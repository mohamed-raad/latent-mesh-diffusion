"""
latent_observatory.py -- Phase 13: Emergence Measurement Framework.

Six experiments that answer:

  Is the model organizing knowledge?
  Is the graph meaningful?
  Are experts specializing?
  Is routing improving?
  Are latent concepts stable?

Instead of adding more modules, this measures what the latent space is doing.
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F


# Experiment 1: Semantic Stability

@dataclass
class ConceptProbe:
    """Tracks which textual concepts each latent node maps to over time."""

    node_concept_history: dict[int, list[tuple[str, float]]] = field(default_factory=dict)

    concept_vocab: list[str] = field(default_factory=lambda: [
        "authentication", "authorization", "database", "sql", "api", "routing",
        "planning", "reasoning", "logic", "math", "arithmetic", "algebra",
        "recursion", "iteration", "sorting", "searching", "encoding", "decoding",
        "memory", "attention", "classification", "regression", "optimization",
        "physics", "chemistry", "biology", "geography", "history", "language",
        "translation", "summarization", "question_answering", "code_generation",
        "debugging", "testing", "deployment", "security", "networking",
    ])

    def probe(
        self,
        node_state: torch.Tensor,
        node_id: int,
        top_k: int = 3,
    ) -> list[tuple[str, float]]:
        """Find the closest concept labels to a node's latent state."""
        device = node_state.device
        if node_id not in self.node_concept_history:
            self.node_concept_history[node_id] = []

        concept_embs = []
        for concept in self.concept_vocab:
            rng = torch.Generator(device=device).manual_seed(hash(concept) % (2**31 - 1))
            emb = torch.randn(node_state.size(-1), generator=rng, device=device)
            concept_embs.append(F.normalize(emb, dim=-1))
        concept_embs = torch.stack(concept_embs)

        node_norm = F.normalize(node_state.unsqueeze(0), dim=-1)
        sims = (node_norm @ concept_embs.T).squeeze(0)
        top_sims, top_idx = sims.topk(min(top_k, len(self.concept_vocab)))

        results = [(self.concept_vocab[idx.item()], sim.item()) for idx, sim in zip(top_idx, top_sims)]
        self.node_concept_history[node_id].append((time.time(), results))
        return results

    def stability_score(self, node_id: int, window: int = 5) -> float:
        """Measure how stable a node's top concept is over recent probes.
        1.0 = same top concept every time, 0.0 = completely random.
        """
        history = self.node_concept_history.get(node_id, [])
        if len(history) < 2:
            return 0.0
        recent = history[-window:]
        top_concepts = [h[1][0][0] if h[1] else "" for h in recent]
        if not top_concepts:
            return 0.0
        most_common = max(set(top_concepts), key=top_concepts.count)
        return top_concepts.count(most_common) / len(top_concepts)


# Experiment 2: Graph Stability

@dataclass
class GraphStability:
    """Measures how similar latent graphs are for identical inputs across runs."""

    graph_hashes: list[tuple[str, float]] = field(default_factory=list)

    def compare_graphs(
        self,
        graph_a_nodes: dict[str, torch.Tensor],
        graph_b_nodes: dict[str, torch.Tensor],
    ) -> float:
        """Compute graph similarity as mean cosine similarity of matched nodes."""
        if not graph_a_nodes or not graph_b_nodes:
            return 0.0

        keys_a = list(graph_a_nodes.keys())
        keys_b = list(graph_b_nodes.keys())

        if len(keys_a) < len(keys_b):
            keys_a, keys_b = keys_b, keys_a
            graph_a_nodes, graph_b_nodes = graph_b_nodes, graph_a_nodes

        total_sim = 0.0
        count = 0
        for kid in keys_b:
            state_b = graph_b_nodes[kid]
            if state_b.dim() > 1:
                state_b = state_b.squeeze()
            state_b_norm = F.normalize(state_b.unsqueeze(0), dim=-1)

            best_sim = -1.0
            for ka_id in keys_a:
                state_a = graph_a_nodes[ka_id]
                if state_a.dim() > 1:
                    state_a = state_a.squeeze()
                sim = (state_b_norm @ F.normalize(state_a.unsqueeze(0), dim=-1).T).item()
                if sim > best_sim:
                    best_sim = sim
            total_sim += best_sim
            count += 1

        return total_sim / max(count, 1)


# Experiment 3: Expert Identity

@dataclass
class ExpertIdentity:
    """Tracks which experts handle which categories to measure specialization."""

    expert_domains: dict[str, dict[str, int]] = field(default_factory=dict)

    def record_routing(self, expert_id: str, domain_label: str):
        if expert_id not in self.expert_domains:
            self.expert_domains[expert_id] = {}
        self.expert_domains[expert_id][domain_label] = \
            self.expert_domains[expert_id].get(domain_label, 0) + 1

    def specialization_entropy(self, expert_id: str) -> float:
        """Entropy of the domain distribution for an expert.
        Lower entropy = more specialized. 0.0 = only handles one domain.
        """
        domains = self.expert_domains.get(expert_id, {})
        if not domains:
            return 0.0
        total = sum(domains.values())
        if total == 0:
            return 0.0
        entropy = 0.0
        for count in domains.values():
            p = count / total
            if p > 0:
                entropy -= p * torch.tensor(p).log().item()
        return entropy / max(len(domains), 1)

    def top_domain(self, expert_id: str) -> tuple[str, float]:
        domains = self.expert_domains.get(expert_id, {})
        if not domains:
            return ("", 0.0)
        total = sum(domains.values())
        best = max(domains, key=domains.get)
        return (best, domains[best] / total)


# Experiment 5: Consensus Value

@dataclass
class ConsensusValue:
    """Compare accuracy with and without consensus."""

    with_consensus: list[float] = field(default_factory=list)
    without_consensus: list[float] = field(default_factory=list)

    def record(self, acc_with: float, acc_without: float):
        self.with_consensus.append(acc_with)
        self.without_consensus.append(acc_without)

    def consensus_gain(self) -> float:
        if not self.with_consensus:
            return 0.0
        return sum(self.with_consensus) / len(self.with_consensus) - \
               sum(self.without_consensus) / len(self.without_consensus)


# Experiment 6: Mitosis Value

@dataclass
class MitosisValue:
    """Measure whether spawned experts improve performance."""

    before_spawn: list[float] = field(default_factory=list)
    after_spawn: list[float] = field(default_factory=list)
    spawn_times: list[int] = field(default_factory=list)

    def record_spawn(self, step: int, loss_before: float, loss_after: float):
        self.spawn_times.append(step)
        self.before_spawn.append(loss_before)
        self.after_spawn.append(loss_after)

    def spawn_benefit(self) -> float:
        if not self.before_spawn:
            return 0.0
        return (sum(self.before_spawn) / len(self.before_spawn) -
                sum(self.after_spawn) / len(self.after_spawn))


# Main Observatory - aggregates all experiments

@dataclass
class LatentObservatory:
    """Dashboard for emergence measurement across all six experiments."""

    semantic_stability: ConceptProbe = field(default_factory=ConceptProbe)
    graph_stability: GraphStability = field(default_factory=GraphStability)
    expert_identity: ExpertIdentity = field(default_factory=ExpertIdentity)
    consensus_value: ConsensusValue = field(default_factory=ConsensusValue)
    mitosis_value: MitosisValue = field(default_factory=MitosisValue)

    # Experiment 4: latent compression results
    compression_results: dict[int, float] = field(default_factory=dict)

    step: int = 0
    last_report_step: int = 0
    report_interval: int = 50

    def probe_nodes(
        self,
        latent_nodes: torch.Tensor,
        top_k: int = 3,
    ) -> dict[int, list[tuple[str, float]]]:
        """Probe all latent nodes and return their top concept matches."""
        results = {}
        for i in range(latent_nodes.size(1)):
            node_state = latent_nodes[0, i]
            if node_state.dim() > 1:
                node_state = node_state.squeeze()
            matches = self.semantic_stability.probe(node_state, i, top_k=top_k)
            results[i] = matches
        return results

    def compare_graphs(
        self,
        graph_a: dict[str, torch.Tensor],
        graph_b: dict[str, torch.Tensor],
        input_hash: str = "",
    ) -> float:
        """Measure similarity between two latent graphs."""
        sim = self.graph_stability.compare_graphs(graph_a, graph_b)
        if input_hash:
            self.graph_stability.graph_hashes.append((input_hash, sim))
        return sim

    def record_routing(self, expert_id: str, domain_label: str):
        self.expert_identity.record_routing(expert_id, domain_label)

    def record_consensus(self, acc_with: float, acc_without: float):
        self.consensus_value.record(acc_with, acc_without)

    def record_mitosis(self, step: int, loss_before: float, loss_after: float):
        self.mitosis_value.record_spawn(step, loss_before, loss_after)

    def record_compression(self, n_nodes: int, accuracy: float):
        self.compression_results[n_nodes] = accuracy

    def report(self, force: bool = False) -> str:
        """Generate a human-readable emergence report (ASCII-only)."""
        if not force and (self.step - self.last_report_step) < self.report_interval:
            return ""
        self.last_report_step = self.step

        lines = [
            "",
            "=" * 60,
            f"LATENT OBSERVATORY -- Step {self.step}",
            "=" * 60,
        ]

        # Semantic Stability
        if self.semantic_stability.node_concept_history:
            lines.append("")
            lines.append("[Semantic Stability]")
            n_nodes = len(self.semantic_stability.node_concept_history)
            stable_count = 0
            for nid in self.semantic_stability.node_concept_history:
                score = self.semantic_stability.stability_score(nid, window=3)
                if score > 0.5:
                    stable_count += 1
            pct = stable_count / max(n_nodes, 1) * 100
            lines.append(f"  Nodes with stable top concept: {stable_count}/{n_nodes} ({pct:.0f}%)")
            stabilities = [(nid, self.semantic_stability.stability_score(nid))
                           for nid in self.semantic_stability.node_concept_history]
            stabilities.sort(key=lambda x: -x[1])
            for nid, s in stabilities[:5]:
                matches = self.semantic_stability.node_concept_history[nid]
                if matches:
                    concepts = ", ".join(f"{c}({sim:.2f})" for c, sim in matches[-1][1][:2])
                    lines.append(f"  Node {nid}: stability={s:.2f} => [{concepts}]")

        # Expert Identity
        if self.expert_identity.expert_domains:
            lines.append("")
            lines.append("[Expert Identity]")
            for eid in sorted(self.expert_identity.expert_domains.keys()):
                top_domain, pct = self.expert_identity.top_domain(eid)
                entropy = self.expert_identity.specialization_entropy(eid)
                lines.append(f"  Expert {eid}: {pct*100:.0f}% {top_domain}  (entropy={entropy:.3f})")

        # Consensus Value
        if self.consensus_value.with_consensus:
            gain = self.consensus_value.consensus_gain()
            lines.append("")
            lines.append("[Consensus Value]")
            lines.append(f"  Gain vs no-consensus: {gain:+.4f}")

        # Mitosis Value
        if self.mitosis_value.spawn_times:
            benefit = self.mitosis_value.spawn_benefit()
            lines.append("")
            lines.append("[Mitosis Value]")
            lines.append(f"  Spawn events: {len(self.mitosis_value.spawn_times)}")
            lines.append(f"  Avg loss benefit: {benefit:+.4f}")

        # Compression
        if self.compression_results:
            lines.append("")
            lines.append("[Latent Compression]")
            for n_nodes, acc in sorted(self.compression_results.items()):
                lines.append(f"  {n_nodes} nodes => acc={acc:.4f}")

        lines.append("")
        lines.append("=" * 60)
        lines.append("")
        return "\n".join(lines)

    def step_report(self, step: int):
        self.step = step
        report = self.report()
        if report:
            print(report)
