"""
Mesh Router — expert routing, universal latent space, planning, execution graphs.
"""
from __future__ import annotations
import os
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional

import torch.nn as nn
import torch.nn.functional as F

import torch

# ═══════════════════════════════════════════════════
# Expert Lifecycle & Metadata
# ═══════════════════════════════════════════════════

class ExpertLifecycle(Enum):
    CREATED = "created"
    EVALUATING = "evaluating"
    ACTIVE = "active"
    MERGING = "merging"
    COMPRESSING = "compressing"
    ARCHIVED = "archived"
    DELETED = "deleted"


@dataclass
class ExpertMetadata:
    accuracy: float = 0.0
    latency_ms: float = 0.0
    usage_count: int = 0
    last_active: float = 0.0
    version: str = "1.0"
    status: ExpertLifecycle = ExpertLifecycle.CREATED
    hallucination_rate: float = 0.0
    failure_count: int = 0
    energy_cost: float = 0.0
    domain: str = ""
    dependencies: list[str] = field(default_factory=list)


@dataclass
class MeshNode:
    node_id: str
    anchor_path: str
    anchor_embedding: torch.Tensor
    rolling_loss: list[float] = field(default_factory=list)
    loss_window: int = 100
    mitosis_threshold: float = 0.5
    metadata: ExpertMetadata = field(default_factory=ExpertMetadata)
    adapter: Optional[nn.Module] = None

    def update_loss(self, loss: float):
        self.rolling_loss.append(loss)
        if len(self.rolling_loss) > self.loss_window:
            self.rolling_loss.pop(0)

    def sustained_high_error(self) -> bool:
        if len(self.rolling_loss) < self.loss_window // 2:
            return False
        recent = torch.tensor(self.rolling_loss[-self.loss_window // 2:])
        return recent.mean().item() > self.mitosis_threshold


# ═══════════════════════════════════════════════════
# Universal Latent Space — Upgraded
# Produces N semantic latent nodes from token embeddings.
# Input:  [B, S, D_model]  →  Output: [B, N, D_latent]
# ═══════════════════════════════════════════════════

class LatentEncoderLayer(nn.Module):
    """One layer: cross-attend to tokens + self-attend among latents + FFN."""

    def __init__(self, d_latent: int, n_heads: int):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_latent, n_heads, batch_first=True)
        self.cross_norm = nn.LayerNorm(d_latent)
        self.self_attn = nn.MultiheadAttention(d_latent, n_heads, batch_first=True)
        self.self_norm = nn.LayerNorm(d_latent)
        self.ffn = nn.Sequential(
            nn.Linear(d_latent, d_latent * 4),
            nn.GELU(),
            nn.Linear(d_latent * 4, d_latent),
        )
        self.ffn_norm = nn.LayerNorm(d_latent)

    def forward(
        self,
        queries: torch.Tensor,
        token_latents: torch.Tensor,
    ) -> torch.Tensor:
        x, _ = self.cross_attn(queries, token_latents, token_latents)
        x = self.cross_norm(queries + x)

        y, _ = self.self_attn(x, x, x)
        y = self.self_norm(x + y)

        z = self.ffn(y)
        return self.ffn_norm(y + z)


class UniversalLatentSpace(nn.Module):
    """Produces N semantic latent nodes from token embeddings.

    Instead of projecting every token to a latent vector (old: [B, S, d_latent]),
    learned semantic queries cross-attend over the token sequence and produce
    a fixed set of N semantic latent nodes (new: [B, N, d_latent]).

    This is semantic compression — 2048 token embeddings → 64-96 concept nodes.
    Each node captures a distinct semantic facet of the input.

    When use_vae=True, the encoder is replaced with a TextVAE (variational
    autoencoder with KL-regularized latent space), providing a learned
    continuous latent space with proper semantic structure.
    """

    def __init__(
        self,
        d_model: int = 1024,
        d_latent: int = 256,
        n_latent_nodes: int = 64,
        n_heads: int = 8,
        n_depth: int = 2,
        use_vae: bool = False,
        vae_config: dict | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_latent = d_latent
        self.n_latent_nodes = n_latent_nodes
        self.use_vae = use_vae

        if use_vae:
            from text_vae import TextVAE, TextVAEConfig
            vae_cfg_kwargs = vae_config or {}
            vae_cfg = TextVAEConfig(
                d_model=d_model,
                d_latent=d_latent,
                n_latent_nodes=n_latent_nodes,
                n_heads=n_heads,
                **vae_cfg_kwargs,
            )
            self.vae = TextVAE(vae_cfg)
        else:
            self.input_proj = nn.Sequential(
                nn.Linear(d_model, d_latent),
                nn.LayerNorm(d_latent),
            )

            self.latent_queries = nn.Parameter(
                torch.randn(n_latent_nodes, d_latent) * 0.02
            )

            self.layers = nn.ModuleList([
                LatentEncoderLayer(d_latent, n_heads) for _ in range(n_depth)
            ])

            self.final_norm = nn.LayerNorm(d_latent)

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """Encode token embeddings into N semantic latent nodes.

        Args:
            token_embeddings: [B, S, D_model] token embeddings.

        Returns:
            [B, N, D_latent] semantic latent nodes.
        """
        if self.use_vae:
            dist = self.vae.encoder(token_embeddings)
            return dist.mode()

        B = token_embeddings.shape[0]
        token_latents = self.input_proj(token_embeddings)
        queries = self.latent_queries.unsqueeze(0).expand(B, -1, -1)

        x = queries
        for layer in self.layers:
            x = layer(x, token_latents)

        return self.final_norm(x)

    def project_tokens(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """Old behavior: project every token to latent space.

        Returns [B, S, D_latent] — one latent per token, no semantic compression.
        Provided for backward compatibility.
        """
        if self.use_vae:
            return self.vae.project_tokens(token_embeddings)
        return self.input_proj(token_embeddings)

    def vae_loss(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor | None = None,
        dist: object | None = None,
        beta: float | None = None,
    ) -> dict[str, torch.Tensor] | None:
        """Compute VAE loss when use_vae=True. Returns None if VAE is disabled."""
        if not self.use_vae:
            return None
        if dist is None:
            dist = self.vae.encoder(x)
        if x_hat is None:
            z = dist.sample()
            x_hat = self.vae.decoder(z)
        return self.vae.loss(x, dist, x_hat, beta)

    def encode_vae(self, x: torch.Tensor) -> object | None:
        """Encode to VAE distribution. Returns None if VAE is disabled."""
        if not self.use_vae:
            return None
        return self.vae.encoder(x)

    def decode_vae(self, z: torch.Tensor) -> torch.Tensor | None:
        """Decode from VAE latents. Returns None if VAE is disabled."""
        if not self.use_vae:
            return None
        return self.vae.decoder(z)


# ═══════════════════════════════════════════════════
# Latent Decoder — maps latents back to embeddings (Phase 7)
# ═══════════════════════════════════════════════════

class LatentDecoder(nn.Module):
    """Maps latent nodes back to token embeddings for the diffusion decoder.

    The diffusion model (NoPropBlock + DiffusionDecoder) operates at the
    embedding level [B, S, d_model]. The latent graph operates at the
    latent level [B, N, d_latent]. This decoder bridges the two:
    latent nodes → token embeddings → diffusion → text.

    Architecture:
      - Learned output queries attend to latent nodes via cross-attention
      - Produces [B, S_out, d_model] token embeddings
      - These feed into the existing DiffusionDecoder
    """

    def __init__(
        self,
        d_latent: int = 256,
        d_model: int = 1024,
        n_output: int = 2048,
        n_heads: int = 8,
    ):
        super().__init__()
        self.d_latent = d_latent
        self.d_model = d_model
        self.n_output = n_output

        self.latent_proj = nn.Linear(d_latent, d_model)

        self.out_queries = nn.Parameter(
            torch.randn(n_output, d_model) * 0.02
        )

        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        latent_nodes: torch.Tensor,
        output_len: int | None = None,
    ) -> torch.Tensor:
        """Decode latent nodes into token embeddings.

        Args:
            latent_nodes: [B, N, d_latent] from UniversalLatentSpace / MeshOfThought.
            output_len: Number of output embeddings (defaults to n_output).

        Returns:
            [B, S_out, d_model] token embeddings for the diffusion decoder.
        """
        B = latent_nodes.shape[0]
        S = output_len or self.n_output

        latent_d_model = self.latent_proj(latent_nodes)  # [B, N, d_model]

        queries = self.out_queries[:S].unsqueeze(0).expand(B, -1, -1)

        x, _ = self.cross_attn(queries, latent_d_model, latent_d_model)
        x = self.norm(queries + x)

        y = self.ffn(x)
        y = self.ffn_norm(x + y)

        return y


# ═══════════════════════════════════════════════════
# Latent Memory — stores reasoning graphs (Phase 8)
# ═══════════════════════════════════════════════════

class LatentMemory:
    """Stores Latent Graphs for reasoning reuse.

    Instead of RAG (retrieving raw text), this retrieves entire reasoning
    graphs. When a similar query arrives, the stored graph is loaded and
    reasoning continues from where it left off — enabling cumulative
    reasoning across sessions.

    Example:
        Rest API graph → stored as "rest_api"
        Next REST API request → load graph → continue reasoning
    """

    def __init__(self, max_graphs: int = 100):
        self.graphs: dict[str, LatentGraph] = {}
        self.keys: dict[str, torch.Tensor] = {}
        self.max_graphs = max_graphs

    def store(
        self,
        key: str,
        graph: LatentGraph,
        embedding: torch.Tensor | None = None,
    ):
        self.graphs[key] = graph
        if embedding is not None:
            self.keys[key] = embedding
        if len(self.graphs) > self.max_graphs:
            oldest = next(iter(self.graphs))
            del self.graphs[oldest]
            self.keys.pop(oldest, None)

    def retrieve(self, key: str) -> LatentGraph | None:
        return self.graphs.get(key)

    def search(
        self,
        query_embedding: torch.Tensor,
        top_k: int = 3,
    ) -> list[tuple[str, LatentGraph, float]]:
        """Find stored graphs most similar to a query embedding."""
        if not self.keys:
            return []
        query_norm = F.normalize(query_embedding, dim=-1)
        results: list[tuple[str, LatentGraph, float]] = []
        for key, emb in self.keys.items():
            emb_norm = F.normalize(emb.unsqueeze(0), dim=-1)
            sim = (query_norm @ emb_norm.T).item()
            results.append((key, self.graphs[key], sim))
        results.sort(key=lambda x: -x[2])
        return results[:top_k]

    def merge_into(
        self,
        target_graph: LatentGraph,
        key: str,
    ) -> LatentGraph:
        """Merge a stored graph's nodes into a target graph."""
        stored = self.graphs.get(key)
        if stored is None:
            return target_graph
        for nid, node in stored.nodes.items():
            if nid not in target_graph.nodes:
                target_graph.nodes[nid] = node
        return target_graph


# ═══════════════════════════════════════════════════
# Mitosis Analyzer — smart expert creation (Phase 9)
# ═══════════════════════════════════════════════════

class MitosisAnalyzer:
    """Analyzes latent nodes to create specialized experts.

    Instead of creating a generic expert on high loss (old behavior),
    this analyzes which latent nodes the current expert handles poorly
    and creates a specialized expert for that semantic cluster.

    Steps:
      1. Collect failing nodes (nodes with low confidence from current expert).
      2. Cluster failing nodes by semantic similarity.
      3. For each cluster, create a new expert with an anchor near that cluster.
      4. Register the new expert with the router and planner.
    """

    def __init__(
        self,
        router: 'MeshRouter | None' = None,
        similarity_threshold: float = 0.6,
        min_nodes_for_new_expert: int = 3,
    ):
        self.router = router
        self.similarity_threshold = similarity_threshold
        self.min_nodes_for_new_expert = min_nodes_for_new_expert

    def analyze(
        self,
        graph: LatentGraph,
        low_confidence_threshold: float = 0.3,
    ) -> list[dict]:
        """Analyze latent graph and propose new expert configurations.

        Args:
            graph: LatentGraph to analyze.
            low_confidence_threshold: Nodes below this confidence are "failing".

        Returns:
            List of proposals, each with:
              - anchor: proposed anchor embedding for the new expert
              - label: semantic label from node concepts
              - n_nodes: number of nodes in the cluster
              - confidence: average cluster confidence
        """
        # 1. Find low-confidence nodes
        failing = [
            node for node in graph.nodes.values()
            if node.confidence < low_confidence_threshold
        ]
        if len(failing) < self.min_nodes_for_new_expert:
            return []

        # 2. Cluster by similarity
        clusters = self._cluster_nodes(failing)
        proposals = []
        for cluster in clusters:
            if len(cluster) < self.min_nodes_for_new_expert:
                continue
            states = torch.stack([n.state for n in cluster])
            anchor = states.mean(dim=0)
            anchor = F.normalize(anchor, dim=-1)
            avg_conf = sum(n.confidence for n in cluster) / len(cluster)
            labels = [n.concept_label for n in cluster if n.concept_label]
            label = max(set(labels), key=labels.count) if labels else ""
            proposals.append({
                "anchor": anchor.detach().cpu(),
                "label": label or f"cluster_{len(proposals)}",
                "n_nodes": len(cluster),
                "confidence": avg_conf,
            })
        return proposals

    def _cluster_nodes(
        self,
        nodes: list[LatentNode],
    ) -> list[list[LatentNode]]:
        """Group nodes by cosine similarity."""
        if not nodes:
            return []
        states = torch.stack([n.state for n in nodes])
        states_norm = F.normalize(states, dim=-1)
        sim = states_norm @ states_norm.T

        assigned = set()
        clusters: list[list[LatentNode]] = []
        for i in range(len(nodes)):
            if i in assigned:
                continue
            cluster = [nodes[i]]
            assigned.add(i)
            for j in range(i + 1, len(nodes)):
                if j in assigned:
                    continue
                if sim[i, j].item() > self.similarity_threshold:
                    cluster.append(nodes[j])
                    assigned.add(j)
            clusters.append(cluster)
        return clusters


class ExpertAdapter(nn.Module):
    """Bidirectional adapter between latent space and expert space."""
    def __init__(self, d_latent: int, d_expert: int):
        super().__init__()
        self.encode = nn.Linear(d_latent, d_expert, bias=False)
        self.decode = nn.Linear(d_expert, d_latent, bias=False)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.encode(latent)

    def decode_from_expert(self, expert_out: torch.Tensor) -> torch.Tensor:
        return self.decode(expert_out)


# ═══════════════════════════════════════════════════
# Intent Detection & Difficulty (AF.md #3)
# ═══════════════════════════════════════════════════

class IntentDetector(nn.Module):
    """Classifies domain/intent from backbone hidden state."""
    def __init__(self, d_model: int, n_domains: int = 16, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_domains),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.mean(dim=1))


class DifficultyEstimator(nn.Module):
    """Regression head estimating difficulty 1-7."""
    def __init__(self, d_model: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.mean(dim=1)) * 6.0 + 1.0  # Scale to [1, 7]


# ═══════════════════════════════════════════════════
# ═══════════════════════════════════════════════════
# Latent Graph — structured latent nodes for reasoning (Phase 2)
# ═══════════════════════════════════════════════════

@dataclass
class LatentNode:
    """A structured node in the latent graph.

    Each node captures one semantic concept with:
      - state:        current latent vector [d_latent]
      - memory:       accumulated context (for reasoning reuse)
      - confidence:   how certain we are of this node's content
      - importance:   relevance to the current query/task
      - parents:      reasoning dependencies (directed edges)
      - children:     reasoning consequences
      - neighbors:    undirected connections
      - owner:        which expert is responsible for this node

    This is the core data structure for Mesh of Thought (Phase 3)
    and Memory (Phase 8).
    """
    node_id: str
    state: torch.Tensor
    memory: torch.Tensor
    confidence: float = 0.0
    importance: float = 0.0
    parents: list[str] = field(default_factory=list)
    children: list[str] = field(default_factory=list)
    neighbors: list[str] = field(default_factory=list)
    owner: str = ""
    created_at: int = 0
    updated_at: int = 0
    concept_label: str = ""

    def connect(self, other_id: str):
        if other_id not in self.children:
            self.children.append(other_id)

    def similarity(self, other: 'LatentNode') -> float:
        return F.cosine_similarity(
            self.state.unsqueeze(0),
            other.state.unsqueeze(0),
        ).item()


class LatentGraph:
    """A directed graph of LatentNodes for reasoning.

    Fits beside ExpertGraph — not replacing it. Supports traversal,
    partitioning, pruning, serialization, and similarity computation.
    """

    def __init__(self):
        self.nodes: dict[str, LatentNode] = {}
        self.root_ids: list[str] = []
        self.global_step: int = 0

    def add_node(self, node: LatentNode) -> str:
        self.nodes[node.node_id] = node
        if node.node_id not in self.root_ids:
            self.root_ids.append(node.node_id)
        node.updated_at = self.global_step
        return node.node_id

    def remove_node(self, node_id: str):
        node = self.nodes.get(node_id)
        if node is None:
            return
        for n in self.nodes.values():
            n.children = [c for c in n.children if c != node_id]
            n.parents = [p for p in n.parents if p != node_id]
            n.neighbors = [nb for nb in n.neighbors if nb != node_id]
        if node_id in self.root_ids:
            self.root_ids.remove(node_id)
        del self.nodes[node_id]

    def get_node(self, node_id: str) -> LatentNode | None:
        return self.nodes.get(node_id)

    def connect(self, parent_id: str, child_id: str):
        parent = self.nodes.get(parent_id)
        child = self.nodes.get(child_id)
        if parent and child:
            if child_id not in parent.children:
                parent.children.append(child_id)
            if parent_id not in child.parents:
                child.parents.append(parent_id)

    def get_neighbors(self, node_id: str) -> list[LatentNode]:
        node = self.nodes.get(node_id)
        if node is None:
            return []
        result: list[LatentNode] = []
        seen: set[str] = set()
        for nid in node.children + node.parents + node.neighbors:
            if nid in self.nodes and nid not in seen:
                result.append(self.nodes[nid])
                seen.add(nid)
        return result

    def subgraph(self, node_ids: list[str]) -> 'LatentGraph':
        g = LatentGraph()
        for nid in node_ids:
            if nid in self.nodes:
                g.nodes[nid] = self.nodes[nid]
        g.root_ids = [r for r in self.root_ids if r in node_ids]
        return g

    def partition_by_owner(self) -> dict[str, 'LatentGraph']:
        partitions: dict[str, LatentGraph] = {}
        for nid, node in self.nodes.items():
            owner = node.owner or "_unassigned"
            if owner not in partitions:
                partitions[owner] = LatentGraph()
            partitions[owner].nodes[nid] = node
        return partitions

    def prune_low_importance(self, threshold: float = 0.1):
        to_remove = [
            nid for nid, node in self.nodes.items()
            if node.importance < threshold
        ]
        for nid in to_remove:
            self.remove_node(nid)

    def to_tensor(self) -> torch.Tensor:
        if not self.nodes:
            return torch.empty(0, 0)
        return torch.stack([node.state for node in self.nodes.values()])

    def from_tensor(self, tensor: torch.Tensor, prefix: str = "latent"):
        self.nodes = {}
        for i in range(tensor.shape[0]):
            nid = f"{prefix}_{i:04d}"
            self.nodes[nid] = LatentNode(
                node_id=nid,
                state=tensor[i],
                memory=torch.zeros_like(tensor[i]),
            )

    def similarity_matrix(self) -> torch.Tensor:
        t = self.to_tensor()
        if t.shape[0] < 2:
            return torch.eye(t.shape[0])
        t_norm = F.normalize(t, dim=-1)
        return t_norm @ t_norm.T


# ═══════════════════════════════════════════════════
# Mesh of Thought — graph-based reasoning (Phase 3)
# ═══════════════════════════════════════════════════

class MeshOfThought:
    """Graph-based reasoning replacing chain-of-thought.

    Instead of A→B→C (linear), builds a reasoning graph where:
      - Multiple branches from a single node (parallel exploration)
      - Experts process different node groups concurrently
      - Consensus integrates parallel updates
      - The graph structure reflects reasoning dependencies

    Usage:
        mot = MeshOfThought(router, latent_dim=256, iters=3)
        final_graph = mot.reason(initial_graph)
    """

    def __init__(
        self,
        router: 'MeshRouter | None' = None,
        latent_dim: int = 256,
        max_iterations: int = 3,
        consensus_rounds: int = 2,
    ):
        self.router = router
        self.latent_dim = latent_dim
        self.max_iterations = max_iterations
        self.consensus_rounds = consensus_rounds

    def reason(
        self,
        graph: LatentGraph,
        blocks: dict | None = None,
    ) -> LatentGraph:
        """Run Mesh of Thought on a latent graph for max_iterations rounds.

        Each round:
          1. Assign unowned nodes to experts via router.
          2. Each expert improves its assigned nodes (if blocks provided).
          3. Form new connections between similar nodes.
          4. Run consensus to resolve conflicts.

        Returns the updated graph.
        """
        for _ in range(self.max_iterations):
            graph.global_step += 1

            # 1. Assign unowned nodes to experts
            if self.router is not None and blocks is not None:
                self._assign_nodes(graph)

            # 2. Each expert improves its nodes
            if blocks is not None:
                self._expert_update(graph, blocks)

            # 3. Form connections between similar nodes
            self._form_connections(graph)

            # 4. Consensus
            for _ in range(self.consensus_rounds):
                self._consensus_step(graph)

        return graph

    def _assign_nodes(self, graph: LatentGraph):
        """Route each unowned node to the best-matching expert."""
        for nid, node in graph.nodes.items():
            if node.owner:
                continue
            query = F.normalize(node.state.unsqueeze(0).unsqueeze(0), dim=-1)
            if self.router is None:
                continue
            results = self.router.route(query)
            if results:
                node.owner = results[0][0]

    def _expert_update(
        self,
        graph: LatentGraph,
        blocks: dict,
    ):
        """Each expert processes its assigned nodes (latent → latent)."""
        partitions = graph.partition_by_owner()
        for owner_nid, subgraph in partitions.items():
            if owner_nid == "_unassigned":
                continue
            block = blocks.get(owner_nid)
            if block is None:
                continue
            node_tensor = subgraph.to_tensor()
            if node_tensor.dim() != 2:
                continue
            B, D = node_tensor.shape
            t_zero = torch.zeros(B, 1, device=node_tensor.device).fill_(0.0)
            # Experts process latents directly (Phase 5: latent → latent)
            updated = block(node_tensor.unsqueeze(1), t_zero).squeeze(1)
            for i, nid in enumerate(subgraph.nodes):
                if nid in graph.nodes:
                    graph.nodes[nid].state = updated[i]
                    graph.nodes[nid].confidence = min(
                        1.0, graph.nodes[nid].confidence + 0.1
                    )

    def _form_connections(self, graph: LatentGraph):
        """Connect nodes with similarity above threshold."""
        sim = graph.similarity_matrix()
        threshold = 0.7
        nids = list(graph.nodes.keys())
        for i in range(len(nids)):
            for j in range(i + 1, len(nids)):
                if sim[i, j].item() > threshold:
                    graph.connect(nids[i], nids[j])
                    graph.nodes[nids[i]].neighbors.append(nids[j])
                    graph.nodes[nids[j]].neighbors.append(nids[i])

    def _consensus_step(self, graph: LatentGraph):
        """Resolve conflicts: similar nodes owned by different experts converge."""
        sim = graph.similarity_matrix()
        nids = list(graph.nodes.keys())
        threshold = 0.8
        for i in range(len(nids)):
            for j in range(i + 1, len(nids)):
                if sim[i, j].item() > threshold:
                    ni_id, nj_id = nids[i], nids[j]
                    ni, nj = graph.nodes[ni_id], graph.nodes[nj_id]
                    ci, cj = ni.confidence, nj.confidence
                    total = ci + cj + 1e-8
                    merged = (ni.state * ci + nj.state * cj) / total
                    ni.state = merged
                    nj.state = merged
                    ni.confidence = (ci + cj) / 2.0
                    nj.confidence = (ci + cj) / 2.0


# ═══════════════════════════════════════════════════
# Expert Graph & Planner (AF.md #3 + #4)
# ═══════════════════════════════════════════════════

@dataclass
class ExpertTreeNode:
    name: str
    children: list['ExpertTreeNode'] = field(default_factory=list)
    expert_id: str | None = None
    depth: int = 0

    def is_leaf(self) -> bool:
        return self.expert_id is not None


class ExpertGraph:
    """Hierarchical expert tree. Math → {Algebra, Geometry, Calculus, Statistics}."""
    def __init__(self):
        self.roots: list[ExpertTreeNode] = []
        self._leaf_map: dict[str, ExpertTreeNode] = {}

    def add_leaf(self, path: list[str], expert_id: str):
        if not path:
            return
        current_roots = self.roots
        parent: ExpertTreeNode | None = None
        for i, seg in enumerate(path):
            found = None
            for child in current_roots:
                if child.name == seg:
                    found = child
                    break
            if found is None:
                found = ExpertTreeNode(name=seg, depth=i)
                current_roots.append(found)
            parent = found
            current_roots = found.children
        if parent is not None:
            parent.expert_id = expert_id
            self._leaf_map[expert_id] = parent

    def get_leaf(self, path: list[str]) -> str | None:
        current = self.roots
        for seg in path:
            found = None
            for child in current:
                if child.name == seg:
                    found = child
                    break
            if found is None:
                return None
            current = found.children
            if found.expert_id is not None:
                return found.expert_id
        return None

    def traverse(self, query_embedding: torch.Tensor, router_anchors: dict[str, torch.Tensor],
                 top_k: int = 3) -> list[str]:
        scores = []
        for eid, emb in router_anchors.items():
            sim = F.cosine_similarity(query_embedding.unsqueeze(0), emb.unsqueeze(0))
            scores.append((eid, sim.item()))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [eid for eid, _ in scores[:top_k]]


class RouterPlanner(nn.Module):
    """Takes intent + difficulty → traverses ExpertGraph → returns execution plan."""
    def __init__(self, d_model: int, n_domains: int = 16):
        super().__init__()
        self.intent_detector = IntentDetector(d_model, n_domains)
        self.difficulty_estimator = DifficultyEstimator(d_model)

    def forward(self, x: torch.Tensor, expert_graph: ExpertGraph,
                router_anchors: dict[str, torch.Tensor], top_k: int = 3) -> tuple[list[str], dict]:
        intent_logits = self.intent_detector(x)
        intent_id = intent_logits.argmax(dim=-1).item()
        difficulty = self.difficulty_estimator(x).item()

        query = x.mean(dim=1)
        selected_experts = expert_graph.traverse(query, router_anchors, top_k=top_k)

        plan = {
            "intent_id": intent_id,
            "difficulty": round(difficulty, 1),
            "num_experts": len(selected_experts),
            "experts": selected_experts,
            "requires_verification": difficulty > 4.0,
            "requires_tools": difficulty > 5.0,
        }
        return selected_experts, plan


class ExecutionGraph:
    """Step-by-step execution plan: sequential/parallel expert calls."""
    def __init__(self):
        self.steps: list[dict] = []

    def build(self, expert_ids: list[str], plan: dict) -> list[dict]:
        self.steps = []
        for eid in expert_ids:
            self.steps.append({
                "expert_id": eid,
                "requires_tool": plan.get("requires_tools", False),
                "requires_verification": plan.get("requires_verification", False),
            })
        if plan.get("requires_verification", False):
            self.steps.append({"expert_id": "__verifier__", "requires_tool": False, "requires_verification": False})
        return self.steps


# ═══════════════════════════════════════════════════
# Confidence Engine (AF.md Infrastructure)
# ═══════════════════════════════════════════════════

@dataclass
class FactEntry:
    fact: str
    confidence: float = 0.5
    timestamp: float = 0.0
    sources: list[str] = field(default_factory=list)
    verification_count: int = 0
    contradictions: list[str] = field(default_factory=list)


class ConfidenceEngine:
    """Tracks per-fact confidence with timestamps and contradictions."""
    def __init__(self):
        self.facts: dict[str, FactEntry] = {}

    def add_fact(self, fact: str, source: str = "", initial_confidence: float = 0.5):
        now = time.time()
        if fact not in self.facts:
            self.facts[fact] = FactEntry(fact=fact, confidence=initial_confidence, timestamp=now)
        entry = self.facts[fact]
        entry.timestamp = now
        entry.verification_count += 1
        if source and source not in entry.sources:
            entry.sources.append(source)
        entry.confidence = min(1.0, entry.confidence + 0.05)

    def add_contradiction(self, fact: str, contradiction: str):
        if fact in self.facts:
            self.facts[fact].contradictions.append(contradiction)
            self.facts[fact].confidence = max(0.01, self.facts[fact].confidence - 0.2)

    def get_confidence(self, fact: str) -> float:
        return self.facts.get(fact, FactEntry(fact=fact)).confidence


# ═══════════════════════════════════════════════════
# Consensus Layer (Phase 6)
# ═══════════════════════════════════════════════════

@dataclass
class ExpertVote:
    expert_id: str
    latent: torch.Tensor
    confidence: float
    domain: str = ""

class ConsensusEngine:
    """Replaces mean(predictions) with confidence-weighted consensus.

    Each expert gives: (updated_latent, confidence).
    Consensus chooses the best update — not an average — which
    preserves expert specialization and avoids dilution.

    Supports:
      - Confidence-weighted selection (pick best)
      - Agreement detection (experts converge)
      - Conflict detection (experts disagree → flag for re-processing)
    """

    def __init__(self, agreement_threshold: float = 0.85):
        self.agreement_threshold = agreement_threshold

    def consensus(
        self,
        votes: list[ExpertVote],
    ) -> tuple[torch.Tensor, float, dict]:
        """Compute consensus from expert votes.

        Strategy: pick the highest-confidence vote.
        If multiple votes agree (cosine sim > threshold), use their weighted mean.

        Args:
            votes: List of ExpertVote from active experts.

        Returns:
            (consensus_latent, consensus_confidence, metadata)
            metadata includes:
              - selected: expert_id of the chosen/predominant vote
              - agreement: mean pairwise agreement among votes
              - conflict: True if significant disagreement exists
        """
        if not votes:
            raise ValueError("No votes to consense")

        if len(votes) == 1:
            return votes[0].latent, votes[0].confidence, {
                "selected": votes[0].expert_id,
                "agreement": 1.0,
                "conflict": False,
                "n_votes": 1,
            }

        # Compute pairwise agreement
        states = torch.stack([v.latent for v in votes])
        states_norm = F.normalize(states, dim=-1)
        sim_matrix = states_norm @ states_norm.T
        n = len(votes)
        total_agreement = 0.0
        pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                total_agreement += sim_matrix[i, j].item()
                pairs += 1
        mean_agreement = total_agreement / max(pairs, 1)

        # Find the highest-confidence vote's agreement cluster
        sorted_votes = sorted(votes, key=lambda v: v.confidence, reverse=True)
        best = sorted_votes[0]

        # Gather agreeing votes
        agreeing = [best]
        for v in sorted_votes[1:]:
            sim = F.cosine_similarity(
                best.latent.unsqueeze(0),
                v.latent.unsqueeze(0),
            ).item()
            if sim > self.agreement_threshold:
                agreeing.append(v)

        if len(agreeing) > 1:
            # Weighted mean of agreeing votes
            weights = torch.tensor(
                [v.confidence for v in agreeing],
                device=best.latent.device,
            )
            weighted_sum = sum(
                v.latent * w for v, w in zip(agreeing, weights)
            )
            consensus = weighted_sum / weights.sum()
            consensus_conf = sum(v.confidence for v in agreeing) / len(agreeing)
        else:
            # No agreement — use best expert's output
            consensus = best.latent
            consensus_conf = best.confidence * 0.8  # discount lone expert

        return consensus, consensus_conf, {
            "selected": best.expert_id,
            "agreement": mean_agreement,
            "conflict": mean_agreement < 0.5,
            "n_votes": n,
            "n_agreeing": len(agreeing),
        }


# ═══════════════════════════════════════════════════
# Expert Health Monitor (AF.md Infrastructure)
# ═══════════════════════════════════════════════════

@dataclass
class HealthMetrics:
    latency_ms: list[float] = field(default_factory=list)
    accuracy: list[float] = field(default_factory=list)
    usage_count: int = 0
    failure_count: int = 0
    hallucination_rate: float = 0.0
    energy_cost: float = 0.0

    @property
    def avg_latency(self) -> float:
        return sum(self.latency_ms) / max(len(self.latency_ms), 1)

    @property
    def avg_accuracy(self) -> float:
        return sum(self.accuracy) / max(len(self.accuracy), 1)


class HealthMonitor:
    def __init__(self):
        self.metrics: dict[str, HealthMetrics] = {}

    def record(self, node_id: str, latency_ms: float, accuracy: float, failed: bool = False):
        if node_id not in self.metrics:
            self.metrics[node_id] = HealthMetrics()
        m = self.metrics[node_id]
        m.latency_ms.append(latency_ms)
        if len(m.latency_ms) > 100:
            m.latency_ms.pop(0)
        m.accuracy.append(accuracy)
        if len(m.accuracy) > 100:
            m.accuracy.pop(0)
        m.usage_count += 1
        if failed:
            m.failure_count += 1

    def get_unhealthy(self, max_latency: float = 500.0, min_accuracy: float = 0.3) -> list[str]:
        return [nid for nid, m in self.metrics.items()
                if m.avg_latency > max_latency or m.avg_accuracy < min_accuracy]


# ═══════════════════════════════════════════════════
# Main MeshRouter
# ═══════════════════════════════════════════════════

class MeshRouter:
    def __init__(
        self,
        top_k: int = 3,
        qb_enabled: bool = True,
        d_model: int = 1024,
        n_domains: int = 16,
        latent_config: 'LatentMeshConfig | dict | None' = None,
    ):
        self.top_k = top_k
        self.qb_enabled = qb_enabled
        self._latent_config: LatentMeshConfig | None = None

        self.nodes: dict[str, MeshNode] = {}
        self.node_ids: list[str] = []
        self.anchor_tensor: torch.Tensor | None = None
        self.qb_betas: torch.Tensor | None = None

        # Parse latent_config into UniversalLatentSpace args
        use_vae = False
        vae_cfg = None
        latent_nodes = 64
        latent_heads = 8
        latent_depth = 2
        d_latent = 256

        if latent_config is not None:
            if isinstance(latent_config, dict):
                latent_config = LatentMeshConfig(**latent_config)
            self._latent_config = latent_config
            latent_nodes = latent_config.latent_nodes
            latent_heads = latent_config.latent_heads
            latent_depth = latent_config.latent_depth
            d_latent = latent_config.d_latent
            use_vae = getattr(latent_config, 'use_vae', False)
            if use_vae:
                vae_cfg = {
                    'kl_beta': getattr(latent_config, 'vae_kl_beta', 0.001),
                    'hierarchical': getattr(latent_config, 'hierarchical', False),
                    'patch_size': getattr(latent_config, 'patch_size', 2),
                    'n_encoder_blocks': getattr(latent_config, 'vae_encoder_blocks', 4),
                    'n_decoder_blocks': getattr(latent_config, 'vae_decoder_blocks', 4),
                }

        self.latent_space = UniversalLatentSpace(
            d_model=d_model,
            d_latent=d_latent,
            n_latent_nodes=latent_nodes,
            n_heads=latent_heads,
            n_depth=latent_depth,
            use_vae=use_vae,
            vae_config=vae_cfg,
        )
        self.router_planner = RouterPlanner(d_model, n_domains)
        self.expert_graph = ExpertGraph()
        self.execution_graph = ExecutionGraph()
        self.confidence_engine = ConfidenceEngine()
        self.health_monitor = HealthMonitor()

    def _compute_qb_betas(self):
        if not self.qb_enabled or self.anchor_tensor is None or self.anchor_tensor.size(0) < 2:
            self.qb_betas = None
            return
        anchors = self.anchor_tensor.to(torch.float32)
        anchors = F.normalize(anchors, dim=-1)
        sim_matrix = anchors @ anchors.T
        n = sim_matrix.size(0)
        triu_vals = sim_matrix[torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)]
        if triu_vals.numel() < 2:
            self.qb_betas = None
            return
        global_med = triu_vals.median()
        betas = []
        for i in range(n):
            row = sim_matrix[i].clone()
            row[i] = float("nan")
            row_vals = row[~row.isnan()]
            if row_vals.numel() < 2:
                betas.append(0.0)
            else:
                below_med = (row_vals < global_med).float().mean().item()
                betas.append((below_med - 0.5) * 0.2)
        self.qb_betas = torch.tensor(betas, dtype=torch.float32)

    def _rebuild_anchor_tensor(self):
        self.node_ids = list(self.nodes.keys())
        if not self.node_ids:
            self.anchor_tensor = None
            self.qb_betas = None
            return
        anchors = [self.nodes[nid].anchor_embedding.detach().cpu() for nid in self.node_ids]
        self.anchor_tensor = torch.stack(anchors)
        self._compute_qb_betas()

    def register_node(self, node: MeshNode, graph_path: list[str] | None = None):
        self.nodes[node.node_id] = node
        if graph_path:
            self.expert_graph.add_leaf(graph_path, node.node_id)
        self._rebuild_anchor_tensor()

    def remove_node(self, node_id: str):
        self.nodes.pop(node_id, None)
        self._rebuild_anchor_tensor()

    def route(self, query: torch.Tensor) -> list[tuple[str, MeshNode, float]]:
        """Route query through latent space → cosine similarity → top-k."""
        if self.anchor_tensor is None or self.anchor_tensor.size(0) == 0:
            return []

        anchors = self.anchor_tensor.to(query.device)
        device = query.device

        # Project query and anchors to same space for comparison
        if query.size(-1) == self.latent_space.d_model and anchors.size(-1) == self.latent_space.d_model:
            # Both in d_model space → project both to d_latent
            self.latent_space = self.latent_space.to(device)
            if query.dim() == 2:
                latent = self.latent_space.project_tokens(query.unsqueeze(0)).mean(dim=1, keepdim=True)
            elif query.size(0) > 1:
                latent = self.latent_space.project_tokens(query).mean(dim=0, keepdim=True)
            else:
                latent = self.latent_space.project_tokens(query).mean(dim=0, keepdim=True)
            anchors = self.latent_space.project_tokens(anchors)
        else:
            # Already in latent space (or unknown) — flatten and compare directly
            if query.dim() > 2:
                latent = query.reshape(-1, query.size(-1)).mean(dim=0, keepdim=True)
            else:
                latent = query
            if anchors.size(-1) != latent.size(-1):
                # Dimensionality mismatch — skip routing
                return []

        query_norm = F.normalize(latent.to(device), dim=-1)
        anchors_norm = F.normalize(anchors, dim=-1)

        sims = query_norm @ anchors_norm.T
        if self.qb_betas is not None:
            sims = sims + self.qb_betas.to(sims.device).unsqueeze(0)

        # Mean-pool over all non-anchor dimensions to get [anchors_size] vector
        while sims.dim() > 1:
            sims = sims.mean(dim=0)
        if sims.dim() == 0:
            sims = sims.unsqueeze(0)
        sims = sims.squeeze(-1)  # ensure 1D

        k = min(self.top_k, self.anchor_tensor.size(0))
        top_scores, top_indices = sims.topk(k)
        result = []
        if top_indices.dim() == 0:
            top_indices = top_indices.unsqueeze(0)
            top_scores = top_scores.unsqueeze(0)
        for idx, score in zip(top_indices.tolist(), top_scores.tolist()):
            nid = self.node_ids[idx]
            result.append((nid, self.nodes[nid], score))
            self.health_monitor.record(nid, 0.0, score)
        return result

    def route_nodes(
        self,
        nodes: list | LatentGraph,
        top_k: int | None = None,
    ) -> dict[str, list[str]]:
        """Route LatentNodes to experts based on state similarity.

        Phase 4: Router routes graph nodes to experts instead of embeddings.
        Each node's state is used to find the best-matching expert,
        then the node is assigned to that expert for processing.

        Args:
            nodes: List of LatentNode or a LatentGraph.
            top_k: Experts per node (defaults to self.top_k).

        Returns:
            Dict mapping expert_id -> list of node_ids assigned to that expert.
        """
        if isinstance(nodes, LatentGraph):
            node_list = list(nodes.nodes.values())
        else:
            node_list = nodes

        if not node_list or self.anchor_tensor is None:
            return {}

        k = top_k or self.top_k
        expert_assignments: dict[str, list[str]] = {}

        # Pre-project anchors once
        anchors_raw = self.anchor_tensor
        if anchors_raw.size(-1) == self.latent_space.d_model:
            anchors_proj = self.latent_space.project_tokens(anchors_raw)
        else:
            anchors_proj = anchors_raw
        anchors_norm = F.normalize(anchors_proj, dim=-1)
        k = min(k, anchors_norm.size(0))

        for node in node_list:
            state = node.state.detach().to(self.anchor_tensor.device)
            if state.dim() == 1:
                state = state.unsqueeze(0)
            state_norm = F.normalize(state, dim=-1)
            sims = state_norm @ anchors_norm.T
            _, top_indices = sims.topk(k)
            for idx in top_indices[0]:
                nid = self.node_ids[idx.item()]
                if nid not in expert_assignments:
                    expert_assignments[nid] = []
                expert_assignments[nid].append(node.node_id)

        return expert_assignments

    def route_with_planning(self, x: torch.Tensor) -> tuple[list[tuple[str, MeshNode, float]], dict]:
        """Full planning path: intent → difficulty → expert selection → execution graph."""
        selected_ids, plan = self.router_planner(x, self.expert_graph,
                                                  {nid: n.anchor_embedding.to(x.device)
                                                   for nid, n in self.nodes.items()},
                                                  top_k=self.top_k)
        steps = self.execution_graph.build(selected_ids, plan)
        results = [(nid, self.nodes[nid], 1.0) for nid in selected_ids if nid in self.nodes]
        return results, {"plan": plan, "steps": steps}

    def check_mitosis(self, node_id: str) -> str | None:
        node = self.nodes.get(node_id)
        if node is None or not node.sustained_high_error():
            return None
        child_id = f"{node_id}_v{len([k for k in self.nodes if k.startswith(node_id)])}"
        anchors_dir = os.path.join("nodes")
        os.makedirs(anchors_dir, exist_ok=True)
        child_path = os.path.join(anchors_dir, f"{child_id}.md")
        with open(child_path, "w") as f:
            f.write(f"# {child_id}\n\nAuto-mitosed from {node_id}\n")
        child_embed = node.anchor_embedding + 0.01 * torch.randn_like(node.anchor_embedding)
        child = MeshNode(
            node_id=child_id,
            anchor_path=child_path,
            anchor_embedding=child_embed,
            mitosis_threshold=node.mitosis_threshold,
        )
        child.metadata = ExpertMetadata(domain=node.metadata.domain, version=node.metadata.version)
        self.register_node(child)
        return child_id

    def merge_similar(self, similarity_threshold: float = 0.95) -> list[tuple[str, str]]:
        """Merge experts whose anchors are nearly identical.
        Returns list of (merged_into, merged_from) pairs."""
        merged: list[tuple[str, str]] = []
        node_ids = list(self.nodes.keys())
        for i in range(len(node_ids)):
            if node_ids[i] not in self.nodes:
                continue
            anchor_i = self.nodes[node_ids[i]].anchor_embedding
            for j in range(i + 1, len(node_ids)):
                if node_ids[j] not in self.nodes:
                    continue
                anchor_j = self.nodes[node_ids[j]].anchor_embedding
                sim = F.cosine_similarity(anchor_i.unsqueeze(0), anchor_j.unsqueeze(0)).item()
                if sim >= similarity_threshold:
                    parent_id = node_ids[i]
                    child_id = node_ids[j]
                    self.nodes[parent_id].rolling_loss.extend(self.nodes[child_id].rolling_loss)
                    self.nodes[child_id].metadata.status = ExpertLifecycle.MERGING
                    anchor_path = self.nodes[parent_id].anchor_path
                    self.remove_node(child_id)
                    with open(anchor_path, "a") as f:
                        f.write(f"\n# merged: {child_id} (sim={sim:.3f})")
                    merged.append((parent_id, child_id))
        return merged

    def prune_dead(self, max_idle_steps: int = 5000, min_loss_window: int = 10) -> list[str]:
        """Remove experts that haven't been used recently or have no training history.
        Returns list of pruned node IDs."""
        pruned: list[str] = []
        for nid, node in list(self.nodes.items()):
            if len(node.rolling_loss) < min_loss_window and len(self.nodes) > 3:
                self.nodes[nid].metadata.status = ExpertLifecycle.ARCHIVED
                self.remove_node(nid)
                pruned.append(nid)
        return pruned

    def to(self, device: torch.device):
        self.latent_space = self.latent_space.to(device)
        self.router_planner = self.router_planner.to(device)
        return self

    def latent_consistency_loss(self, batch_latents: list[torch.Tensor]) -> torch.Tensor:
        """Cosine similarity consistency between same-input latents across experts."""
        if len(batch_latents) < 2:
            return torch.tensor(0.0)
        loss = 0.0
        n = 0
        for i in range(len(batch_latents)):
            for j in range(i + 1, len(batch_latents)):
                loss += 1.0 - F.cosine_similarity(batch_latents[i], batch_latents[j], dim=-1).mean()
                n += 1
        return loss / max(n, 1)


# ═══════════════════════════════════════════════════
# Adaptive Compute (AF.md Infrastructure)
# ═══════════════════════════════════════════════════

class AdaptiveCompute:
    """Easy questions → core only. Hard → planner + experts + tools + verification."""
    def select_mode(self, difficulty: float) -> str:
        if difficulty <= 2.0:
            return "core_only"
        elif difficulty <= 4.0:
            return "core_plus_router"
        elif difficulty <= 5.5:
            return "full_pipeline"
        else:
            return "full_pipeline_with_verification"


# ═══════════════════════════════════════════════════
# World Model (AF.md Infrastructure)
# ═══════════════════════════════════════════════════

class WorldModel:
    """Learns relationships between facts (not isolated facts)."""
    def __init__(self):
        self.relations: dict[tuple[str, str], str] = {}

    def add_relation(self, concept_a: str, concept_b: str, relation: str):
        self.relations[(concept_a, concept_b)] = relation

    def query(self, concept: str) -> list[tuple[str, str, str]]:
        result = []
        for (a, b), rel in self.relations.items():
            if a == concept:
                result.append((a, rel, b))
            elif b == concept:
                result.append((b, rel, a))
        return result


# ═══════════════════════════════════════════════════
# Latent Mesh Losses (Phase 10)
# ═══════════════════════════════════════════════════

def semantic_consistency_loss(
    latent_before: torch.Tensor,
    latent_after: torch.Tensor,
) -> torch.Tensor:
    """Ensure expert updates preserve semantic meaning.

    The updated latent should be directionally consistent with the input
    latent. This prevents experts from drifting into unrelated semantic
    regions while still allowing improvement.

    Loss = 1 - cosine_similarity(before, after)
    Range: [0, 2], lower = more consistent.
    """
    sim = F.cosine_similarity(latent_before, latent_after, dim=-1)
    return (1.0 - sim).mean()


def consensus_loss(
    expert_outputs: list[torch.Tensor],
) -> torch.Tensor:
    """Encourage experts to produce consistent (not identical) outputs.

    Each expert processes the same input. This loss measures the variance
    across expert outputs. Low variance = high consensus.
    Experts should agree on the core content, not on surface features.

    Loss = mean pairwise cosine distance
    Range: [0, 2], lower = more consensus.
    """
    if len(expert_outputs) < 2:
        return torch.tensor(0.0, device=expert_outputs[0].device)

    stacked = torch.stack(expert_outputs)
    n = stacked.shape[0]
    sim_matrix = torch.zeros(n, n, device=stacked.device)
    for i in range(n):
        for j in range(n):
            sim_matrix[i, j] = F.cosine_similarity(
                stacked[i].flatten(), stacked[j].flatten(), dim=0
            )
    # Upper triangle mean (excluding diagonal)
    triu = torch.triu(sim_matrix, diagonal=1)
    count = n * (n - 1) / 2
    mean_sim = triu.sum() / count
    return (1.0 - mean_sim)


def reconstruction_loss(
    original_embedding: torch.Tensor,
    latent_nodes: torch.Tensor,
    latent_decoder: 'LatentDecoder',
) -> torch.Tensor:
    """Embedding → Latent → Embedding cycle consistency.

    The full pipeline should be reversible:
      token_embed → latent nodes → predicted_embedding ≈ token_embed

    This ensures the latent space faithfully represents the input.
    """
    reconstructed = latent_decoder(latent_nodes)
    return F.mse_loss(reconstructed, original_embedding)


# ═══════════════════════════════════════════════════
# Coding Mode — project-level code generation (Phase 11)
# ═══════════════════════════════════════════════════

@dataclass
class CodeNode:
    """A node in the project-level code generation graph.

    Hierarchical layers:
      Project → Architecture → Modules → Files → Classes → Functions → Tests → Deployment
    """
    name: str
    node_type: str  # "project", "architecture", "module", "file", "class", "function", "test", "deployment", "dependency", "docker", "ci"
    description: str = ""
    code: str = ""
    dependencies: list[str] = field(default_factory=list)
    sub_nodes: list['CodeNode'] = field(default_factory=list)


class CodingMode:
    """Project-level parallel code generation.

    Instead of generating tokens one-at-a-time (autoregressive LLM),
    this builds a structured project graph where:
      - Files → Classes → Functions → Tests are planned as a graph
      - Each node is generated by the expert best suited for it
      - Multiple experts generate in parallel
      - The diffusion decoder renders each node's code

    This is where the architecture becomes genuinely different from
    standard LLMs.
    """

    def __init__(
        self,
        router: 'MeshRouter | None' = None,
        latent_space: UniversalLatentSpace | None = None,
        latent_decoder: LatentDecoder | None = None,
    ):
        self.router = router
        self.latent_space = latent_space
        self.latent_decoder = latent_decoder

    def plan_project(self, specification: str) -> CodeNode:
        """Plan a project from a specification string.

        Returns a CodeNode tree with the full hierarchy:
          Project → Architecture → Modules → Files → Classes → Functions → Tests → Deployment

        This mirrors how experienced engineers organize large systems.
        """
        root = CodeNode(name="project", node_type="project", description=specification)

        # Architecture layer
        arch = CodeNode(name="architecture", node_type="architecture",
                        description="System architecture overview and design decisions")
        root.sub_nodes.append(arch)

        # Module layer
        entry_mod = CodeNode(name="entry", node_type="module", description="Application entry point and routing")
        core_mod = CodeNode(name="core", node_type="module", description="Core business logic and data models")
        infra_mod = CodeNode(name="infrastructure", node_type="module", description="Infrastructure, config, deployment")
        arch.sub_nodes.extend([entry_mod, core_mod, infra_mod])

        # File layer under entry
        entry_mod.sub_nodes.append(CodeNode(name="main.py", node_type="file", description="Entry point"))

        # File layer under core
        core_mod.sub_nodes.append(CodeNode(name="models.py", node_type="file", description="Data models"))
        core_mod.sub_nodes.append(CodeNode(name="service.py", node_type="file", description="Business logic"))
        core_mod.sub_nodes.append(CodeNode(
            name="tests", node_type="module", description="Test suite",
            sub_nodes=[
                CodeNode("test_models.py", "file", "Unit tests for data models"),
                CodeNode("test_service.py", "file", "Unit tests for business logic"),
            ],
        ))

        # File layer under infrastructure
        infra_mod.sub_nodes.append(CodeNode(name="config.py", node_type="file", description="Configuration"))
        infra_mod.sub_nodes.append(CodeNode(name="deployment", node_type="module", description="Deployment configs",
            sub_nodes=[
                CodeNode("Dockerfile", "deployment", "Container definition"),
                CodeNode("docker-compose.yml", "deployment", "Multi-service orchestration"),
                CodeNode(".github/workflows/ci.yml", "ci", "CI/CD pipeline"),
            ],
        ))

        return root

    def generate_project(
        self,
        project_graph: CodeNode,
        max_parallel: int = 4,
    ) -> CodeNode:
        """Generate code for a project graph in parallel.

        Leaves of the graph are generated by experts, non-leaves are
        structural (directories).
        """
        leaves = self._collect_leaves(project_graph)

        for leaf in leaves:
            leaf.code = f"# {leaf.name} - {leaf.description}\n# TODO: generate"

        return project_graph

    def _collect_leaves(self, node: CodeNode) -> list[CodeNode]:
        if not node.sub_nodes:
            return [node]
        leaves = []
        for child in node.sub_nodes:
            leaves.extend(self._collect_leaves(child))
        return leaves


# ═══════════════════════════════════════════════════
# Latent Mesh Configuration (Phase 12)
# ═══════════════════════════════════════════════════

@dataclass
class LatentMeshConfig:
    """Configuration for the full Latent Mesh Diffusion Computer.

    Add to an existing MeshRouter config dict or instantiate directly.

    Example:
        cfg = LatentMeshConfig(latent_nodes=96, latent_heads=8, latent_depth=2)
        router = MeshRouter(latent_config=cfg)
    """
    # Number of semantic latent nodes (N)
    latent_nodes: int = 96

    # Cross-attention heads in latent encoder
    latent_heads: int = 8

    # Number of LatentEncoderLayer blocks
    latent_depth: int = 2

    # Latent node state dimension
    d_latent: int = 256

    # Enable TextVAE for variational latent space (Phase 1 upgrade)
    use_vae: bool = False

    # TextVAE settings (only used when use_vae=True)
    vae_kl_beta: float = 0.001
    hierarchical: bool = False
    patch_size: int = 2
    vae_encoder_blocks: int = 4
    vae_decoder_blocks: int = 4

    # MeshOfThought max reasoning iterations
    mot_max_iterations: int = 5

    # Consensus agreement threshold
    consensus_threshold: float = 0.85

    # Mitosis similarity threshold
    mitosis_threshold: float = 0.6

    # Memory max graphs
    memory_max_graphs: int = 100

    # Latent decoder output length
    decoder_output_len: int = 2048

    # Coding mode max parallel generators
    coding_parallel: int = 4

    # Loss weights
    loss_semantic_weight: float = 0.1
    loss_consensus_weight: float = 0.05
    loss_reconstruction_weight: float = 0.2

    def to_dict(self) -> dict:
        return asdict(self)


# ═══════════════════════════════════════════════════
# Utility
# ═══════════════════════════════════════════════════

def load_node_metadata(path: str) -> dict:
    with open(path) as f:
        content = f.read()
    lines = content.strip().split("\n")
    tags = [ln.strip("# ").strip() for ln in lines if ln.startswith("#") and len(ln.strip("# ").strip()) < 80]
    return {"tags": tags, "path": path, "size": len(content)}
