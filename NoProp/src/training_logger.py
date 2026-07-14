"""
training_logger.py — CSV instrumentation for training runs.

Writes structured log files to --log-dir:

  latent_states.csv      Node-level concept probe history
  routing.csv            Per-step expert routing decisions
  consensus.csv          Consensus gain over time
  expert_utilization.csv Expert tier + selection frequency
  mitosis.csv            Spawn events with loss deltas
  tier_ops.csv           GPU/RAM/disk load/evict operations
  training_metrics.csv   Step-level loss / VRAM / speed
"""

import csv
import os
import time
from collections import defaultdict


class TrainingLogger:
    """CSV logger for structured training instrumentation."""

    LOG_NAMES = [
        "latent_states",
        "routing",
        "consensus",
        "expert_utilization",
        "mitosis",
        "tier_ops",
        "training_metrics",
    ]

    FIELDSPEC: dict[str, list[str]] = {
        "latent_states": [
            "step", "node_id", "top_concept_1", "sim_1",
            "top_concept_2", "sim_2", "stability",
        ],
        "routing": [
            "step", "expert_id", "score", "domain",
        ],
        "consensus": [
            "step", "n_experts", "agreement", "gain_vs_no_consensus",
        ],
        "expert_utilization": [
            "step", "expert_id", "tier", "times_selected",
        ],
        "mitosis": [
            "step", "parent_id", "child_id", "loss_before", "loss_after",
        ],
        "tier_ops": [
            "step", "expert_id", "operation", "duration_ms",
        ],
        "training_metrics": [
            "step", "loss", "lr", "vram_gb", "tokens_per_sec", "gpu_experts",
        ],
    }

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self._writers: dict[str, tuple] = {}
        self._counts: dict[str, int] = defaultdict(int)
        self._last_flush: float = time.time()

    def _writer(self, name: str):
        if name not in self._writers:
            path = os.path.join(self.log_dir, f"{name}.csv")
            f = open(path, "w", newline="")
            w = csv.writer(f)
            w.writerow(self.FIELDSPEC[name])
            self._writers[name] = (w, f)
        return self._writers[name][0]

    def log(self, name: str, row: dict):
        if name not in self.FIELDSPEC:
            return
        w = self._writer(name)
        fields = self.FIELDSPEC[name]
        w.writerow([row.get(h, "") for h in fields])
        self._counts[name] += 1
        if time.time() - self._last_flush > 30:
            self.flush()

    def log_latent_states(self, step: int, probe_results: dict[int, list[tuple[str, float]]],
                          stability_scores: dict[int, float]):
        for nid, matches in probe_results.items():
            row = {
                "step": step,
                "node_id": nid,
                "top_concept_1": matches[0][0] if matches else "",
                "sim_1": round(matches[0][1], 4) if matches else "",
                "top_concept_2": matches[1][0] if len(matches) > 1 else "",
                "sim_2": round(matches[1][1], 4) if len(matches) > 1 else "",
                "stability": round(stability_scores.get(nid, 0), 4),
            }
            self.log("latent_states", row)

    def log_routing(self, step: int, routes: list[tuple[str, str, float]]):
        for expert_id, domain, score in routes:
            self.log("routing", {
                "step": step,
                "expert_id": expert_id,
                "score": round(score, 4),
                "domain": domain,
            })

    def log_tier_ops_batch(self, step: int, ops: list[dict]):
        for op in ops:
            op["step"] = step
            self.log("tier_ops", op)

    def log_mitosis(self, step: int, parent: str, child: str,
                    loss_before: float, loss_after: float):
        self.log("mitosis", {
            "step": step,
            "parent_id": parent,
            "child_id": child,
            "loss_before": round(loss_before, 6),
            "loss_after": round(loss_after, 6),
        })

    def flush(self):
        for _, f in self._writers.values():
            f.flush()
        self._last_flush = time.time()

    def close(self):
        self.flush()
        for _, f in self._writers.values():
            f.close()
        self._writers.clear()

    def summary(self) -> str:
        lines = [f"Logs written to {self.log_dir}/"]
        for name in self.LOG_NAMES:
            lines.append(f"  {name}.csv: {self._counts.get(name, 0)} rows")
        return "\n".join(lines)
