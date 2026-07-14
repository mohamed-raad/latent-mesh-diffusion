"""
Expert Lifecycle Manager — AF.md #6.
Spawning, merging, compression, archival of expert nodes.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# ═══════════════════════════════════════════════════
# Lifecycle State Machine
# ═══════════════════════════════════════════════════

class LifecycleState(Enum):
    CREATED = "created"
    EVALUATING = "evaluating"
    ACTIVE = "active"
    IDLE = "idle"
    MERGING = "merging"
    COMPRESSING = "compressing"
    ARCHIVED = "archived"
    DELETED = "deleted"


@dataclass
class LifecycleRecord:
    node_id: str
    state: LifecycleState = LifecycleState.CREATED
    creation_step: int = 0
    last_active_step: int = 0
    usage_count: int = 0
    accuracy_sum: float = 0.0
    idle_steps: int = 0
    merge_candidates: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════
# Lifecycle Manager
# ═══════════════════════════════════════════════════

class LifecycleManager:
    """Manages expert spawning, merging, compression, and archival."""
    def __init__(self, max_experts: int = 64,
                 idle_timeout: int = 1000,
                 accuracy_threshold: float = 0.8):
        self.records: dict[str, LifecycleRecord] = {}
        self.max_experts = max_experts
        self.idle_timeout = idle_timeout
        self.accuracy_threshold = accuracy_threshold

    def register(self, node_id: str, step: int):
        if node_id not in self.records:
            self.records[node_id] = LifecycleRecord(
                node_id=node_id,
                creation_step=step,
                last_active_step=step,
            )

    def activate(self, node_id: str, step: int, accuracy: float):
        rec = self.records.get(node_id)
        if rec is None:
            return
        rec.last_active_step = step
        rec.usage_count += 1
        rec.accuracy_sum += accuracy
        rec.idle_steps = 0
        if rec.state == LifecycleState.CREATED:
            rec.state = LifecycleState.EVALUATING
        elif rec.state == LifecycleState.EVALUATING and rec.usage_count >= 10:
            avg_acc = rec.accuracy_sum / rec.usage_count
            if avg_acc >= self.accuracy_threshold:
                rec.state = LifecycleState.ACTIVE
        elif rec.state == LifecycleState.IDLE:
            rec.state = LifecycleState.ACTIVE

    def tick_idle(self, current_step: int) -> list[str]:
        idle_nodes = []
        for node_id, rec in self.records.items():
            if rec.state in (LifecycleState.ARCHIVED, LifecycleState.DELETED):
                continue
            steps_since_active = current_step - rec.last_active_step
            if steps_since_active > self.idle_timeout:
                if rec.state == LifecycleState.ACTIVE:
                    rec.state = LifecycleState.IDLE
                rec.idle_steps += 1
                if rec.idle_steps > 3:
                    if rec.usage_count < 5:
                        rec.state = LifecycleState.DELETED
                        idle_nodes.append(node_id)
                    else:
                        rec.state = LifecycleState.ARCHIVED
                        idle_nodes.append(node_id)
        return idle_nodes

    def should_spawn(self, current_expert_count: int) -> bool:
        available = current_expert_count < self.max_experts and current_expert_count > 0
        return available or current_expert_count == 0

    def should_merge(self, node_a: str, node_b: str) -> bool:
        ra = self.records.get(node_a)
        rb = self.records.get(node_b)
        if ra is None or rb is None:
            return False
        if ra.state == LifecycleState.ARCHIVED or rb.state == LifecycleState.ARCHIVED:
            return False
        sim = self._compute_similarity(node_a, node_b)
        return sim > 0.85

    def _compute_similarity(self, node_a: str, node_b: str) -> float:
        return 0.0

    def get_merge_candidates(self) -> list[tuple[str, str]]:
        candidates = []
        ids = [nid for nid, r in self.records.items()
               if r.state not in (LifecycleState.ARCHIVED, LifecycleState.DELETED)]
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                if self.should_merge(ids[i], ids[j]):
                    candidates.append((ids[i], ids[j]))
        return candidates

    def merge(self, node_a: str, node_b: str, step: int) -> Optional[str]:
        parent_id = f"{node_a}+{node_b}_merged@{step}"
        self.register(parent_id, step)
        rec = self.records[parent_id]
        rec.state = LifecycleState.EVALUATING
        rec.usage_count = self.records[node_a].usage_count + self.records[node_b].usage_count
        return parent_id

    def get_state(self, node_id: str) -> Optional[LifecycleState]:
        rec = self.records.get(node_id)
        return rec.state if rec is not None else None

    def get_stats(self) -> dict:
        counts = {}
        for r in self.records.values():
            counts[r.state.value] = counts.get(r.state.value, 0) + 1
        return {
            "total": len(self.records),
            "by_state": counts,
        }
