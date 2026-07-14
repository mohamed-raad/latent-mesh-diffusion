"""
Memory Manager — AF.md #7.
Five-tier memory with consolidation, forgetting, semantic indexing.
"""
import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch

# ═══════════════════════════════════════════════════
# Memory Tiers
# ═══════════════════════════════════════════════════

class MemoryTier(Enum):
    WORKING = "working"           # Current context window
    SHORT_TERM = "short_term"     # Recent interactions
    LONG_TERM = "long_term"       # Consolidated knowledge
    EPISODIC = "episodic"         # Timestamped experiences
    SEMANTIC = "semantic"         # Abstracted concepts


@dataclass
class MemoryEntry:
    key: str
    content: Any
    embedding: torch.Tensor
    tier: MemoryTier
    timestamp: float = 0.0
    access_count: int = 0
    importance: float = 0.5


# ═══════════════════════════════════════════════════
# Memory Store (per-tier)
# ═══════════════════════════════════════════════════

class MemoryStore:
    """Fixed-capacity FIFO + importance-pruning store for one tier."""
    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.entries: list[MemoryEntry] = []

    def add(self, entry: MemoryEntry):
        self.entries.append(entry)
        if len(self.entries) > self.capacity:
            self._prune()

    def _prune(self):
        self.entries.sort(key=lambda e: e.importance * (1 + math.log1p(e.access_count)))
        self.entries = self.entries[-self.capacity:]

    def query(self, query_emb: torch.Tensor, k: int = 5) -> list[MemoryEntry]:
        if not self.entries:
            return []
        scores = []
        for entry in self.entries:
            sim = torch.cosine_similarity(query_emb.unsqueeze(0), entry.embedding.unsqueeze(0))
            scores.append((sim.item(), entry))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scores[:k]]

    def clear(self):
        self.entries.clear()


# ═══════════════════════════════════════════════════
# Memory Manager
# ═══════════════════════════════════════════════════

class MemoryManager:
    """Five-tier memory system with consolidation and forgetting."""
    def __init__(self, d_model: int,
                 working_capacity: int = 10,
                 short_term_capacity: int = 500,
                 long_term_capacity: int = 5000,
                 episodic_capacity: int = 1000,
                 semantic_capacity: int = 2000):

        self.tiers = {
            MemoryTier.WORKING: MemoryStore(working_capacity),
            MemoryTier.SHORT_TERM: MemoryStore(short_term_capacity),
            MemoryTier.LONG_TERM: MemoryStore(long_term_capacity),
            MemoryTier.EPISODIC: MemoryStore(episodic_capacity),
            MemoryTier.SEMANTIC: MemoryStore(semantic_capacity),
        }
        self.d_model = d_model
        self._entry_counter = 0

    def store(self, key: str, content: Any, embedding: torch.Tensor,
              tier: MemoryTier = MemoryTier.SHORT_TERM,
              importance: float = 0.5):
        self._entry_counter += 1
        entry = MemoryEntry(
            key=key,
            content=content,
            embedding=embedding.detach().cpu(),
            tier=tier,
            timestamp=time.time(),
            importance=importance,
        )
        entry.access_count = self._entry_counter
        self.tiers[tier].add(entry)

    def recall(self, query_emb: torch.Tensor,
               tier: MemoryTier | None = None,
               k: int = 5) -> list[MemoryEntry]:
        if tier is not None:
            return self.tiers[tier].query(query_emb, k)
        results = []
        for t in MemoryTier:
            results.extend(self.tiers[t].query(query_emb, k // len(MemoryTier)))
        results.sort(key=lambda e: e.importance * (1 + math.log1p(e.access_count)),
                     reverse=True)
        return results[:k]

    def consolidate(self, source_tier: MemoryTier = MemoryTier.SHORT_TERM,
                    dest_tier: MemoryTier = MemoryTier.LONG_TERM,
                    threshold: float = 0.7):
        """Move high-importance entries from short-term to long-term."""
        store = self.tiers[source_tier]
        high_imp = [e for e in store.entries if e.importance > threshold]
        for entry in high_imp:
            entry.tier = dest_tier
            self.tiers[dest_tier].add(entry)
            store.entries.remove(entry)

    def forget(self, tier: MemoryTier = MemoryTier.SHORT_TERM,
               max_age: float = 86400.0):
        """Remove entries older than max_age seconds."""
        now = time.time()
        store = self.tiers[tier]
        store.entries = [e for e in store.entries
                         if (now - e.timestamp) < max_age]

    def get_stats(self) -> dict:
        return {t.value: len(s.entries) for t, s in self.tiers.items()}

    def get_working_context(self) -> list[MemoryEntry]:
        return self.tiers[MemoryTier.WORKING].entries
