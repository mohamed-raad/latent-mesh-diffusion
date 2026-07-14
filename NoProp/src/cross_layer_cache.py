"""
Cross-layer Routing Cache — IndexShare-style routing reuse across layers.
Instead of every layer computing nearest experts, compute once and reuse.
"""
import time
from collections import OrderedDict
import torch


class RoutingCacheEntry:
    __slots__ = ("expert_ids", "scores", "domain", "step", "hit_count")

    def __init__(self, expert_ids: list[str], scores: list[float], domain: str, step: int):
        self.expert_ids = expert_ids
        self.scores = scores
        self.domain = domain
        self.step = step
        self.hit_count = 1

    def to_dict(self) -> dict:
        return {
            "expert_ids": self.expert_ids,
            "scores": self.scores,
            "domain": self.domain,
            "step": self.step,
            "hit_count": self.hit_count,
        }


class CrossLayerRoutingCache:
    """LRU cache for routing decisions. Reuse across attention layers."""

    def __init__(self, max_entries: int = 1024, ttl_steps: int = 10, min_similarity: float = 0.85):
        self.max_entries = max_entries
        self.ttl_steps = ttl_steps
        self.min_similarity = min_similarity
        self._cache: OrderedDict[str, RoutingCacheEntry] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._current_step = 0

    def _make_key(self, embedding: torch.Tensor, domain: str) -> str:
        """Create a cache key from embedding hash + domain."""
        h = hash(embedding.detach().cpu().numpy().tobytes()) ^ hash(domain)
        return f"{domain}_{h}"

    def get(self, embedding: torch.Tensor, domain: str, step: int) -> RoutingCacheEntry | None:
        """Return cached routing if available and fresh."""
        self._current_step = step
        key = self._make_key(embedding, domain)

        if key in self._cache:
            entry = self._cache[key]
            if step - entry.step <= self.ttl_steps:
                self._cache.move_to_end(key)
                entry.hit_count += 1
                self._hits += 1
                return entry

        self._misses += 1
        return None

    def set(self, embedding: torch.Tensor, domain: str, step: int,
            expert_ids: list[str], scores: list[float]):
        """Cache a routing decision."""
        key = self._make_key(embedding, domain)
        entry = RoutingCacheEntry(expert_ids, scores, domain, step)

        if len(self._cache) >= self.max_entries:
            self._cache.popitem(last=False)
        self._cache[key] = entry

    def get_cache_stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / max(total, 1),
            "size": len(self._cache),
            "max_entries": self.max_entries,
        }

    def clear(self):
        self._cache.clear()
        self._hits = 0
        self._misses = 0


class IndexShareAttention:
    """Sparse attention with cross-layer index sharing."""

    def __init__(self, d_model: int, n_heads: int, top_k: int = 32, block_size: int = 64):
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.top_k = top_k
        self.block_size = block_size

    def compute_nearest_neighbors(self, query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """Compute top-k nearest neighbors for sparse attention."""
        scores = torch.matmul(query, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)
        top_k = min(self.top_k, scores.size(-1))
        _, indices = torch.topk(scores, top_k, dim=-1)
        return indices

    def sparse_attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                         indices: torch.Tensor | None = None) -> torch.Tensor:
        """Sparse attention using pre-computed or shared indices."""
        if indices is None:
            indices = self.compute_nearest_neighbors(query, key)

        gathered_keys = torch.gather(key.unsqueeze(-2).expand(-1, -1, indices.size(-1), -1),
                                      -2, indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim))
        gathered_values = torch.gather(value.unsqueeze(-2).expand(-1, -1, indices.size(-1), -1),
                                        -2, indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim))

        scores = torch.matmul(query.unsqueeze(-2), gathered_keys.transpose(-2, -1)) / (self.head_dim ** 0.5)
        scores = scores.squeeze(-2)
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn.unsqueeze(-2), gathered_values).squeeze(-2)
        return output
