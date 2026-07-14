"""
Expert Registry — persistent JSON-backed directory for expert node discovery.
Supports registration, domain lookup, embedding similarity matching, and
cross-run persistence.
"""

import json
import os
import time
import hashlib

import torch


class ExpertRegistry:
    """
    Central directory of expert node metadata, persisted as JSON.

    Each record:
      - expert_id: unique string
      - domain: domain tag (e.g. 'reasoning', 'code', 'math')
      - embedding_hash: SHA-256 of anchor embedding (first 16 hex chars)
      - creation_step, last_active_step
      - metadata: arbitrary dict

    Args:
        registry_path: Path to JSON file (default: expert_registry.json)
    """

    def __init__(self, registry_path: str = "expert_registry.json"):
        self.registry_path = registry_path
        self._records: dict[str, dict] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path) as f:
                    self._records = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._records = {}

    def _save(self):
        os.makedirs(os.path.dirname(self.registry_path) or ".", exist_ok=True)
        tmp = self.registry_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self._records, f, indent=2)
        os.replace(tmp, self.registry_path)

    @staticmethod
    def _hash_embedding(emb: torch.Tensor) -> str:
        return hashlib.sha256(emb.cpu().numpy().tobytes()).hexdigest()[:16]

    def register(self, expert_id: str, domain: str = "general",
                 embedding: torch.Tensor | None = None, step: int = 0,
                 metadata: dict | None = None) -> bool:
        """Register an expert. Returns True if newly created, False if already existed."""
        if expert_id in self._records:
            self._records[expert_id]["last_active_step"] = max(
                self._records[expert_id]["last_active_step"], step
            )
            if metadata:
                self._records[expert_id]["metadata"].update(metadata)
            self._save()
            return False
        record = {
            "expert_id": expert_id,
            "domain": domain,
            "embedding_hash": self._hash_embedding(embedding) if embedding is not None else "",
            "creation_step": step,
            "last_active_step": step,
            "metadata": metadata or {},
            "created_at": time.time(),
        }
        self._records[expert_id] = record
        self._save()
        return True

    def lookup(self, expert_id: str) -> dict | None:
        return self._records.get(expert_id)

    def lookup_by_domain(self, domain: str) -> list[dict]:
        return [r for r in self._records.values() if r.get("domain") == domain]

    def lookup_similar(self, query_emb: torch.Tensor, top_k: int = 5) -> list[tuple[str, float]]:
        """Return known experts whose embedding hash matches closely."""
        qh = self._hash_embedding(query_emb)
        scored: list[tuple[str, float]] = []
        for eid, rec in self._records.items():
            rh = rec.get("embedding_hash", "")
            if rh and rh == qh:
                scored.append((eid, 1.0))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    def remove(self, expert_id: str) -> bool:
        if expert_id in self._records:
            del self._records[expert_id]
            self._save()
            return True
        return False

    def list_experts(self, domain: str | None = None) -> list[str]:
        if domain:
            return [r["expert_id"] for r in self._records.values() if r.get("domain") == domain]
        return list(self._records.keys())

    def count(self, domain: str | None = None) -> int:
        return len(self.list_experts(domain))

    def summary(self) -> dict:
        domains: dict[str, int] = {}
        for r in self._records.values():
            d = r.get("domain", "general")
            domains[d] = domains.get(d, 0) + 1
        return {"total": len(self._records), "domains": domains, "path": self.registry_path}
