"""
Persistent external mesh memory — disk-backed vector store for cross-session
expert state. Uses FAISS for approximate nearest-neighbor search by cosine
similarity.

Each entry stores (embedding, expert_id, timestamp, metadata) and is
persisted to disk as a FAISS index + JSON metadata file.
"""

import json
import os
import time

import numpy as np
import torch


class MeshMemory:
    """
    Disk-backed persistent memory for expert node states.

    Args:
        dim: Embedding dimensionality (default 1024)
        index_path: Path for the FAISS index file
        metadata_path: Path for the JSON metadata file
        use_gpu: Whether to use GPU-accelerated FAISS (default False)
    """

    def __init__(self, dim: int = 1024, index_path: str = "mesh_memory/mem.index",
                 metadata_path: str = "mesh_memory/mem_meta.json", use_gpu: bool = False):
        self.dim = dim
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.use_gpu = use_gpu
        self._index = None
        self._metadata: list[dict] = []
        self._imported = False
        self._load()

    def _check_faiss(self):
        if not self._imported:
            global faiss
            try:
                import faiss as _faiss
                faiss = _faiss
            except ImportError:
                raise ImportError(
                    "faiss is required for MeshMemory. "
                    "Install with: pip install faiss-cpu (or faiss-gpu)"
                )
            self._imported = True

    def _init_index(self):
        self._check_faiss()
        nlist = max(1, min(100, max(len(self._metadata) // 2, 1)))
        quantizer = faiss.IndexFlatIP(self.dim)
        self._index = faiss.IndexIVFFlat(quantizer, self.dim, nlist, faiss.METRIC_INNER_PRODUCT)
        if len(self._metadata) > 0:
            dummy = np.zeros((1, self.dim), dtype=np.float32)
            self._index.train(dummy)

    def _load(self):
        self._check_faiss()
        if os.path.exists(self.index_path):
            try:
                self._index = faiss.read_index(self.index_path)
                if self.use_gpu:
                    res = faiss.StandardGpuResources()
                    self._index = faiss.index_cpu_to_gpu(res, 0, self._index)
            except Exception:
                self._init_index()
        else:
            self._init_index()
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path) as f:
                    self._metadata = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._metadata = []

    def _save(self):
        self._check_faiss()
        os.makedirs(os.path.dirname(self.index_path) or ".", exist_ok=True)
        idx = self._index
        if self.use_gpu:
            idx = faiss.index_gpu_to_cpu(self._index)
        faiss.write_index(idx, self.index_path)
        with open(self.metadata_path, "w") as f:
            json.dump(self._metadata, f, indent=2)

    def insert(self, embedding: torch.Tensor, expert_id: str,
               metadata: dict | None = None) -> int:
        """Insert a memory entry. Returns its index in the store."""
        emb = embedding.detach().cpu().flatten().numpy().astype(np.float32)
        nrm = max(np.linalg.norm(emb), 1e-8)
        emb = (emb / nrm).reshape(1, -1)

        if self._index is None or self._index.ntotal == 0:
            self._init_index()
            self._index.train(emb)

        self._index.add(emb)
        idx = self._index.ntotal - 1
        self._metadata.append({
            "id": idx,
            "expert_id": expert_id,
            "timestamp": time.time(),
            "metadata": metadata or {},
        })
        self._save()
        return idx

    def search(self, query: torch.Tensor, top_k: int = 5) -> list[dict]:
        """Search nearest neighbors by cosine similarity."""
        if self._index is None or self._index.ntotal == 0:
            return []

        q = query.detach().cpu().flatten().numpy().astype(np.float32)
        nrm = max(np.linalg.norm(q), 1e-8)
        q = (q / nrm).reshape(1, -1)

        k = min(top_k, self._index.ntotal)
        distances, indices = self._index.search(q, k)
        results = []
        for dist, ix in zip(distances[0], indices[0]):
            if ix < 0 or ix >= len(self._metadata):
                continue
            rec = self._metadata[ix]
            results.append({
                "expert_id": rec["expert_id"],
                "score": float(dist),
                "metadata": rec.get("metadata", {}),
                "timestamp": rec.get("timestamp", 0),
            })
        return results

    def get_history(self, expert_id: str) -> list[dict]:
        return [r for r in self._metadata if r.get("expert_id") == expert_id]

    def summary(self) -> dict:
        return {
            "dim": self.dim,
            "entries": len(self._metadata),
            "ntotal": self._index.ntotal if self._index is not None else 0,
            "path": self.index_path,
        }
