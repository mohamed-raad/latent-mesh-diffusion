"""
ExpertTierManager — Three-tier expert storage.

  GPU  (VRAM)   Active blocks for forward/backward (limited by --max-gpu-experts)
  RAM  (CPU)    Serialised state_dicts in CPU memory (--max-ram-experts)
  Disk (file)   .pt files under nodes_dir/{expert_id}/block.pt

The router picks top-k experts → tier manager ensures they are on GPU.
Least-recently-used experts get evicted: GPU→RAM→Disk.
"""

import os
import time
from collections import OrderedDict

import torch

from external_nodes import save_expert_block, load_expert_block
from noprop_block import NoPropBlock, inject_lora_into_block


class ExpertTierManager:
    """Manages the GPU → RAM → Disk hierarchy for expert blocks."""

    def __init__(
        self,
        nodes_dir: str,
        embed_dim: int,
        n_heads: int = 4,
        max_gpu: int = 8,
        max_ram: int = 32,
    ):
        self.nodes_dir = nodes_dir
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.max_gpu = max_gpu
        self.max_ram = max_ram

        self.device: torch.device | None = None

        self.gpu: dict[str, NoPropBlock] = OrderedDict()
        self.ram: dict[str, dict[str, torch.Tensor]] = OrderedDict()
        self.disk: set[str] = set()

        self.last_used: dict[str, int] = {}
        self.step: int = 0

        self._ops_log: list[dict] = []

    def set_device(self, device: torch.device):
        self.device = device

    # ---- public API ----

    def ensure_gpu(self, expert_id: str) -> NoPropBlock:
        """Return the expert block on GPU, loading from RAM/disk if needed."""
        t0 = time.perf_counter()
        tier_from = "gpu"

        if expert_id in self.gpu:
            self.last_used[expert_id] = self.step
            self.gpu.move_to_end(expert_id)
            return self.gpu[expert_id]

        block = NoPropBlock(self.embed_dim, num_heads=self.n_heads)

        if expert_id in self.ram:
            tier_from = "ram"
            state = self.ram.pop(expert_id)
            # Check if saved state has LoRA keys
            if any("lora" in k for k in state):
                dummy = torch.randn(1, self.embed_dim)
                block(dummy, torch.tensor([[1.0]]))
                inject_lora_into_block(block)
            block.load_state_dict(state, strict=False)
        elif expert_id in self.disk or os.path.exists(
            os.path.join(self.nodes_dir, expert_id, "block.pt")
        ):
            tier_from = "disk"
            # Probe the saved state to detect LoRA
            path = os.path.join(self.nodes_dir, expert_id, "block.pt")
            if os.path.exists(path):
                saved_state = torch.load(path, weights_only=True, map_location="cpu")
                if any("lora" in k for k in saved_state):
                    dummy = torch.randn(1, self.embed_dim)
                    block(dummy, torch.tensor([[1.0]]))
                    inject_lora_into_block(block)
                block.load_state_dict(saved_state, strict=False)
            self.disk.discard(expert_id)
        else:
            # New expert — initialise fresh
            pass

        # Evict if GPU full
        while len(self.gpu) >= self.max_gpu > 0:
            self._evict_gpu()

        block = block.to(self.device)
        block.train()
        if block.optimizer is None:
            block.configure_optimizer()

        self.gpu[expert_id] = block
        self.last_used[expert_id] = self.step

        elapsed = (time.perf_counter() - t0) * 1000
        self._ops_log.append({
            "step": self.step, "expert_id": expert_id,
            "op": f"load_{tier_from}_to_gpu", "duration_ms": round(elapsed, 2),
        })
        return block

    def get_block(self, expert_id: str) -> NoPropBlock | None:
        return self.gpu.get(expert_id)

    def sync_to_disk(self, expert_id: str):
        """Flush one expert from GPU/RAM to disk."""
        if expert_id in self.gpu:
            save_expert_block(self.gpu[expert_id], self.nodes_dir, expert_id)
        elif expert_id in self.ram:
            block = NoPropBlock(self.embed_dim, num_heads=self.n_heads)
            block.load_state_dict(self.ram[expert_id])
            save_expert_block(block, self.nodes_dir, expert_id)

    def sync_all(self):
        """Flush every expert to disk."""
        for eid in list(self.gpu.keys()):
            self.sync_to_disk(eid)
        for eid in list(self.ram.keys()):
            self.sync_to_disk(eid)
        for eid in self.disk:
            pass  # already on disk

    def remove(self, expert_id: str):
        """Remove expert from all tiers."""
        self.gpu.pop(expert_id, None)
        self.ram.pop(expert_id, None)
        self.disk.discard(expert_id)
        self.last_used.pop(expert_id, None)

    # ---- eviction ----

    def _evict_gpu(self):
        lru_id = min(
            (eid for eid in self.gpu if eid in self.last_used),
            key=lambda eid: self.last_used[eid],
            default=None,
        )
        if lru_id is None:
            return
        block = self.gpu.pop(lru_id)
        t0 = time.perf_counter()
        state = {k: v.detach().cpu() for k, v in block.state_dict().items()}

        # Evict from RAM if full
        while len(self.ram) >= self.max_ram > 0:
            self._evict_ram()

        self.ram[lru_id] = state
        elapsed = (time.perf_counter() - t0) * 1000
        self._ops_log.append({
            "step": self.step, "expert_id": lru_id,
            "op": "evict_gpu_to_ram", "duration_ms": round(elapsed, 2),
        })

    def _evict_ram(self):
        lru_id = min(
            (eid for eid in self.ram if eid in self.last_used),
            key=lambda eid: self.last_used[eid],
            default=None,
        )
        if lru_id is None:
            return
        state = self.ram.pop(lru_id)
        t0 = time.perf_counter()
        block = NoPropBlock(self.embed_dim, num_heads=self.n_heads)
        block.load_state_dict(state)
        save_expert_block(block, self.nodes_dir, lru_id)
        self.disk.add(lru_id)
        elapsed = (time.perf_counter() - t0) * 1000
        self._ops_log.append({
            "step": self.step, "expert_id": lru_id,
            "op": "evict_ram_to_disk", "duration_ms": round(elapsed, 2),
        })

    # ---- status ----

    def tier_of(self, expert_id: str) -> str:
        if expert_id in self.gpu:
            return "gpu"
        if expert_id in self.ram:
            return "ram"
        return "disk"

    def summary(self) -> dict:
        return {
            "gpu": len(self.gpu),
            "ram": len(self.ram),
            "disk": len(self.disk) + sum(
                1 for eid in self.disk
                if eid not in self.gpu and eid not in self.ram
            ),
            "gpu_ids": list(self.gpu.keys()),
            "ram_ids": list(self.ram.keys()),
        }

    def drain_ops_log(self) -> list[dict]:
        logs = self._ops_log
        self._ops_log = []
        return logs
