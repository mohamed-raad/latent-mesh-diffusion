import gc
import math
import glob
import os
import signal
import sys
import threading
import time
from collections import deque

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mesh_tokenizer import VOCAB_SIZE, load_tokenizer, decode as tok_decode, TOKENIZER_NAME
from qat_utils import apply_qat, strip_qat
from diffusion_canvas import DiffusionCanvas
from agk_dataset import AGKDataset
from dspark_speculator import CurriculumDataset, DSparkSpeculator
from export_utils import export_to_gguf, export_to_onnx, export_to_safetensors
from global_cognitive_layer import ConsensusMechanism, GlobalCognitiveLayer, ToolManager
from hub_sync import HubSync
from lifecycle_manager import LifecycleManager
from memory_manager import MemoryManager, MemoryTier
from mesh_router import (
    ExpertAdapter,
    MeshNode,
    MeshRouter,
    UniversalLatentSpace,
    load_node_metadata,
)
from latent_observatory import LatentObservatory
from expert_tier_manager import ExpertTierManager
from training_logger import TrainingLogger
from model_sizes import get_preset, list_presets
from noprop_block import (
    NoPropBlock,
    SinusoidalTimeEmbedding,
    checkpoint_atomic,
    inject_lora_into_block,
    load_checkpoint,
)
from turboquant_attention import TurboQuantKVCompression
from dynamic_quantizer import DynamicQuantizer, quantize_expert_block, dequantize_expert_block
from expert_registry import ExpertRegistry
from mesh_memory import MeshMemory
from cross_layer_cache import CrossLayerRoutingCache
from dynamic_budget import DynamicExpertBudget


def cosine_embedding_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1)


# ═══════════════════════════════════════════════════
# Adaptive Length Bucketing
# ═══════════════════════════════════════════════════

class AdaptiveLengthBuckets:
    """Tracks sequence length distribution and recomputes optimal bucket boundaries.

    Instead of fixed buckets (0-32, 32-64, ...), boundaries adapt to the actual data.
    This ensures sequences within a packed canvas have minimal length variance,
    which reduces the "tail waste" of the final padding token.
    """
    def __init__(self, canvas_len: int, n_buckets: int = 4):
        self.canvas_len = canvas_len
        self.n_buckets = n_buckets
        self._length_history: list[int] = []
        self._boundaries: list[int] = self._default_boundaries()

    def _default_boundaries(self) -> list[int]:
        step = self.canvas_len // self.n_buckets
        return [step * (i + 1) for i in range(self.n_buckets - 1)]

    def observe(self, length: int):
        self._length_history.append(length)
        if len(self._length_history) >= 500:
            self._recompute()

    def _recompute(self):
        if len(self._length_history) < 50:
            return
        arr = sorted(self._length_history)
        n = len(arr)
        boundaries = []
        for i in range(1, self.n_buckets):
            idx = int(n * i / self.n_buckets)
            if idx < n:
                boundaries.append(max(1, arr[idx]))
        if boundaries:
            self._boundaries = boundaries
        self._length_history = self._length_history[-len(self._length_history) // 2:]

    def bucket_for(self, length: int) -> int:
        for i, b in enumerate(self._boundaries):
            if length <= b:
                return i
        return len(self._boundaries)

    @property
    def boundaries(self) -> list[int]:
        return self._boundaries


# ═══════════════════════════════════════════════════
# Domain Queue — persistent per-domain buffer
# ═══════════════════════════════════════════════════

class DomainQueue:
    """Holds incoming sequences for one domain, emits packed canvases when full."""
    def __init__(self, domain: str, packer: "SequencePacker"):
        self.domain = domain
        self.packer = packer
        self.buffer: list[torch.Tensor] = []
        self.buffer_tokens = 0

    def add(self, tokens: torch.Tensor):
        self.buffer.append(tokens)
        self.buffer_tokens += tokens.size(0) + 1

    def pop_packed(self, canvas_len: int, force: bool = False) -> dict | None:
        if not force and self.buffer_tokens < max(8, canvas_len * 0.2):
            return None
        # Take largest sequences first (adaptive bucketing)
        self.buffer.sort(key=lambda x: -x.size(0))
        selected: list[torch.Tensor] = []
        selected_tokens = 0
        while self.buffer and selected_tokens < canvas_len:
            t = self.buffer.pop(0)
            cost = min(t.size(0) + 1, canvas_len)
            if selected_tokens + cost > canvas_len:
                self.buffer.insert(0, t)
                break
            selected.append(t)
            selected_tokens += cost
            self.buffer_tokens -= t.size(0) + 1
        if not selected:
            return None
        packed = self.packer.pack(selected, [self.domain] * len(selected))
        return packed

    def flush(self, canvas_len: int) -> list[dict]:
        canvases = []
        while True:
            c = self.pop_packed(canvas_len, force=True)
            if c is None:
                break
            canvases.append(c)
        return canvases

    @property
    def is_empty(self) -> bool:
        return len(self.buffer) == 0


# ═══════════════════════════════════════════════════
# SequencePacker (enhanced)
# ═══════════════════════════════════════════════════

class SequencePacker:
    """Packs one or more short sequences into a fixed-length canvas with segment masks.

    Features:
    - Dynamic packing: fills each canvas up to capacity, near-zero wasted padding
    - Adaptive length bucketing: sequences sorted longest-first per canvas
    - Expert-aware grouping: preserves domain info in metadata

    Returns dict with:
        input_ids:    [canvas_len] packed tokens
        labels:       [canvas_len] LM targets
        segment_ids:  [canvas_len] which original sequence each token belongs to
        padding_mask: [canvas_len] True=valid, False=padding
        domain:       majority domain of packed sequences
        n_sequences:  number of sequences packed
        pad_fraction: fraction of canvas that is padding (for telemetry)
    """
    def __init__(self, canvas_len: int = 128, eos_id: int = 151643, pad_id: int = 0):
        self.canvas_len = canvas_len
        self.eos_id = eos_id
        self.pad_id = pad_id

    def pack(self, sequences: list[torch.Tensor], domains: list[str] | None = None) -> dict:
        if not sequences:
            return {
                "input_ids": torch.full((self.canvas_len,), self.pad_id, dtype=torch.long),
                "labels": torch.full((self.canvas_len,), self.pad_id, dtype=torch.long),
                "segment_ids": torch.full((self.canvas_len,), -1, dtype=torch.long),
                "padding_mask": torch.zeros(self.canvas_len, dtype=torch.bool),
                "domain": "general",
                "n_sequences": 0,
                "pad_fraction": 1.0,
            }
        # Adaptive length bucketing: longest-first for tighter fit
        if domains is not None:
            pairs = sorted(zip(sequences, domains), key=lambda x: -x[0].size(0))
            sequences = [p[0] for p in pairs]
            domains = [p[1] for p in pairs]
        else:
            sequences = sorted(sequences, key=lambda x: -x.size(0))

        segments: list[torch.Tensor] = []
        seg_ids: list[int] = []
        for i, seq in enumerate(sequences):
            clipped = seq[:self.canvas_len - 1]
            segments.append(clipped)
            segments.append(torch.tensor([self.eos_id], dtype=torch.long))
            seg_ids.extend([i] * (clipped.size(0) + 1))

        full = torch.cat(segments)
        if full.numel() > self.canvas_len:
            full = full[:self.canvas_len]
            seg_ids = seg_ids[:self.canvas_len]

        pad_len = self.canvas_len - full.numel()
        if pad_len > 0:
            full = torch.cat([full, torch.full((pad_len,), self.pad_id, dtype=torch.long)])
            seg_ids.extend([-1] * pad_len)

        from collections import Counter
        majority = (Counter(domains).most_common(1)[0][0]
                    if domains else "general")

        seg_tensor = torch.tensor(seg_ids, dtype=torch.long)
        return {
            "input_ids": full,
            "labels": full.clone(),
            "segment_ids": seg_tensor,
            "padding_mask": seg_tensor >= 0,
            "domain": majority,
            "n_sequences": len(sequences),
            "pad_fraction": pad_len / self.canvas_len,
        }

    def __call__(self, batch: list[dict]) -> dict:
        seqs = []
        domains = []
        for item in batch:
            tokens = item.get("input_ids", item.get("tokens", torch.tensor([], dtype=torch.long)))
            if isinstance(tokens, list):
                tokens = torch.tensor(tokens, dtype=torch.long)
            seqs.append(tokens)
            domains.append(str(item.get("domain", item.get("domain_ids", "general"))))
        return self.pack(seqs, domains)


# ═══════════════════════════════════════════════════
# MeshProfiler — Telemetry Dashboard
# ═══════════════════════════════════════════════════

class MeshProfiler:
    """Real-time telemetry: GPU util, router latency, expert cache hits,
    padding %, data stall time, packing occupancy, per-component breakdown.

    Usage:
        profiler = MeshProfiler()
        profiler.tick_start("data")
        ... load data ...
        profiler.tick_end("data")
        profiler.tick_start("forward")
        ... forward pass ...
        profiler.tick_end("forward")
        profiler.log_step()
    """
    def __init__(self, log_interval: int = 10):
        self.log_interval = log_interval
        self._timers: dict[str, float] = {}
        self._ticks: dict[str, float] = {}
        self.step = 0

        self.gpu_util_samples: list[float] = []
        self.cpu_util_samples: list[float] = []
        self.packing_occupancy: list[float] = []
        self.pad_fraction: list[float] = []
        self.router_latency_ms: list[float] = []
        self.expert_cache_hits: list[float] = []
        self.data_stall_ms: list[float] = []
        self.expert_switches: list[int] = []
        self.router_entropy: list[float] = []
        self.experts_activated: list[int] = []
        self.expert_util_hist: dict[str, int] = {}
        self.vram_gb: float = 0.0

        # Per-component breakdown
        self.component_timers: dict[str, list[float]] = {
            "routing": [], "embed": [], "block_forward": [],
            "block_local_step": [], "mtp": [],
        }

    def tick_start(self, name: str):
        self._ticks[name] = time.perf_counter()

    def tick_end(self, name: str):
        if name in self._ticks:
            elapsed = (time.perf_counter() - self._ticks[name]) * 1000
            if name not in self._timers:
                self._timers[name] = 0.0
            self._timers[name] += elapsed
            del self._ticks[name]

    def tick_component(self, component: str):
        """Context-manager-like helper: call before and after component work."""
        def _end():
            if component in self._ticks:
                elapsed = (time.perf_counter() - self._ticks.pop(component)) * 1000
                self.component_timers.setdefault(component, []).append(elapsed)
        self._ticks[component] = time.perf_counter()
        return _end

    def observe_packing(self, pad_fraction: float, n_seqs: int, canvas_len: int):
        self.pad_fraction.append(pad_fraction)
        occupancy = (1.0 - pad_fraction) * (n_seqs / max(1, canvas_len // 16))
        self.packing_occupancy.append(min(1.0, occupancy))

    def observe_router(self, latency_ms: float, cache_hits: float):
        self.router_latency_ms.append(latency_ms)
        self.expert_cache_hits.append(cache_hits)

    def observe_router_stats(self, experts_activated: int, entropy: float, expert_ids: list[str]):
        self.experts_activated.append(experts_activated)
        self.router_entropy.append(entropy)
        for eid in expert_ids:
            self.expert_util_hist[eid] = self.expert_util_hist.get(eid, 0) + 1

    def observe_gpu(self):
        try:
            free, total = torch.cuda.mem_get_info()
            self.vram_gb = (total - free) / 1e9
        except Exception:
            pass
        try:
            import pynvml
            if not hasattr(self, '_nvml'):
                pynvml.nvmlInit()
                self._nvml = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml)
            self.gpu_util_samples.append(util.gpu / 100.0)
            self.cpu_util_samples.append(0.5)
        except (ImportError, Exception):
            pass

    def log_step(self, step: int):
        self.step = step
        self.observe_gpu()
        if step % self.log_interval == 0:
            self._print_dashboard()

    def _print_dashboard(self):
        def _avg(arr):
            return sum(arr[-self.log_interval:]) / max(1, len(arr[-self.log_interval:]))

        pad_avg = _avg(self.pad_fraction) * 100 if self.pad_fraction else 0
        occ_avg = _avg(self.packing_occupancy) * 100 if self.packing_occupancy else 0
        gpu_avg = _avg(self.gpu_util_samples) * 100 if self.gpu_util_samples else 0
        router_avg = _avg(self.router_latency_ms) if self.router_latency_ms else 0
        cache_avg = _avg(self.expert_cache_hits) * 100 if self.expert_cache_hits else 0
        data_ms = self._timers.get("data", 0)
        forward_ms = self._timers.get("forward", 0)
        total_ms = data_ms + forward_ms
        stall_pct = (data_ms / max(1, total_ms)) * 100

        # Component breakdown
        comp_breakdown = {}
        for comp, vals in self.component_timers.items():
            if vals:
                comp_breakdown[comp] = sum(vals[-self.log_interval:]) / max(1, len(vals[-self.log_interval:]))

        n_activated = self.experts_activated[-1] if self.experts_activated else 0
        entropy_avg = _avg(self.router_entropy) if self.router_entropy else 0

        print("─" * 60)
        print(f"  Step {self.step}  Telemetry Dashboard")
        print("─" * 60)
        print(f"  GPU utilization:     {gpu_avg:5.1f}%")
        print(f"  VRAM:                {self.vram_gb:.2f} GB")
        print(f"  Packing efficiency:  {100-pad_avg:5.1f}%  (occ {occ_avg:.1f}%)")
        print(f"  Router latency:      {router_avg:6.2f} ms")
        print(f"  Experts activated:   {n_activated}")
        print(f"  Router entropy:      {entropy_avg:.4f}")
        print(f"  Expert cache hits:   {cache_avg:5.1f}%")
        print(f"  Data stall time:     {stall_pct:5.1f}%")
        print(f"  Data loading:        {data_ms:.0f} ms / {total_ms:.0f} ms")
        print(f"  Forward compute:     {forward_ms:.0f} ms / {total_ms:.0f} ms")
        if comp_breakdown:
            total_comp = sum(comp_breakdown.values())
            print(f"  ─ Component breakdown (ms / % of forward) ─")
            for comp, avg_ms in sorted(comp_breakdown.items(), key=lambda x: -x[1]):
                pct = (avg_ms / max(1, total_comp)) * 100
                print(f"    {comp:20s}  {avg_ms:7.2f} ms  ({pct:4.1f}%)")
        if self.expert_util_hist:
            top_k = sorted(self.expert_util_hist.items(), key=lambda x: -x[1])[:5]
            print(f"  ─ Top-5 experts by utilization ─")
            for eid, count in top_k:
                print(f"    {eid:20s}  {count} times")
        print("─" * 60)

        # Reset rolling timers
        self._timers.clear()


# ═══════════════════════════════════════════════════
# AsyncPrefetchTokenBucketIterator
# ═══════════════════════════════════════════════════

class AsyncPrefetchTokenBucketIterator:
    """Async-prefetching, dynamic-budget, domain-queued, adaptive-bucketed iterator.

    Combines all 4 optimizations:
    1. **Async prefetch** — background thread continuously builds packed batches
       in a thread-safe queue; GPU never waits for data.
    2. **Dynamic token budget** — queries free VRAM each cycle and adjusts budget.
    3. **Adaptive length bucketing** — observes sequence length distribution and
       adjusts bucket boundaries dynamically.
    4. **Domain-queued expert-aware packing** — persistent per-domain buffers;
       sequences are grouped by domain before packing, maximizing router cache hits.

    Args:
        dataset: PyTorch Dataset yielding dicts with 'input_ids' and optionally 'domain'
        canvas_len: fixed sequence length per canvas
        eos_id: EOS token ID for packing separator
        pad_id: padding token ID
        max_canvases: maximum canvases per batch (VRAM guard)
        dynamic_budget: if True, compute token budget from free VRAM
        min_budget: minimum token budget floor
        max_budget: maximum token budget ceiling
        prefetch_queue_size: number of batches to prefetch (default 2)
        shuffle: shuffle non-iterable datasets
    """
    def __init__(
        self,
        dataset: Dataset,
        canvas_len: int = 128,
        eos_id: int = 151643,
        pad_id: int = 0,
        max_canvases: int = 8,
        dynamic_budget: bool = True,
        min_budget: int = 256,
        max_budget: int = 16384,
        prefetch_queue_size: int = 2,
        shuffle: bool = True,
    ):
        self.dataset = dataset
        self.canvas_len = canvas_len
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.max_canvases = max_canvases
        self.dynamic_budget = dynamic_budget
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.shuffle = shuffle
        self.packer = SequencePacker(canvas_len, eos_id, pad_id)
        self.length_buckets = AdaptiveLengthBuckets(canvas_len)
        self.profiler: MeshProfiler | None = None

        from torch.utils.data import IterableDataset
        self._is_iterable = isinstance(dataset, IterableDataset)
        if not self._is_iterable:
            self._indices = list(range(len(dataset)))
        else:
            self._indices = None
        self._pos = 0

        # Domain queues (persistent across batches)
        self._domain_queues: dict[str, DomainQueue] = {}

        # Async prefetch
        self._queue: deque = deque()
        self._lock = threading.Lock()
        self._prefetch_thread: threading.Thread | None = None
        self._prefetch_running = False
        self._prefetch_queue_size = prefetch_queue_size
        self._batch_count = 0

    def set_profiler(self, profiler: MeshProfiler):
        self.profiler = profiler

    def _compute_token_budget(self) -> int:
        if not self.dynamic_budget or not torch.cuda.is_available():
            return self.max_budget if self.max_budget else 4096
        try:
            free, total = torch.cuda.mem_get_info()
            used_gb = (total - free) / 1e9
            # Target: fill VRAM to ~70% utilization
            target_gb = total / 1e9 * 0.7
            available_for_tokens = max(0.1, target_gb - used_gb)
            target_tokens = int(available_for_tokens * 2048)
            return max(self.min_budget, min(self.max_budget, target_tokens))
        except Exception:
            return self.max_budget // 2 if self.max_budget else 4096

    def _reset(self):
        if not self._is_iterable and self.shuffle:
            import random
            random.shuffle(self._indices)
        self._pos = 0
        self._domain_queues.clear()

    def _get_item(self) -> dict:
        if self._is_iterable:
            item = next(iter(self.dataset))
        else:
            if self._pos >= len(self._indices):
                raise StopIteration
            item = self.dataset[self._indices[self._pos]]
            self._pos += 1
        if isinstance(item, tuple):
            if len(item) >= 2:
                item = {"input_ids": item[0], "labels": item[1], "t": item[2] if len(item) > 2 else torch.zeros(1)}
            else:
                item = {"input_ids": item[0]}
        return item

    def _collect_into_domain_queues(self, budget: int):
        """Pull items from dataset into domain queues until budget reached."""
        total = 0
        while True:
            try:
                item = self._get_item()
            except (StopIteration, IndexError):
                break
            tokens = item.get("input_ids", item.get("tokens"))
            if tokens is None:
                continue
            if isinstance(tokens, (list, tuple)):
                tokens = torch.tensor(tokens, dtype=torch.long)
            seq_len = tokens.size(0)
            domain = str(item.get("domain", item.get("domain_ids", "general")))
            cost = min(seq_len + 1, self.canvas_len)
            if total + cost > budget:
                break
            total += cost
            self.length_buckets.observe(seq_len)
            if domain not in self._domain_queues:
                self._domain_queues[domain] = DomainQueue(domain, self.packer)
            self._domain_queues[domain].add(tokens)

    def _build_batch_from_queues(self) -> dict:
        """Emit one batch by packing from domain queues."""
        canvases: list[dict] = []
        consumed: list[str] = []

        for _ in range(self.max_canvases):
            if not self._domain_queues:
                break
            # Fast O(N) max instead of O(N log N) sort — fine for typical
            # domain counts (4-1000) and called once per batch.
            best_key = max(self._domain_queues, key=lambda k: self._domain_queues[k].buffer_tokens)
            if self._domain_queues[best_key].is_empty:
                consumed.append(best_key)
                continue
            q = self._domain_queues[best_key]
            packed = q.pop_packed(self.canvas_len, force=False)
            if packed is not None:
                canvases.append(packed)
            elif not q.is_empty:
                for c in q.flush(self.canvas_len):
                    canvases.append(c)
                    if len(canvases) >= self.max_canvases:
                        break
            if q.is_empty:
                consumed.append(best_key)

        for k in consumed:
            self._domain_queues.pop(k, None)

        if not canvases:
            raise StopIteration

        # Telemetry
        for c in canvases:
            if self.profiler:
                self.profiler.observe_packing(
                    c.get("pad_fraction", 0), c.get("n_sequences", 1), self.canvas_len
                )

        # Stack into batch [n_canvases, canvas_len]
        batch = torch.stack([c["input_ids"] for c in canvases])
        labels = torch.stack([c["labels"] for c in canvases])
        seg_ids = torch.stack([c["segment_ids"] for c in canvases])
        pad_mask = torch.stack([c["padding_mask"] for c in canvases])

        return {
            "input_ids": batch,
            "labels": labels,
            "segment_ids": seg_ids,
            "padding_mask": pad_mask,
            "domain": canvases[0]["domain"] if canvases else "general",
            "n_canvases": len(canvases),
        }

    def _prefetch_worker(self):
        """Background thread: continuously refill the batch queue."""
        while self._prefetch_running:
            budget = self._compute_token_budget()
            # Collect into domain queues — lock-free: only accessed by this thread
            self._collect_into_domain_queues(budget)
            # Build batches and push to shared queue — brief lock for append only
            while len(self._queue) < self._prefetch_queue_size:
                try:
                    batch = self._build_batch_from_queues()
                    if self.profiler:
                        self.profiler.observe_gpu()
                    with self._lock:
                        self._queue.append(batch)
                except StopIteration:
                    break
            time.sleep(0.001)

    def __iter__(self):
        self._reset()
        # Start prefetch thread
        self._prefetch_running = True
        self._queue.clear()
        self._prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self._prefetch_thread.start()
        return self

    def __next__(self) -> dict:
        if not self._prefetch_running:
            raise StopIteration
        # Wait for next batch from prefetch queue
        while len(self._queue) == 0:
            time.sleep(0.001)
        with self._lock:
            return self._queue.popleft()

    def stop(self):
        self._prefetch_running = False
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=2)

    def __del__(self):
        self.stop()

    def __len__(self) -> int:
        if self._is_iterable:
            return 999999
        budget = self._compute_token_budget()
        return max(1, len(self._indices) // max(1, budget // self.canvas_len))


class SyntheticMeshDataset(Dataset):
    def __init__(self, num_samples: int = 1000, embed_dim: int = 768, num_classes: int = 10):
        self.num_samples = num_samples
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.data = torch.randn(num_samples, embed_dim)
        self.labels = torch.randint(0, num_classes, (num_samples,))
        timesteps = torch.linspace(0, 1, num_samples)
        self.noise_levels = timesteps

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = self.data[idx]
        oh = F.one_hot(self.labels[idx], self.num_classes).float()
        if self.num_classes != self.embed_dim:
            y = F.pad(oh, (0, self.embed_dim - self.num_classes))[:self.embed_dim]
        else:
            y = oh
        t = self.noise_levels[idx].expand(1)
        return x, y, t


class MeshTrainer:
    def __init__(
        self,
        model_size: str = "small",
        embed_dim: int | None = None,
        num_heads: int | None = None,
        top_k: int = 3,
        lr: float = 1e-3,
        nodes_dir: str = "nodes",
        checkpoint_dir: str = "checkpoints/mesh",
        mitosis_threshold: float = 0.5,
        num_draft_tokens: int = 3,
        vocab_size: int = VOCAB_SIZE,
        train_layers: int | None = None,
        use_diffusion_canvas: bool = False,
        canvas_len: int = 512,
        canvas_steps: int = 50,
        canvas_entropy_threshold: float = 0.005,
        mtp_weight: float = 0.1,
        curriculum_dir: str | None = None,
        external_nodes: bool = True,
        max_experts: int = 64,
        log_dir: str | None = None,
        max_gpu_experts: int = 8,
        max_ram_experts: int = 32,
        core_only: bool = False,
        parallel_canvases: int = 1,
        train_experts_only: bool = False,
        domain: str | None = None,
        experts_count: int = 0,
        use_dynamic_quant: bool = False,
        quant_patience: int = 50,
        expert_registry_path: str | None = None,
        mesh_memory_path: str | None = None,
        hub_repo: str | None = None,
        hub_token: str | None = None,
    ):
        # Load model size preset
        self.preset = get_preset(model_size)
        self.embed_dim = embed_dim or self.preset.d_model
        self.n_heads = num_heads or self.preset.n_heads
        self.n_kv_heads = min(self.preset.n_kv_heads, self.n_heads)
        self.n_layers = self.preset.n_layers
        self.d_ff = self.preset.d_ff
        self.model_size_name = model_size

        self.top_k = top_k
        self.lr = lr
        self.nodes_dir = nodes_dir
        self.expert_nodes_dir = os.path.join(nodes_dir, "experts")
        self.checkpoint_dir = checkpoint_dir
        self.mitosis_threshold = mitosis_threshold
        self.num_draft_tokens = num_draft_tokens
        self.vocab_size = vocab_size
        self.train_layers = train_layers
        self.use_diffusion_canvas = use_diffusion_canvas
        self.canvas_len = canvas_len
        self.canvas_steps = canvas_steps
        self.canvas_entropy_threshold = canvas_entropy_threshold

        self.mtp_weight = mtp_weight
        self.curriculum_dir = curriculum_dir
        self.external_nodes = external_nodes
        self.max_experts = max_experts
        self.core_only = core_only
        self.parallel_canvases = max(1, parallel_canvases)
        self.train_experts_only = train_experts_only
        self.domain = domain
        self.experts_count = experts_count
        self.hub_repo = hub_repo
        self.hub_sync: HubSync | None = None

        self.dynamic_quantizer = DynamicQuantizer(patience=quant_patience) if use_dynamic_quant else None
        self.expert_registry = ExpertRegistry(expert_registry_path) if expert_registry_path else None
        self.routing_cache = CrossLayerRoutingCache(max_entries=2048, ttl_steps=10)
        self.dynamic_budget = DynamicExpertBudget(min_experts=2, max_experts=max_experts, easy_experts=4, medium_experts=16, hard_experts=64)
        self._domain_ids: list[str] = []
        mm_path = mesh_memory_path or ""
        self.mesh_memory = MeshMemory(
            dim=self.embed_dim,
            index_path=os.path.join(mm_path, "mem.index") if mesh_memory_path else "",
            metadata_path=os.path.join(mm_path, "mem_meta.json") if mesh_memory_path else "",
        ) if mesh_memory_path else None

        self._qat_enabled = False
        self._interrupted = False
        self._last_saved_step = -1
        self._profiler = None
        signal.signal(signal.SIGINT, self._signal_handler)

        self.token_embedding = nn.Embedding(vocab_size, self.embed_dim)
        self.lm_head = nn.Linear(self.embed_dim, vocab_size)
        self.lm_head.weight = self.token_embedding.weight  # tie weights

        self.router = MeshRouter(top_k=top_k, d_model=self.embed_dim)
        self.latent_space = UniversalLatentSpace(self.embed_dim, d_latent=256)
        self.expert_adapter = ExpertAdapter(256, self.embed_dim)
        self.global_cognitive_layer = GlobalCognitiveLayer(self.embed_dim, n_heads=min(8, self.n_heads))
        self.lifecycle_manager = LifecycleManager(max_experts=max_experts)
        self.memory_manager = MemoryManager(self.embed_dim)
        self.consensus = ConsensusMechanism()
        self.tool_manager = ToolManager()
        self.speculator = DSparkSpeculator(self.embed_dim, vocab_size, num_draft_tokens)
        self.time_emb = SinusoidalTimeEmbedding(self.embed_dim)
        self.kv_compressor = TurboQuantKVCompression(self.embed_dim, use_ste=True)
        if use_diffusion_canvas:
            self.canvas = DiffusionCanvas(
                d_model=self.embed_dim,
                n_layers=self.n_layers,
                n_heads=self.n_heads,
                n_kv_heads=self.n_kv_heads,
                d_ff=self.d_ff,
                vocab_size=vocab_size,
                canvas_len=canvas_len,
                num_steps=canvas_steps,
                entropy_threshold=canvas_entropy_threshold,
                tie_weights=True,
            )
        else:
            self.canvas = None
        self._load_seed_nodes()

        self.step = 0
        self._layer_offset = 0
        self.global_losses: list[float] = []
        self.prefetch_stream: torch.cuda.Stream | None = None
        self._pinned_weights: dict[str, dict[str, torch.Tensor]] = {}
        self.observatory = LatentObservatory()
        self.observatory.report_interval = 10
        self.tier_manager = ExpertTierManager(
            nodes_dir=self.expert_nodes_dir,
            embed_dim=self.embed_dim,
            n_heads=4,
            max_gpu=max_gpu_experts,
            max_ram=max_ram_experts,
        )
        self.logger = TrainingLogger(log_dir) if log_dir else None

    def _embed_tags(self, tags: list[str]) -> torch.Tensor:
        h = torch.zeros(self.embed_dim)
        for i, tag in enumerate(tags):
            tag_hash = hash(tag) % (2**31 - 1)
            rng = torch.Generator().manual_seed(tag_hash)
            h = h + torch.randn(self.embed_dim, generator=rng) * 0.1
        return F.normalize(h, dim=-1)

    def _load_seed_nodes(self):
        os.makedirs(self.nodes_dir, exist_ok=True)
        md_files = []
        for root, dirs, files in os.walk(self.nodes_dir):
            for f in files:
                if f.endswith(".md"):
                    md_files.append(os.path.join(root, f))
        md_files.sort()
        if not md_files:
            anchor = torch.randn(self.embed_dim)
            anchor = F.normalize(anchor, dim=-1)
            node = MeshNode(
                node_id="seed_default",
                anchor_path=os.path.join(self.nodes_dir, "seed_default.md"),
                anchor_embedding=anchor,
                mitosis_threshold=self.mitosis_threshold,
            )
            if self.core_only:
                return
            self.router.register_node(node)
            with open(node.anchor_path, "w") as f:
                f.write("# seed_default\n# Core brain — general reasoning\n#general #reasoning #planning #language\n")
            return

        for path in md_files:
            meta = load_node_metadata(path)
            node_id = os.path.splitext(os.path.basename(path))[0]
            tags = meta.get("tags", [])
            anchor = self._embed_tags(tags)
            rel_path = os.path.relpath(path, self.nodes_dir)
            domain = os.path.dirname(rel_path).replace(os.sep, ".") or "core"
            node = MeshNode(
                node_id=node_id,
                anchor_path=path,
                anchor_embedding=anchor,
                mitosis_threshold=self.mitosis_threshold,
            )
            node.metadata.domain = domain
            node.metadata.dependencies = [p for p in domain.split(".") if p]
            self.router.register_node(node)
            if self.expert_registry:
                self.expert_registry.register(node_id, domain=domain, embedding=anchor, metadata={"tags": tags})

        # Ensure at least experts_count seed nodes (0 = auto, use node files only)
        current_count = len(self.router.nodes)
        if self.experts_count > 0 and current_count < self.experts_count and not self.core_only:
            for i in range(current_count, self.experts_count):
                nid = f"seed_{i}"
                if nid in self.router.nodes:
                    continue
                anchor = F.normalize(torch.randn(self.embed_dim), dim=-1)
                node = MeshNode(
                    node_id=nid,
                    anchor_path=os.path.join(self.nodes_dir, f"{nid}.md"),
                    anchor_embedding=anchor,
                    mitosis_threshold=self.mitosis_threshold,
                )
                node.metadata.domain = self.domain or "general"
                self.router.register_node(node)
                with open(node.anchor_path, "w") as f:
                    f.write(f"# {nid}\n# Auto-created by --experts-count flag\n#{self.domain or 'general'}\n")

        # Seed additional experts from registry (beyond those on disk)
        if self.expert_registry:
            known = set(self.router.nodes.keys())
            for eid in self.expert_registry.list_experts():
                if eid not in known:
                    rec = self.expert_registry.lookup(eid)
                    tags = rec.get("metadata", {}).get("tags", [])
                    anchor = self._embed_tags(tags)
                    node = MeshNode(
                        node_id=eid,
                        anchor_path=os.path.join(self.nodes_dir, f"{eid}.md"),
                        anchor_embedding=anchor,
                        mitosis_threshold=self.mitosis_threshold,
                    )
                    node.metadata.domain = rec.get("domain", "general")
                    self.router.register_node(node)

    def _make_query_embedding(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return F.normalize(x, dim=-1)
        avg = x.mean(dim=1, keepdim=True)
        return F.normalize(avg, dim=-1)

    def _active_nodes(self, x: torch.Tensor, domain_id: int | None = None) -> list[tuple[str, MeshNode, float]]:
        if x.dim() == 3 and x.size(0) > 1:
            query = self._make_query_embedding(x[:1])
        else:
            query = self._make_query_embedding(x)
        domain_str = str(domain_id) if domain_id is not None else "default"
        cached = self.routing_cache.get(query, domain_str, self.step)
        if cached is not None:
            nodes = []
            for eid, score in zip(cached.expert_ids, cached.scores):
                if eid in self.router.nodes:
                    nodes.append((eid, self.router.nodes[eid], score))
            if nodes:
                return nodes
        result = self.router.route(query)
        if result:
            eids = [r[0] for r in result]
            scores = [r[2] if len(r) > 2 else 1.0 for r in result]
            self.routing_cache.set(query, domain_str, self.step, eids, scores)
        return result

    def _select_nodes_for_step(self, candidates: list) -> list:
        if self.train_layers is None or self.train_layers >= len(candidates):
            return candidates
        n = len(candidates)
        start = self._layer_offset % n
        end = min(start + self.train_layers, n)
        selected = candidates[start:end]
        remaining = self.train_layers - len(selected)
        if remaining > 0:
            selected.extend(candidates[:remaining])
        self._layer_offset = (self._layer_offset + self.train_layers) % n
        return selected

    def _embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Convert token IDs to continuous embeddings for mesh processing."""
        return self.token_embedding(token_ids.to(torch.long))

    def _train_step_core_only(self, x: torch.Tensor, target: torch.Tensor, t: torch.Tensor,
                               domain_ids: torch.Tensor | None = None,
                               padding_mask: torch.Tensor | None = None) -> dict[str, float]:
        """Train core engine only: canvas + latent space + speculator. No expert blocks."""
        if self.canvas is None:
            return {"core": 0.0}
        if x.dtype == torch.long:
            labels_ids = target.detach().clone() if target.dtype == torch.long else x.detach().clone()
            if padding_mask is not None:
                labels_ids[~padding_mask] = -100
            x_tokens = x.clone()
            # Parallel canvases: run N trajectories, pick best
            if self.parallel_canvases > 1:
                best_loss_val = float('inf')
                best_logits = None
                for _ in range(self.parallel_canvases):
                    t_i = t + torch.randn_like(t) * 0.15
                    logits_i = self.canvas.model(x_tokens, t_i.view(-1).float())
                    loss_i = F.cross_entropy(logits_i.view(-1, logits_i.size(-1)), labels_ids.view(-1), ignore_index=-100)
                    if loss_i.item() < best_loss_val:
                        best_loss_val = loss_i.item()
                        best_logits = logits_i
                logits = best_logits
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels_ids.view(-1), ignore_index=-100)
            else:
                logits = self.canvas.model(x_tokens, t.view(-1).float())
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels_ids.view(-1), ignore_index=-100)
            # Latent reconstruction loss
            with torch.no_grad():
                x_emb = self._embed_tokens(x_tokens)
            latent = self.latent_space(x_emb)
            recon = self.latent_space.project_tokens(x_emb)
            recon_loss = F.mse_loss(latent.mean(dim=1), recon.mean(dim=1))
            total_loss = loss + 0.05 * recon_loss
            # MTP auxiliary loss
            mtp_loss = self._compute_mtp_loss(logits.detach(), labels_ids)
            if mtp_loss.item() > 0:
                total_loss = total_loss + mtp_loss.item()
            total_loss.backward()
            # Optimize core params
            params = list(self.canvas.model.parameters()) + list(self.latent_space.parameters())
            if not hasattr(self, '_core_optimizer'):
                self._core_optimizer = torch.optim.AdamW(params, lr=self.lr)
            self._core_optimizer.step()
            self._core_optimizer.zero_grad()
            return {"core": total_loss.item()}
        return {"core": 0.0}

    def _train_step_streamed_core_only(self, x: torch.Tensor, target: torch.Tensor, t: torch.Tensor,
                                        domain_ids: torch.Tensor | None = None,
                                        padding_mask: torch.Tensor | None = None) -> dict[str, float]:
        """Streamed version of core-only training."""
        if self.canvas is None:
            return {"core": 0.0}
        if x.dtype == torch.long:
            labels_ids = target.detach().clone() if target.dtype == torch.long else x.detach().clone()
            if padding_mask is not None:
                labels_ids[~padding_mask] = -100
            # Parallel canvases: run N trajectories, pick best
            if self.parallel_canvases > 1:
                best_loss_val = float('inf')
                best_logits = None
                for _ in range(self.parallel_canvases):
                    t_i = t + torch.randn_like(t) * 0.15
                    logits_i = self.canvas.model(x, t_i.view(-1).float())
                    loss_i = F.cross_entropy(logits_i.view(-1, logits_i.size(-1)), labels_ids.view(-1), ignore_index=-100)
                    if loss_i.item() < best_loss_val:
                        best_loss_val = loss_i.item()
                        best_logits = logits_i
                logits = best_logits
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels_ids.view(-1), ignore_index=-100)
            else:
                logits = self.canvas.model(x, t.view(-1).float())
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels_ids.view(-1), ignore_index=-100)
            x_emb = self._embed_tokens(x)
            latent = self.latent_space(x_emb)
            recon = self.latent_space.project_tokens(x_emb)
            recon_loss = F.mse_loss(latent.mean(dim=1), recon.mean(dim=1))
            total_loss = loss + 0.05 * recon_loss
            mtp_loss = self._compute_mtp_loss(logits.detach(), labels_ids)
            if mtp_loss.item() > 0:
                total_loss = total_loss + mtp_loss.item()
            total_loss.backward()
            params = list(self.canvas.model.parameters()) + list(self.latent_space.parameters())
            if not hasattr(self, '_core_optimizer'):
                self._core_optimizer = torch.optim.AdamW(params, lr=self.lr)
            self._core_optimizer.step()
            self._core_optimizer.zero_grad()
            return {"core": total_loss.item()}
        return {"core": 0.0}

    def _train_step(self, x: torch.Tensor, target: torch.Tensor, t: torch.Tensor,
                     domain_ids: torch.Tensor | None = None,
                     padding_mask: torch.Tensor | None = None) -> dict[str, float]:
        if self.core_only:
            return self._train_step_core_only(x, target, t, domain_ids, padding_mask)
        if x.dtype == torch.long:
            labels_ids = target.detach().clone() if target.dtype == torch.long else x.detach().clone()
            if padding_mask is not None:
                labels_ids[~padding_mask] = -100
            clean = self._embed_tokens(target) if target.dtype != torch.long else self._embed_tokens(target)
            x_emb = self._embed_tokens(x)
            noise = torch.randn_like(x_emb)
            noise_scale = t.view(-1, 1, 1).expand_as(x_emb) if t.dim() > 1 else t.view(-1, 1, 1).expand_as(x_emb)
            x = (x_emb + noise * noise_scale).detach()
            target = clean.detach()
        else:
            labels_ids = None
        active = self._active_nodes(x)
        node_losses: dict[str, float] = {}
        expert_outputs: list[torch.Tensor] = []
        expert_confidences: list[float] = []
        expert_latents: list[torch.Tensor] = []

        if not active:
            nodes_list = list(self.router.nodes.values())
            selected = self._select_nodes_for_step(nodes_list)
            for node in selected:
                self.lifecycle_manager.register(node.node_id, self.step)
                block = self._ensure_block(node.node_id)
                block = block.to(x.device)
                block.train()
                if block.optimizer is None:
                    block.configure_optimizer(lr=self.lr)
                pred = block(x, t)
                loss_val = block.local_step(pred, target, t=t)
                node_losses[node.node_id] = loss_val
                node.update_loss(loss_val)
                self.lifecycle_manager.activate(node.node_id, self.step, accuracy=1.0 - loss_val)
                if self.dynamic_quantizer:
                    action = self.dynamic_quantizer.step(node.node_id, block, loss_val)
                    if action == "quantize":
                        quantize_expert_block(block)
                    elif action == "revert":
                        dequantize_expert_block(block)
                if self.expert_registry:
                    self.expert_registry.register(node.node_id, domain=self.domain or "general",
                                                   step=self.step, metadata={"loss": loss_val})
                if self.mesh_memory:
                    with torch.no_grad():
                        mem_emb = pred.mean(dim=1)
                    self.mesh_memory.insert(mem_emb, node.node_id, metadata={"loss": loss_val, "step": self.step})
                expert_outputs.append(pred.detach())
                expert_confidences.append(max(0.0, 1.0 - loss_val))
                latent = self.latent_space.to(pred.device)(pred.mean(dim=1, keepdim=True))
            expert_latents.append(latent)

            # Latent consistency loss
            if len(expert_latents) >= 2:
                lc_loss = self.router.latent_consistency_loss(expert_latents)
                for nid in node_losses:
                    node_losses[nid] = node_losses[nid] + 0.05 * lc_loss.item()

            # GCL pass if 2+ experts
            if len(expert_outputs) >= 2:
                _, gcl_info = self.global_cognitive_layer(
                    expert_outputs, expert_confidences, return_consensus=True
                )
                if gcl_info.get("consensus"):
                    con = gcl_info["consensus"]
                    if con.disagreements:
                        pass  # Could log disagreement info

            mtp_loss = self._compute_mtp_loss(pred.detach(), labels_ids if labels_ids is not None else x)
            if mtp_loss.item() > 0:
                for nid in node_losses:
                    node_losses[nid] = node_losses[nid] + mtp_loss.item()

            return node_losses

        selected = self._select_nodes_for_step(active)
        for node_id, node, _score in selected:
            self.lifecycle_manager.register(node_id, self.step)
            block = self._ensure_block(node_id)
            block = block.to(x.device)
            block.train()
            if block.optimizer is None:
                block.configure_optimizer(lr=self.lr)
            pred = block(x, t)
            loss_val = block.local_step(pred, target, t=t)
            node_losses[node_id] = loss_val
            node.update_loss(loss_val)
            self.lifecycle_manager.activate(node_id, self.step, accuracy=1.0 - loss_val)
            if self.dynamic_quantizer:
                action = self.dynamic_quantizer.step(node_id, block, loss_val)
                if action == "quantize":
                    quantize_expert_block(block)
                elif action == "revert":
                    dequantize_expert_block(block)
            if self.expert_registry:
                self.expert_registry.register(node_id, domain=self.domain or "general",
                                               step=self.step, metadata={"loss": loss_val})
            if self.mesh_memory:
                with torch.no_grad():
                    mem_emb = pred.mean(dim=1)
                self.mesh_memory.insert(mem_emb, node_id, metadata={"loss": loss_val, "step": self.step})
            if pred.dim() == 2:
                expert_outputs.append(pred.detach())
                latent = self.latent_space.to(pred.device)(pred.unsqueeze(1))
            else:
                expert_outputs.append(pred.detach().mean(dim=1))
                latent = self.latent_space.to(pred.device)(pred.mean(dim=1, keepdim=True))
            expert_confidences.append(max(0.0, 1.0 - loss_val))
            expert_latents.append(latent)
            if self.external_nodes:
                save_expert_block(block, self.expert_nodes_dir, node_id)

        if len(expert_latents) >= 2:
            lc_loss = self.router.latent_consistency_loss(expert_latents)
            for nid in node_losses:
                node_losses[nid] = node_losses[nid] + 0.05 * lc_loss.item()

        if len(expert_outputs) >= 2:
            _, gcl_info = self.global_cognitive_layer(
                expert_outputs, expert_confidences, return_consensus=True
            )
            if gcl_info.get("consensus"):
                con = gcl_info["consensus"]
                if con.disagreements:
                    pass

        mtp_loss = self._compute_mtp_loss(pred.detach(), labels_ids if labels_ids is not None else x)
        if mtp_loss.item() > 0:
            for nid in node_losses:
                node_losses[nid] = node_losses[nid] + mtp_loss.item()

        return node_losses

    def _ensure_block(self, node_id: str) -> NoPropBlock:
        block = self.tier_manager.get_block(node_id)
        if block is not None:
            return block
        return self.tier_manager.ensure_gpu(node_id)

    def _pin_block_weights(self, node_id: str):
        block = self.tier_manager.get_block(node_id)
        if block is None:
            return
        state = {}
        for k, v in block.state_dict().items():
            state[k] = v.detach().cpu().pin_memory()
        self._pinned_weights[node_id] = state

    def _prefetch_to_stream(self, node_id: str, device: torch.device):
        if self.prefetch_stream is None:
            self.prefetch_stream = torch.cuda.Stream()
        if node_id not in self._pinned_weights:
            return
        stream = self.prefetch_stream
        with torch.cuda.stream(stream):
            block = self.tier_manager.get_block(node_id)
            if block is not None:
                block.to(device)

    def _sync_prefetch(self):
        if self.prefetch_stream is not None:
            self.prefetch_stream.synchronize()

    def _train_step_streamed(self, x: torch.Tensor, target: torch.Tensor, t: torch.Tensor,
                              domain_ids: torch.Tensor | None = None,
                              padding_mask: torch.Tensor | None = None) -> dict[str, float]:
        if self.core_only:
            return self._train_step_streamed_core_only(x, target, t, domain_ids, padding_mask)
        pf = self._profiler
        if x.dtype == torch.long:
            _ce = pf and pf.tick_component("embed")
            labels_ids = target.detach().clone() if target.dtype == torch.long else x.detach().clone()
            if padding_mask is not None:
                labels_ids[~padding_mask] = -100
            clean = self._embed_tokens(target) if target.dtype != torch.long else self._embed_tokens(target)
            x_emb = self._embed_tokens(x)
            noise = torch.randn_like(x_emb)
            noise_scale = t.view(-1, 1, 1).expand_as(x_emb) if t.dim() > 1 else t.view(-1, 1, 1).expand_as(x_emb)
            x = (x_emb + noise * noise_scale).detach()
            target = clean.detach()
            if _ce: _ce()
        else:
            labels_ids = None
        _cr = pf and pf.tick_component("routing")
        active = self._active_nodes(x)
        if _cr: _cr()
        if pf and active:
            entropy = self.router.compute_routing_entropy() if hasattr(self.router, 'compute_routing_entropy') else 0.0
            pf.observe_router_stats(len(active), entropy, [nid for nid, _, _ in active])
        node_losses: dict[str, float] = {}
        expert_outputs: list[torch.Tensor] = []
        expert_confidences: list[float] = []
        expert_latents: list[torch.Tensor] = []

        if not active:
            all_ids = list(self.router.nodes.keys())
            selected = self._select_nodes_for_step(
                [(nid, self.router.nodes[nid], None) for nid in all_ids]
            )
            selected_ids = [sid for sid, _, _ in selected]
            for i, node_id in enumerate(selected_ids):
                self.lifecycle_manager.register(node_id, self.step)
                block = self._ensure_block(node_id)
                block = block.to(x.device)
                block.train()
                if block.optimizer is None:
                    block.configure_optimizer(lr=self.lr)
                _cb = pf and pf.tick_component("block_forward")
                pred = block(x, t)
                if _cb: _cb()
                _cl = pf and pf.tick_component("block_local_step")
                loss_val = block.local_step(pred, target, t=t)
                if _cl: _cl()
                node_losses[node_id] = loss_val
                self.router.nodes[node_id].update_loss(loss_val)
                self.lifecycle_manager.activate(node_id, self.step, accuracy=1.0 - loss_val)
                if self.dynamic_quantizer:
                    action = self.dynamic_quantizer.step(node_id, block, loss_val)
                    if action == "quantize":
                        quantize_expert_block(block)
                        print(f"  Quantized {node_id} to INT8")
                    elif action == "revert":
                        dequantize_expert_block(block)
                        print(f"  Reverted {node_id} to BF16")
                if self.expert_registry:
                    self.expert_registry.register(node_id, domain=self.domain or "general",
                                                   step=self.step, metadata={"loss": loss_val})
                if self.mesh_memory:
                    with torch.no_grad():
                        mem_emb = pred.mean(dim=1)
                    self.mesh_memory.insert(mem_emb, node_id, metadata={"loss": loss_val, "step": self.step})
            if pred.dim() == 2:
                expert_outputs.append(pred.detach())
                latent = self.latent_space.to(pred.device)(pred.unsqueeze(1))
            else:
                expert_outputs.append(pred.detach().mean(dim=1))
                latent = self.latent_space.to(pred.device)(pred.mean(dim=1, keepdim=True))
            expert_latents.append(latent)
            self._pin_block_weights(node_id)
            next_id = selected_ids[i + 1] if i + 1 < len(selected_ids) else None
            if next_id is not None and torch.cuda.is_available():
                self._prefetch_to_stream(next_id, x.device)
            self._sync_prefetch()

            if len(expert_latents) >= 2:
                lc_loss = self.router.latent_consistency_loss(expert_latents)
                for nid in node_losses:
                    node_losses[nid] = node_losses[nid] + 0.05 * lc_loss.item()

            if len(expert_outputs) >= 2:
                _, gcl_info = self.global_cognitive_layer(
                    expert_outputs, expert_confidences, return_consensus=True
                )

            mtp_loss = self._compute_mtp_loss(pred.detach(), labels_ids if labels_ids is not None else x)
            if mtp_loss.item() > 0:
                for nid in node_losses:
                    node_losses[nid] = node_losses[nid] + mtp_loss.item()

            return node_losses

        selected = self._select_nodes_for_step(active)
        selected_ids = [sid for sid, _, _ in selected]
        for i, node_id in enumerate(selected_ids):
            node = self.router.nodes[node_id]
            self.lifecycle_manager.register(node_id, self.step)
            block = self._ensure_block(node_id)
            block = block.to(x.device)
            block.train()
            if block.optimizer is None:
                block.configure_optimizer(lr=self.lr)
            _cb = pf and pf.tick_component("block_forward")
            pred = block(x, t)
            if _cb: _cb()
            _cl = pf and pf.tick_component("block_local_step")
            loss_val = block.local_step(pred, target, t=t)
            if _cl: _cl()
            node_losses[node_id] = loss_val
            node.update_loss(loss_val)
            self.lifecycle_manager.activate(node_id, self.step, accuracy=1.0 - loss_val)
            if self.dynamic_quantizer:
                action = self.dynamic_quantizer.step(node_id, block, loss_val)
                if action == "quantize":
                    quantize_expert_block(block)
                    print(f"  Quantized {node_id} to INT8")
                elif action == "revert":
                    dequantize_expert_block(block)
                    print(f"  Reverted {node_id} to BF16")
            if self.expert_registry:
                self.expert_registry.register(node_id, domain=self.domain or "general",
                                               step=self.step, metadata={"loss": loss_val})
            if self.mesh_memory:
                with torch.no_grad():
                    mem_emb = pred.mean(dim=1)
                self.mesh_memory.insert(mem_emb, node_id, metadata={"loss": loss_val, "step": self.step})
            if pred.dim() == 2:
                expert_outputs.append(pred.detach())
                latent = self.latent_space.to(pred.device)(pred.unsqueeze(1))
            else:
                expert_outputs.append(pred.detach().mean(dim=1))
                latent = self.latent_space.to(pred.device)(pred.mean(dim=1, keepdim=True))
            expert_confidences.append(max(0.0, 1.0 - loss_val))
            expert_latents.append(latent)
            self._pin_block_weights(node_id)
            next_id = selected_ids[i + 1] if i + 1 < len(selected_ids) else None
            if next_id is not None and torch.cuda.is_available():
                self._prefetch_to_stream(next_id, x.device)
            self._sync_prefetch()

        if len(expert_latents) >= 2:
            lc_loss = self.router.latent_consistency_loss(expert_latents)
            for nid in node_losses:
                node_losses[nid] = node_losses[nid] + 0.05 * lc_loss.item()

        if len(expert_outputs) >= 2:
            _, gcl_info = self.global_cognitive_layer(
                expert_outputs, expert_confidences, return_consensus=True
            )

        mtp_loss = self._compute_mtp_loss(pred.detach(), labels_ids if labels_ids is not None else x)
        if mtp_loss.item() > 0:
            for nid in node_losses:
                node_losses[nid] = node_losses[nid] + mtp_loss.item()

        return node_losses

    def _compute_mtp_loss(self, hidden: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.mtp_weight <= 0:
            return torch.tensor(0.0, device=hidden.device)
        self.speculator = self.speculator.to(hidden.device)
        if hidden.dim() == 2:
            h = hidden.unsqueeze(1)
        else:
            h = hidden.mean(dim=1, keepdim=True).expand(hidden.size(0), -1, -1)
        target_tokens = labels
        if target_tokens.dtype != torch.long and target_tokens.dtype != torch.uint8:
            target_tokens = target_tokens.argmax(dim=-1, keepdim=True).long()
        if target_tokens.dim() == 1:
            target_tokens = target_tokens.unsqueeze(-1)
        if target_tokens.size(-1) < self.num_draft_tokens:
            pad = target_tokens[:, -1:].expand(-1, self.num_draft_tokens - target_tokens.size(-1))
            target_tokens = torch.cat([target_tokens, pad], dim=-1)
        return self.mtp_weight * self.speculator.loss(h, target_tokens[:, :self.num_draft_tokens])

    def _check_mitosis(self):
        spawned: list[str] = []
        for node_id in list(self.router.nodes.keys()):
            child_id = self.router.check_mitosis(node_id)
            if child_id is not None:
                child = self.router.nodes[child_id]
                parent = self.router.nodes.get(node_id)
                block = NoPropBlock(self.embed_dim, num_heads=4)
                if parent is not None:
                    parent_block = getattr(parent, "_block", None)
                    if parent_block is not None:
                        block.load_state_dict(parent_block.state_dict())
                inject_lora_into_block(block, rank=16, alpha=16)
                child.__dict__["_block"] = block
                spawned.append(child_id)
        return spawned

    def _check_merge_prune(self, merge_threshold: float = 0.95, prune_idle: int = 5000):
        if len(self.router.nodes) < 3:
            return [], []
        merged = self.router.merge_similar(similarity_threshold=merge_threshold)
        pruned = self.router.prune_dead(max_idle_steps=prune_idle,
                                         min_loss_window=10 if self.step > 100 else 0)
        return merged, pruned

    # ── multi-notebook helpers ────────────────────────────────────────

    def freeze_experts(self, expert_ids: list[str] | None = None):
        """Freeze (requires_grad=False) specific expert blocks, or all if None."""
        ids = expert_ids or list(self.router.nodes.keys())
        for eid in ids:
            block = self.tier_manager.get_block(eid)
            if block is not None:
                block.requires_grad_(False)
                block.optimizer = None  # drop optimizer — will be recreated with only unfrozen params

    def unfreeze_experts(self, expert_ids: list[str] | None = None):
        """Unfreeze (requires_grad=True) specific expert blocks, or all if None."""
        ids = expert_ids or list(self.router.nodes.keys())
        for eid in ids:
            block = self.tier_manager.get_block(eid)
            if block is not None:
                block.requires_grad_(True)
                block.optimizer = None  # force re-creation on next local_step

    def save_expert_weights(self, expert_id: str, save_dir: str) -> str:
        """Save a single expert's block weights + metadata to *save_dir*/expert_<id>.pt.
        Returns the path saved."""
        os.makedirs(save_dir, exist_ok=True)
        block = self.tier_manager.get_block(expert_id)
        node = self.router.nodes.get(expert_id)
        path = os.path.join(save_dir, f"expert_{expert_id}.pt")
        torch.save({
            "node_id": expert_id,
            "block_state": block.state_dict() if block else {},
            "anchor": node.anchor_embedding.cpu() if node and node.anchor_embedding is not None else None,
            "rolling_loss": node.rolling_loss if node else [],
            "metadata": {
                "domain": getattr(node, "domain", ""),
                "version": getattr(node, "version", "1.0"),
            },
        }, path)
        return path

    def load_expert_weights(self, expert_id: str, load_dir: str) -> bool:
        """Load a single expert's block weights from *load_dir*/expert_<id>.pt.
        Returns True on success."""
        path = os.path.join(load_dir, f"expert_{expert_id}.pt")
        if not os.path.isfile(path):
            return False
        data = torch.load(path, map_location="cpu", weights_only=True)
        block = self._ensure_block(expert_id)
        if block is not None and "block_state" in data and data["block_state"]:
            block.load_state_dict(data["block_state"])
        node = self.router.nodes.get(expert_id)
        if node is not None:
            anchor = data.get("anchor")
            if anchor is not None:
                node.anchor_embedding = anchor
            node.rolling_loss = data.get("rolling_loss", [])
        return True

    def push_expert_shard(self, hub_sync: "HubSync", expert_ids: list[str]):
        """Push this notebook's expert shard to HF Hub with a shard tag."""
        shard_dir = os.path.join(self.checkpoint_dir, f"shard_{hub_sync.notebook_id}")
        for eid in expert_ids:
            self.save_expert_weights(eid, shard_dir)
        hub_sync.push(shard_dir, f"shard_{hub_sync.notebook_id}")
        print(f"  Pushed expert shard ({len(expert_ids)} experts) to {hub_sync.repo_id}")

    def pull_core_from_hub(self, hub_sync: "HubSync") -> bool:
        """Pull the latest master checkpoint and load only core weights.
        Returns True if successfully loaded."""
        latest = hub_sync.checkout_latest(self.checkpoint_dir)
        if latest is None:
            return False
        data = torch.load(latest, map_location="cpu", weights_only=True)
        mesh_state = data.get("model_state_dict", {}).get("mesh", {})
        if not mesh_state:
            mesh_state = data
        canvas_state = mesh_state.get("canvas_state")
        if canvas_state is not None and self.canvas is not None:
            self.canvas.model.load_state_dict(canvas_state)
        # Load expert anchors/router_state but NOT block weights (workers train those)
        router_state = mesh_state.get("router_state", {})
        for node_id, st in router_state.items():
            node = self.router.nodes.get(node_id)
            if node is not None:
                anchor = st.get("anchor")
                if anchor is not None:
                    node.anchor_embedding = anchor
                node.rolling_loss = st.get("rolling_loss", [])
        print(f"  Pulled core checkpoint from {hub_sync.repo_id} (step {mesh_state.get('step', '?')})")
        return True

    def train(
        self,
        dataset: Dataset,
        num_epochs: int = 10,
        batch_size: int = 8,
        log_interval: int = 10,
        val_interval: int = 500,
        mitosis_interval: int = 50,
        ckpt_interval: int = 100,
        resume: bool = True,
        collate_fn=None,
        max_steps: int = 0,
        use_packing: bool = False,
        domain_ids: list[str] | None = None,
        dashboard=None,
        val_dataset: Dataset | None = None,
        hub_repo: str | None = None,
        multi_notebook: bool = False,
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tier_manager.set_device(device)

        # HubSync initialisation
        hub_repo = hub_repo or self.hub_repo
        if hub_repo:
            try:
                self.hub_sync = HubSync(repo_id=hub_repo)
                if multi_notebook:
                    self.hub_sync.advertise()
                latest = self.hub_sync.checkout_latest(self.checkpoint_dir)
                if latest:
                    print(f"  HubSync: pulled {os.path.basename(latest)} from {hub_repo}")
            except Exception as e:
                print(f"  HubSync init skipped ({e})")

        if resume:
            self._load_checkpoint()

        self.token_embedding = self.token_embedding.to(device)
        self.lm_head = self.lm_head.to(device)
        self.router = self.router.to(device)
        self.latent_space = self.latent_space.to(device)
        self.speculator = self.speculator.to(device)
        self.kv_compressor = self.kv_compressor.to(device)
        self.global_cognitive_layer = self.global_cognitive_layer.to(device)
        if self.canvas is not None:
            self.canvas = self.canvas.to(device)

        if self.train_experts_only:
            for p in self.token_embedding.parameters(): p.requires_grad = False
            for p in self.lm_head.parameters(): p.requires_grad = False
            for p in self.latent_space.parameters(): p.requires_grad = False
            for p in self.speculator.parameters(): p.requires_grad = False
            for p in self.kv_compressor.parameters(): p.requires_grad = False
            for p in self.global_cognitive_layer.parameters(): p.requires_grad = False
            if self.canvas is not None:
                for p in self.canvas.parameters(): p.requires_grad = False
            print("Core params frozen — training experts only")

        from torch.utils.data import IterableDataset
        is_iterable = isinstance(dataset, IterableDataset)

        if use_packing:
            profiler = MeshProfiler(log_interval=log_interval)
            bucket_iter = AsyncPrefetchTokenBucketIterator(
                dataset=dataset,
                canvas_len=self.canvas_len,
                eos_id=self.vocab_size - 1,
                max_canvases=batch_size,
                dynamic_budget=True,
                min_budget=256,
                max_budget=getattr(self, '_token_budget', 8192),
                prefetch_queue_size=2,
                shuffle=not is_iterable,
            )
            bucket_iter.set_profiler(profiler)
            self._profiler = profiler
            print(f"AsyncPrefetchTokenBucketIterator: canvas_len={self.canvas_len}, "
                  f"max_canvases={batch_size}, dynamic budget, "
                  f"domain-queued packing, async prefetch")
            loader = DataLoader(dataset, batch_size=1, collate_fn=lambda x: x[0] if x else {})
            total_batches = len(bucket_iter) if not is_iterable else 999999
        else:
            bucket_iter = None
            profiler = None
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=(not is_iterable and collate_fn is None), collate_fn=collate_fn)
            total_batches = len(loader) if not is_iterable else 999999

        _step_times = deque(maxlen=100)
        _step_n_tokens = deque(maxlen=100)
        total_steps_planned = max_steps if max_steps > 0 else total_batches * num_epochs
        train_start_time = time.time()

        for epoch in range(num_epochs):
            epoch_losses: list[float] = []
            self._domain_ids = domain_ids or []
            data_source = bucket_iter if use_packing else loader
            max_batches = min(total_batches, max_steps) if max_steps > 0 else total_batches
            for batch_idx, batch in enumerate(data_source):
                if max_steps > 0 and self.step >= max_steps:
                    break
                if profiler:
                    profiler.tick_end("data_prefetch")
                    profiler.tick_start("forward")
                padding_mask = None
                segment_ids = None
                if isinstance(batch, dict):
                    x = batch.get('input_ids', batch.get('x', batch.get('tokens')))
                    target = batch.get('labels', batch.get('target', x))
                    t = batch.get('t', batch.get('timestep', torch.zeros(1)))
                    domain_ids = batch.get('domain_ids', batch.get('domain'))
                    padding_mask = batch.get('padding_mask')
                    segment_ids = batch.get('segment_ids')
                    x = torch.as_tensor(x)
                    target = torch.as_tensor(target)
                    if not isinstance(t, torch.Tensor):
                        t = torch.tensor(t)
                    if x.dim() == 1:
                        x = x.unsqueeze(0)
                    if target.dim() == 1:
                        target = target.unsqueeze(0)
                elif isinstance(batch, (list, tuple)):
                    x = batch[0]
                    target = batch[1] if len(batch) > 1 else x
                    t = batch[2] if len(batch) > 2 else torch.zeros(x.size(0), 1)
                    domain_ids = batch[4] if len(batch) >= 5 else None
                else:
                    x, target, t = batch, batch, torch.zeros(batch.size(0), 1)
                    domain_ids = None
                # Track data stall: time from batch request to batch available
                x = x.to(device)
                target = target.to(device) if target is not None else x
                t = t.to(device).float()
                if t.dim() == 1:
                    t = t.view(-1, 1)
                if padding_mask is not None:
                    padding_mask = padding_mask.to(device)
                if segment_ids is not None:
                    segment_ids = segment_ids.to(device)
                    if padding_mask is not None:
                        _loss_mask = ~padding_mask
                    else:
                        _loss_mask = None

                _t0 = time.perf_counter()
                if self.prefetch_stream is not None or torch.cuda.is_available():
                    node_losses = self._train_step_streamed(x, target, t, domain_ids=domain_ids,
                                                           padding_mask=_loss_mask if segment_ids is not None else None)
                else:
                    node_losses = self._train_step(x, target, t, domain_ids=domain_ids,
                                                   padding_mask=_loss_mask if segment_ids is not None else None)
                _dt = time.perf_counter() - _t0

                if profiler:
                    profiler.tick_end("forward")
                    profiler.tick_start("data_prefetch")
                avg_loss = sum(node_losses.values()) / max(len(node_losses), 1)
                epoch_losses.append(avg_loss)
                self.global_losses.append(avg_loss)

                _n_tok = x.numel()
                _step_times.append(_dt)
                _step_n_tokens.append(_n_tok)

                self.step += 1

                if dashboard:
                    router_entropy = getattr(self.router, 'compute_routing_entropy', lambda: 0.0)()
                    dashboard.log_step(
                        loss=avg_loss, lr=self.lr, vram_gb=torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
                        tok_s=tok_s, num_gpu_experts=len(node_losses), router_entropy=router_entropy,
                        expert_losses=node_losses, phase=getattr(self, '_current_phase', ''),
                    )

                if max_steps > 0 and self.step >= max_steps:
                    print(f"Reached max steps {max_steps}, stopping.")
                    self._save_checkpoint(final=True)
                    if use_packing and bucket_iter is not None:
                        bucket_iter.stop()
                    return

                if self._interrupted:
                    print("\n=== Graceful stop. Saving checkpoint... ===")
                    self._save_checkpoint(final=True)
                    if use_packing and bucket_iter is not None:
                        bucket_iter.stop()
                    print("=== Done. You can resume later with resume=True ===")
                    return

                if self.step % log_interval == 0 and not use_packing:
                    mem_stats = self.memory_manager.get_stats()
                    avg_s = sum(_step_times) / max(len(_step_times), 1)
                    avg_tok = sum(_step_n_tokens) / max(len(_step_n_tokens), 1)
                    tok_s = avg_tok / avg_s if avg_s > 0 else 0
                    elapsed = time.time() - train_start_time
                    steps_done = self.step - (epoch * total_batches + batch_idx) + 1
                    remaining = total_steps_planned - self.step
                    eta_s = remaining * avg_s if remaining > 0 else 0
                    eta_str = f"{int(eta_s//3600):02d}:{int((eta_s%3600)//60):02d}:{int(eta_s%60):02d}" if eta_s < 86400 else f"{eta_s/3600:.1f}h"
                    if self.core_only:
                        print(
                            f"Epoch {epoch+1}/{num_epochs}  Batch {batch_idx+1}/{total_batches}  "
                            f"Step {self.step}  CoreLoss {avg_loss:.6f}  "
                            f"Canvas {self.canvas_len}x{self.canvas_steps}  "
                            f"{tok_s:,.0f} tok/s  {avg_s*1000:.1f}ms/step  ETA {eta_str}"
                        )
                    else:
                        tiers = self.tier_manager.summary()
                        n_experts = len(node_losses)
                        hit_rate = 0.0
                        if self.tier_manager._ops_log:
                            gpu_loads = sum(1 for o in self.tier_manager._ops_log if "gpu" in o.get("op", ""))
                            total = len(self.tier_manager._ops_log)
                            hit_rate = (total - gpu_loads) / max(total, 1) * 100
                        print(
                            f"Epoch {epoch+1}/{num_epochs}  Batch {batch_idx+1}/{total_batches}  "
                            f"Step {self.step}  AvgLoss {avg_loss:.6f}  "
                            f"Experts {n_experts}  "
                            f"{tok_s:,.0f} tok/s  {avg_s*1000:.1f}ms/step  ETA {eta_str}  "
                            f"CacheHit {hit_rate:.0f}%  "
                            f"GPU[{tiers['gpu']}] RAM[{tiers['ram']}] Disk[{tiers['disk']}]  "
                            f"NodeIDs {list(node_losses.keys())}"
                        )

                # Latent Observatory — probe semantic nodes every report_interval
                if self.step % self.observatory.report_interval == 0:
                    with torch.no_grad():
                        probe_x = x[:1]
                        if probe_x.dtype == torch.long:
                            probe_x = self._embed_tokens(probe_x)
                        if probe_x.dim() == 2:
                            probe_x = probe_x.unsqueeze(1)
                        latent = self.latent_space(probe_x)
                        probe_results = self.observatory.probe_nodes(latent, top_k=3)
                        for nid, _ in self.router.nodes.items():
                            self.observatory.record_routing(nid, nid.split("_")[0] if "_" in nid else nid)
                        # CSV logging
                        if self.logger:
                            stability_scores = {
                                nid: self.observatory.semantic_stability.stability_score(nid)
                                for nid in probe_results
                            }
                            self.logger.log_latent_states(self.step, probe_results, stability_scores)
                            routes = [(eid, nid.split("_")[0] if "_" in nid else nid, 1.0)
                                      for eid in self.router.nodes for nid in [eid]]
                            self.logger.log_routing(self.step, routes)
                            self.logger.log_tier_ops_batch(self.step, self.tier_manager.drain_ops_log())
                    self.observatory.step_report(self.step)

                # Store training info in working memory
                self.memory_manager.store(
                    key=f"step_{self.step}",
                    content={"loss": avg_loss, "nodes": list(node_losses.keys())},
                    embedding=x.float().mean(dim=0),
                    tier=MemoryTier.WORKING,
                    importance=1.0 / (1.0 + avg_loss),
                )

                if self.step % mitosis_interval == 0:
                    spawned = self._check_mitosis()
                    if spawned:
                        print(f"Mitosis: new nodes {spawned}")
                        if self.logger:
                            for child_id in spawned:
                                self.logger.log_mitosis(
                                    self.step, child_id.rsplit("_v", 1)[0] if "_v" in child_id else "root",
                                    child_id, 0.0, 0.0,
                                )
                    merged, pruned = self._check_merge_prune(
                        merge_threshold=0.95,
                        prune_idle=max(100, self.step // 2),
                    )
                    if merged:
                        print(f"Merge: {len(merged)} pairs merged")
                    if pruned:
                        print(f"Prune: removed {pruned}")

                # Refresh Hub advertisement periodically (multi-notebook)
                if self.hub_sync is not None and multi_notebook and self.step % 50 == 0:
                    self.hub_sync.refresh_advertisement()

                # Validation
                if val_dataset is not None and self.step % val_interval == 0:
                    val_metrics = self._validate(val_dataset, device)
                    perplexity = math.exp(min(val_metrics["loss"], 20))
                    print(f"  VALIDATION — loss {val_metrics['loss']:.4f}  "
                          f"perplexity {perplexity:.2f}  "
                          f"top1 {val_metrics['top1']:.2%}  "
                          f"top5 {val_metrics['top5']:.2%}")
                    if dashboard:
                        dashboard.writer.add_scalar("eval/loss", val_metrics["loss"], self.step)
                        dashboard.writer.add_scalar("eval/perplexity", perplexity, self.step)
                        dashboard.writer.add_scalar("eval/top1_accuracy", val_metrics["top1"], self.step)
                        dashboard.writer.add_scalar("eval/top5_accuracy", val_metrics["top5"], self.step)

                if self.step % 50 == 0:
                    archived = self.lifecycle_manager.tick_idle(self.step)
                    if archived:
                        print(f"Lifecycle: archived {archived}")

                # Consolidate memory every 200 steps
                if self.step > 0 and self.step % 200 == 0:
                    self.memory_manager.consolidate()
                    self.memory_manager.forget()

                if self.step % ckpt_interval == 0:
                    self._save_checkpoint()

                del x, target, t
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

            avg_epoch = sum(epoch_losses) / max(len(epoch_losses), 1)
            print(f"=== Epoch {epoch+1} done — avg loss {avg_epoch:.6f} ===")

        if use_packing and bucket_iter is not None:
            bucket_iter.stop()
        self.tier_manager.sync_all()
        self._save_checkpoint(final=True)
        if self.logger:
            self.logger.log_tier_ops_batch(self.step, self.tier_manager.drain_ops_log())
            self.logger.close()
            print(self.logger.summary())
        tiers = self.tier_manager.summary()
        print(f"Training complete. {len(self.router.nodes)} nodes in mesh. "
              f"Tiers: GPU[{tiers['gpu']}] RAM[{tiers['ram']}] Disk[{tiers['disk']}]")

    def save_checkpoint(self, *args, **kwargs):
        self._save_checkpoint(*args, **kwargs)

    def _save_checkpoint(self, final: bool = False, latest: bool = False):
        if not final and not latest and self.step == self._last_saved_step:
            return
        state = {
            "router_state": {},
            "step": self.step,
            "global_losses": self.global_losses[-1000:],
            "mitosis_threshold": self.mitosis_threshold,
            "canvas_state": None,
            "external_nodes": self.external_nodes,
            "use_qat": getattr(self, '_qat_enabled', False),
            "qat_bits": getattr(self, 'qat_bits', 8),
        }
        if self.canvas is not None:
            state["canvas_state"] = self.canvas.model.state_dict()
        for node_id, node in self.router.nodes.items():
            self.tier_manager.sync_to_disk(node_id)
            entry = {
                "anchor": node.anchor_embedding.cpu(),
                "rolling_loss": node.rolling_loss,
            }
            state["router_state"][node_id] = entry
        if final:
            tag = "final"
        elif latest:
            tag = "latest"
        else:
            tag = self.step
        self._last_saved_step = self.step
        checkpoint_atomic(
            self.checkpoint_dir,
            tag,
            {"mesh": state},
            {},
            {"step": self.step, "final": final},
        )
        print(f"Checkpoint saved at step {self.step}")
        if self.hub_sync is not None:
            try:
                self.hub_sync.push(self.checkpoint_dir, tag, final=final)
                if not final:
                    self.hub_sync.push_metadata({"step": self.step, "loss": float(sum(self.global_losses[-100:])/max(len(self.global_losses[-100:]),1))})
            except Exception as e:
                print(f"  HubSync push skipped ({e})")

    def _load_checkpoint(self, step_tag: str | int | None = None):
        ckpt_dir = self.checkpoint_dir
        if not os.path.isdir(ckpt_dir):
            print("No checkpoint directory found, starting fresh")
            return
        if step_tag is not None:
            path = os.path.join(ckpt_dir, f"step_{step_tag}.pt")
            if not os.path.isfile(path):
                print(f"Specified checkpoint step_{step_tag}.pt not found, starting fresh")
                return
            ckpt_paths = [path]
        else:
            # Prefer: final > latest > highest numeric
            final_path = os.path.join(ckpt_dir, "step_final.pt")
            latest_path = os.path.join(ckpt_dir, "step_latest.pt")
            if os.path.isfile(final_path):
                ckpt_paths = [final_path]
            elif os.path.isfile(latest_path):
                ckpt_paths = [latest_path]
            else:
                pattern = os.path.join(ckpt_dir, "step_*.pt")
                ckpts = sorted(glob.glob(pattern))
                # Filter out named checkpoints, keep only numeric ones
                num_ckpts = [p for p in ckpts if os.path.basename(p).replace("step_", "").replace(".pt", "").isdigit()]
                if not num_ckpts:
                    print("No checkpoint found, starting fresh")
                    return
                ckpt_paths = sorted(num_ckpts, key=lambda p: int(os.path.basename(p).replace("step_", "").replace(".pt", "")))
        path = ckpt_paths[-1]
        print(f"Resuming from {path}")
        data = load_checkpoint(path)
        mesh_state = data.get("mesh", {}).get("mesh", {})
        if not mesh_state:
            return
        self.step = mesh_state.get("step", 0)
        self.global_losses = mesh_state.get("global_losses", [])
        canvas_state = mesh_state.get("canvas_state")
        if canvas_state is not None and self.canvas is not None:
            self.canvas.model.load_state_dict(canvas_state)
        router_state = mesh_state.get("router_state", {})
        external = mesh_state.get("external_nodes", self.external_nodes)
        for node_id, state in router_state.items():
            if node_id in self.router.nodes:
                node = self.router.nodes[node_id]
                anchor = state.get("anchor")
                if anchor is not None:
                    node.anchor_embedding = anchor.to(
                        node.anchor_embedding.device if node.anchor_embedding is not None else "cpu"
                    )
                node.rolling_loss = state.get("rolling_loss", [])
                if not external:
                    block = NoPropBlock(self.embed_dim, num_heads=4)
                    model_state = state.get("model")
                    if model_state:
                        block.load_state_dict(model_state)
                    block.configure_optimizer(lr=self.lr)
                    opt_state = state.get("optimizer")
                    if opt_state and block.optimizer:
                        block.optimizer.load_state_dict(opt_state)
                    node.__dict__["_block"] = block

    def _signal_handler(self, sig, frame):
        if self._interrupted:
            print("\n\n=== Second SIGINT — forcing exit. ===")
            sys.exit(130)
        print("\n\n=== SIGINT received. Will stop after current step... ===")
        print("=== Press Ctrl+C again to force exit. ===")
        self._interrupted = True

    def _validate(self, val_dataset: Dataset, device: torch.device) -> dict:
        self.router.eval()
        total_loss = 0.0
        total_correct_1 = 0
        total_correct_5 = 0
        total_seen = 0
        n_batches = 0
        loader = DataLoader(val_dataset, batch_size=8, shuffle=False, drop_last=False,
                            collate_fn=lambda b: {
                                "input_ids": torch.stack([x["input_ids"] if isinstance(x, dict) else x[0] for x in b]),
                                "labels": torch.stack([x["labels"] if isinstance(x, dict) else x[1] for x in b]),
                            })
        with torch.no_grad():
            for batch in loader:
                x = batch["input_ids"].to(device)
                target = batch["labels"].to(device)
                x_emb = self._embed_tokens(x)
                logits = self.router(x_emb, return_logits=True)
                if isinstance(logits, tuple):
                    logits = logits[0]
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1), reduction="mean")
                total_loss += loss.item()
                probs = F.softmax(logits, dim=-1)
                top5 = probs.topk(5, dim=-1).indices
                target_flat = target.view(-1, 1)
                correct_1 = (top5[:, :1] == target_flat).any(dim=-1).sum().item()
                correct_5 = (top5 == target_flat).any(dim=-1).sum().item()
                total_correct_1 += correct_1
                total_correct_5 += correct_5
                total_seen += target.numel()
                n_batches += 1
        self.router.train()
        avg_loss = total_loss / max(n_batches, 1)
        return {"loss": avg_loss, "top1": total_correct_1 / max(total_seen, 1), "top5": total_correct_5 / max(total_seen, 1)}

    def generate_text(self, batch_size: int = 1, max_blocks: int = 1,
                       device: torch.device | None = None) -> torch.Tensor:
        if self.canvas is None:
            raise RuntimeError(
                "DiffusionCanvas is not initialized. Set use_diffusion_canvas=True "
                "when constructing MeshTrainer."
            )
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.canvas = self.canvas.to(device)
        self.canvas.model.eval()
        return self.canvas.generate(batch_size=batch_size, device=device, max_blocks=max_blocks)

    def chat(self, prompt_ids: torch.Tensor,
             device: torch.device | None = None,
             max_new_tokens: int | None = None) -> torch.Tensor:
        if self.canvas is None:
            raise RuntimeError(
                "DiffusionCanvas is not initialized. Set use_diffusion_canvas=True "
                "when constructing MeshTrainer."
            )
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.canvas = self.canvas.to(device)
        self.canvas.model.eval()
        return self.canvas.generate_conditional(
            prompt_ids.to(device), device=device, max_new_tokens=max_new_tokens
        )

    def enable_qat(self, num_bits: int = 8):
        """Apply QAT to the CanvasTransformer backbone for INT8 calibration."""
        if self.canvas is None:
            print("QAT requires DiffusionCanvas (use --use-diffusion-canvas)")
            return
        apply_qat(self.canvas.model, num_bits=num_bits)
        self._qat_enabled = True
        print(f"QAT enabled ({num_bits}-bit). Export with export_model() for INT8.")

    def disable_qat(self):
        """Strip QAT wrappers, restore nn.Linear."""
        if self.canvas is not None and self._qat_enabled:
            strip_qat(self.canvas.model)
            self._qat_enabled = False
            print("QAT disabled, back to BF16.")

    def generate_from_text(self, prompt: str, max_new_tokens: int | None = None,
                           device: torch.device | None = None) -> str:
        """Tokenizes a text prompt, generates tokens via canvas, decodes."""
        tok = load_tokenizer()
        enc = tok(prompt, return_tensors="pt")
        prompt_ids = enc["input_ids"]
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        prompt_ids = prompt_ids.to(device)
        output_ids = self.chat(prompt_ids, device=device, max_new_tokens=max_new_tokens)
        return tok_decode(tok, output_ids[0])

    def infer(self, x: torch.Tensor, t: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = x.to(device)
        if t is None:
            t = torch.full((x.size(0), 1), 0.5, device=device)
        else:
            t = t.to(device)

        self.latent_space = self.latent_space.to(device)
        self.router = self.router.to(device)
        self.kv_compressor = self.kv_compressor.to(device)
        self.speculator = self.speculator.to(device)
        self.global_cognitive_layer = self.global_cognitive_layer.to(device)

        active = self._active_nodes(x)
        outputs: list[torch.Tensor] = []
        confidences: list[float] = []

        active_ids = [nid for nid, _, _ in active]
        for i, (node_id, node, score) in enumerate(active):
            block = self._ensure_block(node_id)
            block = block.to(device)
            block.eval()
            with torch.no_grad():
                out = block(x, t)
                out = self.kv_compressor.compress_attention(out, out, out)
            outputs.append(out)
            confidences.append(score)
            next_id = active_ids[i + 1] if i + 1 < len(active_ids) else None
            if next_id is not None and torch.cuda.is_available():
                self._prefetch_to_stream(next_id, device)
            self._sync_prefetch()

        if not outputs:
            output = torch.zeros_like(x)
        elif len(outputs) >= 2:
            # Use GCL for multi-expert fusion
            fused, gcl_info = self.global_cognitive_layer(outputs, confidences, return_consensus=True)
            output = fused
            # Store in episodic memory
            for i, (node_id, _, score) in enumerate(active):
                self.memory_manager.store(
                    key=f"infer_{node_id}_{self.step}",
                    content={"output_norm": outputs[i].norm().item(), "confidence": score},
                    embedding=outputs[i].mean(dim=0),
                    tier=MemoryTier.EPISODIC,
                    importance=score,
                )
        else:
            output = outputs[0]

        draft_tokens, confidence = self.speculator.speculate(output)
        return output, {
            "draft_tokens": draft_tokens,
            "confidence": confidence,
            "active_nodes": [(node_id, score) for node_id, _, score in active],
        }

    def summary(self):
        print(f"Mesh router: {len(self.router.nodes)} nodes")
        for node_id, node in self.router.nodes.items():
            block = getattr(node, "_block", None)
            params = sum(p.numel() for p in block.parameters()) if block else 0
            life_state = self.lifecycle_manager.get_state(node_id)
            life_str = life_state.value if life_state else "unknown"
            print(f"  {node_id}: {params:,} params, loss_window={len(node.rolling_loss)}, status={life_str}")
        print(f"Compressor: PolarQuant + QJL ({self.embed_dim}-dim)")
        print(f"Speculator: {self.num_draft_tokens}-token MTP heads")
        print(f"GlobalCognitiveLayer: {min(8, self.n_heads)}-head cross-expert attention")
        print(f"LifecycleManager: {self.lifecycle_manager.get_stats().get('total', 0)} tracked experts")
        mem_stats = self.memory_manager.get_stats()
        print(f"MemoryManager: WK={mem_stats.get('working', 0)} ST={mem_stats.get('short_term', 0)} LT={mem_stats.get('long_term', 0)} EP={mem_stats.get('episodic', 0)} SM={mem_stats.get('semantic', 0)}")
        if self.canvas is not None:
            print(f"DiffusionCanvas: {self.canvas_len}-token canvas, {self.canvas_steps} steps")

    def export_model(self, output_path: str, fmt: str = "safetensors", **kwargs):
        mesh_state = {
            "step": self.step,
            "global_losses": self.global_losses[-1000:],
            "mitosis_threshold": self.mitosis_threshold,
            "external_nodes": self.external_nodes,
            "router_state": {},
        }
        for node_id, node in self.router.nodes.items():
            block = getattr(node, "_block", None)
            if block is None:
                continue
            entry = {
                "anchor": node.anchor_embedding.cpu(),
                "rolling_loss": node.rolling_loss,
            }
            if not self.external_nodes:
                entry["model"] = block.state_dict()
            mesh_state["router_state"][node_id] = entry

        meta = {
            "name": kwargs.get("name", "noprop-mesh-model"),
            "embed_dim": self.embed_dim,
            "top_k": self.top_k,
            "mitosis_threshold": self.mitosis_threshold,
            "num_draft_tokens": self.num_draft_tokens,
            "vocab_size": self.vocab_size,
        }

        if fmt == "safetensors":
            used_fallback = export_to_safetensors(mesh_state, output_path, meta)
            tag = "safetensors (fallback .pt)" if used_fallback else "safetensors"
            print(f"Model exported to {output_path} [{tag}]")
        elif fmt == "onnx":
            if len(self.router.nodes) == 0:
                raise RuntimeError("No nodes to export")
            first_id = list(self.router.nodes.keys())[0]
            block = self._ensure_block(first_id)
            export_to_onnx(block, output_path, self.embed_dim)
            print(f"Model exported to {output_path} [ONNX]")
        elif fmt == "gguf":
            n = export_to_gguf(mesh_state, output_path, meta)
            print(f"Model exported to {output_path} [GGUF, {n} tensors]")
        else:
            raise ValueError(f"Unknown format: {fmt}. Use 'safetensors', 'onnx', 'gguf', or 'pt'.")

        if self.canvas is not None:
            canvas_path = output_path.rsplit(".", 1)[0] + "_canvas.pt"
            torch.save(self.canvas.model.state_dict(), canvas_path)
            print(f"Canvas model exported to {canvas_path}")


def main():
    import argparse
    valid_sizes = ", ".join(list_presets())
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", type=str, default="tiny",
                        help=f"Backbone size: {valid_sizes}")
    parser.add_argument("--embed-dim", type=int, default=None,
                        help="Override backbone embedding dim")
    parser.add_argument("--num-heads", type=int, default=None,
                        help="Override attention heads")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--nodes-dir", type=str, default="nodes")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/mesh")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--mitosis-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--canvas-len", type=int, default=512)
    parser.add_argument("--canvas-steps", type=int, default=50)
    parser.add_argument("--mtp-weight", type=float, default=0.1,
                        help="MTP auxiliary loss weight (0=disabled)")
    parser.add_argument("--curriculum-dir", type=str, default=None,
                        help="Path to curriculum JSONL dataset dir")
    parser.add_argument("--agk-dir", type=str, default=None,
                        help="Path to AGK .md dataset dir")
    parser.add_argument("--num-draft-tokens", type=int, default=3)
    parser.add_argument("--vocab-size", type=int, default=VOCAB_SIZE,
                        help=f"Vocabulary size (default: {VOCAB_SIZE} from Qwen3)")
    parser.add_argument("--online", type=str, default=None,
                        help="HF dataset path for online streaming (e.g. HuggingFaceFW/fineweb-edu)")
    parser.add_argument("--online-split", type=str, default="train",
                        help="HF dataset split for online streaming")
    parser.add_argument("--online-text-key", type=str, default="text",
                        help="HF dataset text field name")
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Max training steps (0 = use num_epochs)")
    parser.add_argument("--agentic", action="store_true",
                        help="Use agentic streaming (reasoning + tool-use + planning datasets)")
    parser.add_argument("--qat", action="store_true",
                        help="Enable Quantization-Aware Training (INT8 backbone)")
    parser.add_argument("--qat-bits", type=int, default=8,
                        help="QAT bit-width (default 8)")
    parser.add_argument("--no-external-nodes", action="store_true",
                        help="Disable external node files (embed in checkpoint)")
    parser.add_argument("--packing", action="store_true",
                        help="Enable token-based batching + packing for 2-3x throughput")
    parser.add_argument("--token-budget", type=int, default=0,
                        help="Token budget for packing (0=auto based on canvas_len*batch_size)")
    parser.add_argument("--max-experts", type=int, default=64,
                        help="Maximum number of MoE experts (default 64)")
    parser.add_argument("--max-gpu-experts", type=int, default=8,
                        help="Max experts on GPU at once (default 8)")
    parser.add_argument("--max-ram-experts", type=int, default=32,
                        help="Max experts in CPU RAM cache (default 32)")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="Directory for CSV instrumentation logs")
    parser.add_argument("--core-only", action="store_true",
                        help="Train core engine only (canvas + latent + speculator), no experts")
    parser.add_argument("--mix", type=str, default=None,
                        help="Comma-separated HF datasets with weights for mixed streaming, e.g. 'HuggingFaceFW/fineweb-edu:0.7,codeparrot/github-code:0.3'")
    parser.add_argument("--phase", type=str, default=None,
                        help="Training phase: core, domain, topic, specialist")
    parser.add_argument("--train-experts-only", action="store_true",
                        help="Freeze core engine, train only expert blocks")
    parser.add_argument("--domain", type=str, default=None,
                        help="Domain tag for training (e.g. 'code', 'math', 'reasoning')")
    parser.add_argument("--experts-count", type=int, default=0,
                        help="Initial number of expert nodes (0=auto from node files)")
    parser.add_argument("--datasets", type=str, default=None,
                        help="Alias for --mix: comma-separated HF datasets with weights")
    parser.add_argument("--parallel-canvases", type=int, default=1,
                        help="Number of parallel diffusion trajectories (pick best, default 1)")
    parser.add_argument("--dynamic-quant", action="store_true",
                        help="Enable dynamic per-expert quantization (BF16↔INT8)")
    parser.add_argument("--quant-patience", type=int, default=50,
                        help="Steps without improvement before quantizing an expert (default 50)")
    parser.add_argument("--expert-registry", type=str, default=None,
                        help="Path to expert registry JSON file")
    parser.add_argument("--mesh-memory", type=str, default=None,
                        help="Path to persistent mesh memory directory (FAISS + JSON)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    mix_spec = args.mix or args.datasets
    if mix_spec:
        from online_dataset import MixedOnlineDataset, collate_fn
        sources = []
        for entry in mix_spec.split(","):
            parts = entry.rsplit(":", 1)
            hf_path = parts[0]
            weight = float(parts[1]) if len(parts) > 1 else 1.0
            sources.append({"hf_path": hf_path, "weight": weight})
        dataset = MixedOnlineDataset(sources, max_seq_len=args.canvas_len)
        use_collate = collate_fn
    elif args.agentic:
        from agentic_dataset import AgenticStreamingDataset, collate_fn as agentic_collate
        dataset = AgenticStreamingDataset(max_seq_len=args.canvas_len)
        use_collate = agentic_collate
    elif args.online:
        from online_dataset import StreamingHFDataset, collate_fn
        dataset = StreamingHFDataset(
            hf_path=args.online,
            split=args.online_split,
            max_seq_len=args.canvas_len,
            text_key=args.online_text_key,
        )
        use_collate = collate_fn
        if args.domain:
            print(f"Filtering dataset by domain: {args.domain}")
    elif args.agk_dir is not None:
        dataset = AGKDataset(
            agk_dir=args.agk_dir,
            max_seq_len=args.canvas_len,
        )
        use_collate = None
    elif args.curriculum_dir is not None:
        dataset = CurriculumDataset(
            data_dir=args.curriculum_dir,
            max_seq_len=args.canvas_len,
            tokenizer_vocab_size=args.vocab_size,
        )
        use_collate = None
    else:
        dataset = SyntheticMeshDataset(
            num_samples=args.num_samples,
            embed_dim=args.embed_dim or get_preset(args.model_size).d_model,
            num_classes=10,
        )
        use_collate = None

    trainer = MeshTrainer(
        model_size=args.model_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        top_k=args.top_k,
        lr=args.lr,
        nodes_dir=args.nodes_dir,
        checkpoint_dir=args.checkpoint_dir,
        mitosis_threshold=args.mitosis_threshold,
        canvas_len=args.canvas_len,
        canvas_steps=args.canvas_steps,
        use_diffusion_canvas=True,
        mtp_weight=args.mtp_weight,
        num_draft_tokens=args.num_draft_tokens,
        vocab_size=args.vocab_size,
        external_nodes=not args.no_external_nodes,
        max_experts=args.max_experts,
        log_dir=args.log_dir,
        max_gpu_experts=args.max_gpu_experts,
        max_ram_experts=args.max_ram_experts,
        core_only=args.core_only,
        parallel_canvases=args.parallel_canvases,
        train_experts_only=args.train_experts_only,
        domain=args.domain,
        experts_count=args.experts_count,
        use_dynamic_quant=args.dynamic_quant,
        quant_patience=args.quant_patience,
        expert_registry_path=args.expert_registry,
        mesh_memory_path=args.mesh_memory,
    )

    if args.phase:
        print(f"Phase: {args.phase}")
    if args.token_budget > 0:
        trainer._token_budget = args.token_budget

    domain_list = []
    if args.domain:
        domain_list = [d.strip() for d in args.domain.split(",")]
    elif args.phase:
        domain_list = [args.phase]

    trainer.train(
        dataset=dataset,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        log_interval=5,
        mitosis_interval=10,
        ckpt_interval=20,
        resume=args.resume,
        collate_fn=use_collate if (args.online or args.agentic or mix_spec) else None,
        max_steps=args.max_steps,
        use_packing=args.packing,
        domain_ids=domain_list or None,
    )

    if args.qat and trainer.canvas is not None:
        print("Enabling QAT for INT8 calibration...")
        trainer.enable_qat(num_bits=args.qat_bits)
        print("QAT calibration done. Export with trainer.export_model() after training.")

    trainer.summary()

    tok = load_tokenizer()
    sample_text = "What is deep learning?"
    enc = tok(sample_text, return_tensors="pt")
    print(f"\nGenerated text from prompt: '{sample_text}'")
    print(f"Encoded: {enc['input_ids'][0].tolist()[:10]}... ({enc['input_ids'].size(1)} tokens)")


if __name__ == "__main__":
    main()
