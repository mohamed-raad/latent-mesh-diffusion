"""
Comprehensive scalability benchmark suite for mesh training pipeline.
Sweeps: expert count, canvas size, domain count, sequence variance.
Tracks real-time CPU/RAM/GPU + live loss curve.
"""
import csv, gc, os, sys, time, math, json, signal
from dataclasses import dataclass, field
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

torch.set_float32_matmul_precision("high")

from mesh_router import MeshRouter, MeshNode, UniversalLatentSpace, ExpertAdapter
from noprop_block import NoPropBlock
from dspark_speculator import DSparkSpeculator
from global_cognitive_layer import GlobalCognitiveLayer, ConsensusMechanism, ToolManager
from lifecycle_manager import LifecycleManager
from memory_manager import MemoryManager
from train_mesh import (
    SequencePacker,
    AsyncPrefetchTokenBucketIterator,
    MeshProfiler,
)
from model_sizes import get_preset

SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "benchmarks")


# ═══════════════════════════════════════════════════
# Live system monitor (CPU / RAM / GPU)
# ═══════════════════════════════════════════════════

class LiveMonitor:
    """Reads real-time CPU%, RAM%, GPU%, VRAM every sample_interval steps."""
    def __init__(self):
        self._psutil = None
        self._pynvml = None
        try:
            import psutil as _ps
            self._psutil = _ps
        except ImportError:
            pass
        try:
            import pynvml as _nv
            _nv.nvmlInit()
            self._pynvml = _nv
            self._nv_handle = _nv.nvmlDeviceGetHandleByIndex(0)
        except ImportError:
            pass

    def sample(self) -> dict:
        d = {"cpu_pct": 0.0, "ram_pct": 0.0, "ram_gb": 0.0, "gpu_pct": 0.0, "vram_gb": 0.0}
        if self._psutil:
            d["cpu_pct"] = self._psutil.cpu_percent(interval=0.1)
            d["ram_pct"] = self._psutil.virtual_memory().percent
            d["ram_gb"] = self._psutil.virtual_memory().used / 1e9
        if self._pynvml:
            try:
                util = self._pynvml.nvmlDeviceGetUtilizationRates(self._nv_handle)
                d["gpu_pct"] = util.gpu
                mem = self._pynvml.nvmlDeviceGetMemoryInfo(self._nv_handle)
                d["vram_gb"] = mem.used / 1e9
            except Exception:
                pass
        # Always get current VRAM via torch
        if torch.cuda.is_available():
            try:
                free, total = torch.cuda.mem_get_info()
                d["vram_gb"] = (total - free) / 1e9
            except Exception:
                pass
        return d


# ═══════════════════════════════════════════════════
# Parameterized synthetic dataset
# ═══════════════════════════════════════════════════

class ScaleDataset(Dataset):
    """Dataset with controllable domain count and sequence length range."""
    def __init__(self, n_samples: int = 5000, n_domains: int = 4,
                 seq_min: int = 20, seq_max: int = 60, vocab: int = 151643,
                 seed: int = 42):
        self.vocab = vocab
        self.seed = seed
        g = torch.Generator().manual_seed(seed)
        g2 = torch.Generator().manual_seed(seed + 1)
        self.data = []
        for i in range(n_samples):
            l = int(torch.randint(seq_min, seq_max + 1, (1,), generator=g).item())
            ids = torch.randint(4, vocab - 2, (l,), generator=g2)
            domain = str(hash(i) % n_domains)
            self.data.append({"input_ids": ids, "labels": ids, "domain": domain})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ═══════════════════════════════════════════════════
# Trainer factory with N synthetic experts
# ═══════════════════════════════════════════════════

def make_trainer(n_experts: int, canvas_len: int, embed_dim: int, vocab: int,
                 external_nodes: bool = False, device: str = "cuda"):
    """Build a MeshTrainer with N synthetic expert anchors (no .md files)."""
    from train_mesh import MeshTrainer

    t = MeshTrainer(
        model_size="tiny",
        embed_dim=embed_dim,
        vocab_size=vocab,
        canvas_len=canvas_len,
        top_k=3,
        lr=1e-4,
        external_nodes=external_nodes,
        checkpoint_dir=os.path.join(SAVE_DIR, "_tmp_ckpt"),
    )
    # Move all core components to device (trainer.train() normally does this)
    t.token_embedding = t.token_embedding.to(device)
    t.lm_head = t.lm_head.to(device)
    t.latent_space = t.latent_space.to(device)
    t.speculator = t.speculator.to(device)
    t.kv_compressor = t.kv_compressor.to(device)
    t.global_cognitive_layer = t.global_cognitive_layer.to(device)

    # Override the router with N synthetic experts
    d_model = t.embed_dim
    t.router = MeshRouter(top_k=3, d_model=d_model).to(device)
    for i in range(n_experts):
        anchor = F.normalize(torch.randn(d_model, device=device), dim=-1)
        node = MeshNode(
            node_id=f"expert_{i:04d}",
            anchor_path="",
            anchor_embedding=anchor,
            mitosis_threshold=0.5,
        )
        node.metadata.domain = str(i % max(1, n_experts))
        t.router.register_node(node)
    return t


# ═══════════════════════════════════════════════════
# Single benchmark run
# ═══════════════════════════════════════════════════

@dataclass
class BenchResult:
    label: str; n_experts: int; canvas_len: int; n_domains: int; seq_range: tuple
    steps: int; elapsed_s: float; step_s: float
    tok_s: float; vram_gb: float; gpu_pct: float; cpu_pct: float; ram_gb: float; ram_pct: float
    padding_pct: float; router_latency_ms: float; cache_hit_pct: float
    packing_occupancy_pct: float
    avg_loss: float; final_loss: float
    loss_curve: list = field(default_factory=list)


def run_bench(label: str, n_experts: int, canvas_len: int, n_domains: int,
              seq_min: int, seq_max: int, n_steps: int = 100,
              use_packing: bool = True, verbose: bool = True,
              device: str = "cuda") -> BenchResult:
    """Run a single training benchmark with live monitoring."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(42)

    embed_dim = get_preset("tiny").d_model
    vocab = 151643

    # Dataset
    ds = ScaleDataset(
        n_samples=5000, n_domains=n_domains,
        seq_min=seq_min, seq_max=seq_max, vocab=vocab,
    )

    # Trainer
    trainer = make_trainer(n_experts, canvas_len, embed_dim, vocab, device=device)
    trainer.step = 0
    trainer.global_losses = []

    monitor = LiveMonitor()
    sys_samples = []
    loss_curve = []
    router_latencies = []
    cache_hit_ratios = []
    prev_experts: set | None = None

    if use_packing:
        profiler = MeshProfiler(log_interval=999)
        it = AsyncPrefetchTokenBucketIterator(
            dataset=ds,
            canvas_len=canvas_len,
            eos_id=vocab - 1,
            max_canvases=2,
            dynamic_budget=True,
            min_budget=256,
            max_budget=8192,
            prefetch_queue_size=2,
            shuffle=True,
        )
        it.set_profiler(profiler)
        data_source = it
    else:
        profiler = None
        data_source = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=_pad_collate)

    start = time.time()
    step = 0
    for batch in data_source:
        if step >= n_steps:
            break

        # System sample
        sys_samples.append(monitor.sample())

        # Train step
        x = batch["input_ids"].to(device)
        target = batch["labels"].to(device)
        if x.dim() == 1:
            x = x.unsqueeze(0); target = target.unsqueeze(0)
        t = torch.zeros(x.size(0), 1, device=device)
        pm = batch.get("padding_mask")
        if pm is not None:
            pm = pm.to(device)
            if pm.dim() == 1:
                pm = pm.unsqueeze(0)

        losses = trainer._train_step_streamed(x, target, t, padding_mask=pm)
        avg = sum(losses.values()) / max(1, len(losses))
        loss_curve.append(avg)
        trainer.step += 1
        step = trainer.step

        # Evict unused expert blocks from GPU to prevent VRAM accumulation
        if step % 10 == 0:
            for node in trainer.router.nodes.values():
                blk = getattr(node, "_block", None)
                if blk is not None and next(blk.parameters(), torch.tensor(0)).device.type == "cuda":
                    # Keep only the blocks used this step on GPU
                    pass
            # Move blocks not used this step back to CPU
            used_nodes = set(losses.keys())
            for nid, node in trainer.router.nodes.items():
                if nid not in used_nodes:
                    blk = getattr(node, "_block", None)
                    if blk is not None and hasattr(blk, "parameters"):
                        dev = next(blk.parameters()).device
                        if dev.type == "cuda":
                            blk = blk.cpu()
                            node.__dict__["_block"] = blk

        # Router latency
        query = F.normalize(trainer.router.nodes[list(trainer.router.nodes.keys())[0]].anchor_embedding, dim=-1).unsqueeze(0)
        t0 = time.perf_counter()
        active = trainer.router.route(query)
        router_latencies.append((time.perf_counter() - t0) * 1000)

        # Cache hits: expert overlap with previous step
        curr = set(nid for nid, _, _ in active) if active else set()
        if prev_experts is not None and curr:
            overlap = len(prev_experts & curr)
            cache_hit_ratios.append(overlap / max(len(prev_experts), len(curr)))
        prev_experts = curr

        if verbose and step % 20 == 0:
            s = sys_samples[-1]
            loss_trend = f"{loss_curve[-1]:.4f} ({loss_curve[-20] if len(loss_curve) >= 20 else loss_curve[0]:.4f})" if len(loss_curve) >= 2 else f"{avg:.4f}"
            print(f"  [{label}] step {step:>4d}/{n_steps}  loss={avg:.4f}  "
                  f"VRAM={s['vram_gb']:.1f}GB  GPU={s['gpu_pct']:.0f}%  "
                  f"CPU={s['cpu_pct']:.0f}%  RAM={s['ram_gb']:.1f}GB")

    elapsed = time.time() - start
    if use_packing:
        it.stop()

    # Aggregated metrics
    avg_sys = {k: sum(d[k] for d in sys_samples) / len(sys_samples) for k in sys_samples[0]} if sys_samples else {}
    avg_router = sum(router_latencies) / len(router_latencies) if router_latencies else 0
    avg_cache = sum(cache_hit_ratios) / len(cache_hit_ratios) * 100 if cache_hit_ratios else 0
    mem_gb = torch.cuda.max_memory_allocated() / 1e9

    # Padding estimate
    pad_list = []
    if profiler and profiler.pad_fraction:
        pad_list = profiler.pad_fraction

    # Cleanup
    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    return BenchResult(
        label=label, n_experts=n_experts, canvas_len=canvas_len,
        n_domains=n_domains, seq_range=(seq_min, seq_max),
        steps=step, elapsed_s=round(elapsed, 2),
        step_s=round(elapsed / max(1, step), 5),
        tok_s=round(step * canvas_len * 2 / elapsed, 0),
        vram_gb=round(mem_gb, 2),
        gpu_pct=round(avg_sys.get("gpu_pct", 0), 1),
        cpu_pct=round(avg_sys.get("cpu_pct", 0), 1),
        ram_gb=round(avg_sys.get("ram_gb", 0), 2),
        ram_pct=round(avg_sys.get("ram_pct", 0), 1),
        padding_pct=round(sum(pad_list) / len(pad_list) * 100, 1) if pad_list else 0,
        router_latency_ms=round(avg_router, 3),
        cache_hit_pct=round(avg_cache, 1),
        packing_occupancy_pct=round(
            sum(profiler.packing_occupancy[-10:]) / max(1, len(profiler.packing_occupancy[-10:])) * 100, 1
        ) if profiler and profiler.packing_occupancy else 0,
        avg_loss=round(sum(loss_curve) / len(loss_curve), 5),
        final_loss=round(loss_curve[-1], 5) if loss_curve else 0,
        loss_curve=[round(v, 5) for v in loss_curve],
    )


def _pad_collate(batch):
    max_len = max(b["input_ids"].size(0) for b in batch)
    padded_i, padded_l = [], []
    for b in batch:
        t = b["input_ids"]
        if t.size(0) < max_len:
            t = torch.cat([t, torch.full((max_len - t.size(0),), 0, dtype=torch.long)])
        padded_i.append(t)
        padded_l.append(t.clone())
    return {"input_ids": torch.stack(padded_i), "labels": torch.stack(padded_l)}


# ═══════════════════════════════════════════════════
# CSV helpers
# ═══════════════════════════════════════════════════

FIELDS = [
    "label", "n_experts", "canvas_len", "n_domains", "seq_min", "seq_max",
    "steps", "elapsed_s", "step_s", "tok_s",
    "vram_gb", "gpu_pct", "cpu_pct", "ram_gb", "ram_pct",
    "padding_pct", "router_latency_ms", "cache_hit_pct",
    "packing_occupancy_pct", "avg_loss", "final_loss",
]


def save_results(name: str, results: list[BenchResult]):
    os.makedirs(SAVE_DIR, exist_ok=True)
    path = os.path.join(SAVE_DIR, f"{name}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in results:
            d = {k: getattr(r, k, "") for k in FIELDS}
            d["seq_min"] = r.seq_range[0]
            d["seq_max"] = r.seq_range[1]
            w.writerow(d)
    print(f"  → Saved {path}")


# ═══════════════════════════════════════════════════
# Loss-curve extended training (longer run)
# ═══════════════════════════════════════════════════

def run_extended_training(label: str, n_steps: int = 500, **kwargs):
    """Run longer training and save the full loss curve."""
    if isinstance(n_steps, str):
        n_steps = int(n_steps)
    print(f"\n{'=' * 72}")
    print(f"  EXTENDED TRAINING: {label}  ({n_steps} steps)")
    print(f"{'=' * 72}")
    r = run_bench(label, n_steps=n_steps, verbose=True, **kwargs)

    # Save loss curve separately
    curve_path = os.path.join(SAVE_DIR, f"loss_curve_{label.lower().replace(' ','_')}.csv")
    with open(curve_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "loss", "elapsed_s"])
        for i, loss in enumerate(r.loss_curve):
            frac = (i + 1) / r.steps
            w.writerow([i + 1, loss, round(r.elapsed_s * frac, 3)])

    print(f"  → Saved loss curve: {curve_path}")
    print(f"  Summary: {r.tok_s:.0f} tok/s, "
          f"avg_loss={r.avg_loss:.5f}, final_loss={r.final_loss:.5f}, "
          f"{r.elapsed_s:.1f}s elapsed")
    return r


# ═══════════════════════════════════════════════════
# Sweep runners
# ═══════════════════════════════════════════════════

def sweep_experts():
    print(f"\n{'=' * 72}")
    print("  SWEEP: Expert count")
    print(f"{'=' * 72}")
    results = []
    for n in [4, 16, 64, 128, 256]:
        print(f"\n  --- {n} experts ---")
        r = run_bench(f"E{n}", n_experts=n, canvas_len=128, n_domains=4,
                      seq_min=20, seq_max=60, n_steps=80)
        results.append(r)
        print(f"  tok/s={r.tok_s:.0f}  VRAM={r.vram_gb:.1f}GB  router={r.router_latency_ms:.2f}ms  "
              f"cache={r.cache_hit_pct:.1f}%  loss={r.final_loss:.5f}")
    save_results("expert_scale", results)
    return results


def sweep_canvas():
    print(f"\n{'=' * 72}")
    print("  SWEEP: Canvas length")
    print(f"{'=' * 72}")
    results = []
    for c in [128, 256, 512, 1024]:
        print(f"\n  --- canvas={c} ---")
        n_steps = max(30, 100 * 128 // c)  # more steps for small canvas
        r = run_bench(f"C{c}", n_experts=16, canvas_len=c, n_domains=4,
                      seq_min=20, seq_max=60, n_steps=n_steps)
        results.append(r)
        print(f"  tok/s={r.tok_s:.0f}  VRAM={r.vram_gb:.1f}GB  step={r.step_s:.4f}s  "
              f"pad={r.padding_pct:.1f}%  loss={r.final_loss:.5f}")
    save_results("canvas_scale", results)
    return results


class UniformDomainDataset(Dataset):
    """Wraps a dataset, overriding domain to a single value."""
    def __init__(self, base, domain="general"):
        self.base = base
        self.domain = domain
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        item = dict(self.base[idx])
        item["domain"] = self.domain
        return item


def sweep_domains():
    print(f"\n{'=' * 72}")
    print("  SWEEP: Domain count (Domain Queue vs Random)")
    print(f"{'=' * 72}")
    results = []
    for d in [4, 50, 200, 1000]:
        # Domain mode: real domain IDs → domain-queued packing
        label = f"D{d}_domain"
        print(f"\n  --- {d} domains, Domain Queue ---")
        r = run_bench(label, n_experts=16, canvas_len=128, n_domains=d,
                      seq_min=20, seq_max=60, n_steps=100)
        results.append(r)
        print(f"  tok/s={r.tok_s:.0f}  VRAM={r.vram_gb:.1f}GB  "
              f"cache={r.cache_hit_pct:.1f}%  router={r.router_latency_ms:.2f}ms")
        # Random mode: all items get domain="general" → single domain queue
        label_r = f"D{d}_random"
        print(f"  --- {d} domains, Random ---")
        # We need to override the domain in the dataset; simplest: modify _make_trainer
        # For random mode, we re-run but post-process domain to "general"
        # Use a closure approach: create a wrapper dataset
        base_ds = ScaleDataset(n_samples=5000, n_domains=d, seq_min=20, seq_max=60, vocab=151643)
        uniform_ds = UniformDomainDataset(base_ds)

        # Re-implement run_bench inline with uniform domain
        r_r = _run_bench_with_ds(label_r, uniform_ds, n_experts=16, canvas_len=128, n_steps=100)
        results.append(r_r)
        print(f"  tok/s={r_r.tok_s:.0f}  VRAM={r_r.vram_gb:.1f}GB  "
              f"cache={r_r.cache_hit_pct:.1f}%  router={r_r.router_latency_ms:.2f}ms")
    save_results("domain_scale", results)
    return results


def _run_bench_with_ds(label, ds, n_experts, canvas_len, n_steps):
    """Minimal run_bench that accepts a pre-built dataset."""
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(42)
    embed_dim = get_preset("tiny").d_model
    vocab = 151643
    trainer = make_trainer(n_experts, canvas_len, embed_dim, vocab, device="cuda")
    trainer.step = 0
    monitor = LiveMonitor()
    loss_curve = []
    router_latencies = []
    cache_hit_ratios = []
    prev_experts = None
    profiler = MeshProfiler(log_interval=999)
    it = AsyncPrefetchTokenBucketIterator(
        dataset=ds, canvas_len=canvas_len, eos_id=vocab - 1,
        max_canvases=2, dynamic_budget=True, min_budget=256, max_budget=8192,
        prefetch_queue_size=2, shuffle=True,
    )
    it.set_profiler(profiler)
    start = time.time()
    step = 0
    for batch in it:
        if step >= n_steps:
            break
        x = batch["input_ids"].to("cuda"); target = batch["labels"].to("cuda")
        if x.dim() == 1: x = x.unsqueeze(0); target = target.unsqueeze(0)
        t = torch.zeros(x.size(0), 1, device="cuda")
        pm = batch.get("padding_mask")
        if pm is not None: pm = pm.to("cuda")
        if pm is not None and pm.dim() == 1: pm = pm.unsqueeze(0)
        losses = trainer._train_step_streamed(x, target, t, padding_mask=pm)
        loss_curve.append(sum(losses.values()) / max(1, len(losses)))
        trainer.step += 1; step = trainer.step
        if step % 10 == 0:
            used = set(losses.keys())
            for nid, node in trainer.router.nodes.items():
                if nid not in used:
                    blk = getattr(node, "_block", None)
                    if blk is not None and hasattr(blk, "parameters") and next(blk.parameters()).device.type == "cuda":
                        node.__dict__["_block"] = blk.cpu()
        query = F.normalize(list(trainer.router.nodes.values())[0].anchor_embedding, dim=-1).unsqueeze(0)
        t0 = time.perf_counter()
        active = trainer.router.route(query)
        router_latencies.append((time.perf_counter() - t0) * 1000)
        curr = set(nid for nid, _, _ in active) if active else set()
        if prev_experts is not None and curr:
            cache_hit_ratios.append(len(prev_experts & curr) / max(len(prev_experts), len(curr)))
        prev_experts = curr
    elapsed = time.time() - start
    it.stop()
    mem_gb = torch.cuda.max_memory_allocated() / 1e9
    avg_router = sum(router_latencies) / len(router_latencies) if router_latencies else 0
    avg_cache = sum(cache_hit_ratios) / len(cache_hit_ratios) * 100 if cache_hit_ratios else 0
    gc.collect(); torch.cuda.empty_cache()
    return BenchResult(
        label=label, n_experts=n_experts, canvas_len=canvas_len,
        n_domains=len(set(ds[i]["domain"] for i in range(min(100, len(ds))))),  # approximate
        seq_range=(20, 60), steps=step,
        elapsed_s=round(elapsed, 2), step_s=round(elapsed / max(1, step), 5),
        tok_s=round(step * canvas_len * 2 / elapsed, 0),
        vram_gb=round(mem_gb, 2), gpu_pct=0, cpu_pct=0, ram_gb=0, ram_pct=0,
        padding_pct=round(sum(profiler.pad_fraction) / max(1, len(profiler.pad_fraction)) * 100, 1) if profiler.pad_fraction else 0,
        router_latency_ms=round(avg_router, 3), cache_hit_pct=round(avg_cache, 1),
        packing_occupancy_pct=0, avg_loss=round(sum(loss_curve)/len(loss_curve), 5),
        final_loss=round(loss_curve[-1], 5) if loss_curve else 0,
    )


def sweep_variance():
    print(f"\n{'=' * 72}")
    print("  SWEEP: Sequence length variance")
    print(f"{'=' * 72}")
    ranges = [(20, 60), (10, 400), (50, 800), (100, 1500)]
    results = []
    for lo, hi in ranges:
        label = f"V{lo}-{hi}"
        print(f"\n  --- seq {lo}-{hi} ---")
        r = run_bench(label, n_experts=16, canvas_len=512, n_domains=4,
                      seq_min=lo, seq_max=hi, n_steps=60)
        results.append(r)
        print(f"  tok/s={r.tok_s:.0f}  VRAM={r.vram_gb:.1f}GB  pad={r.padding_pct:.1f}%  "
              f"step={r.step_s:.4f}s  loss={r.final_loss:.5f}")
    save_results("variance_scale", results)
    return results


# ═══════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", type=str, default="all",
                        choices=["all", "experts", "canvas", "domains", "variance", "extended"])
    parser.add_argument("--steps", type=int, default=0,
                        help="Override steps for extended training")
    args = parser.parse_args()

    print("=" * 72)
    print("  MESH TRAINING — SCALABILITY BENCHMARKS")
    dev = torch.cuda.get_device_properties(0)
    print(f"  GPU: {torch.cuda.get_device_name(0)}  VRAM: {dev.total_memory/1e9:.1f}GB")
    print(f"  Model: tiny (768-dim)  Device: cuda")
    print("=" * 72)

    overall_start = time.time()

    if args.sweep in ("all", "experts"):
        sweep_experts()

    if args.sweep in ("all", "canvas"):
        sweep_canvas()

    if args.sweep in ("all", "domains"):
        sweep_domains()

    if args.sweep in ("all", "variance"):
        sweep_variance()

    if args.sweep in ("all", "extended"):
        run_extended_training(
            "Domain Queue 500 steps",
            n_steps=args.steps or 500,
            n_experts=16, canvas_len=256, n_domains=4,
            seq_min=20, seq_max=60,
        )

    elapsed = time.time() - overall_start
    print(f"\n{'=' * 72}")
    print(f"  All sweeps complete in {elapsed:.0f}s")
    print(f"  Results in: {SAVE_DIR}")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
