"""
Benchmark matrix: measure throughput impact of each pipeline optimization.
Saves CSV for tracking over time.

Configurations:
  1. Baseline          — DataLoader, pad-to-batch
  2. + Packing          — DataLoader + SequencePacker as collate_fn
  3. + Async Everything — AsyncPrefetchTokenBucketIterator (all features)
  4. Random Packing     — Full pipeline, shuffled domains
  5. Domain Queue       — Full pipeline, natural domain grouping
"""
import csv, gc, os, sys, time, random as _random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mesh_router import MeshRouter, MeshNode, UniversalLatentSpace, ExpertAdapter
from noprop_block import NoPropBlock
from train_mesh import (
    SequencePacker,
    AsyncPrefetchTokenBucketIterator,
    MeshProfiler,
)

DOMAINS = ["python", "math", "rust", "physics"]
VOCAB = 151643
CANVAS = 128
STEPS = 60
BATCH = 2
SEED = 42
N_SAMPLES = 2000  # pre-generated random-access pool


class VarLenDataset(Dataset):
    """Non-iterable pool of variable-length sequences across 4 domains."""
    def __init__(self, n=N_SAMPLES, seed=SEED, shuffle_domains=False):
        _random.seed(seed)
        self.data = []
        g = torch.Generator().manual_seed(seed)
        g2 = torch.Generator().manual_seed(seed + 1)
        for _ in range(n):
            l = int(torch.randint(20, 61, (1,), generator=g).item())
            ids = torch.randint(4, VOCAB - 2, (l,), generator=g2)
            if shuffle_domains:
                d = _random.choice(DOMAINS)
            else:
                d = DOMAINS[hash(l) % 4]
            self.data.append({"input_ids": ids, "labels": ids, "domain": d})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def pad_collate(batch):
    max_len = max(b["input_ids"].size(0) for b in batch)
    padded_i, padded_l = [], []
    for b in batch:
        t = b["input_ids"]
        if t.size(0) < max_len:
            t = torch.cat([t, torch.full((max_len - t.size(0),), 0, dtype=torch.long)])
        padded_i.append(t)
        padded_l.append(t.clone())
    return {"input_ids": torch.stack(padded_i), "labels": torch.stack(padded_l),
            "domain": "mixed"}


def _make_model(device):
    embed_dim = 256
    token_embedding = nn.Embedding(VOCAB, embed_dim).to(device)
    lm_head = nn.Linear(embed_dim, VOCAB).to(device)
    lm_head.weight = token_embedding.weight
    router = MeshRouter(top_k=3, d_model=embed_dim).to(device)
    for i, d in enumerate(DOMAINS):
        anchor = F.normalize(torch.randn(embed_dim, device=device), dim=-1)
        node = MeshNode(node_id=f"expert_{d}", anchor_path="",
                        anchor_embedding=anchor, mitosis_threshold=0.5)
        node.metadata.domain = d
        router.register_node(node)
    return {"token_embedding": token_embedding, "lm_head": lm_head,
            "router": router, "embed_dim": embed_dim}


def run_step(model, batch, device):
    x = batch["input_ids"].to(device)
    target = batch["labels"].to(device)
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if target.dim() == 1:
        target = target.unsqueeze(0)
    t = torch.zeros(x.size(0), 1, device=device)
    padding_mask = batch.get("padding_mask")
    if padding_mask is not None:
        padding_mask = padding_mask.to(device)
        if padding_mask.dim() == 1:
            padding_mask = padding_mask.unsqueeze(0)
    labels_ids = target.detach().clone()
    if padding_mask is not None:
        labels_ids[~padding_mask] = -100
    clean = model["token_embedding"](target)
    x_emb = model["token_embedding"](x)
    noise = torch.randn_like(x_emb)
    noise_scale = t.view(-1, 1, 1).expand_as(x_emb)
    x_t = (x_emb + noise * noise_scale).detach()
    target_t = clean.detach()
    query = F.normalize(x_t.mean(dim=1, keepdim=True), dim=-1)
    active = model["router"].route(query)
    if not active:
        sel = list(model["router"].nodes.keys())[:3]
    else:
        sel = [nid for nid, _, _ in active[:3]]
    node_losses = {}
    for nid in sel:
        node = model["router"].nodes[nid]
        block = getattr(node, "_block", None)
        if block is None:
            block = NoPropBlock(model["embed_dim"], num_heads=4).to(device)
            node.__dict__["_block"] = block
        block.train()
        if block.optimizer is None:
            block.configure_optimizer(lr=1e-4)
        pred = block(x_t, t)
        loss_val = block.local_step(pred, target_t, t=t)
        node_losses[nid] = loss_val
        node.update_loss(loss_val)
    return node_losses


def benchmark_config(name: str, device="cuda",
                     use_packing=False, async_kwargs=None,
                     shuffle_domains=False) -> dict:
    """Run one config. async_kwargs enables async path."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(SEED)

    ds = VarLenDataset(n=N_SAMPLES, seed=SEED + 10 * hash(name) % (2**31),
                       shuffle_domains=shuffle_domains)
    m = _make_model(device)
    profiler = MeshProfiler(log_interval=999)

    if async_kwargs is not None:
        it = AsyncPrefetchTokenBucketIterator(
            dataset=ds,
            canvas_len=CANVAS,
            eos_id=VOCAB - 1,
            max_canvases=async_kwargs.get("max_canvases", BATCH),
            dynamic_budget=async_kwargs.get("dynamic_budget", True),
            min_budget=256,
            max_budget=8192,
            prefetch_queue_size=async_kwargs.get("prefetch_queue_size", 2),
            shuffle=True,
        )
        it.set_profiler(profiler)
        data_source = it
    elif use_packing:
        packer_fn = SequencePacker(canvas_len=CANVAS, eos_id=VOCAB - 1).__call__
        data_source = DataLoader(ds, batch_size=BATCH, collate_fn=packer_fn, shuffle=True)
    else:
        data_source = DataLoader(ds, batch_size=BATCH, collate_fn=pad_collate, shuffle=True)

    start = time.time()
    step_count = 0
    router_latencies = []
    # Track how many of the top-K experts are the same as previous step
    prev_experts: set[str] | None = None
    cache_hit_ratios = []
    pad_list = []

    for batch in data_source:
        if step_count >= STEPS:
            break

        _ = run_step(m, batch, device)

        # Router latency
        x = batch["input_ids"].to(device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        query = F.normalize(m["token_embedding"](x).mean(dim=1, keepdim=True), dim=-1)
        t0 = time.perf_counter()
        active = m["router"].route(query)
        router_latencies.append((time.perf_counter() - t0) * 1000)

        # Cache hit: how many experts overlap with previous step
        curr = set(nid for nid, _, _ in active)
        if prev_experts is not None and curr:
            overlap = len(prev_experts & curr)
            total = max(len(prev_experts), len(curr))
            cache_hit_ratios.append(overlap / total if total > 0 else 0)
        prev_experts = curr

        pf = batch.get("pad_fraction", None)
        if pf is None:
            pm = batch.get("padding_mask")
            if pm is not None:
                pm = pm.float()
                pf = float(1.0 - pm.mean().item())
            else:
                pf = 0.0
        pad_list.append(float(pf))

        step_count += 1

    elapsed = time.time() - start
    if async_kwargs is not None:
        it.stop()

    mem_gb = torch.cuda.max_memory_allocated() / 1e9
    tok_s = STEPS * CANVAS * BATCH / elapsed

    gpu_util = 0
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_util = util.gpu
    except Exception:
        pass

    avg_pad = (sum(pad_list) / len(pad_list) * 100) if pad_list else 0
    avg_router = (sum(router_latencies) / len(router_latencies)) if router_latencies else 0
    avg_cache = (sum(cache_hit_ratios) / len(cache_hit_ratios) * 100) if cache_hit_ratios else 0

    return {
        "config": name,
        "steps": STEPS,
        "time_s": round(elapsed, 2),
        "step_s": round(elapsed / STEPS, 5),
        "tok_s": round(tok_s, 0),
        "vram_gb": round(mem_gb, 2),
        "gpu_util_pct": round(gpu_util, 1),
        "padding_pct": round(avg_pad, 1),
        "router_latency_ms": round(avg_router, 3),
        "cache_hit_pct": round(avg_cache, 1),
        "packing_occupancy_pct": round(
            sum(profiler.packing_occupancy[-10:]) / max(1, len(profiler.packing_occupancy[-10:])) * 100, 1
        ) if profiler.packing_occupancy else 0,
    }


def main():
    print("=" * 72)
    print("MESH TRAINING PIPELINE — BENCHMARK MATRIX")
    print(f"Canvas: {CANVAS}  Steps: {STEPS}  Batch: {BATCH}  Vocab: {VOCAB}")
    print(f"Samples: {N_SAMPLES}  Domains: {DOMAINS}")
    dev = torch.cuda.get_device_properties(0)
    print(f"Device: {torch.cuda.get_device_name(0)}  VRAM: {dev.total_memory / 1e9:.1f}GB")
    print("=" * 72)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    configs = [
        ("Baseline",           dict(use_packing=False, async_kwargs=None,                      shuffle_domains=False)),
        ("+ Packing",          dict(use_packing=True,  async_kwargs=None,                      shuffle_domains=False)),
        ("+ Async Everything", dict(use_packing=False, async_kwargs={"max_canvases": BATCH,
                                       "dynamic_budget": True, "prefetch_queue_size": 2,},   shuffle_domains=False)),
        ("Random Packing",     dict(use_packing=False, async_kwargs={"max_canvases": BATCH,
                                       "dynamic_budget": True, "prefetch_queue_size": 2,},   shuffle_domains=True)),
        ("Domain Queue",       dict(use_packing=False, async_kwargs={"max_canvases": BATCH,
                                       "dynamic_budget": True, "prefetch_queue_size": 2,},   shuffle_domains=False)),
    ]

    results = []
    for name, kwargs in configs:
        try:
            r = benchmark_config(name, device=device, **kwargs)
            results.append(r)
            print(f"  ✓ {r['config']:25s}  {r['tok_s']:>7.0f} tok/s  "
                  f"{r['vram_gb']:>5.1f}GB  {r['step_s']:.4f}s/step  "
                  f"pad={r['padding_pct']:>4.1f}%  "
                  f"cache={r['cache_hit_pct']:>4.1f}%  "
                  f"router={r['router_latency_ms']:>5.2f}ms")
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  ✗ {name}: {e}")

    csv_dir = os.path.join(os.path.dirname(__file__), "..", "benchmarks")
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, "benchmark_results.csv")
    fieldnames = ["config", "steps", "time_s", "step_s", "tok_s", "vram_gb",
                  "gpu_util_pct", "padding_pct", "router_latency_ms",
                  "cache_hit_pct", "packing_occupancy_pct"]
    existing = []
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            existing = list(csv.DictReader(f))
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(existing)
        for r in results:
            w.writerow(r)

    print(f"\n{'=' * 72}")
    print(f"  Results saved to {csv_path}")
    print()
    print(f"{'Config':25s} {'tok/s':>8s} {'VRAM':>6s} {'s/step':>8s} "
          f"{'Pad%':>5s} {'Cache%':>7s} {'Router':>7s} {'Util%':>6s}")
    print(f"{'-' * 72}")
    for r in results:
        print(f"{r['config']:25s} {r['tok_s']:>8.0f} {r['vram_gb']:>5.1f}GB"
              f" {r['step_s']:>7.4f}s {r['padding_pct']:>4.1f}%"
              f" {r['cache_hit_pct']:>6.1f}% {r['router_latency_ms']:>6.2f}ms"
              f" {r['gpu_util_pct']:>5.1f}%")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
