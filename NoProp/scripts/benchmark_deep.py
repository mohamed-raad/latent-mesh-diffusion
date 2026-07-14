"""
Deep training vs inference benchmark with per-component GPU breakdown.
Uses torch.cuda.synchronize() around each manually segmented component
for accurate GPU timing (no CUDA events required).
"""
import csv, gc, os, sys, time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

torch.set_float32_matmul_precision("high")

from mesh_router import MeshRouter, MeshNode
from noprop_block import NoPropBlock

SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "benchmarks")
VOCAB = 151643
SEED = 42


# ─── GPU Monitor via nvml ───────────────────────────────────────────────

class GPUMonitor:
    def __init__(self):
        self._handle = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._nvml = pynvml
        except ImportError:
            pass

    def sample(self) -> dict:
        d = {"gpu_pct": 0.0, "mem_pct": 0.0, "vram_gb": 0.0}
        if self._handle:
            try:
                util = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
                d["gpu_pct"] = util.gpu
                mem = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
                d["mem_pct"] = (mem.used / mem.total) * 100
                d["vram_gb"] = mem.used / 1e9
            except Exception:
                pass
        if torch.cuda.is_available():
            try:
                free, total = torch.cuda.mem_get_info()
                d["vram_gb"] = (total - free) / 1e9
            except Exception:
                pass
        return d


# ─── Model + expert setup ───────────────────────────────────────────────

def make_model(n_experts: int = 16, embed_dim: int = 1024):
    router = MeshRouter(top_k=3, d_model=embed_dim)
    for i in range(n_experts):
        a = F.normalize(torch.randn(embed_dim), dim=-1)
        node = MeshNode(node_id=f"e{i:04d}", anchor_path="",
                        anchor_embedding=a, mitosis_threshold=0.5)
        router.register_node(node)
    blocks = {}
    for nid in router.nodes:
        b = NoPropBlock(embed_dim, num_heads=8).cuda()
        b.configure_optimizer(lr=1e-4)
        blocks[nid] = b
    return router, blocks


# ─── Dataset ────────────────────────────────────────────────────────────

class LongSeqDataset(Dataset):
    def __init__(self, n: int, seq_len: int, vocab: int = VOCAB):
        g = torch.Generator().manual_seed(SEED)
        g2 = torch.Generator().manual_seed(SEED + 1)
        self.data = []
        for _ in range(n):
            ids = torch.randint(4, vocab - 1, (seq_len,), generator=g2)
            self.data.append(ids)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]


# ─── Timed helpers ──────────────────────────────────────────────────────

def timed(fn, *args, sync=True, **kwargs):
    """Execute fn(*args, **kwargs) and return (result, elapsed_s)."""
    if sync:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    if sync:
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return result, t1 - t0


# ─── Benchmark runners ──────────────────────────────────────────────────

@torch.no_grad()
def run_inference(router, blocks, embed, loader, n_steps: int, monitor: GPUMonitor):
    times = []
    gpu_utils = []
    for tokens in loader:
        if len(times) >= n_steps:
            break
        x = tokens.cuda()
        t_vec = torch.zeros(x.size(0), 1, device="cuda")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        x_emb = embed(x)
        noise = torch.randn_like(x_emb)
        x_t = (x_emb + noise * t_vec.view(-1, 1, 1).expand_as(x_emb)).detach()
        query = F.normalize(x_t.mean(dim=1, keepdim=True), dim=-1)
        active = router.route(query)
        for nid, _, _ in active[:3]:
            blocks[nid](x_t, t_vec)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        gpu_utils.append(monitor.sample())
    avg_s = sum(times) / len(times)
    avg_gpu = {k: sum(d[k] for d in gpu_utils)/len(gpu_utils) for k in gpu_utils[0]} if gpu_utils else {}
    return avg_s, avg_gpu


def run_training(router, blocks, embed, loader, n_steps: int, monitor: GPUMonitor):
    """Training benchmark with per-component GPU breakdown.
    
    Uses synchronize-based wall-clock timing for accurate GPU measurements.
    """
    times_total = []
    times_embed = []
    times_router = []
    times_forward = []
    times_backward = []
    times_opt = []
    gpu_utils = []

    for tokens in loader:
        if len(times_total) >= n_steps:
            break
        x = tokens.cuda()
        B = x.size(0)
        t_vec = torch.zeros(B, 1, device="cuda")

        # Entire step
        torch.cuda.synchronize()
        t_step = time.perf_counter()

        # Embed + noise
        x_emb = embed(x)
        noise = torch.randn_like(x_emb)
        x_t = (x_emb + noise * t_vec.view(-1, 1, 1).expand_as(x_emb)).detach()
        target = x_t.detach().clone()
        torch.cuda.synchronize()
        te = time.perf_counter() - t_step
        times_embed.append(te)

        # Router
        query = F.normalize(x_t.mean(dim=1, keepdim=True), dim=-1)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        active = router.route(query)
        torch.cuda.synchronize()
        tr = time.perf_counter() - t0
        times_router.append(tr)

        selected = [(nid, blocks[nid]) for nid, _, _ in active[:3]]

        # Per-expert: forward, backward, optimizer
        step_fwd = 0.0
        step_bwd = 0.0
        step_opt = 0.0
        for nid, block in selected:
            # Forward
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            pred = block(x_t, t_vec)
            torch.cuda.synchronize()
            step_fwd += time.perf_counter() - t0

            # Backward
            loss = block.local_loss(pred, target, t=t_vec)
            block.optimizer.zero_grad()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(block.parameters(), 1.0)
            torch.cuda.synchronize()
            step_bwd += time.perf_counter() - t0

            # Optimizer
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            block.optimizer.step()
            torch.cuda.synchronize()
            step_opt += time.perf_counter() - t0

        total = te + tr + step_fwd + step_bwd + step_opt
        times_total.append(total)
        times_forward.append(step_fwd)
        times_backward.append(step_bwd)
        times_opt.append(step_opt)
        gpu_utils.append(monitor.sample())

    avg_gpu = {k: sum(d[k] for d in gpu_utils)/len(gpu_utils) for k in gpu_utils[0]} if gpu_utils else {}
    return {
        "total_s": sum(times_total) / len(times_total),
        "embed_s": sum(times_embed) / len(times_embed),
        "router_s": sum(times_router) / len(times_router),
        "forward_s": sum(times_forward) / len(times_forward),
        "backward_s": sum(times_backward) / len(times_backward),
        "opt_s": sum(times_opt) / len(times_opt),
        "gpu_util": avg_gpu,
    }


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    dev_props = torch.cuda.get_device_properties(0)
    print("=" * 72)
    print(f"  DEEP BENCHMARK - Training vs Inference + Component Breakdown")
    print(f"  GPU: {torch.cuda.get_device_name(0)}  VRAM: {dev_props.total_memory/1e9:.1f}GB")
    print(f"  SM: {dev_props.multi_processor_count}  Capability: {dev_props.major}.{dev_props.minor}")
    print("=" * 72)

    embed_dim = 1024
    n_experts = 16
    n_steps = 30
    batch_size = 2
    monitor = GPUMonitor()
    csv_path = os.path.join(SAVE_DIR, "deep_benchmark.csv")

    fieldnames = [
        "mode", "seq_len", "tok_s", "step_s", "gpu_pct", "mem_pct", "vram_gb",
        "embed_ms", "router_ms", "forward_ms", "backward_ms", "opt_ms",
    ]
    rows = []

    for seq_len in [128, 512, 1024, 2048]:
        print(f"\n--- seq_len={seq_len} ---")
        ds = LongSeqDataset(n=2000, seq_len=seq_len)
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
        embed = nn.Embedding(VOCAB, embed_dim).cuda()

        # ── Inference ──
        router, blocks = make_model(n_experts, embed_dim)
        avg_s, util = run_inference(router, blocks, embed, loader, n_steps, monitor)
        tok_s = n_steps * batch_size * seq_len / (avg_s * n_steps)
        print(f"  INFERENCE:  {tok_s:>7.0f} tok/s  step={avg_s:.4f}s  "
              f"GPU={util.get('gpu_pct',0):.0f}%  Mem={util.get('mem_pct',0):.0f}%  "
              f"VRAM={util.get('vram_gb',0):.1f}GB")
        rows.append({"mode": "inference", "seq_len": seq_len,
                      "tok_s": round(tok_s), "step_s": round(avg_s, 5),
                      "gpu_pct": round(util.get("gpu_pct", 0), 1),
                      "mem_pct": round(util.get("mem_pct", 0), 1),
                      "vram_gb": round(util.get("vram_gb", 0), 2),
                      "embed_ms": "", "router_ms": "", "forward_ms": "", "backward_ms": "", "opt_ms": ""})
        del router, blocks
        gc.collect(); torch.cuda.empty_cache()

        # ── Training ──
        router2, blocks2 = make_model(n_experts, embed_dim)
        embed2 = nn.Embedding(VOCAB, embed_dim).cuda()
        times = run_training(router2, blocks2, embed2, loader, n_steps, monitor)
        tok_s = n_steps * batch_size * seq_len / (times["total_s"] * n_steps)
        util = times["gpu_util"]
        total_ms = times["total_s"] * 1000
        print(f"  TRAINING:   {tok_s:>7.0f} tok/s  step={times['total_s']:.4f}s  "
              f"GPU={util['gpu_pct']:.0f}%  Mem={util['mem_pct']:.0f}%  "
              f"VRAM={util['vram_gb']:.1f}GB")
        embed_ms = times["embed_s"] * 1000
        router_ms = times["router_s"] * 1000
        forward_ms = times["forward_s"] * 1000
        backward_ms = times["backward_s"] * 1000
        opt_ms = times["opt_s"] * 1000
        other_ms = total_ms - embed_ms - router_ms - forward_ms - backward_ms - opt_ms
        print(f"    embed={embed_ms:.1f}ms({embed_ms/total_ms*100:.0f}%)  "
              f"router={router_ms:.1f}ms({router_ms/total_ms*100:.0f}%)  "
              f"forward={forward_ms:.1f}ms({forward_ms/total_ms*100:.0f}%)  "
              f"backward={backward_ms:.1f}ms({backward_ms/total_ms*100:.0f}%)  "
              f"opt={opt_ms:.1f}ms({opt_ms/total_ms*100:.0f}%)"
              f"{f'  other={other_ms:.1f}ms' if other_ms > 1 else ''}")
        rows.append({"mode": "training", "seq_len": seq_len,
                      "tok_s": round(tok_s), "step_s": round(times["total_s"], 5),
                      "gpu_pct": round(util["gpu_pct"], 1),
                      "mem_pct": round(util["mem_pct"], 1),
                      "vram_gb": round(util["vram_gb"], 2),
                      "embed_ms": round(embed_ms, 2),
                      "router_ms": round(router_ms, 2),
                      "forward_ms": round(forward_ms, 2),
                      "backward_ms": round(backward_ms, 2),
                      "opt_ms": round(opt_ms, 2)})
        del router2, blocks2
        gc.collect(); torch.cuda.empty_cache()

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\n  -> {csv_path}")
    monitor.close()

    # Summary
    print(f"\n{'=' * 82}")
    print(f"{'mode':>10s} {'seq':>5s} {'tok/s':>8s} {'s/step':>8s} "
          f"{'GPU%':>5s} {'emb':>6s} {'rt':>6s} {'fwd':>6s} {'bwd':>6s} {'opt':>6s}")
    print(f"{'-' * 82}")
    for r in rows:
        e = r.get("embed_ms", "")
        rt = r.get("router_ms", "")
        fwd = r.get("forward_ms", "")
        bwd = r.get("backward_ms", "")
        o = r.get("opt_ms", "")
        print(f"{r['mode']:>10s} {r['seq_len']:>5d} {r['tok_s']:>8.0f} {r['step_s']:>7.4f}s "
              f"{r['gpu_pct']:>4.0f}% "
              f"{str(e)+'ms' if e!='' else '':>7s}"
              f"{str(rt)+'ms' if rt!='' else '':>6s}"
              f"{str(fwd)+'ms' if fwd!='' else '':>7s}"
              f"{str(bwd)+'ms' if bwd!='' else '':>7s}"
              f"{str(o)+'ms' if o!='' else '':>7s}")
    print(f"{'=' * 82}")

    # GPU utilization note
    print(f"\n  For Tensor Core % and mem bandwidth %:")
    print(f"    nvidia-smi dmon -s pucvmet -d 1")


if __name__ == "__main__":
    main()
