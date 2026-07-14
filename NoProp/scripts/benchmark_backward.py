"""
Per-component backward breakdown for NoPropBlock.
Hooks: attn, ff_0, ff_2, norm1, norm2, input_proj, time_emb.
Uses torch.cuda.Event for accurate GPU timing.
"""
import csv, gc, os, sys, time
from collections import defaultdict

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


# ─── GPU Monitor ────────────────────────────────────────────────────────

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


# ─── Backward timing via hooks ──────────────────────────────────────────

class BackwardTimer:
    """Attaches backward pre/post hooks to submodules for per-component GPU timing."""

    def __init__(self):
        self.times = defaultdict(list)
        self._hooks = []

    def instrument(self, block):
        mods = [
            ("attn",       block.attn),
            ("ff_0",       block.ff[0]),
            ("ff_2",       block.ff[2]),
            ("norm1",      block.norm1),
            ("norm2",      block.norm2),
            ("input_proj", block.input_proj),
            ("time_emb",   block.time_emb),
        ]
        for name, mod in mods:
            self._hook(name, mod)
        return self

    def _hook(self, name, mod):
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)

        def pre_hook(m, gO):
            start.record()

        def post_hook(m, gI, gO):
            end.record()
            torch.cuda.synchronize()
            self.times[name].append(start.elapsed_time(end))

        h1 = mod.register_full_backward_pre_hook(pre_hook)
        h2 = mod.register_full_backward_hook(post_hook)
        self._hooks.extend([h1, h2])

    def clear(self):
        for k in self.times:
            self.times[k].clear()

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def averages(self) -> dict:
        return {k: sum(v)/len(v) for k, v in self.times.items() if v}


# ─── Model setup ────────────────────────────────────────────────────────

def make_model(n_experts=16, embed_dim=1024):
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
    def __init__(self, n, seq_len, vocab=VOCAB):
        g2 = torch.Generator().manual_seed(SEED + 1)
        self.data = [torch.randint(4, vocab - 1, (seq_len,), generator=g2) for _ in range(n)]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]


# ─── Main benchmark ─────────────────────────────────────────────────────

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    dev = torch.cuda.get_device_properties(0)
    print("=" * 80)
    print(f"  BACKWARD BREAKDOWN BENCHMARK")
    print(f"  GPU: {torch.cuda.get_device_name(0)}  VRAM: {dev.total_memory/1e9:.1f}GB")
    print("=" * 80)

    embed_dim = 1024
    n_experts = 16
    n_steps   = 40
    batch_size = 2
    monitor = GPUMonitor()

    csv_path = os.path.join(SAVE_DIR, "backward_breakdown.csv")
    fields = ["seq_len", "backward_total_ms", "attn_ms", "ff_0_ms", "ff_2_ms",
              "norm1_ms", "norm2_ms", "input_proj_ms", "time_emb_ms",
              "ff_total_ms", "norm_total_ms", "other_ms",
              "gpu_pct", "vram_gb"]
    rows = []

    for seq_len in [128, 512, 1024, 2048]:
        print(f"\n--- seq_len={seq_len} ---")
        ds = LongSeqDataset(n=2000, seq_len=seq_len)
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

        router, blocks = make_model(n_experts, embed_dim)
        embed = nn.Embedding(VOCAB, embed_dim).cuda()

        # Instrument each block with backward hooks
        timers = {nid: BackwardTimer().instrument(b) for nid, b in blocks.items()}

        total_times = []
        gpu_utils = []
        step_count = 0

        for tokens in loader:
            if step_count >= n_steps:
                break
            x = tokens.cuda()
            B = x.size(0)
            t_vec = torch.zeros(B, 1, device="cuda")

            # forward + backward
            x_emb = embed(x)
            noise = torch.randn_like(x_emb)
            x_t = (x_emb + noise * t_vec.view(-1, 1, 1).expand_as(x_emb)).detach()
            target = x_t.detach().clone()

            query = F.normalize(x_t.mean(dim=1, keepdim=True), dim=-1)
            active = router.route(query)
            selected = [nid for nid, _, _ in active[:3]]

            torch.cuda.synchronize()
            t0 = time.perf_counter()

            for nid in selected:
                block = blocks[nid]
                pred = block(x_t, t_vec)
                loss = block.local_loss(pred, target, t=t_vec)
                block.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(block.parameters(), 1.0)
                block.optimizer.step()

            torch.cuda.synchronize()
            total_times.append(time.perf_counter() - t0)
            gpu_utils.append(monitor.sample())
            step_count += 1

        avg_total_s = sum(total_times) / len(total_times)
        avg_gpu = {k: sum(d[k] for d in gpu_utils)/len(gpu_utils) for k in gpu_utils[0]}

        # Aggregate backward timing across experts
        agg = defaultdict(list)
        for nid in blocks:
            t = timers[nid].averages()
            for k, v in t.items():
                if k != "embed":
                    agg[k].append(v)

        bwd_avg = {k: sum(v)/len(v) for k, v in agg.items()}

        attn_ms   = bwd_avg.get("attn", 0)
        ff_0_ms   = bwd_avg.get("ff_0", 0)
        ff_2_ms   = bwd_avg.get("ff_2", 0)
        norm1_ms  = bwd_avg.get("norm1", 0)
        norm2_ms  = bwd_avg.get("norm2", 0)
        inp_ms    = bwd_avg.get("input_proj", 0)
        time_ms   = bwd_avg.get("time_emb", 0)

        ff_total   = ff_0_ms + ff_2_ms
        norm_total = norm1_ms + norm2_ms
        accounted  = attn_ms + ff_total + norm_total + inp_ms + time_ms
        bwd_per_expert = accounted  # sum of all components

        # Per-step backward: 3 experts
        bwd_total_per_step = bwd_per_expert * 3

        # Print
        print(f"  Avg step: {avg_total_s*1000:.1f}ms  GPU: {avg_gpu['gpu_pct']:.0f}%  VRAM: {avg_gpu['vram_gb']:.1f}GB")
        print(f"  Backward per step (3 experts): {bwd_total_per_step:.1f}ms")
        print(f"  Backward per expert: {bwd_per_expert:.2f}ms")
        print(f"    attn:         {attn_ms:>6.2f}ms  {attn_ms/bwd_per_expert*100:>5.1f}%")
        print(f"    ff_0 (in):    {ff_0_ms:>6.2f}ms  {ff_0_ms/bwd_per_expert*100:>5.1f}%")
        print(f"    ff_2 (out):   {ff_2_ms:>6.2f}ms  {ff_2_ms/bwd_per_expert*100:>5.1f}%")
        print(f"    ff_total:     {ff_total:>6.2f}ms  {ff_total/bwd_per_expert*100:>5.1f}%")
        print(f"    norm1:        {norm1_ms:>6.2f}ms  {norm1_ms/bwd_per_expert*100:>5.1f}%")
        print(f"    norm2:        {norm2_ms:>6.2f}ms  {norm2_ms/bwd_per_expert*100:>5.1f}%")
        print(f"    norm_total:   {norm_total:>6.2f}ms  {norm_total/bwd_per_expert*100:>5.1f}%")
        print(f"    input_proj:   {inp_ms:>6.2f}ms  {inp_ms/bwd_per_expert*100:>5.1f}%")
        print(f"    time_emb:     {time_ms:>6.2f}ms  {time_ms/bwd_per_expert*100:>5.1f}%")
        print(f"    accounted:    {accounted:>6.2f}ms")

        rows.append({
            "seq_len": seq_len,
            "backward_total_ms": round(bwd_total_per_step, 2),
            "attn_ms":     round(attn_ms, 2),
            "ff_0_ms":     round(ff_0_ms, 2),
            "ff_2_ms":     round(ff_2_ms, 2),
            "norm1_ms":    round(norm1_ms, 2),
            "norm2_ms":    round(norm2_ms, 2),
            "input_proj_ms": round(inp_ms, 2),
            "time_emb_ms": round(time_ms, 2),
            "ff_total_ms": round(ff_total, 2),
            "norm_total_ms": round(norm_total, 2),
            "other_ms":    round(max(0, bwd_per_expert - accounted), 2),
            "gpu_pct":     round(avg_gpu["gpu_pct"], 1),
            "vram_gb":     round(avg_gpu["vram_gb"], 2),
        })

        # Cleanup
        for t in timers.values():
            t.remove()
        del router, blocks, timers
        gc.collect()
        torch.cuda.empty_cache()

    # Save CSV
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\n  -> {csv_path}")

    # Summary
    print(f"\n{'=' * 80}")
    hdr = f"{'Seq':>5s} {'Bwd/step':>10s} {'attn':>8s} {'ff_0':>8s} {'ff_2':>8s} {'ff_tot':>8s} {'norm':>8s} {'inp':>8s} {'time':>8s} {'GPU%':>5s}"
    print(hdr)
    print("-" * 80)
    for r in rows:
        print(f"{r['seq_len']:>5d} {r['backward_total_ms']:>8.1f}ms "
              f"{r['attn_ms']:>6.1f}ms {r['ff_0_ms']:>6.1f}ms {r['ff_2_ms']:>6.1f}ms "
              f"{r['ff_total_ms']:>6.1f}ms {r['norm_total_ms']:>6.1f}ms "
              f"{r['input_proj_ms']:>6.1f}ms {r['time_emb_ms']:>6.1f}ms "
              f"{r['gpu_pct']:>4.0f}%")
    print(f"{'=' * 80}")
    print(f"\n  FF dominates at short seq; attention catches up at long seq (O(S^2)).")

    # Per-expert average with percentage
    print(f"\n  Per-expert backward breakdown (%):")
    print(f"  {'Seq':>5s}  {'attn':>7s}  {'ff_in':>7s}  {'ff_out':>7s}  {'norm':>7s}  {'proj':>7s}  {'time':>7s}")
    print(f"  {'-'*52}")
    for r in rows:
        total = r["attn_ms"] + r["ff_0_ms"] + r["ff_2_ms"] + r["norm1_ms"] + r["norm2_ms"] + r["input_proj_ms"] + r["time_emb_ms"]
        print(f"  {r['seq_len']:>5d}  {r['attn_ms']/total*100:>6.1f}%  {r['ff_0_ms']/total*100:>6.1f}%  {r['ff_2_ms']/total*100:>6.1f}%  "
              f"{(r['norm1_ms']+r['norm2_ms'])/total*100:>6.1f}%  {r['input_proj_ms']/total*100:>6.1f}%  {r['time_emb_ms']/total*100:>6.1f}%")

    if hasattr(monitor, 'close'):
        monitor.close()


if __name__ == "__main__":
    main()
