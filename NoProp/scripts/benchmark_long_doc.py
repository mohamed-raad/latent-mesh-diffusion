"""
Long-document benchmark: measure actual canvas utilization and step time
for fixed-length sequences of 512, 1024, 2048 tokens.
Also tracks per-component breakdown (embed, routing, block_forward, local_step).
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
from train_mesh import (
    SequencePacker,
    AsyncPrefetchTokenBucketIterator,
    MeshProfiler,
    MeshTrainer,
)
from model_sizes import get_preset

SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "benchmarks")
VOCAB = 151643
BATCH = 2
SEED = 42


class LongDocDataset(Dataset):
    """Fixed-length sequences at a given token count."""
    def __init__(self, n: int, seq_len: int, seed: int = SEED):
        self.seq_len = seq_len
        g = torch.Generator().manual_seed(seed)
        g2 = torch.Generator().manual_seed(seed + 1)
        self.data = []
        for i in range(n):
            ids = torch.randint(4, VOCAB - 1, (seq_len,), generator=g2)
            self.data.append({"input_ids": ids, "labels": ids})
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]


def pad_collate(batch):
    """All same length, just stack."""
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


def make_trainer(canvas_len: int, n_experts: int = 16):
    embed_dim = get_preset("tiny").d_model
    t = MeshTrainer(
        model_size="tiny",
        vocab_size=VOCAB,
        canvas_len=canvas_len,
        top_k=3,
        lr=1e-4,
        external_nodes=False,
        checkpoint_dir=os.path.join(SAVE_DIR, "_tmp_ckpt"),
    )
    dev = "cuda"
    t.token_embedding = t.token_embedding.to(dev)
    t.lm_head = t.lm_head.to(dev)
    t.latent_space = t.latent_space.to(dev)
    t.speculator = t.speculator.to(dev)
    t.kv_compressor = t.kv_compressor.to(dev)
    t.global_cognitive_layer = t.global_cognitive_layer.to(dev)
    t.router = MeshRouter(top_k=3, d_model=embed_dim)
    for i in range(n_experts):
        a = F.normalize(torch.randn(embed_dim), dim=-1)
        node = MeshNode(
            node_id=f"expert_{i:04d}", anchor_path="",
            anchor_embedding=a, mitosis_threshold=0.5,
        )
        t.router.register_node(node)
    t._profiler = None
    return t


def run_bench(seq_len: int, n_steps: int = 50, n_experts: int = 16, use_packing: bool = False):
    """Benchmark training at a given sequence length. Measures actual tokens processed."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(SEED)

    trainer = make_trainer(seq_len, n_experts=n_experts)
    ds = LongDocDataset(n=2000, seq_len=seq_len)
    profiler = MeshProfiler(log_interval=999)

    if use_packing:
        it = AsyncPrefetchTokenBucketIterator(
            dataset=ds, canvas_len=seq_len, eos_id=VOCAB - 1,
            max_canvases=BATCH, dynamic_budget=True, min_budget=256,
            max_budget=seq_len * BATCH * 2,
            prefetch_queue_size=2, shuffle=True,
        )
        it.set_profiler(profiler)
        data_source = it
    else:
        data_source = torch.utils.data.DataLoader(
            ds, batch_size=BATCH, shuffle=True, collate_fn=pad_collate
        )

    step = 0
    total_actual_tokens = 0
    start = time.time()
    component_times = {"routing": [], "embed": [], "block_forward": [], "block_local_step": []}

    for batch in data_source:
        if step >= n_steps:
            break
        x = batch["input_ids"].cuda()
        target = batch["labels"].cuda()
        if x.dim() == 1:
            x = x.unsqueeze(0); target = target.unsqueeze(0)

        # Actual non-padding tokens
        pm = batch.get("padding_mask")
        if pm is not None:
            actual = pm.float().sum().item()
        else:
            actual = x.numel()
        total_actual_tokens += actual

        # Match _train_step_streamed: embed, add noise, then route + compute
        t_vec = torch.zeros(x.size(0), 1, device="cuda")
        x_emb = trainer.token_embedding(x)
        noise = torch.randn_like(x_emb)
        noise_scale = t_vec.view(-1, 1, 1).expand_as(x_emb)
        x_t = (x_emb + noise * noise_scale).detach()
        clean = trainer.token_embedding(target)
        target_t = clean.detach()

        # Per-component timing
        t0 = time.perf_counter()
        active = trainer._active_nodes(x_t)
        component_times["routing"].append((time.perf_counter() - t0) * 1000)

        selected = [nid for nid, _, _ in active[:3]] if active else list(trainer.router.nodes.keys())[:3]
        for nid in selected:
            node = trainer.router.nodes[nid]
            block = getattr(node, "_block", None)
            if block is None:
                block = NoPropBlock(trainer.embed_dim, num_heads=4).cuda()
                node.__dict__["_block"] = block
            block = block.cuda()
            block.train()
            if block.optimizer is None:
                block.configure_optimizer(lr=1e-4)

            t0 = time.perf_counter()
            pred = block(x_t, t_vec)
            component_times["block_forward"].append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            # local_step does: zero_grad → loss → backward → clip → step
            _ = block.local_step(pred, target_t, t=t_vec)
            component_times["block_local_step"].append((time.perf_counter() - t0) * 1000)

        step += 1
        if step % 10 == 0:
            # Evict unused blocks to save VRAM
            used = set(selected)
            for nid, node in trainer.router.nodes.items():
                if nid not in used:
                    blk = getattr(node, "_block", None)
                    if blk is not None and next(blk.parameters()).device.type == "cuda":
                        node.__dict__["_block"] = blk.cpu()

    elapsed = time.time() - start
    if use_packing:
        it.stop()

    tot = step * BATCH * seq_len
    vr = torch.cuda.max_memory_allocated() / 1e9
    avg_c = {k: (sum(v) / len(v)) if v else 0 for k, v in component_times.items()}
    avg_c_sum = sum(avg_c.values())

    return {
        "seq_len": seq_len,
        "steps": step,
        "elapsed_s": round(elapsed, 2),
        "step_s": round(elapsed / step, 5),
        "tok_s": round(step * BATCH * seq_len / elapsed, 0),
        "actual_tok_s": round(total_actual_tokens / elapsed, 0),
        "actual_per_canvas": round(total_actual_tokens / step / BATCH, 1),
        "vram_gb": round(vr, 2),
        "routing_ms": round(avg_c["routing"], 3),
        "block_forward_ms": round(avg_c["block_forward"], 3),
        "block_local_step_ms": round(avg_c["block_local_step"], 3),
        "avg_loss": 0,
        "routing_pct": round(avg_c["routing"] / avg_c_sum * 100, 1) if avg_c_sum > 0 else 0,
        "forward_pct": round(avg_c["block_forward"] / avg_c_sum * 100, 1) if avg_c_sum > 0 else 0,
        "local_step_pct": round(avg_c["block_local_step"] / avg_c_sum * 100, 1) if avg_c_sum > 0 else 0,
    }


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    print("=" * 72)
    print("  LONG-DOCUMENT BENCHMARK")
    dev = torch.cuda.get_device_properties(0)
    print(f"  GPU: {torch.cuda.get_device_name(0)}  VRAM: {dev.total_memory/1e9:.1f}GB")
    print("=" * 72)

    seq_lens = [128, 512, 1024, 2048]
    csv_path = os.path.join(SAVE_DIR, "long_doc_benchmark.csv")
    fieldnames = [
        "seq_len", "steps", "elapsed_s", "step_s", "tok_s",
        "actual_tok_s", "actual_per_canvas", "vram_gb",
        "routing_ms", "block_forward_ms", "block_local_step_ms",
        "routing_pct", "forward_pct", "local_step_pct",
    ]

    all_rows = []
    for sl in seq_lens:
        n_steps = max(30, 100 * 128 // sl)
        try:
            r = run_bench(sl, n_steps=n_steps, n_experts=16)
            all_rows.append(r)
            comp = f"r={r['routing_ms']:.1f}ms({r['routing_pct']:.0f}%) f={r['block_forward_ms']:.1f}ms({r['forward_pct']:.0f}%) l={r['block_local_step_ms']:.1f}ms({r['local_step_pct']:.0f}%)"
            print(f"  seq={sl:>4d}  {r['tok_s']:>7.0f} tok/s  actual={r['actual_tok_s']:>7.0f} tok/s  "
                  f"fill={r['actual_per_canvas']:>5.1f}/{sl}  "
                  f"step={r['step_s']:.4f}s  VRAM={r['vram_gb']:.1f}GB")
            print(f"          component: {comp}")
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  seq={sl}: FAILED — {e}")

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    print(f"\n  → Saved {csv_path}")

    # Summary table
    print(f"\n{'=' * 72}")
    print(f"{'seq_len':>8s} {'tok/s':>8s} {'actual/s':>9s} {'fill':>7s} {'s/step':>8s} {'routing':>8s} {'forward':>8s} {'local':>8s} {'VRAM':>6s}")
    print(f"{'-' * 72}")
    for r in all_rows:
        print(f"{r['seq_len']:>8d} {r['tok_s']:>8.0f} {r['actual_tok_s']:>9.0f} "
              f"{r['actual_per_canvas']:>5.1f}/{r['seq_len']:<3d} "
              f"{r['step_s']:>7.4f}s {r['routing_ms']:>7.2f}ms {r['block_forward_ms']:>7.2f}ms "
              f"{r['block_local_step_ms']:>7.2f}ms {r['vram_gb']:>5.1f}GB")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
