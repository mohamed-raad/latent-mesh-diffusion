import os
import sys
import time
import json
import torch
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_mesh import MeshTrainer, MeshRouter, MeshNode
from noprop_block import NoPropBlock
from turboquant_attention import TurboQuantKVCompression


def benchmark_inference(trainer: MeshTrainer, num_runs: int = 50, embed_dim: int = 128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(1, embed_dim, device=device)

    trainer.kv_compressor = TurboQuantKVCompression(embed_dim)
    latencies: list[float] = []

    for node in trainer.router.nodes.values():
        block = getattr(node, "_block", None)
        if block is None:
            block = NoPropBlock(embed_dim, num_heads=4)
            node.__dict__["_block"] = block
            block.configure_optimizer(lr=1e-3)
        block.to(device)
        block.eval()

    for _ in range(5):
        trainer.infer(x)

    for _ in range(num_runs):
        t0 = time.perf_counter()
        output, info = trainer.infer(x)
        latencies.append(time.perf_counter() - t0)

    avg_ms = sum(latencies) / len(latencies) * 1000
    min_ms = min(latencies) * 1000
    max_ms = max(latencies) * 1000

    print(f"Inference benchmark ({num_runs} runs):")
    print(f"  Avg: {avg_ms:.2f} ms")
    print(f"  Min: {min_ms:.2f} ms")
    print(f"  Max: {max_ms:.2f} ms")
    print(f"  Output shape: {output.shape}")
    print(f"  Active nodes: {info['active_nodes']}")
    print(f"  Draft tokens shape: {info['draft_tokens'].shape}")

    return {"avg_ms": avg_ms, "min_ms": min_ms, "max_ms": max_ms, "num_nodes": len(trainer.router.nodes)}


def run_memory_profile(trainer: MeshTrainer, embed_dim: int = 128):
    if not torch.cuda.is_available():
        print("CUDA not available — skipping memory profile")
        return {"cuda_available": False}

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    x = torch.randn(1, embed_dim)
    _ = trainer.infer(x)

    current = torch.cuda.memory_allocated() / 1024**2
    peak = torch.cuda.max_memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2

    print(f"VRAM profile:")
    print(f"  Current allocated: {current:.2f} MB")
    print(f"  Peak allocated:    {peak:.2f} MB")
    print(f"  Reserved:          {reserved:.2f} MB")

    assert current < 8192, f"VRAM budget exceeded: {current:.2f} MB > 8192 MB"

    return {"current_mb": current, "peak_mb": peak, "reserved_mb": reserved}


def run_compression_report(trainer: MeshTrainer, embed_dim: int = 128):
    k = torch.randn(1, 64, embed_dim)
    v = torch.randn(1, 64, embed_dim)

    tqc = TurboQuantKVCompression(embed_dim)
    kq, vq, corr = tqc.compress(k, v)

    orig_bytes = k.numel() * 4 + v.numel() * 4
    compressed_bytes = kq.numel() * (3 / 8) + vq.numel() * (3 / 8) + corr.numel() * (1 / 8)
    ratio = orig_bytes / max(compressed_bytes, 1)

    print(f"KV-Cache compression:")
    print(f"  Original:  {orig_bytes / 1024:.1f} KB")
    print(f"  Compressed: {compressed_bytes / 1024:.1f} KB")
    print(f"  Ratio:      {ratio:.1f}x")

    return {"orig_kb": orig_bytes / 1024, "compressed_kb": compressed_bytes / 1024, "ratio": ratio}


def run_speculator_report(trainer: MeshTrainer):
    print(f"Speculator (MTP): {trainer.num_draft_tokens} draft tokens")
    print(f"  Confidence threshold: {trainer.speculator.confidence_threshold}")
    print(f"  Vocabulary size:      {trainer.speculator.predictor.heads[0].lm_head.out_features}")
    total_params = sum(
        p.numel() for head in trainer.speculator.predictor.heads for p in head.parameters()
    )
    print(f"  Total MTP params:     {total_params:,}")
    return {"num_draft_tokens": trainer.num_draft_tokens, "mtp_params": total_params}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--nodes-dir", type=str, default="nodes")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/mesh")
    parser.add_argument("--num-bench-runs", type=int, default=30)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")

    trainer = MeshTrainer(
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        top_k=args.top_k,
        nodes_dir=args.nodes_dir,
        checkpoint_dir=args.checkpoint_dir,
    )

    print(f"\n{'='*60}")
    print("MESH EVALUATION REPORT")
    print(f"{'='*60}\n")

    trainer.summary()
    print()

    bench = benchmark_inference(trainer, num_runs=args.num_bench_runs, embed_dim=args.embed_dim)
    print()

    memory = run_memory_profile(trainer, embed_dim=args.embed_dim)
    print()

    compression = run_compression_report(trainer, embed_dim=args.embed_dim)
    print()

    spec = run_speculator_report(trainer)
    print()

    report = {
        "benchmark": bench,
        "memory": memory,
        "compression": compression,
        "speculator": spec,
        "nodes": len(trainer.router.nodes),
        "embed_dim": args.embed_dim,
        "top_k": args.top_k,
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {args.output}")

    print(f"{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
