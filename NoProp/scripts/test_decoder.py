"""
test_decoder.py — Test the diffusion decoder with a trained checkpoint.

Usage:
    python test_decoder.py --checkpoint checkpoints/core_100m_sanity/step_latest.pt
    python test_decoder.py --checkpoint checkpoints/core_100m_sanity/step_latest.pt --prompt "Hello"
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
torch.set_float32_matmul_precision("high")

from mesh_router import MeshRouter, MeshNode
from noprop_block import NoPropBlock
from diffusion_decoder import DiffusionDecoder

VOCAB = 151643


def load_checkpoint(path: str):
    state = torch.load(path, map_location="cuda", weights_only=False)
    cfg = state.get("config", {})
    d_model = cfg.get("d_model", 1024)
    n_heads = cfg.get("n_heads", 16)
    ff_mult = cfg.get("ff_mult", 4)
    n_experts = cfg.get("n_experts", 7)
    top_k = cfg.get("top_k", 2)

    router = MeshRouter(top_k=top_k, d_model=d_model)
    for i in range(n_experts):
        a = F.normalize(torch.randn(d_model), dim=-1)
        node = MeshNode(node_id=f"e{i:04d}", anchor_path="",
                        anchor_embedding=a, mitosis_threshold=0.5)
        router.register_node(node)

    blocks = {}
    dummy = torch.randn(1, 1, d_model, device="cuda")
    t_dummy = torch.zeros(1, 1, device="cuda")
    for nid in router.nodes:
        b = NoPropBlock(d_model, num_heads=n_heads, ff_mult=ff_mult).cuda()
        b(dummy, t_dummy)
        blocks[nid] = b

    embed = nn.Embedding(VOCAB, d_model).cuda()

    if "embed_state" in state:
        embed.load_state_dict(state["embed_state"])
    for nid, block_data in state.get("blocks", {}).items():
        if nid in blocks:
            blocks[nid].load_state_dict(block_data["model"])
    for nid, anchor in state.get("router_anchors", {}).items():
        if nid in router.nodes:
            router.nodes[nid].anchor_embedding = anchor.cuda()

    step = state.get("step", 0)
    print(f"  Loaded step {step}: d={d_model} h={n_heads} experts={n_experts} top_k={top_k}")
    return router, blocks, embed, step, top_k


def main():
    parser = argparse.ArgumentParser(description="Test diffusion decoder")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/core_100m_sanity/step_latest.pt")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--canvas", type=int, default=64)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--entropy-bound", type=float, default=0.1)
    parser.add_argument("--no-self-conditioning", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    device = torch.cuda.get_device_properties(0)
    print("=" * 60)
    print(f"  DIFFUSION DECODER TEST")
    print(f"  GPU: {torch.cuda.get_device_name(0)}  VRAM: {device.total_memory/1e9:.1f}GB")
    print("=" * 60)

    router, blocks, embed, step, top_k = load_checkpoint(args.checkpoint)
    decoder = DiffusionDecoder(router, blocks, embed, top_k=top_k)

    vocab_size = embed.num_embeddings
    print(f"  Vocab size: {vocab_size}")

    if args.interactive:
        print("\n  Interactive generation. Type a prompt (or 'quit'):\n")
        while True:
            try:
                prompt = input("  Prompt: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not prompt or prompt.lower() in ("quit", "exit", "q"):
                break

            ascii_ids = [min(max(ord(c), 4), VOCAB - 1) for c in prompt[:args.canvas // 2]]
            prompt_tensor = torch.tensor([ascii_ids], dtype=torch.long, device="cuda")

            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            t0.record()

            tokens, trajectory = decoder.generate(
                canvas_length=args.canvas,
                prompt_ids=prompt_tensor if len(ascii_ids) > 0 else None,
                max_denoising_steps=args.steps,
                entropy_bound=args.entropy_bound,
                return_trajectory=True,
                vocab_range=(4, 1000),
            )

            t1.record()
            torch.cuda.synchronize()
            elapsed = t0.elapsed_time(t1) / 1000

            output = tokens[0].cpu().tolist()
            display_ids = [i if 3 < i < 1000 else (46 if i == 2 else 46) for i in output[:200]]
            output_chars = "".join(
                chr(d) if 31 < d < 127 else "."
                for d in display_ids
            )
            print(f"  Output ({len(output)} tokens, {elapsed:.2f}s):")
            print(f"    {output_chars}\n")

            for traj_step, traj_canvas in enumerate(trajectory):
                if traj_step % max(1, args.steps // 4) == 0 or traj_step == len(trajectory) - 1:
                    traj_ids = [int(t.item()) for t in traj_canvas[0][:min(80, traj_canvas.size(1))]]
                    traj_display = "".join(
                        chr(d) if 31 < d < 127 else "."
                        for d in [i if 3 < i < 1000 else 46 for i in traj_ids]
                    )
                    print(f"    step {traj_step:>2d}: {traj_display}")

        return

    # ── Automatic test ──
    prompt = args.prompt
    if prompt:
        ascii_ids = [min(max(ord(c), 4), VOCAB - 1) for c in prompt[:args.canvas // 2]]
        prompt_tensor = torch.tensor([ascii_ids], dtype=torch.long, device="cuda")
    else:
        prompt_tensor = None

    print(f"  Prompt: {prompt or '(none)'}")
    print(f"  Canvas: {args.canvas}  Steps: {args.steps}  Entropy bound: {args.entropy_bound}")
    print(f"  Self-conditioning: {'OFF' if args.no_self_conditioning else 'ON'}")
    print()

    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()

    tokens, trajectory = decoder.generate(
        canvas_length=args.canvas,
        prompt_ids=prompt_tensor,
        max_denoising_steps=args.steps,
        entropy_bound=args.entropy_bound,
        return_trajectory=True,
        vocab_range=(4, 1000),
    )

    t1.record()
    torch.cuda.synchronize()
    elapsed = t0.elapsed_time(t1) / 1000

    print(f"  Generated {tokens.size(1)} tokens in {elapsed:.2f}s ({args.steps} steps)")
    print(f"  Tokens per second: {tokens.size(1) / elapsed:.0f}")
    print()

    output = tokens[0].cpu().tolist()
    display_ids = [i if 3 < i < 1000 else 46 for i in output[:500]]
    output_chars = "".join(
        chr(d) if 31 < d < 127 else "."
        for d in display_ids
    )
    print(f"  Output text ({len(output)} tokens):")
    print(f"    {output_chars[:500]}")
    print()

    print("  Denoising trajectory:")
    for traj_step, traj_canvas in enumerate(trajectory):
        if max(1, args.steps // 5) == 0 or traj_step % max(1, args.steps // 5) == 0 or traj_step == len(trajectory) - 1:
            traj_ids = [int(t.item()) for t in traj_canvas[0][:min(80, traj_canvas.size(1))]]
            traj_display = "".join(
                chr(d) if 31 < d < 127 else "."
                for d in [i if 3 < i < 1000 else 46 for i in traj_ids]
            )
            print(f"    step {traj_step:>2d}: {traj_display}")

    vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"\n  Peak VRAM: {vram:.2f}GB")


if __name__ == "__main__":
    main()
