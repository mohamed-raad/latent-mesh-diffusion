"""
eval_core.py — Evaluate denoising autoencoder checkpoints.

The NoPropBlock is trained as a denoising autoencoder:
  x_emb → add noise → x_t → NoPropBlock → pred ≈ x_emb

Metrics:
  - Validation MSE loss (what the model is trained on)
  - Cosine similarity between pred and target embeddings
  - Reconstruction accuracy (top-1 token match rate)
  - Expert activation distribution

Usage:
  python eval_core.py --checkpoint path/to/step_latest.pt
  python eval_core.py --checkpoint path/to/step_latest.pt --interactive
"""
import argparse
import json
import math
import os
import sys
import time
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
torch.set_float32_matmul_precision("high")

from mesh_router import MeshRouter, MeshNode
from noprop_block import NoPropBlock

VOCAB = 151643


def load_checkpoint(path: str):
    """Load a saved checkpoint and return model components."""
    print(f"  Loading {path}")
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
    for nid in router.nodes:
        b = NoPropBlock(d_model, num_heads=n_heads, ff_mult=ff_mult).cuda()
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
    print(f"  Loaded step {step}: {d_model}d, {n_heads}h, {n_experts} experts")
    return router, blocks, embed, step, d_model, top_k


class SyntheticEvalDataset(Dataset):
    """Small synthetic set for reconstruction evaluation."""
    def __init__(self, n=500, min_len=32, max_len=256, seed=42):
        g = torch.Generator().manual_seed(seed)
        g2 = torch.Generator().manual_seed(seed + 1)
        self.data = []
        for _ in range(n):
            l = int(torch.randint(min_len, max_len + 1, (1,), generator=g).item())
            ids = torch.randint(4, 1000, (l,), generator=g2)
            self.data.append(ids)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]


@torch.no_grad()
def evaluate(router, blocks, embed, loader, d_model, top_k):
    """Compute reconstruction metrics: MSE loss, cosine sim, token accuracy."""
    total_mse = 0.0
    total_cosine = 0.0
    total_tokens = 0
    correct_top1 = 0
    act_counter = Counter()
    steps = 0

    for batch in loader:
        rows, _, _ = batch
        for row in rows:
            x = row.cuda().unsqueeze(0)
            B, S = x.shape
            x_emb = embed(x)

            # Test at t=0 (no noise) and t=0.5 (medium noise)
            for t_val in [0.0, 0.3, 0.7]:
                noise = torch.randn_like(x_emb) * t_val
                x_t = x_emb + noise
                t_2d = torch.zeros(B, 1, device="cuda").fill_(t_val)

                query = F.normalize(x_t.mean(dim=1, keepdim=True), dim=-1)
                active = router.route(query)
                for nid, _, _ in active[:top_k]:
                    act_counter[nid] += 1

                # Ensemble prediction over active experts
                pred_sum = None
                for nid, _, _ in active[:top_k]:
                    p = blocks[nid](x_t, t_2d)
                    pred_sum = p if pred_sum is None else pred_sum + p
                pred = pred_sum / top_k

                mse = F.mse_loss(pred, x_emb).item()
                cos_sim = F.cosine_similarity(pred.view(-1, d_model), x_emb.view(-1, d_model)).mean().item()

                # Token accuracy: does the closest embedding match the original token?
                pred_flat = pred.view(-1, d_model)
                target_ids = x.view(-1)
                sim_all = pred_flat @ embed.weight.T
                pred_ids = sim_all.argmax(dim=-1)
                correct = (pred_ids == target_ids).sum().item()

                total_mse += mse
                total_cosine += cos_sim
                total_tokens += B * S
                correct_top1 += correct
                steps += 1

        if steps >= 100:
            break

    return {
        "mse": total_mse / steps,
        "cosine_sim": total_cosine / steps,
        "token_accuracy": correct_top1 / max(total_tokens, 1),
        "expert_activations": dict(act_counter.most_common()),
    }


@torch.no_grad()
def interactive_demo(router, blocks, embed, d_model, top_k):
    """Show reconstruction quality for typed input."""
    print("\n  Interactive reconstruction demo.")
    print("  Type a phrase and see how well the model denoises it.")
    print("  (type 'quit' to exit)\n")

    while True:
        try:
            text = input("  Input: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in ("quit", "exit", "q"):
            break

        ids = [min(max(ord(c), 4), VOCAB - 1) for c in text[:200]]
        x = torch.tensor(ids, dtype=torch.long, device="cuda").unsqueeze(0)
        x_emb = embed(x)

        for t_val in [0.0, 0.3, 0.7, 1.0]:
            noise = torch.randn_like(x_emb) * t_val
            x_t = x_emb + noise
            t_2d = torch.zeros(1, 1, device="cuda").fill_(t_val)

            query = F.normalize(x_t.mean(dim=1, keepdim=True), dim=-1)
            active = router.route(query)

            pred_sum = None
            for nid, _, _ in active[:top_k]:
                p = blocks[nid](x_t, t_2d)
                pred_sum = p if pred_sum is None else pred_sum + p
            pred = pred_sum / top_k

            # Decode predicted tokens
            sim = pred[0] @ embed.weight.T
            pred_ids = sim.argmax(dim=-1)

            mse = F.mse_loss(pred, x_emb).item()
            cos = F.cosine_similarity(pred.view(-1, d_model), x_emb.view(-1, d_model)).mean().item()

            # Simple char decode
            input_chars = "".join(chr(min(i, 255)) if 4 <= i <= 255 else "?" for i in ids)
            pred_chars = "".join(chr(min(i.item(), 255)) if 4 <= i.item() <= 255 else "?" for i in pred_ids.cpu())

            print(f"  t={t_val:.1f}  MSE={mse:.4f}  cos={cos:.4f}")
            print(f"    Input: {input_chars[:60]}")
            print(f"    Recon: {pred_chars[:60]}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Core 100M denoising model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--save-dir", type=str, default="eval_reports")
    args = parser.parse_args()

    device = torch.cuda.get_device_properties(0)
    print("=" * 60)
    print(f"  EVAL CORE — Denoising Autoencoder Evaluation")
    print(f"  GPU: {torch.cuda.get_device_name(0)}  VRAM: {device.total_memory/1e9:.1f}GB")
    print("=" * 60)

    router, blocks, embed, step, d_model, top_k = load_checkpoint(args.checkpoint)

    if args.interactive:
        interactive_demo(router, blocks, embed, d_model, top_k)
        return

    # Reconstruction evaluation
    ds = SyntheticEvalDataset(n=500)
    from train_core import collate_packed
    loader = DataLoader(ds, batch_size=2, collate_fn=lambda b: collate_packed(b, 2048))
    results = evaluate(router, blocks, embed, loader, d_model, top_k)

    print(f"\n  Step {step} — Reconstruction Metrics:")
    print(f"    MSE:           {results['mse']:.6f}")
    print(f"    Cosine Sim:    {results['cosine_sim']:.4f}")
    print(f"    Token Acc@1:   {results['token_accuracy']:.4f} ({results['token_accuracy']*100:.1f}%)")
    print(f"    Expert activations: {len(results['expert_activations'])} experts used")

    # Show which experts are most/least used
    acts = results["expert_activations"]
    if acts:
        top_e = sorted(acts.items(), key=lambda x: -x[1])[:3]
        bot_e = sorted(acts.items(), key=lambda x: x[1])[:3]
        print(f"    Top 3: {top_e}")
        print(f"    Bottom 3: {bot_e}")

    # Save report
    os.makedirs(args.save_dir, exist_ok=True)
    report_path = os.path.join(args.save_dir, f"eval_step_{step}.json")
    with open(report_path, "w") as f:
        json.dump({"step": step, "metrics": results}, f, indent=2)
    print(f"\n  Report: {report_path}")


if __name__ == "__main__":
    main()
