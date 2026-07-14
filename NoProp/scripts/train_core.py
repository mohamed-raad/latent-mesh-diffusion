"""
train_core.py — Core model training pipeline.

Trains a Diffusion Mesh model with:
  - MeshRouter + NoPropBlock experts
  - Flat-packing for variable-length curriculum data
  - Atomic checkpointing with resume
  - Validation every N steps
  - Fixed seed for reproducibility

Usage:
  python train_core.py                          # fresh training
  python train_core.py --resume checkpoint.pt   # resume from checkpoint
  python train_core.py --steps 500              # sanity run
"""
import argparse
import gc
import json
import os
import random
import sys
import time
from collections import defaultdict, Counter
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
torch.set_float32_matmul_precision("high")

from mesh_router import MeshRouter, MeshNode
from noprop_block import NoPropBlock, checkpoint_atomic, load_checkpoint

# ─── Config ─────────────────────────────────────────────────────────────

VOCAB = 151643
PAD_ID = 2
SEED = 42


@dataclass
class CoreConfig:
    d_model: int = 1024
    n_heads: int = 16
    ff_mult: int = 4
    n_experts: int = 7
    top_k: int = 2
    canvas_len: int = 2048
    batch_size: int = 2
    lr: float = 1e-4
    weight_decay: float = 1e-5
    max_steps: int = 100_000
    save_every: int = 2_000
    val_every: int = 500
    checkpoint_dir: str = "checkpoints/core_100m"
    resume_path: str = ""
    seed: int = SEED

    @property
    def param_estimate(self) -> str:
        per_expert = (6 + 2 * self.ff_mult) * self.d_model ** 2 + 8 * self.d_model
        core = self.n_experts * per_expert
        total = VOCAB * self.d_model + core
        return f"core={core/1e6:.0f}M  total={total/1e6:.0f}M"


# ─── Reproducibility ────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─── Dataset ────────────────────────────────────────────────────────────

class TextDataset(Dataset):
    """Loads tokenized text sequences from a JSONL file."""

    def __init__(self, jsonl_path: str, max_len: int = 2048):
        self.data = []
        self.lengths = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                ids = item.get("input_ids") or item.get("tokens") or item.get("text")
                if isinstance(ids, str):
                    continue
                ids = torch.tensor(ids, dtype=torch.long)
                if ids.numel() < 8:
                    continue
                if ids.numel() > max_len:
                    ids = ids[:max_len]
                self.data.append(ids)
                self.lengths.append(ids.size(0))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def avg_len(self):
        return sum(self.lengths) / len(self.lengths) if self.lengths else 0


class SyntheticDataset(Dataset):
    """Synthetic text-like sequences for debugging / sanity checks."""

    def __init__(self, n: int, min_len: int = 32, max_len: int = 512, seed: int = SEED):
        g = torch.Generator().manual_seed(seed)
        g2 = torch.Generator().manual_seed(seed + 1)
        self.data = []
        self.lengths = []
        for _ in range(n):
            l = int(torch.randint(min_len, max_len + 1, (1,), generator=g).item())
            ids = torch.randint(4, 1000, (l,), generator=g2)
            self.data.append(ids)
            self.lengths.append(l)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def avg_len(self):
        return sum(self.lengths) / len(self.lengths) if self.lengths else 0


# ─── Packing collator ───────────────────────────────────────────────────

def collate_packed(batch, canvas_len: int, pad_id: int = PAD_ID):
    """Flat-pack documents into rows of up to canvas_len tokens each.
    Returns (rows, total_real_tokens, total_padding_tokens)."""
    rows = []
    current = []
    current_len = 0
    real_tok = 0
    pad_tok = 0

    for doc in batch:
        L = doc.size(0)
        if current_len + L > canvas_len and current:
            row = torch.cat(current)
            rows.append(row)
            real_tok += row.size(0)
            current = []
            current_len = 0
        if L > canvas_len:
            rows.append(doc[:canvas_len])
            real_tok += canvas_len
        else:
            current.append(doc)
            current_len += L

    if current:
        row = torch.cat(current)
        rows.append(row)
        real_tok += row.size(0)

    return rows, real_tok, pad_tok


# ─── Model factory ──────────────────────────────────────────────────────

def make_model(cfg: CoreConfig):
    router = MeshRouter(top_k=cfg.top_k, d_model=cfg.d_model)
    for i in range(cfg.n_experts):
        a = F.normalize(torch.randn(cfg.d_model), dim=-1)
        node = MeshNode(node_id=f"e{i:04d}", anchor_path="",
                        anchor_embedding=a, mitosis_threshold=0.5)
        router.register_node(node)
    blocks = {}
    # Initialize LazyLinear with dummy forward, then create optimizer
    dummy = torch.randn(1, 1, cfg.d_model, device="cuda")
    t_dummy = torch.zeros(1, 1, device="cuda")  # [B, 1] for the time branching in forward
    for nid in router.nodes:
        b = NoPropBlock(cfg.d_model, num_heads=cfg.n_heads, ff_mult=cfg.ff_mult).cuda()
        b(dummy, t_dummy)  # initializes LazyLinear
        b.configure_optimizer(lr=cfg.lr, weight_decay=cfg.weight_decay)
        blocks[nid] = b
    embed = nn.Embedding(VOCAB, cfg.d_model).cuda()
    return router, blocks, embed


def count_params(blocks: dict, embed: nn.Embedding) -> dict:
    # LazyLinear already initialized by make_model's dummy forward
    expert_p = sum(p.numel() for b in blocks.values() for p in b.parameters())
    embed_p = sum(p.numel() for p in embed.parameters())
    return {"experts": expert_p, "embedding": embed_p, "total": expert_p + embed_p}


# ─── Noise schedule ────────────────────────────────────────────────────

def sample_noise(batch_size: int, device: torch.device, max_t: float = 1.0) -> torch.Tensor:
    """Sample noise levels uniformly from [0, max_t]."""
    return torch.rand(batch_size, 1, device=device) * max_t


# ─── Training step ──────────────────────────────────────────────────────

def train_step(router, blocks, embed, rows, act_counter, max_t=1.0):
    """Process a packed batch with sampled noise schedule.

    Each row in the batch gets a random noise level t ~ Uniform(0, max_t).
    Returns (total_loss, total_tokens, expert_counts, avg_t).
    """
    total_loss = 0.0
    total_tokens = 0
    counts = Counter()
    t_accum = 0.0

    for row in rows:
        row = row.cuda().unsqueeze(0)
        B, S = row.shape
        x_emb = embed(row)
        target = x_emb.detach().clone()

        t_val = torch.rand(1, device="cuda").item() * max_t
        noise = torch.randn_like(x_emb) * (1 - t_val)
        x_t = x_emb + noise
        t_2d = torch.zeros(B, 1, device="cuda").fill_(t_val)

        query = F.normalize(x_t.mean(dim=1, keepdim=True), dim=-1)
        active = router.route(query)
        active_list = list(active[:router.top_k])

        for i, (nid, _, _) in enumerate(active_list):
            counts[nid] += 1
            if act_counter is not None:
                act_counter[nid] += 1
            block = blocks[nid]
            pred = block(x_t, t_2d)
            loss = block.local_loss(pred, target, t=t_2d)
            block.optimizer.zero_grad()
            retain = (i < len(active_list) - 1)
            loss.backward(retain_graph=retain)
            torch.nn.utils.clip_grad_norm_(block.parameters(), 1.0)
            block.optimizer.step()
            total_loss += loss.item()

        total_tokens += S
        t_accum += t_val

    avg_t = t_accum / max(len(rows), 1)
    return total_loss, total_tokens, counts, avg_t


@torch.no_grad()
def validate(router, blocks, embed, val_loader):
    """Compute MSE loss at three noise levels: t=0.0, 0.3, 0.7."""
    total_loss = 0.0
    total_tokens = 0
    n = 0

    for batch in val_loader:
        rows, real_tok, _ = batch
        for row in rows:
            row = row.cuda().unsqueeze(0)
            B, S = row.shape
            x_emb = embed(row)
            target = x_emb.detach().clone()

            for t_val in [0.0, 0.3, 0.7]:
                noise = torch.randn_like(x_emb) * (1 - t_val)
                x_t = x_emb + noise
                t_2d = torch.zeros(B, 1, device="cuda").fill_(t_val)

                query = F.normalize(x_t.mean(dim=1, keepdim=True), dim=-1)
                active = router.route(query)
                for nid, _, _ in active[:router.top_k]:
                    pred = blocks[nid](x_t, t_2d)
                    total_loss += F.mse_loss(pred, target).item()
            total_tokens += S * 3  # times 3 noise levels
        n += 1
        if n >= 10:
            break

    return total_loss / max(total_tokens, 1)


# ─── Checkpointing ──────────────────────────────────────────────────────

def save_training_state(cfg: CoreConfig, step: int, router, blocks, embed, optimizer_states: dict):
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    path = os.path.join(cfg.checkpoint_dir, f"step_{step}.pt")

    state = {
        "step": step,
        "config": cfg.__dict__,
        "embed_state": embed.state_dict(),
        "optimizer_states": optimizer_states,
        "router_anchors": {nid: router.nodes[nid].anchor_embedding.cpu() for nid in router.nodes},
        "blocks": {},
    }
    for nid, block in blocks.items():
        state["blocks"][nid] = {
            "model": block.state_dict(),
        }

    checkpoint_atomic(cfg.checkpoint_dir, step, state["blocks"], optimizer_states, {"step": step})
    # Also save a simple .pt for easy resume
    torch.save(state, path)
    print(f"  [save] step_{step}.pt  ({sum(p.numel() for p in embed.parameters())/1e6:.0f}M embed + {sum(sum(p.numel() for p in blocks[n].parameters()) for n in blocks)/1e6:.0f}M experts)")


def load_training_state(path: str, cfg: CoreConfig):
    print(f"  [load] {path}")
    state = torch.load(path, map_location="cuda", weights_only=False)
    # Restore config from checkpoint
    for k, v in state.get("config", {}).items():
        setattr(cfg, k, v)

    router, blocks, embed = make_model(cfg)

    # Load embeddings
    if "embed_state" in state:
        embed.load_state_dict(state["embed_state"])

    # Load expert blocks
    for nid, block_data in state.get("blocks", {}).items():
        if nid in blocks:
            blocks[nid].load_state_dict(block_data["model"])
            blocks[nid].configure_optimizer(lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Load optimizer states
    optimizer_states = state.get("optimizer_states", {})
    for nid in blocks:
        if nid in optimizer_states and blocks[nid].optimizer is not None:
            blocks[nid].optimizer.load_state_dict(optimizer_states[nid])

    # Restore router anchors
    for nid, anchor in state.get("router_anchors", {}).items():
        if nid in router.nodes:
            router.nodes[nid].anchor_embedding = anchor.cuda()

    step = state.get("step", 0)
    print(f"  [load] resumed at step {step}")
    return router, blocks, embed, step


# ─── Benchmark prompts for evaluation ───────────────────────────────────

BENCHMARK_PROMPTS = [
    # Arithmetic
    ("arithmetic_1", "Hello."),
    ("arithmetic_2", "What is 5 + 7?"),
    ("arithmetic_3", "What is 25 * 4?"),
    ("arithmetic_4", "If you have 12 apples and eat 3, how many remain?"),
    # Reasoning
    ("reasoning_1", "If all cats are mammals and all mammals are animals, are all cats animals?"),
    ("reasoning_2", "A bat and a ball cost $1.10. The bat costs $1.00 more than the ball. How much does the ball cost?"),
    # Coding
    ("coding_1", "Write a Python function that prints Hello World."),
    ("coding_2", "Write a Python loop that sums numbers 1 to 10."),
    # Language
    ("language_1", "The quick brown fox jumps over the lazy dog."),
    ("language_2", "Continue this sentence: Once upon a time, in a land far away..."),
    # Knowledge
    ("knowledge_1", "Why is the sky blue?"),
    ("knowledge_2", "What is gravity?"),
    ("knowledge_3", "Explain how rain forms."),
]


@torch.no_grad()
def generate_response(router, blocks, embed, prompt_ids: torch.Tensor, max_new: int = 50, t: float = 0.0):
    """Generate a continuation by iteratively denoising."""
    x = prompt_ids.cuda().unsqueeze(0) if prompt_ids.dim() == 1 else prompt_ids.cuda()
    generated = x.clone()
    B, S = generated.shape
    t_vec = torch.tensor([[t]], device="cuda").expand(B, S, -1)

    for _ in range(max_new):
        x_emb = embed(generated)
        noise = torch.randn_like(x_emb) * (1 - t)
        x_t = x_emb + noise
        query = F.normalize(x_t.mean(dim=1, keepdim=True), dim=-1)
        active = router.route(query)

        pred_sum = None
        for nid, _, _ in active[:router.top_k]:
            pred = blocks[nid](x_t, t_vec)
            if pred_sum is None:
                pred_sum = pred
            else:
                pred_sum = pred_sum + pred

        if pred_sum is None:
            break
        pred_avg = pred_sum / router.top_k

        # Predict next token from last position
        logits = pred_avg[:, -1, :]  # [B, d_model]
        # Map to vocab via cosine sim with embedding weights
        sim = logits @ embed.weight.T  # [B, VOCAB]
        next_id = sim.argmax(dim=-1, keepdim=True)  # [B, 1]
        generated = torch.cat([generated, next_id], dim=1)
        S += 1
        t_vec = torch.tensor([[t]], device="cuda").expand(B, S, -1)
        if S > 2048:
            break

    return generated


# ─── Main training loop ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Core 100M")
    parser.add_argument("--resume", type=str, default="", help="Resume from checkpoint path")
    parser.add_argument("--steps", type=int, default=0, help="Override max steps (for sanity run)")
    parser.add_argument("--checkpoint-dir", type=str, default="", help="Override checkpoint dir")
    parser.add_argument("--data", type=str, default="", help="Path to JSONL training data")
    parser.add_argument("--val-data", type=str, default="", help="Path to JSONL validation data")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data (default if no --data)")
    parser.add_argument("--lr", type=float, default=0, help="Override learning rate")
    parser.add_argument("--canvas", type=int, default=0, help="Override canvas length")
    parser.add_argument("--batch-size", type=int, default=0, help="Override data loader batch size")
    parser.add_argument("--val-every", type=int, default=0, help="Override validation interval")
    parser.add_argument("--save-every", type=int, default=0, help="Override checkpoint interval")
    parser.add_argument("--eval", action="store_true", help="Run evaluation on benchmark prompts")
    parser.add_argument("--interactive", action="store_true", help="Interactive generation mode")
    args = parser.parse_args()

    cfg = CoreConfig()

    # Store CLI overrides to re-apply after checkpoint config restoration
    cli_overrides = {}
    if args.steps > 0:
        cli_overrides["max_steps"] = args.steps
    if args.checkpoint_dir:
        cli_overrides["checkpoint_dir"] = args.checkpoint_dir
    if args.lr > 0:
        cli_overrides["lr"] = args.lr
    if args.canvas > 0:
        cli_overrides["canvas_len"] = args.canvas
    if args.batch_size > 0:
        cli_overrides["batch_size"] = args.batch_size
    if args.val_every > 0:
        cli_overrides["val_every"] = args.val_every
    if args.save_every > 0:
        cli_overrides["save_every"] = args.save_every

    device = torch.cuda.get_device_properties(0)
    print("=" * 72)
    print(f"  TRAIN CORE - {cfg.param_estimate}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}  VRAM: {device.total_memory/1e9:.1f}GB")
    print(f"  d_model={cfg.d_model}  heads={cfg.n_heads}  experts={cfg.n_experts}  top_k={cfg.top_k}")
    print(f"  canvas={cfg.canvas_len}  lr={cfg.lr}  steps={cfg.max_steps}")
    print(f"  checkpoint_dir={cfg.checkpoint_dir}")
    print("=" * 72)

    set_seed(cfg.seed)

    # ── Load or create model ──
    start_step = 0
    if args.resume and os.path.exists(args.resume):
        router, blocks, embed, start_step = load_training_state(args.resume, cfg)
    else:
        router, blocks, embed = make_model(cfg)
        p = count_params(blocks, embed)
        print(f"\n  Parameters: {p['experts']/1e6:.0f}M experts + {p['embedding']/1e6:.0f}M embedding = {p['total']/1e6:.0f}M total")

    # Re-apply CLI overrides after checkpoint config restoration
    for k, v in cli_overrides.items():
        setattr(cfg, k, v)

    # ── Dataset ──
    use_synthetic = args.synthetic or not args.data
    if use_synthetic:
        print(f"\n  Using synthetic dataset (10000 sequences)")
        ds = SyntheticDataset(10000, min_len=64, max_len=512)
        n_val = 2000
        n_train = 8000
        train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(cfg.seed))
        print(f"  Train: {n_train}  Val: {n_val}  Avg len: {ds.avg_len():.0f}")
    else:
        print(f"\n  Loading data from {args.data}")
        ds = TextDataset(args.data, max_len=cfg.canvas_len)
        n_val = max(500, int(len(ds) * 0.1))
        n_train = len(ds) - n_val
        train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(cfg.seed))
        print(f"  Train: {n_train}  Val: {n_val}  Avg len: {ds.avg_len():.0f}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_packed(b, cfg.canvas_len),
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_packed(b, cfg.canvas_len),
        num_workers=0,
        pin_memory=False,
    )

    # ── Training loop ──
    step = start_step
    epoch = 0
    best_val_loss = float("inf")
    act_counter = Counter()
    metrics_log = defaultdict(list)

    print(f"\n  Starting training from step {step}...\n")

    # Resume by advancing data loader to correct position
    train_iter = iter(train_loader)
    for _ in range(step):
        try:
            next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)

    while step < cfg.max_steps:
        batch_times = []
        loss_accum = 0.0
        tok_accum = 0

        for batch in train_iter:
            if step >= cfg.max_steps:
                break

            rows, batch_real_tok, _ = batch
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            loss, tok, counts, avg_t_noise = train_step(router, blocks, embed, rows, act_counter)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            batch_times.append(elapsed)
            loss_accum += loss
            tok_accum += tok
            vram = torch.cuda.max_memory_allocated() / 1e9
            tok_s = tok / elapsed

            # ── Logging ──
            if step % 20 == 0:
                avg_batch_t = sum(batch_times[-20:]) / max(len(batch_times[-20:]), 1)
                print(f"  step={step:>5d}  loss={loss:.4f}  t={avg_t_noise:.2f}  "
                      f"{tok_s:>7.0f} tok/s  "
                      f"step={elapsed*1000:.1f}ms  VRAM={vram:.1f}GB"
                      f"{'  [resumed]' if step == start_step and start_step > 0 else ''}")

            # ── Validation ──
            if step > 0 and step % cfg.val_every == 0:
                val_loss = validate(router, blocks, embed, val_loader)
                status = "  [best]" if val_loss < best_val_loss else ""
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                print(f"  {'-' * 50}")
                print(f"  VALIDATION  step={step:>5d}  val_loss={val_loss:.6f}  "
                      f"best={best_val_loss:.6f}{status}")
                metrics_log["step"].append(step)
                metrics_log["val_loss"].append(val_loss)
                metrics_log["train_loss"].append(loss_accum / max(len(batch_times), 1))
                print(f"  {'-' * 50}")

                # Save metrics log
                log_path = os.path.join(cfg.checkpoint_dir, "metrics.json")
                os.makedirs(cfg.checkpoint_dir, exist_ok=True)
                with open(log_path, "w") as f:
                    json.dump(dict(metrics_log), f)

                # Reset VRAM stats
                torch.cuda.reset_peak_memory_stats()

            # ── Checkpoint ──
            if step > 0 and step % cfg.save_every == 0:
                opt_states = {nid: blocks[nid].optimizer.state_dict() for nid in blocks}
                save_training_state(cfg, step, router, blocks, embed, opt_states)
                # Also save latest checkpoint for resume
                latest = os.path.join(cfg.checkpoint_dir, "step_latest.pt")
                state = {
                    "step": step,
                    "config": cfg.__dict__,
                    "embed_state": embed.state_dict(),
                    "optimizer_states": opt_states,
                    "router_anchors": {nid: router.nodes[nid].anchor_embedding.cpu() for nid in router.nodes},
                    "blocks": {nid: {"model": blocks[nid].state_dict()} for nid in blocks},
                }
                torch.save(state, latest)

            step += 1

        # End of epoch
        epoch += 1
        train_iter = iter(train_loader)

    # ── Final save ──
    opt_states = {nid: blocks[nid].optimizer.state_dict() for nid in blocks}
    save_training_state(cfg, step, router, blocks, embed, opt_states)
    latest = os.path.join(cfg.checkpoint_dir, "step_latest.pt")
    state = {
        "step": step,
        "config": cfg.__dict__,
        "embed_state": embed.state_dict(),
        "optimizer_states": opt_states,
        "router_anchors": {nid: router.nodes[nid].anchor_embedding.cpu() for nid in router.nodes},
        "blocks": {nid: {"model": blocks[nid].state_dict()} for nid in blocks},
    }
    torch.save(state, latest)
    print(f"\n  Training complete at step {step}")
    print(f"  Final checkpoint: {latest}")


if __name__ == "__main__":
    main()
