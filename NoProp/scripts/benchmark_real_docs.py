"""
2048 real-document benchmark: Packed vs Unpacked.

Dataset: variable-length documents (512-4096 tokens, mean ~2048).
Packed: flat-packing short docs into max_tokens batches.
Unpacked: pad/truncate all docs to fixed 2048.

Metrics: tok/s, step time, GPU util, padding ratio, router latency,
expert activation distribution, validation loss over time.
"""
import csv, gc, math, os, sys, time
from collections import defaultdict, Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
torch.set_float32_matmul_precision("high")

from mesh_router import MeshRouter, MeshNode
from noprop_block import NoPropBlock

SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "benchmarks")
VOCAB = 151643
SEED = 42
MAX_SEQ = 2048


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


# ─── Variable-length dataset ────────────────────────────────────────────

def _random_length(mean=2048, min_len=512, max_len=4096, generator=None):
    """Log-normal length distribution centered on mean, clipped to [min, max]."""
    g = generator or torch.default_generator
    raw = torch.randn(1, generator=g).item()
    l = int(mean * math.exp(raw * 0.5))
    return int(max(min_len, min(max_len, l)))


class VarLenDataset(Dataset):
    """Synthetic variable-length documents."""

    def __init__(self, n_docs, vocab=VOCAB, mean_len=2048, min_len=512, max_len=4096, seed=SEED):
        g = torch.Generator().manual_seed(seed)
        g2 = torch.Generator().manual_seed(seed + 1)
        self.data = []
        self.lengths = []
        for _ in range(n_docs):
            L = _random_length(mean_len, min_len, max_len, generator=g)
            ids = torch.randint(4, vocab - 1, (L,), generator=g2)
            self.data.append(ids)
            self.lengths.append(L)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def avg_len(self):
        return sum(self.lengths) / len(self.lengths)

    def padding_ratio(self, fixed_len=MAX_SEQ):
        """Ratio of padding tokens if all docs were padded/truncated to fixed_len."""
        total_pad = sum(max(0, fixed_len - L) for L in self.lengths)
        total_real = sum(min(L, fixed_len) for L in self.lengths)
        return total_pad / (total_real + total_pad) if (total_real + total_pad) > 0 else 0


# ─── Flat-packing collate ──────────────────────────────────────────────

PAD_TOKEN_ID = 2


def collate_packed(batch, max_tokens=MAX_SEQ):
    """Concatenate documents into rows, packed by max_tokens budget.
    Returns (rows, total_tokens, padding_tokens)."""
    rows = []
    current = []
    current_len = 0
    total_tok = 0
    padding_tok = 0
    for doc in batch:
        L = doc.size(0) + 1
        if current_len + L > max_tokens and current:
            row = torch.cat(current)
            rows.append(row)
            total_tok += row.size(0)
            current = []
            current_len = 0
        if L > max_tokens:
            row = doc[:max_tokens - 1]
            rows.append(row)
            total_tok += row.size(0)
        else:
            current.append(torch.cat([doc, torch.tensor([PAD_TOKEN_ID])]))
            current_len += L
    if current:
        row = torch.cat(current)
        rows.append(row)
        total_tok += row.size(0)
    return rows, total_tok, padding_tok


def collate_unpacked(batch, fixed_len=MAX_SEQ, pad_id=PAD_TOKEN_ID):
    """Pad or truncate each doc to fixed_len.
    Returns (batch_tensor, total_tokens, padding_tokens)."""
    out = []
    total_tok = 0
    padding_tok = 0
    for doc in batch:
        if doc.size(0) >= fixed_len:
            out.append(doc[:fixed_len])
            total_tok += fixed_len
        else:
            p = fixed_len - doc.size(0)
            out.append(torch.cat([doc, torch.full((p,), pad_id, dtype=doc.dtype)]))
            total_tok += doc.size(0)
            padding_tok += p
    return torch.stack(out), total_tok, padding_tok


# ─── Model ──────────────────────────────────────────────────────────────

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


# ─── Training step ──────────────────────────────────────────────────────

def process_row(router, blocks, embed, row_tensor, act_counter):
    """Process a single row (packed or single doc)."""
    x = row_tensor.cuda().unsqueeze(0)
    S = row_tensor.size(0)
    t = torch.zeros(1, 1, device="cuda")
    x_emb = embed(x)
    noise = torch.randn_like(x_emb)
    x_t = (x_emb + noise * t.view(-1,1,1).expand_as(x_emb)).detach()
    target = x_t.detach().clone()
    query = F.normalize(x_t.mean(dim=1, keepdim=True), dim=-1)
    active = router.route(query)
    loss_val = 0.0
    for nid, _, _ in active[:3]:
        if act_counter is not None:
            act_counter[nid] += 1
        block = blocks[nid]
        pred = block(x_t, t)
        loss = block.local_loss(pred, target, t=t)
        block.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(block.parameters(), 1.0)
        block.optimizer.step()
        loss_val += loss.item()
    return loss_val, S


def process_batch(router, blocks, embed, batch_tensor, act_counter):
    """Process a full batch of fixed-length documents."""
    x = batch_tensor.cuda()
    B, S = x.shape
    t = torch.zeros(B, 1, device="cuda")
    x_emb = embed(x)
    noise = torch.randn_like(x_emb)
    x_t = (x_emb + noise * t.view(-1,1,1).expand_as(x_emb)).detach()
    target = x_t.detach().clone()
    query = F.normalize(x_t.mean(dim=1, keepdim=True), dim=-1)
    active = router.route(query)
    loss_val = 0.0
    for nid, _, _ in active[:3]:
        if act_counter is not None:
            act_counter[nid] += 1
        block = blocks[nid]
        pred = block(x_t, t)
        loss = block.local_loss(pred, target, t=t)
        block.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(block.parameters(), 1.0)
        block.optimizer.step()
        loss_val += loss.item()
    return loss_val, B * S


def train_step(router, blocks, embed, x_batch, act_counter):
    """Route to the appropriate processing function."""
    data, total_tokens, _ = x_batch
    if isinstance(data, list):
        # Packed: process each row individually
        loss = 0.0
        for row in data:
            l, s = process_row(router, blocks, embed, row, act_counter)
            loss += l
        return loss, total_tokens
    else:
        # Unpacked: process full batch
        return process_batch(router, blocks, embed, data, act_counter)


# ─── Evaluation helpers ─────────────────────────────────────────────────

@torch.no_grad()
def eval_row(router, blocks, embed, row_tensor):
    """Forward-only loss on a single row."""
    x = row_tensor.cuda().unsqueeze(0)
    S = row_tensor.size(0)
    t = torch.zeros(1, 1, device="cuda")
    x_emb = embed(x)
    noise = torch.randn_like(x_emb)
    x_t = (x_emb + noise * t.view(-1,1,1).expand_as(x_emb)).detach()
    target = x_t.detach().clone()
    query = F.normalize(x_t.mean(dim=1, keepdim=True), dim=-1)
    active = router.route(query)
    loss_val = 0.0
    for nid, _, _ in active[:3]:
        pred = blocks[nid](x_t, t)
        loss_val += F.mse_loss(pred, target).item()
    return loss_val, S


@torch.no_grad()
def eval_batch(router, blocks, embed, batch_tensor):
    """Forward-only loss on a full batch."""
    x = batch_tensor.cuda()
    B, S = x.shape
    t = torch.zeros(B, 1, device="cuda")
    x_emb = embed(x)
    noise = torch.randn_like(x_emb)
    x_t = (x_emb + noise * t.view(-1,1,1).expand_as(x_emb)).detach()
    target = x_t.detach().clone()
    query = F.normalize(x_t.mean(dim=1, keepdim=True), dim=-1)
    active = router.route(query)
    loss_val = 0.0
    for nid, _, _ in active[:3]:
        pred = blocks[nid](x_t, t)
        loss_val += F.mse_loss(pred, target).item()
    return loss_val, B * S


@torch.no_grad()
def validate(router, blocks, embed, val_loader):
    total_loss = 0.0
    total_tokens = 0
    n = 0
    for batch in val_loader:
        data, total_tok, _ = batch
        if isinstance(data, list):
            for row in data:
                l, s = eval_row(router, blocks, embed, row)
                total_loss += l
                total_tokens += s
        else:
            l, s = eval_batch(router, blocks, embed, data)
            total_loss += l
            total_tokens += s
        n += 1
        if n >= 10:
            break
    return total_loss / total_tokens if total_tokens > 0 else 0.0


# ─── Expert activation tracking ────────────────────────────────────────

activation_counter = Counter()


def track_routing(query, router):
    active = router.route(query)
    for nid, _, _ in active[:3]:
        activation_counter[nid] += 1
    return active


# ─── Main benchmark ─────────────────────────────────────────────────────

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    dev = torch.cuda.get_device_properties(0)
    print("=" * 80)
    print(f"  2048 REAL-DOCUMENT BENCHMARK: Packed vs Unpacked")
    print(f"  GPU: {torch.cuda.get_device_name(0)}  VRAM: {dev.total_memory/1e9:.1f}GB")
    print("=" * 80)

    embed_dim = 1024
    n_experts = 16
    n_steps   = 100
    monitor = GPUMonitor()

    # Dataset: 80% train, 20% validation
    n_docs = 2000
    ds = VarLenDataset(n_docs)
    n_val = int(n_docs * 0.2)
    n_train = n_docs - n_val

    train_ds, val_ds = torch.utils.data.random_split(
        ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED)
    )

    avg_len = ds.avg_len()
    pad_ratio = ds.padding_ratio()
    print(f"\n  Dataset: {n_docs} docs, avg len={avg_len:.0f}, "
          f"padding ratio (if fixed {MAX_SEQ})={pad_ratio:.1%}")

    csv_path = os.path.join(SAVE_DIR, "real_doc_benchmark.csv")
    fields = ["mode", "step_s", "tok_s", "gpu_pct", "vram_gb",
              "avg_loss", "actual_pad_pct", "expert_entropy"]

    configs = [
        ("packed",   32, lambda b: collate_packed(b)),
        ("unpacked",  2, lambda b: collate_unpacked(b)),
    ]

    for mode, bs, collate_fn in configs:
        print(f"\n{'=' * 40}  {mode.upper()} (batch={bs})  {'=' * 40}")
        router, blocks = make_model(n_experts, embed_dim)
        embed = nn.Embedding(VOCAB, embed_dim).cuda()

        train_loader = DataLoader(
            train_ds, batch_size=bs, shuffle=True, collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_ds, batch_size=bs, shuffle=False, collate_fn=collate_fn,
        )

        step_times = []
        losses = []
        token_counts = []
        padding_counts = []
        gpu_utils = []
        activation_counter.clear()

        step = 0
        for batch in train_loader:
            if step >= n_steps:
                break

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            loss, tok = train_step(router, blocks, embed, batch, activation_counter)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            _, _, pad = batch
            step_times.append(elapsed)
            losses.append(loss)
            token_counts.append(tok)
            padding_counts.append(pad)
            gpu_utils.append(monitor.sample())

            if step % 20 == 0:
                val_loss = validate(router, blocks, embed, val_loader)
                print(f"  step={step:>3d}  {tok/elapsed:>8.0f} tok/s  "
                      f"step={elapsed*1000:.1f}ms  loss={loss:.4f}  val_loss={val_loss:.4f}")
            step += 1

        avg_step = sum(step_times) / len(step_times)
        avg_tok_s = sum(token_counts) / sum(step_times)
        avg_gpu = {k: sum(d[k] for d in gpu_utils)/len(gpu_utils) for k in gpu_utils[0]}
        avg_loss = sum(losses) / len(losses)
        total_pad = sum(padding_counts)
        total_real = sum(token_counts)
        actual_pad_pct = total_pad / (total_real + total_pad) * 100 if (total_real + total_pad) > 0 else 0

        # Expert entropy
        total_acts = sum(activation_counter.values())
        if total_acts > 0:
            probs = [c / total_acts for c in activation_counter.values()]
            entropy = -sum(p * math.log2(p) for p in probs if p > 0)
            max_entropy = math.log2(len(activation_counter))
            ent_ratio = entropy / max_entropy if max_entropy > 0 else 0
        else:
            ent_ratio = 0

        print(f"\n  Avg step: {avg_step*1000:.1f}ms  tok/s: {avg_tok_s:.0f}  "
              f"GPU: {avg_gpu['gpu_pct']:.0f}%  actual pad: {actual_pad_pct:.1f}%")
        print(f"  Avg loss: {avg_loss:.4f}  Expert entropy: {ent_ratio:.3f} "
              f"(uniform=1.0)")

        with open(csv_path, "a" if mode == "unpacked" else "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if mode == "packed":
                w.writeheader()
            w.writerow({
                "mode": mode,
                "step_s": round(avg_step, 5),
                "tok_s": round(avg_tok_s),
                "gpu_pct": round(avg_gpu["gpu_pct"], 1),
                "vram_gb": round(avg_gpu["vram_gb"], 2),
                "avg_loss": round(avg_loss, 4),
                "actual_pad_pct": round(actual_pad_pct, 2),
                "expert_entropy": round(ent_ratio, 4),
            })

        del router, blocks
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n  -> {csv_path}")
    print(f"\n{'=' * 80}")
    print(f"  Summary:")
    with open(csv_path) as f:
        print(f.read())

    if hasattr(monitor, 'close'):
        monitor.close()


if __name__ == "__main__":
    main()
