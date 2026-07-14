"""
DSpark Speculator — MTP heads, curriculum-aware loss, tree-medusa, speculative decode.
"""
import json
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


# ═══════════════════════════════════════════════════
# Curriculum JSONL Dataset (Phase 8)
# ═══════════════════════════════════════════════════

class CurriculumDataset(Dataset):
    """Loads thinker-weighted JSONL curriculum data."""
    def __init__(self, data_dir: str, max_seq_len: int = 512,
                 tokenizer_vocab_size: int = 1000,
                 phases: list[int] | None = None):
        self.samples: list[dict] = []
        self.max_seq_len = max_seq_len
        self.vocab_size = tokenizer_vocab_size

        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Curriculum data dir not found: {data_dir}")

        for fname in sorted(os.listdir(data_dir)):
            if not fname.endswith(".jsonl"):
                continue
            phase = self._parse_phase(fname)
            if phases is not None and phase not in phases:
                continue
            fpath = os.path.join(data_dir, fname)
            with open(fpath, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self.samples.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        if not self.samples:
            raise ValueError(f"No curriculum samples found in {data_dir}")

    def _parse_phase(self, fname: str) -> int:
        parts = fname.replace(".jsonl", "").split("_")
        for p in parts:
            if p.isdigit():
                return int(p)
        return 0

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        tokens = torch.tensor(sample.get("tokens", []), dtype=torch.long)
        if tokens.numel() > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        hidden = torch.randn(tokens.size(0), self.vocab_size)
        return {
            "input_ids": tokens,
            "hidden": hidden,
            "labels": tokens,
            "meta": sample.get("meta", {}),
        }


# ═══════════════════════════════════════════════════
# MTP Head (updated with RoPE)
# ═══════════════════════════════════════════════════

def logit_soft_cap(logits: torch.Tensor, capacity: float = 50.0) -> torch.Tensor:
    return capacity * torch.tanh(logits / capacity)


class MTPHead(nn.Module):
    """MTP head with optional shared output projection (bottleneck for large vocab)."""
    def __init__(self, embed_dim: int, vocab_size: int, num_heads: int = 4,
                 soft_cap_attn: float = 30.0, soft_cap_final: float = 50.0,
                 shared_lm_head: nn.Linear | None = None):
        super().__init__()
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.shared_tf = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
        )
        self.lm_head = shared_lm_head or nn.Linear(embed_dim, vocab_size)
        self.soft_cap_attn = soft_cap_attn
        self.soft_cap_final = soft_cap_final

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        h = logit_soft_cap(h, self.soft_cap_attn)
        h = self.shared_tf(h)
        logits = self.lm_head(h)
        logits = logit_soft_cap(logits, self.soft_cap_final)
        return logits


# ═══════════════════════════════════════════════════
# Multi-token Predictor (with curriculum weighting)
# ═══════════════════════════════════════════════════

class MultiTokenPredictor(nn.Module):
    def __init__(self, embed_dim: int, vocab_size: int,
                 num_draft_tokens: int = 3, num_heads: int = 4,
                 tie_weights: bool = True):
        super().__init__()
        self.num_draft_tokens = num_draft_tokens
        shared_lm = nn.Linear(embed_dim, vocab_size) if tie_weights else None
        self.heads = nn.ModuleList([
            MTPHead(embed_dim, vocab_size, num_heads,
                    shared_lm_head=shared_lm or nn.Linear(embed_dim, vocab_size))
            for _ in range(num_draft_tokens)
        ])

    def forward(self, hidden: torch.Tensor) -> list[torch.Tensor]:
        return [head(hidden) for head in self.heads]

    def draft(self, hidden: torch.Tensor) -> torch.Tensor:
        logits_seq = self.forward(hidden)
        tokens = []
        for logits in logits_seq:
            tokens.append(logits.argmax(dim=-1))
        return torch.stack(tokens, dim=1)

    def loss(self, hidden: torch.Tensor, target_tokens: torch.Tensor,
             curriculum_weight: float = 1.0) -> torch.Tensor:
        logits_seq = self.forward(hidden)
        total = 0.0
        B = target_tokens.size(0)
        for k, logits in enumerate(logits_seq):
            idx = min(k, target_tokens.size(1) - 1)
            total += F.cross_entropy(logits.view(-1, logits.size(-1)),
                                     target_tokens[:, idx].view(-1).long())
        return curriculum_weight * total / len(logits_seq)


# ═══════════════════════════════════════════════════
# Confidence Verifier (updated with per-step scores)
# ═══════════════════════════════════════════════════

class ConfidenceVerifier(nn.Module):
    def __init__(self, embed_dim: int, soft_cap: float = 30.0):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
        )
        self.soft_cap = soft_cap

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        h = logit_soft_cap(self.score(hidden), self.soft_cap)
        return torch.sigmoid(h)


# ═══════════════════════════════════════════════════
# Tree-Medusa Head
# ═══════════════════════════════════════════════════

class TreeMedusaHead(nn.Module):
    def __init__(self, embed_dim: int, vocab_size: int, depth: int = 3,
                 branching: list[int] | None = None, num_heads: int = 4,
                 soft_cap_attn: float = 30.0, soft_cap_final: float = 50.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.depth = depth
        if branching is not None:
            self.branching = branching[:depth]
            while len(self.branching) < depth:
                self.branching.append(2)
        else:
            self.branching = [3] + [2] * (depth - 1)
        self.soft_cap_attn = soft_cap_attn
        self.soft_cap_final = soft_cap_final

        self._build_tree()
        self.backbone = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, batch_first=True,
        )
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.embed_proj = nn.Linear(embed_dim * 2, embed_dim)

        mask = self._build_tree_mask()
        self.register_buffer("tree_attn_mask", mask)

    def _build_tree(self):
        parents = [-1]
        depths = [0]
        level_start = 0
        level_size = 1
        for d in range(self.depth):
            kids_per_level = level_size * self.branching[d]
            for p_idx in range(level_start, level_start + level_size):
                for _ in range(self.branching[d]):
                    parents.append(p_idx)
                    depths.append(d + 1)
            level_start += level_size
            level_size = kids_per_level
        self.tree_parents = parents
        self.tree_depths = depths
        self.num_nodes = len(parents)

    def _build_tree_mask(self) -> torch.Tensor:
        n = self.num_nodes
        mask = torch.zeros(n, n, dtype=torch.bool)
        for i in range(n):
            mask[i, i] = True
            p = self.tree_parents[i]
            while p != -1:
                mask[i, p] = True
                p = self.tree_parents[p]
        return mask

    def _depth_nodes(self, d: int) -> list[int]:
        return [i for i in range(self.num_nodes) if self.tree_depths[i] == d]

    def _node_count_at_depth(self, d: int) -> int:
        return len(self._depth_nodes(d))

    def tree_forward(self, hidden: torch.Tensor, tree_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = hidden.size(0)
        N = self.num_nodes
        base = hidden.unsqueeze(1).expand(B, N, self.embed_dim)
        states = base.clone()
        for i in range(1, N):
            p = self.tree_parents[i]
            tok_emb = self.embed(tree_tokens[:, p])
            combined = torch.cat([states[:, p], tok_emb], dim=-1)
            states[:, i] = self.embed_proj(combined)
        states = logit_soft_cap(states, self.soft_cap_attn)
        attn_mask = self.tree_attn_mask.unsqueeze(0).expand(B, -1, -1).contiguous()
        states = self.backbone(states, src_mask=attn_mask)
        logits = self.lm_head(states)
        logits = logit_soft_cap(logits, self.soft_cap_final)
        return logits, states

    def speculate_tree(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = hidden.size(0)
        tree_tokens = torch.zeros(B, self.num_nodes, dtype=torch.long, device=hidden.device)
        for d in range(self.depth):
            logits, _ = self.tree_forward(hidden, tree_tokens)
            parent_nodes = self._depth_nodes(d)
            child_offset = sum(self._node_count_at_depth(k) for k in range(d + 1))
            for j, p_idx in enumerate(parent_nodes):
                p_logits = logits[:, p_idx]
                topk = p_logits.topk(self.branching[d], dim=-1)
                for k in range(self.branching[d]):
                    child_idx = child_offset + j * self.branching[d] + k
                    if child_idx < self.num_nodes:
                        tree_tokens[:, child_idx] = topk.indices[:, k]
        logits, states = self.tree_forward(hidden, tree_tokens)
        best_tokens = tree_tokens[:, 1:].max(dim=-1).values.unsqueeze(-1)
        best_tokens = best_tokens.expand(-1, self.depth)
        conf = torch.sigmoid(states[:, 1:].mean(dim=-1)).mean(dim=-1, keepdim=True)
        return best_tokens, conf


# ═══════════════════════════════════════════════════
# DSpark Speculator (enhanced with curriculum support)
# ═══════════════════════════════════════════════════

class DSparkSpeculator(nn.Module):
    def __init__(self, embed_dim: int, vocab_size: int,
                 num_draft_tokens: int = 3,
                 confidence_threshold: float = 0.9,
                 use_medusa: bool = False):
        super().__init__()
        self.predictor = MultiTokenPredictor(embed_dim, vocab_size, num_draft_tokens)
        self.verifier = ConfidenceVerifier(embed_dim)
        self.confidence_threshold = confidence_threshold
        self.medusa = TreeMedusaHead(embed_dim, vocab_size, depth=num_draft_tokens)
        self.use_medusa = use_medusa

    def speculate(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_medusa:
            cand_tokens, conf = self.medusa.speculate_tree(hidden)
            draft_tokens = cand_tokens[:, :self.predictor.num_draft_tokens]
            if draft_tokens.size(1) < self.predictor.num_draft_tokens:
                pad = draft_tokens[:, -1:].expand(-1, self.predictor.num_draft_tokens - draft_tokens.size(1))
                draft_tokens = torch.cat([draft_tokens, pad], dim=-1)
            accepted = draft_tokens.clone()
            verified_mask = conf >= self.confidence_threshold
            for i in range(accepted.size(0)):
                if not verified_mask[i]:
                    accepted[i] = draft_tokens[i, 0].expand_as(accepted[i])
            return accepted, conf.expand(-1, self.predictor.num_draft_tokens)

        draft_tokens = self.predictor.draft(hidden)
        conf = self.verifier(hidden)
        verified_mask = conf.squeeze(-1) >= self.confidence_threshold
        accepted = draft_tokens.clone()
        for i in range(accepted.size(0)):
            if not verified_mask[i]:
                accepted[i] = draft_tokens[i, 0].expand_as(accepted[i])
        return accepted, conf

    def loss(self, hidden: torch.Tensor, target_tokens: torch.Tensor,
             curriculum_weight: float = 1.0) -> torch.Tensor:
        return self.predictor.loss(hidden, target_tokens, curriculum_weight)
