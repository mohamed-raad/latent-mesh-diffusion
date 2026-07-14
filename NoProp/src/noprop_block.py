import math
import os
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 16, alpha: float = 16.0):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        in_dim, out_dim = base.in_features, base.out_features
        self.lora_a = nn.Parameter(torch.randn(in_dim, rank) * 0.02)
        self.lora_b = nn.Parameter(torch.zeros(rank, out_dim))
        self.scaling = alpha / rank

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.lora_a @ self.lora_b) * self.scaling


def inject_lora_into_block(block: nn.Module, rank: int = 16, alpha: float = 16.0) -> list[str]:
    if not hasattr(block, "_lora_injected"):
        dummy = torch.randn(1, block.embed_dim)
        block(dummy, torch.tensor([[1.0]]))
    names: list[str] = []
    for name, child in block.named_modules():
        if isinstance(child, nn.Linear) and not isinstance(child, LoRALinear) and child.weight.requires_grad:
            parent_path = name.rsplit(".", 1)
            if len(parent_path) == 1:
                parent = block
                key = parent_path[0]
            else:
                parent = dict(block.named_modules())[parent_path[0]]
                key = parent_path[1]
            lora_lin = LoRALinear(child, rank, alpha)
            setattr(parent, key, lora_lin)
            names.append(name)
    return names


def lora_parameters(module: nn.Module) -> Iterable[torch.Tensor]:
    for child in module.modules():
        if isinstance(child, LoRALinear):
            yield child.lora_a
            yield child.lora_b

class NoPropBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 4, ff_mult: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * ff_mult),
            nn.GELU(),
            nn.Linear(embed_dim * ff_mult, embed_dim),
        )
        self.input_proj = nn.LazyLinear(embed_dim)
        self.time_emb = nn.Linear(embed_dim, embed_dim)
        self.optimizer = None
        self.node_anchor = None
        self._compiled = None

    def _infer_batch(self, *tensors: torch.Tensor) -> int:
        for t in tensors:
            if t is not None:
                return t.size(0)
        return 1

    def forward(self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        B = self._infer_batch(x, t, context)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if t.dim() == 1:
            t = t.view(-1, 1).expand(B, self.embed_dim).unsqueeze(1)
        elif t.dim() == 2:
            if t.size(-1) == 1:
                t = t.expand(B, self.embed_dim).unsqueeze(1)
            else:
                t = t.unsqueeze(1)

        h = self.input_proj(x)
        t_feat = self.time_emb(t)
        h = h + t_feat

        attn_out, _ = self.attn(h, h, h)
        h = self.norm1(h + attn_out)
        ff_out = self.ff(h)
        h = self.norm2(h + ff_out)
        return h.squeeze(1) if h.size(1) == 1 else h

    def compiled_forward(self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        if self._compiled is None:
            self._compiled = torch.compile(
                self.forward,
                mode="max-autotune-no-cudagraphs",
                fullgraph=True,
            )
        return self._compiled(x, t, context)

    def local_loss(self, pred: torch.Tensor, target: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        mse = F.mse_loss(pred, target)
        if t is not None:
            mse = mse * snr_grad_weight(t)
        return mse

    def configure_optimizer(self, lr: float = 1e-3, weight_decay: float = 0.0):
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

    def local_step(self, pred: torch.Tensor, target: torch.Tensor, retain_graph: bool = False,
                   t: torch.Tensor | None = None) -> float:
        if self.optimizer is None:
            self.configure_optimizer()
        self.optimizer.zero_grad()
        loss = self.local_loss(pred, target, t=t)
        loss.backward(retain_graph=retain_graph)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32) / max(half - 1, 1))
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        args = t.unsqueeze(-1).float() * self.freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)


def checkpoint_atomic(save_dir: str, step: int, model_state: dict, optimizer_state: dict, metadata: dict):
    os.makedirs(save_dir, exist_ok=True)
    tmp = os.path.join(save_dir, f"step_{step}.tmp")
    final = os.path.join(save_dir, f"step_{step}.pt")
    torch.save({
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer_state,
        "metadata": metadata,
    }, tmp)
    os.replace(tmp, final)


def load_checkpoint(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


def snr_grad_weight(t: torch.Tensor, eta: float = 1.0) -> torch.Tensor:
    t_safe = t.float().clamp(min=1e-6, max=1.0 - 1e-6)
    alpha_bar = (torch.cos(t_safe * math.pi / 2.0) ** 2).clamp(min=1e-6, max=1.0 - 1e-6)
    dalpha = (-math.pi / 2.0 * torch.sin(t_safe * math.pi)).abs()
    gamma = alpha_bar / (1.0 - alpha_bar).clamp(min=1e-6)
    w = (0.5 * eta * gamma * dalpha).detach()
    return w.mean().clamp(min=1e-6, max=10.0)
