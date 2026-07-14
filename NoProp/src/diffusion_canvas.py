"""
CanvasTransformer — multi-layer backbone with GQA attention.
Supports model size presets (tiny/small/standard/large) via model_sizes.py.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_alpha_bar(t: torch.Tensor) -> torch.Tensor:
    return (torch.cos(t * math.pi / 2.0) ** 2).clamp(min=1e-6, max=1.0 - 1e-6)


class UniformStateDiffusion:
    def __init__(self, num_steps: int = 50, schedule: str = "cosine"):
        self.num_steps = num_steps
        self.schedule = schedule

    def alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        if self.schedule == "cosine":
            return cosine_alpha_bar(t)
        return cosine_alpha_bar(t)

    def corrupt(self, clean_emb: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a = self.alpha_bar(t)
        noise = torch.randn_like(clean_emb)
        return a.sqrt() * clean_emb + (1.0 - a).sqrt() * noise, noise

    def timesteps(self, device: torch.device) -> torch.Tensor:
        return torch.linspace(1.0, 0.0, self.num_steps + 1, device=device)[:-1]


class GroupedQueryAttention(nn.Module):
    """Grouped-query attention for efficient inference."""
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.n_rep = n_heads // n_kv_heads

        self.wq = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        q = self.wq(x).view(B, L, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, L, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, L, self.n_kv_heads, self.head_dim)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=2)
            v = v.repeat_interleave(self.n_rep, dim=2)

        q, k = q.transpose(1, 2), k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        attn = attn.transpose(1, 2).contiguous().view(B, L, -1)
        return self.wo(attn)


class CanvasBlock(nn.Module):
    """Single transformer block: time embed -> GQA -> FFN."""
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm_in = nn.LayerNorm(d_model)
        self.attn = GroupedQueryAttention(d_model, n_heads, n_kv_heads)
        self.norm_mid = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor,
                frozen_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.norm_in(x + t_emb)
        attn_out = self.attn(h)
        h = self.norm_mid(h + attn_out)
        ff_out = self.ff(h)
        if frozen_mask is not None:
            ff_out = ff_out * (~frozen_mask).unsqueeze(-1).float()
        h = self.norm_out(h + ff_out)
        delta = h
        if frozen_mask is not None:
            delta = delta * (~frozen_mask).unsqueeze(-1).float()
        return x + delta


class CanvasTransformer(nn.Module):
    """Stack of CanvasBlocks forming the diffusion backbone."""
    def __init__(self, d_model: int, n_layers: int, n_heads: int,
                 n_kv_heads: int, d_ff: int, vocab_size: int,
                 tie_weights: bool = True):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            CanvasBlock(d_model, n_heads, n_kv_heads, d_ff)
            for _ in range(n_layers)
        ])
        self.lm_head = nn.Linear(d_model, vocab_size)
        if tie_weights:
            self.lm_head.weight = self.token_embed.weight

    @staticmethod
    def time_embed(t: torch.Tensor, dim: int) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half - 1, 1))
        args = t.unsqueeze(-1).float() * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)

    def forward(self, tokens: torch.Tensor, t: torch.Tensor,
                frozen_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, L = tokens.shape
        tok_emb = self.token_embed(tokens)
        t_emb = self.time_embed(t, self.d_model)
        h = tok_emb + t_emb
        for block in self.blocks:
            h = block(h, t_emb, frozen_mask=frozen_mask)
        logits = self.lm_head(h)
        return logits


class DiffusionCanvas:
    """Orchestrates diffusion over tokens using CanvasTransformer backbone."""
    def __init__(self, d_model: int, n_layers: int, n_heads: int,
                 n_kv_heads: int, d_ff: int, vocab_size: int,
                 canvas_len: int = 512, num_steps: int = 50,
                 entropy_threshold: float = 0.005,
                 global_cognitive_layer: nn.Module | None = None,
                 tie_weights: bool = True):
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.canvas_len = canvas_len
        self.num_steps = num_steps
        self.entropy_threshold = entropy_threshold
        self.gcl = global_cognitive_layer

        self.model = CanvasTransformer(d_model, n_layers, n_heads, n_kv_heads, d_ff, vocab_size,
                                       tie_weights=tie_weights)
        self.diffusion = UniformStateDiffusion(num_steps)

    def to(self, device: torch.device):
        self.model = self.model.to(device)
        if self.gcl is not None:
            self.gcl = self.gcl.to(device)
        return self

    def init_canvas(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.vocab_size, (batch_size, self.canvas_len), device=device)

    def denoise_step(self, canvas: torch.Tensor, t: torch.Tensor,
                     prev_pred: torch.Tensor | None = None,
                     frozen_mask: torch.Tensor | None = None
                     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.model(canvas, t, frozen_mask=frozen_mask)
        if self.gcl is not None:
            logits = self.gcl(logits)
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs.clamp(min=1e-8))).sum(dim=-1)
        cur_pred = logits.argmax(dim=-1)
        if prev_pred is not None:
            stable = (cur_pred == prev_pred).float()
            low_ent = (entropy < self.entropy_threshold).float()
            new_frozen = (stable * low_ent) > 0.5
            frozen_mask = frozen_mask | new_frozen if frozen_mask is not None else new_frozen
        else:
            frozen_mask = torch.zeros_like(cur_pred, dtype=torch.bool)
        return logits, cur_pred, entropy, frozen_mask

    def generate(self, batch_size: int = 1, device: torch.device | None = None,
                 max_blocks: int = 1) -> torch.Tensor:
        if device is None:
            device = next(self.model.parameters()).device
        self.model.eval()
        output_blocks = []
        for _ in range(max_blocks):
            canvas = self.init_canvas(batch_size, device)
            timesteps = self.diffusion.timesteps(device)
            prev_pred = None
            frozen_mask = None
            for step_idx, t_val in enumerate(timesteps):
                t = t_val.expand(batch_size, 1)
                with torch.no_grad():
                    logits, cur_pred, entropy, frozen_mask = self.denoise_step(
                        canvas, t, prev_pred=prev_pred, frozen_mask=frozen_mask
                    )
                if prev_pred is not None:
                    locked = frozen_mask
                    noise_tokens = torch.randint(0, self.vocab_size, canvas.shape, device=device)
                    canvas = torch.where(locked, cur_pred, noise_tokens)
                    if locked.float().mean() > 0.95:
                        canvas = cur_pred
                        break
                else:
                    canvas = logits.argmax(dim=-1)
                prev_pred = cur_pred
            output_blocks.append(canvas)
        return torch.cat(output_blocks, dim=-1) if len(output_blocks) > 1 else output_blocks[0]

    def generate_conditional(self, prompt_ids: torch.Tensor, device: torch.device | None = None,
                             max_new_tokens: int | None = None) -> torch.Tensor:
        if device is None:
            device = next(self.model.parameters()).device
        self.model.eval()
        B, prompt_len = prompt_ids.shape
        total_len = self.canvas_len if max_new_tokens is None else prompt_len + max_new_tokens
        if total_len > self.canvas_len:
            total_len = self.canvas_len

        canvas = torch.randint(0, self.vocab_size, (B, self.canvas_len), device=device)
        canvas[:, :prompt_len] = prompt_ids[:, :prompt_len]

        timesteps = self.diffusion.timesteps(device)
        prev_pred = None
        frozen_mask = torch.zeros(B, self.canvas_len, dtype=torch.bool, device=device)
        frozen_mask[:, :prompt_len] = True
        for step_idx, t_val in enumerate(timesteps):
            t = t_val.expand(B, 1)
            with torch.no_grad():
                frozen_only_gen = frozen_mask.clone()
                frozen_only_gen[:, :prompt_len] = False
                logits, cur_pred, entropy, frozen_mask = self.denoise_step(
                    canvas, t, prev_pred=prev_pred, frozen_mask=frozen_only_gen
                )
            frozen_mask[:, :prompt_len] = True
            if prev_pred is not None:
                noise_tokens = torch.randint(0, self.vocab_size, canvas.shape, device=device)
                noise_tokens[:, :prompt_len] = prompt_ids[:, :prompt_len]
                canvas = torch.where(frozen_mask, cur_pred, noise_tokens)
                gen_frozen = frozen_mask[:, prompt_len:]
                if gen_frozen.float().mean() > 0.95:
                    canvas = cur_pred
                    break
            else:
                canvas[:, prompt_len:] = cur_pred[:, prompt_len:]
            prev_pred = cur_pred
        return canvas[:, :total_len]

    def compute_loss(self, canvas: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        B, L = canvas.shape
        total_loss = 0.0
        timesteps = self.diffusion.timesteps(canvas.device)
        for t_val in timesteps:
            t = t_val.expand(B, 1)
            logits = self.model(canvas, t)
            if self.gcl is not None:
                logits = self.gcl(logits)
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), target.view(-1))
            total_loss = total_loss + loss
        return total_loss / len(timesteps)
