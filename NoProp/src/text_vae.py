"""
text_vae.py — Text VAE for Latent Mesh Diffusion Computer.

Teaches the latent space to contain semantic information via
variational inference, replacing the simple projection in
UniversalLatentSpace with a proper encoder/decoder pair.

Architecture (borrowed from Cola-DLM, adapted for embedding-level mesh):

    Embedding [B, S, d_model]
        │
        ▼
    TextVAEEncoder (patch conv + transformer)
        │
        ▼
    Diagonal Gaussian (mean, logvar) [B, N, d_latent]
        │
        ├── z ~ q(z|x)  (training — reparameterized sample)
        └── mean        (inference — deterministic mode)
        │
        ▼
    MeshRouter / Experts / Consensus   (unchanged)
        │
        ▼
    TextVAEDecoder (transformer + unpatch)
        │
        ▼
    Reconstructed Embedding [B, S, d_model]

Loss:
    L_VAE = ||x - x_hat||² + beta * KL(q(z|x) || N(0, I))
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════
# Diagonal Gaussian Distribution
# ═══════════════════════════════════════════════════

@dataclass
class GaussianDistribution:
    mean: torch.Tensor
    logvar: torch.Tensor

    def __post_init__(self):
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

    def sample(self) -> torch.Tensor:
        eps = torch.randn_like(self.mean)
        return self.mean + self.std * eps

    def mode(self) -> torch.Tensor:
        return self.mean

    def kl(self) -> torch.Tensor:
        return -0.5 * torch.sum(
            1 + self.logvar - self.mean.pow(2) - self.var,
            dim=-1
        )


# ═══════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════

@dataclass
class TextVAEConfig:
    d_model: int = 1024
    d_latent: int = 256
    n_latent_nodes: int = 64
    patch_size: int = 2
    n_encoder_blocks: int = 4
    n_decoder_blocks: int = 4
    n_heads: int = 8
    ffn_mult: int = 4
    dropout: float = 0.0
    kl_beta: float = 0.001
    use_variation: bool = True
    norm_eps: float = 1e-5
    hierarchical: bool = False


# ═══════════════════════════════════════════════════
# SwiGLU activation
# ═══════════════════════════════════════════════════

class SwiGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


def build_causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """Build a causal attention mask: [seq_len, seq_len].

    Upper triangle (j > i) is -inf, lower triangle (j <= i) is 0.
    """
    mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask


# ═══════════════════════════════════════════════════
# VAE Transformer Block
# ═══════════════════════════════════════════════════

class VAEBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        ffn_dim: int,
        dropout: float = 0.0,
        causal: bool = False,
    ):
        super().__init__()
        self.causal = causal
        self.norm1 = nn.LayerNorm(dim, eps=1e-5)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, eps=1e-5)

        ffn_inner = ffn_dim * 2  # SwiGLU halves it
        self.ffn_proj = nn.Linear(dim, ffn_inner)
        self.ffn_act = SwiGLU()
        self.ffn_out = nn.Linear(ffn_inner // 2, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + self.dropout(attn_out)

        h = self.norm2(x)
        h = self.ffn_proj(h)
        h = self.ffn_act(h)
        h = self.ffn_out(h)
        x = x + self.dropout(h)
        return x


# ═══════════════════════════════════════════════════
# TextVAEEncoder
# ═══════════════════════════════════════════════════

class TextVAEEncoder(nn.Module):
    """Maps token embeddings [B, S, d_model] → Gaussian latents [B, N, d_latent].

    Uses patch convolution for sequence compression, then transformer blocks,
    then projects to mean + logvar.

    When hierarchical=True, transformer blocks use causal masking at the
    block level (Cola-DLM-style): each block attends to previous blocks only.
    """

    def __init__(self, cfg: TextVAEConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_size = cfg.patch_size
        self.hierarchical = cfg.hierarchical

        self.patch_conv = nn.Conv1d(
            cfg.d_model, cfg.d_model,
            kernel_size=cfg.patch_size,
            stride=cfg.patch_size,
        )
        self.norm_pre = nn.LayerNorm(cfg.d_model, eps=cfg.norm_eps)

        self.blocks = nn.ModuleList([
            VAEBlock(
                cfg.d_model, cfg.n_heads, cfg.d_model * cfg.ffn_mult,
                cfg.dropout, causal=cfg.hierarchical,
            )
            for _ in range(cfg.n_encoder_blocks)
        ])

        self.norm_post = nn.LayerNorm(cfg.d_model, eps=cfg.norm_eps)

        out_dim = cfg.d_latent * 2 if cfg.use_variation else cfg.d_latent
        self.out_proj = nn.Linear(cfg.d_model, out_dim)

    def forward(
        self,
        x: torch.Tensor,
    ) -> GaussianDistribution | torch.Tensor:
        B, S, D = x.shape

        # Patch embedding: [B, S, D] → [B, N, D] where N = S // patch_size
        x = x.permute(0, 2, 1)
        x = self.patch_conv(x)
        x = x.permute(0, 2, 1)

        N = x.size(1)
        x = self.norm_pre(x)

        attn_mask = None
        if self.hierarchical:
            attn_mask = build_causal_mask(N, device=x.device)

        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)

        x = self.norm_post(x)
        x = self.out_proj(x)

        if self.cfg.use_variation:
            mean, logvar = x.chunk(2, dim=-1)
            return GaussianDistribution(mean=mean, logvar=logvar)
        else:
            return x


# ═══════════════════════════════════════════════════
# TextVAEDecoder
# ═══════════════════════════════════════════════════

class TextVAEDecoder(nn.Module):
    """Maps latents [B, N, d_latent] → reconstructed embeddings [B, S, d_model].

    Uses transformer blocks then unpatch convolution to recover sequence length.
    """

    def __init__(self, cfg: TextVAEConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_size = cfg.patch_size

        self.in_proj = nn.Linear(cfg.d_latent, cfg.d_model)
        self.norm_pre = nn.LayerNorm(cfg.d_model, eps=cfg.norm_eps)

        self.blocks = nn.ModuleList([
            VAEBlock(cfg.d_model, cfg.n_heads, cfg.d_model * cfg.ffn_mult, cfg.dropout)
            for _ in range(cfg.n_decoder_blocks)
        ])

        self.norm_post = nn.LayerNorm(cfg.d_model, eps=cfg.norm_eps)
        self.unpatch = nn.Linear(cfg.d_model, cfg.patch_size * cfg.d_model)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, N, D_lat = z.shape

        x = self.in_proj(z)
        x = self.norm_pre(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm_post(x)
        x = self.unpatch(x)

        # Unpatch: [B, N, patch_size * D] → [B, N * patch_size, D]
        x = x.view(B, N, self.patch_size, -1)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(B, N * self.patch_size, -1)

        return x


# ═══════════════════════════════════════════════════
# TextVAE — combined module
# ═══════════════════════════════════════════════════

class TextVAE(nn.Module):
    """Full Text VAE: encoder → latent → decoder.

    Drop-in upgrade for UniversalLatentSpace with variational inference.
    """

    def __init__(self, cfg: TextVAEConfig | None = None):
        super().__init__()
        self.cfg = cfg or TextVAEConfig()
        self.encoder = TextVAEEncoder(self.cfg)
        self.decoder = TextVAEDecoder(self.cfg)
        self.d_model = self.cfg.d_model
        self.d_latent = self.cfg.d_latent
        self.n_latent_nodes = self.cfg.n_latent_nodes

    def encode(
        self,
        x: torch.Tensor,
    ) -> GaussianDistribution:
        """Encode embeddings to diagonal Gaussian latent."""
        dist = self.encoder(x)
        return dist

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latents to reconstructed embeddings."""
        x_hat = self.decoder(z)
        return x_hat

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[GaussianDistribution, torch.Tensor]:
        """Full VAE forward: encode → sample → decode.

        Args:
            x: [B, S, d_model] token embeddings.

        Returns:
            (dist, x_hat) where:
              - dist: GaussianDistribution for KL computation
              - x_hat: [B, S', d_model] reconstructed embeddings
        """
        S_orig = x.shape[1]
        dist = self.encoder(x)
        z = dist.sample()
        x_hat = self.decoder(z)

        # Trim or pad to match original length
        if x_hat.shape[1] > S_orig:
            x_hat = x_hat[:, :S_orig, :]
        elif x_hat.shape[1] < S_orig:
            pad = torch.zeros(x.shape[0], S_orig - x_hat.shape[1], x_hat.shape[2], device=x.device)
            x_hat = torch.cat([x_hat, pad], dim=1)

        return dist, x_hat

    def loss(
        self,
        x: torch.Tensor,
        dist: GaussianDistribution,
        x_hat: torch.Tensor,
        beta: float | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute VAE loss: reconstruction + KL.

        Args:
            x: Original embeddings [B, S, d_model].
            dist: GaussianDistribution from encoder.
            x_hat: Reconstructed embeddings [B, S, d_model].
            beta: KL weight (defaults to cfg.kl_beta).

        Returns:
            dict with keys: 'loss', 'recon', 'kl'
        """
        beta = beta if beta is not None else self.cfg.kl_beta

        recon_loss = F.mse_loss(x_hat, x, reduction='mean')

        kl_per_latent = dist.kl()
        kl_loss = kl_per_latent.mean()

        total = recon_loss + beta * kl_loss

        return {
            'loss': total,
            'recon': recon_loss.detach(),
            'kl': kl_loss.detach(),
        }

    @torch.no_grad()
    def project_tokens(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """Backward-compat: encode to latent mode (deterministic mean).

        Matches UniversalLatentSpace.project_tokens interface.
        """
        if token_embeddings.dim() == 2:
            token_embeddings = token_embeddings.unsqueeze(0)
        dist = self.encoder(token_embeddings)
        return dist.mode()
