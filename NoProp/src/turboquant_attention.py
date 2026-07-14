"""
TurboQuant — AF.md #7.
Adaptive-bit-width PolarQuant + streaming Lloyd-Max + cross-layer KV sharing.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════
# Random orthogonal matrix (PolarQuant rotation)
# ═══════════════════════════════════════════════════

def random_orthogonal_matrix(dim: int, device: torch.device | None = None) -> torch.Tensor:
    H = torch.randn(dim, dim, device=device)
    Q, R = torch.linalg.qr(H)
    d = torch.diag(R).sign()
    Q = Q * d.unsqueeze(0)
    return Q


# ═══════════════════════════════════════════════════
# Streaming Lloyd-Max centroids (online update)
# ═══════════════════════════════════════════════════

class StreamingCentroids:
    """Momentum-based online centroid fitting with adaptive bit-width."""
    def __init__(self, dim: int, num_bits: int = 3, momentum: float = 0.9):
        self.dim = dim
        self.num_bits = num_bits
        self.num_levels = 1 << num_bits
        self.momentum = momentum
        self.centroids: torch.Tensor | None = None
        self._count: int = 0

    def update(self, data: torch.Tensor):
        flat = data.view(-1)
        if self.centroids is None:
            mins, maxs = flat.min(), flat.max()
            self.centroids = torch.linspace(mins, maxs, self.num_levels, device=data.device)
        else:
            self.centroids = self.centroids.to(data.device)
        dists = (flat.unsqueeze(-1) - self.centroids.unsqueeze(0)).abs()
        assignments = dists.argmin(dim=-1)
        for i in range(self.num_levels):
            mask = assignments == i
            if mask.any():
                new_c = flat[mask].mean()
                self.centroids[i] = (self.momentum * self.centroids[i] +
                                     (1.0 - self.momentum) * new_c)
        self._count += 1

    def quantize(self, x: torch.Tensor, ste: bool = False) -> torch.Tensor:
        if self.centroids is None:
            return x
        cents = self.centroids.to(x.device)
        flat = x.view(-1)
        dists = (flat.unsqueeze(-1) - cents.unsqueeze(0)).abs()
        assignments = dists.argmin(dim=-1)
        quantized = cents[assignments].view(x.shape)
        if ste:
            return x + (quantized - x).detach()
        return quantized


# ═══════════════════════════════════════════════════
# PolarQuant with adaptive bit-width
# ═══════════════════════════════════════════════════

class AdaptivePolarQuantTransform(nn.Module):
    """PolarQuant rotation + streaming centroids with adaptive bit selection."""
    def __init__(self, dim: int, min_bits: int = 2, max_bits: int = 5,
                 default_bits: int = 3, momentum: float = 0.9):
        super().__init__()
        self.dim = dim
        self.min_bits = min_bits
        self.max_bits = max_bits
        self.default_bits = default_bits

        Pi = random_orthogonal_matrix(dim)
        self.register_buffer("rotation_matrix", Pi)
        self.streaming = StreamingCentroids(dim, default_bits, momentum)

    def select_bits(self, residual_energy: float) -> int:
        if residual_energy > 0.1:
            return self.max_bits
        elif residual_energy > 0.01:
            return self.default_bits
        return self.min_bits

    def fit_centroids(self, kv_samples: torch.Tensor):
        rotated = kv_samples @ self.rotation_matrix.T
        self.streaming.update(rotated)

    def quantize(self, x: torch.Tensor, ste: bool = False) -> torch.Tensor:
        rot = self.rotation_matrix.to(x.device)
        rotated = x @ rot.T
        return self.streaming.quantize(rotated, ste=ste)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.quantize(x)


# ═══════════════════════════════════════════════════
# QJL Residual Correction
# ═══════════════════════════════════════════════════

class QJLResidualCorrection(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        R = torch.randn(dim, dim) / math.sqrt(dim)
        self.register_buffer("random_proj", R)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        orig_shape = v.shape
        flat = v.reshape(-1, orig_shape[-1])
        proj = self.random_proj.to(v.device)
        result = (proj @ flat.T).T
        return result.sign().reshape(orig_shape)


# ═══════════════════════════════════════════════════
# Cross-Layer KV Cache & Sharing
# ═══════════════════════════════════════════════════

class CrossLayerKVCache:
    """KV cache with cross-layer residual sharing and adaptive bit-width."""
    def __init__(self, max_layers: int = 32, max_seq_len: int = 2048):
        self.max_layers = max_layers
        self.max_seq_len = max_seq_len
        self.k_cache: dict[int, torch.Tensor] = {}
        self.v_cache: dict[int, torch.Tensor] = {}
        self.layer_deltas: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def set_layer_kv(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        if layer_idx == 0:
            self.k_cache[layer_idx] = k
            self.v_cache[layer_idx] = v
        else:
            prev_k = self.k_cache.get(layer_idx - 1)
            if prev_k is not None:
                self.layer_deltas[layer_idx] = (k - prev_k, v - self.v_cache.get(layer_idx - 1, v))
            self.k_cache[layer_idx] = k
            self.v_cache[layer_idx] = v
        if len(self.k_cache) > self.max_layers:
            oldest = min(self.k_cache.keys())
            del self.k_cache[oldest]
            del self.v_cache[oldest]
            self.layer_deltas.pop(oldest, None)

    def reconstruct(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_idx in self.k_cache:
            return self.k_cache[layer_idx], self.v_cache[layer_idx]
        if layer_idx in self.layer_deltas and layer_idx - 1 in self.k_cache:
            dk, dv = self.layer_deltas[layer_idx]
            return self.k_cache[layer_idx - 1] + dk, self.v_cache[layer_idx - 1] + dv
        raise KeyError(f"Layer {layer_idx} not in cache")


# ═══════════════════════════════════════════════════
# TurboQuantKVCompression (updated)
# ═══════════════════════════════════════════════════

class TurboQuantKVCompression(nn.Module):
    def __init__(self, dim: int, num_quant_bits: int = 3, use_ste: bool = True):
        super().__init__()
        self.dim = dim
        self.use_ste = use_ste
        self.polar = AdaptivePolarQuantTransform(dim, default_bits=num_quant_bits)
        self.qjl = QJLResidualCorrection(dim)
        self.kv_cache = CrossLayerKVCache()

    def compress(self, key: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        k_quant = self.polar(key)
        v_quant = self.polar(value)
        k_residual = key - k_quant
        v_residual = value - v_quant
        k_sign = self.qjl(k_residual)
        v_sign = self.qjl(v_residual)
        return k_quant, v_quant, k_sign * v_sign

    def compress_attention(self, query: torch.Tensor, key: torch.Tensor,
                           value: torch.Tensor, layer_idx: int | None = None) -> torch.Tensor:
        k_quant, v_quant, _ = self.compress(key, value)
        if layer_idx is not None:
            self.kv_cache.set_layer_kv(layer_idx, k_quant, v_quant)
        scores = query @ k_quant.transpose(-2, -1) / math.sqrt(query.size(-1))
        weights = F.softmax(scores, dim=-1)
        return weights @ v_quant

    def compress_attention_cached(self, query: torch.Tensor, layer_idx: int) -> torch.Tensor:
        k, v = self.kv_cache.reconstruct(layer_idx)
        scores = query @ k.transpose(-2, -1) / math.sqrt(query.size(-1))
        weights = F.softmax(scores, dim=-1)
        return weights @ v

    def get_stats(self) -> dict:
        return {
            "cache_layers": len(self.kv_cache.k_cache),
            "active_bits": self.polar.streaming.num_bits,
            "centroid_count": self.polar.streaming.num_levels,
        }


# ═══════════════════════════════════════════════════
# Legacy aliases (backward compat)
# ═══════════════════════════════════════════════════

def lloyd_max_centroids(data: torch.Tensor, num_bits: int = 3, num_iters: int = 20) -> torch.Tensor:
    sc = StreamingCentroids(data.size(-1), num_bits)
    sc.centroids = None
    for _ in range(num_iters):
        sc.update(data)
    return sc.centroids if sc.centroids is not None else torch.linspace(data.min(), data.max(), 1 << num_bits)


class PolarQuantTransform(AdaptivePolarQuantTransform):
    def __init__(self, dim: int, num_quant_bits: int = 3):
        super().__init__(dim=dim, default_bits=num_quant_bits)
