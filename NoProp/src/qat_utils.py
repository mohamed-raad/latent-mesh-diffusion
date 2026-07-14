"""
QAT (Quantization-Aware Training) — multi-bit quantization (4-bit NF4, 8-bit INT8).
Training in BF16; apply QAT after training for INT8/NF4 calibration and export.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# NF4 quantile levels (from QLoRA: 16 levels for 4-bit normal float)
NF4_LEVELS = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
])


def _scale_from_levels(w: torch.Tensor, levels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize weights using given levels, return (quantized_dequantized, scale)."""
    w_flat = w.flatten()
    abs_max = w_flat.abs().max().clamp(min=1e-6)
    scale = abs_max / levels[-1]
    indices = (w_flat / scale).clip(levels[0], levels[-1])
    dist = (indices.unsqueeze(-1) - levels.unsqueeze(0)).abs()
    nearest = distances.min(dim=-1).indices
    w_q = levels[nearest].reshape(w.shape) * scale
    return w_q, scale


class FakeQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, num_bits, use_nf4):
        if num_bits == 4 and use_nf4:
            device = x.device
            levels = NF4_LEVELS.to(device)
            w_q, _ = _scale_from_levels(x, levels)
            return w_q
        else:
            scale = x.abs().max().clamp(min=1e-6) / (2 ** (num_bits - 1) - 1)
            x_q = (x / scale).round().clamp(-2 ** (num_bits - 1), 2 ** (num_bits - 1) - 1)
            return x_q * scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class QATLinear(nn.Module):
    """Linear with fake-quant weights; preserves dtype. Supports 4-bit NF4 and 8-bit INT8."""
    def __init__(self, base: nn.Linear, num_bits: int = 8, use_nf4: bool = True):
        super().__init__()
        self.num_bits = num_bits
        self.use_nf4 = use_nf4 and num_bits == 4
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.weight = nn.Parameter(base.weight.data.clone())
        self.bias = nn.Parameter(base.bias.data.clone()) if base.bias is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            w_q = FakeQuantizeSTE.apply(self.weight, self.num_bits, self.use_nf4)
        else:
            w_q = self.weight
        return F.linear(x, w_q, self.bias)

    def export_quantized(self) -> tuple[torch.Tensor, torch.Tensor | None, int]:
        """Export weight as quantized int type + scale. Returns (qweight, scale, num_bits)."""
        if self.num_bits == 4 and self.use_nf4:
            device = self.weight.device
            levels = NF4_LEVELS.to(device)
            w_flat = self.weight.data.flatten()
            abs_max = w_flat.abs().max().clamp(min=1e-6)
            scale = abs_max / levels[-1]
            indices = (w_flat / scale).clip(levels[0], levels[-1])
            dists = (indices.unsqueeze(-1) - levels.unsqueeze(0)).abs()
            nearest = dists.min(dim=-1).indices.to(torch.uint8)
            return nearest, scale, 4
        else:
            w = self.weight.data
            scale = w.abs().max().clamp(min=1e-6) / (2 ** (self.num_bits - 1) - 1)
            qw = (w / scale).round().clamp(-2 ** (self.num_bits - 1), 2 ** (self.num_bits - 1) - 1)
            if self.num_bits <= 8:
                return qw.to(torch.int8), scale, self.num_bits
            return qw.to(torch.int16), scale, self.num_bits


def apply_qat(module: nn.Module, num_bits: int = 8, use_nf4: bool = True) -> nn.Module:
    """Replace nn.Linear with QATLinear. Skips embedding/lm_head."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            if name in ("token_embed", "lm_head"):
                continue
            ql = QATLinear(child, num_bits=num_bits, use_nf4=use_nf4)
            setattr(module, name, ql)
        else:
            apply_qat(child, num_bits=num_bits, use_nf4=use_nf4)
    return module


def strip_qat(module: nn.Module) -> nn.Module:
    """Convert QATLinear back to nn.Linear."""
    for name, child in list(module.named_children()):
        if isinstance(child, QATLinear):
            lin = nn.Linear(child.in_features, child.out_features,
                            bias=child.bias is not None)
            with torch.no_grad():
                lin.weight.copy_(child.weight.to(lin.weight.dtype))
                if child.bias is not None and lin.bias is not None:
                    lin.bias.copy_(child.bias.to(lin.bias.dtype))
            setattr(module, name, lin)
        else:
            strip_qat(child)
    return module
