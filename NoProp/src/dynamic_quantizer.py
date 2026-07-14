"""
Dynamic per-expert quantization — BF16↔INT8 runtime switching per block.
Monitors loss plateau per expert; switches non-improving blocks to INT8,
reverts if loss spikes after quantization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Int8Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight_int8", torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer("scale", torch.ones(out_features, 1))
        self.register_buffer("bias", torch.zeros(out_features))

    def calibrate(self, weight: torch.Tensor, bias: torch.Tensor | None = None):
        with torch.no_grad():
            absmax = weight.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
            self.scale = absmax / 127.0
            self.weight_int8 = (weight / self.scale).round().clamp(-128, 127).to(torch.int8)
            if bias is not None:
                self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight_int8.to(x.dtype) * self.scale.to(x.dtype)
        return F.linear(x, w, self.bias.to(x.dtype))


def _has_lora(block: nn.Module) -> bool:
    for m in block.modules():
        if hasattr(m.__class__.__name__, "LoRALinear") and "LoRA" in type(m).__name__:
            return True
    return False


def quantize_expert_block(block: nn.Module):
    """Replace selected Linears in a NoPropBlock with Int8Linear (ff layers + projections)."""
    _names = {"ff.0", "ff.2", "input_proj", "time_emb"}
    for name, child in list(block.named_children()):
        if name not in _names:
            continue
        if isinstance(child, nn.Linear) and not isinstance(child, Int8Linear):
            qlin = Int8Linear(child.in_features, child.out_features)
            qlin.calibrate(child.weight.data, child.bias.data if child.bias is not None else None)
            setattr(block, name, qlin)


def dequantize_expert_block(block: nn.Module):
    """Restore Int8Linear back to nn.Linear inside a NoPropBlock."""
    _names = {"ff.0", "ff.2", "input_proj", "time_emb"}
    for name, child in list(block.named_children()):
        if name not in _names:
            continue
        if isinstance(child, Int8Linear):
            w = child.weight_int8.to(torch.float32) * child.scale.to(torch.float32)
            lin = nn.Linear(child.in_features, child.out_features)
            lin.weight.data.copy_(w)
            lin.bias.data.copy_(child.bias)
            setattr(block, name, lin)


class DynamicQuantizer:
    """
    Per-expert loss-monitoring quantizer.

    Each block starts in BF16. After `patience` steps without improvement,
    it is quantized to INT8. If loss spikes after quantization, it reverts.

    Args:
        patience: Non-improvement steps before quantizing (default 50)
        cooldown: Steps before requantizing after revert (default 100)
        improvement_ratio: Minimum relative improvement (default 0.01)
    """

    def __init__(self, patience: int = 50, cooldown: int = 100, improvement_ratio: float = 0.01):
        self.patience = patience
        self.cooldown = cooldown
        self.improvement_ratio = improvement_ratio
        self._best_loss: dict[str, float] = {}
        self._stagnant_steps: dict[str, int] = {}
        self._cooldown_remaining: dict[str, int] = {}
        self._is_quantized: dict[str, bool] = {}

    def step(self, expert_id: str, block: nn.Module, loss_val: float) -> str | None:
        """Returns 'quantize', 'revert', or None."""
        if expert_id not in self._best_loss:
            self._best_loss[expert_id] = loss_val
            self._stagnant_steps[expert_id] = 0
            self._cooldown_remaining[expert_id] = 0
            self._is_quantized[expert_id] = False
            return None

        if self._cooldown_remaining[expert_id] > 0:
            self._cooldown_remaining[expert_id] -= 1
            return None

        rel_improvement = (self._best_loss[expert_id] - loss_val) / max(self._best_loss[expert_id], 1e-8)

        if rel_improvement > self.improvement_ratio:
            self._best_loss[expert_id] = loss_val
            self._stagnant_steps[expert_id] = 0
            if self._is_quantized[expert_id]:
                self._is_quantized[expert_id] = False
                self._cooldown_remaining[expert_id] = self.cooldown
                return "revert"
            return None

        self._stagnant_steps[expert_id] += 1

        if self._stagnant_steps[expert_id] >= self.patience and not self._is_quantized[expert_id]:
            self._is_quantized[expert_id] = True
            self._stagnant_steps[expert_id] = 0
            return "quantize"

        return None

    def is_quantized(self, expert_id: str) -> bool:
        return self._is_quantized.get(expert_id, False)

    def summary(self) -> dict:
        quantized = [eid for eid, q in self._is_quantized.items() if q]
        bf16 = [eid for eid, q in self._is_quantized.items() if not q]
        return {"quantized": quantized, "bf16": bf16, "total": len(self._is_quantized)}
