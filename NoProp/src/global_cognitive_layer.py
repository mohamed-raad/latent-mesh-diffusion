"""
Global Cognitive Layer (GCL) — AF.md #5.
Cross-expert reasoning, consensus, verification, tool orchestration.
"""
from dataclasses import dataclass, field
from typing import Any

import torch.nn as nn
import torch.nn.functional as F

import torch

# ═══════════════════════════════════════════════════
# Cross-Attention Consensus (AF.md #5)
# ═══════════════════════════════════════════════════

class CrossExpertAttention(nn.Module):
    """Each expert attends to all others via multi-head cross-attention."""
    def __init__(self, d_model: int, n_heads: int = 8):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, N, D = x.shape  # N = number of experts
        Q = self.q_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(1) == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ V).transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out)


# ═══════════════════════════════════════════════════
# Consensus Mechanism
# ═══════════════════════════════════════════════════

@dataclass
class ConsensusResult:
    agreed: bool
    confidence: float
    majority: int
    total: int
    disagreements: list[tuple[int, float]] = field(default_factory=list)


class ConsensusMechanism:
    """Weighted voting with confidence threshold."""
    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold

    def vote(self, expert_outputs: list[torch.Tensor],
             expert_confidences: list[float]) -> ConsensusResult:
        n = len(expert_outputs)
        if n == 0:
            return ConsensusResult(agreed=False, confidence=0.0, majority=0, total=0)

        stacked = torch.stack(expert_outputs)
        mean_out = stacked.mean(dim=0)
        votes_for = 0
        disagreements = []
        for i, out in enumerate(expert_outputs):
            sim = F.cosine_similarity(out.flatten().unsqueeze(0),
                                      mean_out.flatten().unsqueeze(0)).item()
            if sim >= self.threshold:
                votes_for += 1
            else:
                disagreements.append((i, sim))

        majority = votes_for / n
        avg_confidence = sum(expert_confidences) / n
        agreed = majority > 0.5
        return ConsensusResult(
            agreed=agreed,
            confidence=avg_confidence * majority,
            majority=votes_for,
            total=n,
            disagreements=disagreements,
        )


# ═══════════════════════════════════════════════════
# Verification Module
# ═══════════════════════════════════════════════════

class VerificationModule(nn.Module):
    """Checks consistency & correctness of expert outputs."""
    def __init__(self, d_model: int, hidden: int = 128):
        super().__init__()
        self.consistency_check = nn.Sequential(
            nn.Linear(d_model * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, expert_output: torch.Tensor,
                reference: torch.Tensor) -> torch.Tensor:
        concat = torch.cat([expert_output, reference], dim=-1)
        return self.consistency_check(concat)


# ═══════════════════════════════════════════════════
# Tool Manager
# ═══════════════════════════════════════════════════

@dataclass
class Tool:
    name: str
    execute: callable
    description: str = ""
    required_confidence: float = 0.7


class ToolManager:
    """Registers and invokes external tools (calculator, search, code runner)."""
    def __init__(self):
        self.tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        self.tools[tool.name] = tool

    def call(self, name: str, *args, **kwargs) -> Any:
        tool = self.tools.get(name)
        if tool is None:
            raise ValueError(f"Tool '{name}' not registered")
        return tool.execute(*args, **kwargs)


# ═══════════════════════════════════════════════════
# Global Cognitive Layer
# ═══════════════════════════════════════════════════

class GlobalCognitiveLayer(nn.Module):
    """Coordinates cross-expert reasoning, consensus, verification."""
    def __init__(self, d_model: int, n_heads: int = 8, max_experts: int = 16):
        super().__init__()
        self.cross_attention = CrossExpertAttention(d_model, n_heads)
        self.consensus = ConsensusMechanism()
        self.verifier = VerificationModule(d_model)
        self.tool_manager = ToolManager()

        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

        self.expert_proj = nn.Linear(d_model, d_model)

    def forward(self, expert_outputs: list[torch.Tensor],
                confidences: list[float] | None = None,
                return_consensus: bool = True) -> tuple[torch.Tensor, dict]:
        N = len(expert_outputs)
        if N == 0:
            return torch.zeros(1, 1), {"consensus": None, "gcl_output": None}

        stacked = torch.stack(expert_outputs, dim=1)
        B, E, D = stacked.shape

        attended = self.cross_attention(stacked)
        gcl_out = attended.mean(dim=1)

        gate_weights = self.gate(gcl_out)
        weighted_sum = (attended * gate_weights.unsqueeze(-1)).sum(dim=1)

        result = {
            "gcl_output": gcl_out,
            "weighted_sum": weighted_sum,
            "gate_weights": gate_weights.detach(),
        }

        if return_consensus and confidences is not None:
            con_result = self.consensus.vote(
                [o.mean(dim=0) for o in expert_outputs],
                confidences,
            )
            result["consensus"] = con_result

        return weighted_sum, result
