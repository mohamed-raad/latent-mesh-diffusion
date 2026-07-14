"""
Dynamic Thinking Budget — router predicts difficulty, allocates variable experts.
GLM-5 style: easy queries use fewer experts, hard queries use more.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DifficultyPredictor(nn.Module):
    """Lightweight MLP that predicts query difficulty from embeddings."""

    def __init__(self, d_model: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 3)  # easy, medium, hard
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.mean(dim=1) if x.dim() == 3 else x
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = F.gelu(self.fc2(x))
        x = self.dropout(x)
        return F.softmax(self.fc3(x), dim=-1)


class DynamicExpertBudget:
    """Allocates experts based on predicted difficulty."""

    def __init__(
        self,
        min_experts: int = 2,
        max_experts: int = 64,
        easy_experts: int = 4,
        medium_experts: int = 16,
        hard_experts: int = 64,
    ):
        self.min_experts = min_experts
        self.max_experts = max_experts
        self.budget_map = {
            0: easy_experts,    # easy
            1: medium_experts,  # medium
            2: hard_experts,    # hard
        }
        self.difficulty_predictor: DifficultyPredictor | None = None

    def build_predictor(self, d_model: int, hidden_dim: int = 128):
        self.difficulty_predictor = DifficultyPredictor(d_model, hidden_dim)

    def predict_difficulty(self, embedding: torch.Tensor) -> torch.Tensor:
        """Returns difficulty distribution [batch, 3]."""
        if self.difficulty_predictor is None:
            return torch.tensor([[0.0, 0.0, 1.0]] * embedding.size(0))  # default: medium
        return self.difficulty_predictor(embedding)

    def get_expert_budget(self, difficulty_probs: torch.Tensor) -> torch.Tensor:
        """Returns number of experts to use per sample."""
        difficulty_classes = difficulty_probs.argmax(dim=-1)
        budgets = torch.tensor([self.budget_map[d.item()] for d in difficulty_classes])
        return budgets.clamp(self.min_experts, self.max_experts)

    def compute_budget_loss(
        self,
        difficulty_probs: torch.Tensor,
        actual_loss: torch.Tensor,
        budget_penalty: float = 0.01,
    ) -> torch.Tensor:
        """Penalize using too many experts for easy queries."""
        difficulty = difficulty_probs.argmax(dim=-1)
        budgets = self.get_expert_budget(difficulty_probs)

        penalty = budgets.float().mean() * budget_penalty * actual_loss.mean()
        return penalty
