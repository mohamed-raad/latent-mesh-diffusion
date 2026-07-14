import pytest
import torch
import sys

sys.path.insert(0, "NoProp/src")

from noprop_block import NoPropBlock


def test_gradient_isolation():
    block = NoPropBlock(embed_dim=128, num_heads=2)
    block.configure_optimizer(lr=1e-3)

    B = 4
    x = torch.randn(B, 128)
    t = torch.rand(B)
    target = torch.randn(B, 128)

    pred = block(x, t)
    loss = block.local_loss(pred, target)
    loss.backward()

    frozen_grads = 0
    active_grads = 0
    for name, param in block.named_parameters():
        if any(frozen_prefix in name for frozen_prefix in ["input_proj"]):
            frozen_grads += 1
        else:
            if param.grad is not None:
                active_grads += 1

    assert active_grads > 0, "No active parameters received gradients — training cannot proceed"


def test_no_gradient_leak_to_frozen_params():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    block = NoPropBlock(embed_dim=128, num_heads=2).to(device)

    for param in block.input_proj.parameters():
        param.requires_grad = False

    block.configure_optimizer(lr=1e-3)

    B = 4
    x = torch.randn(B, 128, device=device)
    t = torch.rand(B, device=device)
    target = torch.randn(B, 128, device=device)

    pred = block(x, t)
    loss = block.local_loss(pred, target)
    loss.backward()

    for name, param in block.input_proj.named_parameters():
        assert param.grad is None or param.grad.abs().sum().item() == 0.0, (
            f"Gradient leaked to frozen param: {name}"
        )

    any_active_grad = False
    for name, param in block.named_parameters():
        if "input_proj" not in name and param.requires_grad and param.grad is not None:
            if param.grad.abs().sum().item() > 0:
                any_active_grad = True
                break
    assert any_active_grad, "No active param received a non-zero gradient"
