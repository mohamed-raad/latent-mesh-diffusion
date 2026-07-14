import pytest
import torch
import sys

sys.path.insert(0, "NoProp/src")

from noprop_block import NoPropBlock, SinusoidalTimeEmbedding


def test_block_forward():
    block = NoPropBlock(embed_dim=64, num_heads=2)
    B = 4
    x = torch.randn(B, 64)
    t = torch.rand(B)
    out = block(x, t)
    assert out.shape == (B, 64)
    assert out.dtype == torch.bfloat16 or out.dtype == torch.float32


def test_block_local_step():
    block = NoPropBlock(embed_dim=64, num_heads=2)
    B = 4
    x = torch.randn(B, 64)
    t = torch.rand(B)
    target = torch.randn(B, 64)
    pred = block(x, t)
    loss_val = block.local_step(pred, target)
    assert isinstance(loss_val, float)
    assert loss_val >= 0.0


def test_sinusoidal_time_embedding():
    emb = SinusoidalTimeEmbedding(dim=64)
    t = torch.tensor([0.0, 0.5, 1.0])
    out = emb(t)
    assert out.shape == (3, 64)
    assert torch.isfinite(out).all()


def test_checkpoint_atomic(tmp_path):
    import os as _os
    from noprop_block import checkpoint_atomic, load_checkpoint
    save_dir = _os.path.join(str(tmp_path), "ckpt")
    model_state = {"weight": torch.randn(3, 3)}
    opt_state = {"lr": 1e-3}
    metadata = {"step": 42}
    checkpoint_atomic(save_dir, 42, model_state, opt_state, metadata)
    ckpt_path = _os.path.join(save_dir, "step_42.pt")
    assert _os.path.exists(ckpt_path)
    loaded = load_checkpoint(ckpt_path)
    assert loaded["metadata"]["step"] == 42
