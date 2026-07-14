import pytest
import torch
import sys

sys.path.insert(0, "NoProp/src")

from noprop_block import NoPropBlock
from mesh_router import MeshRouter, MeshNode


def test_variable_input_shapes():
    block = NoPropBlock(embed_dim=64, num_heads=2)
    shapes = [(1, 64), (4, 64), (16, 64), (1, 1, 64), (4, 1, 64)]
    for shape in shapes:
        x = torch.randn(*shape)
        t = torch.rand(shape[0])
        out = block(x, t)
        assert out.shape[-1] == 64, f"Failed at shape {shape}"


def test_extreme_context_window():
    block = NoPropBlock(embed_dim=64, num_heads=2)
    long_seq = torch.randn(1, 2048, 64)
    t = torch.rand(1)
    out = long_seq
    for _ in range(10):
        out = block(out[:, -64:, :], t)
    assert out.shape[-1] == 64


def test_high_entropy_noise():
    block = NoPropBlock(embed_dim=64, num_heads=2)
    B = 8
    x = torch.randn(B, 64) * 100.0
    t = torch.rand(B)
    target = torch.randn(B, 64) * 100.0
    pred = block(x, t)
    loss = block.local_loss(pred, target)
    assert torch.isfinite(loss).all(), "Loss diverged on high-entropy input"


def test_router_empty_state():
    router = MeshRouter()
    query = torch.randn(1, 64)
    results = router.route(query)
    assert results == []


def test_router_single_node():
    router = MeshRouter()
    node = MeshNode(
        node_id="test_node",
        anchor_path="nodes/test_node.md",
        anchor_embedding=torch.randn(64),
    )
    router.register_node(node)
    query = node.anchor_embedding.unsqueeze(0)
    results = router.route(query)
    assert len(results) == 1
    assert results[0][0] == "test_node"


def test_mitosis_trigger():
    node = MeshNode(
        node_id="test_node",
        anchor_path="nodes/test_node.md",
        anchor_embedding=torch.randn(64),
        mitosis_threshold=0.1,
    )
    for _ in range(60):
        node.update_loss(0.9)
    assert node.sustained_high_error()


def test_mitosis_no_trigger():
    node = MeshNode(
        node_id="test_node",
        anchor_path="nodes/test_node.md",
        anchor_embedding=torch.randn(64),
        mitosis_threshold=0.9,
    )
    for _ in range(60):
        node.update_loss(0.1)
    assert not node.sustained_high_error()
