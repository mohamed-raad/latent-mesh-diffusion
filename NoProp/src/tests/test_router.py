import pytest
import torch
import tempfile
import os
import sys

sys.path.insert(0, "NoProp/src")

from mesh_router import MeshRouter, MeshNode, load_node_metadata


def test_register_and_route():
    router = MeshRouter(top_k=2)
    node_a = MeshNode("a", "nodes/a.md", torch.randn(64))
    node_b = MeshNode("b", "nodes/b.md", torch.randn(64))
    node_c = MeshNode("c", "nodes/c.md", torch.randn(64))
    router.register_node(node_a)
    router.register_node(node_b)
    router.register_node(node_c)
    query = torch.randn(1, 64)
    results = router.route(query)
    assert len(results) == 2
    assert all(isinstance(r[2], float) for r in results)


def test_remove_node():
    router = MeshRouter()
    node = MeshNode("x", "nodes/x.md", torch.randn(64))
    router.register_node(node)
    router.remove_node("x")
    assert router.nodes == {}


def test_mitosis_creates_child(tmp_path):
    router = MeshRouter(top_k=2)
    base_embed = torch.randn(64)
    node = MeshNode(
        node_id="root",
        anchor_path=str(tmp_path / "root.md"),
        anchor_embedding=base_embed,
        mitosis_threshold=0.1,
    )
    router.register_node(node)
    for _ in range(60):
        node.update_loss(0.95)
    child_id = router.check_mitosis("root")
    assert child_id is not None
    assert child_id in router.nodes
    assert child_id != "root"


def test_load_node_metadata(tmp_path):
    file_path = tmp_path / "test_node.md"
    file_path.write_text("# Python Coding Logic\n# attention\nsome content")
    meta = load_node_metadata(str(file_path))
    assert "Python Coding Logic" in meta["tags"]
    assert "attention" in meta["tags"]
    assert meta["size"] > 0
