import pytest
import torch
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mitosis_hook import MitosisHook
from mesh_router import MeshRouter, MeshNode
from noprop_block import NoPropBlock


@pytest.fixture
def seeded_router():
    router = MeshRouter(top_k=2)
    for i, tag in enumerate(["alpha", "beta", "gamma", "delta"]):
        node = MeshNode(
            node_id=tag,
            anchor_path=f"nodes/{tag}.md",
            anchor_embedding=torch.randn(64),
            mitosis_threshold=0.5,
        )
        router.register_node(node)
    return router


@pytest.fixture
def hook(seeded_router):
    return MitosisHook(
        router=seeded_router,
        embed_dim=64,
        num_heads=2,
        lora_rank=4,
        lora_alpha=4.0,
        oov_threshold=0.2,
        confidence_threshold=0.4,
        nodes_dir=os.path.join(tempfile.gettempdir(), "mitosis_test_nodes"),
        max_nodes=10,
    )


def test_hook_init(seeded_router):
    hook = MitosisHook(
        router=seeded_router,
        embed_dim=64,
        nodes_dir=os.path.join(tempfile.gettempdir(), "mitosis_test_init"),
    )
    assert hook.hook_calls == 0
    assert hook.spawned_count == 0
    assert hook.router is seeded_router


def test_evaluate_query_returns_similarities(hook):
    query = torch.randn(64)
    sims, nearest_id, max_sim = hook.evaluate_query(query)
    assert sims.shape == (4,)
    assert nearest_id in hook.router.node_ids
    assert -1.0 <= max_sim <= 1.0


def test_should_mitose_oov(hook):
    assert hook.should_mitose(0.05) == 'create'
    assert hook.should_mitose(0.19) == 'create'


def test_should_mitose_low_confidence(hook):
    assert hook.should_mitose(0.25) == 'create'
    assert hook.should_mitose(0.35) == 'create'


def test_should_mitose_above_threshold(hook):
    assert hook.should_mitose(0.45) == 'skip'
    assert hook.should_mitose(0.9) == 'update'


def test_should_mitose_respects_max_nodes(seeded_router):
    hook = MitosisHook(
        router=seeded_router,
        embed_dim=64,
        max_nodes=4,
        oov_threshold=0.5,
        confidence_threshold=0.5,
        nodes_dir=os.path.join(tempfile.gettempdir(), "mitosis_test_max"),
    )
    assert hook.should_mitose(0.01) == 'skip'


def test_spawn_node_creates_new_node(hook, seeded_router):
    query = torch.randn(64)
    child_id = hook.spawn_node(query, "alpha")
    assert child_id.startswith("mitosed_alpha_")
    assert child_id in seeded_router.nodes
    assert seeded_router.nodes[child_id].anchor_embedding.shape == (64,)
    assert hasattr(seeded_router.nodes[child_id], "_block")
    block = seeded_router.nodes[child_id].__dict__["_block"]
    assert isinstance(block, NoPropBlock)
    assert hook.spawned_count == 1


def test_spawn_node_creates_block_with_lora(hook, seeded_router):
    query = torch.randn(64)
    child_id = hook.spawn_node(query, "alpha")
    block = seeded_router.nodes[child_id].__dict__["_block"]
    lora_found = any(
        "lora_a" in n or "lora_b" in n for n, _ in block.named_parameters()
    )
    assert lora_found


def test_call_routes_and_returns_mitosis_info(hook, seeded_router):
    query = torch.randn(64) * 0.1
    routes, mitosed_id, action = hook(query)
    assert len(routes) > 0
    assert mitosed_id is not None
    assert mitosed_id.startswith("mitosed_")
    assert action in ('create', 'update', 'skip')
    assert hook.hook_calls == 1


def test_call_does_not_mitose_confident_query(hook):
    alpha_emb = hook.router.nodes["alpha"].anchor_embedding
    query = alpha_emb.clone()
    routes, mitosed_id, action = hook(query)
    assert len(routes) > 0
    assert action == 'update'
    assert mitosed_id is not None


def test_multiple_calls_increment_count(hook):
    for _ in range(3):
        alpha_emb = hook.router.nodes["alpha"].anchor_embedding
        hook(alpha_emb.clone())
    assert hook.hook_calls == 3


def test_state_dict_roundtrip(hook):
    alpha_emb = hook.router.nodes["alpha"].anchor_embedding
    hook(alpha_emb.clone() + torch.randn(64) * 0.5)
    state = hook.state_dict()
    assert state["spawned_count"] == hook.spawned_count
    assert state["hook_calls"] == hook.hook_calls
    assert len(state["spawned_ids"]) == hook.spawned_count

    hook2 = MitosisHook(
        router=hook.router,
        embed_dim=64,
        nodes_dir=os.path.join(tempfile.gettempdir(), "mitosis_test_state"),
    )
    hook2.load_state_dict(state)
    assert hook2.spawned_count == state["spawned_count"]
    assert hook2.hook_calls == state["hook_calls"]
    assert hook2.spawned_ids == state["spawned_ids"]


def test_spawned_node_has_mitosis_threshold_from_parent(hook, seeded_router):
    seeded_router.nodes["alpha"].mitosis_threshold = 0.75
    child_id = hook.spawn_node(torch.randn(64), "alpha")
    child_node = seeded_router.nodes[child_id]
    assert child_node.mitosis_threshold == 0.75


def test_update_node_in_anchor(hook, seeded_router):
    alpha_node = seeded_router.nodes["alpha"]
    old_anchor = alpha_node.anchor_embedding.clone()
    query = torch.randn(64)
    result_id = hook.update_node(query, alpha_node)
    assert result_id == "alpha"
    assert not torch.equal(old_anchor, alpha_node.anchor_embedding)
    assert hook.updated_count == 1
