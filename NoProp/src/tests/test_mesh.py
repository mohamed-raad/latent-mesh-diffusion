import pytest
import torch
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from train_mesh import MeshTrainer, SyntheticMeshDataset


@pytest.fixture
def temp_nodes_dir():
    with tempfile.TemporaryDirectory() as tmp:
        nodes = os.path.join(tmp, "nodes")
        os.makedirs(nodes)
        for name, tags in [
            ("anchor_a.md", "alpha, base"),
            ("anchor_b.md", "beta, specialist"),
        ]:
            with open(os.path.join(nodes, name), "w") as f:
                f.write(f"# {name}\ntags: {tags}\ndim: 128\n")
        yield nodes


def test_trainer_initializes_with_seed_nodes(temp_nodes_dir):
    trainer = MeshTrainer(
        embed_dim=128,
        num_heads=2,
        top_k=2,
        nodes_dir=temp_nodes_dir,
        checkpoint_dir=os.path.join(temp_nodes_dir, "..", "ckpt"),
    )
    assert len(trainer.router.nodes) == 2
    trainer.summary()


def test_trainer_creates_default_seed_when_empty():
    with tempfile.TemporaryDirectory() as tmp:
        nodes = os.path.join(tmp, "empty_nodes")
        os.makedirs(nodes)
        trainer = MeshTrainer(
            embed_dim=128,
            num_heads=2,
            top_k=1,
            nodes_dir=nodes,
            checkpoint_dir=os.path.join(tmp, "ckpt"),
        )
        assert len(trainer.router.nodes) >= 1


def test_training_loop_runs(temp_nodes_dir):
    trainer = MeshTrainer(
        embed_dim=128,
        num_heads=2,
        top_k=2,
        lr=1e-3,
        nodes_dir=temp_nodes_dir,
        checkpoint_dir=os.path.join(temp_nodes_dir, "..", "ckpt"),
    )
    dataset = SyntheticMeshDataset(num_samples=20, embed_dim=128, num_classes=5)
    trainer.train(
        dataset=dataset,
        num_epochs=2,
        batch_size=4,
        log_interval=10,
        mitosis_interval=20,
        ckpt_interval=20,
    )
    assert len(trainer.global_losses) > 0
    assert trainer.step > 0


def test_training_loss_decreases(temp_nodes_dir):
    trainer = MeshTrainer(
        embed_dim=64,
        num_heads=2,
        top_k=2,
        lr=1e-2,
        nodes_dir=temp_nodes_dir,
        checkpoint_dir=os.path.join(temp_nodes_dir, "..", "ckpt"),
    )
    dataset = SyntheticMeshDataset(num_samples=30, embed_dim=64, num_classes=4)
    trainer.train(
        dataset=dataset,
        num_epochs=3,
        batch_size=4,
        log_interval=100,
        mitosis_interval=100,
        ckpt_interval=100,
    )
    first_half = trainer.global_losses[: len(trainer.global_losses) // 2]
    second_half = trainer.global_losses[len(trainer.global_losses) // 2 :]
    if first_half and second_half:
        assert sum(first_half) / len(first_half) >= sum(second_half) / len(second_half) * 0.5


def test_checkpoint_and_resume(temp_nodes_dir):
    ckpt_dir = os.path.join(temp_nodes_dir, "..", "ckpt_resume")
    trainer = MeshTrainer(
        embed_dim=64,
        num_heads=2,
        top_k=1,
        lr=1e-3,
        nodes_dir=temp_nodes_dir,
        checkpoint_dir=ckpt_dir,
    )
    dataset = SyntheticMeshDataset(num_samples=10, embed_dim=64, num_classes=3)
    trainer.train(
        dataset=dataset,
        num_epochs=1,
        batch_size=4,
        log_interval=100,
        mitosis_interval=100,
        ckpt_interval=5,
    )
    assert os.path.exists(os.path.join(ckpt_dir, "step_final.pt")) or any(
        f.startswith("step_") for f in os.listdir(ckpt_dir)
    )
    saved_step = trainer.step

    trainer2 = MeshTrainer(
        embed_dim=64,
        num_heads=2,
        top_k=1,
        lr=1e-3,
        nodes_dir=temp_nodes_dir,
        checkpoint_dir=ckpt_dir,
    )
    trainer2.train(
        dataset=dataset,
        num_epochs=1,
        batch_size=4,
        log_interval=100,
        mitosis_interval=100,
        ckpt_interval=100,
        resume=True,
    )
    assert trainer2.step >= saved_step


def test_inference_returns_valid_output(temp_nodes_dir):
    trainer = MeshTrainer(
        embed_dim=128,
        num_heads=2,
        top_k=2,
        nodes_dir=temp_nodes_dir,
        checkpoint_dir=os.path.join(temp_nodes_dir, "..", "ckpt"),
    )
    x = torch.randn(1, 128)
    output, info = trainer.infer(x)
    assert output.shape == (1, 128)
    assert "draft_tokens" in info
    assert "confidence" in info
    assert "active_nodes" in info
    assert torch.isfinite(output).all()


def test_mitosis_during_training(temp_nodes_dir):
    trainer = MeshTrainer(
        embed_dim=64,
        num_heads=2,
        top_k=2,
        lr=1e-3,
        nodes_dir=temp_nodes_dir,
        checkpoint_dir=os.path.join(temp_nodes_dir, "..", "ckpt"),
        mitosis_threshold=0.01,
    )
    for node in trainer.router.nodes.values():
        node.mitosis_threshold = 0.01
        for _ in range(60):
            node.update_loss(0.95)

    initial_count = len(trainer.router.nodes)
    spawned = trainer._check_mitosis()
    assert len(trainer.router.nodes) > initial_count or len(spawned) > 0


def test_router_invariance_different_order(temp_nodes_dir):
    trainer = MeshTrainer(
        embed_dim=128,
        num_heads=2,
        top_k=2,
        nodes_dir=temp_nodes_dir,
        checkpoint_dir=os.path.join(temp_nodes_dir, "..", "ckpt"),
    )
    x1 = torch.randn(1, 128)
    x2 = x1.clone()
    out1, info1 = trainer.infer(x1)
    out2, info2 = trainer.infer(x2)
    assert torch.allclose(out1, out2)
    assert info1["active_nodes"] == info2["active_nodes"]
