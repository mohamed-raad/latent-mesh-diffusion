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


def test_canvas_disabled_by_default(temp_nodes_dir):
    trainer = MeshTrainer(
        embed_dim=128, num_heads=2, top_k=2,
        nodes_dir=temp_nodes_dir,
        checkpoint_dir=os.path.join(temp_nodes_dir, "..", "ckpt"),
    )
    assert trainer.canvas is None
    assert trainer.use_diffusion_canvas is False


def test_canvas_init_enabled(temp_nodes_dir):
    trainer = MeshTrainer(
        embed_dim=128, num_heads=2, top_k=2,
        nodes_dir=temp_nodes_dir,
        checkpoint_dir=os.path.join(temp_nodes_dir, "..", "ckpt"),
        use_diffusion_canvas=True,
        canvas_len=16,
        canvas_steps=5,
        canvas_entropy_threshold=0.01,
    )
    assert trainer.canvas is not None
    assert trainer.canvas_len == 16
    assert trainer.canvas_steps == 5
    assert trainer.canvas_entropy_threshold == 0.01


def test_canvas_generate_text(temp_nodes_dir):
    trainer = MeshTrainer(
        embed_dim=64, num_heads=2, top_k=2,
        nodes_dir=temp_nodes_dir,
        checkpoint_dir=os.path.join(temp_nodes_dir, "..", "ckpt"),
        use_diffusion_canvas=True,
        canvas_len=8,
        canvas_steps=5,
        vocab_size=100,
    )
    tokens = trainer.generate_text(batch_size=2, max_blocks=1)
    assert tokens.shape == (2, 8)
    assert tokens.dtype == torch.long
    assert (tokens >= 0).all() and (tokens < 100).all()


def test_canvas_generate_text_multiblock(temp_nodes_dir):
    trainer = MeshTrainer(
        embed_dim=64, num_heads=2, top_k=2,
        nodes_dir=temp_nodes_dir,
        checkpoint_dir=os.path.join(temp_nodes_dir, "..", "ckpt"),
        use_diffusion_canvas=True,
        canvas_len=8,
        canvas_steps=3,
        vocab_size=100,
    )
    tokens = trainer.generate_text(batch_size=2, max_blocks=3)
    assert tokens.shape == (2, 24)
    assert tokens.dtype == torch.long


def test_canvas_no_canvas_error(temp_nodes_dir):
    trainer = MeshTrainer(
        embed_dim=128, num_heads=2, top_k=2,
        nodes_dir=temp_nodes_dir,
        checkpoint_dir=os.path.join(temp_nodes_dir, "..", "ckpt"),
    )
    with pytest.raises(RuntimeError, match="DiffusionCanvas is not initialized"):
        trainer.generate_text(batch_size=2)


def test_canvas_infer_unchanged_with_canvas_enabled(temp_nodes_dir):
    trainer = MeshTrainer(
        embed_dim=128, num_heads=2, top_k=2,
        nodes_dir=temp_nodes_dir,
        checkpoint_dir=os.path.join(temp_nodes_dir, "..", "ckpt"),
        use_diffusion_canvas=True,
        canvas_len=8,
        canvas_steps=3,
    )
    x = torch.randn(1, 128)
    output, info = trainer.infer(x)
    assert output.shape == (1, 128)
    assert "draft_tokens" in info
    assert "confidence" in info
    assert "active_nodes" in info
    assert torch.isfinite(output).all()


def test_canvas_summary_reports_canvas(temp_nodes_dir, capsys):
    trainer = MeshTrainer(
        embed_dim=64, num_heads=2, top_k=2,
        nodes_dir=temp_nodes_dir,
        checkpoint_dir=os.path.join(temp_nodes_dir, "..", "ckpt"),
        use_diffusion_canvas=True,
        canvas_len=16,
        canvas_steps=5,
    )
    trainer.summary()
    captured = capsys.readouterr()
    assert "DiffusionCanvas" in captured.out
    assert "16-token canvas" in captured.out
    assert "5 steps" in captured.out


def test_canvas_checkpoint_save_load(temp_nodes_dir):
    ckpt_dir = os.path.join(temp_nodes_dir, "..", "ckpt_canvas")
    trainer = MeshTrainer(
        embed_dim=64, num_heads=2, top_k=2,
        nodes_dir=temp_nodes_dir,
        checkpoint_dir=ckpt_dir,
        use_diffusion_canvas=True,
        canvas_len=8,
        canvas_steps=3,
        vocab_size=100,
    )
    # get a reference output before save
    ref = trainer.generate_text(batch_size=1, max_blocks=1)
    # save
    trainer._save_checkpoint(final=True)
    assert os.path.exists(os.path.join(ckpt_dir, "step_final.pt"))

    # new trainer loads the checkpoint
    trainer2 = MeshTrainer(
        embed_dim=64, num_heads=2, top_k=2,
        nodes_dir=temp_nodes_dir,
        checkpoint_dir=ckpt_dir,
        use_diffusion_canvas=True,
        canvas_len=8,
        canvas_steps=3,
        vocab_size=100,
    )
    trainer2._load_checkpoint()
    post = trainer2.generate_text(batch_size=1, max_blocks=1)
    assert post.shape == ref.shape
    assert post.dtype == ref.dtype
    assert (post >= 0).all() and (post < 100).all()
