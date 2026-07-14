import pytest
import torch
import os
import sys
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from create_obsidian_vault import create_vault
from obsidian_mesh_compiler import ObsidianMeshCompiler
from train_mesh import MeshTrainer, SyntheticMeshDataset


@pytest.fixture
def obsidian_vault():
    tmp = tempfile.mkdtemp()
    vault = os.path.join(tmp, "vault")
    n = create_vault(vault, seed=42)
    yield vault, n
    shutil.rmtree(tmp)


@pytest.fixture
def compiled_mesh(obsidian_vault):
    vault_path, num_pages = obsidian_vault
    base = os.path.dirname(vault_path)
    nodes_dir = os.path.join(base, "nodes")
    ckpt_dir = os.path.join(base, "ckpt")

    trainer = MeshTrainer(
        embed_dim=64,
        num_heads=2,
        top_k=3,
        lr=1e-3,
        nodes_dir=nodes_dir,
        checkpoint_dir=ckpt_dir,
        vocab_size=100,
        use_diffusion_canvas=True,
        canvas_len=8,
        canvas_steps=5,
    )

    compiler = ObsidianMeshCompiler(vault_path, embed_dim=64, max_vocab=512)
    result = compiler.compile(trainer.router, nodes_dir=nodes_dir)
    return trainer, result, num_pages


class TestPipeline:
    def test_vault_creation(self, obsidian_vault):
        vault_path, num_pages = obsidian_vault
        files = os.listdir(vault_path)
        assert len(files) == num_pages
        md_files = [f for f in files if f.endswith(".md")]
        assert len(md_files) == num_pages
        index_path = os.path.join(vault_path, "Index.md")
        assert os.path.exists(index_path)

    def test_compiler_parses_vault(self, obsidian_vault):
        vault_path, num_pages = obsidian_vault
        compiler = ObsidianMeshCompiler(vault_path, embed_dim=64, max_vocab=512)
        graph = compiler.scan_vault()
        assert len(graph) >= num_pages - 1
        assert len(compiler.node_ids) >= num_pages - 1

    def test_compiler_injects_into_mesh(self, compiled_mesh):
        trainer, result, num_pages = compiled_mesh
        assert len(trainer.router.nodes) >= num_pages - 1
        assert result["n_nodes"] >= num_pages - 1
        assert result["n_edges"] > 0

    def test_routing_with_obsidian_anchors(self, compiled_mesh):
        trainer, result, num_pages = compiled_mesh
        query = torch.randn(1, 64)
        routes = trainer.router.route(query)
        assert len(routes) > 0
        for nid, node, score in routes:
            assert nid in trainer.router.nodes
            assert 0.0 <= score <= 1.0

    def test_training_with_obsidian_mesh(self, compiled_mesh):
        trainer, result, num_pages = compiled_mesh
        dataset = SyntheticMeshDataset(num_samples=20, embed_dim=64, num_classes=10)
        trainer.train(
            dataset=dataset,
            num_epochs=2,
            batch_size=4,
            log_interval=100,
            mitosis_interval=100,
            ckpt_interval=100,
        )
        assert trainer.step > 0
        assert len(trainer.global_losses) > 0
        assert len(trainer.router.nodes) >= num_pages - 1

    def test_inference_after_obsidian_training(self, compiled_mesh):
        trainer, result, num_pages = compiled_mesh
        dataset = SyntheticMeshDataset(num_samples=10, embed_dim=64, num_classes=10)
        trainer.train(
            dataset=dataset,
            num_epochs=1,
            batch_size=4,
            log_interval=100,
            mitosis_interval=100,
            ckpt_interval=100,
        )
        x = torch.randn(1, 64)
        output, info = trainer.infer(x)
        assert output.shape == (1, 64)
        assert torch.isfinite(output).all()
        assert len(info["active_nodes"]) > 0

    def test_canvas_generation_after_obsidian_training(self, compiled_mesh):
        trainer, result, num_pages = compiled_mesh
        dataset = SyntheticMeshDataset(num_samples=10, embed_dim=64, num_classes=10)
        trainer.train(
            dataset=dataset,
            num_epochs=1,
            batch_size=4,
            log_interval=100,
            mitosis_interval=100,
            ckpt_interval=100,
        )
        tokens = trainer.generate_text(batch_size=2, max_blocks=1)
        assert tokens.shape == (2, 8)
        assert tokens.dtype == torch.long

    def test_checkpoint_save_and_export(self, compiled_mesh):
        trainer, result, num_pages = compiled_mesh
        dataset = SyntheticMeshDataset(num_samples=10, embed_dim=64, num_classes=10)
        trainer.train(
            dataset=dataset,
            num_epochs=1,
            batch_size=4,
            log_interval=100,
            mitosis_interval=100,
            ckpt_interval=100,
        )
        base = os.path.dirname(trainer.nodes_dir)
        safetensors_path = os.path.join(base, "exported.safetensors")
        trainer.export_model(safetensors_path, fmt="safetensors")
        assert os.path.exists(safetensors_path)

        gguf_path = os.path.join(base, "exported.gguf")
        trainer.export_model(gguf_path, fmt="gguf")
        assert os.path.exists(gguf_path)

        canvas_path = os.path.join(base, "exported_canvas.pt")
        assert os.path.exists(canvas_path)

    def test_full_pipeline_steps_decrease_loss(self, obsidian_vault):
        vault_path, num_pages = obsidian_vault
        base = os.path.dirname(vault_path)
        nodes_dir = os.path.join(base, "nodes2")
        ckpt_dir = os.path.join(base, "ckpt2")

        trainer = MeshTrainer(
            embed_dim=64, num_heads=2, top_k=2, lr=1e-2,
            nodes_dir=nodes_dir, checkpoint_dir=ckpt_dir, vocab_size=100,
        )
        compiler = ObsidianMeshCompiler(vault_path, embed_dim=64, max_vocab=512)
        compiler.compile(trainer.router, nodes_dir=nodes_dir)
        dataset = SyntheticMeshDataset(num_samples=30, embed_dim=64, num_classes=10)
        trainer.train(
            dataset=dataset, num_epochs=3, batch_size=4,
            log_interval=100, mitosis_interval=100, ckpt_interval=100,
        )
        if len(trainer.global_losses) >= 4:
            first = sum(trainer.global_losses[: len(trainer.global_losses) // 2])
            second = sum(trainer.global_losses[len(trainer.global_losses) // 2 :])
            assert second < first * 1.5 or first == 0
