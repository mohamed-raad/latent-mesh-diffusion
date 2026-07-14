"""End-to-end demonstration: create vault → compile mesh → train → generate → export."""
import os
import sys
import tempfile
import shutil
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from create_obsidian_vault import create_vault
from obsidian_mesh_compiler import ObsidianMeshCompiler
from train_mesh import MeshTrainer, SyntheticMeshDataset


def main():
    tmp = tempfile.mkdtemp()
    print("=" * 60)
    print("Step 1: Create synthetic Obsidian vault")
    print("=" * 60)
    vault = os.path.join(tmp, "vault")
    n = create_vault(vault, seed=42)
    print(f"Created {n} markdown pages:")
    for f in sorted(os.listdir(vault)):
        print(f"  {f}")

    print()
    print("=" * 60)
    print("Step 2: Initialize MeshTrainer with diffusion canvas")
    print("=" * 60)
    nodes_dir = os.path.join(tmp, "nodes")
    ckpt_dir = os.path.join(tmp, "ckpt")
    export_dir = os.path.join(tmp, "export")

    torch.set_float32_matmul_precision("high")

    trainer = MeshTrainer(
        embed_dim=128,
        num_heads=4,
        top_k=3,
        lr=1e-3,
        nodes_dir=nodes_dir,
        checkpoint_dir=ckpt_dir,
        vocab_size=1000,
        use_diffusion_canvas=True,
        canvas_len=16,
        canvas_steps=10,
    )
    print(f"Seed nodes: {len(trainer.router.nodes)}")

    print()
    print("=" * 60)
    print("Step 3: Compile Obsidian vault into mesh")
    print("=" * 60)
    compiler = ObsidianMeshCompiler(vault, embed_dim=128, max_vocab=1024)
    result = compiler.compile(trainer.router, nodes_dir=nodes_dir)
    print(f"Compiled {result['n_nodes']} nodes with {result['n_edges']} wiki-link edges")

    print()
    print("=" * 60)
    print("Step 4: Train the mesh (local NoProp blocks)")
    print("=" * 60)
    dataset = SyntheticMeshDataset(num_samples=100, embed_dim=128, num_classes=20)
    trainer.train(
        dataset=dataset,
        num_epochs=5,
        batch_size=8,
        log_interval=50,
        mitosis_interval=100,
        ckpt_interval=200,
    )
    print(f"Done. {len(trainer.router.nodes)} nodes, {trainer.step} steps")
    if trainer.global_losses:
        print(f"Loss range: {min(trainer.global_losses):.6f} – {max(trainer.global_losses):.6f}")

    print()
    print("=" * 60)
    print("Step 5: Inference (mesh forward + speculative decoding)")
    print("=" * 60)
    x = torch.randn(1, 128)
    output, info = trainer.infer(x)
    print(f"Output shape: {output.shape}")
    print(f"Draft tokens shape: {info['draft_tokens'].shape}")
    print(f"Active nodes: {[nid for nid, _ in info['active_nodes']]}")
    print(f"Confidence: {info['confidence'][0].tolist()}")

    print()
    print("=" * 60)
    print("Step 6: Diffusion canvas generation")
    print("=" * 60)
    tokens = trainer.generate_text(batch_size=2, max_blocks=2)
    print(f"Generated token IDs shape: {tokens.shape}")
    print(f"Batch 0: {tokens[0].tolist()}")
    print(f"Batch 1: {tokens[1].tolist()}")
    print(f"Vocabulary range: min={tokens.min().item()}, max={tokens.max().item()}")

    print()
    print("=" * 60)
    print("Step 7: Export model (3 formats)")
    print("=" * 60)
    os.makedirs(export_dir, exist_ok=True)
    trainer.export_model(os.path.join(export_dir, "mesh.safetensors"), fmt="safetensors")
    trainer.export_model(os.path.join(export_dir, "mesh.gguf"), fmt="gguf")
    trainer.summary()

    print()
    print(f"All artifacts in: {tmp}")
    shutil.rmtree(tmp)
    print("Done.")


if __name__ == "__main__":
    main()
