"""CLI: create a new mesh model from an Obsidian vault."""
import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from create_obsidian_vault import create_vault
from obsidian_mesh_compiler import ObsidianMeshCompiler
from train_mesh import MeshTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", default="vault")
    parser.add_argument("--ckpt", default="checkpoints/mesh")
    parser.add_argument("--nodes", default="nodes")
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--vocab-size", type=int, default=50257)
    parser.add_argument("--canvas-len", type=int, default=64)
    parser.add_argument("--canvas-steps", type=int, default=15)
    args = parser.parse_args()

    os.makedirs(args.ckpt, exist_ok=True)

    if not os.path.isdir(args.vault) or not os.listdir(args.vault):
        print(f"No vault at {args.vault} — creating synthetic vault")
        create_vault(args.vault, seed=42)

    print("Creating mesh trainer...")
    trainer = MeshTrainer(
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        top_k=args.top_k,
        lr=args.lr,
        nodes_dir=args.nodes,
        checkpoint_dir=args.ckpt,
        vocab_size=args.vocab_size,
        use_diffusion_canvas=True,
        canvas_len=args.canvas_len,
        canvas_steps=args.canvas_steps,
    )

    print(f"Compiling vault '{args.vault}' into mesh...")
    compiler = ObsidianMeshCompiler(args.vault, embed_dim=args.embed_dim, max_vocab=1024)
    result = compiler.compile(trainer.router, nodes_dir=args.nodes)
    print(f"  Nodes: {result['n_nodes']}, Edges: {result['n_edges']}")
    trainer.summary()


if __name__ == "__main__":
    main()
