"""Interactive chat with a trained mesh model."""
import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from train_mesh import MeshTrainer
from text_pipeline import TextMeshPipeline, ChatInterface


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Chat with a trained mesh model")
    parser.add_argument("--checkpoint-dir", default="checkpoints/mesh",
                        help="Path to trained checkpoint directory")
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--tokenizer", default="gpt2",
                        help="HuggingFace tokenizer name")
    parser.add_argument("--canvas-len", type=int, default=64)
    parser.add_argument("--canvas-steps", type=int, default=15)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--nodes-dir", default="nodes")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    pipe = TextMeshPipeline(model_name=args.tokenizer, max_length=args.canvas_len)
    vocab = args.vocab_size or pipe.vocab_size

    trainer = MeshTrainer(
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        top_k=args.top_k,
        nodes_dir=args.nodes_dir,
        checkpoint_dir=args.checkpoint_dir,
        vocab_size=vocab,
        use_diffusion_canvas=True,
        canvas_len=args.canvas_len,
        canvas_steps=args.canvas_steps,
    )
    trainer._load_checkpoint()
    trainer.canvas.model.eval()

    chat = ChatInterface(pipe, trainer)
    print(f"Mesh chat ready ({len(trainer.router.nodes)} nodes).")
    print(f"Type your messages. '/quit' to exit, '/reset' to clear history.\n")

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user == "/quit":
            break
        if user == "/reset":
            chat.history.clear()
            print("History cleared.\n")
            continue
        if user == "/summary":
            trainer.summary()
            continue

        text = chat.reply(user, max_new_tokens=args.max_new_tokens)
        print(f"Mesh: {text}\n")


if __name__ == "__main__":
    main()
