"""CLI: generate text from a trained mesh given a prompt."""
import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from train_mesh import MeshTrainer
from text_pipeline import TextMeshPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--ckpt", default="checkpoints/mesh")
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--vocab-size", type=int, default=50257)
    parser.add_argument("--canvas-len", type=int, default=64)
    parser.add_argument("--canvas-steps", type=int, default=15)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--tokenizer", default="gpt2")
    args = parser.parse_args()

    pipe = TextMeshPipeline(model_name=args.tokenizer, max_length=args.canvas_len)
    vocab = args.vocab_size or pipe.vocab_size

    trainer = MeshTrainer(
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        top_k=args.top_k,
        nodes_dir="nodes",
        checkpoint_dir=args.ckpt,
        vocab_size=vocab,
        use_diffusion_canvas=True,
        canvas_len=args.canvas_len,
        canvas_steps=args.canvas_steps,
    )
    trainer._load_checkpoint()

    prompt_ids = pipe.encode(args.prompt)
    out = trainer.chat(prompt_ids, max_new_tokens=args.max_new_tokens)

    prompt_len = prompt_ids.size(1)
    generated = out[:, prompt_len:]
    generated = generated[generated != pipe.tokenizer.pad_token_id]
    text = pipe.tokenizer.decode(generated.tolist(), skip_special_tokens=True)

    print(f"\nPrompt: {args.prompt}")
    print(f"Generated: {text}\n")


if __name__ == "__main__":
    main()
