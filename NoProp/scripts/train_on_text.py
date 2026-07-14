"""CLI: train a mesh model on text files — turbo mode with compile + mixed precision."""
import os
import sys
import argparse
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import torch

# Max-speed matmul config
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

from train_mesh import MeshTrainer
from text_pipeline import TextMeshPipeline, TextDataset
from training_monitor import monitor


def _monitor_poll_loop(trainer_ref, interval=2.0):
    while True:
        try:
            step = getattr(trainer_ref, "step", 0)
            losses = getattr(trainer_ref, "global_losses", [])
            loss = losses[-1] if losses else None
            node_cnt = len(getattr(trainer_ref.router, "nodes", {}))
            if step > 0 and loss is not None:
                monitor.record_step(step, loss, node_cnt)
        except Exception:
            pass
        time.sleep(interval)


def _maybe_compile(model, name="model"):
    """Apply torch.compile if available (PyTorch 2.x) and Triton is available (not on Windows)."""
    if hasattr(torch, "compile") and torch.cuda.is_available() and not sys.platform.startswith("win32"):
        try:
            compiled = torch.compile(model, mode="reduce-overhead")
            print(f"  {name}: torch.compile enabled (reduce-overhead)")
            return compiled
        except Exception as e:
            print(f"  {name}: torch.compile failed ({e}), using eager")
    elif sys.platform.startswith("win32"):
        print(f"  {name}: torch.compile skipped (Windows — no Triton). Using TF32+cuDNN+autocast instead.")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--ckpt", default="checkpoints/mesh")
    parser.add_argument("--nodes", default="nodes")
    parser.add_argument("--embed-dim", type=int, default=768)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--vocab-size", type=int, default=50257)
    parser.add_argument("--canvas-len", type=int, default=512)
    parser.add_argument("--canvas-steps", type=int, default=5)
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    print("=" * 60)
    print("MESH TRAINING — TURBO MODE")
    print(f" embed_dim={args.embed_dim}  num_heads={args.num_heads}")
    print(f" canvas_len={args.canvas_len}  canvas_steps={args.canvas_steps}")
    print(f" batch={args.batch}  epochs={args.epochs}  lr={args.lr}")
    print(f" compile=enabled  tf32=enabled  cudnn_benchmark=enabled")
    print("=" * 60)

    print("Loading tokenizer...")
    pipe = TextMeshPipeline(model_name=args.tokenizer, max_length=args.canvas_len)

    print(f"Loading text files from '{args.data}'...")
    texts = pipe.load_text_files(args.data)
    if not texts:
        print(f"No text files found in '{args.data}'.")
        print("Populate it with .txt, .md, .py, .json, .csv files and try again.")
        return

    print(f"  Found {len(texts)} documents")
    chunks = pipe.chunk_texts(texts)
    print(f"  Created {chunks.size(0)} training chunks of length {args.canvas_len}")

    vocab = args.vocab_size or pipe.vocab_size
    dataset = TextDataset(chunks, embed_dim=args.embed_dim, vocab_size=vocab)

    print("Loading mesh...")
    trainer = MeshTrainer(
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        top_k=args.top_k,
        lr=args.lr,
        nodes_dir=args.nodes,
        checkpoint_dir=args.ckpt,
        vocab_size=vocab,
        use_diffusion_canvas=True,
        canvas_len=args.canvas_len,
        canvas_steps=args.canvas_steps,
    )
    trainer._load_checkpoint()
    node_cnt = len(trainer.router.nodes)
    print(f"  {node_cnt} nodes loaded")

    # Compile canvas model for max speed (move to GPU first, fallback to eager if compile fails)
    if trainer.canvas is not None and trainer.canvas.model is not None:
        if torch.cuda.is_available():
            trainer.canvas.model = trainer.canvas.model.cuda()
        try:
            trainer.canvas.model = _maybe_compile(trainer.canvas.model, "canvas_model")
        except Exception as e:
            print(f"  canvas_model: compile failed ({e}), using eager")

    poll_thread = threading.Thread(target=_monitor_poll_loop, args=(trainer,), daemon=True)
    poll_thread.start()

    print(f"Training mesh for {args.epochs} epochs...")
    trainer.train(
        dataset=dataset,
        num_epochs=args.epochs,
        batch_size=args.batch,
        log_interval=50,
        mitosis_interval=200,
        ckpt_interval=500,
    )

    if trainer.canvas is not None:
        print(f"Training canvas for {args.epochs // 2} epochs on raw tokens...")
        trainer.canvas.model.train()
        canvas_opt = torch.optim.AdamW(trainer.canvas.model.parameters(), lr=3e-4)

        # Use autocast for mixed precision
        autocast_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.amp.autocast("cuda", dtype=torch.float16) if torch.cuda.is_available() else torch.no_grad()

        canvas_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(chunks), batch_size=args.batch, shuffle=True
        )
        for epoch in range(max(1, args.epochs // 2)):
            total = 0.0
            count = 0
            for (batch_ids,) in canvas_loader:
                batch_ids = batch_ids.cuda() if torch.cuda.is_available() else batch_ids
                canvas_opt.zero_grad()
                with autocast_ctx:
                    loss = trainer.canvas.compute_loss(batch_ids, batch_ids)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainer.canvas.model.parameters(), 1.0)
                canvas_opt.step()
                total += loss.item()
                count += 1
            avg = total / max(count, 1)
            print(f"  Canvas epoch {epoch+1} — loss {avg:.6f}")
            if args.epochs > 0:
                monitor.record_step(trainer.step + epoch + 1, avg, len(trainer.router.nodes), "canvas")
        trainer.canvas.model.eval()

    trainer._save_checkpoint(final=True)
    trainer.summary()
    print("=" * 60)
    print("TRAINING COMPLETE — TURBO MODE")
    print("=" * 60)


if __name__ == "__main__":
    main()
