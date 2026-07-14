# Train the Latent Mesh Diffusion Computer — Tiny config, fastest speed
# RTX 5060 8GB — optimized for tok/s and s/step

$VENV = "E:\my apps\NN\.venv\Scripts\python.exe"
$SRC = "E:\my apps\NN\NoProp\src"
$NODES = "E:\my apps\NN\NoProp\nodes"
$CKPT = "E:\my apps\NN\NoProp\checkpoints\mesh"

# Tiny model: d_model=768, n_layers=8, n_heads=12, n_kv_heads=4, d_ff=2048 (~250M params)
# Fits 8.5GB VRAM comfortably with batch_size=4, canvas_len=256

& $VENV -m train_mesh `
    --model-size tiny `
    --top-k 2 `
    --lr 1e-3 `
    --num-epochs 5 `
    --batch-size 4 `
    --num-samples 1000 `
    --canvas-len 256 `
    --canvas-steps 25 `
    --mtp-weight 0.05 `
    --num-draft-tokens 2 `
    --nodes-dir "$NODES" `
    --checkpoint-dir "$CKPT" `
    --max-steps 500 `
    --packing `
    --token-budget 2048

# Flags explained:
#   --model-size tiny     Smallest preset, fastest throughput
#   --top-k 2            Fewer experts = less VRAM per batch
#   --batch-size 4       Conservative for 8GB GPU
#   --canvas-len 256     Shorter sequences = faster per step
#   --canvas-steps 25    Fewer diffusion steps = faster
#   --mtp-weight 0.05    Lighter auxiliary loss
#   --num-draft-tokens 2 Fewer MTP heads = less compute
#   --max-steps 500      Total steps (not per epoch)
#   --packing            Token-level batching for 2-3x throughput
#   --token-budget 2048  Max tokens per packed batch
