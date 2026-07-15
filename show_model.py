import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "NoProp", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "NoProp", "scripts"))


import torch
torch.set_float32_matmul_precision("high")

from train_mesh import MeshTrainer

PROJ = os.path.join(os.path.dirname(__file__), "NoProp")

print("=" * 70)
print("LOADING TRAINED MESH MODEL")
print("=" * 70)

trainer = MeshTrainer(
    embed_dim=768,
    num_heads=8,
    top_k=3,
    lr=3e-4,
    nodes_dir=os.path.join(PROJ, "nodes"),
    checkpoint_dir=os.path.join(PROJ, "checkpoints/mesh"),
    vocab_size=50257,
    use_diffusion_canvas=True,
    canvas_len=512,
    canvas_steps=5,
)
trainer._load_checkpoint()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\nDevice: {device}")
print(f"Steps trained: {trainer.step}")
print(f"Mesh nodes: {len(trainer.router.nodes)}")
print()

# Show each node's block architecture
for node_id, node in trainer.router.nodes.items():
    block = getattr(node, "_block", None)
    if block is not None:
        print("=" * 70)
        print(f"NODE: {node_id}")
        print("=" * 70)
        print(f"Total parameters: {sum(p.numel() for p in block.parameters()):,}")
        trainable = sum(p.numel() for p in block.parameters() if p.requires_grad)
        print(f"Trainable params: {trainable:,}")
        print(f"Optimizer: {type(block.optimizer).__name__ if block.optimizer else 'None'}")
        print(f"\nArchitecture layers:")
        for name, mod in block.named_children():
            params = sum(p.numel() for p in mod.parameters())
            print(f"  +-- {name}: {type(mod).__name__} ({params:,} params)")
            for sub_name, sub_mod in mod.named_children():
                sub_params = sum(p.numel() for p in sub_mod.parameters())
                print(f"  |   +-- {sub_name}: {type(sub_mod).__name__} ({sub_params:,} params)")
        print()

# Show canvas model architecture
if trainer.canvas is not None:
    print("=" * 70)
    print("DIFFUSION CANVAS MODEL")
    print("=" * 70)
    canvas = trainer.canvas.model
    total = sum(p.numel() for p in canvas.parameters())
    print(f"Total parameters: {total:,}")
    print(f"\nArchitecture layers:")
    for name, mod in canvas.named_children():
        params = sum(p.numel() for p in mod.parameters())
        print(f"  +-- {name}: {type(mod).__name__} ({params:,} params)")
        if hasattr(mod, "named_children"):
            for sub_name, sub_mod in mod.named_children():
                sub_params = sum(p.numel() for p in sub_mod.parameters())
                print(f"  |   +-- {sub_name}: {type(sub_mod).__name__} ({sub_params:,} params)")
    print()

# Test inference
print("=" * 70)
print("TEST INFERENCE")
print("=" * 70)
try:
    sample = torch.randn(1, 768, device=device)
    with torch.no_grad():
        output, info = trainer.infer(sample)
    print(f"Input shape: {sample.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Inference info: {info}")
    print("Inference successful!")
except Exception as e:
    print(f"Inference failed: {e}")

# Full summary
print()
print("=" * 70)
print("MODEL SUMMARY")
print("=" * 70)
all_params = sum(p.numel() for p in trainer.canvas.model.parameters()) if trainer.canvas else 0
for node_id, node in trainer.router.nodes.items():
    block = getattr(node, "_block", None)
    if block:
        all_params += sum(p.numel() for p in block.parameters())
print(f"Total parameters (all nodes + canvas): {all_params:,}")
if trainer.canvas:
    print(f"  - Canvas model: {sum(p.numel() for p in trainer.canvas.model.parameters()):,}")
for node_id, node in trainer.router.nodes.items():
    block = getattr(node, "_block", None)
    if block:
        print(f"  - Node '{node_id}': {sum(p.numel() for p in block.parameters()):,}")
