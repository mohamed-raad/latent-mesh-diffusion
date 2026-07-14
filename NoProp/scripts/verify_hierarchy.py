"""Quick smoke-test: SequencePacker + expert hierarchy loading."""
import sys, os, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from train_mesh import SequencePacker, MeshTrainer

# 1) SequencePacker
packer = SequencePacker(canvas_len=128, eos_id=151643, pad_id=0)
seqs = [torch.tensor([101, 102, 103]), torch.tensor([201, 202, 203, 204]), torch.tensor([301, 302])]
result = packer.pack(seqs)
assert result["input_ids"].size(0) == 128
assert result["segment_ids"].size(0) == 128
n_pad = (~result["padding_mask"]).sum().item()
print(f"SequencePacker OK  (input_ids[:20]={result['input_ids'].tolist()[:20]}  pad_tokens={n_pad})")

# 2) Expert hierarchy
torch.cuda.set_per_process_memory_fraction(0.85)
t = MeshTrainer(model_size="tiny", vocab_size=151643, external_nodes=False,
                nodes_dir="NoProp/nodes",
                checkpoint_dir=os.path.expanduser("~/checkpoints/mesh_test"))
nodes = t.router.nodes
print(f"\nLoaded {len(nodes)} expert nodes:")
for nid, node in sorted(nodes.items()):
    print(f"  {nid:30s}  domain={node.metadata.domain}  tags={[nid]}")

# 3) Quick forward
print("\nQuick forward pass...")
x = torch.randint(0, 151643, (1, 128))
t_emb = torch.zeros(1, 1)
losses = t._train_step(x, x, t_emb)
print(f"  Nodes activated: {list(losses.keys())}")
print(f"  Losses: { {k: f'{v:.4f}' for k, v in losses.items()} }")
print("\nAll checks passed!")
