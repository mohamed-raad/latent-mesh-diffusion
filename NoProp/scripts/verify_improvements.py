"""Verify all improvements from new features.txt."""
import sys, os, torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_packer():
    from train_mesh import SequencePacker
    packer = SequencePacker(canvas_len=64, eos_id=1000, pad_id=0)
    seqs = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6, 7, 8]), torch.tensor([9, 10, 11])]
    domains = ["python", "python", "rust"]
    result = packer.pack(seqs, domains)
    assert result["input_ids"].size(0) == 64
    assert result["segment_ids"].size(0) == 64
    assert result["padding_mask"].size(0) == 64
    assert result["domain"] in ("python", "rust")
    print(f"  Packer: {result['n_sequences']} seqs packed, domain={result['domain']}")

    batch = [{"input_ids": [1, 2, 3], "domain": "python"},
             {"input_ids": [4, 5, 6, 7], "domain": "python"}]
    result2 = packer(batch)
    assert result2["input_ids"].size(0) == 64
    print(f"  Collate: {result2['n_sequences']} seqs packed")


def test_token_bucket():
    from train_mesh import TokenBucketIterator
    from torch.utils.data import Dataset

    class TestDataset(Dataset):
        def __init__(self):
            self.data = [
                {"input_ids": torch.randint(0, 1000, (n,)),
                 "domain": ["python", "rust", "math"][i % 3]}
                for i, n in enumerate([20, 50, 100, 30, 80, 45, 60, 15, 90, 35])
            ]

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    ds = TestDataset()
    it = TokenBucketIterator(ds, token_budget=256, canvas_len=64, eos_id=1000,
                             min_canvases=1, shuffle=False)
    batches = list(it)
    print(f"  TokenBucket: {len(batches)} batches from {len(ds)} samples")
    for i, b in enumerate(batches):
        print(f"    batch {i}: {list(b['input_ids'].shape)} domain={b['domain']}")
    assert len(batches) > 0
    assert batches[0]["input_ids"].dim() == 2


def test_merge_prune():
    from mesh_router import MeshRouter, MeshNode
    import torch.nn.functional as F

    router = MeshRouter(top_k=3, d_model=8)
    for i in range(5):
        anchor = F.normalize(torch.randn(8), dim=-1)
        router.register_node(MeshNode(node_id=f"expert_{i}", anchor_path="",
                                      anchor_embedding=anchor))
    merged = router.merge_similar(similarity_threshold=0.5)
    print(f"  Merge: {len(merged)} pairs merged, {len(router.nodes)} remain")
    pruned = router.prune_dead(max_idle_steps=0, min_loss_window=0)
    print(f"  Prune: {len(pruned)} nodes pruned, {len(router.nodes)} remain")

    from train_mesh import MeshTrainer
    torch.cuda.set_per_process_memory_fraction(0.8)
    t = MeshTrainer(model_size="tiny", vocab_size=1000, external_nodes=False,
                    nodes_dir="NoProp/nodes",
                    checkpoint_dir=os.path.expanduser("~/checkpoints/mesh_test"))
    m2, p2 = t._check_merge_prune(merge_threshold=0.5, prune_idle=0)
    print(f"  Trainer merge: {len(m2)}, prune: {len(p2)}")


def test_forward():
    from train_mesh import MeshTrainer
    t = MeshTrainer(model_size="tiny", vocab_size=1000, external_nodes=False,
                    nodes_dir="NoProp/nodes",
                    checkpoint_dir=os.path.expanduser("~/checkpoints/mesh_test2"))
    x = torch.randint(0, 1000, (1, 128))
    t_emb = torch.zeros(1, 1)
    losses = t._train_step(x, x, t_emb)
    print(f"  Forward: {len(losses)} nodes, losses={ {k: f'{v:.4f}' for k, v in losses.items()} }")


if __name__ == "__main__":
    print("=== Verification ===")
    print("\n1) SequencePacker:")
    test_packer()
    print("\n2) TokenBucketIterator:")
    test_token_bucket()
    print("\n3) Merge/Prune:")
    test_merge_prune()
    print("\n4) Forward pass:")
    test_forward()
    print("\n=== All tests passed! ===")
