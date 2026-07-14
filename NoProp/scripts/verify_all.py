"""Verify all 4 optimizations + telemetry."""
import sys, os, torch, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

def test_adaptive_length_buckets():
    from train_mesh import AdaptiveLengthBuckets
    b = AdaptiveLengthBuckets(canvas_len=128, n_buckets=4)
    assert len(b.boundaries) == 3
    for _ in range(600):
        b.observe(torch.randint(10, 100, (1,)).item())
    assert len(b.boundaries) == 3
    assert all(1 <= x <= 128 for x in b.boundaries)
    print(f"  AdaptiveLengthBuckets OK: boundaries={b.boundaries}")

def test_domain_queue():
    from train_mesh import DomainQueue, SequencePacker
    packer = SequencePacker(canvas_len=64, eos_id=999, pad_id=0)
    q = DomainQueue("python", packer)
    for _ in range(10):
        q.add(torch.randint(0, 999, (20,)))
    assert q.buffer_tokens > 0
    packed = q.pop_packed(64)
    assert packed is not None
    assert packed["input_ids"].size(0) == 64
    assert packed["domain"] == "python"
    print(f"  DomainQueue OK: packed {packed['n_sequences']} seqs")

def test_mesh_profiler():
    from train_mesh import MeshProfiler
    p = MeshProfiler(log_interval=2)
    p.tick_start("forward")
    time.sleep(0.01)
    p.tick_end("forward")
    p.observe_packing(0.05, 4, 128)
    p.observe_router(0.8, 0.95)
    p.log_step(2)
    assert len(p.pad_fraction) == 1
    assert len(p.router_latency_ms) == 1
    print(f"  MeshProfiler OK")

def test_async_iterator():
    from train_mesh import AsyncPrefetchTokenBucketIterator
    from torch.utils.data import Dataset, IterableDataset

    class TestIter(IterableDataset):
        def __iter__(self):
            while True:
                n = torch.randint(10, 40, (1,)).item()
                yield {"input_ids": torch.randint(0, 999, (n,)), "domain": "python"}

    ds = TestIter()
    it = AsyncPrefetchTokenBucketIterator(
        ds, canvas_len=64, eos_id=999,
        max_canvases=2, dynamic_budget=False, prefetch_queue_size=2,
    )
    batches = []
    for i, batch in enumerate(it):
        batches.append(batch)
        if i >= 3:
            break
    it.stop()
    assert len(batches) == 4
    for b in batches:
        assert b["input_ids"].dim() == 2
        assert b["input_ids"].size(1) == 64
    print(f"  AsyncPrefetchIterator OK: {len(batches)} batches, shape={batches[0]['input_ids'].shape}")

def test_integrated_forward():
    from train_mesh import MeshTrainer
    from torch.utils.data import IterableDataset

    class TestIter(IterableDataset):
        def __iter__(self):
            while True:
                n = torch.randint(10, 40, (1,)).item()
                yield {"input_ids": torch.randint(0, 1000, (n,)), "domain": "python"}

    torch.cuda.set_per_process_memory_fraction(0.8)
    vocab = 1001  # must be > eos_id=1000
    t = MeshTrainer(model_size="tiny", vocab_size=vocab, external_nodes=False,
                    nodes_dir="NoProp/nodes")
    t.train(dataset=TestIter(), num_epochs=999, batch_size=2,
            log_interval=4, mitosis_interval=999, ckpt_interval=999,
            max_steps=8, resume=False, use_packing=True)
    print(f"  Integrated forward OK ({len(t.router.nodes)} nodes)")

if __name__ == "__main__":
    print("=== Verifying all optimizations ===\n")
    test_adaptive_length_buckets()
    test_domain_queue()
    test_mesh_profiler()
    test_async_iterator()
    print("\n--- Forward pass with full pipeline ---")
    test_integrated_forward()
    print("\n=== All tests passed! ===")
