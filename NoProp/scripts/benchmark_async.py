"""Benchmark async prefetch + dynamic budget + domain-queued packing."""
import sys, os, time, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from train_mesh import MeshTrainer
from torch.utils.data import IterableDataset

class BenchIter(IterableDataset):
    def __iter__(self):
        while True:
            n = torch.randint(20, 61, (1,)).item()
            domain = ["python", "math", "rust", "physics"][torch.randint(0, 4, (1,)).item()]
            yield {"input_ids": torch.randint(4, 1000, (n,)), "domain": domain}

CANVAS = 128
STEPS = 40

# — Packing + async prefetch + dynamic budget + domain queues —
print("=== Packing (full pipeline: async + dynamic + domain queues + telemetry) ===")
torch.cuda.reset_peak_memory_stats()
ds = BenchIter()
t = MeshTrainer(model_size="tiny", vocab_size=1001, external_nodes=False,
                canvas_len=CANVAS,
                checkpoint_dir=os.path.expanduser("~/checkpoints/mesh_bench3"))
start = time.time()
t.train(dataset=ds, num_epochs=999, batch_size=2,
        log_interval=999, mitosis_interval=999, ckpt_interval=999,
        max_steps=STEPS, resume=False, use_packing=True)
pack_time = time.time() - start
pack_mem = torch.cuda.max_memory_allocated() / 1e9
print(f"  {STEPS} steps in {pack_time:.1f}s = {pack_time/STEPS:.3f}s/step, VRAM={pack_mem:.2f}GB")
del t
