"""Benchmark packing speedup vs baseline — same canvas_len."""
import sys, os, time, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from train_mesh import MeshTrainer
from torch.utils.data import IterableDataset


class BenchDataset(IterableDataset):
    def __init__(self, fixed_len=0):
        self.fixed_len = fixed_len

    def __iter__(self):
        while True:
            n = self.fixed_len if self.fixed_len > 0 else torch.randint(20, 61, (1,)).item()
            ids = torch.randint(4, 1000, (n,))
            yield {"input_ids": ids, "labels": ids, "domain": "python"}


CANVAS = 128
BATCH = 2
STEPS = 40

torch.cuda.reset_peak_memory_stats()

# — Baseline: canvas=128, batch=2, fixed-length 128 —
print(f"=== Baseline: canvas={CANVAS}, batch={BATCH}, fixed-length {CANVAS} ===")
ds = BenchDataset(fixed_len=CANVAS)
t = MeshTrainer(model_size="tiny", vocab_size=151643, external_nodes=False,
                canvas_len=CANVAS,
                checkpoint_dir=os.path.expanduser("~/checkpoints/mesh_bench"))
start = time.time()
t.train(dataset=ds, num_epochs=999, batch_size=BATCH,
        log_interval=999, mitosis_interval=999, ckpt_interval=999,
        max_steps=STEPS, resume=False, use_packing=False)
base_time = time.time() - start
base_mem = torch.cuda.max_memory_allocated() / 1e9
base_tokens = STEPS * BATCH * CANVAS
print(f"  {STEPS} steps in {base_time:.1f}s = {base_time/STEPS:.3f}s/step")
print(f"  {base_tokens} tokens in {base_time:.1f}s = {base_tokens/base_time:.0f} tok/s, VRAM={base_mem:.2f}GB")
del t

# — Packing: canvas=128, variable-length 20-60 —
print(f"\n=== Packing: canvas={CANVAS}, variable-length 20-60, budget={CANVAS*BATCH*2} ===")
torch.cuda.reset_peak_memory_stats()
ds2 = BenchDataset(fixed_len=0)
t2 = MeshTrainer(model_size="tiny", vocab_size=151643, external_nodes=False,
                 canvas_len=CANVAS,
                 checkpoint_dir=os.path.expanduser("~/checkpoints/mesh_bench2"))
start = time.time()
t2.train(dataset=ds2, num_epochs=999, batch_size=BATCH,
         log_interval=999, mitosis_interval=999, ckpt_interval=999,
         max_steps=STEPS, resume=False, use_packing=True)
pack_time = time.time() - start
pack_mem = torch.cuda.max_memory_allocated() / 1e9

# Estimate tokens processed with packing: each step budget * steps
budget = CANVAS * BATCH * 2
pack_tokens = STEPS * budget
print(f"  {STEPS} steps in {pack_time:.1f}s = {pack_time/STEPS:.3f}s/step")
print(f"  ~{pack_tokens} tokens in {pack_time:.1f}s = {pack_tokens/pack_time:.0f} tok/s, VRAM={pack_mem:.2f}GB")

print(f"\n=== RESULTS ===")
print(f"  Tokens/sec baseline: {base_tokens/base_time:.0f}")
print(f"  Tokens/sec packing:  {pack_tokens/pack_time:.0f}")
print(f"  Speedup: {pack_tokens/pack_time / (base_tokens/base_time):.1f}x")
print(f"  VRAM baseline={base_mem:.2f}GB  packing={pack_mem:.2f}GB")
