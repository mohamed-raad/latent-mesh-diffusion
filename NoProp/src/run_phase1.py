"""
Phase 1: The Acceleration Engine
  canvas_len=128, batch_size=2, small (1024-dim, 500M)
  ~6 hrs for 35K steps, checkpoint every 500 steps, logs every 100
  Ctrl+C to stop gracefully, re-run to resume
"""
import sys, os, time, torch
from torch.utils.data import IterableDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['HF_TOKEN'] = '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'

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Dual-output: console + project log file
_log_file = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "phase1_log.txt"), "a", encoding="utf-8")
_log_file.write(f"\n=== Phase 1 started at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
_log_file.flush()
_orig_print = print
def print(*a, **kw):
    kw.setdefault('flush', True)
    _orig_print(*a, **kw)
    _log_file.write(' '.join(str(x) for x in a) + '\n')
    _log_file.flush()

from train_mesh import MeshTrainer

# ---------- config ----------
MODEL_SIZE = "small"       # 1024-dim, 500M
SEQ_LEN = 128
BATCH_SIZE = 2
MAX_STEPS = 35000
LOG_INTERVAL = 100
CKPT_INTERVAL = 500
OUT_DIR = os.path.expanduser("~/checkpoints/mesh_phase1")
# ---------------------------

class SynthIter(IterableDataset):
    """Random token data for training."""
    def __init__(self, seq_len=SEQ_LEN, vocab=151643):
        self.seq_len = seq_len
        self.vocab = vocab
    def __iter__(self):
        while True:
            ids = torch.randint(4, self.vocab, (self.seq_len,))
            yield {'input_ids': ids, 'labels': ids, 'domain': 'general'}

print(f"""
==== Phase 1 - The Acceleration Engine ==========
  Model: {MODEL_SIZE} (1024-dim)
  Canvas: {SEQ_LEN} tokens, batch={BATCH_SIZE}
  Steps: {MAX_STEPS} ({MAX_STEPS*3} expert updates)
  VRAM est: ~4.5 GB
  Time est: ~6 hrs @ 0.62s/step
  Press Ctrl+C to stop (saves and exits)
  Re-run to auto-resume
=================================================
""")

trainer = MeshTrainer(
    model_size=MODEL_SIZE,
    vocab_size=151643,
    external_nodes=False,
    checkpoint_dir=OUT_DIR,
)

start = time.time()
trainer.train(
    dataset=SynthIter(),
    num_epochs=1,
    batch_size=BATCH_SIZE,
    log_interval=LOG_INTERVAL,
    mitosis_interval=999999,
    ckpt_interval=CKPT_INTERVAL,
    max_steps=MAX_STEPS,
    resume=True,
)
elapsed = time.time() - start
print(f"\n=== Phase 1 done: {MAX_STEPS} steps in {elapsed/3600:.1f} hrs ({elapsed/MAX_STEPS:.3f}s/step) ===")
