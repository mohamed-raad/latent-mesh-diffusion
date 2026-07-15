"""
MASTER notebook — trains the core engine (canvas + latent space).
Creates seed expert nodes so workers know what to claim, then FREEZES them
so only the core is updated.  Pushes checkpoint to HF Hub every N steps.

Workers pull from Hub, unfreeze their expert shard, train, push back.

Usage on Colab:
    !python scripts/run_master.py --hub_repo mohamed99raad/Latent-Mesh-Model
"""

import sys, os, time, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "NoProp", "src"))

from train_mesh import MeshTrainer

# ---------- config ----------
MODEL_SIZE = "small"
SEQ_LEN = 128
BATCH_SIZE = 2
MAX_STEPS = 50000
LOG_INTERVAL = 50
CKPT_INTERVAL = 500
OUT_DIR = os.path.expanduser("~/checkpoints/mesh_master")
HUB_REPO = "mohamed99raad/Latent-Mesh-Model"
NUM_SEED_EXPERTS = 8
# ---------------------------

class SynthIter(torch.utils.data.IterableDataset):
    def __init__(self, seq_len=SEQ_LEN, vocab=151643):
        self.seq_len = seq_len
        self.vocab = vocab
    def __iter__(self):
        while True:
            ids = torch.randint(4, self.vocab, (self.seq_len,))
            yield {'input_ids': ids, 'labels': ids, 'domain': 'general'}

print(f"""
==== MASTER — Core Engine Training ===============
  Model: {MODEL_SIZE} (1024-dim)
  Canvas: {SEQ_LEN} tokens, batch={BATCH_SIZE}
  Steps: {MAX_STEPS}
  Seed experts: {NUM_SEED_EXPERTS} (frozen — workers train them)
  Hub: {HUB_REPO}
  Pushing every {CKPT_INTERVAL} steps
  Ctrl+C to stop (saves and exits)
  Re-run to auto-resume from last pushed checkpoint
=================================================
""")

trainer = MeshTrainer(
    model_size=MODEL_SIZE,
    vocab_size=151643,
    external_nodes=True,            # create seed experts
    experts_count=NUM_SEED_EXPERTS, # how many to auto-create
    checkpoint_dir=OUT_DIR,
    hub_repo=HUB_REPO,
)

# FREEZE all expert nodes — master only trains the core
all_experts = list(trainer.router.nodes.keys())
trainer.freeze_experts(all_experts)
print(f"Frozen {len(all_experts)} experts — only core will train")

# Push initial state to Hub so workers can discover experts + IDs
trainer._save_checkpoint()
if trainer.hub_sync:
    trainer.hub_sync.push(OUT_DIR, trainer.step or 0)
    trainer.hub_sync.push_metadata({
        "step": trainer.step,
        "expert_ids": all_experts,
        "role": "master",
    })

trainer.train(
    dataset=SynthIter(),
    num_epochs=1,
    batch_size=BATCH_SIZE,
    log_interval=LOG_INTERVAL,
    mitosis_interval=999999,
    ckpt_interval=CKPT_INTERVAL,
    max_steps=MAX_STEPS,
    resume=True,
    hub_repo=HUB_REPO,
)
print(f"\n=== Master done ===")
