"""
WORKER notebook — pulls the latest master core checkpoint from HF Hub,
CLAIMS a unique expert shard (round-robin across active workers),
freezes core params, trains ONLY claimed experts, pushes back.

Launch multiple copies simultaneously — each gets a different shard.

Usage on Colab:
    !python scripts/run_worker.py --hub_repo mohamed99raad/Latent-Mesh-Model
"""

import sys, os, time, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "NoProp", "src"))

from train_mesh import MeshTrainer
from hub_sync import HubSync

# ---------- config ----------
MODEL_SIZE = "small"
SEQ_LEN = 128
BATCH_SIZE = 4
MAX_STEPS = 20000
LOG_INTERVAL = 50
CKPT_INTERVAL = 500
PUSH_INTERVAL = 250
OUT_DIR = os.path.expanduser("~/checkpoints/mesh_worker")
HUB_REPO = "mohamed99raad/Latent-Mesh-Model"
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
==== WORKER — Expert Shard Training ==============
  Model: {MODEL_SIZE} (1024-dim)
  Canvas: {SEQ_LEN} tokens, batch={BATCH_SIZE}
  Steps: {MAX_STEPS}
  Hub: {HUB_REPO}
  Pushing every {PUSH_INTERVAL} steps
  Core frozen, training experts only
=================================================
""")

# 1. Init trainer with same config as master
trainer = MeshTrainer(
    model_size=MODEL_SIZE,
    vocab_size=151643,
    external_nodes=True,
    experts_count=8,            # must match master
    checkpoint_dir=OUT_DIR,
    hub_repo=HUB_REPO,
)

# 2. Pull core from Hub (loads latest master checkpoint)
if trainer.hub_sync is not None:
    pulled = trainer.pull_core_from_hub(trainer.hub_sync)
    if not pulled:
        print("  No master checkpoint found on Hub — waiting for master to push first.")
        print(f"  Check {HUB_REPO} for a step_*.pt file.")
        sys.exit(1)
else:
    print("  HubSync not available — check HF_TOKEN")
    sys.exit(1)

# 3. Discover all expert IDs from Hub metadata
meta = {}
try:
    meta_path = trainer.hub_sync.api.hf_hub_download(
        repo_id=HUB_REPO, filename=".metadata.json",
        token=trainer.hub_sync.token, repo_type="model",
    )
    import json
    with open(meta_path) as f:
        meta = json.load(f)
except Exception:
    pass

all_experts = meta.get("expert_ids") or list(trainer.router.nodes.keys())
if not all_experts:
    print("  No experts found — master hasn't created them yet.")
    sys.exit(1)

# 4. Advertise this worker and claim shard
trainer.hub_sync.advertise()
my_experts = trainer.hub_sync.claim_experts(
    all_experts=all_experts,
    min_per_notebook=1,
)
print(f"  Claimed {len(my_experts)} experts: {my_experts}")

# 5. Freeze ALL experts, then unfreeze only our shard
trainer.freeze_experts()
trainer.unfreeze_experts(my_experts)
print(f"  Frozen {len(all_experts) - len(my_experts)} remote experts — training only my {len(my_experts)}")

# 6. Freeze core params
for p in trainer.token_embedding.parameters(): p.requires_grad = False
for p in trainer.lm_head.parameters(): p.requires_grad = False
for p in trainer.latent_space.parameters(): p.requires_grad = False
for p in trainer.speculator.parameters(): p.requires_grad = False
for p in trainer.kv_compressor.parameters(): p.requires_grad = False
for p in trainer.global_cognitive_layer.parameters(): p.requires_grad = False
if trainer.canvas is not None:
    for p in trainer.canvas.parameters(): p.requires_grad = False
print("  Core params frozen")

# 7. Patch _save_checkpoint to also push shard to Hub
_save_orig = trainer._save_checkpoint
def _save_and_push_shard(final=False, latest=False):
    _save_orig(final=final, latest=latest)
    if trainer.hub_sync is not None and trainer.step % PUSH_INTERVAL == 0:
        trainer.push_expert_shard(trainer.hub_sync, my_experts)
trainer._save_checkpoint = _save_and_push_shard

# 8. Train
trainer.train(
    dataset=SynthIter(),
    num_epochs=1,
    batch_size=BATCH_SIZE,
    log_interval=LOG_INTERVAL,
    mitosis_interval=999999,
    ckpt_interval=CKPT_INTERVAL,
    max_steps=MAX_STEPS,
    resume=False,               # don't load worker's own checkpoint — load from Hub
    hub_repo=HUB_REPO,
    multi_notebook=True,
)

# 9. Final shard push
if trainer.hub_sync:
    trainer.push_expert_shard(trainer.hub_sync, my_experts)
    trainer.hub_sync.release()

print(f"\n=== Worker done — trained experts {my_experts} ===")
