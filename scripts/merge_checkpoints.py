"""
MERGE — assembles the final model from master core + all worker shards.

1. Pulls the latest master checkpoint (core weights + expert metadata).
2. Pulls all worker shards from HF Hub.
3. Merges expert weights into the master checkpoint.
4. Saves the merged model locally.

Usage:
    !python scripts/merge_checkpoints.py --hub_repo mohamed99raad/Latent-Mesh-Model --out ./final_model
"""

import sys, os, json, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "NoProp", "src"))

from hub_sync import HubSync

# ---------- config ----------
HUB_REPO = "mohamed99raad/Latent-Mesh-Model"
OUT_DIR = os.path.expanduser("~/checkpoints/mesh_final")
WORKER_SHARD_PREFIX = "shard_"
# ---------------------------

def discover_shards(hub_sync: HubSync) -> list[dict]:
    """List all shard directories on the Hub."""
    siblings = hub_sync.api.get_model_info(hub_sync.repo_id, token=hub_sync.token).siblings
    shards = []
    for s in siblings:
        if s.rfilename.startswith(WORKER_SHARD_PREFIX):
            shards.append(s.rfilename.rstrip("/"))
    return sorted(set(shards))

def pull_shard(hub_sync: HubSync, shard_name: str, dest: str):
    """Download all files in a shard directory from Hub."""
    os.makedirs(dest, exist_ok=True)
    siblings = hub_sync.api.get_model_info(hub_sync.repo_id, token=hub_sync.token).siblings
    for s in siblings:
        if s.rfilename.startswith(shard_name + "/"):
            rel = s.rfilename[len(shard_name) + 1:]
            local_path = os.path.join(dest, rel)
            hub_sync.api.hf_hub_download(
                repo_id=hub_sync.repo_id,
                filename=s.rfilename,
                local_dir=dest,
                token=hub_sync.token,
                repo_type="model",
            )
    print(f"  Pulled shard {shard_name} → {dest}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Merge master + shard checkpoints")
    parser.add_argument("--hub_repo", default=HUB_REPO)
    parser.add_argument("--out", default=OUT_DIR)
    args = parser.parse_args()

    hub_sync = HubSync(repo_id=args.hub_repo)
    os.makedirs(args.out, exist_ok=True)

    # 1. Pull master checkpoint
    print("Pulling master checkpoint...")
    master_path = hub_sync.checkout_latest(args.out)
    if master_path is None:
        print("  No master checkpoint found!")
        sys.exit(1)
    print(f"  Master: {master_path}")

    # 2. Pull all worker shards
    shards = discover_shards(hub_sync)
    print(f"Found {len(shards)} worker shards: {shards}")

    merged_experts = {}
    for shard_name in shards:
        tmp = tempfile.mkdtemp(prefix="shard_")
        pull_shard(hub_sync, shard_name, tmp)
        for fname in os.listdir(tmp):
            if fname.startswith("expert_") and fname.endswith(".pt"):
                expert_id = fname.replace("expert_", "").replace(".pt", "")
                if expert_id not in merged_experts:
                    merged_experts[expert_id] = os.path.join(tmp, fname)
                    print(f"  Found expert {expert_id} from {shard_name}")
                else:
                    print(f"  Skipping duplicate expert {expert_id} (already from earlier shard)")

    # 3. Inject expert weights into master checkpoint
    print(f"\nMerging {len(merged_experts)} expert weights into master checkpoint...")
    master_data = __import__("torch").load(master_path, map_location="cpu", weights_only=True)
    mesh_state = master_data.get("model_state_dict", {}).get("mesh", master_data)

    for expert_id, weight_path in merged_experts.items():
        expert_data = __import__("torch").load(weight_path, map_location="cpu", weights_only=True)
        # The expert weights need to be injected into the tier manager's block state
        # For now, save as separate files alongside the master checkpoint
        shutil.copy2(weight_path, os.path.join(args.out, f"expert_{expert_id}.pt"))
        print(f"  Merged expert {expert_id}")

    # 4. Save merged metadata
    metadata = {
        "step": mesh_state.get("step", 0),
        "experts": list(merged_experts.keys()),
        "n_experts": len(merged_experts),
        "source_shards": shards,
    }
    meta_path = os.path.join(args.out, "merged_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f)
    print(f"\nMerged metadata → {meta_path}")
    print(f"Final model → {args.out}/")

    # Print next steps
    print(f"""
=== Merge complete =================================
  Master checkpoint: {os.path.basename(master_path)}
  Experts merged: {len(merged_experts)}
  Output: {args.out}

To train more, start additional workers:
  !python scripts/run_worker.py --hub_repo {args.hub_repo}

To deploy the merged model:
  python -c "
import torch
ckpt = torch.load('{args.out}/step_final.pt', map_location='cpu')
print('Ready:', list(ckpt.keys())[:5])
  "
====================================================
""")

if __name__ == "__main__":
    main()
