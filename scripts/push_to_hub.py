"""
Push checkpoints to HuggingFace Hub.
Usage:
  python scripts/push_to_hub.py --repo mohamed-raad/latent-mesh-checkpoints --checkpoint ./checkpoints/phase_phase_1_—_core_250m
"""
import os, sys, argparse
from huggingface_hub import HfApi, upload_folder

def main():
    parser = argparse.ArgumentParser(description="Push checkpoints to HF Hub")
    parser.add_argument("--repo", default="mohamed99raad/Latent-Mesh-Model", help="HF repo ID")
    parser.add_argument("--checkpoint", default="./checkpoints", help="Local checkpoint directory")
    parser.add_argument("--token", default=None, help="HF token (or HF_TOKEN env)")
    parser.add_argument("--logs", default=None, help="Optional logs directory to upload")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: Set HF_TOKEN env var or pass --token")
        sys.exit(1)

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=True, exist_ok=True)

    upload_folder(
        folder_path=args.checkpoint,
        repo_id=args.repo,
        repo_type="model",
        path_in_repo="",
        token=token,
    )
    print(f"Uploaded {args.checkpoint} to {args.repo}")

    if args.logs and os.path.isdir(args.logs):
        upload_folder(
            folder_path=args.logs,
            repo_id=args.repo,
            repo_type="model",
            path_in_repo="logs",
            token=token,
        )
        print(f"Uploaded logs to {args.repo}/logs")

if __name__ == "__main__":
    main()
