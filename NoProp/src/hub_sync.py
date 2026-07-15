"""Multi-notebook checkpoint sync via HuggingFace Hub.

Usage:
    sync = HubSync(repo_id="mohamed99raad/Latent-Mesh-Model", token=HF_TOKEN)
    sync.checkout_latest(local_dir="./checkpoints")   # pull before training
    sync.push(local_dir="./checkpoints", step=100)     # push every N steps

Multi-notebook coordination:
    sync = HubSync(repo_id=..., token=..., notebook_id="n1")
    sync.advertise()                         # create notebook lock
    assigned = sync.claim_experts(           # get expert shard
        all_experts=["expert_0", ...],
        min_per_notebook=1,
    )
    sync.release()                           # remove lock when done
"""

import json
import os
import tempfile
import time
import uuid
from pathlib import Path

try:
    from huggingface_hub import HfApi, upload_folder, create_commit, CommitOperationAdd, CommitOperationDelete
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False


EXPERTS_LOCK_FILE = ".expert_lock.txt"
NOTEBOOK_LOCK_PREFIX = "notebook_locks/"


class HubSync:
    """Synchronise checkpoints and coordinate expert shards via HF Hub."""

    def __init__(
        self,
        repo_id: str = "mohamed99raad/Latent-Mesh-Model",
        token: str | None = None,
        notebook_id: str | None = None,
    ):
        if not _HF_AVAILABLE:
            raise ImportError("huggingface_hub is required. Run: pip install huggingface_hub")
        self.repo_id = repo_id
        self.token = token or os.environ.get("HF_TOKEN")
        if not self.token:
            raise ValueError("Set HF_TOKEN env var or pass token= to HubSync")
        self.notebook_id = notebook_id or f"nb_{uuid.uuid4().hex[:8]}"
        self.api = HfApi()
        self.api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
        self._lock_acquired = False

    # ── checkpoint sync ──────────────────────────────────────────────

    def latest_step(self) -> int:
        """Return highest numeric step tag found in the Hub repo."""
        files = self._list_files()
        steps = []
        for f in files:
            name = f.rstrip("/")
            if name.startswith("step_") and name.endswith(".pt"):
                tag = name.replace("step_", "").replace(".pt", "")
                if tag.isdigit():
                    steps.append(int(tag))
            if name == "step_latest.pt":
                # treat latest as marker – we prefer the highest numeric
                pass
        return max(steps) if steps else 0

    def checkout_latest(self, local_dir: str = "checkpoints/mesh") -> str | None:
        """Download the latest checkpoint into *local_dir*.  Returns the path or None."""
        latest = self.latest_step()
        if latest == 0:
            return None
        os.makedirs(local_dir, exist_ok=True)
        filename = f"step_{latest}.pt"
        local_path = os.path.join(local_dir, filename)
        if os.path.isfile(local_path):
            return local_path
        try:
            self.api.hf_hub_download(
                repo_id=self.repo_id,
                filename=filename,
                local_dir=local_dir,
                token=self.token,
                repo_type="model",
            )
            return local_path
        except Exception:
            return None

    def push(self, local_dir_or_file: str, tag: int | str, final: bool = False):
        """Upload a checkpoint to the hub repo.

        If *local_dir_or_file* contains a *step_<tag>.pt* or *expert_<tag>.pt* file,
        that single file is uploaded.  Otherwise the entire directory is uploaded
        via ``upload_folder`` (used for expert shards).

        Also updates *step_latest.pt* for master checkpoints.
        """
        tag_str = "final" if final else str(tag)
        # Try single-file push first
        for prefix in ("step_", "expert_", "shard_"):
            candidate = os.path.join(local_dir_or_file, f"{prefix}{tag_str}.pt")
            if os.path.isfile(candidate):
                remote_name = f"{prefix}{tag_str}.pt"
                self.api.upload_file(
                    path_or_fileobj=candidate,
                    path_in_repo=remote_name,
                    repo_id=self.repo_id,
                    repo_type="model",
                    token=self.token,
                )
                if prefix == "step_" and not final:
                    self.api.upload_file(
                        path_or_fileobj=candidate,
                        path_in_repo="step_latest.pt",
                        repo_id=self.repo_id,
                        repo_type="model",
                        token=self.token,
                    )
                print(f"  HubSync: pushed {remote_name} → {self.repo_id}")
                return
        # Fall back to directory upload (expert shard or other directory)
        if os.path.isdir(local_dir_or_file):
            upload_folder(
                folder_path=local_dir_or_file,
                repo_id=self.repo_id,
                repo_type="model",
                path_in_repo=tag_str,
                token=self.token,
            )
            print(f"  HubSync: pushed {local_dir_or_file} → {self.repo_id}/{tag_str}")

    def push_metadata(self, metadata: dict):
        """Upload a small JSON metadata file for coordination state."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(metadata, f)
            tmp = f.name
        try:
            self.api.upload_file(
                path_or_fileobj=tmp,
                path_in_repo=".metadata.json",
                repo_id=self.repo_id,
                repo_type="model",
                token=self.token,
            )
        finally:
            os.unlink(tmp)

    # ── multi-notebook coordination ──────────────────────────────────

    def advertise(self, ttl_seconds: int = 600):
        """Create a lock file on the hub advertising this notebook is alive."""
        lock_path = f"{NOTEBOOK_LOCK_PREFIX}{self.notebook_id}.lock"
        payload = json.dumps({
            "notebook_id": self.notebook_id,
            "pid": os.getpid(),
            "expires_at": time.time() + ttl_seconds,
            "created_at": time.time(),
        })
        self.api.upload_file(
            path_or_fileobj=payload.encode(),
            path_in_repo=lock_path,
            repo_id=self.repo_id,
            repo_type="model",
            token=self.token,
        )
        self._lock_acquired = True
        self._lock_path = lock_path
        self._lock_expires = time.time() + ttl_seconds

    def refresh_advertisement(self, ttl_seconds: int = 600):
        """Refresh the TTL on the lock so other notebooks know we are still alive."""
        if not self._lock_acquired:
            return
        payload = json.dumps({
            "notebook_id": self.notebook_id,
            "pid": os.getpid(),
            "expires_at": time.time() + ttl_seconds,
            "created_at": time.time(),
        })
        try:
            self.api.upload_file(
                path_or_fileobj=payload.encode(),
                path_in_repo=self._lock_path,
                repo_id=self.repo_id,
                repo_type="model",
                token=self.token,
            )
            self._lock_expires = time.time() + ttl_seconds
        except Exception:
            pass

    def release(self):
        """Remove the notebook lock from the hub."""
        if not self._lock_acquired:
            return
        try:
            self.api.delete_file(
                path_in_repo=self._lock_path,
                repo_id=self.repo_id,
                repo_type="model",
                token=self.token,
            )
        except Exception:
            pass
        self._lock_acquired = False

    def get_active_notebooks(self) -> list[dict]:
        """List all notebooks with active (non-expired) locks."""
        files = self._list_files()
        locks = [f for f in files if f.startswith(NOTEBOOK_LOCK_PREFIX) and f.endswith(".lock")]
        active = []
        now = time.time()
        for lock in locks:
            try:
                content = self.api.hf_hub_download(
                    repo_id=self.repo_id,
                    filename=lock,
                    token=self.token,
                    repo_type="model",
                )
                data = json.loads(Path(content).read_text())
                if data.get("expires_at", 0) > now:
                    active.append(data)
            except Exception:
                pass
        return active

    def claim_experts(
        self,
        all_experts: list[str],
        min_per_notebook: int = 1,
    ) -> list[str]:
        """Coordinate expert sharding across active notebooks.

        Returns the list of expert IDs assigned to *this* notebook
        based on a deterministic hash of (expert, notebook_id).
        """
        n_notebooks = len(self.get_active_notebooks()) or 1
        if n_notebooks <= 1:
            return all_experts  # single notebook trains everything

        # Sort experts and notebooks deterministically
        sorted_experts = sorted(all_experts)
        self.advertise()

        # Simple round-robin assignment
        idx = sorted(self.get_active_notebooks(), key=lambda x: x["notebook_id"])
        my_pos = next(i for i, nb in enumerate(idx) if nb["notebook_id"] == self.notebook_id)
        assigned = sorted_experts[my_pos::n_notebooks]
        if len(assigned) < min_per_notebook:
            assigned = sorted_experts[:min_per_notebook]
        return assigned

    # ── internal helpers ─────────────────────────────────────────────

    def _list_files(self) -> list[str]:
        try:
            info = self.api.get_repo(self.repo_id, repo_type="model", token=self.token)
            siblings = self.api.get_model_info(self.repo_id, token=self.token).siblings
            return [s.rfilename for s in siblings]
        except Exception:
            return []

    def close(self):
        self.release()
