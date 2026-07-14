"""
Binary converter: HF dataset → compact binary shards.
Uses temp HF cache then cleans up. Training reads only from binary.
"""
import os, json, struct, tempfile, time
import numpy as np
import torch

from mesh_tokenizer import load_tokenizer


class BinWriter:
    """Write tokenized samples to compact binary shards (stream-optimized)."""
    def __init__(self, out_dir: str, shard_size: int = 10000):
        self.out_dir = out_dir
        self.shard_size = shard_size
        os.makedirs(out_dir, exist_ok=True)
        self.shard_idx = 0
        self.fh = None
        self._count = 0
        self._shard_counts = []

    def _open(self):
        self._close()
        p = os.path.join(self.out_dir, f"data.{self.shard_idx:05d}.bin")
        self.fh = open(p, "wb")
        self._shard_count = 0

    def _close(self):
        if self.fh is not None:
            self.fh.close()
            self.fh = None
            self._shard_counts.append(self._shard_count)

    def write(self, token_ids: bytes, mask_bytes: bytes):
        if self.fh is None:
            self._open()
        data = struct.pack("<II", len(token_ids), len(mask_bytes)) + token_ids + mask_bytes
        self.fh.write(data)
        self._shard_count += 1
        self._count += 1
        if self._shard_count >= self.shard_size:
            self._close()
            self.shard_idx += 1

    def close(self):
        self._close()
        shards = []
        for i in range(self.shard_idx + 1):
            p = os.path.join(self.out_dir, f"data.{i:05d}.bin")
            if os.path.isfile(p):
                shards.append(f"data.{i:05d}.bin")
        with open(os.path.join(self.out_dir, "index.json"), "w") as f:
            json.dump({"shards": shards, "total": self._count, "version": 2}, f)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class BinReader:
    """Read binary shard files."""
    def __init__(self, local_dir: str):
        self.local_dir = local_dir
        index = json.load(open(os.path.join(local_dir, "index.json")))
        self.shards = index["shards"]

    def iter_all(self):
        for s in self.shards:
            path = os.path.join(self.local_dir, s)
            if not os.path.isfile(path):
                continue
            with open(path, "rb") as f:
                while True:
                    header = f.read(8)
                    if len(header) < 8:
                        break
                    len_ids, len_mask = struct.unpack("<II", header)
                    ids = np.frombuffer(f.read(len_ids), dtype=np.uint32).astype(np.int64)
                    mask = np.frombuffer(f.read(len_mask), dtype=np.uint8).astype(np.int64)
                    yield {
                        "input_ids": torch.tensor(ids),
                        "attention_mask": torch.tensor(mask),
                        "labels": torch.tensor(ids),
                    }


def convert_hf_to_bin(hf_path: str, out_dir: str, hf_config: str | None = None,
                       max_docs: int = -1, shard_size: int = 10000,
                       max_seq_len: int = 2048) -> str:
    """Convert HF dataset to binary shards using temp cache."""
    import ssl; ssl._create_default_https_context = ssl._create_unverified_context

    old_cache = os.environ.get("HF_DATASETS_CACHE", "")
    tmpdir = tempfile.mkdtemp(prefix="hf_bin_")
    os.environ["HF_DATASETS_CACHE"] = tmpdir

    try:
        from datasets import load_dataset
        tok = load_tokenizer()
        kw = {"split": "train", "streaming": True}
        if hf_config:
            kw["name"] = hf_config
        ds = load_dataset(hf_path, **kw)

        os.makedirs(out_dir, exist_ok=True)
        with BinWriter(out_dir, shard_size=shard_size) as w:
            count = 0
            t0 = time.time()
            for ex in ds:
                text = None
                for k in ["text", "content"]:
                    if k in ex and isinstance(ex[k], str) and len(ex[k]) > 10:
                        text = ex[k]
                        break
                if text is None:
                    continue
                enc = tok(text, truncation=True, max_length=max_seq_len,
                          padding="max_length", return_tensors="pt")
                ids = enc["input_ids"].squeeze(0).numpy().astype(np.uint32)
                mask = enc["attention_mask"].squeeze(0).numpy().astype(np.uint8)
                w.write(ids.tobytes(), mask.tobytes())
                count += 1
                if count % 5000 == 0:
                    elapsed = time.time() - t0
                    rate = count / max(elapsed, 0.1)
                    print(f"  {count} docs ({rate:.0f}/s)...")
                if 0 < max_docs <= count:
                    break
            print(f"Done: {count} docs -> {out_dir}")
        return out_dir
    finally:
        if old_cache:
            os.environ["HF_DATASETS_CACHE"] = old_cache
        else:
            os.environ.pop("HF_DATASETS_CACHE", None)
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
