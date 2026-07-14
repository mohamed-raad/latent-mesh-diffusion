"""
Fast binary streaming — lightweight MDS-compatible format for Windows.
Self-contained: no mosaicml/streaming dependency needed.

Format:
  shard: [n_samples: uint32] [offsets: n_samples*uint32] [data: bytes...]
  index.json: {shards: [...], columns: {...}, total_samples: N, compression: null}
"""
import os
import json
import struct
import time
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader
from mesh_tokenizer import VOCAB_SIZE, load_tokenizer


# ═══════════════════════════════════════════════════════════
# Writer
# ═══════════════════════════════════════════════════════════

class FastBinWriter:
    """Write tokenized samples to binary shards."""

    def __init__(self, out_dir: str, columns: dict[str, str], compression: str | None = None,
                 shard_size: int = 10000):
        self.out_dir = out_dir
        self.columns = columns
        self.compression = compression
        self.shard_size = shard_size
        os.makedirs(self.out_dir, exist_ok=True)
        self.shard_idx = 0
        self.shard_offsets: list[int] = []
        self.shard_file: str | None = None
        self.fh = None

    def _open_shard(self):
        self._close_shard()
        self.shard_file = os.path.join(self.out_dir, f"shard.{self.shard_idx:05d}.bin")
        self.fh = open(self.shard_file, "wb")
        self.fh.write(struct.pack("<I", 0))
        self.shard_offsets = []

    def _close_shard(self):
        if self.fh is not None:
            n = len(self.shard_offsets)
            self.fh.seek(0)
            self.fh.write(struct.pack("<I", n))
            for offset in self.shard_offsets:
                self.fh.write(struct.pack("<I", offset))
            self.fh.close()
            self.fh = None

    def write(self, sample: dict):
        if self.fh is None:
            self._open_shard()

        data = b""
        for col_name in self.columns:
            raw = sample.get(col_name, b"")
            if isinstance(raw, bytes):
                data += struct.pack("<I", len(raw)) + raw
            else:
                data += struct.pack("<I", 0)

        self.shard_offsets.append(self.fh.tell())
        self.fh.write(data)

        if len(self.shard_offsets) >= self.shard_size:
            self._close_shard()
            self.shard_idx += 1

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._close_shard()
        self._write_index()

    def _write_index(self):
        shards = []
        for i in range(self.shard_idx + 1):
            p = f"shard.{i:05d}.bin"
            fp = os.path.join(self.out_dir, p)
            if os.path.isfile(fp):
                shards.append(p)
        index = {
            "shards": shards,
            "columns": self.columns,
            "total_samples": sum(len(self._get_shard_samples(p)) if False else 0 for p in shards),
            "compression": self.compression,
            "version": 1,
        }
        with open(os.path.join(self.out_dir, "index.json"), "w") as f:
            json.dump(index, f)

    def _get_shard_samples(self, shard: str) -> list:
        return []


# ═══════════════════════════════════════════════════════════
# Reader
# ═══════════════════════════════════════════════════════════

class FastBinReader:
    """Read binary shard files."""

    def __init__(self, local_dir: str):
        self.local_dir = local_dir
        self.index_path = os.path.join(local_dir, "index.json")
        if not os.path.isfile(self.index_path):
            raise FileNotFoundError(f"No index.json in {local_dir}")
        with open(self.index_path) as f:
            self.index = json.load(f)
        self.columns = self.index.get("columns", {})
        self.shards = self.index.get("shards", [])
        self._sample_count = 0

    def read_shard(self, shard_name: str) -> list[dict]:
        path = os.path.join(self.local_dir, shard_name)
        with open(path, "rb") as f:
            header = f.read(4)
            n = struct.unpack("<I", header)[0]
            offsets = []
            for _ in range(n):
                offsets.append(struct.unpack("<I", f.read(4))[0])
            samples = []
            col_names = list(self.columns.keys())
            for i, offset in enumerate(offsets):
                f.seek(4 + 4 * n + offset)
                sample = {}
                for col_name in col_names:
                    sz = struct.unpack("<I", f.read(4))[0]
                    sample[col_name] = f.read(sz)
                samples.append(sample)
            return samples

    def iter_all(self):
        for shard in self.shards:
            for sample in self.read_shard(shard):
                yield sample


# ═══════════════════════════════════════════════════════════
# HF → Binary conversion
# ═══════════════════════════════════════════════════════════

def convert_hf_to_binary(
    hf_path: str,
    out_dir: str,
    split: str = "train",
    max_docs: int = -1,
    hf_config: str | None = None,
    text_key: str = "text",
    max_seq_len: int = 2048,
    shard_size: int = 5000,
):
    """Convert a HF dataset to binary shards (pre-tokenized)."""
    from datasets import load_dataset
    import ssl; ssl._create_default_https_context = ssl._create_unverified_context

    tok = load_tokenizer()
    os.makedirs(out_dir, exist_ok=True)

    kw = {"split": split, "streaming": True}
    if hf_config:
        kw["name"] = hf_config
    ds = load_dataset(hf_path, **kw)

    columns = {"input_ids": "bytes", "attention_mask": "bytes", "labels": "bytes"}
    with FastBinWriter(out_dir=out_dir, columns=columns, shard_size=shard_size) as writer:
        count = 0
        for example in ds:
            text = None
            for key in [text_key, "text", "content"]:
                if key in example and isinstance(example[key], str) and len(example[key]) > 10:
                    text = example[key]
                    break
            if text is None:
                continue

            enc = tok(text, truncation=True, max_length=max_seq_len, padding="max_length", return_tensors="pt")
            ids = enc["input_ids"].squeeze(0).numpy().astype(np.uint32)
            mask = enc["attention_mask"].squeeze(0).numpy().astype(np.uint8)

            writer.write({
                "input_ids": ids.tobytes(),
                "attention_mask": mask.tobytes(),
                "labels": ids.tobytes(),
            })
            count += 1
            if count % shard_size == 0:
                print(f"  Converted {count} docs...")
            if 0 < max_docs <= count:
                break

    print(f"Done: {count} docs -> {out_dir}")
    return out_dir


# ═══════════════════════════════════════════════════════════
# PyTorch Dataset
# ═══════════════════════════════════════════════════════════

class FastBinMeshDataset(IterableDataset):
    """Stream pre-tokenized binary shards at GPU throughput."""

    def __init__(self, local_dir: str, max_seq_len: int = 2048, repeat: bool = True):
        self.local_dir = local_dir
        self.max_seq_len = max_seq_len
        self.repeat = repeat
        self.reader = FastBinReader(local_dir)

    def __iter__(self):
        while True:
            for sample in self.reader.iter_all():
                input_ids = np.frombuffer(sample["input_ids"], dtype=np.uint32).astype(np.int64)
                attention_mask = np.frombuffer(sample["attention_mask"], dtype=np.uint8).astype(np.int64)
                if len(input_ids) > self.max_seq_len:
                    input_ids = input_ids[:self.max_seq_len]
                    attention_mask = attention_mask[:self.max_seq_len]
                yield {
                    "input_ids": torch.tensor(input_ids),
                    "attention_mask": torch.tensor(attention_mask),
                    "labels": torch.tensor(input_ids),
                }
            if not self.repeat:
                break


class MixedFastBinDataset(IterableDataset):
    """Interleave multiple binary datasets with weights."""

    def __init__(self, sources: list[dict], max_seq_len: int = 2048):
        self.datasets = []
        self.weights = []
        for src in sources:
            ds = FastBinMeshDataset(
                local_dir=src["local_dir"],
                max_seq_len=max_seq_len,
                repeat=src.get("repeat", True),
            )
            self.datasets.append(ds)
            self.weights.append(src.get("weight", 1.0))

    def __iter__(self):
        iters = [iter(ds) for ds in self.datasets]
        import random
        while True:
            idx = random.choices(range(len(iters)), weights=self.weights, k=1)[0]
            try:
                yield next(iters[idx])
            except StopIteration:
                iters[idx] = iter(self.datasets[idx])
                yield next(iters[idx])


# ═══════════════════════════════════════════════════════════
# Convenience converters
# ═══════════════════════════════════════════════════════════

def convert_fineweb_edu(out_dir: str = "./bin_data/fineweb_edu", max_docs: int = -1):
    return convert_hf_to_binary("HuggingFaceFW/fineweb-edu", out_dir, max_docs=max_docs)


def convert_fineweb2_ar(out_dir: str = "./bin_data/fineweb2_ar", max_docs: int = -1):
    return convert_hf_to_binary("HuggingFaceFW/fineweb-2", out_dir, hf_config="ara_Arab", max_docs=max_docs)


def convert_fineweb2_en(out_dir: str = "./bin_data/fineweb2_en", max_docs: int = -1):
    return convert_hf_to_binary("HuggingFaceFW/fineweb-2", out_dir, hf_config="eng_Latn", max_docs=max_docs)


# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert HF datasets to fast binary shards")
    parser.add_argument("--dataset", choices=["fineweb-edu", "fineweb2-ar", "fineweb2-en", "all"], default="all")
    parser.add_argument("--out-dir", default="./bin_data")
    parser.add_argument("--max-docs", type=int, default=-1)
    parser.add_argument("--shard-size", type=int, default=5000)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    args = parser.parse_args()

    tasks = []
    if args.dataset in ("fineweb-edu", "all"):
        tasks.append((convert_fineweb_edu, f"{args.out_dir}/fineweb_edu"))
    if args.dataset in ("fineweb2-ar", "all"):
        tasks.append((convert_fineweb2_ar, f"{args.out_dir}/fineweb2_ar"))
    if args.dataset in ("fineweb2-en", "all"):
        tasks.append((convert_fineweb2_en, f"{args.out_dir}/fineweb2_en"))

    for fn, out_dir in tasks:
        print(f"\nConverting {fn.__name__} -> {out_dir}...")
        fn(out_dir, max_docs=args.max_docs)
