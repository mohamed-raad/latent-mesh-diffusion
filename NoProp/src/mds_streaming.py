"""
MDS Streaming — MosaicML streaming format for FineWeb-Edu + FineWeb-2 (AR, EN).
Pre-tokenizes to .mds shards for zero-bottleneck GPU streaming.
"""
import os
import json
import time
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader
from mesh_tokenizer import VOCAB_SIZE, load_tokenizer

try:
    from streaming import MDSWriter, StreamingDataset
    HAS_MOSAICML = True
except ImportError:
    HAS_MOSAICML = False
    print("WARNING: mosaicml-streaming not installed. Falling back to HF streaming.")


def convert_to_mds(
    hf_path: str,
    out_dir: str,
    split: str = "train",
    max_docs: int = -1,
    hf_config: str | None = None,
    text_key: str = "text",
    compression: str | None = "zstd",
    batch_size: int = 1000,
):
    """Convert a HF dataset to MDS shards (pre-tokenized)."""
    from datasets import load_dataset
    import ssl; ssl._create_default_https_context = ssl._create_unverified_context

    tok = load_tokenizer()
    os.makedirs(out_dir, exist_ok=True)

    kw = {"split": split, "streaming": True}
    if hf_config:
        kw["name"] = hf_config
    ds = load_dataset(hf_path, **kw)

    columns = {"input_ids": "bytes", "attention_mask": "bytes", "labels": "bytes"}
    with MDSWriter(out=out_dir, columns=columns, compression=compression) as writer:
        count = 0
        for example in ds:
            for key in [text_key, "text", "content"]:
                if key in example and isinstance(example[key], str) and len(example[key]) > 10:
                    text = example[key]
                    break
            else:
                continue

            enc = tok(text, truncation=True, max_length=2048, padding="max_length", return_tensors="pt")
            ids = enc["input_ids"].squeeze(0).numpy().astype(np.uint32)
            mask = enc["attention_mask"].squeeze(0).numpy().astype(np.uint8)

            writer.write({
                "input_ids": ids.tobytes(),
                "attention_mask": mask.tobytes(),
                "labels": ids.tobytes(),
            })
            count += 1
            if count % batch_size == 0:
                print(f"  Converted {count} docs...")
            if 0 < max_docs <= count:
                break

    print(f"Done: {count} docs -> {out_dir}")
    return out_dir


class MDSMeshDataset(IterableDataset):
    """Stream pre-tokenized MDS shards at maximum GPU throughput."""
    def __init__(
        self,
        local_dir: str,
        max_seq_len: int = 2048,
        shuffle: bool = True,
        repeat: bool = True,
    ):
        self.local_dir = local_dir
        self.max_seq_len = max_seq_len
        self.shuffle = shuffle
        self.repeat = repeat
        self._dataset = None

    def _init_dataset(self):
        if not HAS_MOSAICML:
            raise RuntimeError("mosaicml-streaming not installed. Run: pip install mosaicml-streaming")
        self._dataset = StreamingDataset(
            local=self.local_dir,
            shuffle=self.shuffle,
            batch_size=1,
        )

    def __iter__(self):
        if self._dataset is None:
            self._init_dataset()
        while True:
            for idx in range(len(self._dataset)):
                sample = self._dataset[idx]
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


class MixedMDSDataset(IterableDataset):
    """Interleave multiple MDS datasets with weights."""
    def __init__(
        self,
        sources: list[dict],
        max_seq_len: int = 2048,
    ):
        self.datasets = []
        self.weights = []
        for src in sources:
            ds = MDSMeshDataset(
                local_dir=src["local_dir"],
                max_seq_len=max_seq_len,
                shuffle=src.get("shuffle", True),
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


def convert_fineweb_edu_mds(out_dir: str = "./mds_data/fineweb_edu", max_docs: int = -1):
    """Convert FineWeb-Edu to MDS."""
    return convert_to_mds(
        hf_path="HuggingFaceFW/fineweb-edu",
        out_dir=out_dir,
        max_docs=max_docs,
    )


def convert_fineweb2_ar_mds(out_dir: str = "./mds_data/fineweb2_ar", max_docs: int = -1):
    """Convert FineWeb-2 Arabic to MDS."""
    return convert_to_mds(
        hf_path="HuggingFaceFW/fineweb-2",
        out_dir=out_dir,
        hf_config="ara_Arab",
        text_key="text",
        max_docs=max_docs,
    )


def convert_fineweb2_en_mds(out_dir: str = "./mds_data/fineweb2_en", max_docs: int = -1):
    """Convert FineWeb-2 English to MDS."""
    return convert_to_mds(
        hf_path="HuggingFaceFW/fineweb-2",
        out_dir=out_dir,
        hf_config="eng_Latn",
        text_key="text",
        max_docs=max_docs,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["fineweb-edu", "fineweb2-ar", "fineweb2-en", "all"], default="all")
    parser.add_argument("--out-dir", default="./mds_data")
    parser.add_argument("--max-docs", type=int, default=-1)
    args = parser.parse_args()

    tasks = []
    if args.dataset in ("fineweb-edu", "all"):
        tasks.append((convert_fineweb_edu_mds, f"{args.out_dir}/fineweb_edu"))
    if args.dataset in ("fineweb2-ar", "all"):
        tasks.append((convert_fineweb2_ar_mds, f"{args.out_dir}/fineweb2_ar"))
    if args.dataset in ("fineweb2-en", "all"):
        tasks.append((convert_fineweb2_en_mds, f"{args.out_dir}/fineweb2_en"))

    for fn, out_dir in tasks:
        print(f"\nConverting {fn.__name__} to {out_dir}...")
        fn(out_dir, max_docs=args.max_docs)
