"""
Unified streaming — auto-selects best available backend:
1. Fast binary shards (preferred, no HF cache)
2. HF streaming (fallback)
"""
import os
from torch.utils.data import IterableDataset
from online_dataset import MixedOnlineDataset, collate_fn
from bin_converter import BinReader


class BinaryDataset(IterableDataset):
    """Stream from pre-converted binary shards. Zero HF cache."""
    def __init__(self, local_dir: str, max_seq_len: int = 2048, repeat: bool = True):
        self.reader = BinReader(local_dir)
        self.max_seq_len = max_seq_len
        self.repeat = repeat

    def __iter__(self):
        while True:
            for sample in self.reader.iter_all():
                if len(sample["input_ids"]) > self.max_seq_len:
                    sample["input_ids"] = sample["input_ids"][:self.max_seq_len]
                    sample["attention_mask"] = sample["attention_mask"][:self.max_seq_len]
                    sample["labels"] = sample["labels"][:self.max_seq_len]
                yield sample
            if not self.repeat:
                break


class MixedBinaryDataset(IterableDataset):
    """Interleave multiple binary datasets with weights."""
    def __init__(self, sources: list[dict], max_seq_len: int = 2048):
        self.datasets = []
        self.weights = []
        for src in sources:
            ds = BinaryDataset(src["local_dir"], max_seq_len=max_seq_len)
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


def create_mixed_dataset(sources: list[dict], max_seq_len: int = 2048) -> IterableDataset:
    """Create best available dataset (prefers binary)."""
    bin_sources = [s for s in sources if "local_dir" in s and os.path.isdir(s["local_dir"])]
    hf_sources = [s for s in sources if "hf_path" in s]

    if bin_sources and not hf_sources:
        return MixedBinaryDataset(sources=bin_sources, max_seq_len=max_seq_len)
    return MixedOnlineDataset(sources=sources, max_seq_len=max_seq_len)
