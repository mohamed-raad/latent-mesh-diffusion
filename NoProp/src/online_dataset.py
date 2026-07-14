"""
Online Dataset — streams HuggingFace datasets tokenized on-the-fly.
No disk download: uses HF `streaming=True` + on-the-fly Qwen3 tokenization.
"""
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
from torch.utils.data import IterableDataset
from mesh_tokenizer import load_tokenizer, VOCAB_SIZE


class StreamingHFDataset(IterableDataset):
    """Streams a HuggingFace dataset without downloading to disk."""
    def __init__(
        self,
        hf_path: str = "HuggingFaceFW/fineweb-edu",
        hf_config: str | None = None,
        split: str = "train",
        max_seq_len: int = 512,
        text_key: str = "text",
        tokenizer=None,
        shuffle_buffer: int = 10000,
    ):
        from datasets import load_dataset
        self.max_seq_len = max_seq_len
        self.text_key = text_key
        self.tokenizer = tokenizer or load_tokenizer()
        self.shuffle_buffer = shuffle_buffer
        kw = {"split": split, "streaming": True}
        if hf_config is not None:
            kw["name"] = hf_config
        self.dataset = load_dataset(hf_path, **kw)

    def _resolve_text_key(self, example: dict) -> str | None:
        """Try the configured text_key, fall back to common names."""
        for key in [self.text_key, "text", "content", "prompt", "response", "code"]:
            if key in example and isinstance(example[key], str) and len(example[key]) > 10:
                return example[key]
        return None

    def __iter__(self):
        buffer = []
        for example in self.dataset:
            text = self._resolve_text_key(example)
            if text is None:
                continue
            enc = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_seq_len,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].squeeze(0)
            mask = enc["attention_mask"].squeeze(0)
            buffer.append({"input_ids": input_ids, "attention_mask": mask, "labels": input_ids.clone()})
            if len(buffer) >= self.shuffle_buffer:
                import random
                random.shuffle(buffer)
                for item in buffer:
                    yield item
                buffer = []
        for item in buffer:
            yield item


class MixedOnlineDataset(IterableDataset):
    """Interleaves multiple streaming datasets with mixing weights."""
    def __init__(
        self,
        sources: list[dict],
        max_seq_len: int = 512,
        tokenizer=None,
    ):
        """
        sources: list of dicts with keys: hf_path, split, text_key, weight
        """
        self.tokenizer = tokenizer or load_tokenizer()
        self.max_seq_len = max_seq_len
        self.datasets = []
        self.weights = []
        for src in sources:
            ds = StreamingHFDataset(
                hf_path=src["hf_path"],
                hf_config=src.get("hf_config"),
                split=src.get("split", "train"),
                max_seq_len=max_seq_len,
                text_key=src.get("text_key", "text"),
                tokenizer=self.tokenizer,
            )
            self.datasets.append(ds)
            self.weights.append(src.get("weight", 1.0))
        self.total_weight = sum(self.weights)

    def __iter__(self):
        iters = [iter(ds) for ds in self.datasets]
        import random
        while True:
            source_idx = random.choices(range(len(iters)), weights=self.weights, k=1)[0]
            try:
                yield next(iters[source_idx])
            except StopIteration:
                iters[source_idx] = iter(self.datasets[source_idx])
                yield next(iters[source_idx])


def collate_fn(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch]) if batch[0].get("attention_mask") is not None else None
    labels = torch.stack([b["labels"] for b in batch]) if batch[0].get("labels") is not None else input_ids.clone()
    t = torch.linspace(0, 1, input_ids.size(0)).unsqueeze(-1).float()
    return input_ids, labels, t, attention_mask
