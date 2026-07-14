"""
Direct streaming from HuggingFace Datasets Server API.
No `datasets` library needed — pure HTTP, no parquet downloads.
"""
import json
import random
import requests
import torch
from torch.utils.data import IterableDataset
from mesh_tokenizer import load_tokenizer


API_BASE = "https://datasets-server.huggingface.co/rows"

DATASETS = {
    "fineweb-edu": {
        "dataset": "HuggingFaceFW/fineweb-edu",
        "config": "default",
        "split": "train",
        "total_rows": 13440000,
        "text_key": "text",
        "weight": 3.0,
        "domain": "general",
    },
    "open-orca": {
        "dataset": "Open-Orca/OpenOrca",
        "config": "default",
        "split": "train",
        "total_rows": 4200000,
        "text_key": "question",  # needs conversation formatting
        "weight": 4.0,
        "domain": "reasoning",
    },
}


class DirectStreamDataset(IterableDataset):
    """Streams HF dataset rows via Datasets Server API — zero downloads."""
    def __init__(
        self,
        source: str = "fineweb-edu",
        max_seq_len: int = 512,
        start_offset: int = 0,
        tokenizer=None,
        shuffle_buffer: int = 1000,
    ):
        self.cfg = DATASETS.get(source, DATASETS["fineweb-edu"])
        self.max_seq_len = max_seq_len
        self.offset = start_offset
        self.tokenizer = tokenizer or load_tokenizer()
        self.shuffle_buffer = shuffle_buffer
        self.text_key = self.cfg["text_key"]
        self.total = self.cfg["total_rows"]
        self.step = 100  # fetch 100 rows per API call

    def _fetch_batch(self, offset: int) -> list[dict]:
        url = f"{API_BASE}?dataset={self.cfg['dataset']}&config={self.cfg['config']}&split={self.cfg['split']}&offset={offset}&length={self.step}"
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return data.get("rows", [])
        except Exception as e:
            print(f"API error at offset {offset}: {e}")
            return []

    def _format_row(self, row: dict) -> str | None:
        """Extract text from a dataset row, handling different formats."""
        row_data = row.get("row", {})
        if not row_data:
            return None
        text = row_data.get(self.text_key, "")
        if not text:
            return None
        return str(text)

    def __iter__(self):
        offset = self.offset
        buffer = []
        while offset < self.total:
            rows = self._fetch_batch(offset)
            if not rows:
                offset += self.step
                continue
            for row in rows:
                text = self._format_row(row)
                if not text:
                    continue
                enc = self.tokenizer(
                    text, truncation=True, max_length=self.max_seq_len,
                    padding="max_length", return_tensors="pt",
                )
                buffer.append({
                    "input_ids": enc["input_ids"].squeeze(0),
                    "labels": enc["input_ids"].squeeze(0).clone(),
                    "domain": self.cfg.get("domain", "general"),
                })
                if len(buffer) >= self.shuffle_buffer:
                    random.shuffle(buffer)
                    for item in buffer:
                        yield item
                    buffer = []
            offset += self.step
        for item in buffer:
            yield item


def collate_fn(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    t = torch.linspace(0, 1, input_ids.size(0)).unsqueeze(-1).float()
    if "domain" in batch[0]:
        domain_ids = torch.tensor([["general", "reasoning", "tool_use", "planning"].index(b.get("domain", "general")) for b in batch])
        return input_ids, labels, t, None, domain_ids
    return input_ids, labels, t, None
