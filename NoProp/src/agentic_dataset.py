"""
Agentic Dataset — domain-aware streaming with reasoning, tool-use, planning datasets.
Wraps HF datasets into chat format, detects domain for router routing.

Sources and their domain labels:
  - HuggingFaceFW/fineweb-edu          → general (3x weight)
  - Open-Orca/OpenOrca                 → reasoning (4x)
  - glaiveai/function-calling-v1       → tool_use (2x)
  - totally-not-an-llm/agentic-v0.1    → planning (2x)
"""
import random
import torch
from torch.utils.data import IterableDataset
from mesh_tokenizer import load_tokenizer
from thinking_utils import format_chat, detect_domain


# Source configurations with domain weights
DEFAULT_SOURCES = [
    {
        "hf_path": "HuggingFaceFW/fineweb-edu",
        "split": "train",
        "text_key": "text",
        "domain": "general",
        "weight": 3.0,
        "formatter": None,  # raw text, already chat-like
    },
    {
        "hf_path": "Open-Orca/OpenOrca",
        "split": "train",
        "text_key": "conversations",
        "domain": "reasoning",
        "weight": 4.0,
        "formatter": "openorca",  # special formatter
    },
    {
        "hf_path": "glaiveai/function-calling-v1",
        "split": "train",
        "text_key": "conversations",
        "domain": "tool_use",
        "weight": 2.0,
        "formatter": "glaive",
    },
    {
        "hf_path": "totally-not-an-llm/thinking",
        "split": "train",
        "text_key": "text",
        "domain": "reasoning",
        "weight": 2.0,
        "formatter": None,
    },
]


def _format_openorca(conversations) -> str:
    """Convert OpenOrca conversation list to chat format."""
    if not isinstance(conversations, list):
        return str(conversations)
    messages = []
    for turn in conversations:
        if isinstance(turn, dict):
            role = turn.get("from", "human")
            value = turn.get("value", "")
            if role == "human":
                messages.append({"role": "user", "content": value})
            elif role in ("gpt", "assistant"):
                messages.append({"role": "assistant", "content": value})
    return format_chat(messages) if messages else ""


def _format_glaive(conversations) -> str:
    """Convert Glaive function-calling conversation to chat format."""
    if not isinstance(conversations, list):
        return str(conversations)
    messages = []
    for turn in conversations:
        if isinstance(turn, dict):
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if role == "function":
                messages.append({"role": "tool", "content": content})
            else:
                messages.append({"role": role, "content": content})
    return format_chat(messages) if messages else ""


_FORMATTERS = {
    "openorca": _format_openorca,
    "glaive": _format_glaive,
}


class AgenticStreamingDataset(IterableDataset):
    """Multi-source streaming dataset with domain labels for router routing."""
    def __init__(
        self,
        sources: list[dict] | None = None,
        max_seq_len: int = 512,
        tokenizer=None,
        shuffle_buffer: int = 5000,
    ):
        self.sources = sources or DEFAULT_SOURCES
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer or load_tokenizer()
        self.shuffle_buffer = shuffle_buffer
        self._iterators = {}
        self._weights = [s["weight"] for s in self.sources]

    def _get_iter(self, source_idx: int):
        """Get or create iterator for a source."""
        if source_idx not in self._iterators:
            src = self.sources[source_idx]
            from datasets import load_dataset
            ds = load_dataset(src["hf_path"], split=src["split"], streaming=True, trust_remote_code=True)
            self._iterators[source_idx] = iter(ds)
        return self._iterators[source_idx]

    def _process_sample(self, example: dict, source_idx: int) -> dict | None:
        src = self.sources[source_idx]
        text_key = src["text_key"]
        formatter_name = src.get("formatter")
        domain = src["domain"]

        raw = example.get(text_key, "")
        if not raw:
            return None

        if formatter_name and formatter_name in _FORMATTERS:
            text = _FORMATTERS[formatter_name](raw)
        elif isinstance(raw, list):
            text = " ".join(str(x) for x in raw)
        else:
            text = str(raw)

        if not text.strip():
            return None

        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_len,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": enc["input_ids"].squeeze(0).clone(),
            "domain": domain,
            "domain_id": ["general", "reasoning", "tool_use", "planning"].index(domain),
        }

    def __iter__(self):
        buffer = []
        while True:
            source_idx = random.choices(range(len(self.sources)), weights=self._weights, k=1)[0]
            try:
                it = self._get_iter(source_idx)
                example = next(it)
            except StopIteration:
                self._iterators[source_idx] = None
                continue

            sample = self._process_sample(example, source_idx)
            if sample is None:
                continue

            buffer.append(sample)
            if len(buffer) >= self.shuffle_buffer:
                random.shuffle(buffer)
                for item in buffer:
                    yield item
                buffer = []

        for item in buffer:
            yield item


def collate_fn(batch):
    """Collate with domain labels for router routing."""
    input_ids = torch.stack([b["input_ids"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch]) if batch[0].get("attention_mask") is not None else None
    t = torch.linspace(0, 1, input_ids.size(0)).unsqueeze(-1).float()
    domains = [b.get("domain", "general") for b in batch]
    domain_ids = torch.tensor([b.get("domain_id", 0) for b in batch])
    return input_ids, labels, t, attention_mask, domains, domain_ids
