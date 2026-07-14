"""
AGK Dataset Adapter — reads markdown problem files, tokenizes with Qwen3.
"""
import os
import glob
import torch
from torch.utils.data import Dataset
from mesh_tokenizer import load_tokenizer, VOCAB_SIZE

DOMAIN_WEIGHTS = {
    "reasoning": 4.0,
    "language_grammar": 3.0,
    "coding": 2.0,
    "maths": 3.0,
    "physics": 3.0,
}

DOMAIN_ORDER = ["reasoning", "language_grammar", "coding", "maths", "physics"]


def _phase_to_domain(phase_dir: str) -> str:
    lower = phase_dir.lower()
    if "reasoning" in lower:
        return "reasoning"
    if "language" in lower or "grammar" in lower:
        return "language_grammar"
    if "coding" in lower:
        return "coding"
    if "math" in lower:
        return "maths"
    if "physics" in lower:
        return "physics"
    return "general"


class AGKDataset(Dataset):
    """Tokenizes .md files on-the-fly using Qwen3 tokenizer."""
    def __init__(self, agk_dir: str, max_seq_len: int = 512,
                 max_files: int | None = None, tokenizer=None):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer or load_tokenizer()
        self.num_classes = 5
        self.samples: list[dict] = []

        phase_dirs = sorted(glob.glob(os.path.join(agk_dir, "phase*")))
        if not phase_dirs:
            raise FileNotFoundError(f"No phase directories found in {agk_dir}")

        for phase_dir in phase_dirs:
            domain = _phase_to_domain(os.path.basename(phase_dir))
            md_files = sorted(glob.glob(os.path.join(phase_dir, "*.md")))
            if max_files is not None:
                md_files = md_files[:max_files]
            for fpath in md_files:
                with open(fpath, encoding="utf-8") as f:
                    text = f.read()
                self.samples.append({"text": text, "domain": domain, "path": fpath})

        if not self.samples:
            raise ValueError(f"No .md files found in {agk_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        text = sample["text"]
        domain = sample["domain"]

        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_len,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        labels = input_ids.clone()

        domain_idx = DOMAIN_ORDER.index(domain) if domain in DOMAIN_ORDER else 0
        oh = torch.nn.functional.one_hot(torch.tensor(domain_idx), self.num_classes).float()

        domain_weight = DOMAIN_WEIGHTS.get(domain, 1.0)
        t = torch.tensor([domain_weight / 5.0])

        return input_ids, labels, t
