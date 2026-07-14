"""Text pipeline: tokenizer, text→training data, chat interface for the mesh."""
import os
import glob
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class TextMeshPipeline:
    def __init__(self, model_name: str = "gpt2", max_length: int = 256):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or "<|pad|>"
        self.vocab_size = self.tokenizer.vocab_size

    def encode(self, text: str) -> torch.Tensor:
        return self.tokenizer.encode(text, return_tensors="pt")

    def decode(self, token_ids: torch.Tensor) -> str:
        return self.tokenizer.decode(token_ids.squeeze().tolist(), skip_special_tokens=True)

    def load_text_files(self, data_dir: str) -> list[str]:
        texts = []
        for ext in ("*.txt", "*.md", "*.json", "*.py", "*.js", "*.html", "*.csv"):
            for fp in glob.glob(os.path.join(data_dir, "**", ext), recursive=True):
                try:
                    with open(fp, encoding="utf-8", errors="replace") as f:
                        content = f.read()
                    if len(content.strip()) > 50:
                        texts.append(content)
                except OSError:
                    pass
        return texts

    def chunk_texts(self, texts: list[str], stride: int | None = None) -> torch.Tensor:
        if stride is None:
            stride = self.max_length // 2
        all_chunks = []
        for text in texts:
            ids = self.encode(text).squeeze(0)
            if ids.numel() == 0:
                continue
            for start in range(0, ids.numel(), stride):
                chunk = ids[start:start + self.max_length]
                if chunk.numel() < 10:
                    continue
                if chunk.numel() < self.max_length:
                    pad_len = self.max_length - chunk.numel()
                    chunk = F.pad(chunk, (0, pad_len), value=self.tokenizer.pad_token_id)
                all_chunks.append(chunk)
        return torch.stack(all_chunks) if all_chunks else torch.zeros(0, self.max_length, dtype=torch.long)


class TextDataset(Dataset):
    def __init__(self, token_chunks: torch.Tensor, embed_dim: int = 128, vocab_size: int = 50257):
        self.tokens = token_chunks
        self.embed_dim = embed_dim
        self.embedding = torch.nn.Embedding(
            min(int(token_chunks.max().item()) + 2, vocab_size), embed_dim, padding_idx=0
        )

    def __len__(self):
        return self.tokens.size(0)

    def __getitem__(self, idx):
        ids = self.tokens[idx]
        emb = self.embedding(ids).mean(dim=0).detach()
        t = torch.full((1,), 0.5, dtype=torch.float)
        return emb.clone(), emb.clone(), t.clone()


class ChatInterface:
    def __init__(self, pipeline: TextMeshPipeline, trainer):
        self.pipeline = pipeline
        self.trainer = trainer
        self.history: list[dict] = []

    def reply(self, user_msg: str, max_new_tokens: int = 64) -> str:
        prompt = user_msg.strip()
        prompt_ids = self.pipeline.encode(prompt)
        prompt_len = prompt_ids.size(1)

        out = self.trainer.chat(prompt_ids, max_new_tokens=max_new_tokens)
        generated = out[:, prompt_len:]
        generated = generated[generated != self.pipeline.tokenizer.pad_token_id]
        text = self.pipeline.tokenizer.decode(generated.tolist(), skip_special_tokens=True)
        self.history.append({"user": user_msg, "bot": text})
        return text

    def train_on_texts(self, texts: list[str], trainer, num_epochs: int = 5, batch_size: int = 8):
        chunks = self.pipeline.chunk_texts(texts)
        dataset = TextDataset(chunks)
        original_train = trainer.train
        trainer.train(dataset=dataset, num_epochs=num_epochs, batch_size=batch_size,
                      log_interval=50, mitosis_interval=200, ckpt_interval=500)
