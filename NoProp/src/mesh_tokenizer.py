"""
Mesh Tokenizer — wraps Qwen3 tokenizer for real text I/O.
Weight-tying friendly: tokenizer maps text→ids, model maps ids→text.
"""
import torch
from transformers import AutoTokenizer

TOKENIZER_NAME = "Qwen/Qwen3-4B"
PAD_TOKEN = "<|endoftext|>"
EOS_TOKEN = "<|im_end|>"
BOS_TOKEN = "<|im_start|>"


def load_tokenizer() -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = PAD_TOKEN
    tok.padding_side = "left"
    return tok


def get_vocab_size(tok: AutoTokenizer | None = None) -> int:
    if tok is None:
        tok = load_tokenizer()
    max_id = max(tok.added_tokens_decoder.keys()) if tok.added_tokens_decoder else tok.vocab_size - 1
    return max(max_id, tok.vocab_size - 1) + 1


VOCAB_SIZE = get_vocab_size()


def encode(tok: AutoTokenizer, text: str, max_len: int = 512) -> dict:
    enc = tok(
        text,
        truncation=True,
        max_length=max_len,
        padding="max_length",
        return_tensors="pt",
    )
    return {"input_ids": enc["input_ids"].squeeze(0), "attention_mask": enc["attention_mask"].squeeze(0)}


def decode(tok: AutoTokenizer, token_ids: torch.Tensor) -> str:
    if token_ids.dim() > 1:
        token_ids = token_ids[0]
    return tok.decode(token_ids.tolist(), skip_special_tokens=True)
