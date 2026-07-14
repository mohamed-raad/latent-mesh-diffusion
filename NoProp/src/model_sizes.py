"""
Model size presets for the Diffusion Mesh backbone.

Usage:
    from config.model_sizes import get_preset, SizePreset
    cfg = get_preset("small")   # 500M — recommended for RTX 5060 8GB
"""
from dataclasses import dataclass


@dataclass
class SizePreset:
    name: str
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    d_ff: int
    max_seq_len: int
    num_experts: int
    param_estimate: str


PRESETS = {
    "tiny": SizePreset(
        name="tiny",
        d_model=768,
        n_layers=8,
        n_heads=12,
        n_kv_heads=4,
        d_ff=2048,
        max_seq_len=4096,
        num_experts=8,
        param_estimate="250M",
    ),
    "small": SizePreset(
        name="small",
        d_model=1024,
        n_layers=12,
        n_heads=16,
        n_kv_heads=4,
        d_ff=4096,
        max_seq_len=8192,
        num_experts=16,
        param_estimate="500M",
    ),
    "standard": SizePreset(
        name="standard",
        d_model=1536,
        n_layers=16,
        n_heads=24,
        n_kv_heads=8,
        d_ff=6144,
        max_seq_len=16384,
        num_experts=32,
        param_estimate="1.0B",
    ),
    "xstandard": SizePreset(
        name="xstandard",
        d_model=1792,
        n_layers=20,
        n_heads=28,
        n_kv_heads=8,
        d_ff=7168,
        max_seq_len=16384,
        num_experts=48,
        param_estimate="1.5B",
    ),
    "large": SizePreset(
        name="large",
        d_model=2048,
        n_layers=24,
        n_heads=32,
        n_kv_heads=8,
        d_ff=8192,
        max_seq_len=32768,
        num_experts=64,
        param_estimate="2.0B",
    ),
}


def get_preset(name: str) -> SizePreset:
    if name not in PRESETS:
        valid = list(PRESETS.keys())
        raise ValueError(f"Unknown preset '{name}'. Valid: {valid}")
    return PRESETS[name]


def list_presets() -> list[str]:
    return list(PRESETS.keys())
