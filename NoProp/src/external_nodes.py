"""
External Nodes — each expert stored as a separate file on disk.
Main checkpoint stores only routing graph + metadata.
Lazy loading: weights fetched from disk only when expert is activated.
"""
import os
import json
import torch
from noprop_block import NoPropBlock, inject_lora_into_block


def _node_dir(base_dir: str, node_id: str) -> str:
    return os.path.join(base_dir, node_id)


def _block_path(base_dir: str, node_id: str) -> str:
    return os.path.join(_node_dir(base_dir, node_id), "block.pt")


def _meta_path(base_dir: str, node_id: str) -> str:
    return os.path.join(_node_dir(base_dir, node_id), "meta.json")


def save_expert_block(block: NoPropBlock, base_dir: str, node_id: str):
    os.makedirs(_node_dir(base_dir, node_id), exist_ok=True)
    torch.save(block.state_dict(), _block_path(base_dir, node_id))


def load_expert_block(block: NoPropBlock, base_dir: str, node_id: str) -> NoPropBlock:
    path = _block_path(base_dir, node_id)
    if os.path.exists(path):
        block.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
    return block


def save_expert_meta(base_dir: str, node_id: str, meta: dict):
    os.makedirs(_node_dir(base_dir, node_id), exist_ok=True)
    with open(_meta_path(base_dir, node_id), "w") as f:
        json.dump(meta, f)


def load_expert_meta(base_dir: str, node_id: str) -> dict:
    path = _meta_path(base_dir, node_id)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def delete_expert(base_dir: str, node_id: str):
    import shutil
    path = _node_dir(base_dir, node_id)
    if os.path.exists(path):
        shutil.rmtree(path)


def list_experts(base_dir: str) -> list[str]:
    if not os.path.isdir(base_dir):
        return []
    return [d for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d)) and os.path.exists(
                _block_path(base_dir, d))]
