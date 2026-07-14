import os
import json
import torch
import struct
from typing import Any
try:
    from mesh_tokenizer import TOKENIZER_NAME
except ImportError:
    TOKENIZER_NAME = "unknown"

GGUF_MAGIC = 0x46554747


def export_to_safetensors(
    mesh_state: dict,
    output_path: str,
    metadata: dict | None = None,
):
    try:
        from safetensors.torch import save_file
        tensors = {}
        for node_id, node_state in mesh_state.get("router_state", {}).items():
            model_sd = node_state.get("model", {})
            for k, v in model_sd.items():
                tensors[f"{node_id}.{k}"] = v.contiguous().cpu()
            anchor = node_state.get("anchor")
            if anchor is not None:
                tensors[f"{node_id}.anchor"] = anchor.contiguous().cpu()
        meta = {
            "format": "noprop-mesh-safetensors",
            "node_count": str(len(mesh_state.get("router_state", {}))),
            "step": str(mesh_state.get("step", 0)),
            "tokenizer": TOKENIZER_NAME,
        }
        if metadata:
            meta.update({k: str(v) for k, v in metadata.items()})
        save_file(tensors, output_path, meta)
        return True
    except ImportError:
        return _export_fallback_pytorch(mesh_state, output_path, metadata)


def _export_fallback_pytorch(mesh_state: dict, output_path: str, metadata: dict | None = None):
    state = {
        "mesh_state": mesh_state,
        "metadata": metadata or {},
    }
    torch.save(state, output_path)
    return False


def export_to_onnx(
    block: torch.nn.Module,
    output_path: str,
    embed_dim: int = 768,
    opset_version: int = 17,
):
    block.eval()
    dummy_x = torch.randn(1, embed_dim)
    dummy_t = torch.tensor([[0.5]])
    with torch.no_grad():
        torch.onnx.export(
            block,
            (dummy_x, dummy_t),
            output_path,
            input_names=["x", "t"],
            output_names=["output"],
            dynamic_axes={
                "x": {0: "batch_size"},
                "t": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
            opset_version=opset_version,
        )
    return True


GGUF_MAGIC = 0x46554747


def _gguf_type_from_tensor(t: torch.Tensor) -> int:
    dtypes = {
        torch.float32: 0,
        torch.float16: 1,
        torch.bfloat16: 25,
        torch.int8: 26,
        torch.int16: 27,
        torch.int32: 28,
        torch.int64: 29,
    }
    return dtypes.get(t.dtype, 0)


def export_to_gguf(
    mesh_state: dict,
    output_path: str,
    metadata: dict | None = None,
    external_nodes_dir: str | None = None,
):
    metadata = metadata or {}
    router_state = mesh_state.get("router_state", {})
    node_ids = list(router_state.keys())
    num_nodes = len(node_ids)

    header_kv = {
        "general.architecture": "noprop-mesh",
        "general.name": metadata.get("name", "noprop-mesh-model"),
        "general.tokenizer": TOKENIZER_NAME,
        "noprop.num_nodes": num_nodes,
        "noprop.embed_dim": metadata.get("embed_dim", 768),
        "noprop.top_k": metadata.get("top_k", 2),
        "noprop.mitosis_threshold": metadata.get("mitosis_threshold", 0.5),
        "noprop.external_nodes": str(external_nodes_dir is not None),
    }

    tensor_data = []
    for nid in node_ids:
        ns = router_state[nid]
        model_sd = ns.get("model", {})
        # If external nodes, load from disk
        if not model_sd and external_nodes_dir is not None:
            ext_path = os.path.join(external_nodes_dir, nid, "block.pt")
            if os.path.exists(ext_path):
                model_sd = torch.load(ext_path, weights_only=True, map_location="cpu")
        for k, v in model_sd.items():
            tensor_data.append((f"{nid}.{k}", v.contiguous().cpu()))
        anchor = ns.get("anchor")
        if anchor is not None:
            tensor_data.append((f"{nid}.anchor", anchor.contiguous().cpu()))

    with open(output_path, "wb") as f:
        f.write(struct.pack("<I", GGUF_MAGIC))
        f.write(struct.pack("<I", 3))
        f.write(struct.pack("<Q", len(header_kv)))
        for k, v in header_kv.items():
            _write_gguf_key_value(f, k, v)
        f.write(struct.pack("<Q", len(tensor_data)))
        offset = 0
        info_entries = []
        for name, tensor in tensor_data:
            nbytes = tensor.element_size() * tensor.numel()
            shape = list(tensor.shape)
            info_entries.append((name, shape, _gguf_type_from_tensor(tensor), offset, nbytes))
            offset += nbytes
        f.write(struct.pack("<Q", 0))
        for name, shape, dtype, off, nbytes in info_entries:
            _write_gguf_tensor_info(f, name, shape, dtype, off)
        for name, tensor in tensor_data:
            f.write(tensor.numpy().tobytes())

    return len(tensor_data)


def _write_gguf_key_value(f, key: str, value: Any):
    key_bytes = key.encode("utf-8")
    f.write(struct.pack("<Q", len(key_bytes)))
    f.write(key_bytes)
    if isinstance(value, str):
        f.write(struct.pack("<I", 8))
        val_bytes = value.encode("utf-8")
        f.write(struct.pack("<Q", len(val_bytes)))
        f.write(val_bytes)
    elif isinstance(value, (int, float)):
        f.write(struct.pack("<I", 4 if isinstance(value, int) else 5))
        f.write(struct.pack("<I" if isinstance(value, int) else "<f", value))
    else:
        f.write(struct.pack("<I", 8))
        val_bytes = str(value).encode("utf-8")
        f.write(struct.pack("<Q", len(val_bytes)))
        f.write(val_bytes)


def _write_gguf_tensor_info(f, name: str, shape: list[int], dtype: int, offset: int):
    name_bytes = name.encode("utf-8")
    f.write(struct.pack("<Q", len(name_bytes)))
    f.write(name_bytes)
    f.write(struct.pack("<I", len(shape)))
    for s in shape:
        f.write(struct.pack("<Q", s))
    f.write(struct.pack("<I", dtype))
    f.write(struct.pack("<Q", offset))


def extract_experts_from_gguf(gguf_path: str, output_dir: str):
    """Extract expert weights from a GGUF file back to external node files."""
    from external_nodes import save_expert_block
    from noprop_block import NoPropBlock
    ckpt = torch.load(gguf_path, weights_only=True, map_location="cpu")
    expert_tensors = {}
    for key, tensor in ckpt.items():
        parts = key.split(".")
        if len(parts) >= 2:
            node_id = parts[0]
            param_name = ".".join(parts[1:])
            if node_id not in expert_tensors:
                expert_tensors[node_id] = {}
            expert_tensors[node_id][param_name] = tensor
    for node_id, params in expert_tensors.items():
        embed_dim = params.get("norm1.weight", params.get("attn.in_proj_weight", torch.zeros(1))).shape[0]
        block = NoPropBlock(embed_dim, num_heads=4)
        block.load_state_dict(params)
        save_expert_block(block, output_dir, node_id)
        print(f"Extracted expert {node_id} to {output_dir}/{node_id}/block.pt")
