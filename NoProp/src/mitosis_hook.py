"""
Adaptive Node Scaler — smart mitosis that decides when to create new mesh nodes
vs update existing nodes based on novelty, confidence, and accuracy.

- High similarity + high confidence → update existing node (fine-tune)
- Medium similarity + low confidence → create new specialized node
- Low similarity (novel) → create new node (mitosis)
"""
import os
import torch
import torch.nn.functional as F
from mesh_router import MeshNode
from noprop_block import NoPropBlock, inject_lora_into_block


class MitosisHook:
    def __init__(
        self,
        router,
        embed_dim: int = 768,
        num_heads: int = 4,
        lora_rank: int = 16,
        lora_alpha: float = 16.0,
        lr: float = 1e-3,
        oov_threshold: float = 0.20,
        confidence_threshold: float = 0.40,
        update_threshold: float = 0.70,
        nodes_dir: str = "nodes",
        max_nodes: int = 512,
    ):
        self.router = router
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lr = lr
        self.oov_threshold = oov_threshold
        self.confidence_threshold = confidence_threshold
        self.update_threshold = update_threshold
        self.nodes_dir = nodes_dir
        self.max_nodes = max_nodes

        os.makedirs(nodes_dir, exist_ok=True)

        self.spawned_count: int = 0
        self.updated_count: int = 0
        self.hook_calls: int = 0
        self.spawned_ids: list[str] = []

    def evaluate_query(self, query: torch.Tensor) -> tuple[torch.Tensor, str | None, float]:
        if self.router.anchor_tensor is None or self.router.anchor_tensor.size(0) == 0:
            return torch.tensor([]), None, 0.0
        q = F.normalize(query, dim=-1)
        anchors = F.normalize(self.router.anchor_tensor.to(q.device), dim=-1)
        sims = q @ anchors.T
        max_sim, argmax = sims.squeeze(0).max(dim=0)
        max_sim_val = max_sim.item()
        nearest_id = self.router.node_ids[argmax.item()]
        return sims.squeeze(0), nearest_id, max_sim_val

    def _node_confidence(self, node: MeshNode) -> float:
        if not node.rolling_loss:
            return 0.5
        recent = torch.tensor(node.rolling_loss[-min(50, len(node.rolling_loss)):])
        avg_loss = recent.mean().item()
        confidence = 1.0 - min(avg_loss / 2.0, 1.0)
        return max(0.1, min(0.95, confidence))

    def should_mitose(self, max_similarity: float, nearest_node=None) -> str:
        """
        Returns: 'create' (mitosis), 'update' (fine-tune), or 'skip'
        """
        if len(self.router.nodes) >= self.max_nodes:
            return 'skip'

        if max_similarity < self.oov_threshold:
            return 'create'

        if max_similarity >= self.update_threshold:
            conf = self._node_confidence(nearest_node) if nearest_node else 0.5
            if conf >= self.confidence_threshold:
                return 'update'
            return 'create'

        if max_similarity < self.confidence_threshold:
            return 'create'

        return 'skip'

    def spawn_node(self, query: torch.Tensor, nearest_node_id: str) -> str:
        parent_node = self.router.nodes.get(nearest_node_id)
        if parent_node is None:
            parent_idx = 0
        else:
            existing = [k for k in self.router.nodes if k.startswith("mitosed_")]
            parent_idx = len(existing) + self.spawned_count

        child_id = f"mitosed_{nearest_node_id}_{parent_idx}"
        child_path = os.path.join(self.nodes_dir, f"{child_id}.md")
        with open(child_path, "w") as f:
            f.write(f"# {child_id}\n\n")
            f.write(f"Auto-mitosed from `{nearest_node_id}`\n")
            f.write(f"Query norm: {query.norm().item():.4f}\n")

        child_anchor = F.normalize(query.squeeze().detach().cpu(), dim=-1)
        child = MeshNode(
            node_id=child_id,
            anchor_path=child_path,
            anchor_embedding=child_anchor,
            mitosis_threshold=(
                parent_node.mitosis_threshold if parent_node else 0.5
            ),
        )
        self.router.register_node(child)

        parent_block = getattr(parent_node, "_block", None) if parent_node else None
        block = NoPropBlock(self.embed_dim, num_heads=self.num_heads)
        if parent_block is not None:
            parent_sd = parent_block.state_dict()
            clean_sd = {}
            for k, v in parent_sd.items():
                new_k = k.replace(".base.", ".")
                if not new_k.endswith("lora_a") and not new_k.endswith("lora_b"):
                    clean_sd[new_k] = v
            block.load_state_dict(clean_sd, strict=False)
        inject_lora_into_block(block, rank=self.lora_rank, alpha=self.lora_alpha)
        block.configure_optimizer(lr=self.lr)
        child.__dict__["_block"] = block

        self.spawned_count += 1
        self.spawned_ids.append(child_id)
        return child_id

    def update_node(self, query: torch.Tensor, node: MeshNode) -> str:
        """
        Update a node's anchor and confidence based on new data.
        Lightweight operation — actual block fine-tuning happens in the
        main training loop. This updates the anchor prototype towards
        the new query and logs an estimated loss for confidence tracking.
        """
        node.update_loss(0.5)
        new_anchor = F.normalize(
            0.9 * node.anchor_embedding + 0.1 * query.squeeze().detach().cpu(),
            dim=-1,
        )
        node.anchor_embedding = new_anchor
        self._write_node_metadata(node)
        self.updated_count += 1
        return node.node_id

    def _write_node_metadata(self, node: MeshNode):
        path = node.anchor_path.replace(".md", "_meta.json")
        meta = {
            "node_id": node.node_id,
            "avg_loss": sum(node.rolling_loss) / max(len(node.rolling_loss), 1),
            "samples": len(node.rolling_loss),
            "confidence": self._node_confidence(node),
        }
        try:
            import json
            with open(path, "w") as f:
                json.dump(meta, f, indent=2)
        except Exception:
            pass

    def __call__(self, query: torch.Tensor, target_embed=None) -> tuple[list, str | None, str]:
        self.hook_calls += 1
        routes = self.router.route(query)
        if not routes:
            return routes, None, 'skip'

        top_node_id, top_node, top_score = routes[0]
        action = self.should_mitose(top_score, nearest_node=top_node)
        result_id: str | None = None

        if action == 'create':
            result_id = self.spawn_node(query, top_node_id)
        elif action == 'update':
            result_id = self.update_node(query, top_node)

        return routes, result_id, action

    def summary(self) -> dict:
        return {
            "hook_calls": self.hook_calls,
            "spawned": self.spawned_count,
            "updated": self.updated_count,
            "total_nodes": len(self.router.nodes),
            "spawned_ids": self.spawned_ids[-10:],
        }

    def state_dict(self) -> dict:
        return {
            "spawned_count": self.spawned_count,
            "updated_count": self.updated_count,
            "hook_calls": self.hook_calls,
            "spawned_ids": self.spawned_ids,
        }

    def load_state_dict(self, state: dict):
        self.spawned_count = state.get("spawned_count", 0)
        self.updated_count = state.get("updated_count", 0)
        self.hook_calls = state.get("hook_calls", 0)
        self.spawned_ids = state.get("spawned_ids", [])
