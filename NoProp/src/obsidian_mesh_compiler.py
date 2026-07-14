import os
import re
import glob
import math
import torch
import torch.nn.functional as F
from typing import Literal

WIKI_LINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")
FRONT_MATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)
HEADER_RE = re.compile(r"^#{1,6}\s+.*$", re.MULTILINE)
STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very", "just",
    "because", "and", "but", "or", "if", "while", "although", "this", "that",
    "these", "those", "it", "its", "i", "me", "my", "we", "our", "you",
    "your", "he", "him", "his", "she", "her", "they", "them", "their",
})


def clean_markdown_text(text: str) -> list[str]:
    text = FRONT_MATTER_RE.sub("", text)
    text = HEADER_RE.sub("", text)
    text = re.sub(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]", r"\1", text)
    text = re.sub(r"\*\*|__|\*|_|`|~~", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s'-]", " ", text)
    tokens = [t.lower().strip("'-") for t in text.split() if len(t) > 2]
    return [t for t in tokens if t not in STOP_WORDS]


def parse_wiki_links(content: str) -> list[str]:
    return [m.group(1).strip() for m in WIKI_LINK_RE.finditer(content)]


def resolve_wiki_name(filepath: str) -> str:
    base = os.path.splitext(os.path.basename(filepath))[0]
    base = base.replace("_", " ").replace("-", " ")
    return base.strip()


class LightweightStaticEmbedder:
    _seed: int = 42
    _fitted: bool = False
    _vocab: dict[str, int] = {}
    _projection: torch.Tensor | None = None
    _embed_dim: int = 768
    _max_vocab: int = 8192

    @classmethod
    def fit(cls, all_tokens: list[list[str]], embed_dim: int = 768, max_vocab: int = 8192):
        cls._embed_dim = embed_dim
        cls._max_vocab = max_vocab
        counter: dict[str, int] = {}
        for toks in all_tokens:
            for t in set(toks):
                counter[t] = counter.get(t, 0) + 1
        sorted_terms = sorted(counter, key=lambda k: (-counter[k], k))[:max_vocab]
        cls._vocab = {t: i for i, t in enumerate(sorted_terms)}
        rng = torch.Generator()
        rng.manual_seed(cls._seed)
        cls._projection = torch.randn(len(cls._vocab), embed_dim, generator=rng) * 0.02
        cls._fitted = True

    @classmethod
    def embed(cls, tokens: list[str]) -> torch.Tensor:
        if not cls._fitted:
            raise RuntimeError("LightweightStaticEmbedder.fit() must be called first")
        n = len(tokens)
        if n == 0:
            return torch.zeros(cls._embed_dim)
        indices = [cls._vocab[t] for t in tokens if t in cls._vocab]
        if not indices:
            return torch.zeros(cls._embed_dim)
        bow = torch.zeros(len(cls._vocab))
        for idx in indices:
            bow[idx] += 1.0
        bow = bow / max(bow.sum(), 1.0)
        emb = bow @ cls._projection
        return F.normalize(emb, dim=-1)


class ObsidianMeshCompiler:
    def __init__(
        self,
        vault_path: str,
        embed_dim: int = 768,
        max_vocab: int = 8192,
        embedding_backend: Literal["static", "sentence_transformers"] = "static",
    ):
        self.vault_path = vault_path
        self.embed_dim = embed_dim
        self.max_vocab = max_vocab
        self.embedding_backend = embedding_backend

        self.page_names: list[str] = []
        self.page_paths: list[str] = []
        self.page_tokens: list[list[str]] = []
        self.page_content: list[str] = []
        self.link_graph: dict[str, set[str]] = {}
        self.sparse_adj: torch.Tensor | None = None
        self.anchor_embeddings: list[torch.Tensor] = []
        self.node_ids: list[str] = []

        self._st_embedder = None
        if embedding_backend == "sentence_transformers":
            self._try_import_sentence_transformers()

    def _try_import_sentence_transformers(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._st_embedder = SentenceTransformer(
                "all-MiniLM-L6-v2", device="cpu"
            )
        except ImportError:
            import warnings
            warnings.warn(
                "sentence_transformers not installed. "
                "Falling back to 'static' embedding backend."
            )
            self.embedding_backend = "static"

    def _walk_markdown_files(self) -> list[str]:
        pattern = os.path.join(self.vault_path, "**/*.md")
        files = glob.glob(pattern, recursive=True)
        files = [f for f in files if os.path.isfile(f)]
        return sorted(set(files))

    def scan_vault(self) -> dict[str, set[str]]:
        files = self._walk_markdown_files()
        if not files:
            self.page_names = []
            self.page_paths = []
            self.link_graph = {}
            return self.link_graph

        name_counts: dict[str, int] = {}
        for fp in files:
            name = resolve_wiki_name(fp)
            name_counts[name] = name_counts.get(name, 0) + 1

        self.page_names = []
        self.page_paths = []
        self.link_graph = {}
        for fp in files:
            name = resolve_wiki_name(fp)
            uid = name if name_counts[name] == 1 else f"{name}__{os.path.basename(fp)}"
            self.page_names.append(name)
            self.page_paths.append(fp)
            self.link_graph[uid] = set()
            self.node_ids.append(uid)

        for uid, fp in zip(self.node_ids, self.page_paths):
            try:
                with open(fp, encoding="utf-8", errors="replace") as fh:
                    content = fh.read()
            except OSError:
                continue
            self.page_content.append(content)
            links = parse_wiki_links(content)
            resolved: set[str] = set()
            for link in links:
                target_name = link.replace("_", " ").replace("-", " ").strip()
                target_uids = [
                    nid for nid, pn in zip(self.node_ids, self.page_names)
                    if pn == target_name
                ]
                if target_uids:
                    resolved.add(target_uids[0])
                else:
                    resolved.add(link)
            self.link_graph[uid] = resolved

        return self.link_graph

    def build_sparse_adjacency(self) -> torch.Tensor:
        if not self.node_ids:
            raise RuntimeError("No nodes. Call scan_vault() first.")
        n = len(self.node_ids)
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        id_to_idx = {uid: i for i, uid in enumerate(self.node_ids)}
        for src_uid, targets in self.link_graph.items():
            si = id_to_idx.get(src_uid)
            if si is None:
                continue
            for tgt in targets:
                ti = id_to_idx.get(tgt)
                if ti is not None and si != ti:
                    rows.append(si)
                    cols.append(ti)
                    vals.append(1.0)

        indices = torch.tensor([rows, cols], dtype=torch.long)
        values = torch.tensor(vals, dtype=torch.float32)
        shape = (n, n)
        self.sparse_adj = torch.sparse_coo_tensor(indices, values, shape)
        return self.sparse_adj

    def embed_semantic_anchors(self) -> list[torch.Tensor]:
        self.page_tokens = [clean_markdown_text(c) for c in self.page_content]

        if self.embedding_backend == "sentence_transformers" and self._st_embedder is not None:
            texts = [" ".join(t) for t in self.page_tokens]
            embs = self._st_embedder.encode(texts, convert_to_tensor=True)
            embs = F.normalize(embs, dim=-1)
            self.anchor_embeddings = [embs[i].cpu() for i in range(embs.size(0))]
        else:
            LightweightStaticEmbedder.fit(self.page_tokens, self.embed_dim, self.max_vocab)
            self.anchor_embeddings = [
                LightweightStaticEmbedder.embed(tok).cpu()
                for tok in self.page_tokens
            ]

        return self.anchor_embeddings

    def inject_into_router(self, router, nodes_dir: str | None = None) -> list[str]:
        from mesh_router import MeshNode
        if nodes_dir is None:
            nodes_dir = os.path.join(self.vault_path, "..", "nodes")
        os.makedirs(nodes_dir, exist_ok=True)

        registered: list[str] = []
        for uid, anchor, fp in zip(self.node_ids, self.anchor_embeddings, self.page_paths):
            if uid in router.nodes:
                existing = router.nodes[uid]
                existing.anchor_embedding = anchor
                continue
            node_path = os.path.join(nodes_dir, f"{uid}.md")
            if not os.path.exists(node_path):
                try:
                    with open(fp, encoding="utf-8", errors="replace") as fh:
                        sha = fh.read()[:512]
                    with open(node_path, "w", encoding="utf-8") as fw:
                        fw.write(f"# {uid}\n\n")
                        fw.write(f"Compiled from `{fp}`\n\n")
                        fw.write(f"**Preview:**\n{sha}\n")
                except OSError:
                    with open(node_path, "w", encoding="utf-8") as fw:
                        fw.write(f"# {uid}\n\nCompiled from vault\n")
            node = MeshNode(
                node_id=uid,
                anchor_path=node_path,
                anchor_embedding=anchor.clone(),
                mitosis_threshold=0.5,
            )
            router.register_node(node)
            registered.append(uid)

        return registered

    def compute_adjacency_prior(self, scaling: float = 0.1) -> torch.Tensor:
        if self.sparse_adj is None:
            raise RuntimeError("No sparse adjacency. Call build_sparse_adjacency() first.")
        adj_dense = self.sparse_adj.to_dense()
        n = adj_dense.size(0)
        norm = self.embed_dim ** 0.5
        prior = torch.eye(n) + scaling * adj_dense
        prior = prior / prior.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return prior

    def compile(self, router, nodes_dir: str | None = None) -> dict:
        self.scan_vault()
        self.build_sparse_adjacency()
        self.embed_semantic_anchors()
        registered = self.inject_into_router(router, nodes_dir)
        prior = self.compute_adjacency_prior()
        return {
            "node_ids": self.node_ids,
            "registered": registered,
            "adjacency_prior": prior,
            "n_nodes": len(self.node_ids),
            "n_edges": self.sparse_adj._nnz() if self.sparse_adj is not None else 0,
        }
