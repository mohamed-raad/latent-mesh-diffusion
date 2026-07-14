import pytest
import torch
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from obsidian_mesh_compiler import (
    ObsidianMeshCompiler,
    parse_wiki_links,
    clean_markdown_text,
    resolve_wiki_name,
    LightweightStaticEmbedder,
)
from mesh_router import MeshRouter


@pytest.fixture
def obsidian_vault():
    with tempfile.TemporaryDirectory() as vault:
        pages = {
            "Home.md": "# Home\nWelcome to [[Projects]] and [[Notes]]\n\nTags: knowledge",
            "Projects.md": "# Projects\nSee [[Home]] for context.\nAlso [[Archive]] is related.",
            "Notes.md": "# Notes\nRandom thoughts on [[Home]]\n\nDeep learning is fun.",
            "Archive.md": "# Archive\nOld [[Projects]] and other [[Notes]]\n\nThis is deprecated.",
            "Orphan.md": "# Orphan\nThis page links to nothing.",
        }
        for name, content in pages.items():
            with open(os.path.join(vault, name), "w") as f:
                f.write(content)
        yield vault


@pytest.fixture
def compiler(obsidian_vault):
    return ObsidianMeshCompiler(obsidian_vault, embed_dim=64, max_vocab=512)


def test_parse_wiki_links():
    links = parse_wiki_links("[[Home]] and [[Projects|display text]] and [[Notes]]")
    assert links == ["Home", "Projects", "Notes"]


def test_parse_wiki_links_no_links():
    assert parse_wiki_links("No links here") == []


def test_clean_markdown_text_basic():
    tokens = clean_markdown_text("Hello world, this is a test of tokenization!")
    assert "hello" in tokens
    assert "world" in tokens
    assert "tokenization" in tokens
    assert "the" not in tokens
    assert "a" not in tokens


def test_clean_markdown_text_strips_front_matter():
    md = "---\ntitle: Test\n---\n# Real Content\nHere is the body."
    tokens = clean_markdown_text(md)
    assert set(tokens) == {"body"}


def test_resolve_wiki_name():
    assert resolve_wiki_name("My File.md") == "My File"
    assert resolve_wiki_name("deep_learning_notes.md") == "deep learning notes"


def test_scan_vault_finds_all_pages(compiler, obsidian_vault):
    graph = compiler.scan_vault()
    assert len(graph) == 5
    assert "Home" in graph
    assert "Projects" in graph
    assert "Orphan" in graph


def test_scan_vault_links_correct(compiler):
    graph = compiler.scan_vault()
    assert "Projects" in graph.get("Home", set())
    assert "Notes" in graph.get("Home", set())
    assert "Home" in graph.get("Projects", set())
    assert "Archive" in graph.get("Projects", set())


def test_scan_vault_orphan_has_no_links(compiler):
    graph = compiler.scan_vault()
    assert len(graph.get("Orphan", set())) == 0


def test_build_sparse_adjacency(compiler):
    compiler.scan_vault()
    adj = compiler.build_sparse_adjacency()
    assert adj.is_sparse
    n = len(compiler.node_ids)
    assert adj.shape == (n, n)
    assert adj._nnz() > 0


def test_build_sparse_adjacency_before_scan_raises(compiler):
    with pytest.raises(RuntimeError, match="No nodes"):
        compiler.build_sparse_adjacency()


def test_embed_semantic_anchors_static(compiler):
    compiler.scan_vault()
    compiler.page_content = [
        "deep learning transformer attention",
        "convolutional neural network",
        "reinforcement learning agent",
        "natural language processing",
        "computer vision object detection",
    ]
    embs = compiler.embed_semantic_anchors()
    assert len(embs) == 5
    for e in embs:
        assert e.shape == (64,)
        assert torch.isfinite(e).all()
    assert embs[0] is not None


def test_inject_into_router(compiler):
    compiler.scan_vault()
    compiler.embed_semantic_anchors()
    router = MeshRouter(top_k=2)
    registered = compiler.inject_into_router(router)
    assert len(registered) == 5
    assert len(router.nodes) == 5
    for nid in ["Home", "Projects", "Notes", "Archive", "Orphan"]:
        assert nid in router.nodes
        assert router.nodes[nid].anchor_embedding.shape == (64,)


def test_inject_into_router_skips_existing(compiler):
    compiler.scan_vault()
    compiler.embed_semantic_anchors()
    router = MeshRouter(top_k=2)
    registered1 = compiler.inject_into_router(router)
    registered2 = compiler.inject_into_router(router)
    assert len(registered2) == 0


def test_compute_adjacency_prior(compiler):
    compiler.scan_vault()
    compiler.build_sparse_adjacency()
    prior = compiler.compute_adjacency_prior(scaling=0.1)
    n = len(compiler.node_ids)
    assert prior.shape == (n, n)
    assert torch.isfinite(prior).all()


def test_full_compile_pipeline(compiler):
    router = MeshRouter(top_k=3)
    result = compiler.compile(router)
    assert result["n_nodes"] == 5
    assert result["n_edges"] > 0
    assert len(result["registered"]) == 5
    assert result["adjacency_prior"].shape == (5, 5)
    assert len(router.nodes) == 5


def test_lightweight_static_embedder():
    tokens_batch = [
        ["deep", "learning", "transformer", "attention"],
        ["neural", "network", "convolution"],
        ["reinforcement", "learning", "agent"],
    ]
    LightweightStaticEmbedder.fit(tokens_batch, embed_dim=32, max_vocab=64)
    emb = LightweightStaticEmbedder.embed(["deep", "learning"])
    assert emb.shape == (32,)
    assert torch.isfinite(emb).all()
    norm = emb.norm().item()
    assert abs(norm - 1.0) < 1e-5


def test_lightweight_static_embedder_empty_tokens():
    emb = LightweightStaticEmbedder.embed([])
    assert emb.shape == (32,)
    assert (emb == 0).all()


def test_empty_vault():
    with tempfile.TemporaryDirectory() as empty:
        comp = ObsidianMeshCompiler(empty, embed_dim=32)
        graph = comp.scan_vault()
        assert len(graph) == 0


def test_broken_markdown_link_does_not_crash():
    text = "Some [[Valid Page]] and [[   Spaces ]] and [[Valid|display]]"
    links = parse_wiki_links(text)
    assert "Valid Page" in links
    assert "Spaces" in links


def test_embedding_fallback_works():
    comp = ObsidianMeshCompiler(
        os.path.dirname(__file__),  # dummy path for init only
        embed_dim=32,
        embedding_backend="sentence_transformers",
    )
    assert comp.embedding_backend == "static"
