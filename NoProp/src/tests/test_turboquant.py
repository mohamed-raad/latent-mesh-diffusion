import pytest
import torch
import math
import sys

sys.path.insert(0, "NoProp/src")

from turboquant_attention import (
    random_orthogonal_matrix,
    lloyd_max_centroids,
    PolarQuantTransform,
    QJLResidualCorrection,
    TurboQuantKVCompression,
    StreamingCentroids,
    AdaptivePolarQuantTransform,
    CrossLayerKVCache,
)


def test_random_orthogonal():
    dim = 32
    Q = random_orthogonal_matrix(dim)
    diff = (Q @ Q.T - torch.eye(dim)).abs().max().item()
    assert diff < 1e-5, f"Orthogonality violation: {diff}"


def test_lloyd_max_centroids():
    data = torch.randn(1000, 16)
    centroids = lloyd_max_centroids(data, num_bits=3)
    assert centroids.shape[0] == 8


def test_polar_quant_forward():
    pq = PolarQuantTransform(dim=32, num_quant_bits=3)
    x = torch.randn(4, 32)
    out = pq(x)
    assert out.shape == x.shape


def test_qjl_correction():
    qjl = QJLResidualCorrection(dim=32)
    v = torch.randn(4, 32)
    out = qjl(v)
    assert out.shape == v.shape
    assert (out.abs() - 1.0).abs().max().item() < 1e-5, "Sign vector must be ±1"


def test_turboquant_compress():
    tq = TurboQuantKVCompression(dim=32, num_quant_bits=3)
    k = torch.randn(4, 8, 32)
    v = torch.randn(4, 8, 32)
    kq, vq, corr = tq.compress(k, v)
    assert kq.shape == k.shape
    assert vq.shape == v.shape


def test_turboquant_attention():
    tq = TurboQuantKVCompression(dim=32, num_quant_bits=3)
    q = torch.randn(4, 8, 32)
    k = torch.randn(4, 8, 32)
    v = torch.randn(4, 8, 32)
    out = tq.compress_attention(q, k, v, layer_idx=0)
    assert out.shape == (4, 8, 32)
    assert torch.isfinite(out).all()


def test_streaming_centroids():
    sc = StreamingCentroids(dim=16, num_bits=3)
    data = torch.randn(100, 16)
    sc.update(data)
    assert sc.centroids is not None
    assert sc.centroids.shape[0] == 8


def test_streaming_quantize():
    sc = StreamingCentroids(dim=16, num_bits=3)
    sc.update(torch.randn(100, 16))
    x = torch.randn(4, 16)
    qx = sc.quantize(x)
    assert qx.shape == x.shape


def test_streaming_quantize_ste():
    sc = StreamingCentroids(dim=16, num_bits=3)
    sc.update(torch.randn(100, 16))
    x = torch.randn(4, 16, requires_grad=True)
    qx = sc.quantize(x, ste=True)
    loss = qx.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_adaptive_polar_quant():
    apq = AdaptivePolarQuantTransform(dim=32, default_bits=3)
    x = torch.randn(4, 32)
    out = apq(x)
    assert out.shape == x.shape


def test_adaptive_bit_selection():
    apq = AdaptivePolarQuantTransform(dim=32)
    assert apq.select_bits(0.001) == 2
    assert apq.select_bits(0.05) == 3
    assert apq.select_bits(0.5) == 5


def test_cross_layer_kv_cache():
    cache = CrossLayerKVCache(max_layers=4)
    for i in range(3):
        k = torch.randn(1, 8, 32)
        v = torch.randn(1, 8, 32)
        cache.set_layer_kv(i, k, v)
    k_rec, v_rec = cache.reconstruct(2)
    assert k_rec.shape == (1, 8, 32)
    assert v_rec.shape == (1, 8, 32)
    stats = cache.k_cache
    assert len(stats) == 3


def test_cross_layer_reconstruct():
    cache = CrossLayerKVCache(max_layers=4)
    k0 = torch.randn(1, 8, 32)
    v0 = torch.randn(1, 8, 32)
    cache.set_layer_kv(0, k0, v0)
    k1 = torch.randn(1, 8, 32)
    v1 = torch.randn(1, 8, 32)
    cache.set_layer_kv(1, k1, v1)
    k_rec, v_rec = cache.reconstruct(1)
    assert (k_rec - k1).abs().max().item() < 1e-6


def test_turboquant_get_stats():
    tq = TurboQuantKVCompression(dim=32)
    stats = tq.get_stats()
    assert "cache_layers" in stats
    assert "active_bits" in stats
    assert "centroid_count" in stats


def test_compress_attention_cached():
    tq = TurboQuantKVCompression(dim=32)
    q = torch.randn(4, 8, 32)
    k = torch.randn(4, 8, 32)
    v = torch.randn(4, 8, 32)
    out = tq.compress_attention(q, k, v, layer_idx=0)
    out2 = tq.compress_attention_cached(q, layer_idx=0)
    assert out.shape == out2.shape


def test_legacy_lloyd_max():
    data = torch.randn(1000, 16)
    centroids = lloyd_max_centroids(data, num_bits=3, num_iters=5)
    assert centroids.shape[0] == 8


def test_legacy_polar_quant():
    pq = PolarQuantTransform(dim=32)
    x = torch.randn(4, 32)
    out = pq(x)
    assert out.shape == x.shape
