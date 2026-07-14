import sys
sys.path.insert(0, "NoProp/src")

import torch
import pytest
from text_vae import TextVAE, TextVAEConfig, GaussianDistribution, VAEBlock, build_causal_mask


def test_gaussian_distribution():
    mean = torch.randn(2, 4, 8)
    logvar = torch.randn(2, 4, 8) * 0.1 - 2.0
    dist = GaussianDistribution(mean=mean, logvar=logvar)
    sample = dist.sample()
    assert sample.shape == mean.shape
    assert not (sample == mean).all()  # sample != mean (stochastic)
    mode = dist.mode()
    assert torch.allclose(mode, mean)
    kl = dist.kl()
    assert kl.shape == (2, 4)
    assert (kl >= 0).all()


def test_vae_block_causal():
    block = VAEBlock(dim=64, n_heads=2, ffn_dim=256, causal=True)
    x = torch.randn(2, 8, 64)
    mask = build_causal_mask(8)
    out = block(x, attn_mask=mask)
    assert out.shape == x.shape


def test_text_vae_forward():
    cfg = TextVAEConfig(d_model=64, d_latent=16, n_latent_nodes=8, patch_size=2, n_encoder_blocks=2, n_decoder_blocks=2, n_heads=2)
    vae = TextVAE(cfg)
    x = torch.randn(2, 16, 64)
    dist, x_hat = vae(x)
    assert dist.mean.shape == (2, 8, 16)
    assert x_hat.shape == (2, 16, 64)


def test_text_vae_hierarchical():
    cfg = TextVAEConfig(d_model=64, d_latent=16, n_latent_nodes=8, patch_size=2, n_encoder_blocks=2, n_decoder_blocks=2, n_heads=2, hierarchical=True)
    vae = TextVAE(cfg)
    x = torch.randn(2, 16, 64)
    dist, x_hat = vae(x)
    assert dist.mean.shape == (2, 8, 16)
    assert x_hat.shape == (2, 16, 64)


def test_text_vae_deterministic():
    cfg = TextVAEConfig(d_model=64, d_latent=16, n_latent_nodes=8, patch_size=2, n_encoder_blocks=2, n_decoder_blocks=2, n_heads=2, use_variation=False)
    vae = TextVAE(cfg)
    x = torch.randn(2, 16, 64)
    out = vae.encoder(x)
    assert out.shape == (2, 8, 16)


def test_text_vae_loss():
    cfg = TextVAEConfig(d_model=64, d_latent=16, n_latent_nodes=8, patch_size=2, n_encoder_blocks=2, n_decoder_blocks=2, n_heads=2, kl_beta=0.01)
    vae = TextVAE(cfg)
    x = torch.randn(2, 16, 64)
    dist, x_hat = vae(x)
    losses = vae.loss(x, dist, x_hat)
    assert 'loss' in losses
    assert 'recon' in losses
    assert 'kl' in losses
    assert losses['loss'] > 0
    assert losses['recon'] > 0
    assert losses['kl'] > 0


def test_text_vae_project_tokens():
    cfg = TextVAEConfig(d_model=64, d_latent=16, n_latent_nodes=8, patch_size=2, n_encoder_blocks=2, n_decoder_blocks=2, n_heads=2)
    vae = TextVAE(cfg)
    x = torch.randn(2, 16, 64)
    proj = vae.project_tokens(x)
    assert proj.shape == (2, 8, 16)

    # 2D input (single batch)
    x2 = torch.randn(16, 64)
    proj2 = vae.project_tokens(x2)
    assert proj2.shape == (1, 8, 16)


def test_vae_backward():
    cfg = TextVAEConfig(d_model=64, d_latent=16, n_latent_nodes=8, patch_size=2, n_encoder_blocks=2, n_decoder_blocks=2, n_heads=2)
    vae = TextVAE(cfg)
    x = torch.randn(2, 16, 64)
    dist, x_hat = vae(x)
    losses = vae.loss(x, dist, x_hat)
    losses['loss'].backward()
    # All VAE params should have gradients
    for name, p in vae.named_parameters():
        assert p.grad is not None, f"{name} has no gradient"


def test_gradient_flow():
    cfg = TextVAEConfig(d_model=64, d_latent=16, n_latent_nodes=8, patch_size=2, n_encoder_blocks=2, n_decoder_blocks=2, n_heads=2)
    vae = TextVAE(cfg)
    opt = torch.optim.AdamW(vae.parameters(), lr=1e-3)
    x = torch.randn(2, 16, 64)
    for step in range(5):
        opt.zero_grad()
        dist, x_hat = vae(x)
        losses = vae.loss(x, dist, x_hat)
        losses['loss'].backward()
        opt.step()
    assert True  # 5 steps of VAE training complete


def test_build_causal_mask():
    mask = build_causal_mask(4)
    assert mask.shape == (4, 4)
    # Upper triangle should be -inf
    assert mask[0, 1] == float('-inf')
    assert mask[0, 2] == float('-inf')
    assert mask[1, 3] == float('-inf')
    # Lower triangle and diagonal should be 0
    assert mask[0, 0] == 0.0
    assert mask[1, 0] == 0.0
    assert mask[3, 2] == 0.0


def test_variable_length():
    cfg = TextVAEConfig(d_model=32, d_latent=8, n_latent_nodes=16, patch_size=2, n_encoder_blocks=1, n_decoder_blocks=1, n_heads=2)
    vae = TextVAE(cfg)
    for seq_len in [8, 12, 16, 20]:
        x = torch.randn(1, seq_len, 32)
        dist, x_hat = vae(x)
        assert x_hat.shape == (1, seq_len, 32)
