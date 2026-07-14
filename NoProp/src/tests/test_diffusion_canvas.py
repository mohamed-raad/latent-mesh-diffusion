import torch
from diffusion_canvas import DiffusionCanvas, CanvasBlock, CanvasTransformer, UniformStateDiffusion


def test_canvas_block_output_shape():
    model = CanvasBlock(d_model=64, n_heads=2, n_kv_heads=1, d_ff=256)
    tokens = torch.randn(2, 16, 64)
    t_emb = torch.randn(2, 1, 64).expand(-1, 16, -1)
    out = model(tokens, t_emb)
    assert out.shape == (2, 16, 64)


def test_uniform_diffusion_alpha_bar():
    diffusion = UniformStateDiffusion(num_steps=10)
    t = torch.tensor([0.0, 0.5, 1.0])
    a = diffusion.alpha_bar(t)
    assert a.shape == (3,)
    assert (a[0] > 0.9).all()
    assert (a[-1] < 0.1).all()


def test_uniform_diffusion_corrupt():
    diffusion = UniformStateDiffusion(num_steps=10)
    clean = torch.randn(2, 4)
    t = torch.tensor([[0.5], [0.3]])
    noisy, noise = diffusion.corrupt(clean, t)
    assert noisy.shape == clean.shape
    assert noise.shape == clean.shape
    assert not torch.allclose(clean, noisy)


def test_diffusion_canvas_init():
    canvas = DiffusionCanvas(d_model=64, n_layers=2, n_heads=2, n_kv_heads=1, d_ff=256,
                             vocab_size=100, canvas_len=16, num_steps=5)
    c = canvas.init_canvas(batch_size=2, device=torch.device("cpu"))
    assert c.shape == (2, 16)
    assert c.dtype == torch.long


def test_diffusion_canvas_denoise_step():
    canvas = DiffusionCanvas(d_model=64, n_layers=2, n_heads=2, n_kv_heads=1, d_ff=256,
                             vocab_size=100, canvas_len=16, num_steps=5)
    c = canvas.init_canvas(batch_size=2, device=torch.device("cpu"))
    t = torch.tensor([[0.5], [0.3]])
    logits, cur_pred, entropy, frozen_mask = canvas.denoise_step(c, t)
    assert logits.shape == (2, 16, 100)
    assert entropy.shape == (2, 16)
    assert (entropy >= 0).all()


def test_diffusion_canvas_generate():
    canvas = DiffusionCanvas(d_model=64, n_layers=2, n_heads=2, n_kv_heads=1, d_ff=256,
                             vocab_size=100, canvas_len=16, num_steps=5)
    out = canvas.generate(batch_size=2, device=torch.device("cpu"), max_blocks=1)
    assert out.shape == (2, 16)
    assert out.dtype == torch.long


def test_diffusion_canvas_generate_multiblock():
    canvas = DiffusionCanvas(d_model=64, n_layers=2, n_heads=2, n_kv_heads=1, d_ff=256,
                             vocab_size=100, canvas_len=8, num_steps=3)
    out = canvas.generate(batch_size=2, device=torch.device("cpu"), max_blocks=3)
    assert out.shape == (2, 24)


def test_diffusion_canvas_compute_loss():
    canvas = DiffusionCanvas(d_model=64, n_layers=2, n_heads=2, n_kv_heads=1, d_ff=256,
                             vocab_size=100, canvas_len=16, num_steps=3)
    tokens = torch.randint(0, 100, (2, 16))
    loss = canvas.compute_loss(tokens, tokens)
    assert loss.item() >= 0.0


def test_adaptive_stopping_entropy():
    canvas = DiffusionCanvas(d_model=64, n_layers=2, n_heads=2, n_kv_heads=1, d_ff=256,
                             vocab_size=100, canvas_len=8, num_steps=10,
                             entropy_threshold=0.5)
    assert canvas.entropy_threshold == 0.5


def test_canvas_time_embed():
    emb = CanvasTransformer.time_embed(torch.tensor([[0.5], [0.3]]), dim=64)
    assert emb.shape == (2, 1, 64)
    assert not torch.isnan(emb).any()
