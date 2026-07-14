"""
diffusion_decoder.py — Iterative diffusion decoder for NoPropBlock.

Ports the DiffusionGemma (JAX) sampler pattern to PyTorch, adapted for
embedding-level denoising:

  1. NoPropBlock denoises at the embedding level (predicts clean embeddings
     from noisy ones, matching the MSE training objective).
  2. A weight-tied LM head (embed.weight^T) maps predicted embeddings to
     token logits.
  3. SampleFromPredictions selects high-confidence tokens (low entropy)
     and re-noises the rest.
  4. Self-conditioning feeds the previous step's predicted embeddings
     back into the model via a small projection MLP.
  5. Temperature annealing: high temp early (more exploration), low temp
     late (more conservative).

Usage:
    decoder = DiffusionDecoder(router, blocks, embed, top_k=2)
    tokens = decoder.generate(
        prompt_ids=prompt,
        canvas_length=256,
        max_denoising_steps=48,
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Noise schedule ───────────────────────────────────────────────────────

class LinearSchedule:
    def noise_probability(self, noise_proportion: torch.Tensor) -> torch.Tensor:
        return noise_proportion


# ─── Diffusion process (token-level) ─────────────────────────────────────

class DiffusionProcess:
    """Multinomial diffusion on tokens: replace tokens with random ones."""

    def __init__(self, noise_schedule: LinearSchedule | None = None):
        self.noise_schedule = noise_schedule or LinearSchedule()

    def get_initial_sample(
        self,
        batch_size: int,
        canvas_length: int,
        text_vocab_size: int,
        device: torch.device,
        vocab_range: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Random tokens from uniform distribution over vocabulary (or a range)."""
        lo, hi = vocab_range if vocab_range else (0, text_vocab_size)
        return torch.randint(
            lo, hi, (batch_size, canvas_length), device=device
        )

    def add_noise_to_tokens(
        self,
        tokens: torch.Tensor,
        noise_proportion: torch.Tensor,
        text_vocab_size: int,
    ) -> torch.Tensor:
        """Replace a fraction of tokens with uniform random ones."""
        noise_mask = (
            torch.rand_like(tokens, dtype=torch.float32)
            < noise_proportion[:, None]
        )
        random_tokens = torch.randint(
            0, text_vocab_size, tokens.shape, device=tokens.device
        )
        return torch.where(noise_mask, random_tokens, tokens)


# ─── Confidence-based token selection ────────────────────────────────────

class SampleFromPredictions:
    """Select tokens based on confidence (entropy ordering).

    Accepts tokens with the lowest per-token entropy up to a cumulative
    bound, then re-noises the rest with uniform random tokens.
    """

    def __init__(self, entropy_bound: float = 0.1):
        self.entropy_bound = entropy_bound

    def __call__(
        self,
        logits: torch.Tensor,
        canvas: torch.Tensor,
        text_vocab_size: int,
        vocab_range: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (selected_tokens, selection_mask).

        If vocab_range is given, sampling and renoising are constrained
        to [lo, hi).
        """
        B, L, V = logits.shape
        lo, hi = vocab_range if vocab_range else (0, text_vocab_size)

        # Restrict probs to the valid range
        probs_full = F.softmax(logits.float(), dim=-1)
        log_probs_full = F.log_softmax(logits.float(), dim=-1)

        # Zero out invalid positions for entropy calculation
        mask = torch.arange(V, device=logits.device).unsqueeze(0).unsqueeze(0)
        valid_mask = (mask >= lo) & (mask < hi)
        probs = probs_full * valid_mask.float()
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        log_probs = log_probs_full * valid_mask.float()
        log_probs = torch.where(probs == 0, 0.0, log_probs)

        token_entropy = -(log_probs * probs).sum(-1)

        sorted_entropy, sorted_idx = token_entropy.sort(dim=-1)
        cumsum = sorted_entropy.cumsum(dim=-1)
        sorted_selection = (cumsum - sorted_entropy) <= self.entropy_bound

        selection_mask = torch.zeros_like(sorted_idx, dtype=torch.bool)
        selection_mask.scatter_(1, sorted_idx, sorted_selection)

        # Sample from valid range only
        valid_logits = logits[..., lo:hi]
        sampled_restricted = torch.distributions.Categorical(
            F.softmax(valid_logits.float(), dim=-1)
        ).sample()
        sampled = sampled_restricted + lo

        random_tokens = torch.randint(
            lo, hi, sampled.shape, device=logits.device
        )
        output = torch.where(selection_mask, sampled, random_tokens)
        return output, selection_mask


# ─── Temperature annealing ────────────────────────────────────────────────

class AnnealingTemperatureShaper:
    """Temperature that anneals with noise proportion.

    High temperature early (noise_proportion -> 1), low temperature late
    (noise_proportion -> 0), following a power law.
    """

    def __init__(
        self,
        exponent: float = 1.0,
        max_temperature: float = 0.8,
        min_temperature: float = 0.4,
    ):
        self.exponent = exponent
        self.max_temperature = max_temperature
        self.min_temperature = min_temperature

    def __call__(
        self,
        logits: torch.Tensor,
        noise_proportion: torch.Tensor,
    ) -> torch.Tensor:
        fraction = 1.0 - (1.0 - noise_proportion) ** self.exponent
        temperature = (
            fraction * (self.max_temperature - self.min_temperature)
            + self.min_temperature
        )
        temperature = temperature.clamp(min=1e-12)
        return logits / temperature[:, None, None]


# ─── Self-conditioning projection ────────────────────────────────────────

class SelfConditioning(nn.Module):
    """Projects previous step's predicted embeddings for self-conditioning.

    Matches DiffusionGemma's SelfConditioning:
      pre_norm -> FFN -> post_norm -> add to canvas embeddings
    """

    def __init__(self, d_model: int, hidden_dim: int | None = None):
        super().__init__()
        hidden = hidden_dim or d_model * 4
        self.pre_norm = nn.LayerNorm(d_model)
        self.ffw = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        self.post_norm = nn.LayerNorm(d_model, elementwise_affine=False)

    def forward(
        self,
        canvas_embeddings: torch.Tensor,
        self_conditioning_signal: torch.Tensor,
    ) -> torch.Tensor:
        normed = self.pre_norm(self_conditioning_signal)
        sc_signal = self.ffw(normed)
        combined = canvas_embeddings + sc_signal
        return self.post_norm(combined)


# ─── Early stopping strategies ───────────────────────────────────────────

class EarlyStopFn:
    """Base class for early stopping."""

    def should_stop(
        self,
        step: int,
        canvas: torch.Tensor,
        previous_canvas: torch.Tensor,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError


class NoEarlyStop(EarlyStopFn):
    def should_stop(self, step, canvas, previous_canvas, logits):
        return torch.zeros(canvas.shape[0], dtype=torch.bool, device=canvas.device)


class TokenStabilityEarlyStop(EarlyStopFn):
    """Stop when argmax tokens match previous canvas."""
    def should_stop(self, step, canvas, previous_canvas, logits):
        del step, canvas
        return (logits.argmax(dim=-1) == previous_canvas).all(dim=-1)


class EntropyEarlyStop(EarlyStopFn):
    """Stop when mean entropy falls below threshold."""
    def __init__(self, threshold: float = 0.005):
        self.threshold = threshold

    def should_stop(self, step, canvas, previous_canvas, logits):
        del step, canvas, previous_canvas
        probs = F.softmax(logits.float(), dim=-1)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        safe_lp = torch.where(probs == 0, 0.0, log_probs)
        entropy = -(safe_lp * probs).sum(dim=-1).mean(dim=-1)
        return entropy <= self.threshold


class ChainedEarlyStop(EarlyStopFn):
    """Stop when ALL sub-stoppers agree."""
    def __init__(self, fns: list[EarlyStopFn]):
        self.fns = fns

    def should_stop(self, step, canvas, previous_canvas, logits):
        results = [fn.should_stop(step, canvas, previous_canvas, logits) for fn in self.fns]
        return torch.stack(results).all(dim=0)


# ─── Main Diffusion Decoder ──────────────────────────────────────────────

class DiffusionDecoder:
    """Iterative diffusion decoder wrapping NoPropBlock experts.

    Generates text by starting from pure noise and iteratively denoising
    an entire canvas of tokens in parallel, using confidence-based token
    selection and self-conditioning.

    The underlying NoPropBlock denoises at the embedding level (matching
    its MSE training objective). A weight-tied LM head maps the predicted
    embeddings to vocabulary logits.
    """

    def __init__(
        self,
        router,
        blocks: dict,
        embed: nn.Embedding,
        top_k: int = 2,
        self_conditioning_dim: int | None = None,
        sc_hidden_dim: int | None = None,
    ):
        self.router = router
        self.blocks = blocks
        self.embed = embed
        self.top_k = top_k
        self.d_model = embed.embedding_dim

        sc_dim = self_conditioning_dim or self.d_model
        self.sc_net = SelfConditioning(sc_dim, sc_hidden_dim)
        # Move self-conditioning network to the same device as the embed
        self.sc_net = self.sc_net.to(self.embed.weight.device)

        self.diffusion_process = DiffusionProcess()
        self.sample_from_predictions = SampleFromPredictions(entropy_bound=0.1)
        self.temperature_shaper = AnnealingTemperatureShaper()
        self.early_stop = ChainedEarlyStop([
            TokenStabilityEarlyStop(),
            EntropyEarlyStop(threshold=0.005),
        ])

    # ── Core denoising step ──────────────────────────────────────────────

    @torch.no_grad()
    def denoise_embeddings(
        self,
        noisy_emb: torch.Tensor,
        t: float,
        sc_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run NoPropBlock experts to denoise embeddings.

        Args:
            noisy_emb: Noisy input embeddings [B, L, D].
            t: Noise level (0 = clean, 1 = fully noisy).
            sc_emb: Optional self-conditioning embeddings from previous step.

        Returns:
            Denoised predictions [B, L, D].
        """
        B = noisy_emb.size(0)
        t_2d = torch.zeros(B, 1, device=noisy_emb.device).fill_(t)

        if sc_emb is not None:
            h = self.sc_net(noisy_emb, sc_emb)
        else:
            h = noisy_emb

        query = F.normalize(h.mean(dim=1, keepdim=True), dim=-1)
        active = self.router.route(query)

        pred_sum = None
        for nid, _, _ in active[:self.top_k]:
            pred = self.blocks[nid](h, t_2d)
            pred_sum = pred if pred_sum is None else pred_sum + pred

        if pred_sum is None:
            raise RuntimeError("No experts selected")

        return pred_sum / self.top_k

    # ── Full generation loop (embedding-level noise, matching training) ──

    @torch.no_grad()
    def generate(
        self,
        canvas_length: int,
        prompt_ids: torch.Tensor | None = None,
        max_denoising_steps: int = 48,
        temperature_config: tuple[float, float, float] | None = None,
        entropy_bound: float = 0.1,
        return_trajectory: bool = False,
        vocab_range: tuple[int, int] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Generate a complete canvas of tokens via iterative diffusion.

        Uses embedding-level Gaussian noise (matching the MSE training
        objective), with token-level confidence-based selection.

        Args:
            canvas_length: Total length of the output canvas.
            prompt_ids: Optional prompt tokens [B, prompt_len] to condition on.
            max_denoising_steps: Number of iterative denoising steps.
            temperature_config: (min_temp, max_temp, exponent) for annealing.
            entropy_bound: Confidence threshold for token selection.
            return_trajectory: If True, return list of canvas at each step.
            vocab_range: Optional (lo, hi) to constrain generation tokens.

        Returns:
            Generated tokens [B, canvas_length], or (tokens, trajectory).
        """
        if prompt_ids is not None:
            assert prompt_ids.dim() == 2, f"Expected [B, L], got {prompt_ids.shape}"
            B, prompt_len = prompt_ids.shape
            assert prompt_len <= canvas_length
        else:
            B = 1
            prompt_len = 0

        device = next(self.embed.parameters()).device
        vocab_size = self.embed.num_embeddings

        self.sample_from_predictions.entropy_bound = entropy_bound
        if temperature_config is not None:
            self.temperature_shaper = AnnealingTemperatureShaper(
                exponent=temperature_config[2],
                max_temperature=temperature_config[1],
                min_temperature=temperature_config[0],
            )

        noise_proportions = (
            1.0 - torch.arange(max_denoising_steps + 1) / max_denoising_steps
        )

        # ── Initialize ──
        if prompt_ids is not None:
            prompt_tokens = prompt_ids.clone()
            gen_tokens = self.diffusion_process.get_initial_sample(
                B, canvas_length - prompt_len, vocab_size, device, vocab_range
            )
            canvas = torch.cat([prompt_tokens, gen_tokens], dim=1)
            prompt_mask = torch.cat([
                torch.ones(B, prompt_len, dtype=torch.bool, device=device),
                torch.zeros(B, canvas_length - prompt_len, dtype=torch.bool, device=device),
            ], dim=1)

            # Initial noisy: embed tokens + full noise (matching training distribution)
            prompt_clean = self.embed(prompt_tokens)
            prompt_noisy = prompt_clean + torch.randn_like(prompt_clean)
            gen_clean = self.embed(gen_tokens)
            gen_noisy = gen_clean + torch.randn_like(gen_clean)
            noisy_emb = torch.cat([prompt_noisy, gen_noisy], dim=1)
        else:
            canvas = self.diffusion_process.get_initial_sample(
                B, canvas_length, vocab_size, device, vocab_range
            )
            prompt_mask = None
            clean_emb = self.embed(canvas)
            noisy_emb = clean_emb + torch.randn_like(clean_emb)

        sc_emb = torch.zeros(B, canvas_length, self.d_model, device=device)
        trajectory = [canvas.clone()] if return_trajectory else None
        done = torch.zeros(B, dtype=torch.bool, device=device)

        # ── Denoising loop ──
        for step in range(max_denoising_steps):
            t = noise_proportions[step].item()
            t_batch = torch.full((B,), t, device=device)

            # 1. Denoise at embedding level (matching training: N(0,1) noise, t=0)
            pred_emb = self.denoise_embeddings(noisy_emb, t, sc_emb)

            # Keep prompt predictions pinned to clean prompt embeddings
            if prompt_mask is not None:
                prompt_clean = self.embed(canvas[:, :prompt_len])
                pred_emb = torch.where(
                    prompt_mask.unsqueeze(-1),
                    torch.cat([prompt_clean, pred_emb[:, prompt_len:]], dim=1),
                    pred_emb,
                )

            # 2. Map to logits via weight-tied LM head
            logits = pred_emb @ self.embed.weight.T  # [B, L, V]

            # 3. Temperature shaping
            shaped_logits = self.temperature_shaper(logits, t_batch)

            # 4. Confidence-based token selection (DiffusionGemma style)
            new_canvas, selection_mask = self.sample_from_predictions(
                shaped_logits, canvas, vocab_size, vocab_range
            )
            if prompt_mask is not None:
                new_canvas = torch.where(prompt_mask, canvas, new_canvas)

            # 5. Early stopping
            new_done = self.early_stop.should_stop(step, new_canvas, canvas, shaped_logits)
            new_canvas = torch.where(done[:, None], canvas, new_canvas)
            done = done | new_done
            canvas = new_canvas

            # 6. Prepare next noisy input: re-embed + Gaussian noise
            t_next = noise_proportions[step + 1].item()
            clean_emb = self.embed(canvas)
            noise_next = torch.randn_like(clean_emb) * (1 - t_next)
            noisy_emb = clean_emb + noise_next

            # Keep prompt positions with their own noise
            if prompt_mask is not None:
                prompt_clean = self.embed(canvas[:, :prompt_len])
                prompt_noisy_next = prompt_clean + torch.randn_like(prompt_clean) * (1 - t_next)
                noisy_emb = torch.cat([prompt_noisy_next, noisy_emb[:, prompt_len:]], dim=1)

            # 7. Self-conditioning for next step
            sc_emb = torch.where(
                done[:, None, None], sc_emb, pred_emb.detach()
            )

            if trajectory is not None:
                trajectory.append(canvas.clone())

            if done.all():
                break

        if trajectory is not None:
            return canvas, trajectory

        return canvas

    # ── Convenience: denoise with interactive streaming ──────────────────

    @torch.no_grad()
    def generate_with_prompt(
        self,
        prompt_text: str,
        tokenizer,
        canvas_length: int = 256,
        max_denoising_steps: int = 48,
    ) -> str:
        """Tokenize prompt, generate, decode.

        Args:
            prompt_text: Input text.
            tokenizer: Tokenizer with encode() and decode() methods.
            canvas_length: Total sequence length.
            max_denoising_steps: Denoising steps.

        Returns:
            Decoded generated text.
        """
        prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt")
        prompt_ids = prompt_ids.to(next(self.embed.parameters()).device)
        tokens = self.generate(
            canvas_length=canvas_length,
            prompt_ids=prompt_ids,
            max_denoising_steps=max_denoising_steps,
        )
        return tokenizer.decode(tokens[0], skip_special_tokens=True)
