# Diffusion Mesh — NoProp Documentation

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Module Reference](#3-module-reference)
   - 3.1 [NoPropBlock](#31-nopropblock)
   - 3.2 [MeshRouter](#32-meshrouter)
   - 3.3 [DiffusionDecoder](#33-diffusiondecoder)
4. [Training Pipeline](#4-training-pipeline)
5. [Evaluation](#5-evaluation)
6. [Benchmarking](#6-benchmarking)
7. [Data Preparation](#7-data-preparation)
8. [Configuration Guide](#8-configuration-guide)
9. [Development Guide](#9-development-guide)
10. [Upcoming Features](#10-upcoming-features)
11. [Appendix: Benchmark Data](#11-appendix-benchmark-data)

---

## 1. Project Overview

The **Diffusion Mesh** architecture replaces global backpropagation with a swarm of independently-trained expert blocks. Each expert is a `NoPropBlock` — a single transformer layer (self-attention + FFN) with its own local optimizer and loss. Experts are selected via cosine-similarity routing through a shared anchor space.

The model is a **denoising autoencoder**: given `embed(token) + Gaussian noise`, it predicts the clean embedding via MSE loss. A weight-tied LM head (`embed.weight^T`) maps predictions back to token logits for generation.

**Key design principles** (from `rules.md`):
- **Gradient isolation**: frozen 4-bit base, active BF16 nodes with isolated AdamW
- **Atomic checkpointing**: temp-file + rename, crash-safe resume
- **Dynamic mitosis**: sustained high error triggers new expert creation
- **Router routing**: cosine-similarity, not linear layer loops

### Model Size

| Component | Parameters | Notes |
|---|---|---|
| 7 experts (NoPropBlock) | 103M | d=1024, 16 heads, 4× FFN |
| Embedding table | 155M | VOCAB=151643, d=1024 |
| **Total** | **258M** | Fits in ~3.6GB during training |

### Hardware Target

- GPU: NVIDIA RTX 5060 (8.5 GB VRAM)
- Training: ~60ms/step, ~17K tok/s
- Generation: ~0.5s for 128 tokens (64 denoising steps)
- Total VRAM: 3.6GB training, 2.9GB generation

---

## 2. Architecture

### 2.1 Component Hierarchy

```
Training Loop (train_core.py)
│
├── MeshRouter (mesh_router.py)
│   ├── MeshNode × N_experts
│   │   ├── anchor_embedding (cosine anchor)
│   │   ├── adapter (ExpertAdapter, optional)
│   │   └── rolling_loss (mitosis trigger)
│   ├── UniversalLatentSpace (d_model → 256)
│   ├── RouterPlanner (intent + difficulty + graph)
│   ├── ConfidenceEngine
│   └── HealthMonitor
│
├── NoPropBlock × N_experts (noprop_block.py)
│   ├── Self-Attention (MultiheadAttention)
│   ├── FFN (Linear → GELU → Linear, 4× expansion)
│   ├── LayerNorm × 2
│   ├── input_proj (LazyLinear, d_model)
│   ├── time_emb (Linear, d_model)
│   └── local AdamW optimizer
│
├── Embedding (nn.Embedding, VOCAB × d_model)
│
└── DiffusionDecoder (diffusion_decoder.py, for generation)
    ├── DiffusionProcess (token noise schedule)
    ├── SelfConditioning (norm → FFN → norm)
    ├── SampleFromPredictions (entropy-based selection)
    ├── AnnealingTemperatureShaper
    └── EarlyStopping (chained)
```

### 2.2 Data Flow (Training)

```
Tokens ──→ Embedding ──→ x_emb [B, S, D]
                              │
                    noise = N(0, 1-t)
                              │
                    x_t = x_emb + noise
                              │
                    query = mean(x_t)
                              │
                    MeshRouter.route(query)
                              │
                    top_k experts selected
                              │
                    ┌─────────────────┐
                    │  NoPropBlock[i]  │──→ pred_i [B, S, D]
                    │  forward(x_t, t) │
                    └─────────────────┘
                              │
                    ┌─────────────────┐
                    │  NoPropBlock[j]  │──→ pred_j [B, S, D]
                    │  forward(x_t, t) │  (with retain_graph)
                    └─────────────────┘
                              │
                    pred = mean(pred_i, pred_j)
                    loss = MSE(pred, x_emb)
                    backward + local_step
```

### 2.3 Data Flow (Generation)

```
                    ┌─────────────────────┐
                    │  Initial:            │
                    │  prompt_emb + N(0,1) │──┐
                    │  gen_emb + N(0,1)    │──┤
                    └─────────────────────┘  │
                                              │
                    for step in max_steps:    │
                        │                    │
                    Denoise → pred_emb        │ (self-conditioning
                        │                    │  feeds previous pred)
                    logits = pred_emb @ W^T   │
                        │                    │
                    Temperature shaping      │
                        │                    │
                    SampleFromPredictions    │
                        │  (entropy ordering) │
                    Accept high-confidence,  │
                    re-noise low-confidence   │
                        │                    │
                    Re-embed + add noise     │
                        │                    │
                    └────────────────────┘

```

---

## 3. Module Reference

### 3.1 NoPropBlock

**File:** `src/noprop_block.py`

Single transformer block with local training. The fundamental compute unit of the Diffusion Mesh.

#### Class: `NoPropBlock(embed_dim, num_heads=4, ff_mult=4)`

| Method | Signature | Description |
|---|---|---|
| `forward` | `(x, t, context=None) → Tensor` | Input projection + time emb → self-attn → residual+norm → FFN → norm |
| `compiled_forward` | `(x, t, context=None) → Tensor` | `torch.compile(mode="max-autotune-no-cudagraphs")` wrapper |
| `local_loss` | `(pred, target, t=None) → Tensor` | MSE loss, optionally SNR-weighted |
| `configure_optimizer` | `(lr=1e-3, weight_decay=0.0)` | Creates per-block `AdamW` optimizer |
| `local_step` | `(pred, target, retain_graph, t) → float` | Full train step: backward → clip → step → loss |

#### Class: `LoRALinear(base, rank=16, alpha=16)`

Low-rank adapter for fine-tuning. Forward: `base(x) + (x @ lora_a @ lora_b) * (alpha / rank)`.

#### Class: `SinusoidalTimeEmbedding(dim, max_period=10000.0)`

Standard sinusoidal positional encoding, applied to time `t`.

#### Functions

| Function | Description |
|---|---|
| `inject_lora_into_block(block, rank=16, alpha=16)` | Replaces all Linear layers with LoRALinear |
| `lora_parameters(module)` | Yields LoRA A/B parameters only |
| `checkpoint_atomic(save_dir, step, ...)` | Temp-file + rename atomic save |
| `load_checkpoint(path)` | `torch.load(weights_only=True)` |
| `snr_grad_weight(t, eta=1.0)` | Cosine-based SNR gradient weighting |

### 3.2 MeshRouter

**File:** `src/mesh_router.py`

Expert routing via cosine-similarity through a shared latent space. Supports planning, health monitoring, mitosis, merging, and pruning.

#### Class: `MeshNode`

| Attribute | Type | Description |
|---|---|---|
| `node_id` | str | Unique identifier (e.g. `e0000`) |
| `anchor_embedding` | Tensor | `d_model`-dim normalized anchor |
| `rolling_loss` | list[float] | Loss history (window=100) |
| `mitosis_threshold` | float | Trigger for mitosis (default 0.5) |
| `metadata` | ExpertMetadata | accuracy, latency, usage, etc. |

| Method | Description |
|---|---|
| `update_loss(loss)` | Appends to rolling window |
| `sustained_high_error()` | True if recent mean > threshold |

#### Class: `MeshRouter(top_k=3, qb_enabled=True, d_model=1024, n_domains=16)`

| Method | Returns | Description |
|---|---|---|
| `register_node(node, graph_path)` | — | Register expert node |
| `remove_node(node_id)` | — | Remove expert |
| `route(query)` | [(nid, node, score)] | Cosine-sim routing through latent space + QB betas |
| `route_with_planning(x)` | [(nid, node, score)], plan_dict | Full planning: intent → difficulty → graph → execution |
| `check_mitosis(node_id)` | str or None | Create child with perturbed anchor |
| `merge_similar(threshold=0.95)` | — | Merge near-identical anchors |
| `prune_dead(max_idle=5000, min_window=10)` | — | Remove idle/unhealthy experts |
| `latent_consistency_loss(batch_latents)` | Tensor | Cosine-sim consistency loss |

**Sub-modules:**

| Class | Purpose |
|---|---|
| `UniversalLatentSpace(d_model, d_latent=256)` | Projects hidden states to shared latent space |
| `ExpertAdapter(d_latent, d_expert)` | Bidirectional encode/decode between latent and expert spaces |
| `IntentDetector(d_model)` | 2-layer MLP: classifies input domain (16 types) |
| `DifficultyEstimator(d_model)` | Regression: outputs difficulty score [1, 7] |
| `ExpertGraph` | Hierarchical tree for semantic expert traversal |
| `RouterPlanner(d_model)` | Combines intent + difficulty + graph for full planning |
| `ExecutionGraph` | Builds step-by-step execution plan from expert selections |
| `ConfidenceEngine` | Fact confidence tracking with contradictions |
| `HealthMonitor` | Per-node latency/accuracy tracking, unhealthy detection |

### 3.3 DiffusionDecoder

**File:** `src/diffusion_decoder.py`

Iterative denoising decoder for text generation. Ports the DiffusionGemma (JAX) sampler pattern to PyTorch, adapted for embedding-level denoising.

#### Class: `DiffusionDecoder(router, blocks, embed, top_k=2, ...)`

| Method | Returns | Description |
|---|---|---|
| `denoise_embeddings(noisy_emb, t, sc_emb)` | Tensor [B, L, D] | Self-conditioning → router → expert forward → ensemble mean |
| `generate(canvas_length, ...)` | Tensor or (Tensor, list) | Full diffusion generation loop |
| `generate_with_prompt(prompt_text, tokenizer, ...)` | str | Convenience: encode → generate → decode |

#### `generate()` Parameters

| Parameter | Default | Description |
|---|---|---|
| `canvas_length` | — | Total output sequence length |
| `prompt_ids` | None | Optional [B, L] prompt tokens |
| `max_denoising_steps` | 48 | Number of iterative denoising steps |
| `temperature_config` | (0.4, 0.8, 1.0) | (min_temp, max_temp, exponent) |
| `entropy_bound` | 0.1 | Confidence threshold for token acceptance |
| `return_trajectory` | False | Return list of canvas at each step |
| `vocab_range` | None | (lo, hi) to constrain token generation |

#### Support Classes

| Class | Purpose |
|---|---|
| `LinearSchedule` | Identity noise schedule: `noise_prob(np) = np` |
| `DiffusionProcess` | Multinomial token noise: `add_noise_to_tokens`, `get_initial_sample` |
| `SampleFromPredictions` | Entropy-based acceptance: sort by entropy, accept where cumulative ≤ bound, re-noise rest |
| `AnnealingTemperatureShaper` | Power-law temperature: `T(np) = minT + (maxT-minT) × (1 - (1-np)^exponent)` |
| `SelfConditioning(nn.Module)` | `pre_norm → FFN → post_norm`, matches DiffusionGemma |
| `NoEarlyStop` | Never stop early |
| `TokenStabilityEarlyStop` | Stop when argmax matches previous canvas |
| `EntropyEarlyStop(threshold=0.005)` | Stop when mean entropy ≤ threshold |
| `ChainedEarlyStop(fns)` | Stop when ALL sub-stoppers agree |

#### Generation Loop Pseudocode

```
canvas = [prompt_tokens || random_tokens]
noisy_emb = embed(canvas) + N(0, 1)     # match training distribution
sc_emb = zeros

for step in range(max_denoising_steps):
    t = 1.0 - step / max_steps

    # Denoise
    pred_emb = model(noisy_emb, t, sc_emb)
    prompt_emb = embed(prompt_tokens)     # keep prompt pinned

    # Map to logits
    logits = pred_emb @ embed.weight.T   # weight-tied LM head
    shaped = temperature(logits, t)

    # Select tokens by confidence
    new_tokens = sample_from_predictions(shaped, canvas)
    canvas[~prompt_mask] = new_tokens[~prompt_mask]

    # Prepare next step
    clean_emb = embed(canvas)
    noisy_emb = clean_emb + N(0, 1-t_next)
    sc_emb = pred_emb.detach()

return canvas
```

---

## 4. Training Pipeline

**Script:** `scripts/train_core.py`

### 4.1 Starting a Fresh Training

```powershell
# Sanity run (synthetic data, 100 steps)
python train_core.py --steps 100 --synthetic --checkpoint-dir checkpoints/sanity

# Full training on curriculum data
python train_core.py --steps 100000 --data curriculum_tokens.jsonl --checkpoint-dir checkpoints/core_100m

# With overrides
python train_core.py --steps 5000 --data curriculum_tokens.jsonl --canvas 512 --batch-size 2 --lr 1e-4
```

### 4.2 Resuming from Checkpoint

```powershell
python train_core.py --resume checkpoints/core_100m/step_latest.pt --steps 10000 --data curriculum_tokens.jsonl
```

CLI overrides (`--steps`, `--checkpoint-dir`, `--lr`, etc.) take precedence over the checkpoint's saved config.

### 4.3 Noise Schedule

Each training step samples `t ~ Uniform(0, 1.0)`. Noise is applied as:

```python
noise = torch.randn_like(x_emb) * (1 - t)
x_t = x_emb + noise
```

The model receives the same `t` as a two-dimensional tensor via its time embedding, learning to denoise at all noise levels. Validation evaluates at three fixed levels: `t = 0.0, 0.3, 0.7`.

### 4.4 Gradient Isolation

Each expert has its own `AdamW` optimizer. Gradient flow is isolated per-expert:

```python
for i, (nid, _, _) in enumerate(active_list):
    block = blocks[nid]
    pred = block(x_t, t_2d)
    loss = block.local_loss(pred, target)
    block.optimizer.zero_grad()
    retain = (i < len(active_list) - 1)
    loss.backward(retain_graph=retain)      # retain_graph for multi-expert
    torch.nn.utils.clip_grad_norm_(block.parameters(), 1.0)
    block.optimizer.step()
```

### 4.5 Flat-Packing Collation

Documents are concatenated into rows of up to `canvas_len` tokens, eliminating padding. The `collate_packed` function:

```
Docs:  [doc_A(500), doc_B(700), doc_C(600), doc_D(400)]
              │
              ▼
Rows:  [A(500)|B(700)]  → canvas(1200)  [saved: 848 padding tokens]
       [C(600)|D(400)]  → canvas(1000)  [saved: 1048 padding tokens]
```

### 4.6 Checkpointing

```
checkpoints/
├── core_100m/
│   ├── step_500.pt       # Full state (model + optimizer + config)
│   ├── step_1000.pt
│   ├── step_2000.pt
│   ├── step_latest.pt    # Always most recent
│   └── metrics.json      # Validation metrics log
```

Atomic save: `step_N.tmp` → `os.replace()` → `step_N.pt` (crash-safe).

### 4.7 Training Metrics (63M model, curriculum data)

| Step | Train Loss | Val Loss (avg t=0/0.3/0.7) | Throughput |
|---|---|---|---|
| 500 | ~0.05 | 8.4e-5 | ~17K tok/s |
| 1000 | ~0.03 | 3.9e-5 | ~17K tok/s |
| 1500 | ~0.02 | 2.2e-5 | ~17K tok/s |
| 2000 | ~0.01 | 1.7e-5 | ~17K tok/s |
| 3000 | ~0.01 | 1.3e-5 | ~17K tok/s |
| 4000 | ~0.01 | 9.7e-6 | ~17K tok/s |
| 5000 | ~0.01 | — | ~17K tok/s |

### 4.8 Configuration File

```python
@dataclass
class CoreConfig:
    d_model: int = 1024
    n_heads: int = 16
    ff_mult: int = 4
    n_experts: int = 7
    top_k: int = 2
    canvas_len: int = 2048
    batch_size: int = 2
    lr: float = 1e-4
    weight_decay: float = 1e-5
    max_steps: int = 100_000
    save_every: int = 2_000
    val_every: int = 500
    checkpoint_dir: str = "checkpoints/core_100m"
    resume_path: str = ""
    seed: int = 42
```

---

## 5. Evaluation

**Script:** `scripts/eval_core.py`

### 5.1 Reconstruction Evaluation

```powershell
python eval_core.py --checkpoint checkpoints/core_100m_text/step_latest.pt
```

Reports:
- **MSE Loss** at t=0.0, 0.3, 0.7
- **Cosine Similarity** between pred and target embeddings
- **Token Accuracy@1** (argmax of `pred @ embed.weight.T` matches original token)
- **Expert Activation Distribution** (which experts are used)

Example output (step 500, synthetic data):
```
MSE:           0.455567
Cosine Sim:    0.7467
Token Acc@1:   1.0000 (100.0%)
Expert activations: 6 experts used
```

### 5.2 Interactive Reconstruction Demo

```powershell
python eval_core.py --checkpoint checkpoints/core_100m_text/step_latest.pt --interactive
```

Type text and see how well the model reconstructs it from noise at different levels.

### 5.3 Text Generation (Diffusion Decoder)

**Script:** `scripts/test_decoder.py`

```powershell
# Automatic mode with prompt
python test_decoder.py --checkpoint checkpoints/core_100m_text/step_latest.pt --canvas 128 --steps 64 --prompt "What is"

# No prompt (pure generation)
python test_decoder.py --checkpoint checkpoints/core_100m_text/step_latest.pt --canvas 64 --steps 32

# Interactive mode
python test_decoder.py --checkpoint checkpoints/core_100m_text/step_latest.pt --interactive
```

Key observations (curriculum-trained model at step 5000):
- Prompt faithfully preserved
- Output consists of valid characters matching English letter frequencies
- Letter bigrams and spacing patterns partially learned
- To achieve coherent text: requires orders of magnitude more training data (billions of tokens) or a language modeling head trained with next-token prediction

---

## 6. Benchmarking

### 6.1 Deep Benchmark (`benchmark_deep.py`)

Training vs inference comparison across sequence lengths, with per-component breakdown.

#### Results Summary

| Seq Len | Mode | Time/Step (ms) | Tokens/s | Embed | Router | Forward | Backward |
|---|---|---|---|---|---|---|---|
| 128 | Inf | 1.8 | 140,800 | 0.6ms | 0.1ms | 0.6ms | — |
| 128 | Train | 5.2 | 48,600 | — | — | 0.6ms | 4.6ms |
| 512 | Inf | 5.1 | 200,000 | 1.5ms | 0.1ms | 3.1ms | — |
| 512 | Train | 28.5 | 36,000 | — | — | — | 18ms |
| 1024 | Inf | 11.8 | 173,000 | 2.1ms | 0.2ms | 9.6ms | — |
| 1024 | Train | 63.1 | 16,200 | — | — | — | 44ms |
| 2048 | Inf | 29.3 | 139,000 | 3.3ms | 0.4ms | 25.5ms | — |
| 2048 | Train | 148.5 | 13,800 | — | — | — | 71ms |

**Key insight:** Backward pass dominates training time (60%+ at 2048). Optimizer step is flat ~10ms regardless of sequence length.

### 6.2 Backward Breakdown (`benchmark_backward.py`)

Per-component backward timing using PyTorch hooks with CUDA events.

#### Component Breakdown at 2048 (per expert)

| Component | Time (ms) | Share |
|---|---|---|
| Attention backward | 12.95 | 62.1% |
| FF layer 0 backward | 4.14 | 19.9% |
| FF layer 2 backward | 3.28 | 15.7% |
| Norm 1 backward | 0.20 | 1.0% |
| Norm 2 backward | 0.17 | 0.8% |
| Input proj backward | 0.08 | 0.4% |
| Time emb backward | 0.02 | 0.1% |
| **Total** | **20.84** | **100%** |

**Actionable insight:** FlashAttention would help at long sequences (62% of backward). FlashFFN at short-medium sequences.

### 6.3 Real-Doc Benchmark (`benchmark_real_docs.py`)

Packed vs unpacked at 2048 canvas with variable-length real documents (~2232 avg tokens).

| Method | Tok/s | Step Time | Padding | Expert Entropy |
|---|---|---|---|---|
| Packed (batch=32) | 28,819 | 64.2ms | 0% | 2.10 |
| Unpacked (batch=2) | 34,270 | 55.4ms | 12% | 1.89 |

**Key insight:** Unpacked is 19% faster at 2048 because each row processes one full batch. Packing only helps when average doc length << canvas length (saves more padding than it loses in batch parallelism).

### 6.4 Running Benchmarks

```powershell
# Deep benchmark (training vs inference, 128-2048)
python benchmark_deep.py

# Backward breakdown (per-component)
python benchmark_backward.py

# Real-doc benchmark (packed vs unpacked)
python benchmark_real_docs.py
```

Results saved to `benchmarks/*.csv`.

---

## 7. Data Preparation

### 7.1 Curriculum Data

**Script:** `scripts/prepare_curriculum.py`

Converts multi-phase curriculum JSONL (from `curriculum_data/phaseXX_*/samples.jsonl`) to char-level token sequences.

```powershell
# Default: phases 0-5
python prepare_curriculum.py --output curriculum_tokens.jsonl

# Custom phases
python prepare_curriculum.py --phases 0,1,2,3,4,5 --max-len 2048
```

**Data format:** Extracts `input`, `analysis`, `final_answer`, `verification` fields, concatenates with ` | ` separator, maps each character to integer ID [4, 255].

**Available phases:**

| Phase | Name | Sequences | Characters |
|---|---|---|---|
| 0 | Foundation | 255 | 450,782 |
| 1 | Knowledge | 101 | 164,043 |
| 2 | Relationships | 61 | 124,928 |
| 3 | Reasoning | 20 | 38,048 |
| 4 | Programming | 452 | 924,798 |
| 5 | Mathematics | 1 | 2,048 |
| **Total** | | **890** | **1,704,647** |

### 7.2 Synthetic Data

Built into `train_core.py` via `SyntheticDataset(n, min_len, max_len, seed)`. Creates random integer sequences in range [4, 1000) for debugging.

### 7.3 Text Data

`TextDataset(jsonl_path, max_len)` loads pre-tokenized JSONL with `input_ids` or `tokens` fields.

---

## 8. Configuration Guide

### 8.1 Core Architecture

| Parameter | Default | Range | Effect |
|---|---|---|---|
| `d_model` | 1024 | 512–2048 | Embedding dimension |
| `n_heads` | 16 | 4–32 | Attention heads |
| `ff_mult` | 4 | 2–8 | FFN expansion factor |
| `n_experts` | 7 | 2–64 | Number of expert blocks |
| `top_k` | 2 | 1–n_experts | Experts selected per forward |

### 8.2 Training

| Parameter | Default | Effect |
|---|---|---|
| `lr` | 1e-4 | AdamW learning rate |
| `weight_decay` | 1e-5 | AdamW weight decay |
| `canvas_len` | 2048 | Maximum sequence length |
| `batch_size` | 2 | DataLoader batch size |
| `max_steps` | 100,000 | Total training steps |
| `save_every` | 2,000 | Checkpoint interval |
| `val_every` | 500 | Validation interval |

### 8.3 Generation

| Parameter | Default | Effect |
|---|---|---|
| `max_denoising_steps` | 48 | Iterative denoising steps |
| `entropy_bound` | 0.1 | Confidence threshold for token acceptance |
| `min_temperature` | 0.4 | Temperature at noise=0 |
| `max_temperature` | 0.8 | Temperature at noise=1 |
| `temperature_exponent` | 1.0 | Power-law curve shape |
| `vocab_range` | (4, 1000) | Valid token range (synthetic default) |

### 8.4 Memory Contract (from `rules.md`)

Every run enforces:
1. `bitsandbytes` importable
2. Base model loads `load_in_4bit=True` (NF4, `requires_grad=False`)
3. Active mesh nodes run in BF16 with isolated `AdamW`
4. `torch.set_float32_matmul_precision("high")`

---

## 9. Development Guide

### 9.1 Environment Setup

```powershell
uv sync                          # Create .venv with all dependencies
.\scripts\run-noprop.ps1 -Script train_core.py   # Run via launcher
.\scripts\run-tests.ps1          # Run test suite
```

### 9.2 Linting

```powershell
ruff check src/ scripts/
ruff format src/ scripts/ --check
```

Configured in root `pyproject.toml`: line-length=100, target=py312, selects E/F/I/N/W.

### 9.3 Testing

Tests in `src/tests/`:

| Test File | What It Verifies |
|---|---|
| `test_zero_grad.py` | Frozen 4-bit grad isolation, BF16 nodes get gradients |
| `test_adversarial.py` | Variable shapes, extreme contexts, high-entropy noise |
| `test_block.py` | Forward, local_step, time embedding, checkpoint |
| `test_router.py` | Registration, routing, mitosis trigger, metadata |
| `test_turboquant.py` | Orthogonal matrix, centroid fitting, QJL sign |
| `test_speculator.py` | MTP head, multi-token draft, confidence scores |

```powershell
.\scripts\run-tests.ps1                             # All tests
.\scripts\run-tests.ps1 -TestPath ...\test_block.py -- -v  # Single test verbose
```

### 9.4 Adding a New Script

1. Place in `NoProp/scripts/`
2. Use `sys.path.insert(0, ...)` to import from `NoProp/src/`
3. Follow existing patterns for argument parsing and device management
4. Run via `.\scripts\run-noprop.ps1 -Script your_script.py`

### 9.5 Known Issues & Quirks

- **`retain_graph=True`** is required in multi-expert backward when multiple experts process the same input tensor
- **`torch.backends.cudnn.deterministic = True`** ensures bit-exact reproducibility
- **LazyLinear** needs a dummy forward pass before parameter counting
- **Resume** CLI overrides are re-applied after checkpoint config restoration
- **VOCAB=151643** is sized for a full LLM tokenizer; char-level training uses only IDs 4–255

---

## 10. Upcoming Features

### 10.1 High Priority

| Feature | Description | Depends On |
|---|---|---|
| **Full-scale language model head** | Replace weight-tied LM head with a proper `nn.Linear(d_model, vocab_size)` trained with cross-entropy on token-level diffusion (matching DiffusionGemma) | Training pipeline changes |
| **DiffusionGemma training** | Train the NoPropBlock + LM head with token-level noise (replace tokens with random ones) and cross-entropy loss on corrupted positions, instead of MSE on embeddings | Architecture change |
| **Large-scale text data** | Train on billions of tokens with a proper tokenizer (e.g., Gemma tokenizer) to learn language structure | Tokenizer, data pipeline |
| **Sliding-window attention** | Extend generation beyond canvas_length using KV-cache-style sliding window | MeshRouter, Decoder |

### 10.2 Medium Priority

| Feature | Description | Depends On |
|---|---|---|
| **FlashAttention integration** | Replace `nn.MultiheadAttention` with FlashAttention for 2-3x speedup at long sequences | — |
| **BF16 training** | Enable `torch.autocast` for BF16 to reduce VRAM and improve throughput | — |
| **Distributed training** | DDP/FSDP across multiple GPUs for larger models | — |
| **Multi-epoch curriculum** | Train through all 11+ curriculum phases in order, with difficulty-based sample weighting | Data pipeline |
| **Top-k entropy regularization** | Encourage router to distribute load evenly across experts | MeshRouter |

### 10.3 Architecture Evolution (from `MESH_REFACTOR_PLAN.md`)

| Phase | Description | Status |
|---|---|---|
| **Phase 0** | DSpark Windows Port (NCCL→Gloo, Triton conditional) | Not started |
| **Phase 1** | Backbone Scaling: Configurable 500M+ backbone with GQA CanvasTransformer (AF.md #1) | Not started |
| **Phase 2** | Universal Latent Space: `d_model → d_latent(256)` projection with bidirectional adapters (AF.md #2) | Skeleton exists |
| **Phase 3** | Hierarchical Router + Planner: IntentDetector + DifficultyEstimator + ExpertGraph (AF.md #3/#4) | Skeleton exists |
| **Phase 4** | Global Cognitive Layer: transformer on latent space + consensus mechanism (AF.md #7) | Not started |
| **Phase 5** | Expert Lifecycle: Create → Evaluate → Improve → Merge → Compress → Archive (AF.md #5) | Skeleton exists |

### 10.4 Research Directions

| Direction | Description |
|---|---|
| **Scaling laws for denied gradient** | How does the mesh scale with number of experts vs single large model? |
| **Mitosis dynamics** | Does mitosis create functionally specialized experts, or duplicates? |
| **Self-conditioning ablation** | What is the contribution of self-conditioning in embedding-level vs token-level diffusion? |
| **NoProp vs backprop at scale** | At what model size does independent local training match global backprop quality? |
| **Embedding-level vs token-level diffusion** | Which noise process works better for the mesh architecture? |

---

## 11. Appendix: Benchmark Data

### 11.1 Deep Benchmark CSV

File: `benchmarks/deep_benchmark.csv`

```
seq_len,mode,step_time_ms,tok_s,embed_ms,router_ms,forward_ms,backward_ms,optim_ms,util_pct,vram_gb
128,inference,1.82,140800,0.6,0.1,0.6,0.0,0.0,72.5,3.67
128,training,5.27,48600,0.6,0.1,0.6,4.6,11.2,68.2,3.46
512,inference,5.12,200000,1.5,0.1,3.1,0.0,0.0,74.8,3.62
512,training,28.46,36000,1.6,0.1,3.1,18.0,11.8,78.3,3.55
1024,inference,11.82,173200,2.1,0.2,9.6,0.0,0.0,71.2,3.75
1024,training,63.12,16200,2.1,0.2,9.6,43.6,11.3,75.6,3.65
2048,inference,29.34,139600,3.3,0.4,25.5,0.0,0.0,69.8,3.71
2048,training,148.50,13800,3.3,0.4,25.5,70.8,10.5,73.4,3.58
```

### 11.2 Backward Breakdown CSV

File: `benchmarks/backward_breakdown.csv`

```
seq_len,attn_ms,ff_0_ms,ff_2_ms,norm1_ms,norm2_ms,proj_ms,time_ms,total_ms
128,0.25,0.12,0.11,0.01,0.01,0.01,0.00,0.51
512,1.55,0.82,0.61,0.05,0.04,0.02,0.00,3.12
1024,4.38,2.15,1.72,0.11,0.09,0.05,0.01,8.52
2048,12.95,4.14,3.28,0.20,0.17,0.08,0.02,20.84
```

### 11.3 Real-Doc Benchmark CSV

File: `benchmarks/real_doc_benchmark.csv`

Contains packed vs unpacked timing, padding ratios, expert entropy, and GPU utilization.

---

*Documentation generated from codebase at `E:\my apps\NN\NoProp\`. Last updated: 2026-07-13.*
