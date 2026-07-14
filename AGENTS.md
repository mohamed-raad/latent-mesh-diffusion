# AGENTS.md — NN monorepo

Three independent subprojects in one tree. Always identify which one you are working in before editing or running commands.

---

## DeepSpec — Speculative-decoding draft models

Training and evaluation of DSpark, DFlash, and Eagle3 draft models.

### Entrypoints

| Command | What it does |
|---|---|
| `bash scripts/train/train.sh` | Train a draft model |
| `bash scripts/eval/eval.sh` | Evaluate a trained draft checkpoint |
| `python scripts/data/prepare_data.sh` | Full data-preparation pipeline |

### Training

`train.py` uses `torch.multiprocessing.spawn` — NOT `torchrun`. RANK/WORLD_SIZE mean node_rank/node_count (WORLD_SIZE=1 for single-node).

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export RANK=0
export WORLD_SIZE=1

python train.py \
    --config config/dspark/dspark_qwen3_4b.py \
    --opts "data.target_cache_path=${HOME}/.cache/deepspec/qwen3_4b_target_cache"
```

Override any config field via `--opts "<dotted.key.path>=<value>"` (repeatable). Checkpoints land in `~/checkpoints/<project_name>/<exp_name>/step_*`.

Available configs: `config/{dspark,dflash,eagle3}/d{spark,flash,eagle3}_{qwen3_4b,qwen3_8b,qwen3_14b,gemma4_12b}.py`.

### Evaluation

```bash
bash scripts/eval/eval.sh
# or directly:
python eval.py \
    --target_name_or_path Qwen/Qwen3-4B \
    --draft_name_or_path ~/checkpoints/deepspec/dspark_block7_qwen3_4b/step_latest
```

Uses `torch.multiprocessing.spawn` (same pattern as training). Evaluates on all 9 benchmarks in `eval_datasets/`.

### Data preparation (3 stages)

1. **Download & split** — `python scripts/data/download_and_split.py`
2. **Regenerate answers** — requires a running SGLang server (`bash scripts/data/launch_sglang_server.sh`; install separately: `pip install "sglang[all]"`)
3. **Build target cache** — `python scripts/data/prepare_target_cache.py`

The target cache is ~38 TB for the default `Qwen/Qwen3-4B` setting. Make sure the output filesystem has adequate space.

### Quirks

- `sglang` is NOT in `requirements.txt` — install separately.
- Configs are Python files with a `finalize_cfg(cfg)` hook that sets `checkpoint_dir` and `tensorboard_dir` from `BASE_CKPT_DIR`/`BASE_TB_DIR`.
- `pip install -r requirements.txt` pins `torch==2.9.1`, `transformers==5.10.2`.

---

## NoProp — Community NoProp Implementation

PyTorch-based image classification without backpropagation.

### Entrypoints

| Command | What it does |
|---|---|
| `python src/noprop_simple.py --dataset cifar10 --backbone resnet50` | Train on CIFAR-10 (default 400 epochs) |
| `python src/noprop_simple.py --dataset cifar100 --backbone resnet152` | Train on CIFAR-100 |
| `python src/nopropct_mnist.py` | Train on MNIST |

Datasets: `mnist`, `cifar10`, `cifar100`. Backbones: `resnet18`, `resnet50`, `resnet152`.

---

## gemma — Google DeepMind Gemma (reference library)

JAX-based LLM library. **Do not edit unless instructed** — this is a vendored upstream dependency. Key facts:

- Entry: `from gemma import gm` (NOT `import gemma as gm`; the `__init__.py` catches the wrong form and raises an error).
- Install: `pip install gemma` or `pip install -e .` from the `gemma/` directory (requires JAX first).
- Requires **Python >= 3.12** and JAX.
- Formatting: `pyink` (Google style, 80 cols, 2-space indent, `pip install -e ".[dev]"`).
- Testing: `pytest` with `pytest-xdist`.

---

## Root architecture docs (Diffusion Mesh project context)

These `.MD` files at the repo root describe the custom "Diffusion Mesh" architecture that this research project is building. Treat them as design specifications:

| File | Content |
|---|---|
| `SYSTEM-PROMPT.MD` | Full architectural paradigm — Diffusion Mesh, local training, node mitosis, TurboQuant, MTP heads |
| `rules.md` | Strict source-code contracts — no placeholders, module isolation, adversarial test scaffolding, async checkpointing |
| `NOPROB REWRITING.MD` | Blueprint for adapting NoProp mechanics into the custom mesh |
| `functional reference library.MD` | Strategy for using Google's gemma repo as a functional reference rather than a wrapper |
| `AF.md` | Architectural critique — 7 critical changes (backbone, latent space, router, life cycle, memory, global layer, etc.) |
| `MESH_REFACTOR_PLAN.md` | **Implementation plan** — 6-phase refactor with file-level steps, based on AF.md |
| `MESH_REFACTOR_CHECKLIST.md` | **Step-by-step checklist** — track progress through all 40+ tasks |
| `dataset-structure.md` | Curriculum design specification — 11-phase progression, JSONL sample format |
| `generate_curriculum.bat` | **One-click launcher** — generates dataset in JSONL format via llama-server (port 8080) |

---

## Workspace layout (uv-based monorepo)

Only `NoProp/` is an active workspace member. `DeepSpec/` and `gemma/` are **read-only reference vaults** — never edit.

### Key files

| File | Purpose |
|---|---|
| `pyproject.toml` (root) | Virtual workspace root, CUDA 12.8 index, ruff config |
| `NoProp/pyproject.toml` | Package `noprop-mesh` with torch>=2.5, bitsandbytes, transformers |
| `.venv/` | Shared uv venv (created on first `uv sync`) |
| `uv.lock` | Single lockfile for entire workspace |

### Launcher scripts

| Script | What it does |
|---|---|
| `scripts\run-noprop.ps1` | Activates venv, verifies bitsandbytes, runs a script from `NoProp/src/` |
| `scripts\run-tests.ps1` | Runs `pytest` on `NoProp/src/tests/` through the venv |
| `scripts\run-wsl-ref.ps1` | Opens WSL2 shell in `DeepSpec/` or `gemma/` for read-only inspection |
| `scripts\wsl-setup.ps1` | One-time: bootstraps WSL2 Ubuntu with Python 3.12 + uv + CUDA toolkit |

### Commands

```powershell
# Run a mesh module
.\scripts\run-noprop.ps1 -Script noprop_block.py

# Run all tests
.\scripts\run-tests.ps1

# Run a specific test with verbose output
.\scripts\run-tests.ps1 -TestPath NoProp/src/tests/test_zero_grad.py -Verbose

# Inspect reference code in WSL2
.\scripts\run-wsl-ref.ps1 -Target DeepSpec
```

### Dataset generation (curriculum, JSONL format)

```bat
:: Make sure llama-server is running (port 8080), then:
generate_curriculum.bat          # All 11 phases, 100 samples each
generate_curriculum.bat 3,4 500  # Phases 3+4 only, 500 each
generate_curriculum.bat 0 1000 --no-resume  # Phase 0, 1000 fresh
```

Or directly from Python:
```powershell
uv run --no-sync --package noprop-mesh python scripts/curriculum_generator.py --phases 0,1,2 --samples 500
```

### Memory contract (8 GB VRAM ceiling)

Every run launched through `run-noprop.ps1` enforces:
1. `bitsandbytes` is importable (abort if missing).
2. Base model loads with `load_in_4bit=True` (NF4 format, `requires_grad=False`).
3. Active mesh nodes run in BF16 with isolated `torch.optim.AdamW`.
4. `torch.set_float32_matmul_precision("high")` is set.
5. TextVAE is **optional** (`use_vae=False` default) — no VRAM increase unless explicitly enabled.

See `rules.md` for the full gradient-isolation and checkpointing contract.

### Mesh source modules (`NoProp/src/`)

| Module | Responsibility |
|---|---|
| `mesh_router.py` | **Latent Mesh Diffusion Computer** — UniversalLatentSpace (N=64-96 nodes, VAE upgrade path), LatentGraph + LatentNode (state/memory/confidence/importance/edges), MeshOfThought (graph-based reasoning), ConsensusEngine (confidence-weighted), LatentDecoder (latent→embedding), LatentMemory (graph store/retrieve), MitosisAnalyzer (semantic clustering), ExpertGraph + RouterPlanner + ExecutionGraph, CodingMode (parallel codegen), 3 latent losses, LatentMeshConfig (12-phase config) |
| `noprop_block.py` | BF16 single-block layer + local AdamW + atomic checkpointing |
| `turboquant_attention.py` | PolarQuant rotation + Lloyd-Max centroids + QJL residual |
| `dspark_speculator.py` | MTP heads + confidence verifier + speculative decode loop |
| `text_vae.py` | **Text VAE** (Cola-DLM inspired) — patch-conv encoder, SwiGLU transformer, diagonal Gaussian latent, hierarchical causal masking, decoder with unpatch. Optional upgrade to UniversalLatentSpace via `use_vae=True` |
| `diffusion_canvas.py` | **Diffusion renderer** — CanvasTransformer backbone with GQA, UniformStateDiffusion, adaptive entropy stopping, conditional/ unconditional generation. Stays as the final text renderer (latent→embedding→diffusion→text) |
| `model_sizes.py` | SizePreset dataclass with tiny/small/standard/xstandard/large presets |
| `train_mesh.py` | MeshTrainer — orchestrates blocks, router, speculator, canvas, checkpointing, mitosis |

### Test suite (`NoProp/src/tests/`) — 127 tests, all passing

| Test | What it verifies |
|---|---|
| `test_text_vae.py` | Gaussian distribution, VAEBlock causal, TextVAE forward/hierarchical/deterministic, loss/backward/gradient flow, causal mask builder, variable length inputs (11 tests) |
| `test_zero_grad.py` | Frozen 4-bit params have `param.grad is None`; active BF16 nodes get non-zero gradients |
| `test_adversarial.py` | Variable shapes, extreme context windows, high-entropy noise, router edge cases (7 tests) |
| `test_block.py` | noprop_block forward, local_step, time embedding, atomic checkpoint |
| `test_diffusion_canvas.py` | CanvasBlock, UniformStateDiffusion, DiffusionCanvas init/denoise/generate/multiblock/loss/entropy stopping, time embedding (9 tests) |
| `test_e2e_pipeline.py` | Full Obsidian→Mesh→Training→Canvas→Checkpoint pipeline (10 tests) |
| `test_mesh.py` | MeshTrainer init, training loop, loss decreases, checkpoint/resume, inference, mitosis, router invariance (8 tests) |
| `test_mesh_canvas_integration.py` | Canvas disabled/enabled, generate text (single + multiblock), error handling, inference unchanged, summary reporting, checkpoint save/load (9 tests) |
| `test_router.py` | Registration, routing, mitosis trigger, metadata loading |
| `test_mitosis_hook.py` | Hook init, evaluate, mitosis triggers, spawning, state dict roundtrip (14 tests) |
| `test_obsidian_compiler.py` | Wiki parsing, markdown cleaning, vault scanning, sparse adjacency, static embedding, inject into router, compile pipeline (18 tests) |
| `test_speculator.py` | MTP head, multi-token predictor/draft, confidence verifier, speculation loop, loss, curriculum dataset (12 tests) |
| `test_turboquant.py` | Random orthogonal, Lloyd-Max centroids, PolarQuant, QJL, streaming quantize, cross-layer KV cache, attention compression (19 tests) |

### Architecture overview: Latent Mesh Diffusion Computer

The original Diffusion Mesh has been transformed into a **Latent Mesh Diffusion Computer** — a 12-phase upgrade that adds structured latent reasoning on top of the existing mesh/diffusion pipeline:

1. **UniversalLatentSpace** — Cross-attention encoder compresses `[B,S,d_model]` → `[B,N,d_latent]` semantic nodes (N=64-96), with optional TextVAE upgrade (`use_vae=True`)
2. **LatentGraph + LatentNode** — Reasoning graph with state, memory, confidence, importance, parents/children/neighbors, owner tracking
3. **MeshOfThought** — Graph-based reasoning replacing chain-of-thought: assign experts → process latents → form connections → confidence-weighted consensus
4. **Router→graph nodes** — `route_nodes()` maps LatentNodes to experts by state similarity
5. **Experts→latent→latent** — Experts process latent vectors directly (not embeddings)
6. **ConsensusEngine** — Confidence-weighted consensus (not mean): highest-confidence vote, agreement detection, conflict detection
7. **LatentDecoder** — Learned cross-attention maps `[B,N,d_latent]` → `[B,S,d_model]` embeddings for diffusion renderer
8. **LatentMemory** — Graph store/retrieve/search/merge for cumulative reasoning across sessions
9. **MitosisAnalyzer** — Clusters low-confidence latent nodes by cosine similarity, proposes specialized expert anchors
10. **Three losses** — `semantic_consistency_loss` (1-cos_sim), `consensus_loss` (pairwise cos distance), `reconstruction_loss` (embedding→latent→embedding cycle MSE)
11. **CodingMode** — CodeNode graph for parallel project generation (files→classes→functions→tests)
12. **LatentMeshConfig** — Unified config with `latent_nodes=96`, `latent_heads=8`, `latent_depth=2`, `d_latent=256`, `use_vae`, loss weights, etc. Pass to `MeshRouter(latent_config=cfg)`

### How the model works (data flow)

```
Input tokens [B, S]
    │
    ▼
token_embedding → [B, S, d_model]
    │
    ├─── UniversalLatentSpace (cross-attention queries + VAE option)
    │    └─── [B, N, d_latent] semantic latent nodes
    │
    ├─── MeshOfThought (iterative reasoning graph)
    │    ├── route nodes to experts
    │    ├── experts update latents (latent→latent, no embedding)
    │    ├── form graph connections (similarity-based)
    │    └── confidence-weighted consensus
    │
    ├─── LatentDecoder (cross-attention)
    │    └─── [B, S_out, d_model] reconstructed embeddings
    │
    ├─── Diffusion renderer (CanvasTransformer + UniformStateDiffusion)
    │    └─── logits → argmax → output tokens
    │
    └─── MTP speculator (multi-token prediction heads)
         └─── draft tokens for speculative decoding
```

**Key insight:** The mesh no longer routes raw embeddings to experts. Instead, it:
1. Compresses embeddings into N semantic latent nodes (lossy, structure-preserving)
2. Routes nodes to experts for reasoning in latent space
3. Resolves conflicting expert outputs via confidence-weighted consensus
4. Decodes latents back to embeddings for the diffusion renderer
5. Diffusion generates the final text (unchanged, acts as the "display")

This decouples **reasoning** (latent graph, experts, consensus) from **generation** (diffusion), which is the fundamental architectural difference from standard autoregressive LLMs.

### Toolchain

- **Package manager:** `uv` (Astral, Rust-based). `uv sync` installs everything.
- **Linter/formatter:** `ruff` configured in root `pyproject.toml`.
- **Testing:** `pytest` (`python -m pytest` or `.\scripts\run-tests.ps1`).
