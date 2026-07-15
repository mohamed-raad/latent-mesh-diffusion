# Diffusion Mesh — Complete Documentation

The architecture where **the model designs itself**: nodes are born, learn, merge, and die as training progresses. No static architecture — a self-organizing neural ecosystem.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Quick Start](#2-quick-start)
3. [Dataset Generation](#3-dataset-generation)
4. [Training](#4-training)
5. [Evaluation & Inference](#5-evaluation--inference)
6. [DeepSpec / DSpark (Speculative Decoding)](#6-deepspec--dspark-speculative-decoding)
7. [Module Reference](#7-module-reference)
8. [Configuration](#8-configuration)
9. [Windows-Specific Notes](#9-windows-specific-notes)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      User Query                              │
│                           │                                   │
│                           ▼                                   │
│  ┌──────────────────────────────────────────┐                │
│  │          Intent Detector                  │                │
│  │  (classifies domain, difficulty 1-7)      │                │
│  └──────────────┬───────────────────────────┘                │
│                 │                                             │
│                 ▼                                             │
│  ┌──────────────────────────────────────────┐                │
│  │          Router Planner                   │                │
│  │  (builds execution graph: which experts,  │                │
│  │   which tools, what memory to query)      │                │
│  └──────────────┬───────────────────────────┘                │
│                 │                                             │
│                 ▼                                             │
│  ┌──────────────────────────────────────────┐                │
│  │     Global Cognitive Layer (GCL)          │                │
│  │  ┌─────────┬──────────┬──────────────┐   │                │
│  │  │ Memory  │Expert Gr.│ Tool Manager │   │                │
│  │  └────┬────┴────┬─────┴──────┬───────┘   │                │
│  └───────┼─────────┼────────────┼───────────┘                │
│          │         │            │                             │
│          ▼         ▼            ▼                             │
│  ┌──────────────────────────────────────────┐                │
│  │        Execution Graph                    │                │
│  │  Expert A → Expert B → Consensus → Out   │                │
│  └──────────────┬───────────────────────────┘                │
│                 │                                             │
│                 ▼                                             │
│  ┌──────────────────────────────────────────┐                │
│  │    Consensus & Verification               │                │
│  └──────────────┬───────────────────────────┘                │
│                 │                                             │
│                 ▼                                             │
│  ┌──────────────────────────────────────────┐                │
│  │             Response                      │                │
│  └──────────────────────────────────────────┘                │
└─────────────────────────────────────────────────────────────┘
```

### Core Principles

| Principle | Description |
|-----------|-------------|
| **Mesh, not monolithic** | Multiple expert nodes (each a BF16 transformer block) coordinated by a shared backbone |
| **Local learning** | Each node trains independently with its own AdamW optimizer — no global backprop |
| **Self-organization** | Nodes are born (mitosis/neurogenesis), grow (training), merge (fusion), and die (pruning) |
| **Shared latent space** | All experts communicate through a 256-dim universal projection — representations stay compatible |
| **Global cognitive layer** | A transformer between router and experts that orchestrates memory, tools, and consensus |
| **Diffusion generation** | Output is produced via iterative denoising on a token canvas (discrete diffusion) |
| **Speculative decoding** | MTP (multi-token prediction) heads accelerate generation by predicting 3-5 tokens per step |
| **Thinker-weighted curriculum** | Core phases (linguistics, reasoning, math) get 3-4x more training data |

### The Three Brains

| Layer | Role | Parameters |
|-------|------|------------|
| **Backbone** (CanvasTransformer) | Language understanding, token diffusion, response generation | 500M |
| **Expert Nodes** | Domain-specific knowledge (math, code, physics, etc.) | 1-8M each |
| **Global Cognitive Layer** | Coordination, planning, memory, consensus, tool use | 50-100M |
| **DSpark Speculator** | Multi-token prediction heads for fast inference | 10-50M |

### File Map

```
E:\my apps\NN
├── DOCUMENTATION.md                ← You are here
├── AGENTS.md                       ← Project brief for AI coding agents
├── AF.md                           ← Architectural critique (the "why")
├── MESH_REFACTOR_PLAN.md           ← Implementation plan (the "how")
├── MESH_REFACTOR_CHECKLIST.md      ← Step-by-step task tracker
├── dataset-structure.md            ← Curriculum design specification
├── generate_curriculum.bat         ← One-click dataset generator
├── rules.md                        ← Source code contracts
├── NoProp/
│   ├── src/
│   │   ├── model_sizes.py          ← Model size presets (250M/500M/1B/2B)
│   │   ├── diffusion_canvas.py     ← Backbone transformer + diffusion process
│   │   ├── mesh_router.py          ← Expert router, latent space, executor
│   │   ├── global_cognitive_layer.py ← GCL: memory, tools, consensus
│   │   ├── memory_manager.py       ← 5-tier memory (working→archived)
│   │   ├── lifecycle_manager.py    ← Expert lifecycle (create→delete)
│   │   ├── noprop_block.py         ← BF16 single-block + local AdamW
│   │   ├── turboquant_attention.py ← PolarQuant + QJL compression
│   │   ├── dspark_speculator.py    ← MTP heads + confidence verifier
│   │   ├── train_mesh.py           ← Main training orchestrator
│   │   ├── export_utils.py         ← safetensors/ONNX/GGUF export
│   │   └── tests/                  ← Pytest test suite
│   ├── scripts/
│   │   ├── curriculum_generator.py ← JSONL dataset generator (11 phases)
│   │   ├── train_on_text.py        ← Turbo training script
│   │   └── mesh_dashboard.py       ← Web dashboard for monitoring
│   └── curriculum_data/            ← Generated datasets
├── DeepSpec/
│   ├── train.py                    ← DSpark training (works on Windows)
│   ├── eval.py                     ← DSpark evaluation
│   ├── deepspec/modeling/dspark/   ← Markov heads, confidence, MTP
│   ├── config/dspark/              ← Config files (Qwen3, Gemma4)
│   └── scripts/train/train.bat     ← Windows launcher
└── .venv/                          ← Shared Python venv
```

---

## 2. Quick Start

### Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| Windows 11 | — | Tested on RTX 5060 8GB |
| uv | 0.6+ | Astral's Python package manager |
| llama-server | Latest | From `E:\my apps\LLAMA\` — Gemma 4 E2B Q4_K_S |
| CUDA | 12.8 | Included with PyTorch via uv |

### Setup

```powershell
# 1. Create venv (one-time)
uv sync

# 2. Verify bitsandbytes
python -c "import bitsandbytes; print('OK')"

# 3. Start llama-server for dataset generation
E:\my apps\LLAMA\1.bat          # 52K ctx TurboQuant mode
# OR for faster generation:
generate_curriculum.bat          # auto-starts speed-optimized server

# 4. Generate training data
generate_curriculum.bat          # 11 phases, thinker-weighted, 50 base
generate_curriculum.bat 3,5 2000 # reasoning + math only, 2000 base

# 5. Train the mesh
.venv\Scripts\python .\NoProp\src\train_mesh.py --embed-dim 512 --num-epochs 10
```

---

## 3. Dataset Generation

### Curriculum Generator

The `generate_curriculum.bat` launcher (at repo root) starts llama-server + runs the generator:

```bat
generate_curriculum.bat                    :: all 11 phases, 50 base
generate_curriculum.bat 3,4 500            :: reasoning + programming only
generate_curriculum.bat 0 1000 --no-resume  :: foundation only, 1000 fresh
```

### Phase Weights (Thinker Mode)

| Tier | Weight | Phases | Description |
|------|--------|--------|-------------|
| ★ CORE | 3-4x | Foundation(3), Mathematics(3), Reasoning(4) | Language, formal reasoning, precise thinking |
| ◆ HIGH | 2x | Knowledge, Relationships, Programming, Self_Improvement | Facts, connections, code, meta-learning |
| Support | 1x | Tool_Use, Long_Context, Memory, Multi_Agent | Auxiliary skills |

### Output Format (JSONL)

Each phase directory contains `samples.jsonl` — one JSON object per line:

```json
{
  "id": "Foundation_000042_a1b2c3d4",
  "domain": "language",
  "difficulty": 3,
  "concepts": ["Parts of Speech", "verbs — transitive, intransitive"],
  "dependencies": [],
  "requires_memory": false,
  "requires_tools": false,
  "reasoning_type": "recognition",
  "input": "What is the difference between a transitive and intransitive verb?",
  "analysis": "Transitive verbs require a direct object...",
  "verification": "Check that the answer correctly identifies whether ...",
  "final_answer": "A transitive verb requires a direct object...",
  "quality": 0.95,
  "teacher": "Gemma4"
}
```

### Direct Python Invocation

```powershell
uv run --no-sync --package noprop-mesh python NoProp/scripts/curriculum_generator.py --phases 3,5 --samples 2000
```

---

## 4. Training

### Mesh Training (main pipeline)

```powershell
.venv\Scripts\python NoProp\src\train_mesh.py \
    --embed-dim 768 \
    --num-heads 8 \
    --top-k 3 \
    --lr 1e-3 \
    --num-epochs 10 \
    --batch-size 8 \
    --checkpoint-dir checkpoints/mesh \
    --mitosis-threshold 0.5
```

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-size` | `small` | Backbone size: `tiny`, `small`, `standard`, `large` |
| `--embed-dim` | 768 | Override backbone embedding dim |
| `--top-k` | 3 | Number of experts to route each query to |
| `--train-layers` | None | Limit active experts per step (round-robin) |
| `--canvas-len` | 512 | Diffusion canvas token length |
| `--canvas-steps` | 50 | Number of diffusion denoising steps |
| `--num-draft-tokens` | 3 | MTP prediction heads |

### Training on Text (turbo training)

```powershell
.venv\Scripts\python NoProp\scripts\train_on_text.py \
    --dataset curriculum_data/phase03_Reasoning/samples.jsonl \
    --embed-dim 512 \
    --epochs 20 \
    --batch-size 4
```

### Checkpointing

Checkpoints are saved atomically to `checkpoints/mesh/step_*`:

```
checkpoints/mesh/
├── step_100/
├── step_200/
└── step_final/
```

Each contains:
- `mesh.pt` — all expert weights + router state + optimizer states
- `meta.json` — step, loss, timestamp

Resume training:
```powershell
.venv\Scripts\python NoProp\src\train_mesh.py --resume
```

### Model Export

```powershell
# Export to safetensors (default)
python -c "
from train_mesh import MeshTrainer
trainer = MeshTrainer()
trainer._load_checkpoint()
trainer.export_model('output/mesh-model.safetensors', fmt='safetensors')
"
```

Formats: `safetensors` (default), `onnx`, `gguf`, `pt`

---

## 5. Evaluation & Inference

### Generate text

```python
from train_mesh import MeshTrainer

trainer = MeshTrainer(use_diffusion_canvas=True)
trainer._load_checkpoint("checkpoints/mesh/step_final")
tokens = trainer.generate_text(batch_size=1, max_blocks=3)
print(trainer.tokenizer.decode(tokens[0]))
```

### Chat (conditional generation)

```python
prompt_ids = tokenizer.encode("Explain quantum entanglement", return_tensors="pt")
output = trainer.chat(prompt_ids, max_new_tokens=256)
print(tokenizer.decode(output[0]))
```

### Inference with routing

```python
import torch
x = torch.randn(1, 768)
output, info = trainer.infer(x)
print(f"Draft tokens: {info['draft_tokens'].shape}")
print(f"Confidence: {info['confidence']}")
print(f"Active experts: {info['active_nodes']}")
```

---

## 6. DeepSpec / DSpark (Speculative Decoding)

DSpark is a speculative decoding architecture that trains **draft model heads** on top of a frozen target model. The mesh uses this for fast inference.

### Windows Port Status

DSpark now runs on Windows. Key changes from upstream:
- **NCCL → Gloo** for distributed backend (`distributed.py`)
- **Triton made optional** (`requirements.txt` — conditional dependency)
- **`torch_compile=False`** in configs (falls back to eager mode)
- **Shell scripts → `.bat`** wrappers

### Train on Windows

```batch
scripts\train\train.bat --config config/dspark/dspark_qwen3_4b.py
```

Or directly:

```powershell
set CUDA_VISIBLE_DEVICES=0
set MASTER_ADDR=127.0.0.1
set MASTER_PORT=29500
set RANK=0
set WORLD_SIZE=1

.venv\Scripts\python DeepSpec\train.py ^
    --config DeepSpec\config\dspark\dspark_qwen3_4b.py ^
    --opts "data.target_cache_path=%USERPROFILE%\.cache\deepspec\qwen3_4b_target_cache"
```

### Evaluate on Windows

```batch
scripts\eval\eval.bat
```

Or directly:

```powershell
.venv\Scripts\python DeepSpec\eval.py ^
    --target_name_or_path Qwen/Qwen3-4B ^
    --draft_name_or_path checkpoints/draft/step_latest
```

Evaluates on all 9 benchmarks:
`gsm8k(500)`, `math500(500)`, `aime25(30)`, `humaneval(164)`, `mbpp(256)`, `livecodebench(500)`, `mt-bench(80)`, `alpaca(500)`, `arena-hard-v2(500)`

### Data Preparation

```powershell
# Stage 1: Download & split
.venv\Scripts\python depspec/scripts/data/download_and_split.py

# Stage 2: Regenerate answers (requires SGLang — install separately)
pip install "sglang[all]"

# Stage 3: Build target cache (~38 TB for Qwen3-4B!)
.venv\Scripts\python depspec/scripts/data/prepare_target_cache.py
```

### Integration with Mesh

The DSpark MTP heads are wired into `train_mesh.py` via `DSparkSpeculator`:
- During training: MTP heads predict the next 3-5 tokens, loss is backpropagated into expert nodes
- During inference: heads propose draft tokens, confidence verifier accepts/rejects them
- Curriculum dataset provides the training text

---

## 7. Module Reference

### `NoProp/src/model_sizes.py`

```python
SizePreset(
    name="small",
    d_model=1024,       # Embedding dimension
    n_layers=12,        # Transformer layers
    n_heads=16,         # Attention heads
    d_ff=4096,          # FFN hidden dimension
    max_seq_len=8192,   # Maximum sequence length
    num_experts=16,     # Maximum active expert nodes
)
```

### `NoProp/src/mesh_router.py`

| Class | Responsibility |
|-------|---------------|
| `MeshNode` | Per-expert: anchor embedding, rolling loss, lifecycle state, health metrics |
| `UniversalLatentSpace` | Projects `d_model → d_latent(256)` for shared representation |
| `ExpertAdapter` | Bidirectional `d_latent ↔ d_expert` projection per expert |
| `IntentDetector` | 2-layer MLP classifying domain + difficulty from backbone state |
| `DifficultyEstimator` | Regression head for difficulty 1-7 |
| `ExpertTreeNode` | Name, children, expert_id for hierarchical expert tree |
| `ExpertGraph` | Root nodes, path-based leaf lookup, routing traversal |
| `RouterPlanner` | Intent + difficulty → traverse ExpertGraph → execution plan |
| `ExecutionGraph` | Sequential/parallel step execution of planned expert calls |
| `MeshRouter` | Orchestrates all of the above |

### `NoProp/src/diffusion_canvas.py`

| Class | Responsibility |
|-------|---------------|
| `UniformStateDiffusion` | Cosine schedule diffusion, corrupt/denoise helpers |
| `CanvasBlock` | Single transformer block (token embed + time embed + MHA + FFN) |
| `CanvasTransformer` | Stacked CanvasBlocks (n_layers deep), main backbone |
| `DiffusionCanvas` | Orchestrates diffusion: init → denoise → generate |

### `NoProp/src/global_cognitive_layer.py`

| Class | Responsibility |
|-------|---------------|
| `GlobalCognitiveLayer` | Transformer operating on shared latent space |
| `ConsensusMechanism` | Confidence-weighted voting across expert outputs |
| `VerificationModule` | Output consistency checks, factual accuracy |
| `ToolManager` | Tool registry, dispatch, result interpretation |

### `NoProp/src/memory_manager.py`

| Class | Responsibility |
|-------|---------------|
| `WorkingMemory` | Last-N tokens/vectors (context window) |
| `SessionMemory` | Summary + KV store for current session |
| `EpisodicMemory` | Experiences with timestamps, recency retrieval |
| `SemanticMemory` | Knowledge graph with facts, confidence, sources |
| `ArchivedMemory` | Compressed long-term LRU storage |
| `MemoryManager` | Routes reads/writes to correct memory tier |

### `NoProp/src/lifecycle_manager.py`

| Function | Responsibility |
|----------|---------------|
| `evaluate_expert()` | Benchmark on validation set, update metadata |
| `merge_experts(ids)` | Knowledge distillation from multiple into one |
| `compress_expert(id)` | Prune, quantize, or distill an expert |
| `archive_expert(id)` | Freeze state, remove from routing |
| `delete_expert(id)` | Remove permanently, free parameters |

### `NoProp/src/tests/`

| Test | Verifies |
|------|----------|
| `test_zero_grad.py` | Frozen 4-bit params have `grad is None`; active nodes get gradients |
| `test_adversarial.py` | Variable shapes, extreme context, high-entropy noise, edge cases |
| `test_block.py` | NoPropBlock forward, local_step, time embed, atomic checkpoint |
| `test_router.py` | Registration, routing, mitosis trigger, metadata |
| `test_turboquant.py` | Orthogonal matrix, centroid fitting, QJL sign, compress |
| `test_speculator.py` | MTP head, multi-token draft, confidence, speculation loop |
| `test_gcl.py` | Global Cognitive Layer forward, consensus, verification |
| `test_memory.py` | All 5 memory tiers, orchestrator routing |
| `test_lifecycle.py` | State transitions, evaluate, merge, compress, archive, delete |
| `test_dataset.py` | Curriculum generator phases, JSONL output, resume |

---

## 8. Configuration

### Backbone Sizes (`model_sizes.py`)

| Preset | `d_model` | `n_layers` | `n_heads` | `d_ff` | Params | VRAM (BF16) |
|--------|-----------|------------|-----------|--------|--------|-------------|
| `tiny` | 768 | 8 | 12 | 3072 | 250M | ~2.5 GB |
| `small` | 1024 | 12 | 16 | 4096 | 500M | ~5.0 GB |
| `standard` | 1536 | 16 | 24 | 6144 | 1.0B | ~10 GB |
| `large` | 2048 | 24 | 32 | 8192 | 2.0B | ~20 GB |

### Curriculum Phases

| ID | Name | Weight | Domain | Difficulty Range |
|----|------|--------|--------|-----------------|
| 0 | Foundation | 3 | language | 1-2 |
| 1 | Knowledge | 2 | general_knowledge | 2-3 |
| 2 | Relationships | 2 | relational_knowledge | 3-4 |
| 3 | Reasoning | 4 | logical_reasoning | 4-6 |
| 4 | Programming | 2 | coding | 5-6 |
| 5 | Mathematics | 3 | mathematics | 5-7 |
| 6 | Tool_Use | 1 | tool_use | 4-5 |
| 7 | Long_Context | 1 | long_context | 6-7 |
| 8 | Memory | 1 | memory | 5-6 |
| 9 | Multi_Agent | 1 | multi_agent | 7 |
| 10 | Self_Improvement | 2 | meta_learning | 7 |

### Server Configs (llama-server)

| Mode | Ctx Size | Batch | Cache KV | Flash Attn | Use Case |
|------|----------|-------|----------|------------|----------|
| **Speed** | 8,192 | 2,048 | q8_0/q4_0 | on | Dataset generation (fast) |
| **TurboQuant** | 53,248 | 1,024 | q8_0/q4_0 | on | Full inference (52K ctx) |
| **Fallback** | 8,192 | 512 | f16/f16 | off | If flash-attn crashes |

---

## 9. Windows-Specific Notes

### Known Issues

| Issue | Cause | Workaround |
|-------|-------|------------|
| `torch.compile` fails | Triton is Linux-only | `torch.compile` disabled in configs; TF32 + autocast used instead |
| NCCL not available | Windows doesn't support NCCL | All distributed calls use Gloo backend |
| `bitsandbytes` 4-bit | Windows CUDA 12.8 support | Install via `uv sync` (pinned in pyproject.toml) |
| Long paths (>260 chars) | Windows MAX_PATH limit | Use `\\?\` prefix or enable long paths in Group Policy |

### Path Conventions

- All paths use **forward slashes** in Python (works everywhere)
- Batch files use **backslashes** with quotes
- Dataset generator outputs to `NoProp/curriculum_data/`

### Performance Tuning

```powershell
# Force TF32 matmul (default in train_mesh.py)
$env:TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

# cuDNN benchmark mode
$env:TORCH_CUDNN_BENCHMARK=1

# Disable deterministic mode (faster but non-deterministic)
# Already set in train_mesh.py
```

---

## 10. Troubleshooting

### "CUDA out of memory"

| Fix | Details |
|-----|---------|
| Reduce `--embed-dim` | 768 → 512 → 384 |
| Reduce `--canvas-len` | 512 → 256 → 128 |
| Reduce `--batch-size` | 8 → 4 → 2 → 1 |
| Use `--train-layers` | Limit active experts (1-2) |
| Enable gradient checkpointing | Add `--gradient-checkpoint` |

### "Server not responding on port 8080"

1. Check `E:\my apps\LLAMA\` window for errors
2. Kill old processes: `taskkill /f /im llama-server.exe`
3. Re-run `generate_curriculum.bat` (starts fresh server)

### "JSON parse error" in dataset generation

- Model occasionally outputs markdown fences around JSON — the parser handles ` ```json `
- If too many failures, reduce temperature: edit `temp = 0.7 + (written % 5) * 0.05` in generator

### "Training loss not decreasing"

1. Check that expert nodes are being created (`_load_seed_nodes()`)
2. Reduce mitosis threshold (0.5 → 0.3)
3. Increase `--lr` (1e-3 → 3e-3)
4. Ensure at least 100 samples per phase

### "DSpark train.py crashes on Windows"

1. Ensure NCCL → Gloo patch applied (see `deepspec/utils/distributed.py`)
2. Set `torch_compile=False` in config
3. Run with `WORLD_SIZE=1` (single-node only)
