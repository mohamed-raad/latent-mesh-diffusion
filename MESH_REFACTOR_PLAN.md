# Mesh Refactor — Integrated Implementation Plan

Based on `AF.md` architectural critique + DSpark Windows port + curriculum training pipeline.

---

## Phase 0: DSpark Windows Port (parallel track)

DeepSpec code is pure PyTorch except 3 blockers. Port is 5-8 small file changes.

| Step | File | Change | Why |
|------|------|--------|-----|
| 0.1 | `deepspec/utils/distributed.py` | `backend="nccl"` → `backend="gloo"` | NCCL doesn't exist on Windows |
| 0.2 | `deepspec/trainer/base_trainer.py` | Fallback to Gloo if NCCL init fails | Graceful handling |
| 0.3 | `DeepSpec/requirements.txt` | `triton==3.5.1` → `triton; sys_platform != 'win32'` | Triton is Linux-only |
| 0.4 | `config/dspark/*.py` | `torch_compile=True` → `torch_compile=False` | No Triton → no compile |
| 0.5 | `scripts/train/train.ps1` (new) | Windows wrapper for train.py | Replaces train.sh |
| 0.6 | `scripts/eval/eval.ps1` (new) | Windows wrapper for eval.py | Replaces eval.sh |
| 0.7 | `deepspec/modeling/dspark/qwen3/modeling.py` | Move `import triton` inside conditional | Avoid import error on Windows |

---

## Phase 1: Backbone Scaling (AF.md #1)

**Current:** Single `CanvasBlock`, `embed_dim=768`, 89M total (77M vocab projections, 12M compute)  
**Target:** Configurable 500M backbone via `model_sizes.py`

| Step | File | Change |
|------|------|--------|
| 1.1 | `config/model_sizes.py` (new) | `SizePreset` dataclass: `tiny(250M)`, `small(500M)`, `standard(1B)`, `large(2B)` |
| 1.2 | `diffusion_canvas.py` | Replace single `CanvasBlock` with `CanvasTransformer` — N stacked blocks |
| 1.3 | `diffusion_canvas.py` | Parameterize `d_model`, `n_layers`, `n_heads`, `d_ff` from preset |
| 1.4 | `diffusion_canvas.py` | Replace `nn.MultiheadAttention` with GQA (grouped-query attention) for speed |
| 1.5 | `train_mesh.py` | Read `model_size` arg, pass to `CanvasTransformer` |
| 1.6 | `train_mesh.py` | Add `model_sizes` import + `--model-size` CLI arg |
| 1.7 | `mesh_router.py` | Update anchor embedding dim from model preset |

**VRAM budget (500M small):**
- Backbone: ~5 GB (BF16)
- 8 expert nodes: ~1 GB (BF16, 1-2M each)
- KV cache (8K ctx): ~0.5 GB
- Activations + buffers: ~1 GB
- **Total: ~7.5 GB — fits RTX 5060 8GB**

---

## Phase 2: Universal Latent Space (AF.md #2)

**Current:** No shared representation — each expert uses incompatible embeddings  
**Target:** `d_model → d_latent(256)` projection + bidirectional adapters per expert

| Step | File | Change |
|------|------|--------|
| 2.1 | `mesh_router.py` | Add `UniversalLatentSpace` — `nn.Linear(d_model, d_latent)` + LayerNorm |
| 2.2 | `mesh_router.py` | Add `ExpertAdapter` — per-expert `nn.Linear(d_latent, d_expert)` + reverse |
| 2.3 | `mesh_router.py` | Modify `MeshRouter.route()` — project query through latent space first |
| 2.4 | `mesh_router.py` | Add `latent_consistency_loss()` — cosine sim between same-batch projections |
| 2.5 | `train_mesh.py` | Add `latent_consistency_weight` to total loss |

---

## Phase 3: Hierarchical Router + Planner (AF.md #3 + #4)

**Current:** `query → cosine_sim → expert`  
**Target:** `query → IntentDetector → DifficultyEstimator → RouterPlanner → ExpertGraph → ExecutionGraph`

| Step | File | Change |
|------|------|--------|
| 3.1 | `mesh_router.py` | Add `IntentDetector` — 2-layer MLP, backbone hidden → domain logits |
| 3.2 | `mesh_router.py` | Add `DifficultyEstimator` — regression head → difficulty 1-7 |
| 3.3 | `mesh_router.py` | Define `ExpertTreeNode` — name, children list, expert_id (leaf only) |
| 3.4 | `mesh_router.py` | Define `ExpertGraph` — root nodes, path traversal, leaf lookup |
| 3.5 | `mesh_router.py` | Add `RouterPlanner` — intent + difficulty → traverse ExpertGraph → execution plan |
| 3.6 | `mesh_router.py` | Add `ExecutionGraph` — sequential/parallel step execution |
| 3.7 | `train_mesh.py` | Add intent + difficulty prediction loss from dataset metadata |
| 3.8 | `train_mesh.py` | Support hierarchical expert registration |

---

## Phase 4: Global Cognitive Layer (AF.md #7 — essential)

**Current:** `Router → Experts`  
**Target:** `Planner → GCL → {Memory, ExpertGraph, ToolManager} → ExecutionGraph → Consensus & Verification`

| Step | File | Change |
|------|------|--------|
| 4.1 | `global_cognitive_layer.py` (new) | `GlobalCognitiveLayer` — transformer on shared latent space |
| 4.2 | `global_cognitive_layer.py` | `ConsensusMechanism` — confidence-weighted voting |
| 4.3 | `global_cognitive_layer.py` | `VerificationModule` — output consistency + factual checks |
| 4.4 | `global_cognitive_layer.py` | `ToolManager` — tool registry + dispatch + result parsing |
| 4.5 | `diffusion_canvas.py` | Insert GCL between router output and response generation |
| 4.6 | `train_mesh.py` | Joint training of GCL + backbone + experts |

---

## Phase 5: Expert Lifecycle (AF.md #5)

**Current:** `Create → Forever`  
**Target:** `Create → Evaluate → Improve → Merge → Compress → Archive → Delete`

| Step | File | Change |
|------|------|--------|
| 5.1 | `mesh_router.py` | Add `ExpertLifecycle` enum (CREATED, EVALUATING, ACTIVE, MERGING, COMPRESSING, ARCHIVED, DELETED) |
| 5.2 | `mesh_router.py` | Add `ExpertMetadata` — accuracy, latency, usage, last_active, version, status |
| 5.3 | `lifecycle_manager.py` (new) | `evaluate_expert()` — benchmark, update metadata |
| 5.4 | `lifecycle_manager.py` | `merge_experts()` — knowledge distillation from multiple → one |
| 5.5 | `lifecycle_manager.py` | `compress_expert()` — prune + quantize |
| 5.6 | `lifecycle_manager.py` | `archive_expert()` — freeze, remove from routing |
| 5.7 | `lifecycle_manager.py` | `delete_expert()` — remove params, free memory |

---

## Phase 6: Memory System (AF.md #8 + New Features)

**Current:** 768d latent vector as only bridge  
**Target:** 5-tier: Working → Session → Episodic → Semantic → Archived

| Step | File | Change |
|------|------|--------|
| 6.1 | `memory_manager.py` (new) | `WorkingMemory` — last-N tokens/vectors |
| 6.2 | `memory_manager.py` | `SessionMemory` — summary + KV store |
| 6.3 | `memory_manager.py` | `EpisodicMemory` — experiences + recency retrieval |
| 6.4 | `memory_manager.py` | `SemanticMemory` — knowledge graph + confidence |
| 6.5 | `memory_manager.py` | `ArchivedMemory` — compressed LRU storage |
| 6.6 | `memory_manager.py` | `MemoryManager` — orchestrator, routes reads/writes |
| 6.7 | `diffusion_canvas.py` | Integrate MemoryManager into forward pass |
| 6.8 | `mesh_router.py` | RouterPlanner queries MemoryManager for context |

---

## Phase 7: Infrastructure (AF.md Remaining Features)

| Step | File | Change |
|------|------|--------|
| 7.1 | `mesh_router.py` | Confidence Engine — per-fact confidence + timestamps + contradictions |
| 7.2 | `mesh_router.py` | Expert Health Monitor — latency, accuracy, hallucination rate |
| 7.3 | `mesh_router.py` | Adaptive Compute — easy→core only, hard→full pipeline |
| 7.4 | `mesh_router.py` | Expert Marketplace — metadata schema |
| 7.5 | `mesh_router.py` | Knowledge Versioning — `Python → {v3.11, v3.12, v3.18}` |
| 7.6 | `mesh_router.py` | World Model — learns relationships between facts |
| 7.7 | `train_mesh.py` | Learning Scheduler — novelty/usefulness/verified filter |
| 7.8 | `mesh_router.py` | Research: Neurogenesis, Synaptic Pruning, Fusion, Distillation, Evolution |

---

## Phase 8: DSpark Integration + Curriculum Training

| Step | File | Change |
|------|------|--------|
| 8.1 | `train_mesh.py` | Wire curriculum JSONL dataset as training data loader |
| 8.2 | `train_mesh.py` | Add DSpark MTP head training as training mode |
| 8.3 | `global_cognitive_layer.py` | Use dspark confidence head for consensus weighting |
| 8.4 | `dspark_speculator.py` | Upgrade to full DSpark Markov head + draft ops pipeline |

---

## Implementation Order

```
Week 1:  Phase 0 (DSpark port) + Phase 1 (backbone)
Week 2:  Phase 2 (latent space) + Phase 3 (router)
Week 3:  Phase 4 (GCL) + Phase 5 (lifecycle)
Week 4:  Phase 6 (memory) + Phase 7 (infrastructure)
Week 5:  Phase 8 (DSpark integration) + testing
```
