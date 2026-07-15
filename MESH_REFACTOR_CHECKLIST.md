# Mesh Refactor — Step-by-Step Checklist

## Phase 0: DSpark Windows Port

- [ ] **0.1** Edit `deepspec/utils/distributed.py` — NCCL→Gloo backend
- [ ] **0.2** Edit `deepspec/trainer/base_trainer.py` — fallback to Gloo if NCCL fails
- [ ] **0.3** Edit `DeepSpec/requirements.txt` — conditional triton
- [ ] **0.4** Edit `config/dspark/dspark_*.py` — `torch_compile=False`
- [ ] **0.5** Create `scripts/train/train.ps1` — Windows wrapper
- [ ] **0.6** Create `scripts/eval/eval.ps1` — Windows wrapper
- [ ] **0.7** Edit `deepspec/modeling/dspark/qwen3/modeling.py` — conditional triton import
- [ ] *Verify:* `python DeepSpec\train.py --help` runs without import errors

---

## Phase 1: Backbone Scaling (500M)

- [ ] **1.1** Create `config/model_sizes.py` — SizePreset dataclass with 4 presets
- [ ] **1.2** Rewrite `diffusion_canvas.py` — `CanvasTransformer` with N stacked blocks
- [ ] **1.3** Parameterize `d_model`, `n_layers`, `n_heads`, `d_ff` from preset
- [ ] **1.4** Replace `nn.MultiheadAttention` with GQA (grouped-query attention)
- [ ] **1.5** Edit `train_mesh.py` — read `model_size` arg, pass to CanvasTransformer
- [ ] **1.6** Edit `train_mesh.py` — add `--model-size` CLI arg
- [ ] **1.7** Edit `mesh_router.py` — update anchor dim from model config
- [ ] *Test:* Run `train_mesh.py` with `--model-size tiny` — verify forward pass, VRAM < 4GB

---

## Phase 2: Universal Latent Space

- [ ] **2.1** Add `UniversalLatentSpace` class to `mesh_router.py`
- [ ] **2.2** Add `ExpertAdapter` (encode/decode) to `mesh_router.py`
- [ ] **2.3** Modify `route()` — project query through latent space first
- [ ] **2.4** Add `latent_consistency_loss()` to `mesh_router.py`
- [ ] **2.5** Wire consistency loss in `train_mesh.py`
- [ ] *Test:* Verify cosine similarity between same-batch projections across experts

---

## Phase 3: Hierarchical Router + Planner

- [ ] **3.1** Add `IntentDetector` classifier to `mesh_router.py`
- [ ] **3.2** Add `DifficultyEstimator` regression head
- [ ] **3.3** Define `ExpertTreeNode` in `mesh_router.py`
- [ ] **3.4** Define `ExpertGraph` with path-based lookup
- [ ] **3.5** Add `RouterPlanner` for execution graph construction
- [ ] **3.6** Add `ExecutionGraph` execution engine
- [ ] **3.7** Add intent/difficulty prediction loss in `train_mesh.py`
- [ ] **3.8** Support hierarchical expert registration in `train_mesh.py`
- [ ] *Test:* Run `test_router.py` — graph traversal, intent classification accuracy

---

## Phase 4: Global Cognitive Layer

- [ ] **4.1** Create `global_cognitive_layer.py` with `GlobalCognitiveLayer`
- [ ] **4.2** Add `ConsensusMechanism` (weighted voting by confidence)
- [ ] **4.3** Add `VerificationModule` (output consistency checks)
- [ ] **4.4** Add `ToolManager` (registry + dispatch + result parsing)
- [ ] **4.5** Integrate GCL into `diffusion_canvas.py` forward pass
- [ ] **4.6** Joint training of GCL + backbone + experts in `train_mesh.py`
- [ ] *Test:* Run full `train_mesh` — verify end-to-end loss convergence

---

## Phase 5: Expert Lifecycle

- [ ] **5.1** Add `ExpertLifecycle` enum to `mesh_router.py`
- [ ] **5.2** Add `ExpertMetadata` tracking to `mesh_router.py`
- [ ] **5.3** Create `lifecycle_manager.py` — `evaluate_expert()`
- [ ] **5.4** `merge_experts()` — knowledge distillation merge
- [ ] **5.5** `compress_expert()` — prune/quantize/distill
- [ ] **5.6** `archive_expert()` — freeze + remove from routing
- [ ] **5.7** `delete_expert()` — permanent removal
- [ ] *Test:* Test lifecycle state transitions

---

## Phase 6: Memory System

- [ ] **6.1** Create `memory_manager.py` — `WorkingMemory`
- [ ] **6.2** `SessionMemory` — summary + KV store
- [ ] **6.3** `EpisodicMemory` — experiences + recency retrieval
- [ ] **6.4** `SemanticMemory` — knowledge graph + confidence tracking
- [ ] **6.5** `ArchivedMemory` — compressed LRU storage
- [ ] **6.6** `MemoryManager` orchestrator — routes reads/writes
- [ ] **6.7** Integrate into `diffusion_canvas.py` forward pass
- [ ] **6.8** RouterPlanner queries MemoryManager
- [ ] *Test:* Run memory recall tests, verify cross-session retention

---

## Phase 7: Infrastructure

- [ ] **7.1** Confidence Engine — per-fact confidence + timestamps + contradictions
- [ ] **7.2** Expert Health Monitor — latency, accuracy, hallucination rate
- [ ] **7.3** Adaptive Compute — easy→core only, hard→full pipeline
- [ ] **7.4** Expert Marketplace — metadata schema (ID, version, dependencies, license)
- [ ] **7.5** Knowledge Versioning — sub-versions per expert
- [ ] **7.6** World Model — relationship graph between concepts
- [ ] **7.7** Learning Scheduler — novelty/usefulness/verified filter in `train_mesh.py`
- [ ] **7.8** Research features — Neurogenesis, Synaptic Pruning, Expert Fusion, Distillation, Evolution
- [ ] *Test:* Full integration test — generate dataset → train mesh → evaluate → next curriculum

---

## Phase 8: DSpark Integration

- [ ] **8.1** Wire curriculum JSONL dataset as training data loader in `train_mesh.py`
- [ ] **8.2** Add DSpark MTP head training mode
- [ ] **8.3** Use dspark confidence head in GCL ConsensusMechanism
- [ ] **8.4** Upgrade `dspark_speculator.py` to full Markov head pipeline
- [ ] *Test:* Train MTP heads on curriculum data, verify loss decreases

---

## Legend

- [ ] Not started
- [x] Completed
- [-] Skipped/blocked
