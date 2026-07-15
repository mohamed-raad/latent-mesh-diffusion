# Training Configuration

## Dataset Mix (all phases)
| Source | Code | Weight | Status |
|--------|------|--------|--------|
| FineWeb-Edu | `HuggingFaceFW/fineweb-edu` | 55-70% | ✅ Streaming |
| FineWeb-2 Arabic (MSA) | `HuggingFaceFW/fineweb-2` / `arb_Arab` | 10-15% | ✅ Streaming |
| FineWeb-2 Arabic (Iraqi) | `HuggingFaceFW/fineweb-2` / `acm_Arab` | 5-6% | ✅ Streaming |
| FineWeb-2 Arabic (Levantine) | `HuggingFaceFW/fineweb-2` / `apc_Arab` | 5-8% | ✅ Streaming |
| FineWeb-2 Arabic (Egyptian) | `HuggingFaceFW/fineweb-2` / `arz_Arab` | 5-8% | ✅ Streaming |
| The Stack v2 (code) | `bigcode/the-stack-v2` | 10% | ❌ Gated (pending access) |

---

## Phase 1 — Core 250M
- **Preset:** tiny  
- **d_model:** 768 | **d_ff:** 3072 | **Layers:** 8 | **Heads:** 12  
- **Canvas:** 512 × 20 steps  
- **Latent:** 64 nodes @ 192 dim | **Heads:** 6 | **MoT iters:** 3  
- **Steps:** 50,000 | **Batch:** 8 | **LR:** 5e-4 | **Experts:** 8  
- **Parallel canvases:** 2  
- **Dataset mix:** 70% EN / 20% AR / 10% code (absorbed)

### Phase 1 Config (Python dict)
```python
PhaseConfig(
    name="Phase 1 — Core 250M",
    preset="tiny",
    d_model=768, n_layers=8, n_heads=12, d_ff=3072,
    max_seq_len=4096, num_experts=8,
    steps=50000, batch_size=8, lr=5e-4,
    canvas_len=512, canvas_steps=20,
    domains=["language", "reasoning", "interactions"],
    datasets=[
        {"hf_path": "HuggingFaceFW/fineweb-edu", "weight": 0.70},
        {"hf_path": "HuggingFaceFW/fineweb-2", "hf_config": "arb_Arab", "weight": 0.10},
        {"hf_path": "HuggingFaceFW/fineweb-2", "hf_config": "acm_Arab", "weight": 0.05},
        {"hf_path": "HuggingFaceFW/fineweb-2", "hf_config": "apc_Arab", "weight": 0.05},
    ],
    latent_nodes=64, d_latent=192, latent_heads=6, mot_iterations=3,
    parallel_canvases=2,
)
```

---

## Phase 2 — Core 500M
- **Preset:** small  
- **d_model:** 1024 | **d_ff:** 4096 | **Layers:** 12 | **Heads:** 16  
- **Canvas:** 768 × 24 steps  
- **Latent:** 96 nodes @ 256 dim | **Heads:** 8 | **MoT iters:** 5  
- **Steps:** 75,000 | **Batch:** 4 | **LR:** 3e-4 | **Experts:** 16  
- **Parallel canvases:** 3  
- **Dataset mix:** 65% EN / 23% AR / 12% code

---

## Phase 3 — Core 750M
- **Preset:** standard (custom: d=1344)  
- **d_model:** 1344 | **d_ff:** 5376 | **Layers:** 14 | **Heads:** 20  
- **Canvas:** 1024 × 28 steps  
- **Latent:** 128 nodes @ 320 dim | **Heads:** 10 | **MoT iters:** 5  
- **Steps:** 100,000 | **Batch:** 4 | **LR:** 2e-4 | **Experts:** 24  
- **Parallel canvases:** 3 | **Consensus:** 0.80  
- **Dataset mix:** 60% EN / 28% AR / 12% code

---

## Phase 4 — Core 1B
- **Preset:** standard  
- **d_model:** 1536 | **d_ff:** 6144 | **Layers:** 16 | **Heads:** 24  
- **Canvas:** 1024 × 32 steps (curriculum: →1536 →2048 optional)  
- **Latent:** 160 nodes @ 384 dim | **Heads:** 12 | **MoT iters:** 5  
- **Steps:** 150,000 | **Batch:** 2 | **LR:** 1e-4 | **Experts:** 32  
- **Parallel canvases:** 4 | **Consensus:** 0.85  
- **Dataset mix:** 55% EN / 28% AR / 17% code

---

## Expert Architecture

Hierarchical: Domain → Language → Skill → Micro-expert

### Coding (600M total, 18 micro-experts)
| Language | Skill | Micro-expert | Params | d_model | Layers |
|----------|-------|-------------|--------|---------|--------|
| python | generation | gen_basic | 32M | 512 | 4 |
| python | generation | gen_advanced | 64M | 768 | 6 |
| python | debugging | debug_basic | 24M | 384 | 3 |
| python | debugging | debug_advanced | 48M | 640 | 5 |
| python | optimization | opt_profiling | 32M | 512 | 4 |
| python | optimization | opt_memory | 48M | 640 | 5 |
| python | async | async_basic | 16M | 320 | 2 |
| python | numpy | numpy_ops | 24M | 384 | 3 |
| python | pytorch | torch_nn | 48M | 640 | 5 |
| python | pytorch | torch_train | 64M | 768 | 6 |
| nodejs | generation | node_gen | 32M | 512 | 4 |
| nodejs | async | node_async | 24M | 384 | 3 |
| cpp | generation | cpp_gen | 48M | 640 | 5 |
| cpp | memory | cpp_mem | 32M | 512 | 4 |
| react | components | react_comp | 24M | 384 | 3 |
| react | state | react_state | 16M | 320 | 2 |
| database | sql | sql_queries | 16M | 320 | 2 |
| database | nosql | nosql_basic | 8M | 256 | 1 |

### Reasoning (256M total)
| Language | Skill | Params |
|----------|-------|--------|
| math | algebra | 48M |
| math | geometry | 48M |
| math | calculus | 64M |
| logic | deductive | 32M |
| planning | task_plan | 64M |

### Language (168M total)
| Language | Skill | Params |
|----------|-------|--------|
| english | grammar | 24M |
| english | writing | 48M |
| arabic | MSA | 32M |
| arabic | dialects | 16M |
| translation | en↔ar | 48M |
