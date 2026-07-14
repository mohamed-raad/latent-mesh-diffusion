"""
Training Pipeline Orchestrator — 4-phase core training + expert specialization.
Phase 1: 250M core (tiny preset) on general domains
Phase 2: 500M core (small preset) on general domains
Phase 3: 750M core (custom preset) on general domains + reasoning
Phase 4: 1B core (standard preset) on all domains
Phase 5+: Expert training per domain (coding, math, reasoning, etc.)
         → Subdomain training (python, nodejs, C++, react, database)
         → Children training (API, ML, functions, variables at 100M each)
"""
import os
import sys
import json
import time
import argparse
from dataclasses import dataclass, field, asdict
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader, IterableDataset

from model_sizes import get_preset, SizePreset
from mesh_tokenizer import VOCAB_SIZE, load_tokenizer
from training_dashboard import TrainingDashboard
from dynamic_budget import DynamicExpertBudget
from cross_layer_cache import CrossLayerRoutingCache
from mesh_router import LatentMeshConfig

# Dataset imports
from streaming import create_mixed_dataset
from bin_converter import convert_hf_to_bin


# ──────────────────────────────────────────────
# Training phases
# ──────────────────────────────────────────────

@dataclass
class PhaseConfig:
    name: str
    preset: str
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    max_seq_len: int
    num_experts: int
    steps: int
    batch_size: int
    lr: float
    canvas_len: int
    canvas_steps: int
    domains: list[str]
    datasets: list[dict]
    latent_nodes: int = 96
    d_latent: int = 256
    latent_heads: int = 8
    mot_iterations: int = 5
    consensus_threshold: float = 0.85
    use_vae: bool = False
    parallel_canvases: int = 3
    max_gpu_experts: int = 8
    max_ram_experts: int = 32


PHASE_CONFIGS = {
    "phase1_250m": PhaseConfig(
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
    ),
    "phase2_500m": PhaseConfig(
        name="Phase 2 — Core 500M",
        preset="small",
        d_model=1024, n_layers=12, n_heads=16, d_ff=4096,
        max_seq_len=8192, num_experts=16,
        steps=75000, batch_size=4, lr=3e-4,
        canvas_len=768, canvas_steps=24,
        domains=["language", "reasoning", "interactions", "learning"],
        datasets=[
            {"hf_path": "HuggingFaceFW/fineweb-edu", "weight": 0.65},
            {"hf_path": "HuggingFaceFW/fineweb-2", "hf_config": "arb_Arab", "weight": 0.12},
            {"hf_path": "HuggingFaceFW/fineweb-2", "hf_config": "acm_Arab", "weight": 0.06},
            {"hf_path": "HuggingFaceFW/fineweb-2", "hf_config": "arz_Arab", "weight": 0.05},
        ],
        latent_nodes=96, d_latent=256, latent_heads=8, mot_iterations=5,
        parallel_canvases=3,
    ),
    "phase3_750m": PhaseConfig(
        name="Phase 3 — Core 750M",
        preset="standard",
        d_model=1344, n_layers=14, n_heads=20, d_ff=5376,
        max_seq_len=8192, num_experts=24,
        steps=100000, batch_size=4, lr=2e-4,
        canvas_len=1024, canvas_steps=28,
        domains=["language", "reasoning", "interactions", "learning", "planning"],
        datasets=[
            {"hf_path": "HuggingFaceFW/fineweb-edu", "weight": 0.60},
            {"hf_path": "HuggingFaceFW/fineweb-2", "hf_config": "arb_Arab", "weight": 0.15},
            {"hf_path": "HuggingFaceFW/fineweb-2", "hf_config": "apc_Arab", "weight": 0.08},
            {"hf_path": "HuggingFaceFW/fineweb-2", "hf_config": "acm_Arab", "weight": 0.05},
        ],
        latent_nodes=128, d_latent=320, latent_heads=10, mot_iterations=5,
        parallel_canvases=3, consensus_threshold=0.80,
    ),
    "phase4_1b": PhaseConfig(
        name="Phase 4 — Core 1B",
        preset="standard",
        d_model=1536, n_layers=16, n_heads=24, d_ff=6144,
        max_seq_len=16384, num_experts=32,
        steps=150000, batch_size=2, lr=1e-4,
        canvas_len=1024, canvas_steps=32,
        domains=["language", "reasoning", "interactions", "learning", "planning", "coding"],
        datasets=[
            {"hf_path": "HuggingFaceFW/fineweb-edu", "weight": 0.55},
            {"hf_path": "HuggingFaceFW/fineweb-2", "hf_config": "arb_Arab", "weight": 0.15},
            {"hf_path": "HuggingFaceFW/fineweb-2", "hf_config": "arz_Arab", "weight": 0.08},
            {"hf_path": "HuggingFaceFW/fineweb-2", "hf_config": "apc_Arab", "weight": 0.05},
        ],
        latent_nodes=160, d_latent=384, latent_heads=12, mot_iterations=5,
        parallel_canvases=4, consensus_threshold=0.85,
    ),
}


# ──────────────────────────────────────────────
# Expert training configs
# ──────────────────────────────────────────────

@dataclass
class MicroExpert:
    name: str
    params_m: int
    d_model: int
    n_layers: int
    n_heads: int
    steps: int
    weight: float
    datasets: list[str]


@dataclass
class SkillExpert:
    name: str
    micro_experts: list[MicroExpert]


@dataclass
class SubDomainExpert:
    name: str
    skills: list[SkillExpert]


@dataclass
class ExpertDomainConfig:
    domain: str
    subdomains: list[SubDomainExpert]


EXPERT_CONFIGS = {
    "coding": ExpertDomainConfig(
        domain="coding",
        subdomains=[
            SubDomainExpert(name="python", skills=[
                SkillExpert(name="generation", micro_experts=[
                    MicroExpert(name="gen_basic", params_m=32, d_model=512, n_layers=4, n_heads=8, steps=10000, weight=1.0, datasets=["bigcode/the-stack-v2/python"]),
                    MicroExpert(name="gen_advanced", params_m=64, d_model=768, n_layers=6, n_heads=12, steps=15000, weight=1.0, datasets=["bigcode/the-stack-v2/python"]),
                ]),
                SkillExpert(name="debugging", micro_experts=[
                    MicroExpert(name="debug_basic", params_m=24, d_model=384, n_layers=3, n_heads=6, steps=8000, weight=1.0, datasets=["bigcode/the-stack-v2/python"]),
                    MicroExpert(name="debug_advanced", params_m=48, d_model=640, n_layers=5, n_heads=10, steps=12000, weight=1.0, datasets=["bigcode/the-stack-v2/python"]),
                ]),
                SkillExpert(name="optimization", micro_experts=[
                    MicroExpert(name="opt_profiling", params_m=32, d_model=512, n_layers=4, n_heads=8, steps=10000, weight=1.0, datasets=["bigcode/the-stack-v2/python"]),
                    MicroExpert(name="opt_memory", params_m=48, d_model=640, n_layers=5, n_heads=10, steps=12000, weight=1.0, datasets=["bigcode/the-stack-v2/python"]),
                ]),
                SkillExpert(name="async", micro_experts=[
                    MicroExpert(name="async_basic", params_m=16, d_model=320, n_layers=2, n_heads=4, steps=6000, weight=1.0, datasets=["bigcode/the-stack-v2/python"]),
                ]),
                SkillExpert(name="numpy", micro_experts=[
                    MicroExpert(name="numpy_ops", params_m=24, d_model=384, n_layers=3, n_heads=6, steps=8000, weight=1.0, datasets=["bigcode/the-stack-v2/python"]),
                ]),
                SkillExpert(name="pytorch", micro_experts=[
                    MicroExpert(name="torch_nn", params_m=48, d_model=640, n_layers=5, n_heads=10, steps=15000, weight=1.0, datasets=["bigcode/the-stack-v2/python"]),
                    MicroExpert(name="torch_train", params_m=64, d_model=768, n_layers=6, n_heads=12, steps=15000, weight=1.0, datasets=["bigcode/the-stack-v2/python"]),
                ]),
            ]),
            SubDomainExpert(name="nodejs", skills=[
                SkillExpert(name="generation", micro_experts=[
                    MicroExpert(name="node_gen", params_m=32, d_model=512, n_layers=4, n_heads=8, steps=10000, weight=1.0, datasets=["bigcode/the-stack-v2/javascript"]),
                ]),
                SkillExpert(name="async", micro_experts=[
                    MicroExpert(name="node_async", params_m=24, d_model=384, n_layers=3, n_heads=6, steps=8000, weight=1.0, datasets=["bigcode/the-stack-v2/javascript"]),
                ]),
            ]),
            SubDomainExpert(name="cpp", skills=[
                SkillExpert(name="generation", micro_experts=[
                    MicroExpert(name="cpp_gen", params_m=48, d_model=640, n_layers=5, n_heads=10, steps=12000, weight=1.0, datasets=["bigcode/the-stack-v2/cpp"]),
                ]),
                SkillExpert(name="memory", micro_experts=[
                    MicroExpert(name="cpp_mem", params_m=32, d_model=512, n_layers=4, n_heads=8, steps=10000, weight=1.0, datasets=["bigcode/the-stack-v2/cpp"]),
                ]),
            ]),
            SubDomainExpert(name="react", skills=[
                SkillExpert(name="components", micro_experts=[
                    MicroExpert(name="react_comp", params_m=24, d_model=384, n_layers=3, n_heads=6, steps=8000, weight=1.0, datasets=["bigcode/the-stack-v2/typescript"]),
                ]),
                SkillExpert(name="state", micro_experts=[
                    MicroExpert(name="react_state", params_m=16, d_model=320, n_layers=2, n_heads=4, steps=6000, weight=1.0, datasets=["bigcode/the-stack-v2/typescript"]),
                ]),
            ]),
            SubDomainExpert(name="database", skills=[
                SkillExpert(name="sql", micro_experts=[
                    MicroExpert(name="sql_queries", params_m=16, d_model=320, n_layers=2, n_heads=4, steps=6000, weight=1.0, datasets=["bigcode/the-stack-v2/sql"]),
                ]),
                SkillExpert(name="nosql", micro_experts=[
                    MicroExpert(name="nosql_basic", params_m=8, d_model=256, n_layers=1, n_heads=4, steps=4000, weight=1.0, datasets=["bigcode/the-stack-v2"]),
                ]),
            ]),
        ],
    ),
    "reasoning": ExpertDomainConfig(
        domain="reasoning",
        subdomains=[
            SubDomainExpert(name="math", skills=[
                SkillExpert(name="algebra", micro_experts=[
                    MicroExpert(name="alg_basic", params_m=48, d_model=640, n_layers=5, n_heads=10, steps=12000, weight=1.0, datasets=["HuggingFaceH4/MATH-500"]),
                ]),
                SkillExpert(name="geometry", micro_experts=[
                    MicroExpert(name="geo_basic", params_m=48, d_model=640, n_layers=5, n_heads=10, steps=12000, weight=1.0, datasets=["HuggingFaceH4/MATH-500"]),
                ]),
                SkillExpert(name="calculus", micro_experts=[
                    MicroExpert(name="calc_basic", params_m=64, d_model=768, n_layers=6, n_heads=12, steps=15000, weight=1.0, datasets=["HuggingFaceH4/MATH-500"]),
                ]),
            ]),
            SubDomainExpert(name="logic", skills=[
                SkillExpert(name="deductive", micro_experts=[
                    MicroExpert(name="deduct_basic", params_m=32, d_model=512, n_layers=4, n_heads=8, steps=10000, weight=1.0, datasets=["HuggingFaceFW/fineweb-2"]),
                ]),
            ]),
            SubDomainExpert(name="planning", skills=[
                SkillExpert(name="task_plan", micro_experts=[
                    MicroExpert(name="plan_basic", params_m=64, d_model=768, n_layers=6, n_heads=12, steps=15000, weight=1.0, datasets=["HuggingFaceFW/fineweb-2"]),
                ]),
            ]),
        ],
    ),
    "language": ExpertDomainConfig(
        domain="language",
        subdomains=[
            SubDomainExpert(name="english", skills=[
                SkillExpert(name="grammar", micro_experts=[
                    MicroExpert(name="eng_grammar", params_m=24, d_model=384, n_layers=3, n_heads=6, steps=8000, weight=1.0, datasets=["HuggingFaceFW/fineweb-2/eng_Latn"]),
                ]),
                SkillExpert(name="writing", micro_experts=[
                    MicroExpert(name="eng_write", params_m=48, d_model=640, n_layers=5, n_heads=10, steps=12000, weight=1.0, datasets=["HuggingFaceFW/fineweb-2/eng_Latn"]),
                ]),
            ]),
            SubDomainExpert(name="arabic", skills=[
                SkillExpert(name="msa", micro_experts=[
                    MicroExpert(name="ara_msa", params_m=32, d_model=512, n_layers=4, n_heads=8, steps=10000, weight=1.0, datasets=["HuggingFaceFW/fineweb-2/ara_Arab"]),
                ]),
                SkillExpert(name="dialects", micro_experts=[
                    MicroExpert(name="ara_dialect", params_m=16, d_model=320, n_layers=2, n_heads=4, steps=6000, weight=1.0, datasets=["HuggingFaceFW/fineweb-2/ara_Arab"]),
                ]),
            ]),
            SubDomainExpert(name="translation", skills=[
                SkillExpert(name="en_ar", micro_experts=[
                    MicroExpert(name="trans_en_ar", params_m=48, d_model=640, n_layers=5, n_heads=10, steps=15000, weight=1.0, datasets=["HuggingFaceFW/fineweb-2"]),
                ]),
            ]),
        ],
    ),
}


# ──────────────────────────────────────────────
# Training orchestrator
# ──────────────────────────────────────────────

class TrainingOrchestrator:
    """Orchestrates multi-phase training with dashboard, streaming, and async RL."""

    def __init__(self, log_dir: str = "./training_logs", checkpoint_base: str = "./checkpoints",
                 use_packing: bool = True):
        self.log_dir = log_dir
        self.checkpoint_base = checkpoint_base
        self._use_packing = use_packing
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_base, exist_ok=True)

        self.dashboard = TrainingDashboard(log_dir=os.path.join(log_dir, "tensorboard"))
        self.dynamic_budget = DynamicExpertBudget(
            min_experts=2, max_experts=64, easy_experts=4, medium_experts=16, hard_experts=64,
        )
        self.routing_cache = CrossLayerRoutingCache(
            max_entries=2048, ttl_steps=10, min_similarity=0.85,
        )
        self.current_phase: str = ""
        self.current_step = 0

    def get_dataset(self, config: PhaseConfig, convert_first: bool = True) -> IterableDataset:
        """Create dataset, auto-converting HF sources to binary if needed."""
        datasets = []
        for ds_spec in config.datasets:
            hf_path = ds_spec["hf_path"]
            hf_config = ds_spec.get("hf_config")
            tag = hf_path.replace("/", "_") + (f"_{hf_config}" if hf_config else "")
            bin_path = os.path.join("./bin_data", tag)

            bin_ok = False
            index_path = os.path.join(bin_path, "index.json")
            if os.path.isfile(index_path):
                try:
                    ver = json.load(open(index_path)).get("version", 0)
                    if ver >= 2:
                        bin_ok = True
                    else:
                        print(f"  Stale binary format (v{ver}), re-converting {tag}...")
                        import shutil; shutil.rmtree(bin_path, ignore_errors=True)
                except Exception:
                    pass

            if convert_first and not bin_ok:
                print(f"  Converting {tag} to binary...")
                convert_hf_to_bin(hf_path, bin_path, hf_config=hf_config,
                                  max_docs=200000, shard_size=10000,
                                  max_seq_len=config.canvas_len)
                bin_ok = True

            if bin_ok:
                datasets.append({"local_dir": bin_path, "weight": ds_spec["weight"]})
            else:
                entry = {"hf_path": hf_path, "weight": ds_spec["weight"]}
                if hf_config:
                    entry["hf_config"] = hf_config
                datasets.append(entry)
        return create_mixed_dataset(sources=datasets, max_seq_len=config.canvas_len)

    def run_phase(self, config: PhaseConfig):
        """Run a single training phase."""
        print(f"\n{'='*60}")
        print(f"  {config.name}")
        print(f"  Preset: {config.preset} | Steps: {config.steps} | LR: {config.lr}")
        print(f"  Canvas: {config.canvas_len}x{config.canvas_steps}")
        print(f"  Latent: {config.latent_nodes} nodes @ {config.d_latent} dim")
        print(f"{'='*60}\n")

        from train_mesh import MeshTrainer
        import importlib; importlib.reload(sys.modules.get('train_mesh'))

        checkpoint_dir = os.path.join(self.checkpoint_base, f"phase_{config.name.lower().replace(' ', '_')}")
        log_dir = os.path.join(self.log_dir, f"phase_{config.name.lower().replace(' ', '_')}")

        trainer = MeshTrainer(
            model_size=config.preset,
            embed_dim=config.d_model,
            num_heads=config.n_heads,
            lr=config.lr,
            checkpoint_dir=checkpoint_dir,
            canvas_len=config.canvas_len,
            canvas_steps=config.canvas_steps,
            use_diffusion_canvas=True,
            max_experts=config.num_experts,
            log_dir=log_dir,
            max_gpu_experts=config.max_gpu_experts,
            max_ram_experts=config.max_ram_experts,
            parallel_canvases=config.parallel_canvases,
            core_only=False,
        )

        if trainer.router._latent_config is None:
            trainer.router._latent_config = LatentMeshConfig()
        trainer.router._latent_config.latent_nodes = config.latent_nodes
        trainer.router._latent_config.d_latent = config.d_latent
        trainer.router._latent_config.latent_heads = config.latent_heads
        trainer.router._latent_config.mot_max_iterations = config.mot_iterations
        trainer.router._latent_config.consensus_threshold = config.consensus_threshold
        trainer.router._latent_config.use_vae = config.use_vae

        dataset = self.get_dataset(config)
        print(f"  Dataset ready | {len(config.domains)} domains")
        print(f"  Starting training...\n")

        trainer.train(
            dataset=dataset,
            num_epochs=5,
            batch_size=config.batch_size,
            log_interval=10,
            ckpt_interval=1000,
            max_steps=config.steps,
            use_packing=self._use_packing,
            domain_ids=config.domains,
        )

        self.current_phase = config.name
        trainer.save_checkpoint()
        print(f"  ✓ Phase complete: {config.name}")

    def run_expert_training(self, expert_config: ExpertDomainConfig, phase_checkpoint: str):
        """Train experts for a specific domain."""
        print(f"\n{'='*60}")
        print(f"  Expert Training: {expert_config.domain}")
        print(f"  Subdomains: {[s['name'] for s in expert_config.subdomains]}")
        print(f"{'='*60}\n")

        from train_mesh import MeshTrainer
        from expert_tier_manager import ExpertTierManager

        checkpoint_dir = os.path.join(self.checkpoint_base, f"expert_{expert_config.domain}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        trainer = MeshTrainer(
            model_size="tiny",
            embed_dim=768,
            num_heads=12,
            lr=expert_config.children_lr,
            checkpoint_dir=checkpoint_dir,
            canvas_len=512,
            canvas_steps=20,
            use_diffusion_canvas=True,
            max_experts=64,
            log_dir=os.path.join(self.log_dir, f"expert_{expert_config.domain}"),
            max_gpu_experts=8,
            max_ram_experts=32,
            core_only=False,
            train_experts_only=True,
        )

        for subdomain in expert_config.subdomains:
            print(f"\n  ── Training subdomain: {subdomain['name']} ──")
            sub_checkpoint = os.path.join(checkpoint_dir, subdomain["name"])
            os.makedirs(sub_checkpoint, exist_ok=True)

            ds = MixedOnlineDataset(
                sources=subdomain.get("datasets", expert_config.datasets),
                max_seq_len=512,
            )

            trainer.nodes_dir = sub_checkpoint
            trainer.expert_nodes_dir = os.path.join(sub_checkpoint, "experts")
            os.makedirs(trainer.expert_nodes_dir, exist_ok=True)

            trainer.train(
                dataset=ds,
                num_epochs=3,
                batch_size=expert_config.children_batch_size,
                log_interval=10,
                save_interval=500,
                max_steps=expert_config.children_steps,
                use_packing=True,
                domain_ids=[expert_config.domain, subdomain["name"]],
            )

            expert_save_path = os.path.join(sub_checkpoint, f"{subdomain['name']}_expert.pt")
            trainer.save_checkpoint(expert_save_path)
            print(f"  ✓ {subdomain['name']} expert saved to {expert_save_path}")

    def run_all_phases(self, start_from: str = "phase1"):
        """Run all phases sequentially."""
        phases = list(PHASE_CONFIGS.keys())
        start_idx = 0
        for i, p in enumerate(phases):
            if p == start_from:
                start_idx = i
                break

        for phase_name in phases[start_idx:]:
            config = PHASE_CONFIGS[phase_name]
            self.run_phase(config)

    def run_all_experts(self, domains: list[str] | None = None):
        """Train all expert domains."""
        if domains is None:
            domains = list(EXPERT_CONFIGS.keys())

        for domain_name in domains:
            if domain_name in EXPERT_CONFIGS:
                phase_checkpoint = os.path.join(self.checkpoint_base, "phase_phase_4_—_core_1b")
                self.run_expert_training(EXPERT_CONFIGS[domain_name], phase_checkpoint)

    def get_training_summary(self) -> dict:
        return {
            "current_phase": self.current_phase,
            "current_step": self.current_step,
            "phases_completed": [k for k in PHASE_CONFIGS],
            "experts_configured": list(EXPERT_CONFIGS.keys()),
            "dashboard": os.path.join(self.log_dir, "tensorboard"),
        }


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Latent Mesh Training Pipeline")
    parser.add_argument("--phase", type=str, default="all",
                        choices=["all", "phase1", "phase2", "phase3", "phase4"],
                        help="Which phase to run")
    parser.add_argument("--experts", type=str, nargs="*",
                        help="Expert domains to train (coding, reasoning, language)")
    parser.add_argument("--log-dir", type=str, default="./training_logs")
    parser.add_argument("--checkpoint-base", type=str, default="./checkpoints")
    parser.add_argument("--start-from", type=str, default="phase1")
    parser.add_argument("--no-packing", action="store_true",
                        help="Disable token packing (simpler, shows progress immediately)")
    args = parser.parse_args()

    orchestrator = TrainingOrchestrator(
        log_dir=args.log_dir,
        checkpoint_base=args.checkpoint_base,
        use_packing=not args.no_packing,
    )

    if args.phase == "all":
        orchestrator.run_all_phases(start_from=args.start_from)
    else:
        phase_name = f"{args.phase}_" + {"phase1": "250m", "phase2": "500m", "phase3": "750m", "phase4": "1b"}[args.phase]
        if phase_name in PHASE_CONFIGS:
            orchestrator.run_phase(PHASE_CONFIGS[phase_name])

    if args.experts:
        orchestrator.run_all_experts(domains=args.experts)

    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Dashboard: tensorboard --logdir={orchestrator.log_dir}")
    print(f"  Checkpoints: {orchestrator.checkpoint_base}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
