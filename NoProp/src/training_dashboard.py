"""
Training Dashboard — live TensorBoard + rich CLI.
Shows loss, VRAM, tok/s, expert stats, latent probe, routing entropy in real-time.
"""
import os
import time
import threading
import numpy as np
from collections import deque
from torch.utils.tensorboard import SummaryWriter


class TrainingDashboard:
    """Real-time training dashboard with TensorBoard + rich terminal output."""

    def __init__(self, log_dir: str, port: int = 6006):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        self.port = port

        self.loss_history = deque(maxlen=1000)
        self.tok_s_history = deque(maxlen=100)
        self.vram_history = deque(maxlen=100)
        self.expert_history = deque(maxlen=100)
        self.router_entropy_history = deque(maxlen=100)
        self.lr_history = deque(maxlen=100)
        self.grad_norm_history = deque(maxlen=100)

        self.step = 0
        self.epoch = 0
        self.start_time = time.time()
        self._last_log_time = time.time()
        self._lock = threading.Lock()

    def log_step(
        self,
        loss: float,
        lr: float,
        vram_gb: float,
        tok_s: float,
        num_gpu_experts: int,
        router_entropy: float = 0.0,
        grad_norm: float = 0.0,
        expert_losses: dict[str, float] | None = None,
        latent_probe: dict | None = None,
        domain: str = "",
        phase: str = "",
    ):
        with self._lock:
            self.step += 1
            self.loss_history.append(loss)
            self.tok_s_history.append(tok_s)
            self.vram_history.append(vram_gb)
            self.expert_history.append(num_gpu_experts)
            self.router_entropy_history.append(router_entropy)
            self.lr_history.append(lr)
            self.grad_norm_history.append(grad_norm)

            self.writer.add_scalar("train/loss", loss, self.step)
            self.writer.add_scalar("train/lr", lr, self.step)
            self.writer.add_scalar("train/tokens_per_sec", tok_s, self.step)
            self.writer.add_scalar("train/vram_gb", vram_gb, self.step)
            self.writer.add_scalar("train/gpu_experts", num_gpu_experts, self.step)
            self.writer.add_scalar("train/router_entropy", router_entropy, self.step)
            self.writer.add_scalar("train/grad_norm", grad_norm, self.step)

            if expert_losses:
                for k, v in expert_losses.items():
                    self.writer.add_scalar(f"experts/{k}", v, self.step)

            if domain:
                self.writer.add_text("meta/domain", domain, self.step)
            if phase:
                self.writer.add_text("meta/phase", phase, self.step)

    def log_epoch(self, epoch: int, avg_loss: float, avg_tok_s: float, accuracy: float = 0.0):
        with self._lock:
            self.epoch = epoch
            self.writer.add_scalar("train/epoch_loss", avg_loss, epoch)
            self.writer.add_scalar("train/epoch_tok_s", avg_tok_s, epoch)
            if accuracy > 0:
                self.writer.add_scalar("eval/accuracy", accuracy, epoch)

    def log_latent_probe(self, node_stats: dict):
        with self._lock:
            for nid, stats in node_stats.items():
                self.writer.add_scalar(f"latent/{nid}_stability", stats.get("stability", 0), self.step)
                self.writer.add_text(f"latent/{nid}_concept", stats.get("top_concept", ""), self.step)

    def log_router_routes(self, routes: list[tuple[str, str, float]]):
        with self._lock:
            for expert_id, domain, score in routes:
                self.writer.add_scalar(f"router/{expert_id}_{domain}", score, self.step)

    def log_checkpoint(self, path: str, size_mb: float):
        self.writer.add_text("checkpoint/path", path, self.step)
        self.writer.add_scalar("checkpoint/size_mb", size_mb, self.step)

    def log_mitosis(self, parent_id: str, child_id: str, loss_before: float, loss_after: float):
        with self._lock:
            self.writer.add_scalar(f"mitosis/{parent_id}_to_{child_id}/loss_before", loss_before, self.step)
            self.writer.add_scalar(f"mitosis/{parent_id}_to_{child_id}/loss_after", loss_after, self.step)

    def print_status(self, batch_idx: int, max_batches: int):
        """Print a compact status line to terminal."""
        elapsed = time.time() - self.start_time
        avg_loss = np.mean(self.loss_history[-20:]) if self.loss_history else 0
        avg_tok_s = np.mean(self.tok_s_history[-10:]) if self.tok_s_history else 0
        avg_vram = np.mean(self.vram_history[-10:]) if self.vram_history else 0
        avg_experts = int(np.mean(self.expert_history[-10:])) if self.expert_history else 0
        avg_entropy = np.mean(self.router_entropy_history[-10:]) if self.router_entropy_history else 0

        progress = batch_idx / max_batches * 100 if max_batches else 0
        eta = (elapsed / (batch_idx + 1)) * (max_batches - batch_idx) if batch_idx > 0 and max_batches else 0

        line = (
            f"Step {self.step:>6d} | "
            f"Loss {avg_loss:.4f} | "
            f"{avg_tok_s:>7.0f} tok/s | "
            f"VRAM {avg_vram:.1f}GB | "
            f"Experts {avg_experts:>2d} | "
            f"Entropy {avg_entropy:.3f} | "
            f"Progress {progress:>5.1f}% | "
            f"ETA {eta:>7.0f}s"
        )
        print(f"\r{' ' * 120}\r{line}", end="", flush=True)

    def print_epoch_summary(self):
        avg_loss = np.mean(self.loss_history[-100:]) if self.loss_history else 0
        avg_tok_s = np.mean(self.tok_s_history[-100:]) if self.tok_s_history else 0
        avg_vram = np.mean(self.vram_history[-100:]) if self.vram_history else 0
        elapsed = time.time() - self.start_time
        print(
            f"\n{'='*70}\n"
            f"  Epoch {self.epoch:<4d} | "
            f"Avg Loss {avg_loss:.4f} | "
            f"{avg_tok_s:>7.0f} tok/s | "
            f"VRAM {avg_vram:.1f}GB | "
            f"Elapsed {elapsed:.0f}s\n"
            f"{'='*70}"
        )

    def flush(self):
        self.writer.flush()

    def close(self):
        self.writer.close()
        print(f"\nDashboard saved to {self.log_dir}")
        print(f"View with: tensorboard --logdir={self.log_dir}")
