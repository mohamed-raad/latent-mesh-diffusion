"""
Training Monitor — tracks VRAM, speed, loss, nodes during mesh training.
Writes status.json that the web dashboard reads.
"""
import os
import json
import time
import subprocess
import threading
from datetime import datetime


MONITOR_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "training_status.json")


def _nvidia_smi_query(fields: list[str]) -> dict:
    try:
        cmd = ["nvidia-smi", "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if r.returncode != 0:
            return {}
        vals = r.stdout.strip().split(", ")
        return dict(zip(fields, vals))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {}


def get_gpu_stats() -> dict:
    info = _nvidia_smi_query([
        "index", "name", "memory.total", "memory.used", "memory.free",
        "utilization.gpu", "temperature.gpu", "power.draw",
    ])
    if info:
        return {
            "gpu_id": info.get("index", "?"),
            "gpu_name": info.get("name", "?"),
            "vram_total_mb": int(float(info.get("memory.total", 0))),
            "vram_used_mb": int(float(info.get("memory.used", 0))),
            "vram_free_mb": int(float(info.get("memory.free", 0))),
            "gpu_util_pct": float(info.get("utilization.gpu", 0)),
            "gpu_temp_c": float(info.get("temperature.gpu", 0)),
            "power_w": float(info.get("power.draw", 0)),
        }
    return {"gpu_id": "N/A", "vram_used_mb": 0}


class TrainingMonitor:
    def __init__(self):
        self.loss_history: list[dict] = []
        self.start_time = time.time()
        self.step_count = 0
        self.node_count = 0
        self.lock = threading.Lock()
        self.benchmarks: dict = {
            "eval_loss": [],
            "accuracy": [],
            "confidence": [],
            "spec_speedup": [],
            "perplexity": [],
        }
        self._workflow_status = {"current": "idle", "progress": "", "started_at": None}
        self._generation_stats = {
            "total_generated": 0,
            "total_ingested": 0,
            "dataset_sizes": {},
        }

    def record_step(self, step: int, loss: float, node_count: int, phase: str = "mesh"):
        with self.lock:
            self.step_count = step
            self.node_count = node_count
            self.loss_history.append({
                "step": step,
                "loss": loss,
                "phase": phase,
                "time": time.time() - self.start_time,
            })
            self._write_status()

    def record_benchmark(self, key: str, value: float):
        with self.lock:
            if key in self.benchmarks:
                self.benchmarks[key].append({"value": value, "time": time.time() - self.start_time})

    def set_workflow(self, status: str, progress: str = ""):
        with self.lock:
            self._workflow_status = {
                "current": status,
                "progress": progress,
                "started_at": self._workflow_status.get("started_at") or (time.time() if status != "idle" else None),
            }

    def update_gen_stats(self, gen_type: str, count: int):
        with self.lock:
            self._generation_stats["total_generated"] += count if gen_type == "generated" else 0
            self._generation_stats["total_ingested"] += count if gen_type == "ingested" else 0

    def get_speed(self) -> float:
        with self.lock:
            elapsed = time.time() - self.start_time
            if elapsed < 1:
                return 0.0
            return self.step_count / elapsed

    def _write_status(self):
        gpu = get_gpu_stats()
        status = {
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": time.time() - self.start_time,
            "step": self.step_count,
            "loss": self.loss_history[-1]["loss"] if self.loss_history else None,
            "loss_history": self.loss_history[-500:],
            "steps_per_sec": round(self.get_speed(), 4),
            "node_count": self.node_count,
            "gpu": gpu,
            "vram_pct": round(
                gpu["vram_used_mb"] / gpu["vram_total_mb"] * 100, 1
            ) if gpu.get("vram_total_mb", 0) > 0 else 0,
            "benchmarks": self.benchmarks,
            "workflow": self._workflow_status,
            "generation_stats": self._generation_stats,
        }
        with open(MONITOR_FILE, "w") as f:
            json.dump(status, f, indent=2)


monitor = TrainingMonitor()
