"""
Async RL Pipeline — separates generation (actors) from training (learner).
Inspired by GLM-5's Asynchronous Reinforcement Learning + Agent RL.

Architecture:
  Mesh Actors (N GPUs) → generate trajectories → Replay Buffer → Learner trains

Benefits:
  - 20-50% higher GPU utilization
  - Never block generation waiting for training
  - Train on yesterday's conversations while generating today's
"""
import os
import json
import time
import random
import threading
import pickle
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import torch
import numpy as np


@dataclass
class TrajectoryStep:
    """A single step in an agent trajectory."""
    state: dict  # Input state (tokens, embeddings, etc.)
    action: dict  # Action taken (expert_id, prediction, etc.)
    reward: float = 0.0
    next_state: dict | None = None
    done: bool = False
    metadata: dict = field(default_factory=dict)


@dataclass
class Trajectory:
    """Full agent trajectory from start to finish."""
    steps: list[TrajectoryStep] = field(default_factory=list)
    total_reward: float = 0.0
    task_type: str = ""
    task_id: str = ""
    length: int = 0

    def add_step(self, step: TrajectoryStep):
        self.steps.append(step)
        self.total_reward += step.reward
        self.length = len(self.steps)

    def get_rewards(self) -> list[float]:
        return [s.reward for s in self.steps]

    def get_actions(self) -> list[dict]:
        return [s.action for s in self.steps]


class ReplayBuffer:
    """Prioritized replay buffer for agent trajectories."""

    def __init__(self, capacity: int = 1_000_000, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer: deque[tuple[float, Trajectory]] = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._count = 0

    def push(self, trajectory: Trajectory, priority: float | None = None):
        if priority is None:
            priority = abs(trajectory.total_reward) + 1e-6
        with self._lock:
            self.buffer.append((priority ** self.alpha, trajectory))
            self._count += 1

    def sample(self, batch_size: int) -> list[Trajectory]:
        with self._lock:
            if len(self.buffer) < batch_size:
                return list(self.buffer)
            total = sum(p for p, _ in self.buffer)
            probs = [p / total for p, _ in self.buffer]
            indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
            return [self.buffer[i][1] for i in indices]

    def __len__(self):
        return len(self.buffer)


class Actor:
    """Mesh actor that generates trajectories."""

    def __init__(self, actor_id: str, mesh, device: torch.device):
        self.actor_id = actor_id
        self.mesh = mesh
        self.device = device
        self.mesh.to(device)

    def generate_trajectory(self, task: dict) -> Trajectory:
        """Generate a full trajectory for a given task."""
        trajectory = Trajectory(
            task_type=task.get("type", "generic"),
            task_id=task.get("id", f"{self.actor_id}_{time.time()}"),
        )

        state = task.get("input", "")
        max_steps = task.get("max_steps", 20)
        done = False
        step_count = 0

        while not done and step_count < max_steps:
            with torch.no_grad():
                action = self.mesh.generate(state) if callable(self.mesh.generate) else self._default_action(state)

            reward = task.get("reward_fn", lambda s, a, t: 0.0)(state, action, step_count)

            step = TrajectoryStep(
                state={"text": state} if isinstance(state, str) else state,
                action={"output": action} if isinstance(action, str) else action,
                reward=reward,
                done=step_count >= max_steps - 1,
            )
            trajectory.add_step(step)

            state = action if isinstance(action, str) else action.get("output", state)
            done = task.get("done_fn", lambda s, a, t: step_count >= max_steps - 1)(state, action, step_count)
            step_count += 1

        return trajectory

    def _default_action(self, state: Any) -> dict:
        return {"output": str(state), "expert": "default"}


class Learner:
    """Trains mesh weights from replay buffer trajectories."""

    def __init__(self, mesh, optimizer, device: torch.device, lr: float = 3e-4):
        self.mesh = mesh
        self.optimizer = optimizer
        self.device = device
        self.lr = lr
        self.mesh.to(device)

    def train_on_trajectory(self, trajectory: Trajectory) -> dict[str, float]:
        """Train on a single trajectory with trajectory-level rewards."""
        total_loss = 0.0
        expert_losses = {}
        n_steps = len(trajectory.steps)

        for i, step in enumerate(trajectory.steps):
            state = step.state
            action = step.action
            reward = step.reward

            if "input_ids" in state:
                x = torch.as_tensor(state["input_ids"]).to(self.device)
                target = torch.as_tensor(action.get("target_ids", x)).to(self.device)
            else:
                continue

            t = torch.tensor([[i / max(n_steps, 1)]]).to(self.device)
            loss_dict = self.mesh._train_step(x.unsqueeze(0), target.unsqueeze(0), t)
            step_loss = sum(loss_dict.values()) / max(len(loss_dict), 1)

            weighted_loss = step_loss * (1.0 - 0.5 * reward)
            weighted_loss.backward()

            total_loss += weighted_loss.item()
            for k, v in loss_dict.items():
                expert_losses[k] = expert_losses.get(k, 0.0) + v

        self.optimizer.step()
        self.optimizer.zero_grad()

        return {"total_loss": total_loss / max(n_steps, 1), **{k: v / max(n_steps, 1) for k, v in expert_losses.items()}}


class AsyncRLPipeline:
    """Full async RL pipeline: N actors → replay buffer → learner."""

    def __init__(
        self,
        mesh,
        optimizer,
        num_actors: int = 4,
        buffer_capacity: int = 100_000,
        batch_size: int = 32,
        learner_device: torch.device | None = None,
        actor_devices: list[torch.device] | None = None,
        replay_dir: str = "./replay_buffer",
    ):
        self.mesh = mesh
        self.optimizer = optimizer
        self.buffer = ReplayBuffer(capacity=buffer_capacity)
        self.batch_size = batch_size
        self.replay_dir = replay_dir
        os.makedirs(replay_dir, exist_ok=True)

        if learner_device is None:
            learner_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if actor_devices is None:
            actor_devices = [learner_device] * num_actors

        self.learner = Learner(mesh, optimizer, learner_device, lr=optimizer.param_groups[0]["lr"] if optimizer.param_groups else 3e-4)

        self.actors: list[Actor] = []
        model_copies = [type(mesh)(**mesh.__dict__) for _ in range(num_actors)]
        for i, (mc, dev) in enumerate(zip(model_copies, actor_devices)):
            self.actors.append(Actor(f"actor_{i}", mc, dev))

        self._running = False
        self._threads: list[threading.Thread] = []
        self._gen_count = 0
        self._train_count = 0

    def start_actors(self, task_generator, num_threads: int = 4):
        """Start actor threads that generate trajectories."""
        self._running = True

        def _actor_loop(actor: Actor):
            while self._running:
                task = task_generator()
                if task is None:
                    time.sleep(0.1)
                    continue
                trajectory = actor.generate_trajectory(task)
                self.buffer.push(trajectory)
                self._gen_count += 1

        for actor in self.actors[:num_threads]:
            t = threading.Thread(target=_actor_loop, args=(actor,), daemon=True)
            t.start()
            self._threads.append(t)

    def train_step(self) -> dict[str, float]:
        """One training step from the replay buffer."""
        if len(self.buffer) < self.batch_size:
            return {"total_loss": 0.0}

        trajectories = self.buffer.sample(self.batch_size)
        total_metrics = {}
        num_traj = 0

        for traj in trajectories:
            metrics = self.learner.train_on_trajectory(traj)
            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0.0) + v
            num_traj += 1

        self._train_count += num_traj
        return {k: v / max(num_traj, 1) for k, v in total_metrics.items()}

    def save_replay_buffer(self, path: str | None = None):
        path = path or os.path.join(self.replay_dir, f"replay_{int(time.time())}.pkl")
        with open(path, "wb") as f:
            pickle.dump(list(self.buffer.buffer), f)
        print(f"Saved {len(self.buffer)} trajectories to {path}")
        return path

    def load_replay_buffer(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        for prio, traj in data:
            self.buffer.push(traj, prio)
        print(f"Loaded {len(data)} trajectories from {path}")

    def sync_actors(self):
        """Sync actor weights from learner."""
        state = self.mesh.state_dict()
        for actor in self.actors:
            actor.mesh.load_state_dict(state)

    def stop(self):
        self._running = False
        for t in self._threads:
            t.join(timeout=5)
        self.save_replay_buffer()

    def get_stats(self) -> dict:
        return {
            "buffer_size": len(self.buffer),
            "generated": self._gen_count,
            "trained": self._train_count,
            "num_actors": len(self.actors),
        }


class AgentRLEngine:
    """Agent RL with trajectory-level rewards (plan → tool → verify → correct → final)."""

    def __init__(self, mesh, router, reward_weights: dict | None = None):
        self.mesh = mesh
        self.router = router
        self.reward_weights = reward_weights or {
            "planning": 0.15,
            "tool_use": 0.20,
            "verification": 0.25,
            "correction": 0.25,
            "final": 0.15,
        }

    def score_trajectory(self, trajectory: Trajectory) -> dict[str, float]:
        """Score each phase of an agent trajectory."""
        scores = {}
        steps = trajectory.steps

        for i, step in enumerate(steps):
            action = step.action
            phase = action.get("phase", "unknown")

            if phase == "planning":
                scores["planning"] = step.reward * self.reward_weights.get("planning", 0.15)
            elif phase == "tool_use":
                scores["tool_use"] = step.reward * self.reward_weights.get("tool_use", 0.20)
            elif phase == "verification":
                scores["verification"] = step.reward * self.reward_weights.get("verification", 0.25)
            elif phase == "correction":
                scores["correction"] = step.reward * self.reward_weights.get("correction", 0.25)
            elif phase == "final":
                scores["final"] = step.reward * self.reward_weights.get("final", 0.15)

        return scores

    def compute_trajectory_reward(self, trajectory: Trajectory) -> float:
        """Compute total weighted trajectory reward."""
        phase_scores = self.score_trajectory(trajectory)
        return sum(phase_scores.values())
