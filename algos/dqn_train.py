from __future__ import annotations

"""
Short Prioritized Recipe (DQN training)

1) Warm-up buffer: run a simple heuristic (local-first) for 5kâ€“20k
   transitions to fill replay before learning.
2) Add strict attempt caps: enforce max_attempts_per_task=3 and penalize
   retries in the reward; reject further retries.
3) Reward redesign: composite reward
     r = 1*success - 0.8*(latency/lat_norm) - 0.4*(energy/eng_norm)
         - 0.7*retry_penalty - 0.2*cloud_penalty, then clip to [-1,1].
4) Training algorithm: Double DQN (+PER already in agent) [n-step n=3
   can be added later]. Sync target every 1000 steps.
5) Monitoring & eval: every 10k env steps, freeze policy and run 3 eval
   episodes on a slightly noisier env (jitter, small failure prob).
6) Ablations: baseline DQN â†’ +PER â†’ +Double â†’ +n-step â†’ +SA.
"""

import os
import random
import time
from typing import Any, Dict, List, Sequence

import yaml
import numpy as np
import torch

from algos.DQN_agent import DQNAgent
from algos.device_utils import select_device
from algos.dqn_utils import compute_reward
from Env.utils import energy_per_cycle_from_power


# ---------- hyperparams ----------
CONFIG: Dict[str, Any] = {
    "seed": 42,
    "device": select_device(),
    "total_env_steps": 300_000,
    "warmup_steps": 10_000,
    "batch_size": 64,
    "lr": 5e-4,
    "gamma": 0.99,
    "target_update": 1000,
    # Exploration (delegated to agent's schedule)
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay_steps": 150_000,
    # Attempts and eval
    "max_attempts": 3,
    "eval_interval": 10_000,
    "eval_episodes": 3,
    # Env
    "env_cfg_path": os.path.join("data", "env.yaml"),
    "edges_count": 10,
}
# ----------------------------------


def set_seed(seed: int, device: str | None = None) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device is None:
        device = select_device()
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class MECTrainEnv:
    """Lightweight training wrapper for the MEC setting.

    This avoids running the full SimPy experiment, while producing
    transitions compatible with the existing DQNAgent (hash featurizer).
    """

    def __init__(self, cfg_path: str, edges_count: int = 10, max_attempts: int = 3):
        raw = _load_yaml(cfg_path)
        # Extract speeds/energies similar to experiments/experiment02.parse_env_config
        net = (raw.get("network") or {})
        bw = (net.get("bandwidth") or {})
        lat = (net.get("latency") or {})
        self.bw_iot_to_mec = _parse_rate(bw.get("iot_to_mec", "50 Mbps"))
        self.lat_iot_to_mec = _parse_time(lat.get("iot_to_mec", "25 ms"))
        self.jitter_s = float((net or {}).get("jitter_ms", 0.0)) / 1000.0

        cell_specs = ((raw.get("cell_devices") or {}).get("specs") or {})
        edge_specs = ((raw.get("edge_devices") or {}).get("specs") or {})
        cloud_specs = ((raw.get("cloud") or {}).get("specs") or {})
        self.cell_cpu = _parse_cpu(cell_specs.get("cpu", "2 GHz"))
        self.edge_cpu = _parse_cpu(edge_specs.get("cpu", "8 GHz"))
        self.cloud_cpu = _parse_cpu(cloud_specs.get("cpu", "16 GHz"))

        # Energy parameters (approximate)
        self.cell_power_w = 2.0
        self.cell_energy_per_cycle = energy_per_cycle_from_power(self.cell_power_w, self.cell_cpu)
        self.edge_compute_energy_per_cycle = 5e-12
        self.cloud_compute_energy_per_cycle = 5e-12
        self.tx_energy_per_bit = float(((raw.get("cell_devices") or {}).get("specs") or {}).get("tx_energy_per_bit", 1e-9))

        # Failure probabilities and caps
        sim = (raw.get("simulation") or {})
        fp = (sim.get("failure_prob") or {})
        self.fail_p_local = float(fp.get("local", 0.02))
        self.fail_p_edge = float(fp.get("edge", 0.02))
        self.fail_p_cloud = float(fp.get("cloud", 0.01))
        self.max_attempts = int(sim.get("max_attempts_per_task", max_attempts))

        # Action space
        self.edges: List[str] = [f"edge-{i}" for i in range(int(edges_count))]
        self.feature_dim: int = 384  # matches DQN_agent default featurizer

        # Task template
        self.task_bits = 1_000_000
        self.task_cycles = int(5e8)
        self.deadline_s = 1.0

        # Runtime
        self._attempt: int = 0
        self._eval_mode: bool = False
        self._state: str = "cell-0"

    def reset(self, eval_mode: bool = False) -> str:
        self._attempt = 0
        self._eval_mode = bool(eval_mode)
        # Randomize cell id a bit for variety
        self._state = f"cell-{random.randrange(16)}"
        return self._state

    def valid_actions(self, state: Any | None = None) -> List[Any]:
        return ["local", "cloud"] + [("edge", e) for e in self.edges]

    def heuristic_select(self, state: Any, actions: Sequence[Any]) -> Any:
        # Simple local-first heuristic
        if "local" in actions:
            return "local"
        return actions[0]

    def _sample_failure(self, offload_type: str) -> bool:
        # True if failure occurs
        if offload_type == "local":
            p = self.fail_p_local
        elif offload_type == "cloud":
            p = self.fail_p_cloud
        else:
            p = self.fail_p_edge
        # Slightly noiser during eval
        if self._eval_mode:
            p = min(1.0, p * 1.5)
        return random.random() < p

    def step(self, action: Any):
        # Compute latency/energy based on simple models
        bit_size = self.task_bits
        cycles = self.task_cycles
        jitter = (self.jitter_s * 1.5) if self._eval_mode else self.jitter_s

        if action == "local":
            offload_type = "local"
            tx_time = 0.0
            compute_time = cycles / max(1e-8, self.cell_cpu)
            energy = cycles * self.cell_energy_per_cycle
        elif action == "cloud":
            offload_type = "cloud"
            tx_time = (bit_size / max(1e-8, self.bw_iot_to_mec)) + max(0.0, self.lat_iot_to_mec + random.uniform(-jitter, jitter))
            compute_time = cycles / max(1e-8, self.cloud_cpu)
            energy = bit_size * self.tx_energy_per_bit + cycles * self.cloud_compute_energy_per_cycle
        elif isinstance(action, tuple) and action[0] == "edge":
            offload_type = "edge"
            tx_time = (bit_size / max(1e-8, self.bw_iot_to_mec)) + max(0.0, self.lat_iot_to_mec + random.uniform(-jitter, jitter))
            compute_time = cycles / max(1e-8, self.edge_cpu)
            energy = bit_size * self.tx_energy_per_bit + cycles * self.edge_compute_energy_per_cycle
        else:
            # Unknown action: treat as local
            offload_type = "local"
            tx_time = 0.0
            compute_time = cycles / max(1e-8, self.cell_cpu)
            energy = cycles * self.cell_energy_per_cycle

        latency = tx_time + compute_time

        # Attempt accounting
        self._attempt += 1
        failed = self._sample_failure(offload_type)
        success = not failed
        done = False
        if success:
            done = True
        elif self._attempt >= self.max_attempts:
            # Cap reached â€” reject more retries
            done = True

        info = {
            "success": bool(success),
            "latency": float(latency),
            "energy": float(energy),
            "attempt": int(self._attempt),
            "offload_type": offload_type,
            "completed": bool(success),
            # normalizers for reward
            "latency_norm": 2.0,
            "energy_norm": 5.0,
        }

        # State is coarse here (cell id); next_actions always available
        s2 = self._state if not done else self.reset(eval_mode=self._eval_mode)
        next_actions = self.valid_actions(s2)
        base_reward = 0.0
        return s2, base_reward, done, {**info, "next_actions": next_actions}


def _parse_cpu(val: Any) -> float:
    s = str(val).lower()
    import re
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", s)
    num = float(m.group(1)) if m else 1.0
    if "ghz" in s or " g" in s:
        return num * 1e9
    if "mhz" in s or " m" in s:
        return num * 1e6
    return float(num)


def _parse_rate(s: Any) -> float:
    if not isinstance(s, str):
        return float(s)
    ss = s.strip().lower()
    import re
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", ss)
    num = float(m.group(1)) if m else 0.0
    if "gbps" in ss or " g" in ss:
        return num * 1e9
    if "mbps" in ss or " m" in ss:
        return num * 1e6
    if "kbps" in ss or " k" in ss:
        return num * 1e3
    return num


def _parse_time(s: Any) -> float:
    try:
        if isinstance(s, (int, float)):
            return float(s)
        ss = str(s).strip().lower()
        import re
        m = re.search(r"([0-9]+(?:\.[0-9]+)?)", ss)
        num = float(m.group(1)) if m else 0.0
        if "ms" in ss:
            return num / 1000.0
        if "s" in ss:
            return num
        return num
    except Exception:
        return 0.0


def warmup(env: MECTrainEnv, agent: DQNAgent, n_steps: int) -> None:
    s = env.reset()
    for _ in range(int(n_steps)):
        actions = env.valid_actions(s)
        a = env.heuristic_select(s, actions)
        s2, base_r, done, info = env.step(a)
        r = compute_reward(info, base_r)
        agent.update(s, a, r, s2, info.get("next_actions", []), done)
        s = env.reset() if done else s2


def evaluate_policy(env: MECTrainEnv, agent: DQNAgent, episodes: int = 3) -> Dict[str, float]:
    stats = {"lat": [], "succ": []}
    for _ in range(int(episodes)):
        s = env.reset(eval_mode=True)
        done = False
        ep_lat: List[float] = []
        succ = 0
        # Run until a fixed number of completions to stabilize stats
        completes = 0
        while completes < 50:
            acts = env.valid_actions(s)
            a = agent.select_action(s, acts)
            s, base_r, done, info = env.step(a)
            if info.get("completed", False):
                succ += 1
                completes += 1
            if "latency" in info:
                ep_lat.append(float(info["latency"]))
        stats["lat"].append(float(np.mean(ep_lat) if ep_lat else 0.0))
        stats["succ"].append(float(succ))
    return {"lat": float(np.mean(stats["lat"])), "succ": float(np.mean(stats["succ"]))}


def train() -> None:
    device = CONFIG["device"]
    set_seed(int(CONFIG["seed"]), device=device)

    env = MECTrainEnv(
        cfg_path=str(CONFIG["env_cfg_path"]),
        edges_count=int(CONFIG["edges_count"]),
        max_attempts=int(CONFIG["max_attempts"]),
    )
    agent = DQNAgent(
        alpha=float(CONFIG["lr"]),
        gamma=float(CONFIG["gamma"]),
        input_dim=int(getattr(env, "feature_dim", 384)),
        target_update_freq=int(CONFIG["target_update"]),
        epsilon=float(CONFIG["epsilon_start"]),
        epsilon_end=float(CONFIG["epsilon_end"]),
        epsilon_decay_steps=int(CONFIG["epsilon_decay_steps"]),
        device=device,
        use_double_dqn=True,
        use_per=True,
    )

    # Warmup using heuristic
    print("Warming replay buffer...")
    warmup(env, agent, int(CONFIG["warmup_steps"]))

    steps = 0
    s = env.reset()
    start_t = time.time()
    while steps < int(CONFIG["total_env_steps"]):
        actions = env.valid_actions(s)
        a = agent.select_action(s, actions)
        s2, base_r, done, info = env.step(a)
        r = compute_reward(info, base_r)
        agent.update(s, a, r, s2, info.get("next_actions", []), done)
        steps += 1

        if (steps % int(CONFIG["eval_interval"])) == 0:
            eval_stats = evaluate_policy(env, agent, episodes=int(CONFIG["eval_episodes"]))
            dt = time.time() - start_t
            print(f"[{steps}] eval lat={eval_stats['lat']:.4f} succ={eval_stats['succ']:.2f} eps={agent.epsilon:.3f} wall={dt/60.0:.1f}m")

        s = env.reset() if done else s2

    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(agent.q.state_dict(), os.path.join("models", "dqn_final.pth"))
    print("Saved model to models/dqn_final.pth")


if __name__ == "__main__":
    train()



