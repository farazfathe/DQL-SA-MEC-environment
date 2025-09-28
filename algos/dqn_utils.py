from __future__ import annotations

from typing import Any, Dict


def compute_reward(info: Dict[str, Any], base_reward: float = 0.0) -> float:
    """Composite reward shaping for MEC offloading.

    r = +1 * success
        - 0.8 * (latency/latency_norm)
        - 0.4 * (energy/energy_norm)
        - 0.7 * retry_penalty
        - 0.2 * cloud_penalty

    Returns a value clipped to [-1, 1].
    """
    latency_norm = float(info.get("latency_norm", 2.0))  # seconds
    energy_norm = float(info.get("energy_norm", 5.0))    # joules
    attempt = int(info.get("attempt", 1))
    retry_penalty = 1.0 if attempt > 1 else 0.0
    offload_type = info.get("offload_type")
    cloud_pen = 0.2 if offload_type == "cloud" else 0.0

    r = 0.0
    if bool(info.get("success", False)):
        r += 1.0
    r += -0.8 * (float(info.get("latency", 0.0)) / max(1e-8, latency_norm))
    r += -0.4 * (float(info.get("energy", 0.0)) / max(1e-8, energy_norm))
    r += -0.7 * retry_penalty
    r += -cloud_pen

    # Clip final reward
    if r > 1.0:
        return 1.0
    if r < -1.0:
        return -1.0
    return r


def normalize_obs(obs: Any) -> Any:
    """Placeholder for observation normalization.

    The current DQNAgent uses a hash-based featurizer, so observation
    normalization is a no-op. This function exists to keep parity with
    training scaffolds that expect a normalization hook.
    """
    return obs

