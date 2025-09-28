from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple
import random
import numpy as np


Transition = Tuple[Any, Any, float, Any, List[Any], bool]


@dataclass
class PrioritizedReplayBuffer:
    """Simple Prioritized Experience Replay (PER) buffer.

    - Proportional prioritization (p_i = |delta_i| + eps)^alpha
    - Importance sampling weights with annealing beta
    """

    capacity: int
    alpha: float = 0.6
    eps: float = 1e-5

    def __post_init__(self) -> None:
        self.storage: List[Transition] = []
        self.priorities: List[float] = []
        self.next_idx: int = 0

    def __len__(self) -> int:
        return len(self.storage)

    def size(self) -> int:
        return len(self.storage)

    def add(self, s: Any, a: Any, r: float, s2: Any, next_actions: List[Any], done: bool) -> None:
        data = (s, a, float(r), s2, list(next_actions), bool(done))
        if len(self.storage) < self.capacity:
            self.storage.append(data)
            # New transition priority: max existing or 1
            p0 = max(self.priorities) if self.priorities else 1.0
            self.priorities.append(p0)
        else:
            self.storage[self.next_idx] = data
            p0 = max(self.priorities) if self.priorities else 1.0
            self.priorities[self.next_idx] = p0
            self.next_idx = (self.next_idx + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4):
        assert len(self.storage) > 0, "buffer empty"
        probs = np.array(self.priorities, dtype=np.float64)
        probs = (probs + self.eps) ** float(self.alpha)
        probs /= probs.sum()

        idxs = np.random.choice(len(self.storage), size=batch_size, replace=False, p=probs)
        batch = [self.storage[i] for i in idxs]

        # Importance sampling weights
        weights = (len(self.storage) * probs[idxs]) ** (-beta)
        weights = weights / (weights.max() + 1e-8)
        return batch, idxs, weights.astype(np.float32)

    def update_priorities(self, idxs, priorities) -> None:
        for i, p in zip(idxs, priorities):
            self.priorities[int(i)] = float(p)

