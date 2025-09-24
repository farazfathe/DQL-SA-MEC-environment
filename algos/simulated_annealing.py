from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence, Tuple, TypeVar


Action = TypeVar("Action")


EnergyFn = Callable[[Action], float]
NeighborFn = Callable[[Sequence[Action], int], int]


def default_neighbor(actions: Sequence[Action], idx: int) -> int:
    if not actions:
        raise ValueError("actions must be non-empty")
    if len(actions) == 1:
        return 0
    # Propose a different random index
    j = idx
    while j == idx:
        j = random.randrange(0, len(actions))
    return j


@dataclass
class SimulatedAnnealing:
    """Generic simulated annealing over a discrete action set.

    The algorithm searches over indices in the provided actions list using
    an energy/objective function E(action) to minimize.
    """

    initial_temperature: float = 1.0
    cooling_rate: float = 0.99
    min_temperature: float = 1e-3
    steps_per_call: int = 20
    neighbor_fn: NeighborFn = default_neighbor

    def select(self, actions: Sequence[Action], energy_fn: EnergyFn, start_index: int | None = None) -> Tuple[Action, int]:
        if not actions:
            raise ValueError("actions must be non-empty")

        idx = 0 if start_index is None else max(0, min(len(actions) - 1, start_index))
        current_energy = energy_fn(actions[idx])
        best_idx = idx
        best_energy = current_energy
        T = self.initial_temperature

        for _ in range(self.steps_per_call):
            if T <= self.min_temperature:
                break
            j = self.neighbor_fn(actions, idx)
            candidate_energy = energy_fn(actions[j])
            delta = candidate_energy - current_energy
            if delta <= 0:
                idx = j
                current_energy = candidate_energy
            else:
                p = math.exp(-delta / T)
                if random.random() < p:
                    idx = j
                    current_energy = candidate_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_idx = idx
            T *= self.cooling_rate

        return actions[best_idx], best_idx


