from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Hashable, List, Sequence, Tuple

from .Q_learning_agent import QLearningAgent
from .simulated_annealing import SimulatedAnnealing


State = Hashable
Action = Hashable


ScoreFn = Callable[[float], float]


@dataclass
class QLSAWrapper:
    """Combine Q-learning with Simulated Annealing over action indices.

    SA is used to minimize a negative utility based on Q-values (and optional shaping score).
    """

    ql: QLearningAgent
    sa: SimulatedAnnealing

    def select_action(self, state: State, actions: Sequence[Action], score_shaping: Callable[[Action], float] | None = None) -> Action:
        if not actions:
            raise ValueError("actions must be non-empty")

        # Energy/objective: lower is better. Use negative Q and optional shaping term.
        def energy_for(action: Action) -> float:
            base = -self.ql.get_q(state, action)
            if score_shaping is not None:
                base += score_shaping(action)
            return base

        # Run SA over indices into the actions list
        def energy_by_index(idx_action):
            return energy_for(idx_action)

        # SimulatedAnnealing works on the action elements directly via energy fn
        chosen, _ = self.sa.select(actions, energy_for)
        return chosen


