from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Hashable, Sequence, Optional

from .DQN_agent import DQNAgent
from .simulated_annealing import SimulatedAnnealing


State = Hashable
Action = Hashable


@dataclass
class QLSAWrapper:
    """Combine a DQN agent with Simulated Annealing over discrete actions."""

    ql: DQNAgent
    sa: SimulatedAnnealing

    def select_action(
        self,
        state: State,
        actions: Sequence[Action],
        score_shaping: Optional[Callable[[Action], float]] = None,
    ) -> Action:
        if not actions:
            raise ValueError("actions must be non-empty")

        action_list = list(actions)
        if random.random() < self.ql.epsilon:
            return random.choice(action_list)

        q_estimates = {action: self.ql.get_q(state, action) for action in action_list}

        def energy_for(action: Action) -> float:
            energy = -q_estimates[action]
            if score_shaping is not None:
                energy += score_shaping(action)
            return energy

        best_action = max(action_list, key=lambda act: q_estimates[act])
        start_index = action_list.index(best_action)

        chosen, _ = self.sa.select(action_list, energy_for, start_index=start_index)
        return chosen

    def update(
        self,
        state: State,
        action: Action,
        reward: float,
        next_state: State,
        next_actions: Sequence[Action],
        done: Optional[bool] = None,
    ) -> None:
        """Pass-through learning update to the underlying DQN.

        Use this to keep training logic co-located with the selection policy.
        """
        self.ql.update(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            next_actions=next_actions,
            done=done,
        )
