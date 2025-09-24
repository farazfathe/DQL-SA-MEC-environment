from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Hashable, List, Sequence, Tuple
import random


State = Hashable
Action = Hashable


@dataclass
class QLearningAgent:
    """Tabular Q-learning agent.

    This agent maintains Q(s, a) values in a nested dict keyed by state and action.
    """

    alpha: float = 0.1  # learning rate
    gamma: float = 0.99  # discount factor
    epsilon: float = 0.1  # epsilon-greedy exploration

    q_table: Dict[State, Dict[Action, float]] = field(default_factory=dict)

    def get_q(self, state: State, action: Action) -> float:
        return self.q_table.get(state, {}).get(action, 0.0)

    def set_q(self, state: State, action: Action, value: float) -> None:
        self.q_table.setdefault(state, {})[action] = value

    def best_action(self, state: State, actions: Sequence[Action]) -> Tuple[Action, float]:
        if not actions:
            raise ValueError("actions must be non-empty")
        best_a = actions[0]
        best_q = self.get_q(state, best_a)
        for a in actions[1:]:
            q = self.get_q(state, a)
            if q > best_q:
                best_q = q
                best_a = a
        return best_a, best_q

    def select_action(self, state: State, actions: Sequence[Action]) -> Action:
        if random.random() < self.epsilon:
            return random.choice(list(actions))
        best_a, _ = self.best_action(state, actions)
        return best_a

    def update(self, state: State, action: Action, reward: float, next_state: State, next_actions: Sequence[Action]) -> None:
        next_best_q = 0.0
        if next_actions:
            _, next_best_q = self.best_action(next_state, next_actions)
        current_q = self.get_q(state, action)
        target = reward + self.gamma * next_best_q
        new_q = (1 - self.alpha) * current_q + self.alpha * target
        self.set_q(state, action, new_q)


