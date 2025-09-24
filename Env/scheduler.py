from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, Sequence, Tuple

from .task import Task


class Scheduler(Protocol):
    """Protocol for plugging in offloading algorithms.

    A scheduler decides the placement for newly generated tasks and returns three lists:
    - local_tasks: to be executed at the originating cell
    - edge_tasks: (edge_id, task) pairs to offload to edges
    - cloud_tasks: tasks to send to cloud (when edge declines)
    """

    def place(self, cell_id: str, tasks: List[Task]) -> tuple[list[Task], list[tuple[str, Task]], list[Task]]:
        ...


@dataclass
class GreedyLocalFirstScheduler:
    """Baseline scheduler: try local; else offload to the cell's configured edge; else cloud."""

    default_edge_id: str

    def place(self, cell_id: str, tasks: List[Task]) -> tuple[list[Task], list[tuple[str, Task]], list[Task]]:
        local: list[Task] = []
        to_edge: list[tuple[str, Task]] = []
        to_cloud: list[Task] = []
        for t in tasks:
            # Decision is deferred to the environment which will check feasibility; here we just prefer edge by default
            to_edge.append((self.default_edge_id, t))
        return local, to_edge, to_cloud


@dataclass
class QLSASchedulerAdapter:
    """Adapter that delegates action choice to a provided QL+SA policy.

    Actions are discrete placements: "local", ("edge", edge_id), "cloud".
    """

    policy: object  # expects .select_action(state, actions, score_shaping=None)
    edge_ids: Sequence[str]
    bias_edge: bool = True

    def place(self, cell_id: str, tasks: List[Task]) -> tuple[list[Task], list[tuple[str, Task]], list[Task]]:
        local: list[Task] = []
        to_edge: list[tuple[str, Task]] = []
        to_cloud: list[Task] = []

        actions: list[object] = ["local", "cloud"] + [("edge", e) for e in self.edge_ids]
        state = cell_id  # placeholder state; user can extend with richer features

        def shaping(action: object) -> float:
            if not self.bias_edge:
                return 0.0
            # Small penalty for local/cloud to bias toward edge
            if action == "local":
                return 0.1
            if action == "cloud":
                return 0.15
            return 0.0

        for task in tasks:
            # Pass shaping to favor edge when roughly comparable
            try:
                choice = self.policy.select_action(state, actions, score_shaping=shaping)
            except TypeError:
                choice = self.policy.select_action(state, actions)
            if choice == "local":
                local.append(task)
            elif choice == "cloud":
                to_cloud.append(task)
            elif isinstance(choice, tuple) and choice[0] == "edge":
                to_edge.append((choice[1], task))
            else:
                # Fallback: edge 0 if available else local
                if self.edge_ids:
                    to_edge.append((self.edge_ids[0], task))
                else:
                    local.append(task)

        return local, to_edge, to_cloud


