from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol

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


