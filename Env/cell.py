from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional
import simpy

from .task import Task, TaskStatus


TaskGenerator = Callable[[float], Iterable[Task]]


@dataclass
class Cell:
    """Represents a user cell (UE + small cell) that creates tasks and may offload.

    This class handles:
    - Task generation based on a provided generator callback
    - Local execution using a simple CPU model
    - Battery accounting for compute and transmission energy
    - Offloading decisions via a pluggable scheduler (not implemented here)
    """

    cell_id: str
    cpu_rate_cycles_per_s: float  # local CPU capability
    battery_energy_joules: float
    task_generator: TaskGenerator

    # Optional associations
    edge_id: Optional[str] = None

    # Metrics/state
    time_s: float = 0.0
    generated_tasks: List[Task] = field(default_factory=list)
    completed_tasks: List[Task] = field(default_factory=list)
    offloaded_tasks: List[Task] = field(default_factory=list)
    dropped_tasks: List[Task] = field(default_factory=list)

    # Model knobs
    compute_energy_j_per_cycle: float = 0.0
    tx_energy_j_per_bit_to_edge: float = 0.0

    def tick(self, dt_s: float, now_s: Optional[float] = None) -> List[Task]:
        """Advance simulation time and generate tasks.

        Returns the list of newly generated tasks at this tick.
        """
        if dt_s <= 0:
            return []
        self.time_s = (self.time_s if now_s is None else now_s) + dt_s

        new_tasks = list(self.task_generator(self.time_s))
        for task in new_tasks:
            task.source_cell_id = self.cell_id
        self.generated_tasks.extend(new_tasks)
        return new_tasks

    def can_execute_locally(self, task: Task) -> bool:
        if self.battery_energy_joules <= 0:
            return False
        # Simple memory/CPU feasibility could be modeled elsewhere; assume feasible here
        return True

    def execute_locally(self, task: Task, now: float) -> bool:
        """Attempt to run task locally; returns True if completed.

        Deducts energy and marks status; if insufficient battery or deadline hit, returns False.
        """
        if not self.can_execute_locally(task):
            return False

        exec_time = task.estimated_compute_time(self.cpu_rate_cycles_per_s)
        finish_time = now + exec_time
        if finish_time > task.deadline:
            # cannot finish before deadline locally
            return False

        energy_cost = task.cpu_cycles * self.compute_energy_j_per_cycle
        if energy_cost > self.battery_energy_joules:
            return False

        # Execute
        task.mark_started(now)
        task.mark_completed(finish_time)
        self.battery_energy_joules -= energy_cost
        self.completed_tasks.append(task)
        return True

    def offload_to_edge_energy_cost(self, task: Task) -> float:
        return task.bit_size * self.tx_energy_j_per_bit_to_edge

    def prepare_offload_to_edge(self, task: Task) -> Optional[float]:
        """Reserve and deduct transmission energy if possible; returns energy cost or None.
        """
        cost = self.offload_to_edge_energy_cost(task)
        if cost > self.battery_energy_joules:
            return None
        self.battery_energy_joules -= cost
        self.offloaded_tasks.append(task)
        task.status = TaskStatus.QUEUED
        return cost

    # --- SimPy integration ---
    env: simpy.Environment | None = field(default=None, repr=False)

    def start(self, env: simpy.Environment) -> None:
        self.env = env
        env.process(self._generation_loop())

    def _generation_loop(self):
        assert self.env is not None
        # This loop only generates tasks every 1 second by default; users can override by custom generators that ignore time spacing
        while True:
            now = float(self.env.now)
            new_tasks = list(self.task_generator(now))
            for task in new_tasks:
                task.source_cell_id = self.cell_id
            self.generated_tasks.extend(new_tasks)
            yield self.env.timeout(1.0)


