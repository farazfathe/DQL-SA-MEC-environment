from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List
import simpy

from .task import Task, TaskStatus


@dataclass
class Cloud:
    """Cloud layer processing tasks forwarded from edges.

    Attributes
    ----------
    cpu_rate_cycles_per_s: float
        Aggregate compute capacity available.
    energy_joules: float
        Energy reserve for computation (can be set high to ignore energy limits).
    distances_from_edges: Dict[str, float]
        Distance in meters from each edge.
    """

    cpu_rate_cycles_per_s: float
    energy_joules: float
    distances_from_edges: Dict[str, float] = field(default_factory=dict)

    compute_energy_j_per_cycle: float = 0.0
    env: simpy.Environment | None = field(default=None, repr=False)
    in_queue: simpy.Store | None = field(default=None, repr=False)

    def process(self, tasks: List[Task], now: float) -> List[Task]:
        """Process all given tasks sequentially and return those completed.

        This is a simple baseline that assumes cloud is powerful and processes immediately
        if energy allows. No queueing delay is modeled here.
        """
        completed: List[Task] = []
        for task in tasks:
            exec_time = task.estimated_compute_time(self.cpu_rate_cycles_per_s)
            energy_cost = task.cpu_cycles * self.compute_energy_j_per_cycle
            if energy_cost > self.energy_joules:
                # insufficient energy; skip
                continue
            task.mark_started(now)
            task.mark_completed(now + exec_time)
            self.energy_joules -= energy_cost
            completed.append(task)
        return completed

    # --- SimPy integration ---
    def start(self, env: simpy.Environment) -> None:
        self.env = env
        self.in_queue = simpy.Store(env)
        env.process(self._serve_loop())

    def put(self, task: Task) -> None:
        if not self.in_queue:
            raise RuntimeError("Cloud not started; call start(env) first")
        self.in_queue.put(task)

    def _serve_loop(self):
        assert self.env is not None
        assert self.in_queue is not None
        while True:
            task: Task = yield self.in_queue.get()
            exec_time = task.estimated_compute_time(self.cpu_rate_cycles_per_s)
            energy_cost = task.cpu_cycles * self.compute_energy_j_per_cycle
            if energy_cost > self.energy_joules:
                task.mark_dropped()
                continue
            finish_time = self.env.now + exec_time
            if finish_time > task.deadline:
                task.mark_dropped()
                continue
            task.mark_started(self.env.now)
            yield self.env.timeout(exec_time)
            task.mark_completed(self.env.now)
            self.energy_joules -= energy_cost


