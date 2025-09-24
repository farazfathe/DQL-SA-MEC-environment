from __future__ import annotations

from dataclasses import dataclass, field
from typing import Deque, Dict, List
from collections import deque
import simpy

from .task import Task, TaskStatus


@dataclass
class EdgeServer:
    """Edge server that receives offloaded tasks from cells.

    Attributes
    ----------
    edge_id: str
        Unique identifier for the edge.
    cpu_rate_cycles_per_s: float
        Processing rate in cycles per second.
    energy_joules: float
        Available energy reserve for computation.
    heat_celsius: float
        A simple heat metric; can be used to throttle.
    distances_from_cells: Dict[str, float]
        Distance in meters from each cell id.
    distances_from_edges: Dict[str, float]
        Distance between this edge and other edges.
    queue: Deque[Task]
        FIFO queue of tasks awaiting processing.
    """

    edge_id: str
    cpu_rate_cycles_per_s: float
    energy_joules: float
    heat_celsius: float = 25.0
    distances_from_cells: Dict[str, float] = field(default_factory=dict)
    distances_from_edges: Dict[str, float] = field(default_factory=dict)
    # Runtime simpy constructs
    env: simpy.Environment | None = field(default=None, repr=False)
    in_queue: simpy.Store | None = field(default=None, repr=False)
    queue: Deque[Task] = field(default_factory=deque)

    compute_energy_j_per_cycle: float = 0.0

    def enqueue(self, task: Task) -> None:
        task.status = TaskStatus.QUEUED
        self.queue.append(task)

    def step(self, now: float, max_time_s: float) -> List[Task]:
        """Process tasks for up to max_time_s of service time.

        Returns tasks that completed during this step.
        """
        completed: List[Task] = []
        service_budget_s = max_time_s
        while self.queue and service_budget_s > 0:
            task = self.queue[0]
            exec_time = task.estimated_compute_time(self.cpu_rate_cycles_per_s)
            if exec_time > service_budget_s:
                # Partial processing not modeled; break until next step
                break
            energy_cost = task.cpu_cycles * self.compute_energy_j_per_cycle
            if energy_cost > self.energy_joules:
                # Not enough energy; cannot process further this step
                break

            # Execute
            task.mark_started(now)
            now += exec_time
            task.mark_completed(now)
            self.energy_joules -= energy_cost
            self.queue.popleft()
            completed.append(task)
            service_budget_s -= exec_time

        return completed

    # --- SimPy integration ---
    def start(self, env: simpy.Environment) -> None:
        self.env = env
        self.in_queue = simpy.Store(env)
        env.process(self._dequeue_loop())

    def put(self, task: Task) -> None:
        if not self.in_queue:
            raise RuntimeError("EdgeServer not started; call start(env) first")
        self.in_queue.put(task)

    def _dequeue_loop(self):
        assert self.env is not None
        assert self.in_queue is not None
        while True:
            task: Task = yield self.in_queue.get()
            # Push into local FIFO and try to serve immediately serially
            self.enqueue(task)
            # Serve head-of-line tasks one-by-one
            while self.queue:
                head = self.queue[0]
                exec_time = head.estimated_compute_time(self.cpu_rate_cycles_per_s)
                energy_cost = head.cpu_cycles * self.compute_energy_j_per_cycle
                # If not enough energy, wait a bit (could model recharge). For now, drop.
                if energy_cost > self.energy_joules:
                    head.mark_dropped()
                    self.queue.popleft()
                    continue
                # Check deadline feasibility
                finish_time = self.env.now + exec_time
                if finish_time > head.deadline:
                    head.mark_dropped()
                    self.queue.pop()
                    continue
                head.mark_started(self.env.now)
                yield self.env.timeout(exec_time)
                head.mark_completed(self.env.now)
                self.energy_joules -= energy_cost
                self.queue.popleft()


