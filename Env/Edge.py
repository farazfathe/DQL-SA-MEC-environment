from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional
from collections import deque
import simpy
import random

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
    # Modeling knobs
    contention_coeff: float = 0.0  # increases exec_time with queue length
    energy_variability_coeff: float = 0.0  # increases energy with load
    failure_prob: float = 0.0  # probability of processing failure per task
    service_time_jitter_coeff: float = 0.0  # multiplicative jitter on service time
    on_task_completed: Optional[Callable[[Task, float], None]] = field(default=None, repr=False)
    on_task_enqueued: Optional[Callable[[Task, float], None]] = field(default=None, repr=False)
    on_task_started: Optional[Callable[[Task], None]] = field(default=None, repr=False)
    on_task_dropped: Optional[Callable[[Task], None]] = field(default=None, repr=False)

    # Window accumulators for utilization/energy
    busy_time_s_acc: float = 0.0
    completed_count_acc: int = 0
    consumed_energy_j_acc: float = 0.0

    def enqueue(self, task: Task) -> None:
        task.status = TaskStatus.QUEUED
        if task.queued_at is None:
            task.queued_at = self.env.now if self.env is not None else 0.0
        self.queue.append(task)
        if self.on_task_enqueued is not None and self.env is not None:
            try:
                self.on_task_enqueued(task, float(self.env.now))
            except Exception:
                pass

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
                base_exec_time = head.estimated_compute_time(self.cpu_rate_cycles_per_s)
                # Simple contention model: slower when more queued
                load_factor = 1.0 + self.contention_coeff * max(0, len(self.queue) - 1)
                exec_time = base_exec_time * load_factor

                # Energy variability with load
                energy_cost = head.cpu_cycles * self.compute_energy_j_per_cycle
                energy_cost *= (1.0 + self.energy_variability_coeff * max(0, len(self.queue)))
                # If not enough energy, wait a bit (could model recharge). For now, drop.
                if energy_cost > self.energy_joules:
                    head.mark_dropped()
                    if self.on_task_dropped is not None:
                        try:
                            self.on_task_dropped(head)
                        except Exception:
                            pass
                    self.queue.popleft()
                    continue
                # Check deadline feasibility
                finish_time = self.env.now + exec_time
                if finish_time > head.deadline:
                    head.mark_dropped()
                    if self.on_task_dropped is not None:
                        try:
                            self.on_task_dropped(head)
                        except Exception:
                            pass
                    self.queue.popleft()
                    continue
                head.mark_started(self.env.now)
                if self.on_task_started is not None:
                    try:
                        self.on_task_started(head)
                    except Exception:
                        pass
                # queue wait captured via started_at - queued_at in metrics
                # Apply service time jitter if enabled
                if self.service_time_jitter_coeff > 0.0:
                    jit = 1.0 + random.uniform(-self.service_time_jitter_coeff, self.service_time_jitter_coeff)
                    exec_time *= max(0.1, jit)
                yield self.env.timeout(exec_time)
                self.busy_time_s_acc += exec_time
                # Simulate potential processing failure
                failed = (random.random() < self.failure_prob)
                if failed:
                    head.mark_dropped()
                    if self.on_task_dropped is not None:
                        try:
                            self.on_task_dropped(head)
                        except Exception:
                            pass
                else:
                    head.mark_completed(self.env.now)
                self.energy_joules -= energy_cost
                self.consumed_energy_j_acc += energy_cost
                self.queue.popleft()
                if (not failed) and (self.on_task_completed is not None):
                    try:
                        self.on_task_completed(head, energy_cost)
                        self.completed_count_acc += 1
                    except Exception:
                        pass

    def pop_window_counters(self) -> Dict[str, float]:
        stats = {
            "busy_time_s": float(self.busy_time_s_acc),
            "completed": int(self.completed_count_acc),
            "energy_j": float(self.consumed_energy_j_acc),
            "queue_len": int(len(self.queue)),
        }
        self.busy_time_s_acc = 0.0
        self.completed_count_acc = 0
        self.consumed_energy_j_acc = 0.0
        return stats


