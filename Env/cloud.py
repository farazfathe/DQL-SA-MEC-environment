from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional
import simpy
import random

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
    # Modeling knobs
    contention_coeff: float = 0.0
    energy_variability_coeff: float = 0.0
    failure_prob: float = 0.0
    service_time_jitter_coeff: float = 0.0
    env: simpy.Environment | None = field(default=None, repr=False)
    in_queue: simpy.Store | None = field(default=None, repr=False)
    on_task_completed: Optional[Callable[[Task, float], None]] = field(default=None, repr=False)
    on_task_enqueued: Optional[Callable[[Task, float], None]] = field(default=None, repr=False)
    on_task_started: Optional[Callable[[Task], None]] = field(default=None, repr=False)
    on_task_dropped: Optional[Callable[[Task], None]] = field(default=None, repr=False)

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
        if task.queued_at is None and self.env is not None:
            task.queued_at = self.env.now
        self.in_queue.put(task)
        if self.on_task_enqueued is not None and self.env is not None:
            try:
                self.on_task_enqueued(task, float(self.env.now))
            except Exception:
                pass

    def _serve_loop(self):
        assert self.env is not None
        assert self.in_queue is not None
        while True:
            task: Task = yield self.in_queue.get()
            base_exec_time = task.estimated_compute_time(self.cpu_rate_cycles_per_s)
            # Approximate contention using current queue length
            q_len = len(self.in_queue.items) if hasattr(self.in_queue, 'items') else 0
            load_factor = 1.0 + self.contention_coeff * max(0, q_len)
            exec_time = base_exec_time * load_factor

            energy_cost = task.cpu_cycles * self.compute_energy_j_per_cycle
            energy_cost *= (1.0 + self.energy_variability_coeff * max(0, q_len))
            if energy_cost > self.energy_joules:
                task.mark_dropped()
                if self.on_task_dropped is not None:
                    try:
                        self.on_task_dropped(task)
                    except Exception:
                        pass
                continue
            finish_time = self.env.now + exec_time
            if finish_time > task.deadline:
                task.mark_dropped()
                if self.on_task_dropped is not None:
                    try:
                        self.on_task_dropped(task)
                    except Exception:
                        pass
                continue
            task.mark_started(self.env.now)
            if self.on_task_started is not None:
                try:
                    self.on_task_started(task)
                except Exception:
                    pass
            # queue wait captured via started_at - queued_at in metrics
            if self.service_time_jitter_coeff > 0.0:
                jit = 1.0 + random.uniform(-self.service_time_jitter_coeff, self.service_time_jitter_coeff)
                exec_time *= max(0.1, jit)
            yield self.env.timeout(exec_time)
            self.busy_time_s_acc += exec_time
            # Failure chance
            if random.random() < self.failure_prob:
                task.mark_dropped()
                if self.on_task_dropped is not None:
                    try:
                        self.on_task_dropped(task)
                    except Exception:
                        pass
            else:
                task.mark_completed(self.env.now)
            self.energy_joules -= energy_cost
            self.consumed_energy_j_acc += energy_cost
            if self.on_task_completed is not None:
                if task.status == TaskStatus.COMPLETED:
                    try:
                        self.on_task_completed(task, energy_cost)
                        self.completed_count_acc += 1
                    except Exception:
                        pass

    # Window counters similar to Edge
    busy_time_s_acc: float = 0.0
    completed_count_acc: int = 0
    consumed_energy_j_acc: float = 0.0

    def pop_window_counters(self) -> Dict[str, float]:
        stats = {
            "busy_time_s": float(self.busy_time_s_acc),
            "completed": int(self.completed_count_acc),
            "energy_j": float(self.consumed_energy_j_acc),
            "queue_len": int(len(self.in_queue.items)) if (self.in_queue is not None and hasattr(self.in_queue, 'items')) else 0,
        }
        self.busy_time_s_acc = 0.0
        self.completed_count_acc = 0
        self.consumed_energy_j_acc = 0.0
        return stats



