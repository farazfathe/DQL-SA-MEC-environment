from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class EventRecord:
    task_id: str
    time_s: float
    event: str
    node: Optional[str] = None
    details: Dict[str, float] = field(default_factory=dict)


@dataclass
class TaskRecord:
    task_id: str
    source_cell_id: Optional[str]
    created_at: float
    started_at: Optional[float]
    finished_at: Optional[float]
    queued_at: Optional[float]
    deadline: float
    completed_on_time: bool
    was_offloaded: bool
    was_to_cloud: bool
    latency_s: Optional[float]
    wait_s: Optional[float]
    energy_j: float


@dataclass
class StepRecord:
    time_s: float
    completed_tasks: int
    total_energy_j: float
    offloaded_ratio: float
    cloud_offload_ratio: float
    throughput_tasks_per_s: float
    avg_latency_s: Optional[float]
    avg_wait_s: Optional[float]
    # New per-window metrics
    sched_time_s: float = 0.0
    sched_time_per_decision_s: float = 0.0
    decisions: int = 0
    arrivals: int = 0
    acceptance_ratio: Optional[float] = None
    failure_ratio: Optional[float] = None
    busy_time_total_s: float = 0.0
    load_balance_cv: Optional[float] = None
    server_utilization: Dict[str, float] = field(default_factory=dict)
    queue_lengths: Dict[str, int] = field(default_factory=dict)
    node_completed: Dict[str, int] = field(default_factory=dict)
    node_energy_j: Dict[str, float] = field(default_factory=dict)


@dataclass
class RLRecord:
    iteration: int
    temperature: Optional[float] = None
    objective: Optional[float] = None
    accepted_worse: Optional[int] = None
    total_moves: Optional[int] = None
    epsilon: Optional[float] = None
    reward: Optional[float] = None


@dataclass
class MetricsLogger:
    tasks: Dict[str, TaskRecord] = field(default_factory=dict)
    steps: List[StepRecord] = field(default_factory=list)
    rl: List[RLRecord] = field(default_factory=list)
    events: List[EventRecord] = field(default_factory=list)
    attempts: Dict[str, int] = field(default_factory=dict)

    total_energy_j: float = 0.0
    total_completed_tasks: int = 0
    total_local_completed: int = 0
    total_offloaded: int = 0
    total_cloud_offloaded: int = 0
    # Aggregates for algorithm (scheduling) runtime
    total_sched_time_s: float = 0.0
    total_sched_decisions: int = 0

    def log_task(
        self,
        *,
        task_id: str,
        source_cell_id: Optional[str],
        created_at: float,
        started_at: Optional[float],
        finished_at: Optional[float],
        queued_at: Optional[float] = None,
        deadline: float,
        was_offloaded: bool,
        was_to_cloud: bool,
        energy_j: float,
    ) -> None:
        latency_s = None
        wait_s = None
        completed_on_time = False
        if finished_at is not None:
            latency_s = finished_at - created_at
            if started_at is not None:
                base_q = queued_at if queued_at is not None else created_at
                wait_s = max(0.0, started_at - base_q)
            completed_on_time = finished_at <= deadline
        self.tasks[task_id] = TaskRecord(
            task_id=task_id,
            source_cell_id=source_cell_id,
            created_at=created_at,
            started_at=started_at,
            finished_at=finished_at,
            queued_at=queued_at,
            deadline=deadline,
            completed_on_time=completed_on_time,
            was_offloaded=was_offloaded,
            was_to_cloud=was_to_cloud,
            latency_s=latency_s,
            wait_s=wait_s,
            energy_j=energy_j,
        )

        # Aggregate counts and energy
        if finished_at is not None:
            self.total_completed_tasks += 1
            if not was_offloaded:
                self.total_local_completed += 1
        if was_offloaded:
            self.total_offloaded += 1
        if was_to_cloud:
            self.total_cloud_offloaded += 1
        self.total_energy_j += max(0.0, energy_j)

    def mark_completed(self, task_id: str, finished_at: float, energy_j: float) -> None:
        if task_id in self.tasks:
            rec = self.tasks[task_id]
            rec.finished_at = finished_at
            rec.latency_s = finished_at - rec.created_at
            if rec.started_at is not None:
                base_q = rec.queued_at if rec.queued_at is not None else rec.created_at
                rec.wait_s = max(0.0, rec.started_at - base_q)
            rec.completed_on_time = finished_at <= rec.deadline
            self.total_completed_tasks += 1
            self.total_energy_j += max(0.0, energy_j)

    def log_event(self, task_id: str, time_s: float, event: str, node: Optional[str] = None, **details) -> None:
        self.events.append(EventRecord(task_id=task_id, time_s=time_s, event=event, node=node, details=details))
        # Count attempts only on processing starts (exclude tx_start)
        if event in {"local_start", "edge_start", "cloud_start"}:
            self.attempts[task_id] = self.attempts.get(task_id, 0) + 1
        # Update queued_at on enqueue events
        if event.endswith("_enqueue") and task_id in self.tasks:
            self.tasks[task_id].queued_at = time_s
        # Update started_at on start events if task already known
        if event.endswith("_start") and task_id in self.tasks:
            if self.tasks[task_id].started_at is None:
                self.tasks[task_id].started_at = time_s

    def log_step(
        self,
        *,
        time_s: float,
        completed_tasks: int,
        offloaded: int,
        cloud_offloaded: int,
        energy_j: float,
        latencies: Optional[List[float]] = None,
        waits: Optional[List[float]] = None,
        server_utilization: Optional[Dict[str, float]] = None,
        queue_lengths: Optional[Dict[str, int]] = None,
        node_completed: Optional[Dict[str, int]] = None,
        node_energy_j: Optional[Dict[str, float]] = None,
        window_s: float = 1.0,
        # New optional window stats
        sched_time_s: float = 0.0,
        decisions: int = 0,
        arrivals: int = 0,
        acceptance_ratio: Optional[float] = None,
        failure_ratio: Optional[float] = None,
        busy_time_total_s: float = 0.0,
        load_balance_cv: Optional[float] = None,
    ) -> None:
        avg_latency = (sum(latencies) / len(latencies)) if latencies else None
        avg_wait = (sum(waits) / len(waits)) if waits else None
        throughput = completed_tasks / window_s if window_s > 0 else 0.0
        total = completed_tasks + offloaded  # denominator for offload ratio
        off_ratio = (offloaded / total) if total > 0 else 0.0
        cloud_ratio = (cloud_offloaded / total) if total > 0 else 0.0
        # Track scheduling time aggregates
        self.total_sched_time_s += max(0.0, float(sched_time_s))
        self.total_sched_decisions += max(0, int(decisions))
        self.steps.append(
            StepRecord(
                time_s=time_s,
                completed_tasks=completed_tasks,
                total_energy_j=self.total_energy_j,
                offloaded_ratio=off_ratio,
                cloud_offload_ratio=cloud_ratio,
                throughput_tasks_per_s=throughput,
                avg_latency_s=avg_latency,
                avg_wait_s=avg_wait,
                sched_time_s=float(sched_time_s),
                sched_time_per_decision_s=(float(sched_time_s) / float(decisions)) if decisions > 0 else 0.0,
                decisions=int(decisions),
                arrivals=int(arrivals),
                acceptance_ratio=acceptance_ratio,
                failure_ratio=failure_ratio,
                busy_time_total_s=float(busy_time_total_s),
                load_balance_cv=load_balance_cv,
                server_utilization=server_utilization or {},
                queue_lengths=queue_lengths or {},
                node_completed=node_completed or {},
                node_energy_j=node_energy_j or {},
            )
        )

    def log_rl(
        self,
        *,
        iteration: int,
        temperature: float | None = None,
        objective: float | None = None,
        accepted_worse: int | None = None,
        total_moves: int | None = None,
        epsilon: float | None = None,
        reward: float | None = None,
    ) -> None:
        self.rl.append(
            RLRecord(
                iteration=iteration,
                temperature=temperature,
                objective=objective,
                accepted_worse=accepted_worse,
                total_moves=total_moves,
                epsilon=epsilon,
                reward=reward,
            )
        )

    # --- Aggregates ---
    def summary(self) -> Dict[str, float]:
        total_tasks = len(self.tasks)
        completed_on_time = sum(1 for t in self.tasks.values() if t.completed_on_time)
        offloaded = sum(1 for t in self.tasks.values() if t.was_offloaded)
        to_cloud = sum(1 for t in self.tasks.values() if t.was_to_cloud)
        avg_latency = (
            sum(t.latency_s for t in self.tasks.values() if t.latency_s is not None)
            / max(1, sum(1 for t in self.tasks.values() if t.latency_s is not None))
        )
        avg_wait = (
            sum(t.wait_s for t in self.tasks.values() if t.wait_s is not None)
            / max(1, sum(1 for t in self.tasks.values() if t.wait_s is not None))
        )
        attempts_mean = (
            sum(self.attempts.get(tid, 0) for tid in self.tasks.keys()) / float(max(1, total_tasks))
        )
        # Makespan: from first task created to last finished
        first_created = min((t.created_at for t in self.tasks.values()), default=None)
        last_finished = max((t.finished_at for t in self.tasks.values() if t.finished_at is not None), default=None)
        makespan = (last_finished - first_created) if (first_created is not None and last_finished is not None) else 0.0
        # Scheduling time aggregates
        mean_sched_per_decision = (self.total_sched_time_s / float(self.total_sched_decisions)) if self.total_sched_decisions > 0 else 0.0
        # Algorithm objective stats if present
        obj_vals = [r.objective for r in self.rl if r.objective is not None]
        obj_mean = (sum(obj_vals) / float(len(obj_vals))) if obj_vals else None
        obj_final = obj_vals[-1] if obj_vals else None
        # Acceptance/failure
        success_rate = (completed_on_time / float(total_tasks)) if total_tasks else 0.0
        failure_ratio = 1.0 - success_rate if total_tasks else 0.0
        # Total processing (service) time across all nodes over run
        total_processing_time_s = sum((s.busy_time_total_s for s in self.steps), start=0.0)
        return {
            "total_tasks": float(total_tasks),
            "success_rate": success_rate,
            "failure_ratio": failure_ratio,
            "local_completed": float(self.total_local_completed),
            "offloading_ratio": (offloaded / float(total_tasks)) if total_tasks else 0.0,
            "cloud_offloading_ratio": (to_cloud / float(total_tasks)) if total_tasks else 0.0,
            "avg_latency_s": avg_latency,
            "avg_wait_s": avg_wait,
            "total_energy_j": self.total_energy_j,
            "energy_efficiency": (self.total_completed_tasks / self.total_energy_j) if self.total_energy_j > 0 else 0.0,
            "attempts_mean": attempts_mean,
            "makespan_s": makespan,
            "sched_time_total_s": self.total_sched_time_s,
            "sched_time_per_decision_s": mean_sched_per_decision,
            "objective_mean": obj_mean if obj_mean is not None else 0.0,
            "objective_final": obj_final if obj_final is not None else 0.0,
            "total_processing_time_s": float(total_processing_time_s),
        }


