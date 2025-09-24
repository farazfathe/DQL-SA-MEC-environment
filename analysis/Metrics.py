from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class TaskRecord:
    task_id: str
    source_cell_id: Optional[str]
    created_at: float
    started_at: Optional[float]
    finished_at: Optional[float]
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
    server_utilization: Dict[str, float] = field(default_factory=dict)
    queue_lengths: Dict[str, int] = field(default_factory=dict)


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

    total_energy_j: float = 0.0
    total_completed_tasks: int = 0
    total_offloaded: int = 0
    total_cloud_offloaded: int = 0

    def log_task(
        self,
        *,
        task_id: str,
        source_cell_id: Optional[str],
        created_at: float,
        started_at: Optional[float],
        finished_at: Optional[float],
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
                wait_s = started_at - created_at
            completed_on_time = finished_at <= deadline
        self.tasks[task_id] = TaskRecord(
            task_id=task_id,
            source_cell_id=source_cell_id,
            created_at=created_at,
            started_at=started_at,
            finished_at=finished_at,
            deadline=deadline,
            completed_on_time=completed_on_time,
            was_offloaded=was_offloaded,
            was_to_cloud=was_to_cloud,
            latency_s=latency_s,
            wait_s=wait_s,
            energy_j=energy_j,
        )

        if finished_at is not None:
            self.total_completed_tasks += 1
        if was_offloaded:
            self.total_offloaded += 1
        if was_to_cloud:
            self.total_cloud_offloaded += 1
        self.total_energy_j += energy_j

    def mark_completed(self, task_id: str, finished_at: float, energy_j: float) -> None:
        if task_id in self.tasks:
            rec = self.tasks[task_id]
            rec.finished_at = finished_at
            rec.latency_s = finished_at - rec.created_at
            if rec.started_at is not None:
                rec.wait_s = rec.started_at - rec.created_at
            rec.completed_on_time = finished_at <= rec.deadline
            self.total_completed_tasks += 1
            self.total_energy_j += max(0.0, energy_j)

    def log_step(
        self,
        *,
        time_s: float,
        completed_tasks: int,
        offloaded: int,
        cloud_offloaded: int,
        energy_j: float,
        latencies: List[float] | None = None,
        waits: List[float] | None = None,
        server_utilization: Dict[str, float] | None = None,
        queue_lengths: Dict[str, int] | None = None,
        window_s: float = 1.0,
    ) -> None:
        avg_latency = (sum(latencies) / len(latencies)) if latencies else None
        avg_wait = (sum(waits) / len(waits)) if waits else None
        throughput = completed_tasks / window_s if window_s > 0 else 0.0
        total = completed_tasks + offloaded  # denominator for offload ratio
        off_ratio = (offloaded / total) if total > 0 else 0.0
        cloud_ratio = (cloud_offloaded / total) if total > 0 else 0.0
        self.total_energy_j += energy_j
        self.total_completed_tasks += completed_tasks
        self.total_offloaded += offloaded
        self.total_cloud_offloaded += cloud_offloaded
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
                server_utilization=server_utilization or {},
                queue_lengths=queue_lengths or {},
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
        return {
            "total_tasks": float(total_tasks),
            "success_rate": (completed_on_time / float(total_tasks)) if total_tasks else 0.0,
            "offloading_ratio": (offloaded / float(total_tasks)) if total_tasks else 0.0,
            "cloud_offloading_ratio": (to_cloud / float(total_tasks)) if total_tasks else 0.0,
            "avg_latency_s": avg_latency,
            "avg_wait_s": avg_wait,
            "total_energy_j": self.total_energy_j,
            "energy_efficiency": (self.total_completed_tasks / self.total_energy_j) if self.total_energy_j > 0 else 0.0,
        }


