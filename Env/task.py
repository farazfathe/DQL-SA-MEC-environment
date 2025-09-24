from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TaskStatus(Enum):
    CREATED = "created"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    DROPPED = "dropped"  # e.g., missed deadline or energy constraints


@dataclass
class Task:
    """Represents a computation task in the MEC environment.

    Attributes
    ----------
    task_id: str
        Unique identifier for the task.
    bit_size: int
        Size of the task payload in bits (e.g., input data to transmit/offload).
    memory_required: int
        Memory required to execute the task, in bytes.
    cpu_cycles: int
        Total CPU cycles required to complete the task.
    deadline: float
        Absolute deadline timestamp (same time base as env clock) by which the task should be completed.
    created_at: float
        Absolute creation timestamp.
    source_cell_id: Optional[str]
        Identifier of the originating cell.
    status: TaskStatus
        Current lifecycle state of the task.
    started_at: Optional[float]
        Timestamp when execution started.
    finished_at: Optional[float]
        Timestamp when execution finished.
    """

    task_id: str
    bit_size: int
    memory_required: int
    cpu_cycles: int
    deadline: float
    created_at: float
    source_cell_id: Optional[str] = None
    status: TaskStatus = field(default=TaskStatus.CREATED)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None

    def estimated_compute_time(self, cpu_rate_cycles_per_s: float) -> float:
        """Return estimated compute time in seconds for a given CPU rate.

        Parameters
        ----------
        cpu_rate_cycles_per_s: float
            Sustained CPU throughput in cycles per second.
        """
        if cpu_rate_cycles_per_s <= 0:
            raise ValueError("cpu_rate_cycles_per_s must be positive")
        return self.cpu_cycles / float(cpu_rate_cycles_per_s)

    def is_deadline_missed(self, now: float) -> bool:
        return now > self.deadline

    def mark_started(self, now: float) -> None:
        self.status = TaskStatus.RUNNING
        self.started_at = now

    def mark_completed(self, now: float) -> None:
        self.status = TaskStatus.COMPLETED
        self.finished_at = now

    def mark_dropped(self) -> None:
        self.status = TaskStatus.DROPPED


