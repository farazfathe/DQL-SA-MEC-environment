from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json
import time


@dataclass
class TaskLifecycleEvent:
    ts: float
    event: str
    detail: Dict[str, object] = field(default_factory=dict)


@dataclass
class TaskLifecycleLogger:
    """Per-task lifecycle logger: spawn, attempts, complete/fail.

    Lightweight helper to track task attempts and status changes.
    """

    events: Dict[str, List[TaskLifecycleEvent]] = field(default_factory=dict)
    attempts: Dict[str, int] = field(default_factory=dict)

    def _now(self) -> float:
        return time.time()

    def _log(self, task_id: str, event: str, **detail) -> None:
        self.events.setdefault(task_id, []).append(
            TaskLifecycleEvent(ts=self._now(), event=event, detail=dict(detail))
        )

    def spawn(self, task_id: str) -> None:
        self.attempts[task_id] = 0
        self._log(task_id, "spawn")

    def attempt(self, task_id: str) -> int:
        n = int(self.attempts.get(task_id, 0)) + 1
        self.attempts[task_id] = n
        self._log(task_id, "attempt", attempt=n)
        return n

    def complete(self, task_id: str) -> None:
        self._log(task_id, "complete")

    def fail(self, task_id: str, reason: Optional[str] = None) -> None:
        self._log(task_id, "fail", reason=(reason or ""))

    def to_json(self) -> str:
        def _conv(ev: TaskLifecycleEvent):
            return {"ts": ev.ts, "event": ev.event, **(ev.detail or {})}
        obj = {tid: [_conv(ev) for ev in evs] for tid, evs in self.events.items()}
        return json.dumps(obj, indent=2)

