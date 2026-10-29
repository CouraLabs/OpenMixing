from __future__ import annotations

import json
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..core.cancel import JobCancelled
from ..core.errors import classify

STATES = ("queued", "running", "done", "failed", "cancelled", "lost")
MAX_CONCURRENT = 1
KINDS = ("render",)


@dataclass
class Job:
    id: str
    kind: str
    key: str
    state: str = "queued"
    progress: dict = field(default_factory=lambda: {"i": 0, "n": 0, "msg": ""})
    started: float | None = None
    finished: float | None = None
    result: dict | None = None
    error: dict | None = None
    stop: threading.Event = field(default_factory=threading.Event, repr=False)
    future: Future | None = field(default=None, repr=False)

    def cancelled(self) -> bool:
        return self.stop.is_set()

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k not in ("stop", "future")}


class JobManager:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.lock = threading.RLock()
        self.pool = ThreadPoolExecutor(MAX_CONCURRENT)
        self.jobs: dict[str, Job] = {}
        for line in (self.path.read_text().splitlines() if self.path.exists() else []):
            try:
                d = json.loads(line)
                self.jobs[d["id"]] = Job(**d)
            except (ValueError, KeyError, TypeError):
                continue
        for j in self.jobs.values():
            if j.state in ("queued", "running"):
                j.state = "lost"
                self._save(j)

    def _save(self, j: Job) -> None:
        with self.lock, self.path.open("a") as f:
            f.write(json.dumps(j.to_dict()) + "\n")

    def get(self, jid: str) -> Job:
        if jid not in self.jobs:
            raise KeyError(f"no job {jid!r}; call list_jobs")
        return self.jobs[jid]

    def list(self, state: str | None = None) -> list[Job]:
        return [j for j in self.jobs.values() if state in (None, j.state)]

    def submit(self, kind: str, key: str, fn: Callable[[Job], dict], valid: Callable[[dict], bool] = lambda r: True) -> tuple[Job, bool]:
        with self.lock:
            for j in reversed(list(self.jobs.values())):
                if j.kind == kind and j.key == key and (j.state in ("queued", "running") or (j.state == "done" and valid(j.result))):
                    return j, True
            j = Job(uuid.uuid4().hex[:8], kind, key)
            self.jobs[j.id] = j
            self._save(j)
            j.future = self.pool.submit(self._run, j, fn)
            return j, False

    def _run(self, j: Job, fn: Callable[[Job], dict]) -> None:
        with self.lock:
            if j.state != "queued":
                return
            j.state, j.started = "running", time.time()
            self._save(j)
        try:
            result, state, error = fn(j), "done", None
        except JobCancelled:
            result, state, error = None, "cancelled", None
        except Exception as e:
            result, state, error = None, "failed", classify(e).payload()["error"]
        with self.lock:
            j.result, j.state, j.error, j.finished = result, state, error, time.time()
            self._save(j)

    def cancel(self, jid: str) -> Job:
        with self.lock:
            j = self.get(jid)
            if j.state == "queued":
                j.state, j.finished = "cancelled", time.time()
                self._save(j)
            elif j.state == "running":
                j.stop.set()
            return j

    def wait(self, jid: str, timeout: float | None = None) -> Job:
        j = self.get(jid)
        if j.future is not None:
            j.future.result(timeout)
        return j

    def report(self, j: Job, i: int, n: int, msg: str) -> None:
        j.progress = {"i": i, "n": n, "msg": msg}
