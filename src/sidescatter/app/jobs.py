"""Background jobs: heavy work runs on worker threads, so the window never freezes.

    job = run_job("Unmixing 8 files", work, on_done=add_to_workspace)

work(job) runs on a worker thread. It reports with job.progress(0.4, "File 3 of 8")
and calls job.check_cancelled() at checkpoints; both raise JobCancelled once Cancel
was pressed. Its return value goes to on_done(result), which runs on the UI
thread: that is the only place to change the workspace or any widget. So
cancelling, or failing, leaves everything as it was.

key: jobs with the same key never run at the same time. A new job cancels the
running one with its key and starts once that one has stopped, e.g. when a gate
is edited again before the counts from the last edit are in.

Progress shows in the status bar's task area (task_area.py) once a job has run
for SHOW_AFTER seconds, so quick jobs don't flicker.
"""

import os
import time
import threading
import traceback

from PySide6 import QtCore, QtWidgets

WAITING, RUNNING, DONE, FAILED, CANCELLED = "waiting", "running", "done", "failed", "cancelled"
ENDED = (DONE, FAILED, CANCELLED)
SHOW_AFTER = 0.5   # seconds before a job shows up in the task area
KEEP_RECENT = 10   # finished jobs the task list keeps showing


class JobCancelled(Exception):
    """Raised inside a job's work at the next checkpoint after Cancel."""


class Job(QtCore.QObject):
    changed = QtCore.Signal()   # state, progress or text (UI thread)
    finished = QtCore.Signal()  # done, failed or cancelled (UI thread)
    _poke = QtCore.Signal()                        # from the worker: progress moved
    _ended = QtCore.Signal(str, object, str)       # from the worker: state, result/error, details

    def __init__(self, title, fn, on_done=None, on_error=None, on_cancel=None,
                 cancellable=True, key=None, quiet=False):
        super().__init__()
        self.title = title
        self.fn = fn
        self.on_done, self.on_error, self.on_cancel = on_done, on_error, on_cancel
        self.cancellable = cancellable
        self.key = key
        self.quiet = quiet          # forgotten once done (failures are kept)
        self.state = WAITING
        self.fraction = None        # 0..1, or None while it can't tell
        self.text = ""
        self.result = None
        self.error = None
        self.details = ""           # traceback of a failure
        self.created = time.monotonic()
        self.started = None
        self.seen = False           # a failure the user has looked at
        self._cancel = threading.Event()
        self._last_poke = 0.0
        self._poke.connect(self.changed)
        self._ended.connect(self._end)

    # ----- called from the work function (worker thread) -------------------

    def progress(self, fraction=None, text=None):
        """Report progress (fraction 0..1, a short line of text). Also a checkpoint."""
        self.check_cancelled()
        if fraction is not None:
            self.fraction = min(max(float(fraction), 0.0), 1.0)
        if text is not None:
            self.text = str(text)
        now = time.monotonic()
        if now - self._last_poke > 0.05:  # don't flood the UI thread
            self._last_poke = now
            self._poke.emit()

    def check_cancelled(self):
        if self._cancel.is_set():
            raise JobCancelled()

    @property
    def cancel_requested(self):
        return self._cancel.is_set()

    # ----- UI thread ---------------------------------------------------------

    @property
    def active(self):
        return self.state not in ENDED

    def elapsed(self):
        return time.monotonic() - (self.started or self.created)

    def cancel(self):
        if not self.active or self._cancel.is_set():
            return
        self._cancel.set()
        if self.state == WAITING:
            manager()._unqueue(self)
            self._end(CANCELLED, None, "")
        else:
            self.text = "Cancelling…"
            self.changed.emit()

    def wait(self, timeout=120.0):
        """Run the event loop until the job ends; returns its result. Raises the
        job's error if it failed, or JobCancelled. Meant for tests and scripts."""
        if self.active:
            loop = QtCore.QEventLoop()
            self.finished.connect(loop.quit)
            QtCore.QTimer.singleShot(int(timeout * 1000), loop.quit)
            loop.exec()
            if self.active:
                raise TimeoutError(f"{self.title} is still running after {timeout} s")
        if self.state == FAILED:
            raise self.error
        if self.state == CANCELLED:
            raise JobCancelled()
        return self.result

    def _run(self):
        """Worker thread."""
        if self._cancel.is_set():
            return  # cancelled while waiting; already ended
        self.started = time.monotonic()
        self.state = RUNNING
        self._poke.emit()
        try:
            result = self.fn(self)
            self.check_cancelled()  # Cancel was pressed: drop the result
        except JobCancelled:
            self._ended.emit(CANCELLED, None, "")
        except BaseException as exc:  # noqa: B036 - reported to the user, not swallowed
            self._ended.emit(FAILED, exc, traceback.format_exc())
        else:
            self._ended.emit(DONE, result, "")

    def _end(self, state, value, details):
        if not self.active:
            return
        self.state = state
        if state == DONE:
            self.result, self.fraction = value, 1.0
            if self.on_done is not None:
                try:
                    self.on_done(value)
                except Exception as exc:
                    state, value, details = FAILED, exc, traceback.format_exc()
                    self.state = FAILED
        if state == FAILED:
            self.error, self.details = value, details
            self.text = _plain(value)
        if state == FAILED and self.on_error is not None:
            self.on_error(value)
        if state == CANCELLED:
            self.text = "Cancelled"
            if self.on_cancel is not None:
                self.on_cancel()
        self.changed.emit()
        self.finished.emit()


class _Runner(QtCore.QRunnable):
    def __init__(self, job):
        super().__init__()
        self.job = job

    def run(self):
        self.job._run()


class JobManager(QtCore.QObject):
    changed = QtCore.Signal()   # a job started, moved, ended or was dismissed

    def __init__(self, threads=None):
        super().__init__()
        self.pool = QtCore.QThreadPool()
        self.pool.setMaxThreadCount(threads or max(1, (os.cpu_count() or 2) - 1))
        self.jobs = []        # active jobs, then recent finished ones
        self._runners = {}    # job -> its runnable while queued in the pool
        self._held = []       # jobs waiting for a running job with the same key

    def run(self, title, fn, on_done=None, on_error=None, on_cancel=None, cancellable=True,
            key=None, quiet=False):
        job = Job(title, fn, on_done, on_error, on_cancel, cancellable, key, quiet)
        busy = False
        if key is not None:
            for other in self.active(key=key):
                other.cancel()
                busy = busy or other.state == RUNNING
        self.jobs.append(job)
        job.changed.connect(self.changed)
        job.finished.connect(lambda: self._finished(job))
        if busy:
            job.text = "Waiting"
            self._held.append(job)
        else:
            self._start(job)
        self.changed.emit()
        return job

    def active(self, key=None, prefix=None):
        """Jobs that haven't ended, optionally only those with this key, or whose
        key starts with this tuple."""
        out = [j for j in self.jobs if j.active]
        if key is not None:
            out = [j for j in out if j.key == key]
        if prefix is not None:
            out = [j for j in out if isinstance(j.key, tuple) and j.key[:len(prefix)] == tuple(prefix)]
        return out

    def busy(self, key=None, prefix=None):
        return bool(self.active(key, prefix))

    def visible(self):
        """What the task area shows: jobs running long enough to mention, then
        failures not yet looked at, then recent finished ones."""
        now = time.monotonic()
        return [j for j in self.jobs
                if (j.active and now - j.created >= SHOW_AFTER) or not j.active]

    def cancel_all(self):
        for job in self.active():
            job.cancel()

    def wait_idle(self, timeout=120.0):
        """Run the event loop until no job is active (tests, quitting)."""
        deadline = time.monotonic() + timeout
        while self.active():
            if time.monotonic() > deadline:
                raise TimeoutError("background jobs still running: "
                                   + ", ".join(j.title for j in self.active()))
            QtWidgets.QApplication.processEvents(QtCore.QEventLoop.AllEvents, 50)
            time.sleep(0.005)

    def dismiss(self, job):
        if not job.active and job in self.jobs:
            self.jobs.remove(job)
            self.changed.emit()

    def _start(self, job):
        runner = _Runner(job)
        runner.setAutoDelete(False)
        self._runners[job] = runner
        self.pool.start(runner)

    def _unqueue(self, job):
        runner = self._runners.get(job)
        if runner is not None:
            self.pool.tryTake(runner)
        if job in self._held:
            self._held.remove(job)

    def _finished(self, job):
        self._runners.pop(job, None)
        if job.key is not None and not any(j.state == RUNNING and j.key == job.key for j in self.jobs):
            for held in self._held:
                if held.key == job.key:
                    self._held.remove(held)
                    held.text = ""
                    self._start(held)
                    break
        if job.quiet and job.state != FAILED:
            self.jobs.remove(job)
        ended = [j for j in self.jobs if not j.active]
        for old in ended[:-KEEP_RECENT]:
            self.jobs.remove(old)
        self.changed.emit()


_manager = None


def manager():
    global _manager
    if _manager is None:
        _manager = JobManager()
    return _manager


def run_job(title, fn, on_done=None, on_error=None, on_cancel=None, cancellable=True, key=None,
            quiet=False):
    """Run fn(job) on a worker thread; see the module docstring. Returns the Job."""
    return manager().run(title, fn, on_done, on_error, on_cancel, cancellable, key, quiet)


def wait_idle(timeout=120.0):
    manager().wait_idle(timeout)


def _plain(exc):
    """A failure in plain words for the task list."""
    if isinstance(exc, FileNotFoundError):
        return f"A file is missing: {exc}"
    if isinstance(exc, MemoryError):
        return "Ran out of memory"
    text = str(exc).strip() or type(exc).__name__
    return text[0].upper() + text[1:] if text else text
