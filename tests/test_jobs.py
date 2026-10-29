"""Background jobs: progress, Cancel, failures, same-key jobs, and the window staying responsive."""

import threading
import time

import flowkit as fk
import pytest
from PySide6 import QtCore, QtGui, QtWidgets

pytest.importorskip("pytestqt")

from sidescatter.app import jobs  # noqa: E402
from sidescatter.app.gating import service as gating  # noqa: E402
from sidescatter.app.jobs import CANCELLED, DONE, FAILED, JobCancelled, run_job, wait_idle  # noqa: E402
from sidescatter.app.main_window import DEMO_DIR, MainWindow  # noqa: E402
from sidescatter.app.task_area import TaskArea  # noqa: E402
from sidescatter.core.workspace import ROOT, Workspace  # noqa: E402

SID = "PBMC_full_panel.fcs"


def _until(event, timeout=10):
    """Wait on a worker-side event without blocking the UI thread."""
    deadline = time.monotonic() + timeout
    while not event.is_set() and time.monotonic() < deadline:
        QtWidgets.QApplication.processEvents()
        time.sleep(0.005)
    assert event.is_set()


def test_result_goes_to_on_done_on_the_ui_thread(qtbot):
    seen = {}

    def work(job):
        seen["worker"] = threading.current_thread() is not threading.main_thread()
        job.progress(0.5, "halfway")
        return 42

    job = run_job("Answer", work, on_done=lambda r: seen.update(
        result=r, ui=threading.current_thread() is threading.main_thread()))
    assert job.wait() == 42
    assert seen == {"worker": True, "result": 42, "ui": True}
    assert job.state == DONE and job.fraction == 1.0


def test_the_event_loop_keeps_running_while_a_job_works(qtbot):
    release = threading.Event()
    ticks = []
    timer = QtCore.QTimer(interval=10)
    timer.timeout.connect(lambda: ticks.append(1))
    timer.start()
    job = run_job("Slow", lambda job: release.wait(5))
    qtbot.wait(200)
    assert job.active and len(ticks) > 5  # the window would have frozen before
    release.set()
    job.wait()
    timer.stop()


def test_cancel_stops_at_the_next_checkpoint_and_drops_the_result(qtbot):
    started, done = threading.Event(), []

    def work(job):
        started.set()
        while True:
            job.check_cancelled()
            time.sleep(0.01)

    job = run_job("Endless", work, on_done=done.append)
    _until(started)
    job.cancel()
    with pytest.raises(JobCancelled):
        job.wait()
    assert job.state == CANCELLED and done == []


def test_failure_is_kept_with_details(qtbot):
    errors = []

    def work(job):
        raise ValueError("bad keyword in file")

    job = run_job("Reading", work, on_error=errors.append)
    with pytest.raises(ValueError):
        job.wait()
    assert job.state == FAILED and "ad keyword in file" in job.text and "Traceback" in job.details
    assert errors and job in jobs.manager().jobs  # stays listed until looked at
    jobs.manager().dismiss(job)


def test_same_key_replaces_and_never_overlaps(qtbot):
    running, overlap, first_in = [0], [], threading.Event()
    lock = threading.Lock()

    def work(job, n):
        with lock:
            running[0] += 1
            overlap.append(running[0])
        first_in.set()
        try:
            for _ in range(30):
                job.check_cancelled()
                time.sleep(0.01)
            return n
        finally:
            with lock:
                running[0] -= 1

    first = run_job("Count", lambda job: work(job, 1), key=("count", 1))
    _until(first_in)
    second = run_job("Count", lambda job: work(job, 2), key=("count", 1))
    assert second.wait() == 2
    assert first.state == CANCELLED
    assert max(overlap) == 1


def test_task_area_shows_long_jobs_and_failures(qtbot):
    for old in [j for j in jobs.manager().jobs if not j.active]:
        jobs.manager().dismiss(old)
    area = TaskArea()
    qtbot.addWidget(area)
    release = threading.Event()
    job = run_job("Unmixing 3 files", lambda job: (job.progress(0.3, "File 1 of 3"), release.wait(5)))
    assert area.isHidden()  # quick jobs never show
    qtbot.waitUntil(lambda: area.isVisible(), timeout=3000)
    assert "Unmixing 3 files" in area.label.text()
    release.set()
    job.wait()
    bad = run_job("Export", lambda job: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        bad.wait()
    area.update_state()
    assert "Export failed" in area.label.text()
    jobs.manager().dismiss(bad)


@pytest.fixture
def window(qtbot):
    w = MainWindow()
    w.ask_before_discard = False
    qtbot.addWidget(w)
    w.add_samples([str(DEMO_DIR / SID)])
    wait_idle()
    return w


def _big_gate(ws, lo=0.3):
    return fk.gates.RectangleGate("Big", [ws.make_dimension(SID, "FSC-A", lo, None),
                                          ws.make_dimension(SID, "SSC-A", None, None)])


def test_counts_read_dots_until_background_gating_is_done(window, qtbot):
    ws = window.ws
    ws.add_gate(_big_gate(ws))
    window._changed()
    item = window.panel.tree.topLevelItem(0).child(0)
    assert item.text(2) == "…"
    wait_idle()
    qtbot.waitUntil(lambda: window.panel.tree.topLevelItem(0).child(0).text(2) != "…", timeout=3000)
    assert window.panel.tree.topLevelItem(0).child(0).text(2) == f"{ws.count(SID, 'Big'):,}"


def test_stale_results_are_dropped(window, qtbot):
    ws = window.ws
    ws.add_gate(_big_gate(ws))
    token, strategy, sample = ws.gating_snapshot(SID)
    old = ws.compute_results(strategy, sample)
    ws.get_gate("Big", ROOT).dimensions[0].min = 0.6
    ws.invalidate()
    assert not ws.store_results(SID, token, old)  # the gate moved while it was computed
    assert not ws.has_results(SID)


def test_graph_fades_then_redraws_when_results_arrive(window, qtbot):
    ws = window.ws
    v = window.open_graph(SID)
    wait_idle()
    v.refresh()
    assert v._data_items and v.count_label.text().endswith("events") and "…" not in v.count_label.text()
    ws.add_gate(_big_gate(ws))
    window._changed()
    v.refresh()
    assert v._waiting and v.count_label.text() == "… events"
    assert v._data_items  # the old picture stays up meanwhile
    wait_idle()
    qtbot.waitUntil(lambda: not v._waiting, timeout=3000)
    assert v.count_label.text() == f"{ws.samples[SID].event_count:,} events"
    assert not v.busy.isVisible()


def test_editing_again_cancels_the_stale_gating_job(window, qtbot):
    ws = window.ws
    ws.add_gate(_big_gate(ws))
    window._changed()
    gating().ensure(ws, [SID])
    pending = [j for j in jobs.manager().active() if j.key and j.key[0] == "gating"]
    ws.get_gate("Big", ROOT).dimensions[0].min = 0.6
    ws.invalidate()
    window._changed()  # cancels jobs for the old gates
    assert all(j.cancel_requested or not j.active for j in pending)
    wait_idle()
    gating().ensure(ws, [SID])
    wait_idle()
    assert ws.has_results(SID)


def test_open_and_save_run_in_the_background(window, qtbot, tmp_path):
    ws = window.ws
    ws.add_gate(_big_gate(ws))
    path = tmp_path / "w.ssw"
    window.path = str(path)
    window.dirty = True
    window.save().wait()
    assert path.exists() and not window.dirty
    window.load_workspace(str(path)).wait()
    assert window.ws is not ws and window.path == str(path)
    assert window.ws.count(SID, "Big") == ws.count(SID, "Big")


def test_load_reports_progress(tmp_path):
    ws = Workspace()
    ws.add_fcs([DEMO_DIR / SID])
    ws.save(tmp_path / "w.ssw")
    seen = []
    Workspace.load(tmp_path / "w.ssw", progress=lambda f, t: seen.append((f, t)))
    assert seen == [(0.0, f"Loading {SID}")]


def test_cancelled_export_leaves_nothing_behind(window, qtbot, tmp_path):
    tab = window.unmix_tab
    files = sorted(str(p) for p in (DEMO_DIR / "controls").glob("*.fcs")) + [str(DEMO_DIR / SID)]
    tab.add_files(files)
    wait_idle()
    job = tab.export(tmp_path, "Demo", True)
    assert not tab.export_btn.isEnabled()  # no second export of the same experiment meanwhile
    job.cancel()
    wait_idle()
    assert job.state == CANCELLED or job.state == DONE
    if job.state == CANCELLED:
        assert list(tmp_path.iterdir()) == []
    assert tab.export_btn.isEnabled()


def test_quitting_with_jobs_running_asks_first(window, qtbot, monkeypatch):
    release = threading.Event()
    job = run_job("Unmixing 8 files", lambda job: release.wait(5) and job.check_cancelled())
    window.ask_before_discard = True
    window.dirty = False
    asked = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "exec", lambda self: asked.append(self.text()) or 0)
    monkeypatch.setattr(QtWidgets.QMessageBox, "clickedButton", lambda self: None)  # "Keep working"
    event = QtGui.QCloseEvent()
    window.closeEvent(event)
    assert not event.isAccepted() and "1 job is still running" in asked[0]
    release.set()
    job.wait()
    window.ask_before_discard = False
