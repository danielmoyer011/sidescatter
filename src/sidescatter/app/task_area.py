"""Where background jobs show up: a task area at the right of the status bar
(like FlowJo's), a list of jobs with Cancel buttons when it's clicked, failure
details with Copy and Report bug, and a small progress dialog for the few jobs
there's nothing to do but wait for (opening a workspace, FlowJo import).
"""

from PySide6 import QtCore, QtGui, QtWidgets

from . import theme
from .jobs import CANCELLED, DONE, FAILED, RUNNING, manager

FAILED_COLOR = "#D64545"


class Spinner(QtWidgets.QWidget):
    """A small turning arc that says "working on it"."""

    def __init__(self, size=16, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self._angle = 0
        self._timer = QtCore.QTimer(self, interval=40)
        self._timer.timeout.connect(self._turn)

    def _turn(self):
        self._angle = (self._angle + 24) % 360
        self.update()

    def showEvent(self, event):
        self._timer.start()
        super().showEvent(event)

    def hideEvent(self, event):
        self._timer.stop()
        super().hideEvent(event)

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        w = self.width()
        pen = QtGui.QPen(QtGui.QColor(theme.ACCENT), max(1.6, w / 8))
        pen.setCapStyle(QtCore.Qt.RoundCap)
        p.setPen(pen)
        r = QtCore.QRectF(2, 2, w - 4, w - 4)
        p.drawArc(r, -self._angle * 16, 270 * 16)
        p.end()


class TaskArea(QtWidgets.QToolButton):
    """Status-bar entry: the running job's name and progress, or a red failure."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("TaskArea")
        self.setAutoRaise(True)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.spinner = Spinner(14)
        self.label = QtWidgets.QLabel()
        self.bar = QtWidgets.QProgressBar()
        self.bar.setFixedSize(110, 8)
        self.bar.setTextVisible(False)
        self.bar.setRange(0, 1000)
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(8, 0, 8, 0)
        lay.setSpacing(6)
        lay.addWidget(self.spinner)
        lay.addWidget(self.label)
        lay.addWidget(self.bar)
        self.clicked.connect(self.show_list)
        self._popup = None
        self._timer = QtCore.QTimer(self, interval=150)  # jobs become visible after a delay
        self._timer.timeout.connect(self.update_state)
        manager().changed.connect(self.update_state)
        self.update_state()

    def sizeHint(self):
        return self.layout().sizeHint()

    def update_state(self):
        m = manager()
        if m.active():
            self._timer.start()
        else:
            self._timer.stop()
        running = [j for j in m.visible() if j.active]
        failed = [j for j in m.jobs if j.state == FAILED and not j.seen]
        if running:
            job = running[0]
            more = f"  +{len(running) - 1} more" if len(running) > 1 else ""
            pct = f"  {round(job.fraction * 100)}%" if job.fraction is not None and job.state == RUNNING else ""
            self.label.setText(f"{_short(job.title)}{pct}{more}")
            self.label.setStyleSheet("")
            self.spinner.show()
            self.bar.setVisible(job.fraction is not None)
            self.bar.setValue(int((job.fraction or 0) * 1000))
            self.setToolTip("Click to see running jobs and cancel them")
            self.show()
        elif failed:
            self.label.setText(f"⚠  {_short(failed[-1].title)} failed")
            self.label.setStyleSheet(f"color: {FAILED_COLOR}; font-weight: 600;")
            self.spinner.hide()
            self.bar.hide()
            self.setToolTip("Click to see what went wrong")
            self.show()
        else:
            self.hide()
        self.updateGeometry()
        if self._popup is not None and self._popup.isVisible():
            self._popup.rebuild()

    def show_list(self):
        failed = [j for j in manager().jobs if j.state == FAILED and not j.seen]
        running = [j for j in manager().visible() if j.active]
        if failed and not running:
            show_failure(failed[-1], self.window())
            return
        if self._popup is None:
            self._popup = TaskList(self.window())
        self._popup.rebuild()
        self._popup.adjustSize()
        pos = self.mapToGlobal(QtCore.QPoint(self.width(), 0))
        self._popup.move(pos - QtCore.QPoint(self._popup.width(), self._popup.height() + 4))
        self._popup.show()


class TaskList(QtWidgets.QFrame):
    """Every running and recent job, each with Cancel (or Details for a failure)."""

    def __init__(self, parent=None):
        super().__init__(parent, QtCore.Qt.Popup)
        self.setObjectName("TaskList")
        self.setMinimumWidth(380)
        self.setStyleSheet(f"#TaskList {{ background: {theme.SURFACE}; border: 1px solid "
                           f"{theme.BORDER_STRONG}; border-radius: 6px; }}")
        self._rows = QtWidgets.QVBoxLayout(self)
        self._rows.setContentsMargins(10, 8, 10, 8)
        self._rows.setSpacing(6)
        self._shown = None

    def rebuild(self):
        """Lay out a row per job; rows update themselves as their jobs move."""
        jobs = list(reversed(manager().visible()))
        key = [(id(j), j.active, j.state == FAILED) for j in jobs]
        if key == self._shown:
            return
        self._shown = key
        while self._rows.count():
            w = self._rows.takeAt(0).widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self._rows.addWidget(QtWidgets.QLabel("<b>Background jobs</b>"))
        if not jobs:
            self._rows.addWidget(QtWidgets.QLabel("Nothing running."))
        for job in jobs:
            self._rows.addWidget(JobRow(job, self))
        self.adjustSize()


class JobRow(QtWidgets.QWidget):
    def __init__(self, job, parent=None):
        super().__init__(parent)
        self.job = job
        title = QtWidgets.QLabel(_short(job.title, 52))
        title.setToolTip(job.title)
        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)
        self.bar = QtWidgets.QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(6)
        text = QtWidgets.QVBoxLayout()
        text.setSpacing(1)
        text.addWidget(title)
        text.addWidget(self.status)
        text.addWidget(self.bar)
        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(0, 2, 0, 2)
        row.addLayout(text, 1)
        self.button = None
        if job.active and job.cancellable:
            self.button = QtWidgets.QPushButton("Cancel")
            self.button.clicked.connect(job.cancel)
        elif job.state == FAILED:
            self.button = QtWidgets.QPushButton("Details…")
            self.button.clicked.connect(lambda: show_failure(job, self.window().parentWidget()))
        if self.button is not None:
            row.addWidget(self.button)
        job.changed.connect(self._refresh)
        self._refresh()

    def _refresh(self):
        job = self.job
        self.status.setText(_status(job))
        self.status.setStyleSheet(f"color: {FAILED_COLOR if job.state == FAILED else theme.MUTED};")
        self.bar.setVisible(job.active and job.fraction is not None)
        self.bar.setValue(int((job.fraction or 0) * 1000))
        if self.button is not None and job.active:
            self.button.setEnabled(not job.cancel_requested)


def show_failure(job, parent=None):
    """What went wrong, in plain words, with Copy details and Report bug."""
    job.seen = True
    manager().changed.emit()
    box = QtWidgets.QMessageBox(parent)
    box.setIcon(QtWidgets.QMessageBox.Warning)
    box.setWindowTitle(job.title)
    box.setText(f"<b>{job.title} failed.</b>")
    box.setInformativeText(job.text or "Something went wrong.")
    box.setDetailedText(job.details)
    copy = box.addButton("Copy details", QtWidgets.QMessageBox.ActionRole)
    report = box.addButton("Report bug…", QtWidgets.QMessageBox.ActionRole)
    box.addButton(QtWidgets.QMessageBox.Close)
    copy.clicked.disconnect()  # keep the box open after copying
    copy.clicked.connect(lambda: QtWidgets.QApplication.clipboard().setText(failure_text(job)))
    box.exec()
    if box.clickedButton() is report:
        from .suggestion_box import SuggestionDialog

        dlg = SuggestionDialog(parent)
        dlg.prefill_bug(f"{job.title} failed", f"{job.text}\n\n```\n{job.details.strip()}\n```")
        dlg.exec()
    manager().dismiss(job)


def failure_text(job):
    return f"{job.title} failed: {job.text}\n\n{job.details}".strip()


class JobDialog(QtWidgets.QDialog):
    """Progress and Cancel for a job the user has to wait for."""

    def __init__(self, job, parent=None):
        super().__init__(parent)
        self.job = job
        self.setWindowTitle(job.title)
        self.setWindowModality(QtCore.Qt.WindowModal)
        self.setMinimumWidth(420)
        self.label = QtWidgets.QLabel(job.title + "…")
        self.label.setWordWrap(True)
        self.bar = QtWidgets.QProgressBar()
        self.bar.setRange(0, 0)
        self.bar.setTextVisible(False)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Cancel)
        buttons.rejected.connect(self.reject)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.label)
        lay.addWidget(self.bar)
        lay.addWidget(buttons)
        job.changed.connect(self._update)
        job.finished.connect(self.accept)
        self._update()

    def _update(self):
        job = self.job
        self.label.setText(job.text or job.title + "…")
        if job.fraction is None:
            self.bar.setRange(0, 0)
        else:
            self.bar.setRange(0, 1000)
            self.bar.setValue(int(job.fraction * 1000))

    def reject(self):
        self.job.cancel()
        super().reject()


def run_with_dialog(job, parent=None, on_failure=None):
    """Show a JobDialog for this job until it ends. Returns right away; the
    dialog closes itself. A failure is shown straight away, since the user was
    waiting for it: by on_failure(job) if given and it returns True, otherwise
    with show_failure."""
    if job.active:
        dlg = JobDialog(job, parent)
        dlg.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        dlg.show()

    def failed():
        if on_failure is not None and on_failure(job):
            manager().dismiss(job)
        else:
            show_failure(job, parent)

    job.finished.connect(lambda: job.state == FAILED and QtCore.QTimer.singleShot(0, failed))
    return job


def _status(job):
    if job.state == RUNNING:
        pct = f"{round(job.fraction * 100)}%" if job.fraction is not None else ""
        return " · ".join(t for t in (job.text, pct) if t) or "Working…"
    if job.state == DONE:
        return "Done"
    if job.state == CANCELLED:
        return "Cancelled"
    if job.state == FAILED:
        return job.text or "Failed"
    return job.text or "Waiting"


def _short(text, n=40):
    return text if len(text) <= n else text[:n - 1] + "…"

