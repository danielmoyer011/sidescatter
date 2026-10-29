"""UMAP, tSNE, PCA and FlowSOM: the run dialog and the metacluster heatmap.

The dialog picks a population, the samples and channels, and the methods.
The work runs in a background thread with a progress bar and a Cancel
button; the results become channels and populations in the workspace.
"""

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ..core import highdim
from ..core.workspace import ROOT
from . import theme

HEAT_STOPS = ((0.0, "#2B2D6E"), (0.35, "#3E7FC1"), (0.65, "#7CCBA2"), (1.0, "#F8E16C"))


class _Worker(QtCore.QObject):
    progressed = QtCore.Signal(float, str)
    finished = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, inputs):
        super().__init__()
        self.inputs = inputs
        self.cancel_requested = False

    def run(self):
        def progress(fraction, text):
            if self.cancel_requested:
                raise highdim.Cancelled()
            self.progressed.emit(fraction, text)

        try:
            result = highdim.compute(self.inputs, progress)
        except highdim.Cancelled:
            self.failed.emit("")
        except Exception as exc:  # shown to the user, not raised in the thread
            self.failed.emit(str(exc) or type(exc).__name__)
        else:
            self.finished.emit(result)


class AnalysisDialog(QtWidgets.QDialog):
    """Set up and run UMAP / tSNE / PCA / FlowSOM on one population."""

    analysisDone = QtCore.Signal(object)  # the Analysis added to the workspace

    def __init__(self, workspace, population=None, parent_path=ROOT, samples=(), settings=None,
                 parent=None):
        super().__init__(parent)
        self.ws = workspace
        self.setWindowTitle("UMAP, tSNE and FlowSOM")
        self.thread = self.worker = None
        if settings is None:
            usable = [s for s in samples if self.ws.applies_to(s, population, parent_path)
                      and not self._is_concatenated(s)]
            settings = workspace.new_analysis_settings(population, parent_path, usable)
        self.settings = settings
        st = settings

        trail = " › ".join([*st.parent_path[1:], st.population]) if st.population else "All events"
        self.name = QtWidgets.QLineEdit(st.name)
        pop = QtWidgets.QLabel(f"<b>{trail}</b>")

        # ----- samples ---------------------------------------------------------
        self.sample_list = QtWidgets.QListWidget()
        candidates = [s for s in self.ws.samples
                      if not self._is_concatenated(s) and self.ws.applies_to(s, st.population, st.parent_path)]
        for sid in candidates:
            it = QtWidgets.QListWidgetItem(sid)
            it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.Checked if sid in st.samples else QtCore.Qt.Unchecked)
            self.sample_list.addItem(it)
        self.sample_list.itemChanged.connect(self._update_summary)

        # ----- channels --------------------------------------------------------
        self.channel_list = QtWidgets.QListWidget()
        first = st.samples[0] if st.samples else (candidates[0] if candidates else None)
        chans = [c for c in self.ws.channels(first) if c.kind != "derived"] if first else []
        defaults = set(st.channels or highdim.default_channels(chans))
        for c in chans:
            it = QtWidgets.QListWidgetItem(c.label)
            it.setData(QtCore.Qt.UserRole, c.pnn)
            it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.Checked if c.pnn in defaults else QtCore.Qt.Unchecked)
            self.channel_list.addItem(it)
        self.channel_list.itemChanged.connect(self._update_summary)
        ticks = QtWidgets.QHBoxLayout()
        self.tick_buttons = []
        for text, fn in (("Defaults", self._tick_defaults), ("All", lambda: self._tick_all(True)),
                         ("None", lambda: self._tick_all(False))):
            b = QtWidgets.QPushButton(text)
            b.setAutoDefault(False)
            b.clicked.connect(fn)
            ticks.addWidget(b)
            self.tick_buttons.append(b)
        ticks.addStretch()

        # ----- methods ---------------------------------------------------------
        self.method_boxes = {}
        methods = QtWidgets.QHBoxLayout()
        tips = {"pca": "Principal components: fast, linear; PCA-1 and PCA-2 are added",
                "umap": "UMAP map: UMAP-1 and UMAP-2 are added",
                "tsne": "tSNE map (openTSNE): TSNE-1 and TSNE-2 are added",
                "flowsom": "FlowSOM clusters and metaclusters; each metacluster becomes a population"}
        for m in highdim.METHODS:
            box = QtWidgets.QCheckBox(highdim.METHOD_LABELS[m])
            box.setChecked(m in st.methods)
            box.setToolTip(tips[m])
            box.toggled.connect(self._update_summary)
            self.method_boxes[m] = box
            methods.addWidget(box)
        methods.addStretch()

        # ----- numbers ---------------------------------------------------------
        def spin(lo, hi, value, step=1):
            s = QtWidgets.QSpinBox()
            s.setRange(lo, hi)
            s.setSingleStep(step)
            s.setGroupSeparatorShown(True)
            s.setValue(int(value))
            s.valueChanged.connect(self._update_summary)
            return s

        self.per_sample = spin(100, 10_000_000, st.per_sample, 1000)
        self.max_total = spin(1000, 50_000_000, st.max_total, 100_000)
        self.seed = spin(0, 2_147_483_647, st.seed)
        self.n_meta = spin(2, 100, st.n_meta)
        self.som_x = spin(2, 50, st.som_x)
        self.som_y = spin(2, 50, st.som_y)
        self.neighbors = spin(2, 200, st.umap_neighbors)
        self.min_dist = QtWidgets.QDoubleSpinBox()
        self.min_dist.setRange(0.0, 1.0)
        self.min_dist.setSingleStep(0.05)
        self.min_dist.setValue(st.umap_min_dist)
        self.perplexity = QtWidgets.QDoubleSpinBox()
        self.perplexity.setRange(2, 500)
        self.perplexity.setValue(st.tsne_perplexity)

        grid = QtWidgets.QHBoxLayout()
        grid.addWidget(self.som_x)
        grid.addWidget(QtWidgets.QLabel("×"))
        grid.addWidget(self.som_y)

        form = QtWidgets.QFormLayout()
        form.addRow("Events per sample", self.per_sample)
        form.addRow("At most, in total", self.max_total)
        form.addRow("FlowSOM metaclusters", self.n_meta)
        form.addRow("FlowSOM grid", grid)
        form.addRow("UMAP neighbours", self.neighbors)
        form.addRow("UMAP minimum distance", self.min_dist)
        form.addRow("tSNE perplexity", self.perplexity)
        form.addRow("Random seed", self.seed)

        self.summary = QtWidgets.QLabel()
        self.summary.setObjectName("GraphHint")
        self.summary.setWordWrap(True)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(False)
        self.progress_text = QtWidgets.QLabel()
        self.progress_text.setObjectName("GraphHint")
        for w in (self.progress, self.progress_text):
            w.setVisible(False)

        self.buttons = QtWidgets.QDialogButtonBox()
        self.run_button = self.buttons.addButton("Run", QtWidgets.QDialogButtonBox.AcceptRole)
        self.cancel_button = self.buttons.addButton(QtWidgets.QDialogButtonBox.Cancel)
        self.run_button.setDefault(True)
        self.run_button.clicked.connect(self.run)
        self.cancel_button.clicked.connect(self._cancel)

        left = QtWidgets.QVBoxLayout()
        top = QtWidgets.QFormLayout()
        top.addRow("Name", self.name)
        top.addRow("Population", pop)
        left.addLayout(top)
        left.addWidget(QtWidgets.QLabel("Samples"))
        left.addWidget(self.sample_list, 1)
        left.addWidget(QtWidgets.QLabel("Methods"))
        left.addLayout(methods)
        left.addLayout(form)

        right = QtWidgets.QVBoxLayout()
        right.addWidget(QtWidgets.QLabel("Channels (on their display scale)"))
        right.addWidget(self.channel_list, 1)
        right.addLayout(ticks)

        cols = QtWidgets.QHBoxLayout()
        cols.addLayout(left, 1)
        cols.addSpacing(12)
        cols.addLayout(right, 1)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(cols, 1)
        layout.addWidget(self.summary)
        layout.addWidget(self.progress)
        layout.addWidget(self.progress_text)
        layout.addWidget(self.buttons)
        self.resize(820, 620)
        self._update_summary()

    @staticmethod
    def _is_concatenated(sid):
        return sid.startswith("Concatenated (")

    # ----- choices ---------------------------------------------------------

    def _checked(self, widget, role=QtCore.Qt.DisplayRole):
        return [widget.item(i).data(role) for i in range(widget.count())
                if widget.item(i).checkState() == QtCore.Qt.Checked]

    def _tick_all(self, on):
        for i in range(self.channel_list.count()):
            self.channel_list.item(i).setCheckState(QtCore.Qt.Checked if on else QtCore.Qt.Unchecked)

    def _tick_defaults(self):
        sid = self.ws_first_sample()
        if sid is None:
            return
        defaults = set(highdim.default_channels(self.ws.channels(sid)))
        for i in range(self.channel_list.count()):
            it = self.channel_list.item(i)
            it.setCheckState(QtCore.Qt.Checked if it.data(QtCore.Qt.UserRole) in defaults
                             else QtCore.Qt.Unchecked)

    def ws_first_sample(self):
        return self.sample_list.item(0).text() if self.sample_list.count() else None

    def current_settings(self):
        st = highdim.Settings.from_dict(self.settings.to_dict())
        st.name = self.name.text().strip() or self.settings.name
        st.samples = self._checked(self.sample_list)
        st.channels = self._checked(self.channel_list, QtCore.Qt.UserRole)
        st.methods = [m for m, box in self.method_boxes.items() if box.isChecked()]
        st.per_sample, st.max_total, st.seed = (self.per_sample.value(), self.max_total.value(),
                                                self.seed.value())
        st.n_meta, st.som_x, st.som_y = self.n_meta.value(), self.som_x.value(), self.som_y.value()
        st.umap_neighbors, st.umap_min_dist = self.neighbors.value(), self.min_dist.value()
        st.tsne_perplexity = self.perplexity.value()
        return st

    def _problem(self, st):
        if not st.samples:
            return "Tick at least one sample."
        if len(st.channels) < 2:
            return "Tick at least two channels."
        if not st.methods:
            return "Pick at least one method."
        if "flowsom" in st.methods and st.n_meta > st.som_x * st.som_y:
            return "FlowSOM needs fewer metaclusters than grid nodes."
        clash = self.ws.analysis(st.name)
        if clash is not None and st.name != self.settings.name:
            return f"There is already an analysis called {st.name}."
        return None

    def _update_summary(self, *_):
        st = self.current_settings()
        problem = self._problem(st)
        self.run_button.setEnabled(problem is None and self.thread is None)
        if problem:
            self.summary.setText(problem)
            return
        per = st.per_sample_target()
        adds = ", ".join(st.all_output_channels())
        meta = (f" Each of the {st.n_meta} metaclusters becomes a population under this one."
                if "flowsom" in st.methods else "")
        replace = " Running again replaces the earlier result." if self.ws.analysis(st.name) else ""
        self.summary.setText(
            f"Builds the map from up to {per:,} events of each of {len(st.samples)} sample(s), then "
            f"places every other event of the population on it. Adds {adds}.{meta}{replace}")

    # ----- running ---------------------------------------------------------

    def run(self):
        st = self.current_settings()
        if self._problem(st):
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            inputs = highdim.gather(self.ws, st)
        except Exception as exc:
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.warning(self, "UMAP, tSNE and FlowSOM", str(exc))
            return
        QtWidgets.QApplication.restoreOverrideCursor()
        self.settings = st
        self._set_running(True)
        self.thread = QtCore.QThread(self)
        self.worker = _Worker(inputs)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progressed.connect(self._progressed)
        self.worker.finished.connect(self._done)
        self.worker.failed.connect(self._failed)
        self.thread.start()

    def _set_running(self, on):
        for w in (self.progress, self.progress_text):
            w.setVisible(on)
        for w in (self.name, self.sample_list, self.channel_list, *self.method_boxes.values(),
                  self.per_sample, self.max_total, self.seed, self.n_meta, self.som_x, self.som_y,
                  self.neighbors, self.min_dist, self.perplexity, *self.tick_buttons):
            w.setEnabled(not on)
        self.run_button.setEnabled(not on)
        self.cancel_button.setEnabled(True)
        self.cancel_button.setText("Cancel")
        self.progress.setValue(0)
        self.progress_text.setText("Reading the population…" if on else "")

    def _progressed(self, fraction, text):
        self.progress.setValue(int(1000 * fraction))
        self.progress_text.setText(text)

    def _stop_thread(self):
        if self.thread is not None:
            self.thread.quit()
            self.thread.wait()
        self.thread = self.worker = None

    def _done(self, result):
        self._stop_thread()
        record = self.ws.add_analysis(result)
        self.analysisDone.emit(record)
        self.accept()

    def _failed(self, message):
        self._stop_thread()
        self._set_running(False)
        self._update_summary()
        if message:
            QtWidgets.QMessageBox.warning(self, "UMAP, tSNE and FlowSOM", message)

    def _cancel(self):
        if self.worker is None:
            self.reject()
            return
        self.worker.cancel_requested = True
        self.cancel_button.setEnabled(False)
        self.progress_text.setText("Stopping after the current step…")

    def closeEvent(self, event):
        if self.worker is not None:
            self._cancel()
            event.ignore()
        else:
            super().closeEvent(event)

    def reject(self):
        if self.worker is not None:
            self._cancel()
            return
        super().reject()


def heat_colour(value):
    """Colour for a value in 0..1 on the heatmap scale."""
    if not np.isfinite(value):
        return QtGui.QColor(theme.BORDER)
    for (a, ca), (b, cb) in zip(HEAT_STOPS, HEAT_STOPS[1:]):
        if value <= b:
            t = (value - a) / (b - a)
            c1, c2 = QtGui.QColor(ca), QtGui.QColor(cb)
            return QtGui.QColor(int(c1.red() + t * (c2.red() - c1.red())),
                                int(c1.green() + t * (c2.green() - c1.green())),
                                int(c1.blue() + t * (c2.blue() - c1.blue())))
    return QtGui.QColor(HEAT_STOPS[-1][1])


def metacluster_table(ws, record):
    """(row names, column labels, medians, percentages) for an analysis's metaclusters.

    Rows follow the metacluster numbers; each takes its population's current
    name, so renamed metaclusters ("NK cells") show their new names.
    """
    st = record.settings
    names = {k: f"Metacluster {k}" for k in range(1, st.n_meta + 1)}
    xform = ws.transform(st.meta_channel)
    for gname, parent in ws.children(st.population_path):
        k = record.metacluster_of(ws.get_gate(gname, parent), xform)
        if k is not None:
            names[k] = gname
    first = next((s for s in st.samples if s in ws.samples), None)
    labels = {c.pnn: c.label for c in ws.channels(first)} if first else {}
    cols = [labels.get(c, c) for c in st.channels]
    counts = np.asarray(record.meta_counts, dtype=float)
    pct = 100 * counts / counts.sum() if counts.sum() else counts
    return [names[k] for k in range(1, st.n_meta + 1)], cols, np.asarray(record.meta_medians), pct


class HeatmapDialog(QtWidgets.QDialog):
    """Median intensity of each channel in each FlowSOM metacluster."""

    def __init__(self, workspace, record, parent=None):
        super().__init__(parent)
        self.ws, self.record = workspace, record
        self.setWindowTitle(f"Metacluster heatmap — {record.settings.name}")
        self.rows, self.cols, self.medians, self.pct = metacluster_table(workspace, record)

        self.table = QtWidgets.QTableWidget(len(self.rows), len(self.cols) + 1)
        self.table.setHorizontalHeaderLabels(["% of events", *self.cols])
        self.table.setVerticalHeaderLabels(self.rows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        header = self.table.horizontalHeader()
        header.setDefaultAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignBottom)
        # Each channel is coloured on its own range, as FlowSOM heatmaps are.
        lo, hi = np.nanmin(self.medians, axis=0), np.nanmax(self.medians, axis=0)
        scaled = (self.medians - lo) / np.where(hi > lo, hi - lo, 1)
        for r in range(len(self.rows)):
            pct = QtWidgets.QTableWidgetItem(f"{self.pct[r]:.2f}")
            pct.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            self.table.setItem(r, 0, pct)
            for c in range(len(self.cols)):
                v = self.medians[r, c]
                it = QtWidgets.QTableWidgetItem(f"{v:.2f}" if np.isfinite(v) else "")
                colour = heat_colour(scaled[r, c])
                it.setBackground(colour)
                it.setForeground(QtGui.QColor("#FFFFFF" if colour.lightness() < 140 else theme.INK))
                it.setTextAlignment(QtCore.Qt.AlignCenter)
                it.setToolTip(f"{self.rows[r]} · {self.cols[c]}: median {v:.3f} on the display scale")
                self.table.setItem(r, c + 1, it)
        self.table.resizeColumnsToContents()
        for c in range(1, len(self.cols) + 1):
            self.table.setColumnWidth(c, max(64, min(self.table.columnWidth(c), 120)))

        note = QtWidgets.QLabel(
            "Median of each channel on its display scale (0 is the bottom of the plot axis, 1 the top), "
            "from the events the map was "
            "built from. Colours run from each channel's lowest metacluster to its highest. "
            "Rename a metacluster in the workspace to label its row.")
        note.setObjectName("GraphHint")
        note.setWordWrap(True)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        export = buttons.addButton("Export CSV…", QtWidgets.QDialogButtonBox.ActionRole)
        export.clicked.connect(self.export_csv)
        buttons.rejected.connect(self.reject)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.table, 1)
        layout.addWidget(note)
        layout.addWidget(buttons)
        self.resize(min(1200, 160 + 90 * (len(self.cols) + 1)), 140 + 30 * len(self.rows))

    def dataframe(self):
        import pandas as pd

        df = pd.DataFrame(self.medians, index=self.rows, columns=self.cols)
        df.insert(0, "% of events", self.pct)
        df.index.name = "Metacluster"
        return df

    def export_csv(self, path=None):
        if not path:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Export heatmap", f"{self.record.settings.name} metaclusters.csv", "CSV (*.csv)")
        if path:
            self.dataframe().to_csv(path)
        return path
