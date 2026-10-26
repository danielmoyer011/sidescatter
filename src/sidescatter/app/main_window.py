"""Main window: workspace panel on the left, graph window on the right."""

from pathlib import Path

import flowkit as fk
from PySide6 import QtCore, QtGui, QtWidgets

from .. import __version__
from ..core.fcs import find_fcs_files
from ..core.workspace import Workspace
from .graph_view import GraphView
from .table_editor import TableEditor
from .unmixing_dialog import UnmixingDialog
from .workspace_panel import WorkspacePanel

DEMO_DIR = Path(__file__).resolve().parents[1] / "demo_data" / "spectral"
WORKSPACE_FILTER = "Sidescatter workspace (*.ssw)"


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ws = Workspace()
        self.path = None
        self.dirty = False
        self.ask_before_discard = True

        self.panel = WorkspacePanel(self.ws)
        self.graph = GraphView(self.ws)
        self.panel.populationSelected.connect(self.graph.show_population)
        self.panel.changed.connect(self._changed)
        self.graph.gatesChanged.connect(self._changed)

        split = QtWidgets.QSplitter()
        split.addWidget(self.panel)
        split.addWidget(self.graph)
        split.setSizes([420, 880])
        self.setCentralWidget(split)
        self.resize(1320, 840)

        self.setAcceptDrops(True)
        self._build_menus()
        self._build_toolbar()
        self.statusBar().showMessage(
            "Add or drop FCS files, open a workspace, or try the demo spectral data.")
        self._update_title()

    # ----- UI construction -------------------------------------------------

    def _build_menus(self):
        m = self.menuBar().addMenu("&File")
        m.addAction("&New workspace", self.new_workspace, QtGui.QKeySequence.New)
        m.addAction("&Open workspace…", self.open_workspace, QtGui.QKeySequence.Open)
        m.addAction("&Save", self.save, QtGui.QKeySequence.Save)
        m.addAction("Save &as…", self.save_as, QtGui.QKeySequence.SaveAs)
        m.addSeparator()
        m.addAction("Add &samples (FCS)…", self.add_samples, "Ctrl+I")
        m.addAction("Import &FlowJo 10 workspace…", self.import_flowjo)
        m.addAction("Open &demo spectral data", self.open_demo)
        m.addSeparator()
        m.addAction("E&xport plot…", self.export_plot)
        m.addSeparator()
        m.addAction("&Quit", self.close, QtGui.QKeySequence.Quit)

        m = self.menuBar().addMenu("&Analysis")
        m.addAction("Spectral &unmixing…", self.open_unmixing, "Ctrl+U")
        m.addAction("&Table editor…", self.open_table_editor, "Ctrl+T")
        m.addAction("Axis &transform…", self.edit_transform)
        m.addAction("Group samples by &keyword…", self.group_by_keyword)

        m = self.menuBar().addMenu("&Help")
        m.addAction("&About Sidescatter", self.about)

    def _build_toolbar(self):
        tb = self.addToolBar("Gates")
        tb.setMovable(False)
        tb.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        tb.addAction("Rectangle", lambda: self.graph.start_gate("rectangle"))
        tb.addAction("Polygon", lambda: self.graph.start_gate("polygon"))
        tb.addAction("Range", lambda: self.graph.start_gate("range"))
        tb.addSeparator()
        tb.addAction("Unmixing", self.open_unmixing)
        tb.addAction("Table editor", self.open_table_editor)

    # ----- state -----------------------------------------------------------

    def _changed(self):
        self.dirty = True
        self.panel.rebuild_groups()
        self._update_title()

    def _update_title(self):
        name = Path(self.path).name if self.path else "Untitled"
        self.setWindowTitle(f"{name}{' •' if self.dirty else ''} — Sidescatter {__version__}")

    def _set_workspace(self, ws, path=None):
        self.ws = ws
        self.graph.ws = ws
        self.graph.sample_id = None
        self.panel.set_workspace(ws)
        self.path, self.dirty = path, False
        self._update_title()

    def _confirm_discard(self):
        if not self.dirty or not self.ask_before_discard:
            return True
        answer = QtWidgets.QMessageBox.question(
            self, "Unsaved changes", "Discard unsaved changes to this workspace?")
        return answer == QtWidgets.QMessageBox.Yes

    # ----- drag and drop ---------------------------------------------------

    @staticmethod
    def _local_paths(mime):
        if not mime.hasUrls():
            return []
        return [u.toLocalFile() for u in mime.urls() if u.isLocalFile()]

    def dragEnterEvent(self, event):
        if self._local_paths(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        self.dragEnterEvent(event)

    def dropEvent(self, event):
        paths = self._local_paths(event.mimeData())
        if not paths:
            event.ignore()
            return
        event.acceptProposedAction()
        self.add_dropped(paths)

    def add_dropped(self, paths):
        """Add dropped files and folders as samples; report anything that is not FCS."""
        fcs_paths, skipped = find_fcs_files(paths)
        if fcs_paths:
            before = len(self.ws.samples)
            self.add_samples(fcs_paths)
            added = len(self.ws.samples) - before
            self.statusBar().showMessage(f"Added {added} sample(s).", 6000)
        if skipped:
            shown = "\n".join(skipped[:10]) + (f"\n… and {len(skipped) - 10} more" if len(skipped) > 10 else "")
            QtWidgets.QMessageBox.warning(
                self, "Not FCS files",
                f"Skipped {len(skipped)} item(s) that are not FCS files:\n\n{shown}")
        elif not fcs_paths:
            QtWidgets.QMessageBox.warning(self, "No FCS files", "No FCS files were found in what was dropped.")
        return fcs_paths, skipped

    def closeEvent(self, event):
        if self._confirm_discard():
            event.accept()
        else:
            event.ignore()

    # ----- file actions ----------------------------------------------------

    def new_workspace(self):
        if self._confirm_discard():
            self._set_workspace(Workspace())

    def open_workspace(self):
        if not self._confirm_discard():
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open workspace", "", WORKSPACE_FILTER)
        if path:
            self._run(lambda: self._set_workspace(Workspace.load(path), path), "Opening workspace")

    def save(self):
        if self.path is None:
            return self.save_as()
        self._run(lambda: self.ws.save(self.path), "Saving")
        self.dirty = False
        self._update_title()

    def save_as(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save workspace", "workspace.ssw",
                                                        WORKSPACE_FILTER)
        if path:
            self.path = path
            self.save()

    def add_samples(self, paths=None):
        if not paths:
            paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Add samples", "",
                                                              "FCS files (*.fcs *.FCS)")
        if paths:
            self._run(lambda: self.ws.add_fcs(paths), "Loading samples")
            self._changed()

    def import_flowjo(self):
        if not self._confirm_discard():
            return
        wsp, _ = QtWidgets.QFileDialog.getOpenFileName(self, "FlowJo workspace", "",
                                                       "FlowJo workspace (*.wsp)")
        if not wsp:
            return
        fcs_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Folder containing the FCS files", str(Path(wsp).parent))
        if fcs_dir:
            self._run(lambda: self._set_workspace(Workspace.from_flowjo(wsp, fcs_dir)),
                      "Importing FlowJo workspace")

    def open_demo(self):
        self.add_samples([str(DEMO_DIR / "PBMC_full_panel.fcs")])
        controls = sorted(str(p) for p in (DEMO_DIR / "controls").glob("*.fcs"))
        self.open_unmixing(controls)

    def export_plot(self):
        if self.graph.sample_id is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export plot", "plot.png", "PNG image (*.png);;SVG vector (*.svg)")
        if path:
            self.graph.export_image(path)

    # ----- analysis actions ------------------------------------------------

    def open_unmixing(self, control_paths=()):
        dlg = UnmixingDialog(self.ws, self, control_paths=control_paths or ())
        dlg.unmixed.connect(self._after_unmix)
        dlg.exec()

    def _after_unmix(self, new_ids):
        self._changed()
        i = self.panel.group_combo.findData("Unmixed")
        if i >= 0:
            self.panel.group_combo.setCurrentIndex(i)
        if new_ids:
            self.graph.show_population(new_ids[0])
        self.statusBar().showMessage(f"Unmixed {len(new_ids)} sample(s).", 6000)

    def open_table_editor(self):
        samples = self.ws.groups.get(self.panel.current_group(), [])
        if not samples:
            QtWidgets.QMessageBox.information(self, "Table editor", "Add samples first.")
            return
        TableEditor(self.ws, samples, self).exec()

    def edit_transform(self):
        if self.graph.sample_id is None:
            return
        dlg = TransformDialog(self.ws, [self.graph.x_combo.currentData(),
                                        self.graph.y_combo.currentData()], self)
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            self._changed()
            self.graph.refresh()

    def group_by_keyword(self):
        key, ok = QtWidgets.QInputDialog.getText(self, "Group by keyword", "FCS keyword (e.g. $CYT, TUBE NAME):")
        if ok and key.strip():
            made = self.ws.group_by_keyword(key.strip())
            self._changed()
            self.statusBar().showMessage(f"Made {len(made)} group(s).", 6000)

    def about(self):
        QtWidgets.QMessageBox.about(
            self, "About Sidescatter",
            f"<b>Sidescatter {__version__}</b><p>Free, open-source flow cytometry analysis "
            "with spectral unmixing.</p><p>GPL-3.0-or-later. Built on FlowKit, PySide6 and "
            "pyqtgraph. Not affiliated with FlowJo or BD.</p>")

    # ----- helpers ---------------------------------------------------------

    def _run(self, fn, what):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            fn()
        except Exception as exc:
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.critical(self, what, f"{what} failed:\n\n{exc}")
            return
        QtWidgets.QApplication.restoreOverrideCursor()


class TransformDialog(QtWidgets.QDialog):
    """Pick the display transform for the plotted channels."""

    KINDS = ("Logicle (biexponential)", "Arcsinh", "Log", "Linear")

    def __init__(self, workspace, channels, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Axis transform")
        self.ws = workspace
        self.channel = QtWidgets.QComboBox()
        self.channel.addItems([c for c in dict.fromkeys(channels) if c])
        self.kind = QtWidgets.QComboBox()
        self.kind.addItems(self.KINDS)
        self.top = QtWidgets.QDoubleSpinBox()
        self.top.setRange(1, 1e9)
        self.top.setDecimals(0)
        self.width_ = QtWidgets.QDoubleSpinBox()
        self.width_.setRange(0, 3)
        self.width_.setSingleStep(0.1)
        self.decades = QtWidgets.QDoubleSpinBox()
        self.decades.setRange(1, 8)
        self.decades.setSingleStep(0.1)
        self.channel.currentTextChanged.connect(self._load)
        form = QtWidgets.QFormLayout(self)
        form.addRow("Channel", self.channel)
        form.addRow("Scale", self.kind)
        form.addRow("Top of scale (T)", self.top)
        form.addRow("Linear width (W)", self.width_)
        form.addRow("Decades (M)", self.decades)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._apply)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)
        self._load(self.channel.currentText())

    def _load(self, channel):
        t = self.ws.transform(channel)
        self.top.setValue(float(getattr(t, "param_t", 262144)))
        self.width_.setValue(float(getattr(t, "param_w", 0.5)))
        self.decades.setValue(float(getattr(t, "param_m", 4.5)))
        names = {fk.transforms.LogicleTransform: 0, fk.transforms.AsinhTransform: 1,
                 fk.transforms.LogTransform: 2, fk.transforms.LinearTransform: 3}
        self.kind.setCurrentIndex(names.get(type(t), 0))

    def _apply(self):
        t, w, m = self.top.value(), self.width_.value(), self.decades.value()
        kind = self.kind.currentIndex()
        if kind == 0:
            new = fk.transforms.LogicleTransform(param_t=t, param_w=w, param_m=m, param_a=0)
        elif kind == 1:
            new = fk.transforms.AsinhTransform(param_t=t, param_m=m, param_a=0)
        elif kind == 2:
            new = fk.transforms.LogTransform(param_t=t, param_m=m)
        else:
            new = fk.transforms.LinearTransform(param_t=t, param_a=0)
        try:
            self.ws.set_transform(self.channel.currentText(), new)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Transform", str(exc))
            return
        self.accept()
