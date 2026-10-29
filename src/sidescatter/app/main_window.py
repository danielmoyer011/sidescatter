"""Main window, laid out like FlowJo 10: a ribbon on top and the workspace below.

Double-clicking a sample or population opens it in its own graph window,
where the gating tools sit right above the plot. The Layout Editor arranges
plots into printable figures.
"""

import sys
from pathlib import Path

import flowkit as fk
from PySide6 import QtCore, QtGui, QtWidgets

from .. import __version__
from ..core.fcs import find_fcs_files
from ..core.workspace import ROOT, Workspace
from . import icons
from .analysis_dialog import AnalysisDialog, HeatmapDialog
from .graph_view import GraphWindow
from .layout_editor import LayoutWindow
from .ribbon import Ribbon
from .suggestion_box import SuggestionDialog
from .table_editor import TableEditor
from .unmixing_tab import UnmixingTab
from .workspace_panel import WorkspacePanel

DEMO_DIR = Path(__file__).resolve().parents[1] / "demo_data" / "spectral"
WORKSPACE_FILTER = "Sidescatter workspace (*.ssw)"
WEBSITE = "https://github.com/danielmoyer011/sidescatter"


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ws = Workspace()
        self.path = None
        self.dirty = False
        self.ask_before_discard = True
        self.graphs = []   # open GraphWindows
        self.graph = None  # the GraphView used most recently
        self.layout_window = None

        self.panel = WorkspacePanel(self.ws)
        self.panel.populationActivated.connect(self.open_graph)
        self.panel.changed.connect(self._changed)
        self.panel.analysisRequested.connect(self.run_analysis)
        self.unmix_tab = UnmixingTab(self.ws)
        self.unmix_tab.unmixed.connect(self._after_unmix)
        self.unmix_tab.changed.connect(self._changed)
        self.pages = QtWidgets.QTabWidget()
        self.pages.setObjectName("Pages")
        self.pages.setDocumentMode(True)
        self.pages.addTab(self.panel, icons.icon("new_workspace"), "Workspace")
        self.pages.addTab(self.unmix_tab, icons.icon("unmixing"), "Unmixing")
        self.setCentralWidget(self.pages)
        self.resize(1320, 840)

        self.setAcceptDrops(True)
        self._build_menus()
        self._build_toolbar()
        self.license_button = QtWidgets.QPushButton("License…")
        self.license_button.setObjectName("LicenseButton")
        self.license_button.clicked.connect(self.show_license)
        self.statusBar().addPermanentWidget(self.license_button)
        self.statusBar().showMessage(
            "Add or drop FCS files, open a workspace, or try the demo spectral data.")
        self._update_title()

    # ----- UI construction -------------------------------------------------

    def _action(self, text, icon, slot, shortcut=None, ribbon_text=None, tip=None):
        a = QtGui.QAction(icons.icon(icon), text, self)
        a.setIconText(ribbon_text or text.replace("&", "").replace("…", ""))
        a.setToolTip(tip or text.replace("&", ""))
        if shortcut:
            a.setShortcut(QtGui.QKeySequence(shortcut))
        a.triggered.connect(lambda checked=False: slot())
        self.addAction(a)  # shortcuts work with or without a menu bar
        return a

    def _build_menus(self):
        A = self._action
        self.acts = acts = {
            "new": A("&New workspace", "new_workspace", self.new_workspace, QtGui.QKeySequence.New,
                     "New\nWorkspace"),
            "open": A("&Open workspace…", "open", self.open_workspace, QtGui.QKeySequence.Open,
                      "Open\nWorkspace"),
            "save": A("&Save", "save", self.save, QtGui.QKeySequence.Save, "Save"),
            "save_as": A("Save &as…", "save", self.save_as, QtGui.QKeySequence.SaveAs, "Save\nAs…"),
            "add": A("Add &samples (FCS)…", "add_samples", self.add_samples, "Ctrl+I", "Add\nSamples…"),
            "import": A("Import &FlowJo 10 workspace…", "import_flowjo", self.import_flowjo, None,
                        "Import FlowJo\nWorkspace…"),
            "demo": A("Open &demo spectral data", "demo", self.open_demo, None, "Demo\nData"),
            "export": A("E&xport plot…", "export", self.export_plot, None, "Export\nPlot…"),
            "quit": A("&Quit", "quit", self.close, QtGui.QKeySequence.Quit, "Quit"),
            "group": A("&Create group…", "create_group", self.panel.create_group, "Ctrl+G",
                       "Create\nGroup…"),
            "keyword": A("Group samples by &keyword…", "keyword", self.group_by_keyword, None,
                         "Group by\nKeyword…"),
            "rename": A("&Rename population…", "rename", self.panel.rename_selected, "F2",
                        "Rename\nPopulation…"),
            "boolean": A("&Boolean gate…", "boolean", self.panel.make_boolean, "Ctrl+B",
                         "Boolean\nGate…", tip="Combine the selected populations with AND, OR "
                         "and NOT, or make all their combinations"),
            "delete": A("&Delete gate", "delete", self.panel.delete_selected, None, "Delete\nGate"),
            "remove": A("Re&move samples", "remove_sample", self.panel.remove_selected_samples, None,
                        "Remove\nSamples"),
            "graph": A("Open &graph window", "rectangle", self.open_selected_graph, "Ctrl+Return",
                       "Graph\nWindow"),
            "layout": A("&Layout editor…", "layout_editor", self.open_layout_editor, "Ctrl+L",
                        "Layout\nEditor"),
            "unmix": A("Spectral &unmixing", "unmixing", self.open_unmixing, "Ctrl+U",
                       "Spectral\nUnmixing", tip="Open the Unmixing tab"),
            "export_exp": A("Export &experiment…", "export", lambda: self.unmix_tab.export(), None,
                            "Export\nExperiment…",
                            tip="Write a zipped experiment with Raw and Unmixed folders, like SpectroFlo"),
            "analysis": A("UMAP, tSNE and &FlowSOM…", "umap", self.run_analysis, None,
                          "UMAP / tSNE\nFlowSOM…",
                          tip="Map and cluster the selected population (UMAP, tSNE, PCA, FlowSOM)"),
            "heatmap": A("Metacluster &heatmap…", "heatmap", self.show_heatmap, None,
                         "Cluster\nHeatmap…", tip="Median of each channel in each FlowSOM metacluster"),
            "remove_analysis": A("Delete &analysis…", "delete", self.delete_analysis, None,
                                 "Delete\nAnalysis…",
                                 tip="Remove an analysis with its channels and metacluster populations"),
            "table": A("&Table editor…", "table_editor", self.open_table_editor, "Ctrl+T",
                       "Table\nEditor"),
            "transform": A("Axis &transform…", "transform", self.edit_transform, None,
                           "Axis\nTransform…"),
            "license": A("&License…", "license", self.show_license, None, "License"),
            "about": A("&About Sidescatter", "about", self.about, None, "About"),
            "suggest": A("&Suggestion box…", "suggest", self.open_suggestions, None,
                         "Suggest an\nIdea…", "Send an idea or report a bug"),
            "website": A("Sidescatter on &GitHub", "website", self.open_website, None,
                         "Project\nWebsite"),
        }
        menus = (
            ("&File", ["new", "open", "save", "save_as", None, "add", "import", "demo", None,
                       "export", None, "quit"]),
            ("&Edit", ["boolean", "rename", "delete", "remove"]),
            ("&Workspace", ["add", "group", "keyword", None, "graph", "layout"]),
            ("&Tools", ["unmix", "export_exp", None, "analysis", "heatmap", "remove_analysis", None,
                        "table", "transform"]),
            ("&Help", ["suggest", None, "about", "license", "website"]),
        )
        # FlowJo 10 has no menu bar on Windows: the ribbon replaces it. macOS keeps
        # its global menu bar, so the same actions go there too.
        if sys.platform == "darwin":
            bar = self.menuBar()
            for title, keys in menus:
                m = bar.addMenu(title)
                for k in keys:
                    m.addSeparator() if k is None else m.addAction(acts[k])

    def _build_toolbar(self):
        acts = self.acts
        self.ribbon = r = Ribbon()
        r.app_button.setMenu(self._app_menu())
        for k in ("save", "add", "group", "table", "layout", "unmix", "demo"):
            r.add_quick(acts[k])
        r.add_corner(acts["suggest"])
        r.add_corner(acts["license"])
        r.add_corner(acts["about"])

        home = r.add_tab("Sidescatter")
        home.add_group("Navigate", [acts["new"], acts["add"], acts["group"], acts["table"]])
        home.add_group("Experiment", [acts["keyword"], acts["graph"], acts["boolean"], acts["layout"],
                                      acts["demo"]])
        home.add_group("Cytometry", [acts["unmix"], acts["analysis"], acts["transform"], acts["import"]])
        home.add_group("Help", [acts["suggest"], acts["website"], acts["about"], acts["license"]])

        page = r.add_tab("File")
        page.add_group("Workspace", [acts["new"], acts["open"], acts["save"], acts["save_as"]])
        page.add_group("Import", [acts["add"], acts["import"], acts["demo"]])
        page.add_group("Export", [acts["export"], acts["export_exp"]])
        page.add_group("Application", [acts["quit"]])

        page = r.add_tab("Edit")
        page.add_group("Populations", [acts["boolean"], acts["rename"], acts["delete"]])
        page.add_group("Samples", [acts["remove"]])

        page = r.add_tab("Workspace")
        page.add_group("Samples", [acts["add"], acts["demo"]])
        page.add_group("Groups", [acts["group"], acts["keyword"]])
        page.add_group("Gates", [acts["boolean"]])
        page.add_group("Graphs", [acts["graph"], acts["layout"], acts["export"]])

        page = r.add_tab("Tools")
        page.add_group("Spectral", [acts["unmix"], acts["export_exp"]])
        page.add_group("Clustering", [acts["analysis"], acts["heatmap"], acts["remove_analysis"]])
        page.add_group("Statistics", [acts["table"]])
        page.add_group("Figures", [acts["layout"]])

        page = r.add_tab("Configure")
        page.add_group("Display", [acts["transform"]])
        page.add_group("About", [acts["license"], acts["about"]])
        self.setMenuWidget(r)

    def _app_menu(self):
        m = QtWidgets.QMenu(self)
        for k in ("new", "open", "save", "save_as", None, "license", "about", None, "quit"):
            m.addSeparator() if k is None else m.addAction(self.acts[k])
        return m

    # ----- state -----------------------------------------------------------

    def _changed(self):
        self.dirty = True
        self.panel.rebuild_groups()
        self._update_title()
        sender = self.sender()
        for win in list(self.graphs):
            if win.view is not sender:
                QtCore.QTimer.singleShot(0, win.view.refresh)
        if self.layout_window is not None and sender is not self.layout_window:
            QtCore.QTimer.singleShot(0, self.layout_window.refresh)

    def _layouts_changed(self):
        self.dirty = True
        self._update_title()

    def _update_title(self):
        name = Path(self.path).name if self.path else "Untitled"
        self.setWindowTitle(f"{name}{' •' if self.dirty else ''} — Sidescatter {__version__}")

    def _set_workspace(self, ws, path=None):
        self.ws = ws
        for win in list(self.graphs):
            win.close()
        self.panel.set_workspace(ws)
        if self.layout_window is not None:
            self.layout_window.set_workspace(ws)
        self.unmix_tab.set_workspace(ws)
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
        if self.pages.currentWidget() is self.unmix_tab:
            skipped = self.unmix_tab.add_files(paths)
            if skipped:
                QtWidgets.QMessageBox.warning(
                    self, "Not FCS files", f"Skipped {len(skipped)} item(s):\n\n" + "\n".join(skipped[:10]))
            return
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

    # ----- graph windows ---------------------------------------------------

    def open_graph(self, sample_id, name=None, parent_path=ROOT):
        """Open a population in a graph window, or bring its window forward."""
        parent_path = tuple(parent_path)
        for win in self.graphs:
            v = win.view
            if v.sample_id == sample_id and v.population == (name, parent_path):
                win.raise_()
                win.activateWindow()
                self.graph = v
                return v
        win = GraphWindow(self.ws, self)
        v = win.view
        v.group = self.panel.current_group()
        v.gatesChanged.connect(self._changed)
        v.transformRequested.connect(self.edit_transform)
        v.populationShown.connect(lambda *_: setattr(self, "graph", v))
        win.closed.connect(self._graph_closed)
        self.graphs.append(win)
        v.show_population(sample_id, name, parent_path)
        if name is not None and self.ws.is_boolean(name, parent_path):
            # nothing to draw for a Boolean gate: show it on its first input's axes
            axes = self.ws.plot_axes(name, parent_path)
            if len(axes) == 2:
                v.set_axes(*axes)
            elif len(axes) == 1:
                v.set_axes(axes[0], v.y_combo.currentData())
        offset = 28 * ((len(self.graphs) - 1) % 8)
        top_left = self.mapToGlobal(QtCore.QPoint(self.width() // 3, 90))
        win.move(top_left + QtCore.QPoint(offset, offset))
        win.show()
        self.graph = v
        return v

    def open_selected_graph(self):
        sel = self.panel.selected()
        if sel:
            self.open_graph(*sel)
        elif self.panel.tree.topLevelItemCount():
            self.open_graph(*self.panel._key(self.panel.tree.topLevelItem(0)))

    def _graph_closed(self, win):
        if win in self.graphs:
            self.graphs.remove(win)
        if self.graph is win.view:
            self.graph = self.graphs[-1].view if self.graphs else None

    # ----- layout editor ---------------------------------------------------

    def open_layout_editor(self):
        """Open the Layout Editor (one per workspace window), or bring it forward."""
        if self.layout_window is None:
            win = LayoutWindow(self.ws, self, current_population=self.panel.selected,
                               current_group=self.panel.current_group)
            win.axes_for = self._graph_axes
            win.changed.connect(self._layouts_changed)
            win.graphRequested.connect(self.open_graph)
            self.layout_window = win
        win = self.layout_window
        win.show()
        win.raise_()
        win.activateWindow()
        return win

    def _graph_axes(self, sample_id, name, parent_path):
        """(x, y, plot type) of an open graph window showing this population, if any."""
        for win in self.graphs:
            v = win.view
            if v.sample_id == sample_id and v.population == (name, tuple(parent_path)):
                kind = v.type_combo.currentText()
                # Layouts don't draw spectra yet; drop the plot in as a pseudocolour.
                return v.x_combo.currentData(), v.y_combo.currentData(), "Pseudocolor" if kind == "Spectrum" else kind
        return None

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
        self.unmix_tab.store()
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
        """Load the demo controls and sample into a new experiment on the Unmixing tab."""
        files = sorted(str(p) for p in (DEMO_DIR / "controls").glob("*.fcs"))
        files.append(str(DEMO_DIR / "PBMC_full_panel.fcs"))
        tab = self.unmix_tab
        if tab.specs:
            tab.new_setup("Demo")
        tab.add_files(files)
        self.open_unmixing()

    def export_plot(self):
        if self.graph is None or self.graph.sample_id is None:
            QtWidgets.QMessageBox.information(
                self, "Export plot", "Open a graph window first (double-click a population).")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export plot", "plot.png", "PNG image (*.png);;SVG vector (*.svg)")
        if path:
            self.graph.export_image(path)

    # ----- analysis actions ------------------------------------------------

    def open_unmixing(self, paths=()):
        self.pages.setCurrentWidget(self.unmix_tab)
        if paths:
            self.unmix_tab.add_files(paths)

    def _after_unmix(self, new_ids):
        self.pages.setCurrentWidget(self.panel)
        self._changed()
        self.panel.select_group("Unmixed")
        if new_ids:
            self.open_graph(new_ids[0])
        self.statusBar().showMessage(f"Unmixed {len(new_ids)} sample(s).", 6000)

    def run_analysis(self, sample_id=None, name=None, parent_path=ROOT):
        """Set up UMAP / tSNE / PCA / FlowSOM on a population (the selected one by default)."""
        if not self.ws.samples:
            QtWidgets.QMessageBox.information(self, "UMAP, tSNE and FlowSOM", "Add samples first.")
            return None
        if sample_id is None:
            sel = self.panel.selected()
            if sel:
                sample_id, name, parent_path = sel
        group = self.panel.current_group()
        samples = self.ws.groups.get(group) or self.ws.groups.get("All Samples", [])
        if sample_id is not None and sample_id not in samples:
            samples = [sample_id]
        dlg = AnalysisDialog(self.ws, name, parent_path, samples, parent=self)
        dlg.analysisDone.connect(self._after_analysis)
        dlg.exec()
        return dlg

    def _after_analysis(self, record):
        self._changed()
        st = record.settings
        self.panel.select_group(st.name)
        view = self.open_graph(record.concatenated_id, st.population, st.parent_path)
        maps = [m for m in ("umap", "tsne", "pca") if m in st.methods]
        if maps:
            x, y = st.output_channels()[maps[0]]
            view.set_axes(x, y)
        self.statusBar().showMessage(
            f"{st.name}: added {', '.join(st.all_output_channels())} to {len(st.samples)} sample(s).", 8000)

    def _pick_analysis(self, title, flowsom_only=False):
        names = [a.settings.name for a in self.ws.analyses
                 if not flowsom_only or "flowsom" in a.settings.methods]
        if not names:
            QtWidgets.QMessageBox.information(
                self, title, "No FlowSOM analysis yet. Run UMAP, tSNE and FlowSOM first."
                if flowsom_only else "There are no analyses in this workspace.")
            return None
        if len(names) == 1:
            return self.ws.analysis(names[0])
        name, ok = QtWidgets.QInputDialog.getItem(self, title, "Analysis:", names, len(names) - 1, False)
        return self.ws.analysis(name) if ok else None

    def show_heatmap(self):
        record = self._pick_analysis("Metacluster heatmap", flowsom_only=True)
        if record is not None:
            dlg = HeatmapDialog(self.ws, record, self)
            dlg.show()
            return dlg
        return None

    def delete_analysis(self):
        record = self._pick_analysis("Delete analysis")
        if record is None:
            return
        st = record.settings
        answer = QtWidgets.QMessageBox.question(
            self, "Delete analysis",
            f"Delete {st.name}? This removes {', '.join(st.all_output_channels())} from its samples, "
            "its concatenated sample, and its metacluster populations with every gate inside them.")
        if answer == QtWidgets.QMessageBox.Yes:
            self.ws.remove_analysis(st.name)
            self._changed()

    def open_table_editor(self):
        samples = self.ws.groups.get(self.panel.current_group(), [])
        if not samples:
            QtWidgets.QMessageBox.information(self, "Table editor", "Add samples first.")
            return
        TableEditor(self.ws, samples, self).exec()

    def edit_transform(self, view=None):
        view = view or self.graph
        if view is None or view.sample_id is None:
            QtWidgets.QMessageBox.information(
                self, "Axis transform", "Open a graph window first (double-click a population).")
            return
        channels = [view.x_combo.currentData()]
        if not view.is_histogram:
            channels.append(view.y_combo.currentData())
        dlg = TransformDialog(self.ws, channels, self)
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            self._changed()
            view.refresh()

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

    def show_license(self):
        LicenseDialog(self).exec()

    def open_suggestions(self):
        SuggestionDialog(self).exec()

    def open_website(self):
        QtGui.QDesktopServices.openUrl(QtCore.QUrl(WEBSITE))

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


class LicenseDialog(QtWidgets.QDialog):
    """Where other cytometry software asks for a licence key."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("License")
        badge = QtWidgets.QLabel()
        badge.setPixmap(icons.icon("license", 56).pixmap(56, 56))
        title = QtWidgets.QLabel("No License, No Problem")
        title.setStyleSheet("font-size: 18pt; font-weight: 700;")
        body = QtWidgets.QLabel(
            "<p>Sidescatter doesn't do license keys, dongles, seat counts, activation servers "
            "or renewal reminders. It also doesn't stop you at demonstration files.</p>"
            "<p>Load any FCS file from any instrument, on as many computers as you like, for as "
            "long as you like. It's free software under the GNU GPL (version 3 or later).</p>"
            "<p style='color:#6A7187'>License status: <b>not required</b> · "
            f"Version {__version__} · Expires: never</p>")
        body.setWordWrap(True)
        body.setTextFormat(QtCore.Qt.RichText)
        body.setMinimumWidth(420)

        buttons = QtWidgets.QDialogButtonBox()
        read = buttons.addButton("Read the GPL", QtWidgets.QDialogButtonBox.HelpRole)
        read.clicked.connect(lambda: QtGui.QDesktopServices.openUrl(
            QtCore.QUrl("https://www.gnu.org/licenses/gpl-3.0.html")))
        ok = buttons.addButton("Nice", QtWidgets.QDialogButtonBox.AcceptRole)
        ok.setDefault(True)
        buttons.accepted.connect(self.accept)

        text = QtWidgets.QVBoxLayout()
        text.addWidget(title)
        text.addWidget(body)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(badge, 0, QtCore.Qt.AlignTop)
        row.addSpacing(10)
        row.addLayout(text, 1)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 16)
        layout.addLayout(row)
        layout.addSpacing(8)
        layout.addWidget(buttons)
