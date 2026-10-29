"""Smoke tests for the desktop UI (run headless with QT_QPA_PLATFORM=offscreen)."""

import flowkit as fk
import numpy as np
import pytest
from PySide6 import QtCore

pytest.importorskip("pytestqt")

from sidescatter.app.main_window import DEMO_DIR, MainWindow  # noqa: E402
from sidescatter.app.jobs import wait_idle  # noqa: E402
from sidescatter.app.table_editor import TableEditor  # noqa: E402
from sidescatter.app.unmixing_tab import ROLE_CHOICES, UnmixingTab  # noqa: E402
from sidescatter.core.workspace import Workspace  # noqa: E402


@pytest.fixture
def window(qtbot):
    w = MainWindow()
    w.ask_before_discard = False
    qtbot.addWidget(w)
    w.add_samples([str(DEMO_DIR / "PBMC_full_panel.fcs")])
    wait_idle()
    return w


def _demo_files():
    return sorted(str(p) for p in (DEMO_DIR / "controls").glob("*.fcs")) + [str(DEMO_DIR / "PBMC_full_panel.fcs")]


def _role_of(tab, name):
    row = next(i for i, s in enumerate(tab.specs) if s.path.name == name)
    return tab.table.cellWidget(row, 1).currentText()


def test_unmixing_tab_classifies_unmixes_and_exports(window, qtbot, tmp_path, monkeypatch):
    from PySide6 import QtWidgets
    monkeypatch.setattr(QtWidgets.QMessageBox, "show", lambda self: None)
    window.open_demo()
    wait_idle()
    tab = window.unmix_tab
    assert window.pages.currentWidget() is tab
    assert _role_of(tab, "PBMC_full_panel.fcs") == "Multicolor sample"
    assert _role_of(tab, "Beads_unstained.fcs") == "Unstained (beads)"
    assert _role_of(tab, "Beads_PE_X.fcs") == "Single color (beads)"
    assert tab.unmixer is not None and len(tab.unmixer.names) == 13
    assert "Complexity index" in tab.status.text()
    assert tab.preview_sample.currentText() == "PBMC_full_panel.fcs"

    tab.method.setCurrentIndex(tab.method.findData("ols"))
    assert tab.unmixer.method == "ols" and tab.setup.method == "ols"

    with qtbot.waitSignal(tab.unmixed, timeout=120000) as unmixed:
        tab.unmix_into_workspace()
    ids = unmixed.args[0]
    assert window.pages.currentWidget() is window.panel
    assert window.panel.current_group() == "Unmixed"
    assert window.ws.groups["Unmixed"] == ["PBMC_full_panel.fcs (unmixed)"]
    assert len(window.ws.groups["Reference Group"]) == 15 and len(ids) == 16
    assert window.graph.sample_id == ids[0] or window.graph.sample_id in ids

    result = tab.export(tmp_path, "Demo", True).wait()
    assert result.path == tmp_path / "Demo.zip" and result.n_samples == 1

    # The setup and the unmixed samples survive saving and reopening the workspace.
    path = tmp_path / "ws.ssw"
    window.path = str(path)
    window.save()
    wait_idle()
    back = Workspace.load(path)
    assert back.experiments[0]["files"][0]["path"].endswith(".fcs")
    assert "PBMC_full_panel.fcs (unmixed)" in back.samples


def test_changing_roles_on_several_rows(qtbot):
    tab = UnmixingTab(Workspace())
    qtbot.addWidget(tab)
    tab.add_files(_demo_files())
    wait_idle()
    rows = [i for i, s in enumerate(tab.specs) if s.path.name.startswith("Beads_B")]
    tab._select_rows(rows)
    exclude = [label for label, _, _ in ROLE_CHOICES].index("Exclude")
    tab.table.cellWidget(rows[0], 1).setCurrentIndex(exclude)
    assert all(tab.specs[r].role == "exclude" for r in rows)
    assert len(tab.unmixer.names) == 13 - len(rows)
    assert all(tab.table.cellWidget(r, 1).currentText() == "Exclude" for r in rows)
    other = next(i for i, s in enumerate(tab.specs) if s.path.name == "Beads_PE_X.fcs")
    assert tab.specs[other].role == "single_stain"

    # Giving two controls the same fluor blocks unmixing until it's fixed.
    tab.table.item(other, 2).setText("FITC")
    assert tab.unmixer is None and not tab.export_btn.isEnabled()
    assert "Two single-stain controls for FITC" in tab.status.text()
    tab.table.item(other, 2).setText("PE")
    assert tab.unmixer is not None


def test_drop_on_unmixing_tab_adds_files(window, qtbot, tmp_path):
    window.open_unmixing()
    before = len(window.ws.samples)
    target = window.unmix_tab.table.viewport()
    _drop(qtbot, target, [DEMO_DIR / "controls"])
    wait_idle()
    assert len(window.unmix_tab.specs) == 15
    assert len(window.ws.samples) == before  # raw files stay in the tab


def test_several_experiments(qtbot):
    tab = UnmixingTab(Workspace())
    qtbot.addWidget(tab)
    tab.add_files(_demo_files()[:3])
    wait_idle()
    tab.new_setup("Panel B")
    assert tab.specs == [] and tab.exp_combo.count() == 2
    tab.add_files(_demo_files()[3:5])
    wait_idle()
    tab.exp_combo.setCurrentIndex(0)
    assert len(tab.specs) == 3
    assert [d["name"] for d in tab.ws.experiments] == ["Experiment 1", "Panel B"]


def test_gates_show_in_tree_and_table(window, qtbot):
    ws = window.ws
    sid = "PBMC_full_panel.fcs"
    x = ws.display_values(sid, "FSC-A")
    ws.add_gate(fk.gates.RectangleGate("Big", [ws.make_dimension(sid, "FSC-A", float(np.median(x)), None),
                                                ws.make_dimension(sid, "SSC-A", None, None)]))
    window._changed()
    names = [window.panel.tree.topLevelItem(0).child(i).text(0)
             for i in range(window.panel.tree.topLevelItem(0).childCount())]
    assert names == ["Big"]

    window.open_graph(sid)
    window.graph.set_axes("FSC-A", "SSC-A")
    assert window.graph._rois  # the gate is drawn

    editor = TableEditor(ws, [sid], window)
    qtbot.addWidget(editor)
    editor.pop_combo.setCurrentIndex(1)
    editor.stat_combo.setCurrentText("Freq. of Parent")
    editor._add_column()
    wait_idle()
    assert editor.table_df.iloc[0, 1] == pytest.approx(50.0, abs=0.1)


def _drop(qtbot, target, paths):
    from PySide6 import QtCore, QtGui, QtWidgets
    mime = QtCore.QMimeData()
    mime.setUrls([QtCore.QUrl.fromLocalFile(str(p)) for p in paths])
    pos = QtCore.QPointF(target.rect().center())
    enter = QtGui.QDragEnterEvent(pos.toPoint(), QtCore.Qt.CopyAction, mime,
                                  QtCore.Qt.LeftButton, QtCore.Qt.NoModifier)
    QtWidgets.QApplication.sendEvent(target, enter)
    assert enter.isAccepted()
    drop = QtGui.QDropEvent(pos, QtCore.Qt.CopyAction, mime, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier)
    QtWidgets.QApplication.sendEvent(target, drop)


def test_drop_files_and_folders_adds_samples(window, qtbot, tmp_path, monkeypatch):
    import shutil
    from PySide6 import QtWidgets
    warnings = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *a, **k: warnings.append(a[2]))

    folder = tmp_path / "plate" / "row A"
    folder.mkdir(parents=True)
    controls = sorted((DEMO_DIR / "controls").glob("*.fcs"))[:2]
    shutil.copy(controls[0], folder / "a.fcs")
    shutil.copy(controls[1], folder / "b.FCS")
    (folder / "notes.txt").write_text("not data")
    loose = tmp_path / "loose.fcs"
    shutil.copy(controls[0], loose)
    other = tmp_path / "readme.csv"
    other.write_text("x")

    before = len(window.ws.samples)
    _drop(qtbot, window, [tmp_path / "plate", loose, other])
    wait_idle()
    assert len(window.ws.samples) == before + 3
    assert {"a.fcs", "b.FCS", "loose.fcs"} <= set(window.ws.samples)
    assert window.dirty
    assert len(warnings) == 1 and "readme.csv" in warnings[0]


def test_drop_on_plot_reaches_window(window, qtbot, tmp_path, monkeypatch):
    import shutil
    from PySide6 import QtWidgets
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *a, **k: None)
    f = tmp_path / "one.fcs"
    shutil.copy(sorted((DEMO_DIR / "controls").glob("*.fcs"))[0], f)
    before = len(window.ws.samples)
    window.open_graph("PBMC_full_panel.fcs")
    target = window.graph.plot.viewport()  # QGraphicsView accepts drops itself, then passes them up
    assert target.acceptDrops()
    _drop(qtbot, target, [f])
    wait_idle()
    assert len(window.ws.samples) == before + 1


def test_drop_without_fcs_only_warns(window, qtbot, tmp_path, monkeypatch):
    from PySide6 import QtWidgets
    warnings = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *a, **k: warnings.append(a[1]))
    (tmp_path / "empty").mkdir()
    before = len(window.ws.samples)
    window.add_dropped([tmp_path / "empty"])
    wait_idle()
    assert len(window.ws.samples) == before and warnings == ["No FCS files"]


def test_unmixing_steps_and_identify_rows(qtbot):
    tab = UnmixingTab(Workspace())
    qtbot.addWidget(tab)
    tab.add_files(_demo_files())
    wait_idle()
    assert tab.step == 0 and not tab.rows
    tab.next_btn.click()
    assert tab.step == 1
    names = [row.title.text() for row in tab.rows]
    assert len(names) == 15 and names[0] == "AF647 (Beads)" and names[-1].startswith("Unstained")
    row = tab.rows[0]
    assert row.spectrum.counts is not None and row.spectrum.counts.shape[0] == len(tab.unmixer.detectors)
    assert "Positive" in row._regions

    # Dragging the Positive range rebuilds the matrix without rebuilding the rows.
    region = row._regions["Positive"]
    a, b = region.getRegion()
    region.setRegion((a + (b - a) * 0.5, b))
    region.sigRegionChangeFinished.emit(region)
    qtbot.waitUntil(lambda: row.spec.positive_range is not None)
    qtbot.wait(20)
    assert tab.rows[0] is row
    result = next(r for r in tab.results if r.spec is row.spec)
    assert result.peak_detector == row.spec.gate_detector
    assert tab.setup.to_dict()["files"][0]["positive_range"]

    tab.next_btn.click()
    assert tab.step == 2 and tab.next_btn.isHidden()
    assert len(tab.signatures.references) == len(tab.unmixer.names)
    tab.signatures.each.click()
    assert tab.signatures.stack.currentIndex() == 1 and len(tab.signatures.small) == len(tab.unmixer.names)


def test_spectrum_plot_type_in_graph_window(window, qtbot):
    from sidescatter.app.graph_view import GraphView
    v = GraphView(window.ws)
    qtbot.addWidget(v)
    v.show_population("PBMC_full_panel.fcs")
    v.type_combo.setCurrentText("Spectrum")
    wait_idle()
    assert v.spectrum.counts is not None
    assert v.spectrum.det_axis.names[:2] == ["V1", "V2"]
    # Every detector gets its own label.
    ticks = v.spectrum.det_axis.tickValues(0, 1, 100)
    assert len(ticks) == 1 and len(ticks[0][1]) == len(v.spectrum.det_axis.names) == 42
    assert not v.tools["polygon"].isEnabled()
    v.start_gate("polygon")
    assert v.type_combo.currentText() == "Pseudocolor"


def test_per_cell_autofluorescence_in_the_tab(qtbot):
    tab = UnmixingTab(Workspace())
    qtbot.addWidget(tab)
    tab.resize(1400, 900)
    tab.show()
    tab.add_files(_demo_files())
    wait_idle()
    assert tab.setup.af_mode == "per_cell" and tab.unmixer.per_cell_af
    assert tab.af_lib_size.isVisibleTo(tab) and not tab.af_count.isVisibleTo(tab)
    n_lib = len(tab.unmixer.af_library)

    # Draw an AF population on the unstained cells with the mouse.
    tab.tabs.setCurrentWidget(tab.af_view)
    view = tab.af_view
    assert view.file_combo.count() == 2  # PBMC and dead-cell unstained cells
    view.file_combo.setCurrentText("PBMC_unstained.fcs")
    assert view.spec.path.name == "PBMC_unstained.fcs" and view.draw_btn.isEnabled()
    view.draw_btn.setChecked(True)
    vb = view.plot.getViewBox()
    (x0, x1), (y0, y1) = vb.viewRange()
    corners = [(x0 + (x1 - x0) * fx, y0 + (y1 - y0) * fy) for fx, fy in ((0.5, 0.5), (0.95, 0.5), (0.95, 0.95))]
    for cx, cy in corners:
        pos = view.plot.mapFromScene(vb.mapViewToScene(QtCore.QPointF(cx, cy)))
        qtbot.mouseClick(view.plot.viewport(), QtCore.Qt.LeftButton, pos=pos)
    qtbot.mouseClick(view.plot.viewport(), QtCore.Qt.RightButton, pos=pos)
    assert len(view.spec.af_gates) == 1 and not view.draw_btn.isChecked()
    assert view.gate_list.count() == 1
    assert any(f["af_gates"] for f in tab.ws.experiments[0]["files"])

    # Only the drawn population (the dead-cell unstained has no gates, so it keeps its median).
    tab.af_auto.setChecked(False)
    assert len(tab.unmixer.af_library) == 2 < n_lib

    tab.tabs.setCurrentWidget(tab.af_check_view)
    wait_idle()
    assert "less spread" in tab.af_check_view.summary.text()

    tab.af_mode.setCurrentIndex(tab.af_mode.findData("fluors"))
    assert not tab.unmixer.per_cell_af and tab.af_count.isVisibleTo(tab)
    tab.af_mode.setCurrentIndex(tab.af_mode.findData("per_cell"))
    tab.af_auto.setChecked(True)
    with qtbot.waitSignal(tab.unmixed, timeout=120000) as unmixed:
        tab.unmix_into_workspace()
    ids = unmixed.args[0]
    sample = tab.ws.samples[next(i for i in ids if i.startswith("PBMC_full"))]
    assert "AF Index" in sample.pnn_labels and "AF-A" in sample.pnn_labels
