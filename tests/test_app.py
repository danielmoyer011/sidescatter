"""Smoke tests for the desktop UI (run headless with QT_QPA_PLATFORM=offscreen)."""

import flowkit as fk
import numpy as np
import pytest

pytest.importorskip("pytestqt")

from sidescatter.app.main_window import DEMO_DIR, MainWindow  # noqa: E402
from sidescatter.app.table_editor import TableEditor  # noqa: E402
from sidescatter.app.unmixing_dialog import UnmixingDialog  # noqa: E402


@pytest.fixture
def window(qtbot):
    w = MainWindow()
    w.ask_before_discard = False
    qtbot.addWidget(w)
    w.add_samples([str(DEMO_DIR / "PBMC_full_panel.fcs")])
    return w


def test_unmixing_dialog_builds_and_unmixes(window, qtbot):
    controls = sorted(str(p) for p in (DEMO_DIR / "controls").glob("*.fcs"))
    dlg = UnmixingDialog(window.ws, window, control_paths=controls)
    qtbot.addWidget(dlg)
    assert dlg.unmixer is not None and len(dlg.unmixer.names) == 13
    assert "Complexity index" in dlg.complexity.text()

    dlg.method.setCurrentIndex(dlg.method.findData("ols"))
    assert dlg.unmixer.method == "ols"

    window.ws.add_unmixer("Unmixing 1", dlg.unmixer)
    sid = window.ws.unmix("PBMC_full_panel.fcs", "Unmixing 1", group="Unmixed")
    window._after_unmix([sid])
    assert window.panel.current_group() == "Unmixed"
    assert window.graph.sample_id == sid


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

    window.graph.show_population(sid, None)
    window.graph.set_axes("FSC-A", "SSC-A")
    assert window.graph._rois  # the gate is drawn

    editor = TableEditor(ws, [sid], window)
    qtbot.addWidget(editor)
    editor.pop_combo.setCurrentIndex(1)
    editor.stat_combo.setCurrentText("Freq. of Parent")
    editor._add_column()
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
    target = window.graph.plot.viewport()  # QGraphicsView accepts drops itself, then passes them up
    assert target.acceptDrops()
    _drop(qtbot, target, [f])
    assert len(window.ws.samples) == before + 1


def test_drop_without_fcs_only_warns(window, qtbot, tmp_path, monkeypatch):
    from PySide6 import QtWidgets
    warnings = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *a, **k: warnings.append(a[1]))
    (tmp_path / "empty").mkdir()
    before = len(window.ws.samples)
    window.add_dropped([tmp_path / "empty"])
    assert len(window.ws.samples) == before and warnings == ["No FCS files"]
