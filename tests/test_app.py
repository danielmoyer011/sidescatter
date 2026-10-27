"""Smoke tests for the desktop UI (run headless with QT_QPA_PLATFORM=offscreen)."""

import flowkit as fk
import numpy as np
import pytest

pytest.importorskip("pytestqt")

from sidescatter.app.main_window import DEMO_DIR, MainWindow  # noqa: E402
from sidescatter.app.table_editor import TableEditor  # noqa: E402
from sidescatter.app.unmixing_tab import ROLE_CHOICES, UnmixingTab  # noqa: E402
from sidescatter.core.workspace import Workspace  # noqa: E402


@pytest.fixture
def window(qtbot):
    w = MainWindow()
    w.ask_before_discard = False
    qtbot.addWidget(w)
    w.add_samples([str(DEMO_DIR / "PBMC_full_panel.fcs")])
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

    ids = tab.unmix_into_workspace()
    assert window.pages.currentWidget() is window.panel
    assert window.panel.current_group() == "Unmixed"
    assert window.ws.groups["Unmixed"] == ["PBMC_full_panel.fcs (unmixed)"]
    assert len(window.ws.groups["Reference Group"]) == 15 and len(ids) == 16
    assert window.graph.sample_id == ids[0] or window.graph.sample_id in ids

    result = tab.export(tmp_path, "Demo", True)
    assert result.path == tmp_path / "Demo.zip" and result.n_samples == 1

    # The setup and the unmixed samples survive saving and reopening the workspace.
    path = tmp_path / "ws.ssw"
    window.path = str(path)
    window.save()
    back = Workspace.load(path)
    assert back.experiments[0]["files"][0]["path"].endswith(".fcs")
    assert "PBMC_full_panel.fcs (unmixed)" in back.samples


def test_changing_roles_on_several_rows(qtbot):
    tab = UnmixingTab(Workspace())
    qtbot.addWidget(tab)
    tab.add_files(_demo_files())
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
    assert len(window.unmix_tab.specs) == 15
    assert len(window.ws.samples) == before  # raw files stay in the tab


def test_several_experiments(qtbot):
    tab = UnmixingTab(Workspace())
    qtbot.addWidget(tab)
    tab.add_files(_demo_files()[:3])
    tab.new_setup("Panel B")
    assert tab.specs == [] and tab.exp_combo.count() == 2
    tab.add_files(_demo_files()[3:5])
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
    window.open_graph("PBMC_full_panel.fcs")
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
