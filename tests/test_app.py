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
