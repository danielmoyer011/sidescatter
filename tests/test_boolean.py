"""Boolean gates drawn in Sidescatter: AND/OR/NOT, all combinations, placement, delete, save."""

import flowkit as fk
import numpy as np
import pytest

from sidescatter.core.workspace import ROOT, Workspace, boolean_name, combination_name

SID = "PBMC_full_panel.fcs"
CELLS = ("root", "Cells")


@pytest.fixture
def ws(spectral_dir):
    """Cells, then three marker gates inside Cells, and Treg-like gate inside A+."""
    ws = Workspace()
    ws.add_fcs([spectral_dir / SID])

    def above(name, ch, q, parent):
        lo = float(np.quantile(ws.display_values(SID, ch), q))
        ws.add_gate(fk.gates.RectangleGate(name, [ws.make_dimension(SID, ch, lo, None)]), parent)

    above("Cells", "FSC-A", 0.05, ROOT)
    above("A+", "V7-A", 0.5, CELLS)
    above("B+", "B2-A", 0.6, CELLS)
    above("C+", "R1-A", 0.4, CELLS)
    above("Sub", "SSC-A", 0.5, CELLS + ("A+",))
    return ws


def _mask(ws, name, parent=ROOT):
    return ws.membership(SID, name, parent)


def test_and_or_not_match_numpy(ws):
    cells, a, b = _mask(ws, "Cells"), _mask(ws, "A+", CELLS), _mask(ws, "B+", CELLS)
    name, parent = ws.add_boolean([("A+", CELLS, False), ("B+", CELLS, True)], "and")
    assert (name, parent) == ("A+ AND NOT B+", CELLS)
    assert np.array_equal(_mask(ws, name, parent), cells & a & ~b)

    name, parent = ws.add_boolean([("A+", CELLS, False), ("B+", CELLS, False)], "or")
    assert name == "A+ OR B+"
    assert np.array_equal(_mask(ws, name, parent), cells & (a | b))

    name, parent = ws.add_boolean([("A+", CELLS, True)])
    assert name == "NOT A+"
    assert np.array_equal(_mask(ws, name, parent), cells & ~a)
    assert ws.is_boolean(name, parent)
    assert ws.boolean_inputs(name, parent) == ("and", [("A+", CELLS, True)])


def test_inputs_from_different_branches_go_under_shared_parent(ws):
    sub = ("Sub", CELLS + ("A+",), False)
    assert ws.boolean_parent([sub, ("B+", CELLS, False)]) == CELLS
    assert ws.boolean_parent([("Cells", ROOT, False), ("B+", CELLS, False)]) == ROOT
    name, parent = ws.add_boolean([sub, ("B+", CELLS, False)], "and")
    assert parent == CELLS
    expected = _mask(ws, "Sub", CELLS + ("A+",)) & _mask(ws, "B+", CELLS)
    assert np.array_equal(_mask(ws, name, parent), expected)


def test_bad_inputs_are_refused(ws):
    with pytest.raises(ValueError):
        ws.add_boolean([("A+", CELLS, False)])  # one input needs NOT
    with pytest.raises(ValueError):
        ws.add_boolean([("A+", CELLS, False), ("A+", CELLS, True)])
    ws.add_boolean([("A+", CELLS, False), ("B+", CELLS, False)])
    with pytest.raises(ValueError):  # same name twice in one place
        ws.add_boolean([("A+", CELLS, False), ("B+", CELLS, False)])


def test_all_combinations(ws):
    inputs = [("A+", CELLS), ("B+", CELLS), ("C+", CELLS)]
    made = ws.boolean_combinations(inputs)
    names = [n for n, _ in made]
    assert len(made) == 7 and "A- B- C-" not in names
    assert names[0] == "A+ B+ C+"
    a, b, c = (_mask(ws, n, CELLS) for n in ("A+", "B+", "C+"))
    cells = _mask(ws, "Cells")
    assert np.array_equal(_mask(ws, "A+ B- C+", CELLS), cells & a & ~b & c)
    # the combinations split Cells without overlap
    total = sum(ws.count(SID, n, CELLS) for n in names) + int((cells & ~a & ~b & ~c).sum())
    assert total == int(cells.sum())

    made = ws.boolean_combinations(inputs, include_all_negative=True)
    assert len(made) == 8 and made[0][0] == "A+ B+ C+ (2)"
    assert "A- B- C-" in [n for n, _ in made]


def test_names():
    assert boolean_name([("CD4+", (), False), ("Treg", (), True)]) == "CD4+ AND NOT Treg"
    assert boolean_name([("Dump", (), True)]) == "NOT Dump"
    assert combination_name([("IFNg+", (), False), ("TNF", (), True), ("IL-2-", (), False)]) \
        == "IFNg+ TNF- IL-2+"


def test_gating_inside_a_boolean_population(ws):
    name, parent = ws.add_boolean([("A+", CELLS, False), ("B+", CELLS, True)])
    inside = parent + (name,)
    lo = float(np.median(ws.display_values(SID, "SSC-A")))
    ws.add_gate(fk.gates.RectangleGate("Big", [ws.make_dimension(SID, "SSC-A", lo, None)]), inside)
    assert ws.count(SID, "Big", inside) < ws.count(SID, name, parent)
    assert ("Big", inside) in ws.population_paths()


def test_edit_and_rename(ws):
    name, parent = ws.add_boolean([("A+", CELLS, False), ("B+", CELLS, False)])
    name, parent = ws.edit_boolean(name, parent, [("A+", CELLS, False), ("C+", CELLS, True)], "or",
                                   new_name="A or not C")
    assert (name, parent) == ("A or not C", CELLS)
    a, c = _mask(ws, "A+", CELLS), _mask(ws, "C+", CELLS)
    assert np.array_equal(_mask(ws, name, parent), _mask(ws, "Cells") & (a | ~c))

    # renaming an input keeps the Boolean gate pointing at it
    ws.rename_gate("C+", "CD-C+", CELLS)
    assert ws.boolean_inputs(name, parent)[1][1][0] == "CD-C+"
    ws.count(SID, name, parent)

    # moving to a new shared parent
    sub = ("Sub", CELLS + ("A+",), False)
    name, parent = ws.edit_boolean(name, parent, [sub, ("Cells", ROOT, True)], "or")
    assert parent == ROOT

    # it can't use itself
    with pytest.raises(ValueError):
        ws.edit_boolean(name, parent, [("A+", CELLS, False), (name, parent, False)], "and")


def test_deleting_an_input_removes_its_boolean_gates(ws):
    b1 = ws.add_boolean([("Sub", CELLS + ("A+",), False), ("B+", CELLS, False)])
    b2 = ws.add_boolean([(b1[0], b1[1], False), ("C+", CELLS, False)])
    keep = ws.add_boolean([("B+", CELLS, False), ("C+", CELLS, False)], "or")
    assert set(ws.boolean_dependents("A+", CELLS)) == {b1, b2}
    ws.remove_gate("A+", CELLS)
    paths = ws.population_paths()
    assert b1 not in paths and b2 not in paths and keep in paths
    ws.count(SID, *keep)


def test_plot_axes_follow_first_input(ws):
    b1 = ws.add_boolean([("A+", CELLS, False), ("B+", CELLS, False)])
    b2 = ws.add_boolean([(b1[0], b1[1], True), ("C+", CELLS, False)])
    assert ws.plot_axes(*b2) == ["V7-A"]


def test_boolean_gates_apply_to_every_sample(ws, spectral_dir):
    [other] = ws.add_fcs([spectral_dir / SID])
    name, parent = ws.add_boolean([("A+", CELLS, False), ("B+", CELLS, True)])
    assert ws.applies_to(other, name, parent)
    assert ws.count(other, name, parent) == ws.count(SID, name, parent)


def test_boolean_on_a_gate_a_sample_lacks(ws, spectral_dir):
    """A raw spectral file has no unmixed channels: Boolean gates on them don't apply."""
    from sidescatter.core import unmixing

    specs = unmixing.guess_controls(sorted((spectral_dir / "controls").glob("*.fcs")))
    um, _ = unmixing.build_unmixer(specs, method="wls")
    ws.add_unmixer("Aurora", um)
    unmixed = ws.unmix(SID, "Aurora")
    ch = next(c.pnn for c in ws.channels(unmixed) if c.pnn not in ws.samples[SID].pnn_labels
              and c.kind == "fluorescence")
    lo = float(np.median(ws.display_values(unmixed, ch)))
    ws.add_gate(fk.gates.RectangleGate("U+", [ws.make_dimension(unmixed, ch, lo, None)]))
    name, parent = ws.add_boolean([("U+", ROOT, False), ("Cells", ROOT, False)])
    assert ws.applies_to(unmixed, name, parent) and not ws.applies_to(SID, name, parent)
    assert ws.count(unmixed, name, parent) > 0
    ws.count(SID, "Cells")  # the raw file still gates fine without them


def test_save_and_load(ws, tmp_path):
    b1 = ws.add_boolean([("Sub", CELLS + ("A+",), False), ("B+", CELLS, True)])
    ws.boolean_combinations([("A+", CELLS), ("C+", CELLS)])
    before = {p: ws.count(SID, *p) for p in ws.population_paths()}
    ws.save(tmp_path / "w.ssw")
    back = Workspace.load(tmp_path / "w.ssw")
    assert {p: back.count(SID, *p) for p in back.population_paths()} == before
    assert back.boolean_inputs(*b1) == ws.boolean_inputs(*b1)


# ----- the editor and the sample table --------------------------------------

def test_editor_and_tree(qtbot, ws, monkeypatch):
    pytest.importorskip("pytestqt")
    from PySide6 import QtWidgets

    from sidescatter.app.boolean_editor import BooleanDialog
    from sidescatter.app.main_window import MainWindow

    win = MainWindow()
    win.ask_before_discard = False
    qtbot.addWidget(win)
    win._set_workspace(ws)

    dlg = BooleanDialog(ws, [("A+", CELLS), ("B+", CELLS)], sample_id=SID)
    qtbot.addWidget(dlg)
    assert dlg.name_edit.text() == "A+ AND B+"
    dlg.set_negated(1, True)
    assert dlg.name_edit.text() == "A+ AND NOT B+"
    assert "Goes under Cells" in dlg.placement.text()
    expected = int((_mask(ws, "Cells") & _mask(ws, "A+", CELLS) & ~_mask(ws, "B+", CELLS)).sum())
    assert f"{expected:,} events" in dlg.placement.text()
    dlg.or_button.setChecked(True)
    assert dlg.name_edit.text() == "A+ OR NOT B+"
    dlg.apply()
    assert dlg.made == [("A+ OR NOT B+", CELLS)]

    # it shows in the tree with its own icon and the usual statistics
    win.panel.rebuild()
    items = [it for it in win.panel._items() if it.text(0) == "A+ OR NOT B+"]
    assert items and not items[0].icon(0).isNull()
    assert items[0].text(2) == f"{ws.count(SID, 'A+ OR NOT B+', CELLS):,}"

    # editing reopens with its inputs
    edit = BooleanDialog(ws, editing=("A+ OR NOT B+", CELLS))
    qtbot.addWidget(edit)
    assert edit.inputs() == [("A+", CELLS, False), ("B+", CELLS, True)] and edit.or_button.isChecked()
    assert not edit.combos_box.isVisibleTo(edit)
    edit.set_negated(1, False)
    edit.name_edit.setText("A or B")
    edit.apply()
    assert edit.made == [("A or B", CELLS)]

    # selecting populations in the table fills the editor; all combinations from there
    for it in win.panel._items():
        it.setSelected(it.text(0) in ("A+", "B+", "C+"))
    made = {}

    def fake_exec(self):
        assert [k[0] for k in self.inputs()] == ["A+", "B+", "C+"]
        self.all_negative.setChecked(True)
        self.make_combinations()
        made["n"] = len(self.made)
        return QtWidgets.QDialog.Accepted

    monkeypatch.setattr(BooleanDialog, "exec", fake_exec)
    win.acts["boolean"].trigger()
    assert made["n"] == 8

    # double-clicking opens it on its first input's axes, and it can be gated further
    v = win.open_graph(SID, "A or B", CELLS)
    assert v.x_combo.currentData() == "V7-A"
    assert v.count_label.text() == f"{ws.count(SID, 'A or B', CELLS):,} events"
    v.set_axes("FSC-A", "SSC-A")
    assert v.create_gate("rectangle", [(0.2, 0.2), (0.8, 0.8)], name="Inside")
    assert ("Inside", CELLS + ("A or B",)) in ws.population_paths()


def test_delete_warns_about_boolean_gates(qtbot, ws, monkeypatch):
    from PySide6 import QtWidgets

    from sidescatter.app.workspace_panel import WorkspacePanel

    panel = WorkspacePanel(ws)
    qtbot.addWidget(panel)
    ws.add_boolean([("A+", CELLS, False), ("B+", CELLS, True)])
    asked = {}

    def fake_question(parent, title, text):
        asked["text"] = text
        return QtWidgets.QMessageBox.Yes

    monkeypatch.setattr(QtWidgets.QMessageBox, "question", fake_question)
    panel._delete("B+", CELLS)
    assert "A+ AND NOT B+" in asked["text"]
    assert ("A+ AND NOT B+", CELLS) not in ws.population_paths()
