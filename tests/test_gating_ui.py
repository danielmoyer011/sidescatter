"""The FlowJo-style layout: ribbon, workspace tables, graph windows and gating tools."""

import flowkit as fk
import numpy as np
import pytest

pytest.importorskip("pytestqt")

from PySide6 import QtCore  # noqa: E402

from sidescatter.app import graph_view  # noqa: E402
from sidescatter.app.jobs import wait_idle  # noqa: E402
from sidescatter.app.main_window import DEMO_DIR, LicenseDialog, MainWindow  # noqa: E402
from sidescatter.core.workspace import ROOT, Workspace  # noqa: E402

SID = "PBMC_full_panel.fcs"


@pytest.fixture
def window(qtbot):
    w = MainWindow()
    w.ask_before_discard = False
    qtbot.addWidget(w)
    w.add_samples([str(DEMO_DIR / SID)])
    wait_idle()
    return w


@pytest.fixture
def view(window):
    v = window.open_graph(SID)
    v.set_axes("FSC-A", "SSC-A")
    return v


def _freq(ws, name, path=ROOT):
    return ws.statistic(SID, "Freq. of Parent", name, path)


def test_ribbon_and_workspace_tables(window):
    tabs = [window.ribbon.tabs.tabText(i) for i in range(window.ribbon.tabs.count())]
    assert tabs == ["Sidescatter", "File", "Edit", "Workspace", "Tools", "Configure"]
    tree = window.panel.tree
    assert [tree.headerItem().text(i) for i in range(3)] == ["Name", "Statistic", "#Cells"]
    groups = window.panel.groups
    assert groups.topLevelItem(0).text(0) == "All Samples"
    assert groups.topLevelItem(0).text(1) == "1"
    assert tree.topLevelItem(0).text(2) == f"{window.ws.samples[SID].event_count:,}"


def test_double_click_opens_one_graph_window_per_population(window):
    item = window.panel.tree.topLevelItem(0)
    window.panel.tree.itemDoubleClicked.emit(item, 0)
    assert len(window.graphs) == 1 and window.graph.sample_id == SID
    window.panel.tree.itemDoubleClicked.emit(item, 0)
    assert len(window.graphs) == 1  # brought forward, not duplicated
    assert window.graph.tools["rectangle"].isEnabled()
    window.graphs[0].close()
    assert window.graphs == [] and window.graph is None


def test_rectangle_polygon_ellipse_and_freehand(window, view):
    ws = window.ws
    assert view.create_gate("rectangle", [[0.1, 0.1], [0.6, 0.6]], name="Rect") == ["Rect"]
    assert view.create_gate("polygon", [[0.1, 0.1], [0.6, 0.1], [0.35, 0.6]], name="Tri") == ["Tri"]
    assert view.create_gate("ellipse", [[0.1, 0.1], [0.6, 0.6]], name="Oval") == ["Oval"]
    trace = [[0.35 + 0.25 * np.cos(t), 0.35 + 0.25 * np.sin(t)] for t in np.linspace(0, 6.2, 200)]
    assert view.create_gate("freehand", trace, name="Free") == ["Free"]

    rect, oval, free = (ws.count(SID, n) for n in ("Rect", "Oval", "Free"))
    assert 0 < oval < rect  # the inscribed ellipse holds fewer events than its box
    assert free == pytest.approx(oval, rel=0.05)  # a traced circle ~ the same ellipse
    assert len(ws.get_gate("Free", ROOT).vertices) < 200  # the trace is thinned
    assert len(view._rois) == 4


def test_quadrant_gate_makes_four_populations_and_moves_together(window, view):
    ws = window.ws
    made = view.create_gate("quadrant", [[0.4, 0.3]])
    assert [m.split(":")[0] for m in made] == ["Q1", "Q2", "Q3", "Q4"]
    assert sum(_freq(ws, m) for m in made) == pytest.approx(100.0)
    lines = [r for r in view._rois if isinstance(r, graph_view.pg.InfiniteLine)]
    assert len(lines) == 2  # one crosshair, not four rectangles

    before = ws.count(SID, made[1])
    lines[0].setValue(0.5)
    lines[0].sigPositionChangeFinished.emit(lines[0])
    assert all(ws.get_gate(m, ROOT).dimensions[0].min in (None, 0.5) for m in made)
    assert all(ws.get_gate(m, ROOT).dimensions[0].max in (None, 0.5) for m in made)
    assert ws.count(SID, made[1]) < before
    assert sum(_freq(ws, m) for m in made) == pytest.approx(100.0)

    # a second quadrant on the same plot gets its own names
    again = view.create_gate("quadrant", [[0.6, 0.6]])
    assert len(set(again) | set(made)) == 8


def test_range_and_bisector_on_histogram(window, view):
    ws = window.ws
    view.set_axes("FSC-A", "SSC-A")
    view.start_gate("bisector")
    assert view.is_histogram and view.tools["bisector"].isChecked()
    assert not view.tools["rectangle"].isEnabled()
    low, high = view.create_gate("bisector", [[0.3, 0.0]])
    assert _freq(ws, low) + _freq(ws, high) == pytest.approx(100.0)
    assert view.create_gate("range", [[0.2, 0], [0.5, 0]], name="Mid") == ["Mid"]
    assert 0 < ws.count(SID, "Mid") < ws.samples[SID].event_count


def test_drawing_a_rectangle_with_the_mouse(window, view, qtbot, monkeypatch):
    from PySide6 import QtWidgets
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", lambda *a, **k: ("Drawn", True))
    view.parentWidget().show()
    qtbot.waitExposed(view.parentWidget())
    view.start_gate("rectangle")
    vb = view.plot.getViewBox()

    def at(x, y):
        return view.plot.mapFromScene(vb.mapViewToScene(QtCore.QPointF(x, y)))

    vp = view.plot.viewport()
    qtbot.mousePress(vp, QtCore.Qt.LeftButton, pos=at(0.2, 0.2))
    qtbot.mouseMove(vp, at(0.5, 0.5))
    qtbot.mouseRelease(vp, QtCore.Qt.LeftButton, pos=at(0.6, 0.6))
    gate = window.ws.get_gate("Drawn", ROOT)
    (x0, x1), (y0, y1) = ((d.min, d.max) for d in gate.dimensions)
    assert (x0, y0) == pytest.approx((0.2, 0.2), abs=0.02)
    assert (x1, y1) == pytest.approx((0.6, 0.6), abs=0.02)
    assert view._draw_mode is None and view.tools["pointer"].isChecked()
    assert view.gate_at(0.4, 0.4) == ("Drawn", ROOT)
    assert view.gate_at(0.9, 0.9) is None


def test_ellipse_roundtrip_and_save(window, view, tmp_path):
    mean, cov = graph_view.ellipse_gate_params((0.4, 0.5), 0.2, 0.05, np.radians(30))
    gate = fk.gates.EllipsoidGate("E", [window.ws.make_dimension(SID, "FSC-A"),
                                        window.ws.make_dimension(SID, "SSC-A")], mean, cov, 1.0)
    (cx, cy), a, b, angle = graph_view.ellipse_shape(gate)
    assert (cx, cy, a, b) == pytest.approx((0.4, 0.5, 0.2, 0.05))
    assert np.degrees(angle) % 180 == pytest.approx(30)

    view.create_gate("ellipse", [[0.2, 0.2], [0.6, 0.5]], name="Oval")
    count = window.ws.count(SID, "Oval")
    path = tmp_path / "w.ssw"
    window.ws.save(path)
    assert Workspace.load(path).count(SID, "Oval") == count


def test_step_through_samples_keeps_population(window, view):
    window.add_samples([str(DEMO_DIR / SID)])  # a second copy
    wait_idle()
    view.create_gate("rectangle", [[0.1, 0.1], [0.6, 0.6]], name="Rect")
    view.show_population(SID, "Rect", ROOT)
    view.step_sample(+1)
    assert view.sample_id != SID and view.population == ("Rect", ROOT)


def test_license_dialog(window, qtbot):
    from PySide6 import QtWidgets
    dlg = LicenseDialog(window)
    qtbot.addWidget(dlg)
    texts = [lbl.text() for lbl in dlg.findChildren(QtWidgets.QLabel)]
    assert "No License, No Problem" in texts
    assert window.license_button.text() == "License…"


def test_changing_scale_keeps_ellipse_on_same_events(window, view):
    view.create_gate("ellipse", [[0.2, 0.2], [0.6, 0.5]], name="Oval")
    before = window.ws.count(SID, "Oval")
    window.ws.set_transform("FSC-A", fk.transforms.LinearTransform(param_t=262144 * 16, param_a=0))
    assert window.ws.count(SID, "Oval") == pytest.approx(before, rel=0.02)


@pytest.fixture
def drawing(window, view, qtbot, monkeypatch):
    """Start a polygon on a shown graph window; returns (at, viewport)."""
    from PySide6 import QtWidgets
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", lambda *a, **k: ("Poly", True))
    view.parentWidget().show()
    qtbot.waitExposed(view.parentWidget())
    view.start_gate("polygon")
    vb = view.plot.getViewBox()

    def at(x, y):
        return view.plot.mapFromScene(vb.mapViewToScene(QtCore.QPointF(x, y)))

    return at, view.plot.viewport()


def _hover(vp, pos):
    """Send a plain mouse move; qtbot.mouseMove is unreliable once another test's window closed."""
    from PySide6 import QtGui, QtWidgets
    pos = QtCore.QPointF(pos)
    ev = QtGui.QMouseEvent(QtCore.QEvent.MouseMove, pos, vp.mapToGlobal(pos), QtCore.Qt.NoButton,
                           QtCore.Qt.NoButton, QtCore.Qt.NoModifier)
    QtWidgets.QApplication.sendEvent(vp, ev)


def _corners(qtbot, vp, at, pts):
    for x, y in pts:
        _hover(vp, at(x, y))
        qtbot.mouseClick(vp, QtCore.Qt.LeftButton, pos=at(x, y))


TRIANGLE = [(0.2, 0.2), (0.7, 0.2), (0.45, 0.7)]


def test_polygon_closes_on_a_click_near_the_first_corner(window, view, qtbot, drawing):
    at, vp = drawing
    _corners(qtbot, vp, at, TRIANGLE)
    near = at(*TRIANGLE[0]) + QtCore.QPoint(6, -5)  # inside the snap radius, not on the corner
    _hover(vp, near)
    assert len(view._snap_ring.data) == 1
    qtbot.mouseClick(vp, QtCore.Qt.LeftButton, pos=near)
    gate = window.ws.get_gate("Poly", ROOT)
    assert len(gate.vertices) == 3
    assert view._draw_mode is None


def test_click_outside_snap_radius_adds_a_corner(window, view, qtbot, drawing):
    at, vp = drawing
    _corners(qtbot, vp, at, TRIANGLE + [(0.25, 0.4)])
    assert view._draw_mode == "polygon" and len(view._points) == 4


def test_right_click_closes_polygon_but_cancels_when_too_few_corners(window, view, qtbot,
                                                                     drawing):
    at, vp = drawing
    _corners(qtbot, vp, at, TRIANGLE[:2])
    qtbot.mouseClick(vp, QtCore.Qt.RightButton, pos=at(0.5, 0.5))
    assert view._draw_mode is None and window.ws.children(ROOT) == []
    view.start_gate("polygon")
    _corners(qtbot, vp, at, TRIANGLE)
    qtbot.mouseClick(vp, QtCore.Qt.RightButton, pos=at(0.9, 0.9))
    assert len(window.ws.get_gate("Poly", ROOT).vertices) == 3


def test_enter_closes_and_backspace_removes_last_corner(window, view, qtbot, drawing):
    at, vp = drawing
    _corners(qtbot, vp, at, TRIANGLE + [(0.9, 0.9)])
    qtbot.keyClick(view, QtCore.Qt.Key_Backspace)
    assert len(view._points) == 3
    qtbot.keyClick(view, QtCore.Qt.Key_Return)
    gate = window.ws.get_gate("Poly", ROOT)
    assert [v[0] for v in gate.vertices] == pytest.approx([p[0] for p in TRIANGLE], abs=0.02)


def test_closing_edge_preview_follows_the_mouse(view, qtbot, drawing):
    at, vp = drawing
    _corners(qtbot, vp, at, TRIANGLE[:2])
    _hover(vp, at(0.5, 0.8))
    xs, ys = view._close_preview.getData()
    assert xs[-1] == pytest.approx(0.2, abs=0.02) and ys[-1] == pytest.approx(0.2, abs=0.02)
    assert len(view._snap_ring.data) == 0
