"""Axis scales: the scale engine, gates following scale changes, and the scale controls."""

import flowkit as fk
import numpy as np
import pytest

from sidescatter.core import scales, serialize
from sidescatter.core.workspace import ROOT, Workspace

RAW = np.array([-5000.0, -100.0, 0.0, 50.0, 1000.0, 1e5, 262144.0]).reshape(-1, 1)


@pytest.mark.parametrize("kind", list(scales.KINDS))
def test_settings_round_trip(kind):
    xform = scales.make(kind, scales.defaults(kind, 262144.0))
    assert scales.kind_of(xform) == kind
    again, s = scales.settings(xform)
    assert again == kind
    with np.errstate(all="ignore"):
        np.testing.assert_allclose(scales.make(kind, s).apply(RAW), xform.apply(RAW), equal_nan=True)
    lo, hi = scales.raw_range(xform)
    assert lo < hi and hi == pytest.approx(262144.0, rel=0.01)


@pytest.mark.parametrize("kind", list(scales.KINDS))
def test_scale_ends_sit_at_zero_and_one(kind):
    xform = scales.make(kind, scales.defaults(kind, 4194304.0))
    lo, hi = scales.raw_range(xform)
    ends = xform.apply(np.array([[lo], [hi]])).ravel()
    np.testing.assert_allclose(ends, [0.0, 1.0], atol=1e-6)
    inside = np.array([[lo + 0.25 * (hi - lo)], [hi * 0.01]])
    np.testing.assert_allclose(xform.inverse(xform.apply(inside)), inside, rtol=1e-3)


@pytest.mark.parametrize("kind", ["biex", "neglog"])
def test_new_scales_save_and_load(kind):
    xform = scales.make(kind, scales.defaults(kind, 262144.0))
    back = serialize.transform_from_dict(serialize.transform_to_dict(xform))
    assert type(back) is type(xform)
    np.testing.assert_allclose(back.apply(RAW), xform.apply(RAW))


def test_biex_matches_flowjo_scaled_to_one():
    flowjo = fk.transforms.WSPBiexTransform(negative=0.5, width=-100, positive=4.42, max_value=262144)
    ours = scales.BiexTransform(0.5, -100, 4.42, 262144)
    np.testing.assert_allclose(ours.apply(RAW), flowjo.apply(RAW) / 4096.0)


def test_negative_log_is_symmetric():
    xform = scales.SymLogTransform(param_t=1e5, param_n=-1e5, param_c=10.0)
    d = xform.apply(np.array([[-1e4], [0.0], [1e4]])).ravel()
    assert d[1] == pytest.approx(0.5)
    assert d[0] == pytest.approx(1 - d[2])


def test_log_needs_a_positive_minimum():
    with pytest.raises(ValueError):
        scales.make("log", {"min": 0.0, "max": 1000.0})


def test_convert_keeps_the_top():
    biex = scales.make("biex", scales.defaults("biex", 4194304.0))
    for kind in scales.KINDS:
        assert scales.settings(scales.convert(biex, kind))[1]["max"] == pytest.approx(4194304.0)


def test_suggest_width_widens_for_spread_negatives():
    s = scales.defaults("logicle", 262144.0)
    narrow = scales.suggest_width("logicle", s, np.r_[np.full(100, -5.0), np.ones(100)])
    wide = scales.suggest_width("logicle", s, np.r_[np.full(100, -3000.0), np.ones(100)])
    assert wide["w"] > narrow["w"]


@pytest.fixture
def ws(spectral_dir):
    ws = Workspace()
    [sid] = ws.add_fcs([spectral_dir / "PBMC_full_panel.fcs"])
    return ws, sid


def test_fluorescence_starts_on_biex_and_scatter_on_linear(ws):
    ws, sid = ws
    kinds = {c.kind: scales.kind_of(ws.transform(c.pnn)) for c in ws.channels(sid)}
    assert kinds["fluorescence"] == "biex"
    assert kinds["scatter"] == "linear"


def test_saved_default_is_used_for_new_channels(spectral_dir, monkeypatch):
    monkeypatch.setattr(scales, "saved_defaults",
                        {"V1-A": {"kind": "arcsinh", "settings": {"a": 0.0, "cofactor": 2.0}}})
    ws = Workspace()
    ws.add_fcs([spectral_dir / "PBMC_full_panel.fcs"])
    assert scales.kind_of(ws.transform("V1-A")) == "arcsinh"
    assert scales.kind_of(ws.transform("V2-A")) == "biex"


@pytest.mark.parametrize("kind", [k for k in scales.KINDS if k != "biex"])
def test_gates_keep_their_events_when_the_scale_changes(ws, kind):
    ws, sid = ws
    ch = "V1-A"
    raw = ws.scale_values(sid, ch)
    lo, hi = np.quantile(raw[raw > 0], [0.3, 0.9])
    xform = ws.transform(ch)
    d = scales.make("biex", scales.settings(xform)[1]).apply(np.array([[lo], [hi]])).ravel()
    ws.add_gate(fk.gates.RectangleGate("Pos", [ws.make_dimension(sid, ch, d[0], d[1])]))
    before = ws.count(sid, "Pos")
    ws.set_transform(ch, scales.convert(xform, kind))
    assert abs(ws.count(sid, "Pos") - before) <= max(2, 0.001 * before)
    ws.restore_previous_transform(ch)
    assert scales.kind_of(ws.transform(ch)) == "biex"
    assert ws.previous_transform(ch) is None
    assert abs(ws.count(sid, "Pos") - before) <= max(2, 0.001 * before)


def test_flowjo_biex_is_rescaled_to_plot_scale(ws):
    """FlowJo's own biex (0..4096) becomes the same scale on 0..1, same events gated."""
    ws, sid = ws
    ch = "V1-A"
    flowjo = fk.transforms.WSPBiexTransform(negative=0.0, width=-100, positive=5.62, max_value=4194304)
    ws.set_transform(ch, flowjo, remember=False)
    ws.add_gate(fk.gates.RectangleGate("Pos", [ws.make_dimension(sid, ch, 2000.0, 3500.0)]))
    before = ws.count(sid, "Pos")
    ws.rescale_flowjo_biex()
    assert isinstance(ws.transform(ch), scales.BiexTransform)
    assert ws.get_gate("Pos", ROOT).dimensions[0].min == pytest.approx(2000.0 / 4096)
    assert ws.count(sid, "Pos") == before
    assert ws.previous_transform(ch) is None


# ----- the controls ------------------------------------------------------------

pytest.importorskip("pytestqt")


@pytest.fixture
def view(qtbot):
    from sidescatter.app.jobs import wait_idle
    from sidescatter.app.main_window import DEMO_DIR, MainWindow

    w = MainWindow()
    w.ask_before_discard = False
    qtbot.addWidget(w)
    w.add_samples([str(DEMO_DIR / "PBMC_full_panel.fcs")])
    wait_idle()
    v = w.open_graph("PBMC_full_panel.fcs")
    v.set_axes("V1-A", "V2-A")
    return v


def test_scale_buttons_sit_by_each_axis_picker(view):
    assert view.x_scale.text() == "Biex" and view.y_scale.text() == "Biex"
    view.set_scale("V1-A", "neglog")
    assert view.x_scale.text() == "NegLog"
    assert scales.kind_of(view.ws.transform("V1-A")) == "neglog"
    view.restore_scale("V1-A")
    assert view.x_scale.text() == "Biex"


def test_histogram_has_no_y_scale_button(view):
    view.type_combo.setCurrentText("Histogram")
    assert view.x_scale.isVisibleTo(view) and not view.y_scale.isVisibleTo(view)


def test_scale_menu_lists_every_scale(view):
    from sidescatter.app.axis_scales import scale_menu

    menu = scale_menu(view, view.ws, "V1-A", view.set_scale, view.restore_scale, lambda c: None)
    texts = [a.text() for a in menu.actions() if not a.isSeparator()]
    assert texts[:7] == [label for label, _ in scales.KINDS.values()]
    assert [a.text() for a in menu.actions() if a.isChecked()] == ["Biex"]
    assert texts[-1] == "Customize…"


def test_transform_window_previews_then_applies(view, qtbot):
    from sidescatter.app.axis_scales import TransformWindow

    ws = view.ws
    dlg = TransformWindow(ws, view.sample_id, None, "V1-A")
    qtbot.addWidget(dlg)
    dlg.kind.setCurrentIndex(dlg.kind.findData("logicle"))
    dlg._kind_picked()
    w_before = scales.settings(dlg.pending["V1-A"])[1]["w"]
    dlg.sliders[1].set_value(1.5)
    dlg._edited()
    assert scales.settings(dlg.pending["V1-A"])[1]["w"] == pytest.approx(1.5)
    assert w_before != 1.5
    top = dlg.max_box.value()
    dlg.nudge("max", +1)
    assert dlg.max_box.value() == pytest.approx(2 * top)
    dlg.copy_to(["V2-A", "V3-A"])
    assert scales.kind_of(ws.transform("V1-A")) == "biex"  # nothing changes before Apply
    dlg.apply()
    assert {scales.kind_of(ws.transform(c)) for c in ("V1-A", "V2-A", "V3-A")} == {"logicle"}
    assert scales.settings(ws.transform("V2-A"))[1]["w"] == pytest.approx(1.5)


def test_transform_window_rejects_bad_settings(view, qtbot):
    from sidescatter.app.axis_scales import TransformWindow

    dlg = TransformWindow(view.ws, view.sample_id, None, "V1-A")
    qtbot.addWidget(dlg)
    dlg.kind.setCurrentIndex(dlg.kind.findData("log"))
    dlg._kind_picked()
    dlg.min_box.setValue(-5)
    assert "Min must be above 0" in dlg.problem.text()


def test_save_keeps_a_scale_for_new_workspaces(view, qtbot, tmp_path, monkeypatch):
    from PySide6 import QtCore

    from sidescatter.app import axis_scales

    ini = QtCore.QSettings(str(tmp_path / "s.ini"), QtCore.QSettings.IniFormat)
    monkeypatch.setattr(axis_scales, "_settings", lambda: ini)
    monkeypatch.setattr(scales, "saved_defaults", {})
    dlg = axis_scales.TransformWindow(view.ws, view.sample_id, None, "V1-A")
    qtbot.addWidget(dlg)
    dlg.kind.setCurrentIndex(dlg.kind.findData("arcsinh"))
    dlg._kind_picked()
    dlg.save_default()
    assert scales.saved_defaults["V1-A"]["kind"] == "arcsinh"
    scales.saved_defaults.clear()
    axis_scales.load_saved_scales()
    assert scales.saved_defaults["V1-A"]["kind"] == "arcsinh"
    assert "max" not in scales.saved_defaults["V1-A"]["settings"]
