"""The Clean data dialog, the QC window, the PDF report and cleaning in the Unmixing tab."""

import zipfile

import pytest

pytest.importorskip("pytestqt")

from PySide6 import QtCore, QtWidgets  # noqa: E402

from sidescatter.app import qc_report  # noqa: E402
from sidescatter.app.jobs import wait_idle  # noqa: E402
from sidescatter.app.main_window import MainWindow  # noqa: E402
from sidescatter.app.qc_window import CleanDialog  # noqa: E402
from sidescatter.core import fcs  # noqa: E402
from sidescatter.core.workspace import QC_GATE, ROOT  # noqa: E402
from test_cleaning import synthetic_run  # noqa: E402


@pytest.fixture
def window(qtbot, tmp_path):
    w = MainWindow()
    w.ask_before_discard = False
    qtbot.addWidget(w)
    paths = []
    for i, kind in enumerate(({"clog": True}, {"clog": True, "drift": True, "bubble": True}, {})):
        events, names, _ = synthetic_run(seed=i, **kind)
        p = tmp_path / f"run{i}.fcs"
        fcs.write_fcs(p, events, names, keywords={"TIMESTEP": "0.01"})
        paths.append(str(p))
    w.add_samples(paths)
    wait_idle()
    return w


def _accept(monkeypatch, settings=None):
    monkeypatch.setattr(CleanDialog, "exec", lambda self: QtWidgets.QDialog.Accepted)
    if settings is not None:
        monkeypatch.setattr(CleanDialog, "settings", lambda self: settings)


def test_clean_data_batch_and_qc_window(window, qtbot, monkeypatch, tmp_path):
    _accept(monkeypatch)
    done = window.clean_data().wait()
    assert set(done) == set(window.ws.groups["All Samples"])
    assert window.ws.qc_gate() == QC_GATE

    win = window.qc_window
    assert win.isVisible() and win.table.rowCount() == 3
    amber = {win.table.item(r, 0).text() for r in range(3)
             if win.table.item(r, 0).background().color().name().upper() == qc_report.AMBER}
    pct = {win.table.item(r, 0).text(): float(win.table.item(r, 2).text().rstrip("%")) for r in range(3)}
    assert amber == {sid for sid, v in pct.items() if v > 20}
    assert pct["run1.fcs"] > pct["run0.fcs"] > pct["run2.fcs"]

    # sample table shows QC passed above the sample's gates
    tree = window.panel.tree
    top = tree.topLevelItem(0)
    assert top.child(0).text(0) == QC_GATE

    # edit the stretches on the event-rate plot
    win.show_sample("run0.fcs")
    before = window.ws.count("run0.fcs", QC_GATE)
    assert len(win._regions) >= 1
    win._regions.clear()
    win._commit()
    qtbot.waitUntil(lambda: window.ws.qc["run0.fcs"].edited)
    assert window.ws.count("run0.fcs", QC_GATE) > before
    win.show_sample("run0.fcs")
    win.add_stretch()
    assert len(window.ws.qc["run0.fcs"].stretches) == 1
    win.reset_stretches()
    assert not window.ws.qc["run0.fcs"].edited

    pdf = win.save_pdf(str(tmp_path / "qc.pdf"))
    assert (tmp_path / "qc.pdf").read_bytes()[:4] == b"%PDF"
    assert pdf


def test_dialog_presets_and_channels(window):
    sid = window.ws.groups["All Samples"][0]
    dlg = CleanDialog(window.ws, [sid])
    assert dlg.preset.currentText() == "Standard"
    assert dlg.settings().channels is None  # untouched: each sample's own defaults
    dlg.preset.setCurrentText("Strict")
    s = dlg.settings()
    assert (s.mad, s.it_limit, s.preset) == (4.0, 0.55, "Strict")
    dlg.mad.setValue(7)
    assert dlg.preset.currentText() == "Custom"
    item = dlg.channels.item(0)
    item.setCheckState(QtCore.Qt.Unchecked)
    chans = dlg.settings().channels
    assert item.data(QtCore.Qt.UserRole) not in chans and "Time" not in chans


def test_graph_window_follows_its_gate_into_qc_passed(window, monkeypatch):
    import flowkit as fk

    ws = window.ws
    sid = ws.groups["All Samples"][0]
    ws.add_gate(fk.gates.RectangleGate("Pop", [ws.make_dimension(sid, "C1-A", 0.0, 0.5)]))
    view = window.open_graph(sid, "Pop", ROOT)
    _accept(monkeypatch)
    window.clean_data().wait()
    assert view.population == ("Pop", ROOT + (QC_GATE,))
    ws.remove_qc()
    window._changed()
    assert view.population == ("Pop", ROOT)


def test_unmixing_tab_cleans_controls_and_exports_report(qtbot, tmp_path, monkeypatch):
    monkeypatch.setattr(QtWidgets.QMessageBox, "show", lambda self: None)
    w = MainWindow()
    w.ask_before_discard = False
    qtbot.addWidget(w)
    w.open_demo()
    wait_idle()
    tab = w.unmix_tab
    plain = tab.unmixer.matrix.copy()
    tab.clean_check.setChecked(True)
    assert tab.setup.clean_controls and tab.unmixer is not None
    assert "Cleaned 15 control(s)" in tab.status.text()
    assert tab.unmixer.matrix.shape == plain.shape
    assert tab.setup.to_dict()["clean_controls"] is True

    result = tab.export(tmp_path, "Demo", True, clean=True).wait()
    with zipfile.ZipFile(result.path) as zf:
        names = zf.namelist()
        pdf = zf.read("Demo/QC/Demo data cleaning.pdf")
        ssw = zf.read("Demo/Demo.ssw")
    assert pdf[:4] == b"%PDF"
    (tmp_path / "Demo.ssw").write_bytes(ssw)
    with zipfile.ZipFile(tmp_path / "Demo.ssw") as zf:
        doc = __import__("json").loads(zf.read("workspace.json"))
    assert list(doc["qc"]) == ["PBMC_full_panel.fcs"]
    assert doc["gates"][0]["type"] == "QCGate"

    (tmp_path / "plain").mkdir()
    plain_export = tab.export(tmp_path / "plain", "Demo", True, clean=False).wait()
    with zipfile.ZipFile(plain_export.path) as zf:
        assert not any(n.startswith("Demo/QC/") for n in zf.namelist())
