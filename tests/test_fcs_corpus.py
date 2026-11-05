"""Odd and broken FCS files (tests/data/fcs_corpus, made by scripts/make_fcs_corpus.py):
each one loads, loads with a warning, or is refused with a reason in plain words."""

import json
from pathlib import Path

import numpy as np
import pytest

from sidescatter.core import fcs
from sidescatter.core.fcs import UnreadableFCS

CORPUS = Path(__file__).parent / "data" / "fcs_corpus"
EXPECTED = json.loads((CORPUS / "expected.json").read_text())
STANDARD = CORPUS / "fcs31_float_little.fcs"


def test_every_file_in_the_folder_is_listed():
    files = {p.name for p in CORPUS.iterdir() if p.suffix == ".fcs"}
    assert files == set(EXPECTED)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_corpus_file(name):
    want = EXPECTED[name]
    path = CORPUS / name
    if want["outcome"] == "fail":
        with pytest.raises(UnreadableFCS) as err:
            fcs.load_sample(path)
        assert want["words"] in err.value.reason
        assert "Traceback" not in str(err.value) and "\n" not in err.value.reason
        return
    sample = fcs.load_sample(path)
    warnings = " ".join(sample.load_warnings)
    if want["outcome"] == "warn":
        assert want["words"] in warnings
    else:
        assert not sample.load_warnings
    assert sample.event_count > 0
    if want["standard_events"]:
        ref = fcs.load_sample(STANDARD).get_events(source="raw")
        assert np.array_equal(sample.get_events(source="raw"), ref)


@pytest.mark.parametrize("name", sorted(n for n, w in EXPECTED.items() if w["outcome"] == "ok"))
def test_own_reader_matches_flowkit(name):
    """The fallback reader (for files FlowKit can't take) decodes what FlowKit decodes."""
    info = fcs.inspect_fcs(CORPUS / name)
    if info["fallback"]:
        pytest.skip("FlowKit can't read this one as it is")
    ours = fcs._decode(CORPUS / name, info, name)
    theirs = fcs.read_fcs(CORPUS / name)
    assert ours.pnn_labels == theirs.pnn_labels
    assert np.allclose(ours.get_events(source="raw"), theirs.get_events(source="raw"), rtol=1e-6)


def test_spillover_survives_the_fallback_reader():
    sample = fcs.load_sample(CORPUS / "spill.fcs")
    info = fcs.inspect_fcs(CORPUS / "spill.fcs")
    ours = fcs._decode(CORPUS / "spill.fcs", info, "spill.fcs")
    assert sample.get_metadata()["spill"] == ours.get_metadata()["spill"]


def test_unmixed_samples_never_inherit_the_raw_files_spillover():
    raw = fcs.load_sample(CORPUS / "spill.fcs")
    out = fcs.load_sample(raw.get_events(source="raw"), "x", list(raw.pnn_labels), keywords=fcs.keywords(raw))
    assert "spill" not in out.get_metadata() and "spillover" not in out.get_metadata()


def test_app_adds_what_it_can_and_explains_the_rest(qtbot, monkeypatch):
    pytest.importorskip("pytestqt")
    from PySide6 import QtWidgets

    from sidescatter.app.jobs import wait_idle
    from sidescatter.app.main_window import MainWindow

    w = MainWindow()
    w.ask_before_discard = False
    qtbot.addWidget(w)
    shown = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "exec",
                        lambda self: shown.append(self.text() + "\n" + self.informativeText()) or 0)
    w.add_samples([str(CORPUS / n) for n in ("fcs31_float_little.fcs", "truncated.fcs",
                                              "enddata_off_by_one.fcs", "._resource_fork.fcs")])
    wait_idle()
    assert sorted(w.ws.samples) == ["enddata_off_by_one.fcs", "fcs31_float_little.fcs"]
    assert "Couldn't read 2 files" in shown[0] and "The other 2 were added" in shown[0]
    assert "truncated.fcs: the file ends" in shown[0] and "._resource_fork.fcs: not an FCS file" in shown[0]
    rows = {w.panel.tree.topLevelItem(i).text(0): w.panel.tree.topLevelItem(i)
            for i in range(w.panel.tree.topLevelItemCount())}
    assert "data offsets are slightly off" in rows["enddata_off_by_one.fcs"].toolTip(0)
    assert not rows["fcs31_float_little.fcs"].toolTip(0)
    assert "1 opened with a warning" in w.statusBar().currentMessage()
