"""Workspace files over time: every release's example still opens with the same
counts, older formats are upgraded, newer ones refused plainly, saves are
atomic, and samples whose FCS files moved can be found again."""

import json
import os
import shutil
import zipfile
from pathlib import Path

import flowkit as fk
import pytest
from PySide6 import QtCore

from sidescatter import __version__
from sidescatter.core import versioning
from sidescatter.core.versioning import FORMAT_VERSION, NewerWorkspaceError
from sidescatter.core.workspace import Workspace

EXAMPLES = Path(__file__).parent / "workspaces"
FINGERPRINT = EXAMPLES / f"format-v{FORMAT_VERSION}.json"
SID = "PBMC_full_panel.fcs"


def _examples():
    return sorted(p for p in EXAMPLES.glob("v*") if p.is_dir())


def _counts(ws):
    return {sid: {"/".join([*p[1:], n]): ws.count(sid, n, p) for n, p in ws.population_paths()
                  if ws.applies_to(sid, n, p)}
            for sid in ws.samples}


def _close(got, want):
    """Counts may move by an event or two across platforms (unmixing is recomputed)."""
    return abs(got - want) <= max(2, 0.002 * want)


@pytest.mark.parametrize("folder", _examples(), ids=lambda p: p.name)
def test_every_release_example_opens_with_the_same_counts(folder, tmp_path):
    moved = tmp_path / folder.name  # opened somewhere else, as when a user moves a project
    shutil.copytree(folder, moved)
    expected = json.loads((moved / "expected.json").read_text())["counts"]
    ws = Workspace.load(moved / "example.ssw")
    assert not ws.missing
    got = _counts(ws)
    assert set(got) == set(expected)
    for sid, pops in expected.items():
        assert set(got[sid]) == set(pops), sid
        bad = {p: (got[sid][p], n) for p, n in pops.items() if not _close(got[sid][p], n)}
        assert not bad, f"{folder.name} {sid}: (now, expected) {bad}"

    ws.save(tmp_path / "again.ssw")  # and it still saves and reopens the same
    again = Workspace.load(tmp_path / "again.ssw", search_dirs=[moved])
    assert _counts(again) == got


def test_save_format_matches_its_fingerprint():
    """Fails when what a workspace file holds changes. See the message."""
    ws = Workspace.load(_examples()[-1] / "example.ssw")
    shape = versioning.shape(ws.to_document())
    if os.environ.get("SIDESCATTER_RECORD_FORMAT"):
        FINGERPRINT.write_text(json.dumps(shape, indent=1, sort_keys=True) + "\n")
    assert FINGERPRINT.exists(), f"no fingerprint for format {FORMAT_VERSION}: run with SIDESCATTER_RECORD_FORMAT=1"
    recorded = json.loads(FINGERPRINT.read_text())
    assert shape == recorded, (
        "The workspace file format changed (fields added, removed or retyped).\n"
        "- If older Sidescatters can still open the new files and new Sidescatters the old ones\n"
        "  (e.g. a new optional field with a default), re-record the fingerprint:\n"
        "      SIDESCATTER_RECORD_FORMAT=1 pytest tests/test_versioning.py\n"
        "- Otherwise bump FORMAT_VERSION in core/versioning.py, add an upgrade step from the old\n"
        "  version to UPGRADES there, then record the new fingerprint as above.\n"
        "Either way, make an example workspace at release: scripts/make_example_workspace.py")


def _doc(path):
    with zipfile.ZipFile(path) as zf:
        return json.loads(zf.read("workspace.json"))


def _rewrite(path, doc):
    with zipfile.ZipFile(path) as zf:
        others = {n: zf.read(n) for n in zf.namelist() if n != "workspace.json"}
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("workspace.json", json.dumps(doc))
        for n, data in others.items():
            zf.writestr(n, data)


@pytest.fixture
def saved(spectral_dir, tmp_path):
    """A workspace with a gate, a group and a cleaned sample, saved next to its FCS file."""
    shutil.copy(spectral_dir / SID, tmp_path / SID)
    ws = Workspace()
    ws.add_fcs([tmp_path / SID])
    ws.add_gate(fk.gates.RectangleGate("Big", [ws.make_dimension(SID, "FSC-A", 0.3, None)]))
    ws.add_group("Stained", [SID])
    ws.clean([SID])
    ws.save(tmp_path / "w.ssw")
    return ws, tmp_path / "w.ssw"


def test_files_record_who_wrote_them(saved):
    doc = _doc(saved[1])
    assert doc["version"] == FORMAT_VERSION and doc["written_by"] == __version__


def test_newer_files_are_refused_with_a_plain_message(saved):
    _, path = saved
    doc = _doc(path)
    doc.update(version=FORMAT_VERSION + 1, written_by="9.9.0")
    _rewrite(path, doc)
    with pytest.raises(NewerWorkspaceError) as err:
        Workspace.load(path)
    assert str(err.value) == (f"This workspace was saved by Sidescatter 9.9.0, which is newer than "
                              f"this copy (Sidescatter {__version__}). Update Sidescatter to open it.")


def test_older_files_are_upgraded_step_by_step(saved, monkeypatch):
    ws, path = saved
    steps = []

    def v1_to_v2(doc):
        steps.append(1)
        doc["groups"]["Renamed"] = doc["groups"].pop("Stained")
        return doc

    def v2_to_v3(doc):
        steps.append(2)
        return doc

    monkeypatch.setattr(versioning, "FORMAT_VERSION", 3)
    monkeypatch.setattr(versioning, "UPGRADES", {1: v1_to_v2, 2: v2_to_v3})
    back = Workspace.load(path)
    assert steps == [1, 2] and back.upgraded_from == 1
    assert back.groups["Renamed"] == [SID] and back.count(SID, "Big") == ws.count(SID, "Big")
    assert _doc(path)["version"] == 1  # nothing changes on disk until it's saved


def test_a_failed_save_leaves_the_old_file_alone(saved, monkeypatch):
    ws, path = saved
    before = path.read_bytes()
    ws.add_group("New", [SID])

    def full_disk(*_):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(zipfile.ZipFile, "writestr", full_disk)
    with pytest.raises(OSError):
        ws.save(path)
    assert path.read_bytes() == before
    assert sorted(p.name for p in path.parent.iterdir()) == sorted([SID, "w.ssw"])  # no temp file left


@pytest.mark.skipif(os.name == "nt", reason="Windows has no Unix permissions")
def test_saving_keeps_the_files_permissions(saved):
    ws, path = saved
    os.chmod(path, 0o640)
    ws.save(path)
    assert path.stat().st_mode & 0o777 == 0o640


def test_missing_files_open_anyway_and_are_kept(saved, tmp_path):
    ws, path = saved
    big = ws.count(SID, "Big")
    moved = tmp_path / "moved" / "deeper"
    moved.mkdir(parents=True)
    shutil.move(tmp_path / SID, moved / SID)

    back = Workspace.load(path)
    assert list(back.missing) == [SID] and not back.samples
    assert back.groups == {"All Samples": [], "Stained": []}
    qc = ("root", ws.qc_gate())
    assert back.qc_gate() == ws.qc_gate() and back.get_gate("Big", qc) is not None

    back.save(tmp_path / "still.ssw")  # saving doesn't lose the sample, its group or its cleaning
    doc = _doc(tmp_path / "still.ssw")
    assert [s["id"] for s in doc["samples"]] == [SID]
    assert doc["groups"]["Stained"] == [SID] and SID in doc["qc"]

    found = back.find_missing(tmp_path / "moved")  # one folder; files below it count
    assert found == {SID: moved / SID}
    back.restore_missing(back.read_missing(found))
    assert not back.missing and back.groups["Stained"] == [SID]
    assert back.sources[SID]["path"] == str(moved / SID)
    assert back.count(SID, "Big", qc) == ws.count(SID, "Big", qc)
    assert back.qc[SID].n_events == ws.qc[SID].n_events
    assert big == ws.count(SID, "Big")


def test_unmixed_samples_come_back_with_their_raw_file(spectral_dir, tmp_path):
    from sidescatter.core import unmixing

    shutil.copy(spectral_dir / SID, tmp_path / SID)
    ws = Workspace()
    [raw] = ws.add_fcs([tmp_path / SID])
    specs = unmixing.guess_controls(sorted((spectral_dir / "controls").glob("*.fcs")))
    um, _ = unmixing.build_unmixer(specs, method="ols")
    ws.add_unmixer("Aurora", um)
    unmixed = ws.unmix(raw, "Aurora")
    ws.save(tmp_path / "w.ssw")
    (tmp_path / "elsewhere").mkdir()
    shutil.move(tmp_path / SID, tmp_path / "elsewhere" / SID)

    back = Workspace.load(tmp_path / "w.ssw")
    assert set(back.missing) == {raw, unmixed}
    back.restore_missing(back.read_missing(back.find_missing(tmp_path / "elsewhere")))
    assert not back.missing and set(back.samples) == {raw, unmixed}
    assert back.samples[unmixed].event_count == ws.samples[unmixed].event_count



# ----- in the app ------------------------------------------------------------


@pytest.fixture
def window(qtbot):
    from sidescatter.app.main_window import MainWindow

    w = MainWindow()
    w.ask_before_discard = False
    qtbot.addWidget(w)
    return w


def test_app_shows_missing_samples_and_locates_them(window, saved, tmp_path):
    ws, path = saved
    (tmp_path / "new place").mkdir()
    shutil.move(tmp_path / SID, tmp_path / "new place" / SID)
    window.load_workspace(str(path)).wait()
    panel = window.panel
    assert not panel.banner.isHidden() and "1 sample' FCS file couldn't be found" in panel.banner_text.text()
    row = panel.tree.topLevelItem(0)
    assert row.text(0) == SID and row.text(1) == "File not found"
    assert not row.flags() & QtCore.Qt.ItemIsSelectable  # can't be plotted or dragged

    window.locate_files(str(tmp_path / "new place")).wait()
    assert not window.ws.missing and panel.banner.isHidden()
    assert panel.tree.topLevelItem(0).text(1) != "File not found"
    assert "Found all 1 samples" in window.statusBar().currentMessage()


def test_first_save_after_an_upgrade_keeps_the_original(window, saved, monkeypatch):
    _, path = saved
    original = path.read_bytes()
    monkeypatch.setattr(versioning, "FORMAT_VERSION", 2)
    monkeypatch.setattr(versioning, "UPGRADES", {1: lambda doc: doc})
    window.load_workspace(str(path)).wait()
    window.save().wait()
    backup = path.with_name("w.v1-backup.ssw")
    assert backup.read_bytes() == original
    assert "original is kept as w.v1-backup.ssw" in window.statusBar().currentMessage()
    window.save().wait()  # only the first save makes one
    assert sorted(p.name for p in path.parent.glob("*.ssw")) == ["w.ssw", "w.v1-backup.ssw"]


def test_newer_file_offers_the_update(window, saved, monkeypatch, qtbot):
    from PySide6 import QtWidgets

    _, path = saved
    doc = _doc(path)
    doc.update(version=FORMAT_VERSION + 1, written_by="9.9.0")
    _rewrite(path, doc)
    shown = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "exec",
                        lambda self: shown.append((self.text(), [b.text() for b in self.buttons()])) or 0)
    job = window.load_workspace(str(path))
    with pytest.raises(NewerWorkspaceError):
        job.wait()
    qtbot.waitUntil(lambda: bool(shown), timeout=3000)
    text, buttons = shown[0]
    assert "saved by Sidescatter 9.9.0" in text and "Download update" in buttons
    assert "Report bug…" not in buttons  # not a bug
