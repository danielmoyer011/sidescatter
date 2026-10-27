"""Classifying raw files and exporting SpectroFlo-style experiment packages."""

import csv
import io
import shutil
import zipfile

import numpy as np
import pytest

from sidescatter.core import classify, experiment, fcs, spectral, unmixing
from sidescatter.core.workspace import Workspace


@pytest.fixture
def demo_files(spectral_dir):
    return sorted((spectral_dir / "controls").glob("*.fcs")) + [spectral_dir / "PBMC_full_panel.fcs"]


@pytest.mark.parametrize("name, fluors, marker, unstained, carrier", [
    ("Beads_PECy7_CD45RA", ["PE-Cy7"], "CD45RA", False, "beads"),
    ("B2 BV421 (Beads)", ["BV421"], "", False, "beads"),
    ("Unstained (Cells)", [], "", True, "cells"),
    ("A1 Spark Blue 550 CD4 beads", ["Spark Blue 550"], "CD4", False, "beads"),
    ("Alexa Fluor 647 IgM", ["AF647"], "IgM", False, None),
    ("APC-Fire 750 CD8", ["APC-Fire 750"], "CD8", False, None),
    ("PBMC US", [], "", True, "cells"),
    ("cFluor V420 Cells", ["cFluor V420"], "", False, "cells"),
    ("BV421 CD3 PE CD4 FITC CD8", ["BV421", "PE", "FITC"], "CD3 CD4 CD8", False, None),
])
def test_parse_name(name, fluors, marker, unstained, carrier):
    info = classify.parse_name(name)
    assert info.fluors == fluors
    assert info.marker == marker
    assert info.unstained == unstained
    assert info.carrier == carrier


def test_demo_files_classified_by_name(demo_files):
    specs = {s.path.name: s for s in unmixing.guess_files(demo_files)}
    assert specs["PBMC_full_panel.fcs"].role == "sample"
    assert specs["Beads_unstained.fcs"].role == "unstained" and specs["Beads_unstained.fcs"].carrier == "beads"
    assert specs["PBMC_unstained.fcs"].role == "unstained" and specs["PBMC_unstained.fcs"].carrier == "cells"
    pe = specs["Beads_PECy7_CD45RA.fcs"]
    assert (pe.role, pe.carrier, pe.name, pe.marker, pe.guessed_by) == (
        "single_stain", "beads", "PE-Cy7", "CD45RA", "name")
    assert pe.reasons
    assert unmixing.check_specs(list(specs.values())) == []


def test_data_decides_when_names_say_nothing(demo_files, tmp_path):
    anon = []
    for i, f in enumerate(demo_files):
        anon.append(tmp_path / f"file{i:02d}.fcs")
        shutil.copy(f, anon[-1])
    got = {f.name: g for f, g in zip(demo_files, classify.classify_files(anon))}
    assert all(g.source == "data" for g in got.values())
    assert got["PBMC_full_panel.fcs"].role == "sample"
    for name in ("Beads_unstained.fcs", "Dead_unstained.fcs", "PBMC_unstained.fcs"):
        assert got[name].role == "unstained", name
    for name in ("Beads_PE_X.fcs", "Beads_FITC_CD41b.fcs", "Dead_PO_DCM.fcs"):
        assert got[name].role == "single_stain", name
    assert got["Beads_PE_X.fcs"].carrier == "beads"
    assert got["PBMC_unstained.fcs"].carrier == "cells"


def test_spectroflo_folders_and_keywords(spectral_dir, tmp_path):
    ref = tmp_path / "Exp" / "Raw" / "Reference Group"
    samp = tmp_path / "Exp" / "Raw" / "Group_001"
    unmixed = tmp_path / "Exp" / "Unmixed" / "Reference Group"
    for d in (ref, samp, unmixed):
        d.mkdir(parents=True)
    src = spectral_dir / "controls" / "Beads_PE_X.fcs"
    shutil.copy(src, ref / "Tube 3.fcs")
    shutil.copy(src, unmixed / "Tube 3.fcs")

    # A sample whose group keyword says so, and a reference marked only by keyword.
    sample = fcs.load_sample(spectral_dir / "PBMC_full_panel.fcs")
    raw = sample.get_events(source="raw")
    fcs.write_fcs(samp / "Tube 9.fcs", raw, sample.pnn_labels, keywords={"GROUPNAME": "Group_001"})
    control = fcs.load_sample(src)
    fcs.write_fcs(tmp_path / "Tube 4.fcs", control.get_events(source="raw"), control.pnn_labels,
                  keywords={"GROUPNAME": "Reference Group", "TUBENAME": "PE (Beads)"})

    g = classify.classify(ref / "Tube 3.fcs")
    assert (g.role, g.source) == ("single_stain", "folder")
    assert classify.classify(unmixed / "Tube 3.fcs").role == "exclude"
    assert classify.classify(samp / "Tube 9.fcs").role == "sample"
    g = classify.classify(tmp_path / "Tube 4.fcs")
    assert (g.role, g.fluor, g.carrier, g.source) == ("single_stain", "PE", "beads", "keywords")


def test_duplicate_fluors_are_an_error(demo_files):
    specs = unmixing.guess_files(demo_files)
    by_name = {s.path.name: s for s in specs}
    by_name["Dead_PO_DCM.fcs"].name = "PE"
    problems = unmixing.check_specs(specs)
    assert problems[0][0] == "error" and "PE" in problems[0][1]


def test_write_fcs_roundtrip(tmp_path):
    ev = np.random.default_rng(0).normal(size=(2500, 4)).astype(np.float32) * 1e4
    p = tmp_path / "x.fcs"
    fcs.write_fcs(p, ev, ["FSC-A", "BV421-A", "PE-A", "AF-A"], ["", "CD3", "CD4", ""],
                  {"$DATE": "24-Oct-2026", "cyt": "Aurora", "TUBENAME": "a/b|c"}, [4194304] * 4)
    s = fcs.load_sample(p)
    np.testing.assert_array_equal(s.get_events(source="raw"), ev)
    assert s.pns_labels == ["", "CD3", "CD4", ""]
    meta = s.get_metadata()
    assert meta["date"] == "24-Oct-2026" and meta["tubename"] == "a/b|c" and meta["p2r"] == "4194304"


def test_old_unmixers_keep_their_channel_names():
    d = spectral.Unmixer(["a", "b"], [spectral.Reference("PE_X", np.array([1.0, 0.5]))]).to_dict()
    assert spectral.Unmixer.from_dict(d).channel_labels() == [("PE_X-A", "")]
    del d["channel_style"]
    assert spectral.Unmixer.from_dict(d).channel_labels() == [("PE_X", "")]


def _export(demo_files, tmp_path, **kw):
    specs = unmixing.guess_files(demo_files)
    {s.path.name: s for s in specs}["Dead_unstained.fcs"].role = "exclude"
    um, _ = unmixing.build_unmixer(specs)
    return specs, um, experiment.export_experiment(tmp_path, "My Exp", specs, um, **kw)


def test_export_zip_matches_spectroflo_layout(demo_files, tmp_path):
    specs, um, result = _export(demo_files, tmp_path)
    assert result.path == tmp_path / "My Exp.zip"
    assert (result.n_references, result.n_samples) == (14, 1)
    assert [p.name for p in tmp_path.iterdir()] == ["My Exp.zip"]  # no temporary files left
    with zipfile.ZipFile(result.path) as zf:
        names = set(zf.namelist())
        for tree in ("Raw", "Unmixed"):
            assert f"My Exp/{tree}/Samples/PBMC_full_panel.fcs" in names
            assert f"My Exp/{tree}/Reference Group/Beads_PE_X.fcs" in names
            assert f"My Exp/{tree}/Reference Group/Dead_unstained.fcs" not in names
        for extra in ("My Exp unmixing matrix.csv", "manifest.csv", "QC report.txt", "My Exp.ssw"):
            assert f"My Exp/{extra}" in names
        raw = zf.read("My Exp/Raw/Samples/PBMC_full_panel.fcs")
        assert raw == (demo_files[-1]).read_bytes()
        manifest = list(csv.DictReader(io.StringIO(zf.read("My Exp/manifest.csv").decode())))
        matrix = list(csv.reader(io.StringIO(zf.read("My Exp/My Exp unmixing matrix.csv").decode())))
        report = zf.read("My Exp/QC report.txt").decode()
    assert len(manifest) == len(demo_files)
    excluded = next(r for r in manifest if r["File"] == "Dead_unstained.fcs")
    assert excluded["Role"] == "Excluded" and excluded["Raw file"] == ""
    assert matrix[0][5:] == um.detectors and len(matrix) == len(um.references) + 1
    assert "Complexity index" in report


def test_export_folder_unmixed_files_and_workspace(demo_files, tmp_path):
    specs, um, result = _export(demo_files, tmp_path, as_zip=False)
    root = result.path
    unmixed = fcs.load_sample(root / "Unmixed" / "Samples" / "PBMC_full_panel.fcs")
    labels = dict(zip(unmixed.pnn_labels, unmixed.pns_labels))
    assert labels["PE-Cy7-A"] == "CD45RA" and labels["PB-A"] == "CD3" and "AF-A" in labels
    assert unmixed.get_metadata()["cyt"] == "Aurora"
    direct = unmixing.unmix_sample(fcs.load_sample(demo_files[-1]), um)
    np.testing.assert_allclose(unmixed.get_events(source="raw"), direct.get_events(source="raw"), rtol=1e-5)

    ws = Workspace.load(root / "My Exp.ssw")
    assert ws.groups["Unmixed"] == ["PBMC_full_panel.fcs"]
    assert "Beads_PE_X.fcs" in ws.groups["Reference Group"]
    assert ws.samples["PBMC_full_panel.fcs"].event_count == unmixed.event_count
    assert "Unmixing" in ws.unmixers

    with pytest.raises(FileExistsError):
        experiment.export_experiment(tmp_path, "My Exp", specs, um, as_zip=False)


def test_export_refuses_broken_experiments(demo_files, tmp_path):
    specs = unmixing.guess_files(demo_files)
    um, _ = unmixing.build_unmixer(specs)
    for s in specs:
        if s.role == "single_stain":
            s.name = "PE"
    with pytest.raises(ValueError, match="Two single-stain controls"):
        experiment.export_experiment(tmp_path, "x", specs, um)
    assert list(tmp_path.iterdir()) == []


def test_same_file_names_from_different_folders(tmp_path):
    a, b = tmp_path / "day1" / "s.fcs", tmp_path / "day2" / "s.fcs"
    specs = [unmixing.ControlSpec(a, "", "sample"), unmixing.ControlSpec(b, "", "sample")]
    assert experiment.layout(specs) == {a: ("Samples", "s.fcs"), b: ("Samples", "s (2).fcs")}


def test_default_name(demo_files):
    specs = unmixing.guess_files(demo_files)
    assert experiment.default_name(specs) == "PBMC full panel"
    assert experiment.safe_name('a/b:c*?') == "a_b_c__"
