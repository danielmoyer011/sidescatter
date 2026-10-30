"""Exchanging spectra and control files with AutoSpectral."""

import csv
from pathlib import Path

import numpy as np
import pytest

from sidescatter.core import autospectral_io, experiment, spectral, unmixing


def _r_csv(path, names, detectors, rows):
    """A spectra file as R's write.csv writes it (quoted strings, blank first header cell)."""
    lines = ['"",' + ",".join(f'"{d}"' for d in detectors)]
    lines += [f'"{n}",' + ",".join(f"{v:.8f}" for v in r) for n, r in zip(names, rows)]
    Path(path).write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def demo(spectral_dir):
    specs = unmixing.guess_controls(sorted((spectral_dir / "controls").glob("*.fcs")))
    samples = {}
    um, results = unmixing.build_unmixer(specs, samples=samples, af_mode="per_cell")
    return specs, um, results, samples


def test_reads_r_spectra_and_autofluorescence_files(tmp_path):
    dets = ["V1-A", "V2-A", "B1-A"]
    p = _r_csv(tmp_path / "Spectra_spectra_2026.csv", ["BV421", "FITC", "AF"], dets,
               [[1000, 200, 0], [0, 10, 50], [3, 2, 1]])
    s = autospectral_io.read_spectra(p)
    assert s.detectors == dets and s.names == ["BV421", "FITC"]
    np.testing.assert_allclose(s.fluors[0], [1, 0.2, 0])
    assert len(s.af) == 1
    af = autospectral_io.read_spectra(_r_csv(tmp_path / "Lung Autofluorescence 2026.csv",
                                             ["1", "2"], dets, [[1, 2, 3], [3, 2, 1]]))
    assert af.names == [] and len(af.af) == 2
    assert s.signature("bv 421") is not None and s.signature("PE") is None
    with pytest.raises(ValueError, match="another instrument"):
        s.on(["V1-A", "R1-A"])
    np.testing.assert_allclose(s.on(["B1-A", "V1-A", "V2-A"]).fluors[0], [0, 1, 0.2])
    (tmp_path / "bad.csv").write_text("x\n1\n")
    with pytest.raises(ValueError):
        autospectral_io.read_spectra(tmp_path / "bad.csv")


def test_export_reads_back_and_writes_a_control_file(demo, tmp_path):
    specs, um, results, _ = demo
    written = autospectral_io.export(tmp_path, specs, um, results)
    assert [p.name for p in written] == ["Sidescatter_spectra.csv", "Sidescatter_Autofluorescence.csv",
                                         "Sidescatter_control_file.csv"]
    back = autospectral_io.read_spectra(written[0])
    fluors = [r for r in um.references if not r.is_autofluorescence]
    assert back.names == [r.name for r in fluors]
    np.testing.assert_allclose(back.fluors, [r.signature for r in fluors], atol=1e-5)
    assert len(autospectral_io.read_spectra(written[1]).af) == len(um.af_library)

    with open(written[2], newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == autospectral_io.CONTROL_COLUMNS
    by_file = {r["filename"]: r for r in rows}
    assert by_file["PBMC_unstained.fcs"]["fluorophore"] == "AF"
    assert by_file["PBMC_unstained.fcs"]["universal.negative"] == "PBMC_unstained.fcs"
    assert by_file["Beads_unstained.fcs"]["fluorophore"] == "Negative"
    pe = by_file["Beads_PE_X.fcs"]
    assert (pe["fluorophore"], pe["control.type"], pe["universal.negative"], pe["channel"]) == \
        ("PE", "beads", "Beads_unstained.fcs", "B4-A")
    assert by_file["Dead_PO_DCM.fcs"]["is.viability"] == "TRUE"
    assert pe["is.viability"] == "FALSE"


def test_imported_spectra_replace_and_add(demo, tmp_path):
    specs, um, _, samples = demo
    dets = um.detectors
    pe = next(r.signature for r in um.references if r.name == "PE")
    shifted = spectral.normalise(np.roll(pe, 1) + pe)
    extra = spectral.normalise(np.linspace(1, 2, len(dets)))
    af = [spectral.normalise(np.linspace(2, 1, len(dets))), spectral.normalise(np.ones(len(dets)))]
    p = _r_csv(tmp_path / "x.csv", ["PE", "NewDye", "AF1", "AF2"], dets[::-1],
               [shifted[::-1], extra[::-1], af[0][::-1], af[1][::-1]])
    imported = autospectral_io.read_spectra(p)
    um2, res = unmixing.build_unmixer(specs, samples=samples, af_mode="per_cell", imported=imported)
    by_name = {r.name: r for r in um2.references}
    np.testing.assert_allclose(by_name["PE"].signature, shifted, atol=1e-6)
    np.testing.assert_allclose(by_name["NewDye"].signature, extra, atol=1e-6)
    r = next(r for r in res if r.spec.name == "PE")
    assert r.signature_source == "imported" and r.measured_signature is not None
    np.testing.assert_allclose(um2.af_library, af, atol=1e-6)
    um3, _ = unmixing.build_unmixer(specs, samples=samples, af_mode="fluors", imported=imported)
    assert sum(r.is_autofluorescence for r in um3.references) == 2

    setup = experiment.Setup("E", specs, imported=imported)
    again = experiment.Setup.from_dict(setup.to_dict()).imported
    assert again.names == imported.names and np.allclose(again.af, imported.af)
    assert experiment.Setup.from_dict(experiment.Setup("F").to_dict()).imported is None
