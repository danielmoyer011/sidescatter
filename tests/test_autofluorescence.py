"""Per-cell autofluorescence matching, checked on synthetic data with a known answer."""

import csv
import io
import zipfile
from pathlib import Path

import numpy as np
import pytest

from sidescatter.core import experiment, fcs, spectral, unmixing
from sidescatter.core.workspace import Workspace

from synthetic_af import AF_TYPES, DETECTORS, cells, fluor_signatures

F = fluor_signatures(16)
UNSTAINED_SPEC = unmixing.ControlSpec(Path("unstained.fcs"), "", "unstained", "cells")


def _unmixer(mode, Y_unstained, n=1, method="ols", spec=UNSTAINED_SPEC, **kw):
    um = spectral.Unmixer(list(DETECTORS), [spectral.Reference(f"F{i}", s) for i, s in enumerate(F)],
                          method, spectral.robust_sd(Y_unstained) ** 2)
    return unmixing.add_autofluorescence(um, [(spec, Y_unstained)], mode, n, **kw)


@pytest.fixture(scope="module")
def unstained():
    return cells(30_000, seed=1)[0]


def test_per_cell_fit_is_exact_without_noise():
    rng = np.random.default_rng(0)
    L = np.array([t[1] for t in AF_TYPES])
    A = rng.uniform(0, 5000, (400, len(F)))
    pick = rng.integers(len(L), size=400)
    c = rng.uniform(500, 8000, 400)
    Y = A @ F + c[:, None] * L[pick]
    A_hat, af, index = spectral.unmix_af_per_cell(Y, F, L)
    np.testing.assert_array_equal(index, pick + 1)
    np.testing.assert_allclose(af, c, rtol=1e-6)
    np.testing.assert_allclose(A_hat, A, atol=1e-4)
    # Only the allowed entries are used.
    allowed = np.zeros(len(L), bool)
    allowed[0] = True
    assert set(spectral.unmix_af_per_cell(Y, F, L, allowed=allowed)[2]) <= {0, 1}


def test_library_finds_the_af_types(unstained):
    lib, sizes = spectral.af_library(unstained, 30, F)
    assert 4 <= len(lib) <= 30 and sizes[0] == sizes.max()
    sim = spectral.cosine(np.array([t[1] for t in AF_TYPES]), lib).max(axis=1)
    assert (sim > 0.98).all()  # every AF type has a close library entry
    off_diag = spectral.cosine(lib, lib)[~np.eye(len(lib), dtype=bool)]
    assert off_diag.max() <= 0.99  # near-duplicates dropped


@pytest.mark.parametrize("method", ["ols", "wls"])
def test_per_cell_leaves_less_spread_than_af_as_fluors(unstained, method):
    """Unstained cells with mixed AF should unmix to zero in every fluor channel."""
    test = cells(20_000, seed=2)[0]
    spread = {}
    for mode, n in (("fluors", 1), ("fluors", 3), ("per_cell", 1)):
        A = _unmixer(mode, unstained, n, method).unmix(test)
        spread[mode, n] = spectral.robust_sd(A[:, :len(F)]).sum()
    assert spread["per_cell", 1] < 0.25 * spread["fluors", 1]
    assert spread["per_cell", 1] < 0.5 * spread["fluors", 3]


def test_per_cell_recovers_stained_cells(unstained):
    rng = np.random.default_rng(5)
    truth = np.zeros((20_000, len(F)))
    truth[:, 3] = rng.choice([0, 3000, 20000], 20_000)
    truth[:, 9] = rng.choice([0, 8000], 20_000)
    Y, af_type = cells(20_000, truth, F, seed=6)
    err = {}
    for mode, n in (("fluors", 1), ("per_cell", 1)):
        A = _unmixer(mode, unstained, n, "wls").unmix(Y)
        err[mode] = np.sqrt(((A[:, :len(F)] - truth) ** 2).mean())
    assert err["per_cell"] < 0.5 * err["fluors"]
    # AF Index tells the AF types apart: most macrophages share one entry, distinct
    # from the one most lymphocytes use.
    index = _unmixer("per_cell", unstained).unmix(Y)[:, -1].astype(int)
    common = [np.bincount(index[af_type == t]).argmax() for t in range(len(AF_TYPES))]
    assert common[0] != common[1]


def test_outputs_channels_and_roundtrip(unstained):
    um = _unmixer("per_cell", unstained)
    assert um.per_cell_af and um.output_names[-2:] == ["Autofluorescence", "AF Index"]
    labels = [pnn for pnn, _ in um.channel_labels()]
    assert labels[-2:] == ["AF-A", "AF Index"] and len(labels) == len(F) + 2
    Y = unstained[:2000]
    A = um.unmix(Y)
    assert A.shape == (2000, len(F) + 2)
    assert ((A[:, -1] >= 0) & (A[:, -1] <= len(um.af_library))).all()
    assert np.abs(um.residuals(Y, A)).mean() < np.abs(Y).mean() * 0.2
    back = spectral.Unmixer.from_dict(um.to_dict())
    np.testing.assert_allclose(back.unmix(Y), A)
    assert back.af_sources == um.af_sources


def test_drawn_gates_and_auto_off(unstained):
    # Gate the bright macrophages on two detectors where they stand out.
    mac = AF_TYPES[1][1]
    xi, yi = int(np.argmax(mac)), 50
    x, y = unstained[:, xi], unstained[:, yi]
    lo_x, hi_x = np.percentile(x, [80, 100])
    lo_y, hi_y = np.percentile(y, [80, 100])
    gate = {"name": "Macrophages", "x": DETECTORS[xi], "y": DETECTORS[yi],
            "vertices": [[lo_x, lo_y], [hi_x + 1, lo_y], [hi_x + 1, hi_y + 1], [lo_x, hi_y + 1]]}
    spec = unmixing.ControlSpec(Path("u.fcs"), "", "unstained", "cells", af_gates=[gate])
    um = _unmixer("per_cell", unstained, spec=spec, auto=False)
    assert len(um.af_library) == 1
    assert spectral.cosine(um.af_library[0], mac)[0, 0] > 0.97
    both = _unmixer("per_cell", unstained, spec=spec, auto=True)
    assert len(both.af_library) > 1
    np.testing.assert_allclose(both.af_library[0], um.af_library[0])  # drawn gates come first


def test_in_polygon_counts_the_edge():
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    x = np.array([0.5, 1.5, 0.0, 0.5])
    y = np.array([0.5, 0.5, 0.5, 1.0])
    assert unmixing.in_polygon(x, y, square).tolist()[:2] == [True, False]


def test_af_source_restricts_library():
    a = cells(10_000, seed=7)[0]
    # Second "tissue": only lymphocyte-like AF.
    rng = np.random.default_rng(9)
    lymph_only = rng.poisson(600 * rng.lognormal(0, 0.3, (8000, 1)) * AF_TYPES[0][1]).astype(float)
    um = spectral.Unmixer(list(DETECTORS), [spectral.Reference(f"F{i}", s) for i, s in enumerate(F)])
    specs = [unmixing.ControlSpec(Path("tissue_a.fcs"), "", "unstained", "cells"),
             unmixing.ControlSpec(Path("tissue_b.fcs"), "", "unstained", "cells")]
    unmixing.add_autofluorescence(um, list(zip(specs, (a, lymph_only))), "per_cell")
    assert set(um.af_sources) <= {"tissue_a.fcs", "tissue_b.fcs"}
    index = um.unmix(a, af_source="tissue_b.fcs")[:, -1].astype(int)
    used = {um.af_sources[i - 1] for i in np.unique(index) if i > 0}
    assert used <= {"tissue_b.fcs"}


@pytest.fixture
def demo_files(spectral_dir):
    return sorted((spectral_dir / "controls").glob("*.fcs")) + [spectral_dir / "PBMC_full_panel.fcs"]


def test_build_unmix_export_and_reload(demo_files, tmp_path):
    specs = unmixing.guess_files(demo_files)
    um, _ = unmixing.build_unmixer(specs, "wls", af_mode="per_cell", af_library_size=20)
    assert um.per_cell_af and 1 <= len(um.af_library) <= 20
    full = fcs.load_sample(demo_files[-1])
    unmixed = unmixing.unmix_sample(full, um)
    assert unmixed.pnn_labels[-2:] == ["AF-A", "AF Index"]
    assert "AF Index" not in unmixing.unmix_sample(full, um, keep_autofluorescence=False).pnn_labels

    check = unmixing.af_check(specs, um)
    per, flu = check["methods"]["per_cell"], check["methods"]["fluors"]
    assert per["spread"].shape == (len(check["fluors"]),)
    assert per["spread"].sum() <= flu["spread"].sum() * 1.05  # PBMC AF is even; no worse

    result = experiment.export_experiment(tmp_path, "AF", specs, um)
    with zipfile.ZipFile(result.path) as zf:
        report = zf.read("AF/QC report.txt").decode()
        matrix = list(csv.reader(io.StringIO(zf.read("AF/AF unmixing matrix.csv").decode())))
        ws_bytes = zf.read("AF/AF.ssw")
    assert "Autofluorescence check" in report and "per-cell matching" in report
    assert sum(r[0].startswith("AF library") for r in matrix) == len(um.af_library)
    (tmp_path / "AF.ssw").write_bytes(ws_bytes)

    ws = Workspace()
    ws.add_unmixer("U", um)
    unstained = next(s for s in specs if s.path.name == "PBMC_unstained.fcs")
    sid = ws.unmix_file(demo_files[-1], "U", "Unmixed", full, af_source=str(unstained.path))
    assert ws.sources[sid]["af_source"] == str(unstained.path)
    ws.save(tmp_path / "w.ssw")
    back = Workspace.load(tmp_path / "w.ssw")
    np.testing.assert_allclose(back.samples[sid].get_events(source="raw"),
                               ws.samples[sid].get_events(source="raw"), rtol=1e-5)


def test_setup_keeps_af_settings():
    spec = unmixing.ControlSpec(Path("u.fcs"), "", "unstained", "cells",
                                af_gates=[{"name": "g", "x": "V1-A", "y": "B2-A", "vertices": [[0, 0]]}])
    sample = unmixing.ControlSpec(Path("s.fcs"), "", "sample", af_source="u.fcs")
    s = experiment.Setup("x", [spec, sample], af_mode="per_cell", af_library_size=12, af_auto=False)
    back = experiment.Setup.from_dict(s.to_dict())
    assert (back.af_mode, back.af_library_size, back.af_auto) == ("per_cell", 12, False)
    assert back.specs[0].af_gates == spec.af_gates and back.specs[1].af_source == "u.fcs"
    old = s.to_dict()
    del old["af_mode"]
    assert experiment.Setup.from_dict(old).af_mode == "fluors"  # older setups don't change
