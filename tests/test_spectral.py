import csv

import numpy as np
import pytest

from sidescatter.core import fcs, spectral, unmixing


def _random_signatures(k, d, rng):
    centres = np.linspace(0, d - 1, k)
    x = np.arange(d)
    return np.array([spectral.normalise(np.exp(-0.5 * ((x - c) / 3.0) ** 2) + 0.02) for c in centres])


def test_ols_recovers_known_abundances():
    rng = np.random.default_rng(1)
    M = _random_signatures(6, 30, rng)
    A = rng.uniform(0, 1e4, size=(500, 6))
    A_hat = spectral.unmix_ols(A @ M, M)
    np.testing.assert_allclose(A_hat, A, rtol=1e-8, atol=1e-6)


def test_wls_recovers_known_abundances_without_noise():
    rng = np.random.default_rng(2)
    M = _random_signatures(5, 25, rng)
    A = rng.uniform(0, 1e4, size=(300, 5))
    A_hat = spectral.unmix_wls(A @ M, M, background_var=np.full(25, 100.0), chunk=64)
    np.testing.assert_allclose(A_hat, A, rtol=1e-6, atol=1e-4)


def test_wls_beats_ols_under_photon_noise():
    rng = np.random.default_rng(3)
    M = _random_signatures(8, 40, rng)
    A = rng.uniform(0, 5e3, size=(4000, 8))
    Y = rng.poisson(A @ M + 50).astype(float) - 50
    bg = np.full(40, 50.0)
    err_ols = np.abs(spectral.unmix_ols(Y, M) - A).mean()
    err_wls = np.abs(spectral.unmix_wls(Y, M, bg) - A).mean()
    assert err_wls < err_ols


def test_signature_is_peak_normalised_and_background_subtracted():
    neg = np.full((100, 4), 10.0)
    pos = neg + np.array([0, 50, 100, 25])
    np.testing.assert_allclose(spectral.extract_signature(pos, neg), [0, 0.5, 1.0, 0.25])


def test_similarity_and_complexity():
    M = np.array([[1.0, 0.0], [0.0, 1.0]])
    np.testing.assert_allclose(spectral.similarity_matrix(M), np.eye(2))
    assert spectral.complexity_index(M) == pytest.approx(1.0)


def test_fine_tune_undoes_known_spillover():
    rng = np.random.default_rng(4)
    true = rng.uniform(0, 100, size=(50, 3))
    spill = np.array([[1, 0.1, 0], [0, 1, 0.05], [0.02, 0, 1]])
    np.testing.assert_allclose(spectral.apply_fine_tune(true @ spill, spill), true)


def test_unmixer_roundtrip_dict():
    um = spectral.Unmixer(["a", "b"], [spectral.Reference("X", np.array([1.0, 0.5]))], "wls",
                          np.array([1.0, 2.0]))
    back = spectral.Unmixer.from_dict(um.to_dict())
    assert back.names == ["X"] and back.method == "wls"
    np.testing.assert_allclose(back.matrix, um.matrix)


def test_guess_controls_classifies_files(spectral_dir):
    specs = {s.path.name: s for s in unmixing.guess_controls(sorted((spectral_dir / "controls").glob("*.fcs")))}
    assert specs["Beads_unstained.fcs"].role == "unstained"
    assert specs["Beads_unstained.fcs"].carrier == "beads"
    assert specs["PBMC_unstained.fcs"].carrier == "cells"
    assert specs["Beads_PE_X.fcs"].name == "PE_X"


def test_aurora_signatures_match_flowspecs(spectral_dir):
    """Signatures from the Cytek Aurora controls match those published with flowSpecs."""
    specs = unmixing.guess_controls(sorted((spectral_dir / "controls").glob("*.fcs")))
    um, results = unmixing.build_unmixer(specs)
    with open(spectral_dir / "flowSpecs_reference_signatures.csv") as f:
        rows = list(csv.reader(f))
    assert rows[0][1:] == um.detectors
    reference = {r[0]: np.array(r[1:], dtype=float) for r in rows[1:]}
    for ref in um.references:
        if ref.is_autofluorescence:
            continue
        theirs = reference[ref.name]
        cos = ref.signature @ theirs / np.linalg.norm(ref.signature) / np.linalg.norm(theirs)
        assert cos > 0.995, ref.name
    # beads use the unstained beads as negative, the dead-cell stain uses unstained cells
    by_name = {r.spec.name: r for r in results}
    assert by_name["PE_X"].negative_source == "Beads_unstained.fcs"
    assert by_name["PO_DCM"].negative_source.endswith("unstained.fcs")
    assert "Bead" not in by_name["PO_DCM"].negative_source


def test_unmix_full_panel(spectral_dir):
    specs = unmixing.guess_controls(sorted((spectral_dir / "controls").glob("*.fcs")))
    um, _ = unmixing.build_unmixer(specs, method="wls")
    full = fcs.load_sample(spectral_dir / "PBMC_full_panel.fcs")
    unmixed = unmixing.unmix_sample(full, um)
    assert unmixed.event_count == full.event_count
    assert "PE_X" in unmixed.pnn_labels and "Autofluorescence" in unmixed.pnn_labels
    qc = unmixing.unmixing_qc(full, um)
    assert qc["relative_residual_median"] < 0.1
