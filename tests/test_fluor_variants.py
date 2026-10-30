"""Per-cell fluor variation (AutoSpectral-style), on synthetic data where each
cell's true fluor spectrum is one of a few variants."""

import time

import numpy as np
import pytest

from sidescatter.core import control_cleanup, experiment, fcs, fluor_variants, spectral, unmixing

from synthetic_af import DETECTORS, fluor_signatures

D = len(DETECTORS)
F = fluor_signatures(8, seed=1)
SHIFT = np.roll(F[2], 1) * 0.3
TRUE_VARIANTS = [spectral.normalise(F[2] + t * SHIFT) for t in (-0.3, 0.0, 0.3)]


def _control(n, rng):
    which = rng.integers(0, 3, n)
    amount = rng.lognormal(np.log(20000), 0.3, n)
    return np.array([TRUE_VARIANTS[w] for w in which]) * amount[:, None] + rng.normal(0, 30, (n, D))


def _sample(m, rng):
    """Cells with fluor 2 (a random variant) and/or fluor 3; returns (Y, true abundances)."""
    A = np.zeros((m, 8))
    A[:, 2] = rng.lognormal(np.log(20000), 0.5, m) * (rng.random(m) < 0.5)
    A[:, 3] = rng.lognormal(np.log(3000), 0.3, m) * (rng.random(m) < 0.5)
    which = rng.integers(0, 3, m)
    Y = np.array([TRUE_VARIANTS[w] for w in which]) * A[:, [2]] + A[:, [3]] * F[3]
    return Y + rng.normal(0, 30, (m, D)), A


def _unmixer(ctrl):
    M = F.copy()
    M[2] = spectral.normalise(np.median(ctrl, axis=0))
    return spectral.Unmixer(list(DETECTORS), [spectral.Reference(f"F{i}", M[i]) for i in range(8)], "ols")


def test_variants_are_close_to_the_signature():
    ctrl = _control(1500, np.random.default_rng(0))
    sig = spectral.normalise(np.median(ctrl, axis=0))
    V = fluor_variants.variants(ctrl, sig)
    assert 2 <= len(V) <= 100
    assert (spectral.cosine(V, sig[None])[:, 0] >= 0.98).all()
    assert np.allclose(V.max(axis=1), 1.0)
    assert len(fluor_variants.variants(ctrl[:5], sig)) == 0   # too few events for a map


@pytest.mark.parametrize("method", ["ols", "wls"])
def test_variation_cuts_spillover_spread(method):
    rng = np.random.default_rng(1)
    ctrl = _control(1500, rng)
    Y, truth = _sample(20000, rng)
    um = _unmixer(ctrl)
    um.method = method
    um.background_var = np.full(D, 900.0)
    before = um.unmix(Y)
    um.variants = [fluor_variants.variants(ctrl, um.matrix[2]) if i == 2 else np.empty((0, D))
                   for i in range(8)]
    um.variant_thresholds = np.full(8, 300.0)
    after = um.unmix(Y)
    # Fluors that fluor 2's variation spills into: less spread on cells without them.
    spread = lambda A, j: spectral.robust_sd(A[truth[:, j] == 0, j])
    assert sum(spread(after, j) for j in (4, 6)) < 0.8 * sum(spread(before, j) for j in (4, 6))
    err = lambda A: np.abs(A[:, [0, 1, 3, 4, 5, 6, 7]] - truth[:, [0, 1, 3, 4, 5, 6, 7]]).mean()
    assert err(after) < err(before)


def test_variant_channels_and_roundtrip():
    rng = np.random.default_rng(2)
    ctrl = _control(1500, rng)
    um = _unmixer(ctrl)
    um.variants = [fluor_variants.variants(ctrl, um.matrix[2]) if i == 2 else np.empty((0, D))
                   for i in range(8)]
    um.variant_thresholds = np.full(8, 300.0)
    um.variant_channels = True
    assert um.output_names[-8:] == [f"F{i} Variant" for i in range(8)]
    Y, _ = _sample(2000, rng)
    A = um.unmix(Y)
    assert A.shape == (2000, 16)
    assert A[:, 8 + 2].max() >= 1 and (A[:, 8 + 0] == 0).all()
    again = spectral.Unmixer.from_dict(um.to_dict())
    np.testing.assert_allclose(again.unmix(Y), A)


def test_hotspots_point_at_the_varying_fluor():
    ctrl = _control(1500, np.random.default_rng(3))
    M = _unmixer(ctrl).matrix
    sets = [fluor_variants.variants(ctrl, M[2]) if i == 2 else np.empty((0, D)) for i in range(8)]
    H = fluor_variants.hotspots(M, [f"F{i}" for i in range(8)], sets)
    assert np.isnan(H[0]).all() and np.isnan(H[2, 2])
    assert np.nanmax(H[2]) > 0


def test_built_from_control_files(tmp_path):
    """Variants come out of build_unmixer, are used only when turned on, and are saved."""
    rng = np.random.default_rng(4)
    paths = []
    for i in range(4):
        events = (_control(3000, rng) if i == 2 else
                  rng.lognormal(np.log(20000), 0.3, (3000, 1)) * F[i]) + rng.normal(0, 30, (3000, D))
        events[: 1500] = rng.normal(0, 30, (1500, D))   # dim half: the internal negative
        p = tmp_path / f"Beads_F{i}.fcs"
        fcs.write_fcs(p, events.astype(np.float32), list(DETECTORS), ranges=[4194304] * D)
        paths.append(p)
    specs = [unmixing.ControlSpec(p, f"F{i}", "single_stain", "beads", negative="internal")
             for i, p in enumerate(paths)]
    um, res = unmixing.build_unmixer(specs, autofluorescence=False, cleanup=control_cleanup.Settings())
    assert len(res[2].variants) >= 2 and not um.per_cell_variants
    assert unmixing.variant_hotspots(um, res) is not None
    um, res = unmixing.build_unmixer(specs, autofluorescence=False,
                                     variation=fluor_variants.Settings(enabled=True))
    assert um.per_cell_variants and len(um.variant_thresholds) == 4
    setup = experiment.Setup("E", specs)
    setup.variation = fluor_variants.Settings(True, 50, 0.97, True)
    assert experiment.Setup.from_dict(setup.to_dict()).variation == setup.variation
    assert not experiment.Setup("F").variation.enabled
    sample = fcs.load_sample(paths[2])
    events, pnn, *_ = unmixing.unmixed_events(sample, um)
    assert pnn[-4:] == ["F0-A", "F1-A", "F2-A", "F3-A"]


def test_speed_on_a_big_panel():
    """30 fluors with variants: 200,000 events in well under a second per 10,000."""
    rng = np.random.default_rng(5)
    k = 30
    M = fluor_signatures(k, seed=6)
    sets = []
    for i in range(k):
        jitter = M[i] * (1 + 0.05 * rng.standard_normal((40, D)))
        sets.append(jitter / jitter.max(axis=1, keepdims=True))
    um = spectral.Unmixer(list(DETECTORS), [spectral.Reference(f"F{i}", M[i]) for i in range(k)], "ols",
                          variants=sets, variant_thresholds=np.full(k, 200.0))
    n = 200_000
    A = rng.lognormal(np.log(2000), 1.0, (n, k)) * (rng.random((n, k)) < 0.3)
    Y = A @ M + rng.normal(0, 30, (n, D))
    t = time.perf_counter()
    um.unmix(Y)
    assert time.perf_counter() - t < 20
