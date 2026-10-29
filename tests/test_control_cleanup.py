"""Control clean-up and the spectra check (AutoSpectral-style), on synthetic data
where the true fluor signature is known."""

from pathlib import Path

import numpy as np
import pytest

from sidescatter.core import control_cleanup, experiment, fcs, spectra_library, spectral, unmixing

from synthetic_af import AF_TYPES, DETECTORS, fluor_signatures

TOP = 4194304.0
FLUORS = fluor_signatures(4, seed=3)
SMALL_AF, BIG_AF = AF_TYPES[0][1], AF_TYPES[1][1]


def _cells(n, rng, big_share=0.3):
    """Scatter and AF of n cells: small lymphocyte-like cells with dim AF and big
    macrophage-like cells with bright AF of another shape."""
    big = rng.random(n) < big_share
    fsc = np.where(big, rng.lognormal(np.log(9e5), 0.12, n), rng.lognormal(np.log(4e5), 0.12, n))
    ssc = np.where(big, rng.lognormal(np.log(6e5), 0.15, n), rng.lognormal(np.log(1.5e5), 0.15, n))
    af = np.where(big[:, None], 5000.0 * BIG_AF, 500.0 * SMALL_AF) * rng.lognormal(0, 0.2, (n, 1))
    return big, fsc, ssc, af


def _write(path, fsc, ssc, Y, fsc_h=None):
    fsc_h = fsc / 1.3 if fsc_h is None else fsc_h
    data = np.column_stack([fsc, fsc_h, ssc, Y]).astype(np.float32)
    fcs.write_fcs(path, data, ["FSC-A", "FSC-H", "SSC-A", *DETECTORS],
                  ranges=[TOP] * data.shape[1])
    return path


def _noisy(mean, rng):
    return rng.poisson(np.clip(mean, 0, None)).astype(float) + rng.normal(0, 40, mean.shape)


@pytest.fixture
def experiment_files(tmp_path):
    """Unstained cells and a single stain on the big (high-AF) cells only."""
    rng = np.random.default_rng(0)
    _, fsc, ssc, af = _cells(20000, rng)
    unstained = _write(tmp_path / "Cells_Unstained.fcs", fsc, ssc, _noisy(af, rng))
    big, fsc, ssc, af = _cells(20000, rng)
    stain = np.where(big, rng.lognormal(np.log(30000), 0.3, len(big)), 0.0)
    Y = _noisy(af + stain[:, None] * FLUORS[1], rng)
    single = _write(tmp_path / "Cells_X1.fcs", fsc, ssc, Y)
    return unstained, single


def _specs(unstained, single):
    return [unmixing.ControlSpec(Path(unstained), "Unstained", "unstained", "cells"),
            unmixing.ControlSpec(Path(single), "X1", "single_stain", "cells")]


def test_scatter_matching_removes_af_from_the_signature(experiment_files):
    specs = _specs(*experiment_files)
    _, plain = unmixing.build_unmixer(specs, autofluorescence=False)
    _, clean = unmixing.build_unmixer(specs, autofluorescence=False,
                                      cleanup=control_cleanup.Settings())
    truth = FLUORS[1] / FLUORS[1].max()
    err_plain = np.abs(plain[0].signature - truth).max()
    err_clean = np.abs(clean[0].signature - truth).max()
    assert err_clean < err_plain / 3
    assert spectral.cosine(clean[0].signature[None], truth[None])[0, 0] > 0.999
    r = clean[0]
    assert r.cleanup["used"] == 500
    assert r.cleanup["background"].startswith("scatter-matched")
    np.testing.assert_allclose(r.basic_signature, plain[0].signature)


def test_cleanup_off_keeps_the_plain_signature(experiment_files):
    specs = _specs(*experiment_files)
    _, a = unmixing.build_unmixer(specs, autofluorescence=False)
    _, b = unmixing.build_unmixer(specs, autofluorescence=False,
                                  cleanup=control_cleanup.Settings(enabled=False))
    np.testing.assert_array_equal(a[0].signature, b[0].signature)
    assert b[0].cleanup is None


def test_saturated_events_and_doublets_are_dropped(tmp_path):
    rng = np.random.default_rng(1)
    n = 5000
    _, fsc, ssc, af = _cells(n, rng, big_share=0.0)
    stain = rng.lognormal(np.log(50000), 0.3, n)
    Y = _noisy(af + stain[:, None] * FLUORS[2], rng)
    Y[:200] = TOP                        # saturated in every detector
    fsc_h = fsc / 1.3
    fsc_h[200:400] = fsc[200:400] / 2.6  # doublets: twice the area for the height
    path = _write(tmp_path / "Cells_X2.fcs", fsc, ssc, Y, fsc_h)
    sample = fcs.load_sample(path)
    sat_ok, single = unmixing.cleanup_masks(sample, DETECTORS, control_cleanup.Settings())
    assert not sat_ok[:200].any() and sat_ok[200:].all()
    assert (~single[200:400]).mean() > 0.95 and single[400:].mean() > 0.99
    spec = unmixing.ControlSpec(path, "X2", "single_stain", "cells", negative="internal")
    _, res = unmixing.build_unmixer([spec], autofluorescence=False,
                                    cleanup=control_cleanup.Settings(brightest=0))
    assert res[0].cleanup["saturated"] == 200
    assert res[0].cleanup["doublets"] > 0
    assert not res[0].used_mask[:200].any()


def test_brightest_keeps_the_top_events():
    values = np.arange(10.0)
    mask = np.array([True] * 8 + [False] * 2)
    kept = control_cleanup.brightest(values, mask, 3)
    assert np.flatnonzero(kept).tolist() == [5, 6, 7]
    assert control_cleanup.brightest(values, mask, 0) is mask


def test_high_af_positives_are_dropped(experiment_files, tmp_path):
    rng = np.random.default_rng(2)
    n = 10000
    _, fsc, ssc, af = _cells(n, rng)
    weird = rng.random(n) < 0.05          # far brighter AF than any unstained cell
    af[weird] *= 20
    stain = rng.lognormal(np.log(30000), 0.3, n)
    single = _write(tmp_path / "Cells_X1b.fcs", fsc, ssc, _noisy(af + stain[:, None] * FLUORS[1], rng))
    specs = _specs(experiment_files[0], single)
    _, plain = unmixing.build_unmixer(specs, autofluorescence=False,
                                      cleanup=control_cleanup.Settings(brightest=0))
    specs[1].remove_high_af = True
    _, res = unmixing.build_unmixer(specs, autofluorescence=False,
                                    cleanup=control_cleanup.Settings(brightest=0))
    dropped = res[0].positive_mask & ~res[0].used_mask
    assert res[0].cleanup["high_af"] > 0
    assert weird[dropped].mean() > 0.8
    truth = FLUORS[1] / FLUORS[1].max()
    assert np.abs(res[0].signature - truth).max() < np.abs(plain[0].signature - truth).max()


def test_settings_roundtrip_and_old_setups_have_cleanup_off(experiment_files):
    setup = experiment.Setup("E", _specs(*experiment_files))
    assert setup.cleanup.enabled
    setup.cleanup.neighbours = 3
    setup.specs[1].remove_high_af = True
    setup.specs[1].signature_choice = "accepted"
    again = experiment.Setup.from_dict(setup.to_dict())
    assert again.cleanup.neighbours == 3 and again.cleanup.enabled
    assert again.specs[1].remove_high_af and again.specs[1].signature_choice == "accepted"
    old = setup.to_dict()
    del old["cleanup"]
    assert not experiment.Setup.from_dict(old).cleanup.enabled


# ----- the spectra check -----------------------------------------------------

def test_detector_bands_follow_the_instrument():
    bands = spectra_library.detector_bands(DETECTORS)
    assert bands[DETECTORS.index("V1-A")] == (405.0, 428.0, 15.0)
    assert bands[DETECTORS.index("R8-A")] == (640.0, 812.0, 34.0)
    # BD and Sony names carry the band centre; an older 16-detector blue module isn't guessed.
    assert spectra_library.detector_bands(["B530-A", "405-450"]) == [(488.0, 530.0, 20.0),
                                                                     (405.0, 450.0, 20.0)]
    assert spectra_library.detector_bands([f"B{i}-A" for i in range(1, 17)])[0] is None


@pytest.mark.parametrize("fluor,peak", [("BV421", "V1-A"), ("PE", "YG1-A"), ("APC", "R1-A"),
                                        ("PE Cy7", "YG9-A"), ("FITC", "B2-A")])
def test_predicted_signatures_peak_in_the_right_detector(fluor, peak):
    sig, _ = spectra_library.predict(fluor, DETECTORS)
    assert DETECTORS[int(np.argmax(sig))] == peak


def test_check_flags_the_wrong_control():
    bv421, _ = spectra_library.predict("BV421", DETECTORS)
    ok = spectra_library.check("BV421", bv421, DETECTORS)
    assert ok.passed and ok.kind == "predicted"
    bv605, _ = spectra_library.predict("BV605", DETECTORS)
    bad = spectra_library.check("BV421", bv605, DETECTORS)
    assert not bad.passed
    assert spectra_library.check("Unknown dye 123", bv421, DETECTORS) is None


def test_saved_library_wins_and_can_replace_a_bad_signature(tmp_path, experiment_files):
    lib = spectra_library.SpectraLibrary(tmp_path / "lib.json")
    truth = FLUORS[1] / FLUORS[1].max()
    lib.add("X1", DETECTORS, truth, "cells", "earlier.fcs")
    lib.save()
    lib = spectra_library.SpectraLibrary(tmp_path / "lib.json")
    assert lib.find("x1", DETECTORS, "cells")["source"] == "earlier.fcs"
    assert lib.find("X1", DETECTORS[:-1]) is None

    specs = _specs(*experiment_files)
    _, res = unmixing.build_unmixer(specs, autofluorescence=False, library=lib)
    assert res[0].check.kind == "saved"
    specs[1].signature_choice = "library"
    um, res = unmixing.build_unmixer(specs, autofluorescence=False, library=lib)
    assert res[0].signature_source == "library"
    np.testing.assert_allclose(um.references[0].signature, truth, atol=1e-5)
    assert res[0].measured_signature is not None
    assert lib.remove("X1", DETECTORS, "cells") and lib.find("X1", DETECTORS) is None
