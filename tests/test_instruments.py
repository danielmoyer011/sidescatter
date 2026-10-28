"""Spectral workflow on synthetic files laid out like different instruments.

Nothing in Sidescatter assumes a laser or detector count; these tests build
controls and a stained sample for several channel layouts and check that the
known signatures and abundances come back.
"""

import flowio
import numpy as np
import pytest

from sidescatter.core import fcs, spectral, unmixing


def aurora(lasers):
    sizes = {"UV": 16, "V": 16, "B": 14, "YG": 10, "R": 8}
    return [f"{l}{i}-A" for l in lasers for i in range(1, sizes[l] + 1)]


INSTRUMENTS = {
    "Aurora 3L (V/B/R)": aurora(["V", "B", "R"]),
    "Aurora 5L (UV/V/B/YG/R)": aurora(["UV", "V", "B", "YG", "R"]),
    "Northern Lights 2L (V/B)": aurora(["V", "B"]),
    "BD wavelength names": [f"{l}{w}-A" for l, ws in {
        "UV": range(379, 820, 30), "V": range(427, 820, 25), "B": range(510, 820, 30),
        "YG": range(586, 820, 30), "R": range(670, 820, 25)}.items() for w in ws],
    "Sony-style 180 channels": [f"Ch{i}-A" for i in range(1, 181)],
}
SCATTER = ["FSC-A", "FSC-H", "SSC-A", "SSC-B-A", "Time"]


def _write(path, names, events):
    with open(path, "wb") as fh:
        flowio.create_fcs(fh, events.astype(np.float32).flatten().tolist(), names)


def _make_experiment(tmp_path, detectors, n_fluors=6, seed=0):
    rng = np.random.default_rng(seed)
    d = len(detectors)
    x = np.arange(d)
    peaks = np.linspace(d * 0.08, d * 0.92, n_fluors)
    M = np.array([spectral.normalise(np.exp(-0.5 * ((x - p) / (d / 25)) ** 2) + 0.01) for p in peaks])
    af = spectral.normalise(np.exp(-0.5 * ((x - d * 0.2) / (d / 6)) ** 2))
    names = SCATTER + detectors

    def events(abund, n):
        scatter = rng.uniform(2e5, 2e6, size=(n, len(SCATTER)))
        Y = abund @ M + rng.uniform(500, 1500, size=(n, 1)) * af
        Y = rng.normal(Y, np.sqrt(np.abs(Y) + 100))
        return np.hstack([scatter, Y])

    paths = []
    for i in range(n_fluors):
        a = np.zeros((2000, n_fluors))
        a[:, i] = rng.uniform(5e4, 2e5, 2000)
        p = tmp_path / f"Cells_F{i}.fcs"
        _write(p, names, events(a, 2000))
        paths.append(p)
    p = tmp_path / "Cells_unstained.fcs"
    _write(p, names, events(np.zeros((3000, n_fluors)), 3000))
    paths.append(p)

    true = rng.uniform(0, 5e4, size=(5000, n_fluors)) * (rng.random((5000, n_fluors)) < 0.5)
    sample = tmp_path / "Sample.fcs"
    # Columns shuffled: detectors are matched by name, not position.
    order = np.r_[np.arange(len(SCATTER)), len(SCATTER) + rng.permutation(d)]
    ev = events(true, 5000)
    _write(sample, [names[i] for i in order], ev[:, order])
    return paths, sample, M, true


@pytest.mark.parametrize("instrument", INSTRUMENTS)
def test_unmixing_any_instrument(tmp_path, instrument):
    detectors = INSTRUMENTS[instrument]
    controls, sample_path, M, true = _make_experiment(tmp_path, detectors)

    sample = fcs.load_sample(sample_path)
    assert fcs.is_spectral(sample), instrument
    assert [c.pnn for c in fcs.detector_channels(sample)] == sorted(
        detectors, key=[c.pnn for c in fcs.channels(sample)].index)
    assert {c.pnn for c in fcs.channels(sample) if c.kind == "scatter"} == {
        "FSC-A", "FSC-H", "SSC-A", "SSC-B-A"}

    um, results = unmixing.build_unmixer(unmixing.guess_controls(controls), method="wls")
    assert len(um.detectors) == len(detectors)
    for i, ref in enumerate(r for r in um.references if not r.is_autofluorescence):
        theirs = M[i][[detectors.index(d) for d in um.detectors]]
        cos = ref.signature @ theirs / np.linalg.norm(ref.signature) / np.linalg.norm(theirs)
        assert cos > 0.99, (instrument, ref.name, cos)

    unmixed = unmixing.unmix_sample(sample, um)
    got = np.column_stack([unmixed.get_channel_events(unmixed.get_channel_index(f"F{i}-A") + 1, source="raw")
                           for i in range(M.shape[0])])
    # Abundances come back to within a few percent of the brightest signal.
    assert np.median(np.abs(got - true)) < 0.03 * true.max(), instrument


def test_unmixed_or_conventional_files_are_not_raw_spectral(tmp_path):
    fluors = ["BUV395-A", "BUV496-A", "BUV563-A", "BUV615-A", "BUV661-A", "BUV737-A", "BUV805-A",
              "BV421-A", "BV480-A", "BV510-A", "BV570-A", "BV605-A", "BV650-A", "BV711-A",
              "BV750-A", "BV785-A", "FITC-A", "PerCP-Cy5.5-A", "PE-A", "PE-CF594-A", "PE-Cy5-A",
              "PE-Cy7-A", "APC-A", "AF700-A", "APC-Fire750-A", "AF-A"]
    p = tmp_path / "unmixed.fcs"
    _write(p, SCATTER + fluors, np.ones((10, len(SCATTER) + len(fluors))))
    assert not fcs.is_spectral(fcs.load_sample(p))


def test_mismatched_detectors_give_a_clear_error(tmp_path):
    controls, _, _, _ = _make_experiment(tmp_path, INSTRUMENTS["Aurora 3L (V/B/R)"])
    um, _ = unmixing.build_unmixer(unmixing.guess_controls(controls))
    other = tmp_path / "other"
    other.mkdir()
    _, sample_path, _, _ = _make_experiment(other, INSTRUMENTS["Aurora 5L (UV/V/B/YG/R)"])
    # A 5-laser sample has every 3-laser detector, so it unmixes...
    unmixing.unmix_sample(fcs.load_sample(sample_path), um)
    # ...but a 3-laser sample can't be unmixed with 5-laser controls.
    um5, _ = unmixing.build_unmixer(unmixing.guess_controls(
        sorted(p for p in other.glob("Cells_*.fcs"))))
    small = fcs.load_sample(tmp_path / "Sample.fcs")
    with pytest.raises(ValueError, match="same instrument configuration"):
        unmixing.unmix_sample(small, um5)


def test_chunked_unmixing_matches_whole_file(tmp_path, monkeypatch):
    controls, sample_path, _, _ = _make_experiment(tmp_path, INSTRUMENTS["Aurora 3L (V/B/R)"])
    um, _ = unmixing.build_unmixer(unmixing.guess_controls(controls), method="wls")
    sample = fcs.load_sample(sample_path)
    whole = unmixing.unmix_sample(sample, um).get_events(source="raw")
    monkeypatch.setattr(unmixing, "CHUNK", 777)
    chunked = unmixing.unmix_sample(sample, um).get_events(source="raw")
    np.testing.assert_allclose(chunked, whole)
