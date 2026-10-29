"""Laying out raw detectors and gating controls for spectrum plots."""

import numpy as np
import pytest

from sidescatter.core import experiment, fcs, spectrum, unmixing


@pytest.mark.parametrize("name, laser, number", [
    ("V7-A", "V", 7),
    ("UV16-A", "UV", 16),
    ("YG3-A", "YG", 3),
    ("R1-H", "R", 1),
    ("B510-A", "B", 510),        # BD: laser letter + emission wavelength
    ("UV379-A", "UV", 379),
    ("BL1-A", "B", 1),           # Attune-style aliases
    ("405-450", "V", 450),       # Sony: laser wavelength - emission wavelength
    ("637-780/60", "R", 780),
    ("BV421-A", None, None),     # unmixed fluor names aren't detectors
    ("CH12", None, None),
])
def test_parse_detector(name, laser, number):
    assert spectrum.parse_detector(name) == (laser, number)


def test_five_laser_order_ignores_file_order():
    names = ([f"R{i}-A" for i in range(1, 9)] + [f"UV{i}-A" for i in range(1, 17)]
             + [f"B{i}-A" for i in range(1, 15)] + [f"YG{i}-A" for i in range(1, 11)]
             + [f"V{i}-A" for i in range(16, 0, -1)])
    ordered = [names[i] for i in spectrum.detector_order(names)]
    assert ordered[:3] == ["UV1-A", "UV2-A", "UV3-A"]
    assert ordered[16:18] == ["V1-A", "V2-A"]
    assert [spectrum.parse_detector(n)[0] for n in ordered[32::14]] == ["B", "YG", "R"]
    assert ordered[-1] == "R8-A" and len(ordered) == 64
    groups = spectrum.laser_groups(names)
    assert [(g[0], g[2] - g[1] + 1) for g in groups] == [("UV", 16), ("V", 16), ("B", 14), ("YG", 10), ("R", 8)]


def test_unknown_names_keep_file_order_after_known_ones():
    names = ["Zeta", "B2-A", "Alpha", "V1-A"]
    assert [names[i] for i in spectrum.detector_order(names)] == ["V1-A", "B2-A", "Zeta", "Alpha"]
    assert spectrum.laser_color("Zeta") == spectrum.UNKNOWN_COLOR


def test_signature_colour_follows_peak_laser():
    names = ["V1-A", "V2-A", "V3-A", "R1-A", "R2-A"]
    red = spectrum.signature_color(np.array([0, 0, 0.1, 1.0, 0.3]), names)
    violet = spectrum.signature_color(np.array([1.0, 0.2, 0, 0, 0]), names)
    assert red[0] > red[2] and violet[2] > violet[0]
    # Neighbouring detectors of one laser get different shades.
    assert spectrum.detector_color(names, 0) != spectrum.detector_color(names, 2)


def test_density_scales_each_detector_on_its_own():
    rng = np.random.default_rng(0)
    events = np.column_stack([rng.normal(1e3, 50, 5000), rng.normal(1e5, 5e3, 50)[rng.integers(0, 50, 5000)]])
    xform = spectrum.intensity_transform()
    lo = float(spectrum.to_display(xform, spectrum.FLOOR))
    counts = spectrum.density(spectrum.to_display(xform, events), lo, 1.0, bins=100)
    assert counts.shape == (2, 100) and counts.sum() == 10000
    scaled = spectrum.column_scale(counts)
    assert np.allclose(scaled.max(axis=1), 1.0)
    # The dim detector peaks low, the bright one high.
    assert np.argmax(counts[0]) < np.argmax(counts[1])


def test_in_polygon_counts_edges_as_inside():
    square = [[0, 0], [10, 0], [10, 10], [0, 10]]
    x = np.array([5, 0, 10, 5, 11, -1, 5])
    y = np.array([5, 5, 10, 0, 5, 5, 10.0001])
    assert unmixing.in_polygon(x, y, square).tolist() == [True, True, True, True, False, False, False]


@pytest.fixture
def demo_specs(spectral_dir):
    files = sorted((spectral_dir / "controls").glob("*.fcs")) + [spectral_dir / "PBMC_full_panel.fcs"]
    return unmixing.guess_files(files)


def test_scatter_gate_and_drawn_ranges_drive_the_signature(demo_specs):
    samples = {}
    _, results = unmixing.build_unmixer(demo_specs, samples=samples)
    fitc = next(s for s in demo_specs if s.name == "FITC")
    before = next(r for r in results if r.spec is fitc)
    assert before.in_gate.all()

    sample = samples[fitc.path]
    x, y = (sample.get_events(source="raw")[:, sample.get_channel_index(c)]
            for c in fcs.scatter_channels(sample))
    mx, my = np.median(x), np.median(y)
    fitc.scatter_gate = [[0, 0], [mx, 0], [mx, my * 10], [0, my * 10]]   # the left half
    peak = before.peak_detector
    values = unmixing._detector_events(sample, [peak])[:, 0]
    fitc.gate_detector = peak
    fitc.positive_range = (float(np.percentile(values, 50)), float(values.max()))
    _, results = unmixing.build_unmixer(demo_specs, samples=samples)
    after = next(r for r in results if r.spec is fitc)
    assert after.in_gate.sum() < len(after.in_gate)
    assert not after.positive_mask[~after.in_gate].any()
    assert after.positive_mask.sum() < before.positive_mask.sum()
    assert after.peak_detector == peak


def test_internal_negative_range(demo_specs):
    pe = next(s for s in demo_specs if s.name == "PE")
    pe.negative = "internal"
    samples = {}
    _, results = unmixing.build_unmixer(demo_specs, samples=samples)
    r = next(r for r in results if r.spec is pe)
    assert r.negative_source == "internal" and r.negative_mask is not None
    values = unmixing._detector_events(samples[pe.path], [r.peak_detector])[:, 0]
    pe.negative_range = (float(values.min()), float(np.percentile(values, 5)))
    _, results = unmixing.build_unmixer(demo_specs, samples=samples)
    r2 = next(r for r in results if r.spec is pe)
    assert 10 <= r2.n_negative < r.n_negative


def test_gates_are_saved_with_the_setup(demo_specs):
    spec = demo_specs[0]
    spec.scatter_gate = [[1.0, 2.0], [3.0, 2.0], [3.0, 5.0]]
    spec.gate_detector = "R2-A"
    spec.positive_range = (1e4, 1e6)
    back = experiment.Setup.from_dict(experiment.Setup("E", demo_specs).to_dict())
    assert back.specs[0].scatter_gate == [[1.0, 2.0], [3.0, 2.0], [3.0, 5.0]]
    assert back.specs[0].gate_detector == "R2-A"
    assert back.specs[0].positive_range == (1e4, 1e6) and back.specs[0].negative_range is None
    assert back.specs[1].scatter_gate is None and back.specs[1].positive_range is None
