"""Automated data cleaning: the PeacoQC port, flow rate, margins, and the QC passed population."""

from pathlib import Path

import flowkit as fk
import numpy as np
import pytest

from sidescatter.core import cleaning, fcs, unmixing
from sidescatter.core.layout import Layout, LayoutItem
from sidescatter.core.workspace import QC_GATE, ROOT, Workspace

PEACOQC_FILE = Path(__file__).parent / "data" / "peacoqc" / "111_Comp_Trans.fcs"
# The vignette's channels (R's 1, 3, 5:14, 18, 21)
PEACOQC_CHANNELS = [0, 2, *range(4, 14), 17, 20]


# ----- synthetic runs --------------------------------------------------------

def synthetic_run(n=60_000, seed=0, clog=False, bubble=False, drift=False, pause=False):
    """A stable 4-colour run with a Time channel, plus injected anomalies.

    Returns (events, names, regions) with regions {kind: (start, end)} in event index.
    """
    rng = np.random.default_rng(seed)
    pop = rng.random(n) < 0.6
    k = 4
    centres = np.where(pop[:, None], [0.3, 0.6, 0.45, 0.7], [0.6, 0.25, 0.5, 0.3])
    x = centres + rng.normal(0, 0.05, (n, k))
    fsc = 0.5 + rng.normal(0, 0.05, n)
    rate = np.full(n, 1000.0)  # events per Time unit
    regions = {}
    if clog:
        a, b = int(0.30 * n), int(0.34 * n)
        x[a:b] -= 0.12
        rate[a:b] = 250
        regions["clog"] = (a, b)
    if bubble:
        a, b = int(0.62 * n), int(0.635 * n)
        x[a:b, :2] += 0.15
        regions["bubble"] = (a, b)
    if drift:
        a = int(0.8 * n)
        x[a:, 2] += np.linspace(0, 0.3, n - a)
        regions["drift"] = (a, n)
    if pause:
        a, b = int(0.5 * n), int(0.52 * n)
        rate[a:b] = 120  # the flow nearly stops, the signal looks normal
        regions["pause"] = (a, b)
    time = np.cumsum(rng.exponential(1 / rate))
    events = np.column_stack([time * 100, fsc, x])
    return events, ["Time", "FSC-A", "C1-A", "C2-A", "C3-A", "C4-A"], regions


def removed_in(mask, a, b):
    return mask[a:b].mean()


def removed_outside(mask, regions):
    keep = np.ones(len(mask), bool)
    for a, b in regions.values():
        keep[max(a - 1500, 0):b + 1500] = False  # windows straddling an edge go too
    return mask[keep].mean()


def clean_synthetic(events, settings=None):
    return cleaning.clean(events[:, 1:], ["FSC-A", "C1-A", "C2-A", "C3-A", "C4-A"], events[:, 0],
                          settings or cleaning.Settings(margins=False), time_channel="Time")


# ----- matching PeacoQC --------------------------------------------------------

def test_density_uses_r_bandwidth_and_grid():
    # bw.nrd0(c(1, 2, 2.5, 4, 7)) = 0.9 * min(sd, IQR / 1.34) * 5^-0.2 = 0.97358
    xs, ys = cleaning.density(np.array([1, 2, 2.5, 4, 7.0]))
    assert len(xs) == 512
    assert xs[0] == pytest.approx(1 - 3 * 0.97358, abs=1e-4)
    assert xs[-1] == pytest.approx(7 + 3 * 0.97358, abs=1e-4)
    assert ys.sum() * (xs[1] - xs[0]) == pytest.approx(1.0, abs=0.01)


def test_removes_exactly_what_peacoqc_removes():
    sample = fk.Sample(str(PEACOQC_FILE))
    ev = sample.get_events(source="raw")
    names = [sample.pnn_labels[i] for i in PEACOQC_CHANNELS]
    res = cleaning.clean(ev[:, PEACOQC_CHANNELS], names, None,
                         cleaning.Settings(margins=False, flow_rate=False))
    removed = res.removed(None)
    expected = np.zeros(len(ev), bool)
    expected[:1000] = expected[9500:] = True  # from PeacoQC's R code, see data/peacoqc/README.md
    assert res.bins["n_bins"] == 39 and res.bins["events_per_bin"] == 500
    assert len(res.bins["peaks"]) == 19
    assert np.array_equal(removed, expected)
    assert res.removed_by["MAD"] == pytest.approx(100 * 1117 / 9617, abs=1e-3)


def test_peacoqc_helpers():
    assert cleaning.events_per_bin(9617) == 500
    assert cleaning.events_per_bin(1_000_000) == 4500
    br = cleaning.make_breaks(1000, 500)
    assert br.tolist() == [[0, 500], [250, 750], [500, 1000], [750, 1000]]
    good = np.array([True] * 6 + [False] + [True] * 3 + [False] * 2 + [True] * 8)
    out = cleaning.remove_short_regions(good, 5)
    assert out[:6].all() and not out[6:12].any() and out[12:].all()


def test_smooth_spline_is_a_smoother():
    y = np.r_[np.zeros(100), np.ones(100)] + np.random.default_rng(0).normal(0, 0.05, 200)
    s = cleaning.smooth_spline(y)
    assert np.abs(s - y).mean() < 0.1
    assert np.std(np.diff(s)) < np.std(np.diff(y))


# ----- anomalies -----------------------------------------------------------------

def test_clean_run_keeps_almost_everything():
    events, _, _ = synthetic_run()
    res = clean_synthetic(events)
    assert res.percent_removed(events[:, 0]) < 3
    assert not res.warnings


@pytest.mark.parametrize("kind", ["clog", "bubble", "drift"])
def test_finds_injected_anomalies(kind):
    events, _, regions = synthetic_run(**{kind: True})
    res = clean_synthetic(events)
    removed = res.removed(events[:, 0])
    a, b = regions[kind]
    assert removed_in(removed, a, b) > 0.85
    assert removed_outside(removed, regions) < 0.03


def test_all_anomalies_together():
    events, _, regions = synthetic_run(clog=True, bubble=True, drift=True)
    res = clean_synthetic(events)
    removed = res.removed(events[:, 0])
    for a, b in regions.values():
        assert removed_in(removed, a, b) > 0.85
    assert removed_outside(removed, regions) < 0.05
    assert res.removed_by["Isolation tree"] > 0 or res.removed_by["MAD"] > 0


def test_flow_rate_catches_a_pause_the_signal_misses():
    events, _, regions = synthetic_run(pause=True)
    a, b = regions["pause"]
    signal_only = clean_synthetic(events, cleaning.Settings(margins=False, flow_rate=False))
    assert removed_in(signal_only.removed(events[:, 0]), a, b) < 0.2
    res = clean_synthetic(events)
    assert removed_in(res.removed(events[:, 0]), a, b) > 0.8
    assert res.removed_by["Flow rate"] > 1


def test_strict_removes_at_least_as_much_as_lenient():
    events, _, _ = synthetic_run(drift=True, seed=3)
    pct = {name: clean_synthetic(events, cleaning.Settings.from_preset(name, margins=False))
           .percent_removed(events[:, 0]) for name in ("Lenient", "Standard", "Strict")}
    assert pct["Lenient"] <= pct["Standard"] <= pct["Strict"]


def test_events_out_of_time_order_are_sorted_first():
    events, _, regions = synthetic_run(clog=True)
    perm = np.random.default_rng(1).permutation(len(events))
    shuffled = events[perm]
    res = clean_synthetic(shuffled)
    assert any("time order" in w for w in res.warnings)
    removed = np.empty(len(events), bool)
    removed[perm] = res.removed(shuffled[:, 0])
    assert removed_in(removed, *regions["clog"]) > 0.85


def test_no_time_channel_uses_event_order():
    events, _, regions = synthetic_run(clog=True)
    res = cleaning.clean(events[:, 1:], ["FSC-A", "C1-A", "C2-A", "C3-A", "C4-A"], None,
                         cleaning.Settings(margins=False))
    assert any("No Time channel" in w for w in res.warnings)
    assert "Flow rate" not in res.removed_by
    assert removed_in(res.removed(None), *regions["clog"]) > 0.85


def test_margins_only_count_pileups():
    rng = np.random.default_rng(0)
    raw = rng.normal(1000, 100, (5000, 2))
    clean_mask, counts = cleaning.find_margins(raw, [262144, 262144], ["A", "B"])
    assert not clean_mask.any() and counts == {}
    raw[:40, 0] = 262144  # saturated detector
    raw[100:110, 1] = raw[:, 1].min() - 5  # piled up at the bottom
    mask, counts = cleaning.find_margins(raw, [262144, 262144], ["A", "B"])
    assert mask[:40].all() and mask[100:110].all()
    assert counts == {"A": 40, "B": 10} and mask.sum() == 50


def test_result_round_trips_and_edits():
    events, _, _ = synthetic_run(clog=True)
    res = clean_synthetic(events)
    back = cleaning.Result.from_dict(res.to_dict())
    t = events[:, 0]
    assert np.array_equal(back.removed(t), res.removed(t))
    spans = cleaning.stretch_times(res, t)
    assert len(spans) == 1
    a, b = spans[0]
    cleaning.set_stretch_times(res, t, [(a, (a + b) / 2), (t.max() * 0.9, t.max() * 0.95)])
    assert res.edited and len(res.stretches) == 2
    cleaning.set_stretch_times(res, t, spans)
    assert not res.edited


def test_tiny_files_only_lose_margins():
    events, _, _ = synthetic_run(n=800)
    res = clean_synthetic(events)
    assert res.percent_removed(events[:, 0]) == 0
    assert any("too few" in w for w in res.warnings)


def test_default_channels_leave_out_time_and_af():
    chans = [fcs.Channel(0, "Time", "", "time"), fcs.Channel(1, "FSC-A", "", "scatter"),
             fcs.Channel(2, "FSC-H", "", "scatter"), fcs.Channel(3, "SSC-A", "", "scatter"),
             fcs.Channel(4, "BV421-A", "CD3", "fluorescence"), fcs.Channel(5, "AF647-A", "IgM", "fluorescence"),
             fcs.Channel(6, "AF-A", "", "fluorescence"), fcs.Channel(7, "AF2-A", "", "fluorescence")]
    assert cleaning.default_channels(chans) == ["FSC-A", "SSC-A", "BV421-A", "AF647-A"]


# ----- the workspace -------------------------------------------------------------

def _workspace(tmp_path, **kw):
    events, names, regions = synthetic_run(**kw)
    path = tmp_path / "run.fcs"
    fcs.write_fcs(path, events, names, ranges=[events[:, 0].max() + 1] + [1.0] * 5,
                  keywords={"TIMESTEP": "0.01"})
    ws = Workspace()
    sid = ws.add_fcs([path])[0]
    return ws, sid, events, regions


def _linear(ws, sid):
    for ch in ws.channels(sid):
        if ch.kind != "time":
            ws.set_transform(ch.pnn, fk.transforms.LinearTransform(param_t=1.0, param_a=0.0))


def test_qc_passed_wraps_existing_gates(tmp_path):
    ws, sid, events, regions = _workspace(tmp_path, clog=True)
    _linear(ws, sid)
    ws.add_gate(fk.gates.RectangleGate("Pop", [ws.make_dimension(sid, "C1-A", 0.15, 0.45)]))
    ws.add_gate(fk.gates.RectangleGate("Sub", [ws.make_dimension(sid, "C2-A", 0.4, None)]), ROOT + ("Pop",))
    lay = Layout()
    lay.items.append(LayoutItem("plot", population=["root", "Pop", "Sub"]))
    lay.items.append(LayoutItem("plot", population=["root"]))
    ws.layouts.append(lay)
    before = ws.count(sid, "Pop")

    res = ws.clean([sid])[sid]
    top = ROOT + (QC_GATE,)
    assert ws.population_paths() == [(QC_GATE, ROOT), ("Pop", top), ("Sub", top + ("Pop",))]
    keep = ws.qc_keep(sid)
    assert ws.count(sid, QC_GATE) == keep.sum() < len(events)
    assert ws.count(sid, "Pop", top) < before
    assert np.array_equal(ws.membership(sid, "Pop", top), ws.membership(sid, "Pop", top) & keep)
    assert removed_in(~keep, *regions["clog"]) > 0.85
    assert lay.items[0].population == ["root", QC_GATE, "Pop", "Sub"]
    assert lay.items[1].population == ["root"]
    assert res.removed_by["Flow rate"] > 0

    ws.remove_qc()
    assert ws.population_paths() == [("Pop", ROOT), ("Sub", ROOT + ("Pop",))]
    assert ws.count(sid, "Pop") == before
    assert lay.items[0].population == ["root", "Pop", "Sub"]


def test_deleting_qc_gate_keeps_the_gates_inside(tmp_path):
    ws, sid, _, _ = _workspace(tmp_path, clog=True)
    ws.add_gate(fk.gates.RectangleGate("Pop", [ws.make_dimension(sid, "C1-A", 0.15, 0.45)]))
    ws.clean([sid])
    ws.remove_gate(QC_GATE, ROOT)
    assert ws.population_paths() == [("Pop", ROOT)] and not ws.qc


def test_uncleaned_samples_keep_every_event(tmp_path):
    ws, sid, events, _ = _workspace(tmp_path, clog=True)
    other_path = tmp_path / "other.fcs"
    ev2, names, _ = synthetic_run(seed=5)
    fcs.write_fcs(other_path, ev2, names)
    other = ws.add_fcs([other_path])[0]
    ws.clean([sid])
    assert ws.count(other, QC_GATE) == len(ev2)
    assert ws.count(sid, QC_GATE) < len(events)


def test_qc_saved_with_workspace_and_edits_kept(tmp_path):
    ws, sid, events, _ = _workspace(tmp_path, clog=True)
    settings = cleaning.Settings.from_preset("Standard")
    ws.clean([sid], settings)
    t0, t1 = float(events[:, 0].min()), float(events[:, 0].max())
    ws.set_qc_stretches(sid, [(t0 + 0.1 * (t1 - t0), t0 + 0.2 * (t1 - t0))])
    edited = ws.count(sid, QC_GATE)
    assert ws.qc[sid].edited

    path = tmp_path / "ws.ssw"
    ws.save(path)
    back = Workspace.load(path)
    assert back.qc[sid].edited
    assert back.count(sid, QC_GATE) == edited

    back.clean([sid], cleaning.Settings.from_preset("Standard"))
    assert back.qc[sid].edited and back.count(sid, QC_GATE) == edited  # same settings: edits kept
    back.clean([sid], cleaning.Settings.from_preset("Strict"))
    assert not back.qc[sid].edited  # different settings: cleaned afresh

    back.reset_qc_stretches(sid)
    assert not back.qc[sid].edited


def test_export_cleaned_fcs(tmp_path):
    ws, sid, _, _ = _workspace(tmp_path, clog=True)
    ws.clean([sid])
    out = tmp_path / "clean.fcs"
    n = ws.export_cleaned(sid, out)
    s = fk.Sample(str(out))
    assert s.event_count == n == ws.count(sid, QC_GATE)
    assert s.pnn_labels == ws.samples[sid].pnn_labels


def test_unmixed_samples_checked_for_margins_on_raw_detectors(spectral_dir):
    ws = Workspace()
    files = sorted((spectral_dir / "controls").glob("*.fcs"))
    specs = unmixing.guess_controls(files)
    um, _ = unmixing.build_unmixer(specs, method="ols")
    ws.add_unmixer("demo", um)
    raw = ws.add_fcs([spectral_dir / "PBMC_full_panel.fcs"])[0]
    raw_sample = ws.samples[raw]
    top = float(raw_sample.channels.loc[3, "pnr"])
    raw_sample.get_events(source="raw")[:25, 3] = top  # saturate a detector on 25 events
    sid = ws.unmix(raw, "demo")
    res = ws.clean([sid])[sid]
    assert res.margin[:25].all()
    assert raw_sample.pnn_labels[3] in res.margin_counts
    assert any(c.endswith("-A") and not c.startswith(("AF-", "Time")) for c in res.channels)
    assert "AF-A" not in res.channels


def test_cleaning_controls_before_signatures(spectral_dir):
    files = sorted((spectral_dir / "controls").glob("*.fcs"))
    specs = unmixing.guess_controls(files)
    samples = {}
    plain, _ = unmixing.build_unmixer(specs, method="ols", samples=samples)
    pe = next(s for s in specs if s.name == "PE")
    sample = samples[pe.path]
    keep, res = unmixing.control_keep(sample)
    assert keep.shape == (sample.event_count,) and res.n_events == sample.event_count
    # drop a third of the PE control: the signature must come from the rest only
    keep = keep.copy()
    keep[: len(keep) // 3] = False
    um, results = unmixing.build_unmixer(specs, method="ols", samples=samples, keep={pe.path: keep})
    r = next(r for r in results if r.spec.path == pe.path)
    assert not r.positive_mask[~keep].any()
    assert um.names == plain.names
    i = um.names.index("PE")
    assert np.corrcoef(um.matrix[i], plain.matrix[i])[0, 1] > 0.99


def test_plot_data_for_reports(tmp_path):
    ws, sid, _, _ = _workspace(tmp_path, clog=True)
    ws.clean([sid])
    data = ws.qc_plot_data(sid)
    assert data["x_label"] == "Time (s)"
    assert len(data["stretches"]) >= 1
    assert [c["name"] for c in data["channels"]] == ws.qc[sid].channels
    assert data["channels"][1]["hist"].shape == (240, 90)
    assert data["channels"][1]["peaks"]
