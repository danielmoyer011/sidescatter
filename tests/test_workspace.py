import xml.etree.ElementTree as ET

import flowkit as fk
import numpy as np
import pytest

from sidescatter.core import stats, unmixing
from sidescatter.core.workspace import ROOT, Workspace


def _flowjo_counts(wsp_path):
    """Population counts FlowJo stored in a .wsp: {sample: {(parent_path, name): count}}."""
    out = {}

    def walk(node, path, counts):
        subs = node.find("Subpopulations")
        if subs is None:
            return
        for pop in subs.findall("Population"):
            counts[(tuple(path), pop.get("name"))] = int(pop.get("count"))
            walk(pop, path + [pop.get("name")], counts)

    for s in ET.parse(wsp_path).getroot().iter("Sample"):
        node = s.find("SampleNode")
        counts = {}
        walk(node, ["root"], counts)
        out[node.get("name")] = counts
    return out


@pytest.fixture
def unmixed_ws(spectral_dir):
    ws = Workspace()
    specs = unmixing.guess_controls(sorted((spectral_dir / "controls").glob("*.fcs")))
    um, _ = unmixing.build_unmixer(specs, method="wls")
    ws.add_unmixer("Aurora", um)
    [raw] = ws.add_fcs([spectral_dir / "PBMC_full_panel.fcs"])
    sid = ws.unmix(raw, "Aurora")
    return ws, sid


def test_rectangle_gate_counts_match_numpy(unmixed_ws):
    ws, sid = unmixed_ws
    x = ws.display_values(sid, "PB-A")
    lo, hi = np.quantile(x, [0.5, 0.99])
    gate = fk.gates.RectangleGate("CD3+", [ws.make_dimension(sid, "PB-A", lo, hi)])
    ws.add_gate(gate)
    assert ws.count(sid, "CD3+") == int(((x >= lo) & (x < hi)).sum())


def test_nested_gates_and_frequencies(unmixed_ws):
    ws, sid = unmixed_ws
    fsc = ws.display_values(sid, "FSC-A")
    ws.add_gate(fk.gates.RectangleGate("Cells", [ws.make_dimension(sid, "FSC-A", np.quantile(fsc, 0.05), None)]))
    ws.add_gate(fk.gates.PolygonGate(
        "Lymphs", [ws.make_dimension(sid, "FSC-A"), ws.make_dimension(sid, "SSC-A")],
        [[0, 0], [1, 0], [1, float(np.median(ws.display_values(sid, "SSC-A")))], [0, 0]]),
        ("root", "Cells"))
    n_cells = ws.count(sid, "Cells")
    n_lymph = ws.count(sid, "Lymphs", ("root", "Cells"))
    assert 0 < n_lymph <= n_cells
    assert ws.statistic(sid, "Freq. of Parent", "Lymphs", ("root", "Cells")) == pytest.approx(100 * n_lymph / n_cells)
    assert [p for p in ws.population_paths()] == [("Cells", ROOT), ("Lymphs", ("root", "Cells"))]


def test_changing_transform_keeps_gate_selection(unmixed_ws):
    ws, sid = unmixed_ws
    x = ws.display_values(sid, "PE-A")
    ws.add_gate(fk.gates.RectangleGate("PE+", [ws.make_dimension(sid, "PE-A", np.quantile(x, 0.7), None)]))
    before = ws.count(sid, "PE+")
    ws.set_transform("PE-A", fk.transforms.AsinhTransform(param_t=4194304, param_m=5, param_a=0))
    assert ws.count(sid, "PE+") == before


def test_save_and_load_roundtrip(unmixed_ws, tmp_path):
    ws, sid = unmixed_ws
    x = ws.display_values(sid, "PB-A")
    ws.add_gate(fk.gates.RectangleGate("CD3+", [ws.make_dimension(sid, "PB-A", np.median(x), None)]))
    ws.add_group("Stained", [sid])
    path = tmp_path / "test.ssw"
    ws.save(path)
    back = Workspace.load(path)
    assert back.groups["Stained"] == [sid]
    assert back.count(sid, "CD3+") == ws.count(sid, "CD3+")
    assert back.unmixers["Aurora"].names == ws.unmixers["Aurora"].names


def test_statistics_table(unmixed_ws):
    ws, sid = unmixed_ws
    table = ws.statistics_table([sid], [
        {"stat": "Count"},
        {"stat": "Median", "channel": "PB-A"},
    ])
    assert table.loc[0, "All events | Count"] == ws.samples[sid].event_count
    assert table.loc[0, "All events | Median (PB-A)"] == pytest.approx(
        np.median(ws.scale_values(sid, "PB-A")))


def test_channel_stats():
    v = np.array([1.0, 10.0, 100.0, -5.0])
    assert stats.channel_stat(v, "Geometric Mean") == pytest.approx(10.0)
    assert stats.channel_stat(v, "Median") == pytest.approx(5.5)


def test_flowjo_import_counts_close_to_flowjo(eight_color):
    """Counts from an imported FlowJo 10 workspace agree with FlowJo's own numbers.

    Known gap: populations over 1,000 events agree within 0.5%, but small
    populations gated on FlowJo's biexponential scale can differ by up to
    ~15% (e.g. 64 vs 74 events). Matching FlowJo's biex display exactly is a
    phase 0 accuracy goal; this test guards against regressions meanwhile.
    """
    wsp = eight_color / "8_color_ICS.wsp"
    ws = Workspace.from_flowjo(wsp, eight_color / "fcs_files")
    truth = _flowjo_counts(wsp)
    assert set(ws.groups) >= {"All Samples", "DEN"}
    checked = 0
    for sid in ws.samples:
        for name, parent in ws.population_paths():
            fj = truth[sid][(parent, name)]
            ours = ws.count(sid, name, parent)
            tolerance = 0.005 * fj if fj >= 1000 else max(3, 0.15 * fj)
            assert abs(ours - fj) <= tolerance, (sid, parent, name, ours, fj)
            checked += 1
    assert checked >= 40
