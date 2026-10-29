"""UMAP, tSNE, PCA and FlowSOM: the engine, the workspace channels and populations, and the UI."""

import numpy as np
import pytest
import flowkit as fk

from sidescatter.core import fcs, highdim
from sidescatter.core.flowsom import FlowSOM
from sidescatter.core.workspace import ROOT, Workspace

NAMES = ["FSC-A", "SSC-A"] + [f"M{i}-A" for i in range(6)] + ["AF-A", "Time"]
MARKERS = ["", "", "CD3", "CD4", "CD8", "CD19", "CD56", "CD14", "", ""]
CELLS = ("root", "Cells")


def _events(n, k, seed):
    """n events from k well-separated populations; returns (events, population of each)."""
    rng = np.random.default_rng(seed)
    centres = np.random.default_rng(0).uniform(300, 100_000, (k, 6))
    lab = rng.integers(0, k, n)
    fl = centres[lab] * rng.lognormal(0, 0.08, (n, 6))
    ev = np.column_stack([rng.uniform(5e4, 2e5, (n, 2)), fl, rng.uniform(0, 500, n), np.arange(n)])
    return ev, lab


@pytest.fixture
def files(tmp_path):
    paths, truth = [], {}
    for s, n in enumerate((4000, 3000)):
        ev, lab = _events(n, 5, seed=s + 1)
        p = tmp_path / f"sample{s}.fcs"
        fcs.write_fcs(p, ev, NAMES, MARKERS)
        paths.append(str(p))
        truth[p.name] = lab
    return paths, truth


@pytest.fixture
def ws(files):
    w = Workspace()
    ids = w.add_fcs(files[0])
    w.add_gate(fk.gates.RectangleGate("Cells", [w.make_dimension(ids[0], "FSC-A", 0.05, 0.95)]))
    return w


def _pure(truth, clusters):
    """How well each cluster holds a single true population (1 = perfectly)."""
    from sklearn.metrics import homogeneity_score

    return homogeneity_score(truth, clusters)


def test_flowsom_finds_separated_populations():
    ev, lab = _events(6000, 5, seed=3)
    data = np.log10(ev[:, 2:8])
    model = FlowSOM.fit(data, n_meta=8, seed=1)
    clusters, meta = model.predict(data)
    assert clusters.min() >= 1 and clusters.max() <= 100
    assert set(np.unique(meta)) <= set(range(1, 9))
    # Each metacluster holds one population (FlowSOM is meant to over-cluster a little).
    assert _pure(lab, meta) > 0.95
    assert _pure(lab, clusters) > 0.95
    # Same seed, same answer.
    again = FlowSOM.fit(data, n_meta=8, seed=1).predict(data)[1]
    assert np.array_equal(meta, again)


def test_default_channels_skip_scatter_time_and_autofluorescence(ws):
    chans = highdim.default_channels(ws.channels("sample0.fcs"))
    assert chans == [f"M{i}-A" for i in range(6)]
    # Alexa Fluor channels are fluors, not autofluorescence.
    alexa = [fcs.Channel(0, "AF647-A", "IgM", "fluorescence"), fcs.Channel(1, "AF2-A", "", "fluorescence")]
    assert highdim.default_channels(alexa) == ["AF647-A"]


def test_downsampling_is_equal_per_sample_and_capped():
    st = highdim.Settings(samples=["a", "b", "c", "d"], per_sample=20_000, max_total=1_000_000)
    assert st.per_sample_target() == 20_000
    st.max_total = 40_000
    assert st.per_sample_target() == 10_000


def _run(ws, methods, **kw):
    st = ws.new_analysis_settings("Cells", ROOT, list(ws.samples))
    st.methods, st.n_meta, st.per_sample = methods, 5, kw.pop("per_sample", 1500)
    for k, v in kw.items():
        setattr(st, k, v)
    return ws.add_analysis(highdim.compute(highdim.gather(ws, st)))


def test_analysis_adds_channels_populations_and_concatenated_sample(ws, files):
    _, truth = files
    record = _run(ws, ["pca", "flowsom"])
    st = record.settings
    s0 = ws.samples["sample0.fcs"]
    assert s0.pnn_labels[-4:] == ["PCA-1", "PCA-2", "FlowSOM", "FlowSOM meta"]
    assert [c.kind for c in ws.channels("sample0.fcs")][-4:] == ["derived"] * 4

    # Every event of the population has coordinates and a cluster, sampled or not.
    for sid in ("sample0.fcs", "sample1.fcs"):
        cells = ws.membership(sid, "Cells")
        pca = ws.scale_values(sid, "PCA-1")
        meta = ws.scale_values(sid, "FlowSOM meta")
        assert np.isfinite(pca[cells]).all() and (meta[cells] >= 1).all()
        assert np.isnan(pca[~cells]).all() and (meta[~cells] == 0).all()
    assert len(record.sampled["sample0.fcs"]) == 1500

    # Metaclusters are populations under Cells, with counts that add up.
    kids = [n for n, _ in ws.children(CELLS)]
    assert kids == [f"Metacluster {k}" for k in range(1, 6)]
    for sid in ("sample0.fcs", "sample1.fcs"):
        counts = [ws.count(sid, n, CELLS) for n in kids]
        assert sum(counts) == ws.count(sid, "Cells")
        meta = ws.scale_values(sid, "FlowSOM meta")[ws.membership(sid, "Cells")]
        assert _pure(truth[sid][ws.membership(sid, "Cells")], meta) > 0.6
    assert ws.statistic("sample0.fcs", "Freq. of Parent", "Metacluster 1", CELLS) > 0

    # The concatenated sample holds the sampled events with a Sample ID channel.
    cat = record.concatenated_id
    assert ws.samples[cat].event_count == 3000
    sample_id = ws.scale_values(cat, highdim.SAMPLE_ID)
    assert (np.bincount(sample_id.astype(int)) == [0, 1500, 1500]).all()
    assert ws.count(cat, "Cells") == 3000
    assert set(ws.groups[st.name]) == {"sample0.fcs", "sample1.fcs", cat}
    assert record.meta_medians.shape == (5, 6)


def test_renamed_metaclusters_survive_save_and_load(ws, files, tmp_path):
    record = _run(ws, ["pca", "flowsom"])
    ws.rename_gate("Metacluster 2", "NK cells", CELLS)
    before = {n: ws.count("sample1.fcs", n, CELLS) for n, _ in ws.children(CELLS)}
    path = tmp_path / "w.ssw"
    ws.save(path)

    back = Workspace.load(path)
    assert {n: back.count("sample1.fcs", n, CELLS) for n, _ in back.children(CELLS)} == before
    rec = back.analysis(record.settings.name)
    gate = back.get_gate("NK cells", CELLS)
    assert rec.metacluster_of(gate, back.transform(rec.settings.meta_channel)) == 2
    assert np.allclose(back.scale_values("sample0.fcs", "PCA-1"), ws.scale_values("sample0.fcs", "PCA-1"),
                       equal_nan=True)
    assert back.samples[record.concatenated_id].event_count == 3000


def test_second_analysis_gets_its_own_channels_and_delete_removes_one(ws):
    first = _run(ws, ["pca", "flowsom"])
    st = ws.new_analysis_settings("Cells", ROOT, ["sample0.fcs"])
    assert st.name == "Analysis 2" and st.suffix == " (2)"
    st.methods, st.n_meta = ["pca"], 5
    second = ws.add_analysis(highdim.compute(highdim.gather(ws, st)))
    assert "PCA-1 (2)" in ws.samples["sample0.fcs"].pnn_labels
    assert "PCA-1 (2)" not in ws.samples["sample1.fcs"].pnn_labels

    # A gate drawn inside a metacluster goes with it.
    ws.add_gate(fk.gates.RectangleGate("High PCA", [ws.make_dimension("sample0.fcs", "PCA-1", 0.5, 1.0)]),
                CELLS + ("Metacluster 1",))
    ws.remove_analysis(first.settings.name)
    assert ws.children(CELLS) == []
    assert "PCA-1" not in ws.samples["sample0.fcs"].pnn_labels
    assert "PCA-1 (2)" in ws.samples["sample0.fcs"].pnn_labels
    assert first.concatenated_id not in ws.samples and second.concatenated_id in ws.samples
    assert first.settings.name not in ws.groups


def test_rerun_replaces_the_analysis(ws):
    record = _run(ws, ["pca", "flowsom"])
    st = highdim.Settings.from_dict(record.settings.to_dict())
    st.n_meta = 4
    ws.add_analysis(highdim.compute(highdim.gather(ws, st)))
    assert len(ws.analyses) == 1
    assert [n for n, _ in ws.children(CELLS)] == [f"Metacluster {k}" for k in range(1, 5)]
    assert ws.samples["sample0.fcs"].pnn_labels.count("PCA-1") == 1


def test_umap_and_tsne_place_every_event(ws):
    record = _run(ws, ["umap", "tsne"], per_sample=400)
    for sid in ("sample0.fcs", "sample1.fcs"):
        cells = ws.membership(sid, "Cells")
        for ch in ("UMAP-1", "UMAP-2", "TSNE-1", "TSNE-2"):
            v = ws.scale_values(sid, ch)
            assert np.isfinite(v[cells]).all()
            assert v[cells].min() > 0  # shifted onto a linear scale starting at 0
            disp = ws.display_values(sid, ch, cells)
            assert 0 < disp.min() and disp.max() < 1
    assert record.meta_medians is None


def test_cancel_stops_the_run(ws):
    st = ws.new_analysis_settings("Cells", ROOT, list(ws.samples))
    st.methods = ["pca", "flowsom"]
    inputs = highdim.gather(ws, st)

    def progress(fraction, text):
        raise highdim.Cancelled()

    with pytest.raises(highdim.Cancelled):
        highdim.compute(inputs, progress)


# ----- the app ------------------------------------------------------------------


@pytest.fixture
def window(qtbot, files):
    pytest.importorskip("pytestqt")
    from sidescatter.app.main_window import MainWindow

    w = MainWindow()
    w.ask_before_discard = False
    qtbot.addWidget(w)
    w.add_samples(files[0])
    w.ws.add_gate(fk.gates.RectangleGate("Cells", [w.ws.make_dimension("sample0.fcs", "FSC-A", 0.05, 0.95)]))
    w._changed()
    return w


def test_dialog_runs_in_background_and_opens_the_map(window, qtbot):
    from sidescatter.app.analysis_dialog import AnalysisDialog, HeatmapDialog

    dlg = AnalysisDialog(window.ws, "Cells", ROOT, list(window.ws.samples), parent=window)
    assert dlg.name.text() == "Analysis 1"
    ticked = dlg._checked(dlg.channel_list, role=0x0100)
    assert ticked == [f"M{i}-A" for i in range(6)]
    dlg.method_boxes["umap"].setChecked(False)
    dlg.method_boxes["pca"].setChecked(True)
    dlg.n_meta.setValue(5)
    dlg.per_sample.setValue(1000)
    dlg.analysisDone.connect(window._after_analysis)
    with qtbot.waitSignal(dlg.analysisDone, timeout=120_000):
        dlg.run()
        assert dlg.progress.isVisible() or dlg.thread is not None
    assert window.ws.analysis("Analysis 1") is not None
    view = window.graph
    assert view.sample_id == "Concatenated (Analysis 1)"
    assert (view.x_combo.currentData(), view.y_combo.currentData()) == ("PCA-1", "PCA-2")
    view.set_colour("M0-A")
    view.refresh()

    window.ws.rename_gate("Metacluster 3", "B cells", CELLS)
    heat = HeatmapDialog(window.ws, window.ws.analysis("Analysis 1"), window)
    assert heat.rows[2] == "B cells"
    assert heat.table.columnCount() == 7
    assert abs(sum(heat.pct) - 100) < 1e-6
    df = heat.dataframe()
    assert list(df.columns[1:]) == ["CD3 :: M0-A", "CD4 :: M1-A", "CD8 :: M2-A", "CD19 :: M3-A",
                                    "CD56 :: M4-A", "CD14 :: M5-A"]


def test_dialog_cancel(window, qtbot):
    from sidescatter.app.analysis_dialog import AnalysisDialog

    dlg = AnalysisDialog(window.ws, "Cells", ROOT, list(window.ws.samples), parent=window)
    dlg.method_boxes["umap"].setChecked(False)
    dlg.run()
    dlg._cancel()
    qtbot.waitUntil(lambda: dlg.thread is None, timeout=60_000)
    assert window.ws.analyses == []
    assert dlg.run_button.isEnabled()


def test_ribbon_has_clustering_tools(window):
    for key in ("analysis", "heatmap", "remove_analysis"):
        assert key in window.acts
