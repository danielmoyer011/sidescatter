"""GatingML 2.0 reference checks: every gate of the official compliance examples
(tests/data/gatingml, from FlowKit, BSD licence) must give exactly the
reference events in Sidescatter, before and after saving and reopening.

The examples cover rectangle, polygon (including a non-simple one), ellipse,
3D ellipsoid, quadrant and Boolean gates, ratio dimensions, every GatingML
transform, gates on spillover-compensated fluorochromes, and nested gates.
"""

import copy
import re
from pathlib import Path

import flowkit as fk
import numpy as np
import pandas as pd
import pytest

from sidescatter.core.workspace import Strategy, Workspace

DATA = Path(__file__).parent / "data" / "gatingml"
TRUTH = {re.search(r"Results_(.+)\.txt$", p.name).group(1):
         pd.read_csv(p, header=None, dtype="bool").squeeze().values
         for p in sorted((DATA / "truth").glob("Results_*.txt"))}
SID = "data1.fcs"
ALL = "gml_all_gates.xml"
# reference results for gates that aren't in gml_all_gates.xml: result -> (file, gate)
OWN_FILE = {"Ellipsoid3D": ("gml_ellipsoid3d_gate.xml", "Ellipsoid3D"),
            "ParQuadRect": ("gml_parent_quadrant_rect_gate.xml", "ParRectangle1")}
CASES = sorted([(ALL, n, n) for n in TRUTH if n not in OWN_FILE] +
               [(f, gate, n) for n, (f, gate) in OWN_FILE.items()])


def _workspace(gml=ALL):
    """A GatingML file's gates in a Sidescatter workspace, with the reference file."""
    gs = fk.parse_gating_xml(str(DATA / "gml" / gml))
    ws = Workspace()
    ws.strategy = Strategy()
    ws.strategy.__dict__.update(copy.deepcopy(gs.__dict__))
    ws.add_fcs([DATA / SID])
    return ws


_cache = {}


def _opened(gml):
    if gml not in _cache:
        _cache[gml] = _workspace(gml)
    return _cache[gml]


def _reopened(gml, tmp_path_factory):
    key = ("reopened", gml)
    if key not in _cache:
        path = tmp_path_factory.mktemp("gml") / "reference.ssw"
        _workspace(gml).save(path)
        _cache[key] = Workspace.load(path, search_dirs=[DATA])
    return _cache[key]


@pytest.fixture(scope="module")
def opened():
    return _opened(ALL)


@pytest.fixture(scope="module")
def reopened(tmp_path_factory):
    return _reopened(ALL, tmp_path_factory)


def _population(ws, name):
    return next(p for n, p in ws.population_paths() if n == name)


def test_every_reference_result_is_checked(opened):
    names = {n for n, _ in opened.population_paths()}
    assert set(TRUTH) - set(OWN_FILE) <= names
    assert names - set(TRUTH) == {"Quadrant1", "Quadrant2"}  # results are per quadrant


@pytest.mark.parametrize("gml, gate, result", CASES, ids=[c[2] for c in CASES])
def test_gate_matches_reference(gml, gate, result):
    ws = _opened(gml)
    got = ws.membership(SID, gate, _population(ws, gate))
    assert np.array_equal(got, TRUTH[result]), f"{gate}: {int(got.sum())} events, reference {int(TRUTH[result].sum())}"


@pytest.mark.parametrize("gml, gate, result", CASES, ids=[c[2] for c in CASES])
def test_gate_matches_reference_after_saving_and_reopening(gml, gate, result, tmp_path_factory):
    ws = _reopened(gml, tmp_path_factory)
    got = ws.membership(SID, gate, _population(ws, gate))
    assert np.array_equal(got, TRUTH[result]), f"{gate}: {int(got.sum())} events, reference {int(TRUTH[result].sum())}"


def test_reopened_workspace_has_the_same_gates(opened, reopened):
    assert reopened.population_paths() == opened.population_paths()
    assert set(reopened.strategy.comp_matrices) == {"MySpill"}


def test_every_population_opens_in_a_graph_window(qtbot):
    pytest.importorskip("pytestqt")
    from sidescatter.app.main_window import MainWindow

    w = MainWindow()
    w.ask_before_discard = False
    qtbot.addWidget(w)
    for gml in (ALL, *(f for f, _ in OWN_FILE.values())):
        _open_every_graph(w, gml)


def _open_every_graph(w, gml):
    from sidescatter.app.jobs import wait_idle

    w._set_workspace(_workspace(gml))
    wait_idle()
    w.panel.rebuild()  # every population's count shows in the workspace table
    counts = [it.text(2) for it in w.panel._items()]
    assert len(counts) == len(w.ws.population_paths()) + 1 and "…" not in counts
    for name, parent in w.ws.population_paths():
        view = w.open_graph(SID, name, parent)
        wait_idle()
        view.refresh()
        assert view.count_label.text().endswith("events")
        view.window().close()
