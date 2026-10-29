"""Make the example workspace for a release: tests/workspaces/v<version>/.

Run once per release, before tagging it:

    python scripts/make_example_workspace.py 0.2.0

It writes a small workspace that uses every feature (every gate shape, Boolean
gates, groups, unmixing with both kinds of autofluorescence, data cleaning, a
PCA/FlowSOM analysis, a layout page) on the demo PBMC file trimmed to 5000
events, plus the counts it should give. tests/test_versioning.py opens every
example ever made and checks those counts, so a later Sidescatter that can't
read an older file fails CI instead of failing a user.
"""

import argparse
import json
import sys
from pathlib import Path

import flowkit as fk
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sidescatter import __version__  # noqa: E402
from sidescatter.core import fcs, highdim, unmixing  # noqa: E402
from sidescatter.core.cleaning import Settings as CleanSettings  # noqa: E402
from sidescatter.core.experiment import Setup  # noqa: E402
from sidescatter.core.layout import Layout, LayoutItem  # noqa: E402
from sidescatter.core.workspace import ROOT as ALL, Workspace  # noqa: E402

DEMO = ROOT / "src" / "sidescatter" / "demo_data" / "spectral"
RAW = "PBMC_example.fcs"
EVENTS = 5000


def trim(src, dst, n=EVENTS):
    """The first n events of a demo file, as a new FCS file."""
    s = fcs.load_sample(src)
    keywords = {k: v for k, v in fcs.keywords(s).items() if k.lower() in ("cyt", "date", "btim", "etim")}
    fcs.write_fcs(dst, s.get_events(source="raw")[:n], list(s.pnn_labels), list(s.pns_labels),
                  keywords, [float(r or 262144) for r in s.channels["pnr"]])


def build(folder):
    ws = Workspace()
    [raw] = ws.add_fcs([folder / RAW])
    q = lambda sid, ch, p: float(np.quantile(ws.display_values(sid, ch), p))  # noqa: E731
    dim = ws.make_dimension

    # every gate shape, nested
    ws.add_gate(fk.gates.RectangleGate("Cells", [dim(raw, "FSC-A", q(raw, "FSC-A", 0.05), None),
                                                 dim(raw, "SSC-A", None, q(raw, "SSC-A", 0.98))]))
    cells = ALL + ("Cells",)
    fx, fy = q(raw, "FSC-A", 0.5), q(raw, "FSC-H", 0.5)
    ws.add_gate(fk.gates.PolygonGate("Singlets", [dim(raw, "FSC-A"), dim(raw, "FSC-H")],
                                     [[0, 0], [1, 0], [1, 1], [fx, fy + 0.15], [0, 0.1]]), cells)
    sx, sy = q(raw, "FSC-A", 0.5), q(raw, "SSC-A", 0.4)
    ws.add_gate(fk.gates.EllipsoidGate("Lymphocytes", [dim(raw, "FSC-A"), dim(raw, "SSC-A")],
                                       [sx, sy], [[0.01, 0.002], [0.002, 0.006]], 1.0), cells)
    v7 = q(raw, "V7-A", 0.6)
    ws.add_gate(fk.gates.RectangleGate("V7+", [dim(raw, "V7-A", v7, None)]), cells)  # a range gate
    ws.add_gate(fk.gates.RectangleGate("V7-", [dim(raw, "V7-A", None, v7)]), cells)
    ws.add_gate(fk.gates.RectangleGate("B3+", [dim(raw, "B3-A", q(raw, "B3-A", 0.5), None)]), cells)
    ws.add_boolean([("Lymphocytes", cells, False), ("V7+", cells, True)])
    ws.boolean_combinations([("V7+", cells), ("B3+", cells)])

    # unmixing: one unmixer with AF as extra fluors (unmixed from the sample in
    # the workspace), one with per-cell AF matching (from the raw file outside it)
    specs = unmixing.guess_controls(sorted((DEMO / "controls").glob("*.fcs")))
    fluors, _ = unmixing.build_unmixer(specs, method="wls", af_mode="fluors")
    per_cell, _ = unmixing.build_unmixer(specs, method="wls", af_mode="per_cell", af_library_size=10)
    ws.add_unmixer("Aurora", fluors)
    ws.add_unmixer("Aurora per-cell AF", per_cell)
    unmixed = ws.unmix(raw, "Aurora", group="Stained")
    matched = ws.unmix_file(folder / RAW, "Aurora per-cell AF", group="Stained")
    marker = next(c.pnn for c in ws.channels(unmixed) if c.pnn not in ws.samples[raw].pnn_labels
                  and c.kind == "fluorescence")
    ws.add_gate(fk.gates.RectangleGate(f"{marker}+", [dim(unmixed, marker, q(unmixed, marker, 0.7), None)]),
                cells)
    setup = Setup("Example", [s for s in specs], af_mode="per_cell", af_library_size=10)
    ws.experiments = [setup.to_dict()]
    for f in ws.experiments[0]["files"]:  # relative, so the example works from any checkout
        f["path"] = Path(f["path"]).name

    # data cleaning on one sample, then an analysis of the unmixed ones
    ws.clean([raw], CleanSettings())
    st = highdim.Settings(name="Example map", samples=[unmixed, matched], methods=["pca", "flowsom"],
                          channels=[c.pnn for c in ws.channels(unmixed) if c.kind == "fluorescence"][:6],
                          per_sample=2000, som_x=4, som_y=4, n_meta=4)
    ws.add_analysis(highdim.compute(highdim.gather(ws, st)))

    ws.layouts = [Layout("Figure 1", items=[
        LayoutItem("plot", sample_id=raw, population=list(cells), x_channel="FSC-A", y_channel="SSC-A"),
        LayoutItem("text", y=260, text="Example workspace made by Sidescatter " + __version__),
    ])]
    return ws


def counts(ws):
    return {sid: {"/".join([*p[1:], n]): ws.count(sid, n, p) for n, p in ws.population_paths()
                  if ws.applies_to(sid, n, p)}
            for sid in ws.samples}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("version", help="the release this example is for, e.g. 0.2.0")
    ap.add_argument("--force", action="store_true", help="replace an existing example")
    args = ap.parse_args()
    folder = ROOT / "tests" / "workspaces" / f"v{args.version}"
    if folder.exists() and not args.force:
        sys.exit(f"{folder} exists already; examples of past releases must never change (--force to replace)")
    folder.mkdir(parents=True, exist_ok=True)
    trim(DEMO / "PBMC_full_panel.fcs", folder / RAW)
    ws = build(folder)
    for sid, src in ws.sources.items():  # store paths relative to the example, as if moved
        for key in ("path", "unmixed_from_path"):
            if key in src:
                src[key] = Path(src[key]).name
    ws.save(folder / "example.ssw")
    (folder / "expected.json").write_text(json.dumps(
        {"version": args.version, "events": EVENTS, "counts": counts(ws)}, indent=1, sort_keys=True))
    print(f"Wrote {folder}: {len(ws.samples)} samples, {len(ws.population_paths())} populations")


if __name__ == "__main__":
    main()
