"""Speed test on a big full-spectrum file: 5 million events, 64 detectors.

    python scripts/benchmark.py                 # 5M events, compare with the baseline
    python scripts/benchmark.py --events 500000 # quicker, for trying things out
    python scripts/benchmark.py --record        # save these times as the new baseline

It makes the file (fixed seed, Aurora-style detector names: never checked in,
it's about 1.4 GB), then times the steps a user waits for: opening it,
unmixing it, gating 10 nested gates, cleaning it, and exporting it. For each
step it reports the time and the process's peak memory, as a Markdown table
(written to the GitHub job summary in CI).

It fails when a step takes more than twice as long as in
benchmarks/baseline.json, or memory goes over the limit (6 GB by default).
Re-record the baseline (from a CI run) when a step got slower on purpose.
"""

import argparse
import gc
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

import flowkit as fk
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sidescatter.core import fcs  # noqa: E402
from sidescatter.core.spectral import Reference, Unmixer  # noqa: E402
from sidescatter.core.workspace import ROOT as ALL, Workspace  # noqa: E402

BASELINE = ROOT / "benchmarks" / "baseline.json"
EVENTS = 5_000_000
SLOWER = 2.0      # fail when a step takes this many times its baseline
MEMORY_GB = 6.0   # fail when the process's memory goes over this
LASERS = (("UV", 16), ("V", 16), ("B", 14), ("YG", 10), ("R", 8))  # 64 detectors, like a 5-laser Aurora
FLUORS = 12


def detectors():
    return [f"{laser}{i}-A" for laser, n in LASERS for i in range(1, n + 1)]


def signatures(dets, k=FLUORS, seed=1):
    """k smooth, peaked emission spectra over the detectors (peak normalised to 1)."""
    rng = np.random.default_rng(seed)
    x = np.arange(len(dets))
    sigs = []
    for i in range(k):
        peak = (i + 0.5) * len(dets) / k
        width = rng.uniform(2.5, 6.0)
        s = np.exp(-0.5 * ((x - peak) / width) ** 2) + 0.05 * np.exp(-0.5 * ((x - peak - 12) / 8) ** 2)
        sigs.append(s / s.max())
    af = np.exp(-0.5 * ((x - 18) / 14) ** 2)
    return np.array(sigs), af / af.max()


def make_file(path, n, seed=7, block=500_000):
    """The test file: scatter, time and 64 detectors. Written in blocks, so making
    it needs far less memory than the file is big."""
    dets = detectors()
    sigs, af = signatures(dets)
    pnn = ["Time", "FSC-A", "FSC-H", "SSC-A", "SSC-H", *dets]
    rng = np.random.default_rng(seed)
    blocks = []
    for start in range(0, n, block):
        m = min(block, n - start)
        ab = rng.lognormal(mean=6.5, sigma=1.4, size=(m, len(sigs))) * (rng.random((m, len(sigs))) < 0.35)
        spectral = ab @ sigs + rng.lognormal(6.0, 0.4, size=(m, 1)) * af
        spectral += rng.normal(0, 30, size=spectral.shape)
        fsc = rng.normal(90_000, 20_000, size=m).clip(1000)
        ssc = rng.lognormal(10.5, 0.6, size=m)
        cols = [np.arange(start, start + m) / 1000.0, fsc, fsc * rng.normal(0.85, 0.03, m),
                ssc, ssc * rng.normal(0.8, 0.03, m)]
        blocks.append(np.column_stack(cols + [spectral]).astype(np.float32))
    events = np.vstack(blocks)
    del blocks
    ranges = [float(n) / 1000.0 + 1] + [4194304.0] * (len(pnn) - 1)
    fcs.write_fcs(path, events, pnn, keywords={"$CYT": "Aurora (benchmark)", "$TIMESTEP": "1"}, ranges=ranges)
    del events
    gc.collect()
    refs = [Reference(f"F{i + 1}", s) for i, s in enumerate(sigs)]
    refs.append(Reference("AF", af, is_autofluorescence=True))
    return Unmixer(dets, refs, "ols")


class Memory:
    """Peak resident memory while a step runs, sampled every 20 ms (Linux/macOS/Windows)."""

    def __init__(self):
        self.peak, self._stop = 0, threading.Event()

    def __enter__(self):
        self.peak = _rss()
        self._t = threading.Thread(target=self._watch, daemon=True)
        self._t.start()
        return self

    def _watch(self):
        while not self._stop.wait(0.02):
            self.peak = max(self.peak, _rss())

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()
        self.peak = max(self.peak, _rss())


def _rss():
    try:
        import psutil

        return psutil.Process().memory_info().rss
    except ImportError:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def run(n, folder):
    results = []

    def step(name, fn):
        gc.collect()
        with Memory() as mem:
            t0 = time.perf_counter()
            out = fn()
            secs = time.perf_counter() - t0
        results.append({"step": name, "seconds": round(secs, 2), "peak_gb": round(mem.peak / 1e9, 2)})
        print(f"{name:<34} {secs:8.1f} s  {mem.peak / 1e9:6.2f} GB", flush=True)
        return out

    path = folder / "benchmark.fcs"
    t0 = time.perf_counter()
    unmixer = make_file(path, n)
    print(f"Made {n:,} events x {len(unmixer.detectors)} detectors "
          f"({path.stat().st_size / 1e9:.2f} GB) in {time.perf_counter() - t0:.0f} s", flush=True)

    ws = Workspace()
    ws.add_unmixer("Aurora", unmixer)
    [raw] = step("Open the file", lambda: ws.add_fcs([path]))
    unmixed = step("Unmix", lambda: ws.unmix(raw, "Aurora"))
    step("Gate (10 nested gates)", lambda: _gate(ws, unmixed))
    step("Clean data", lambda: ws.clean([unmixed]))
    out = folder / "export.fcs"
    step("Export (cleaned events as FCS)", lambda: ws.export_cleaned(unmixed, out))
    return results


def _gate(ws, sid):
    """10 gates, each inside the last, on scatter and unmixed channels; then count all."""
    dim = ws.make_dimension
    parent = ALL
    channels = ["FSC-A", "SSC-A"] + [f"F{i}" for i in range(1, 9)]
    names = {c.pnn for c in ws.channels(sid)}
    channels = [next(n for n in names if n.startswith(c)) for c in channels]
    for i, ch in enumerate(channels):
        x = ws.display_values(sid, ch)
        lo = float(np.quantile(x, 0.03))
        ws.add_gate(fk.gates.RectangleGate(f"G{i + 1}", [dim(sid, ch, lo, None)]), parent)
        parent = parent + (f"G{i + 1}",)
    return [ws.count(sid, n, p) for n, p in ws.population_paths()]


def table(results, baseline, n):
    rows = ["| Step | Time | Baseline | Peak memory |", "|---|---:|---:|---:|"]
    for r in results:
        base = baseline.get(r["step"])
        rows.append(f"| {r['step']} | {r['seconds']:.1f} s | "
                    f"{f'{base:.1f} s' if base else '–'} | {r['peak_gb']:.2f} GB |")
    return f"### Speed test: {n:,} events × 64 detectors\n\n" + "\n".join(rows) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--events", type=int, default=EVENTS)
    ap.add_argument("--record", action="store_true", help="save these times as the baseline")
    ap.add_argument("--memory-gb", type=float, default=MEMORY_GB)
    ap.add_argument("--json", help="also write the results here")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        results = run(args.events, Path(tmp))
    saved = json.loads(BASELINE.read_text()) if BASELINE.exists() else {}
    baseline = saved.get("seconds", {}) if saved.get("events") == args.events else {}
    report = table(results, baseline, args.events)
    print("\n" + report)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(report)
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1))
    if args.record:
        BASELINE.parent.mkdir(exist_ok=True)
        BASELINE.write_text(json.dumps({"events": args.events, "seconds": {
            r["step"]: r["seconds"] for r in results}}, indent=1) + "\n")
        print(f"Recorded {BASELINE}")
        return 0

    problems = []
    for r in results:
        base = baseline.get(r["step"])
        if base and r["seconds"] > SLOWER * max(base, 1.0):
            problems.append(f"{r['step']} took {r['seconds']:.1f} s, over {SLOWER:g}x its baseline "
                            f"of {base:.1f} s")
        if r["peak_gb"] > args.memory_gb:
            problems.append(f"{r['step']} used {r['peak_gb']:.2f} GB, over the {args.memory_gb:g} GB limit")
    for p in problems:
        print("FAIL:", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
