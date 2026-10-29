"""Turning reference control files into an Unmixer, and unmixing samples.

Controls can be single-stained beads or cells. Each single stain is paired
with a negative: an unstained file of the same kind (beads with unstained
beads, cells with unstained cells) when one exists, otherwise the dim events
inside the control itself. Unstained cells also supply autofluorescence.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import classify, fcs, spectral



@dataclass
class ControlSpec:
    """One raw file in an unmixing experiment and what it is used for."""

    path: Path
    name: str                       # fluor label shown to the user, e.g. "PE-Cy7"
    role: str = "single_stain"      # "single_stain", "unstained", "sample" or "exclude"
    carrier: str = "cells"          # "beads" or "cells"
    negative: str = "auto"          # "auto", "internal", or a path to an unstained file
    positive_mask: np.ndarray = None  # manual gates override the automatic ones
    negative_mask: np.ndarray = None
    marker: str = ""                # antigen, e.g. "CD3"
    reasons: list = field(default_factory=list)  # why the role was guessed
    guessed_by: str = ""            # strongest evidence: folder, name, keywords, data, default
    # Gates drawn in the Identify step, all in raw units and saved with the setup:
    # a polygon on the scatter channels, and Positive/Negative ranges on one detector.
    scatter_gate: list = None       # [[x, y], ...] on fcs.scatter_channels
    gate_detector: str = ""         # detector the ranges are on (the peak channel)
    positive_range: tuple = None    # (low, high)
    negative_range: tuple = None    # (low, high); only used with an internal negative
    # Unstained cells: hand-drawn AF populations, each a dict with name, x and
    # y (detector names) and vertices [[x, y], ...] in raw detector units.
    af_gates: list = field(default_factory=list)
    af_source: str = "shared"       # samples: "shared" AF library, or an unstained file's path

    @property
    def is_reference(self):
        return self.role in ("single_stain", "unstained")


FileSpec = ControlSpec


def _legacy_name(stem):
    return re.sub(r"^(beads?|cells?|pbmc|dead)[_\- ]+", "", stem, flags=re.IGNORECASE) or stem


def guess_files(paths, samples=None, use_data=True):
    """Classify every file of an experiment: references (single stain or
    unstained, on beads or cells), multicolor samples, or files to leave out.
    """
    paths = [Path(p) for p in paths]
    guesses = classify.classify_files(paths, samples, use_data)
    return [ControlSpec(p, g.fluor or _legacy_name(p.stem), g.role, g.carrier, marker=g.marker,
                        reasons=g.reasons, guessed_by=g.source)
            for p, g in zip(paths, guesses)]


def guess_controls(paths, samples=None):
    """Classify files known to be controls: unstained vs single stain, beads vs cells."""
    specs = guess_files(paths, samples)
    for spec in specs:
        if not spec.is_reference:
            spec.role = "single_stain"
    return specs


def check_specs(specs):
    """Problems that stop or weaken unmixing, as (level, message); level is "error" or "warning"."""
    problems = []
    singles = [s for s in specs if s.role == "single_stain"]
    if not singles:
        problems.append(("error", "No single-stain reference controls."))
    seen = {}
    for s in singles:
        key = s.name.strip().upper()
        if not key:
            problems.append(("error", f"{s.path.name} has no fluor name."))
        elif key in seen:
            problems.append(("error", f"Two single-stain controls for {s.name}: {seen[key]} and "
                                      f"{s.path.name}. Set one of them to Exclude."))
        else:
            seen[key] = s.path.name
    for carrier in sorted({s.carrier for s in singles}):
        if not any(u.role == "unstained" and u.carrier == carrier for u in specs):
            problems.append(("warning", f"No unstained {carrier}: {carrier} single stains will use "
                                        "their own dim events as the negative."))
    if not any(s.role == "unstained" and s.carrier == "cells" for s in specs):
        problems.append(("warning", "No unstained cells, so no autofluorescence signature."))
    if not any(s.role == "sample" for s in specs):
        problems.append(("warning", "No multicolor samples to unmix."))
    return problems


def _matching_unstained(spec, specs):
    """The unstained file of the same carrier as spec, if any."""
    for other in specs:
        if other.role == "unstained" and other.carrier == spec.carrier:
            return other
    return None


def _detector_index(sample, detectors, label=None):
    """Column indices of the named detectors, matched by name, in that order."""
    by_name = {c.pnn: c.index for c in fcs.channels(sample)}
    missing = [d for d in detectors if d not in by_name]
    if missing:
        more = f" and {len(missing) - 5} more" if len(missing) > 5 else ""
        raise ValueError(
            f"{label or sample.id} is missing detectors {', '.join(missing[:5])}{more}. "
            "Controls and samples must come from the same instrument configuration.")
    return [by_name[d] for d in detectors]


def _detector_events(sample, detectors, label=None):
    return sample.get_events(source="raw")[:, _detector_index(sample, detectors, label)]


def in_polygon(x, y, vertices):
    """Events inside a polygon (even-odd rule); points on an edge count as inside."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    v = np.asarray(vertices, dtype=float)
    inside = np.zeros(x.shape, dtype=bool)
    on_edge = np.zeros(x.shape, dtype=bool)
    for (x0, y0), (x1, y1) in zip(v, np.roll(v, -1, axis=0)):
        crosses = (y0 > y) != (y1 > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            at = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
        inside ^= crosses & (x < at)
        # On the segment: collinear and within its bounding box.
        cross = (x1 - x0) * (y - y0) - (y1 - y0) * (x - x0)
        scale = max(abs(x1 - x0) + abs(y1 - y0), 1e-12)
        on_edge |= ((np.abs(cross) <= 1e-9 * scale * scale)
                    & (np.minimum(x0, x1) <= x) & (x <= np.maximum(x0, x1))
                    & (np.minimum(y0, y1) <= y) & (y <= np.maximum(y0, y1)))
    return inside | on_edge


def scatter_mask(sample, vertices):
    """Events inside a scatter gate drawn on fcs.scatter_channels; all events without one."""
    pair = fcs.scatter_channels(sample)
    if vertices is None or len(vertices) < 3 or pair is None:
        return np.ones(sample.event_count, dtype=bool)
    raw = sample.get_events(source="raw")
    return in_polygon(raw[:, sample.get_channel_index(pair[0])],
                      raw[:, sample.get_channel_index(pair[1])], vertices)


def _range_mask(values, bounds):
    lo, hi = sorted(map(float, bounds))
    return (values >= lo) & (values <= hi)


def _expand(sub_mask, keep):
    full = np.zeros(keep.shape, dtype=bool)
    full[np.flatnonzero(keep)[sub_mask]] = True
    return full


@dataclass
class ControlResult:
    spec: ControlSpec
    signature: np.ndarray
    peak_detector: str
    positive_mask: np.ndarray
    negative_source: str
    n_positive: int
    n_negative: int
    in_gate: np.ndarray = None        # events inside the scatter gate
    negative_mask: np.ndarray = None  # internal negatives; None when another file is the negative


def build_unmixer(specs, method="ols", n_autofluorescence=1, autofluorescence=True, samples=None,
                  af_mode="fluors", af_library_size=30, af_auto=True):
    """Build an Unmixer from control specs.

    af_mode: "fluors" (n_autofluorescence AF signatures as extra fluors) or
    "per_cell" (an AF library of up to af_library_size spectra per unstained
    cells file, plus its drawn AF gates; af_auto=False keeps only the gates).

    Returns (unmixer, results) where results describes how each signature was
    derived so the UI can show it and let the user correct the gates.
    samples: optional {path: loaded Sample}, filled in as files are loaded.
    """
    specs = [s for s in specs if s.is_reference]
    samples = {} if samples is None else samples

    def sample_for(path):
        if path not in samples:
            samples[path] = fcs.load_sample(path)
        return samples[path]

    first = sample_for(specs[0].path)
    detectors = fcs.detector_channels(first)
    names = [c.pnn for c in detectors]
    by_path = {s.path: s for s in specs}

    def det_events(spec):
        return _detector_events(sample_for(spec.path), names, spec.path.name)

    def gated_events(spec):
        """Detector events inside the file's scatter gate (all events without one)."""
        events = det_events(spec)
        if spec.scatter_gate is None:
            return events
        return events[scatter_mask(sample_for(spec.path), spec.scatter_gate)]

    references, results = [], []
    for spec in specs:
        if spec.role != "single_stain":
            continue
        events = det_events(spec)
        in_gate = scatter_mask(sample_for(spec.path), spec.scatter_gate)

        neg_events, neg_source = None, "internal"
        if spec.negative_mask is not None:
            neg_events = events[spec.negative_mask]
        elif spec.negative not in ("auto", "internal"):
            other = by_path.get(Path(spec.negative)) or ControlSpec(Path(spec.negative), "")
            neg_events, neg_source = gated_events(other), Path(spec.negative).name
        elif spec.negative == "auto":
            unstained = _matching_unstained(spec, specs)
            if unstained is not None:
                neg_events, neg_source = gated_events(unstained), unstained.path.name

        pos_sub, neg_sub, peak = spectral.auto_gate_control(events[in_gate], neg_events)
        pos_mask = _expand(pos_sub, in_gate)
        internal_neg = None if neg_sub is None else _expand(neg_sub, in_gate)
        if spec.gate_detector in names:
            peak = names.index(spec.gate_detector)
        if spec.positive_range is not None:
            drawn = in_gate & _range_mask(events[:, peak], spec.positive_range)
            if drawn.sum() >= 10:
                pos_mask = drawn
        if spec.positive_mask is not None:
            pos_mask = spec.positive_mask
        if neg_events is None:
            if spec.negative_range is not None:
                drawn = in_gate & _range_mask(events[:, peak], spec.negative_range)
                if drawn.sum() >= 10:
                    internal_neg = drawn
            neg_events = events[internal_neg]

        sig = spectral.extract_signature(events[pos_mask], neg_events)
        references.append(spectral.Reference(spec.name, sig, source=spec.path.name, marker=spec.marker))
        results.append(ControlResult(spec, sig, names[peak], pos_mask, neg_source,
                                     int(pos_mask.sum()), len(neg_events), in_gate,
                                     internal_neg if neg_source == "internal" else None))

    background_var = None
    cell_unstained = next((s for s in specs if s.role == "unstained" and s.carrier == "cells"), None)
    any_unstained = cell_unstained or next((s for s in specs if s.role == "unstained"), None)
    if any_unstained is not None:
        background_var = spectral.robust_sd(gated_events(any_unstained)) ** 2
    unmixer = spectral.Unmixer(names, references, method, background_var)
    if autofluorescence and cell_unstained is not None:
        unstained = [(s, gated_events(s)) for s in specs if s.role == "unstained" and s.carrier == "cells"]
        add_autofluorescence(unmixer, unstained, af_mode, n_autofluorescence, af_library_size, af_auto)
    return unmixer, results


AF_MODES = ("per_cell", "fluors")


def add_autofluorescence(unmixer, unstained, mode="fluors", n_signatures=1, library_size=30,
                         auto=True):
    """Add autofluorescence to an Unmixer that has only fluor references.

    unstained: [(ControlSpec, detector events)] for each unstained-cells file.
    mode "fluors": n_signatures AF spectra from the first file become extra
    fluors (SpectroFlo's way). mode "per_cell": every file's AF library
    (automatic clusters and/or the file's drawn AF gates) is pooled, and each
    cell is matched to its best library entry when unmixing.
    """
    if mode not in AF_MODES:
        raise ValueError(f"unknown autofluorescence mode {mode!r}")
    spec, Y = unstained[0]
    if mode == "fluors":
        af = spectral.autofluorescence_signatures(Y, n_signatures)
        for i, sig in enumerate(af):
            label = "Autofluorescence" if len(af) == 1 else f"Autofluorescence {i + 1}"
            unmixer.references.append(spectral.Reference(label, sig, True, spec.path.name))
        return unmixer
    fluors = unmixer.matrix
    spectra, sources = [], []
    for spec, Y in unstained:
        lib = af_library_for(spec, Y, unmixer.detectors, library_size, auto, fluors)
        spectra.extend(lib)
        sources.extend([str(spec.path)] * len(lib))
    # Near-duplicates are dropped within each file but kept across files, so a
    # sample limited to one tissue's unstained still has all of its spectra.
    unmixer.af_library = np.array(spectra)
    unmixer.af_sources = sources
    unmixer.references.append(spectral.Reference(
        "Autofluorescence", spectral.normalise(np.median(unstained[0][1], axis=0)), True,
        ", ".join(s.path.name for s, _ in unstained)))
    return unmixer


_library_cache = {}


def af_library_for(spec, Y, detectors, library_size=30, auto=True, fluors=None):
    """One unstained file's AF library: drawn gates first, then the automatic clusters.

    Automatic clustering is the slow part, so it is cached on the events and settings.
    """
    spectra = []
    for gate in spec.af_gates:
        mask = af_gate_mask(gate, Y, detectors)
        if mask.sum() >= 5:
            try:
                spectra.append(spectral.normalise(np.median(Y[mask], axis=0)))
            except ValueError:
                pass
    if auto or not spectra:
        step = max(1, len(Y) // 997)
        key = (str(spec.path), Y.shape, float(Y[::step].sum()), library_size if auto else 1)
        if key not in _library_cache:
            if len(_library_cache) > 32:
                _library_cache.clear()
            _library_cache[key] = spectral.af_library(Y, library_size if auto else 1)[0]
        lib = _library_cache[key]
        if fluors is not None and len(fluors):
            lib = lib[spectral.cosine(lib, fluors).max(axis=1) < 0.99]
        spectra.extend(lib)
    if not spectra:
        spectra = [spectral.normalise(np.median(Y, axis=0))]
    return np.array(spectra)[spectral.dedupe_spectra(np.array(spectra))]


def af_gate_mask(gate, Y, detectors):
    """Events of Y (columns named by detectors) inside a drawn AF gate."""
    if gate["x"] not in detectors or gate["y"] not in detectors:
        return np.zeros(len(Y), dtype=bool)
    xi, yi = detectors.index(gate["x"]), detectors.index(gate["y"])
    return in_polygon(Y[:, xi], Y[:, yi], np.asarray(gate["vertices"], dtype=float))


# Events per block when unmixing; bounds the extra memory to a few hundred MB
# however many events a file has.
CHUNK = 200_000


def unmix_sample(sample, unmixer, keep_autofluorescence=True, af_source=None, progress=None):
    """Return a new FlowKit Sample: scatter/time channels plus one channel per fluor.

    Works through the events in blocks so a multi-million-event, 64-detector
    file never needs a second full copy of its detector data. The source
    file's keywords (date, cytometer, tube name...) are carried over.
    af_source and progress: see unmixed_events.
    """
    events, pnn, pns, ranges, meta = unmixed_events(sample, unmixer, keep_autofluorescence,
                                                    af_source, progress)
    return fcs.load_sample(events, sample_id=f"{sample.id} (unmixed)", channel_labels=pnn,
                           pns_labels=pns, keywords=meta)


def unmixed_events(sample, unmixer, keep_autofluorescence=True, af_source=None, progress=None):
    """Unmixed events plus what an FCS file of them needs: (events, PnN, PnS, PnR, keywords).

    af_source: with per-cell autofluorescence, the unstained file whose AF
    library entries this sample uses ("shared" or None: all of them).
    progress(done, total): called after each block of events.
    """
    chans = fcs.channels(sample)
    det_idx = _detector_index(sample, unmixer.detectors)
    keep = [c for c in chans if c.kind in ("time", "scatter")]
    fluors = [i for i, r in enumerate(unmixer.references)
              if keep_autofluorescence or not r.is_autofluorescence]
    if unmixer.per_cell_af and keep_autofluorescence:
        fluors.append(len(unmixer.references))  # AF Index
    if af_source == "shared":
        af_source = None
    labels = unmixer.channel_labels()
    pnn = [c.pnn for c in keep] + [labels[i][0] for i in fluors]
    pns = [c.pns for c in keep] + [labels[i][1] for i in fluors]

    raw = sample.get_events(source="raw")  # FlowKit returns its array, not a copy
    out = np.empty((raw.shape[0], len(pnn)), dtype=np.float32)
    out[:, :len(keep)] = raw[:, [c.index for c in keep]]
    for start in range(0, raw.shape[0], CHUNK):
        block = slice(start, start + CHUNK)
        out[block, len(keep):] = unmixer.unmix(raw[block][:, det_idx], af_source)[:, fluors]
        if progress:
            progress(min(start + CHUNK, raw.shape[0]), raw.shape[0])

    pnr = sample.channels["pnr"].tolist()
    det_range = max((pnr[i] for i in det_idx), default=262144)
    ranges = [pnr[c.index] for c in keep] + [det_range] * len(fluors)
    meta = {k: v for k, v in fcs.keywords(sample).items()
            if not re.match(r"^p\d+[a-z]+$", k)}
    meta["SIDESCATTER_UNMIXING"] = f"{unmixer.method.upper()}, {len(unmixer.references)} components"
    if unmixer.per_cell_af:
        meta["SIDESCATTER_UNMIXING"] += f", per-cell AF from {len(unmixer.af_library)} spectra"
    return out, pnn, pns, ranges, meta


def unmixing_qc(sample, unmixer):
    """Quality numbers for one sample: residual size per detector and per event."""
    Y = _detector_events(sample, unmixer.detectors)
    A = unmixer.unmix(Y)
    R = unmixer.residuals(Y, A)
    scale = np.maximum(np.abs(Y).sum(axis=1), 1e-9)
    return {
        "detector_rms": dict(zip(unmixer.detectors, np.sqrt((R ** 2).mean(axis=0)))),
        "relative_residual_median": float(np.median(np.abs(R).sum(axis=1) / scale)),
        "similarity": spectral.similarity_matrix(unmixer.matrix),
        "complexity_index": spectral.complexity_index(unmixer.matrix),
    }


AF_METHOD_LABELS = {"per_cell": "Per-cell AF", "fluors": "AF as fluors"}


def af_check(specs, unmixer, samples=None, n_autofluorescence=1, af_library_size=30, af_auto=True,
             max_events=50_000):
    """Unmix the unstained cells with per-cell AF and with AF as fluors, side by side.

    A good AF model leaves unstained cells at zero in every fluor channel, so
    the spread (robust SD) and median left there, and the per-detector
    residual, show which way fits these cells better. Returns None without
    unstained cells, else {"fluors", "detectors", "n_events", "methods":
    {mode: {"median", "spread", "residual_rms", "unmixer"}}}.
    """
    samples = {} if samples is None else samples
    unstained = [s for s in specs if s.role == "unstained" and s.carrier == "cells"]
    if not unstained:
        return None
    rng = np.random.default_rng(0)
    events = []
    for spec in unstained:
        if spec.path not in samples:
            samples[spec.path] = fcs.load_sample(spec.path)
        Y = _detector_events(samples[spec.path], unmixer.detectors, spec.path.name)
        if spec.scatter_gate is not None:
            Y = Y[scatter_mask(samples[spec.path], spec.scatter_gate)]
        events.append((spec, Y))
    pooled = np.vstack([Y for _, Y in events])
    if len(pooled) > max_events:
        pooled = pooled[np.sort(rng.choice(len(pooled), max_events, replace=False))]
    fluor_refs = [r for r in unmixer.references if not r.is_autofluorescence]
    fluor_idx = list(range(len(fluor_refs)))
    out = {"fluors": [r.name for r in fluor_refs], "detectors": list(unmixer.detectors),
           "n_events": len(pooled), "methods": {}}
    for mode in AF_MODES:
        um = spectral.Unmixer(list(unmixer.detectors), list(fluor_refs), unmixer.method,
                              unmixer.background_var, channel_style=unmixer.channel_style)
        add_autofluorescence(um, events, mode, n_autofluorescence, af_library_size, af_auto)
        A = um.unmix(pooled)
        R = um.residuals(pooled, A)
        out["methods"][mode] = {
            "median": np.median(A[:, fluor_idx], axis=0),
            "spread": spectral.robust_sd(A[:, fluor_idx]),
            "residual_rms": np.sqrt((R ** 2).mean(axis=0)),
            "unmixer": um,
        }
    return out


def af_check_text(check):
    """The AF comparison as lines of text, for the QC report."""
    if check is None:
        return []
    per, flu = check["methods"]["per_cell"], check["methods"]["fluors"]
    width = max([len(n) for n in check["fluors"]] + [12]) + 2
    lines = ["Autofluorescence check: unstained cells unmixed both ways "
             f"({check['n_events']:,} events). Each fluor should sit at 0; lower spread is better.",
             f"  {'Fluor':<{width}}{'Per-cell AF':>24}{'AF as fluors':>24}",
             f"  {'':<{width}}{'median':>12}{'spread':>12}{'median':>12}{'spread':>12}"]
    for i, name in enumerate(check["fluors"]):
        lines.append(f"  {name:<{width}}{per['median'][i]:12.1f}{per['spread'][i]:12.1f}"
                     f"{flu['median'][i]:12.1f}{flu['spread'][i]:12.1f}")
    lines += [f"  Total spread: per-cell {per['spread'].sum():,.0f}, as fluors {flu['spread'].sum():,.0f}",
              f"  Residual RMS per detector (mean): per-cell {per['residual_rms'].mean():,.1f}, "
              f"as fluors {flu['residual_rms'].mean():,.1f}",
              f"  Per-cell AF library: {len(per['unmixer'].af_library)} spectra"]
    return lines
