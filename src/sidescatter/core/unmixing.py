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


def build_unmixer(specs, method="ols", n_autofluorescence=1, autofluorescence=True, samples=None):
    """Build an Unmixer from control specs.

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
    if autofluorescence and cell_unstained is not None:
        af = spectral.autofluorescence_signatures(gated_events(cell_unstained), n_autofluorescence)
        for i, sig in enumerate(af):
            label = "Autofluorescence" if len(af) == 1 else f"Autofluorescence {i + 1}"
            references.append(spectral.Reference(label, sig, True, cell_unstained.path.name))

    unmixer = spectral.Unmixer(names, references, method, background_var)
    return unmixer, results


# Events per block when unmixing; bounds the extra memory to a few hundred MB
# however many events a file has.
CHUNK = 200_000


def unmix_sample(sample, unmixer, keep_autofluorescence=True):
    """Return a new FlowKit Sample: scatter/time channels plus one channel per fluor.

    Works through the events in blocks so a multi-million-event, 64-detector
    file never needs a second full copy of its detector data. The source
    file's keywords (date, cytometer, tube name...) are carried over.
    """
    events, pnn, pns, ranges, meta = unmixed_events(sample, unmixer, keep_autofluorescence)
    return fcs.load_sample(events, sample_id=f"{sample.id} (unmixed)", channel_labels=pnn,
                           pns_labels=pns, keywords=meta)


def unmixed_events(sample, unmixer, keep_autofluorescence=True):
    """Unmixed events plus what an FCS file of them needs: (events, PnN, PnS, PnR, keywords)."""
    chans = fcs.channels(sample)
    det_idx = _detector_index(sample, unmixer.detectors)
    keep = [c for c in chans if c.kind in ("time", "scatter")]
    fluors = [i for i, r in enumerate(unmixer.references)
              if keep_autofluorescence or not r.is_autofluorescence]
    labels = unmixer.channel_labels()
    pnn = [c.pnn for c in keep] + [labels[i][0] for i in fluors]
    pns = [c.pns for c in keep] + [labels[i][1] for i in fluors]

    raw = sample.get_events(source="raw")  # FlowKit returns its array, not a copy
    out = np.empty((raw.shape[0], len(pnn)), dtype=np.float32)
    out[:, :len(keep)] = raw[:, [c.index for c in keep]]
    for start in range(0, raw.shape[0], CHUNK):
        block = slice(start, start + CHUNK)
        out[block, len(keep):] = unmixer.unmix(raw[block][:, det_idx])[:, fluors]

    pnr = sample.channels["pnr"].tolist()
    det_range = max((pnr[i] for i in det_idx), default=262144)
    ranges = [pnr[c.index] for c in keep] + [det_range] * len(fluors)
    meta = {k: v for k, v in fcs.keywords(sample).items()
            if not re.match(r"^p\d+[a-z]+$", k)}
    meta["SIDESCATTER_UNMIXING"] = f"{unmixer.method.upper()}, {len(unmixer.references)} components"
    return out, pnn, pns, ranges, meta


def unmixing_qc(sample, unmixer):
    """Quality numbers for one sample: residual size per detector and per event."""
    Y = _detector_events(sample, unmixer.detectors)
    A = unmixer.unmix(Y)
    R = spectral.residuals(Y, A, unmixer.matrix)
    scale = np.maximum(np.abs(Y).sum(axis=1), 1e-9)
    return {
        "detector_rms": dict(zip(unmixer.detectors, np.sqrt((R ** 2).mean(axis=0)))),
        "relative_residual_median": float(np.median(np.abs(R).sum(axis=1) / scale)),
        "similarity": spectral.similarity_matrix(unmixer.matrix),
        "complexity_index": spectral.complexity_index(unmixer.matrix),
    }
