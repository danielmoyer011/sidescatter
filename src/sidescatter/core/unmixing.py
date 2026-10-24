"""Turning reference control files into an Unmixer, and unmixing samples.

Controls can be single-stained beads or cells. Each single stain is paired
with a negative: an unstained file of the same kind (beads with unstained
beads, cells with unstained cells) when one exists, otherwise the dim events
inside the control itself. Unstained cells also supply autofluorescence.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import fcs, spectral

UNSTAINED_PATTERN = re.compile(r"unstain|unstained|\bUS\b|negative|blank", re.IGNORECASE)
BEAD_PATTERN = re.compile(r"bead", re.IGNORECASE)


@dataclass
class ControlSpec:
    path: Path
    name: str                       # fluor label shown to the user, e.g. "PE-Cy7"
    role: str = "single_stain"      # "single_stain" or "unstained"
    carrier: str = "cells"          # "beads" or "cells"
    negative: str = "auto"          # "auto", "internal", or a path to an unstained file
    positive_mask: np.ndarray = None  # manual gates override the automatic ones
    negative_mask: np.ndarray = None


def guess_controls(paths):
    """Classify control files by name: unstained vs single stain, beads vs cells."""
    specs = []
    for p in map(Path, paths):
        stem = p.stem
        role = "unstained" if UNSTAINED_PATTERN.search(stem) else "single_stain"
        carrier = "beads" if BEAD_PATTERN.search(stem) else "cells"
        name = re.sub(r"^(beads?|cells?|pbmc|dead)[_\- ]+", "", stem, flags=re.IGNORECASE)
        specs.append(ControlSpec(p, name or stem, role, carrier))
    return specs


def _matching_unstained(spec, specs):
    """The unstained file of the same carrier as spec, if any."""
    for other in specs:
        if other.role == "unstained" and other.carrier == spec.carrier:
            return other
    return None


def _detector_events(sample, detectors, label):
    """Events for the named detectors, in that order, matched by name."""
    by_name = {c.pnn: c for c in fcs.channels(sample)}
    missing = [d for d in detectors if d not in by_name]
    if missing:
        more = f" and {len(missing) - 5} more" if len(missing) > 5 else ""
        raise ValueError(
            f"{label} is missing detectors {', '.join(missing[:5])}{more}. Controls and "
            "samples must come from the same instrument configuration.")
    return fcs.raw_events(sample, [by_name[d] for d in detectors])


@dataclass
class ControlResult:
    spec: ControlSpec
    signature: np.ndarray
    peak_detector: str
    positive_mask: np.ndarray
    negative_source: str
    n_positive: int
    n_negative: int


def build_unmixer(specs, method="ols", n_autofluorescence=1, autofluorescence=True):
    """Build an Unmixer from control specs.

    Returns (unmixer, results) where results describes how each signature was
    derived so the UI can show it and let the user correct the gates.
    """
    samples = {}

    def sample_for(path):
        if path not in samples:
            samples[path] = fcs.load_sample(path)
        return samples[path]

    first = sample_for(specs[0].path)
    detectors = fcs.detector_channels(first)
    names = [c.pnn for c in detectors]

    def det_events(spec):
        return _detector_events(sample_for(spec.path), names, spec.path.name)

    references, results = [], []
    for spec in specs:
        if spec.role != "single_stain":
            continue
        events = det_events(spec)

        neg_events, neg_source = None, "internal"
        if spec.negative_mask is not None:
            neg_events = events[spec.negative_mask]
        elif spec.negative not in ("auto", "internal"):
            neg_events, neg_source = det_events(ControlSpec(Path(spec.negative), "")), Path(spec.negative).name
        elif spec.negative == "auto":
            unstained = _matching_unstained(spec, specs)
            if unstained is not None:
                neg_events, neg_source = det_events(unstained), unstained.path.name

        pos_mask, internal_neg, peak = spectral.auto_gate_control(events, neg_events)
        if spec.positive_mask is not None:
            pos_mask = spec.positive_mask
        if neg_events is None:
            neg_events = events[internal_neg]

        sig = spectral.extract_signature(events[pos_mask], neg_events)
        references.append(spectral.Reference(spec.name, sig, source=spec.path.name))
        results.append(ControlResult(spec, sig, names[peak], pos_mask, neg_source,
                                     int(pos_mask.sum()), len(neg_events)))

    background_var = None
    cell_unstained = next((s for s in specs if s.role == "unstained" and s.carrier == "cells"), None)
    any_unstained = cell_unstained or next((s for s in specs if s.role == "unstained"), None)
    if any_unstained is not None:
        background_var = spectral.robust_sd(det_events(any_unstained)) ** 2
    if autofluorescence and cell_unstained is not None:
        af = spectral.autofluorescence_signatures(det_events(cell_unstained), n_autofluorescence)
        for i, sig in enumerate(af):
            label = "Autofluorescence" if len(af) == 1 else f"Autofluorescence {i + 1}"
            references.append(spectral.Reference(label, sig, True, cell_unstained.path.name))

    unmixer = spectral.Unmixer(names, references, method, background_var)
    return unmixer, results


def unmix_sample(sample, unmixer, keep_autofluorescence=True):
    """Return a new FlowKit Sample: scatter/time channels plus one channel per fluor."""
    chans = fcs.channels(sample)
    Y = _detector_events(sample, unmixer.detectors, sample.id)
    A = unmixer.unmix(Y)

    keep = [c for c in chans if c.kind in ("time", "scatter")]
    columns = [fcs.raw_events(sample, keep)]
    labels = [c.pnn for c in keep]
    for i, ref in enumerate(unmixer.references):
        if ref.is_autofluorescence and not keep_autofluorescence:
            continue
        columns.append(A[:, i:i + 1])
        labels.append(ref.name)
    data = np.hstack(columns)
    return fcs.load_sample(data, sample_id=f"{sample.id} (unmixed)", channel_labels=labels)


def unmixing_qc(sample, unmixer):
    """Quality numbers for one sample: residual size per detector and per event."""
    Y = _detector_events(sample, unmixer.detectors, sample.id)
    A = unmixer.unmix(Y)
    R = spectral.residuals(Y, A, unmixer.matrix)
    scale = np.maximum(np.abs(Y).sum(axis=1), 1e-9)
    return {
        "detector_rms": dict(zip(unmixer.detectors, np.sqrt((R ** 2).mean(axis=0)))),
        "relative_residual_median": float(np.median(np.abs(R).sum(axis=1) / scale)),
        "similarity": spectral.similarity_matrix(unmixer.matrix),
        "complexity_index": spectral.complexity_index(unmixer.matrix),
    }
