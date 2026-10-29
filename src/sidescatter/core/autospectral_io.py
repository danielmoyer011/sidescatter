"""Reading and writing AutoSpectral's files, to compare or combine the two tools.

AutoSpectral (Burton et al. 2025, an R package) keeps spectra as CSV files
written by R's write.csv: a blank first header cell, then one column per
detector, and one row per fluor named in the first column ("AF" for
autofluorescence; an autofluorescence file has one row per AF spectrum). Its
control file is a CSV with one row per control: filename, fluorophore
("AF" for unstained cells, "Negative" for unstained beads), marker, channel
(peak detector), control.type ("cells" or "beads"), universal.negative (the
matching unstained file; the AF row names itself), large.gate, gate.name,
gate.define and is.viability.
"""

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import spectral

CONTROL_COLUMNS = ["filename", "fluorophore", "marker", "channel", "control.type",
                   "universal.negative", "large.gate", "gate.name", "gate.define", "is.viability"]
_VIABILITY = re.compile(r"ZOMBIE|LIVE.?DEAD|VIABILITY|FVD|DCM|7-?AAD|^PI$|DAPI|L/D|\bLD\b|DEAD", re.I)


@dataclass
class Spectra:
    """Spectra read from an AutoSpectral CSV."""

    path: Path
    detectors: list
    names: list            # fluor rows, in file order
    fluors: np.ndarray     # (k, d), peak-normalised
    af: np.ndarray         # (m, d) autofluorescence rows, peak-normalised (may be empty)

    def to_dict(self):
        return {"file": str(self.path), "detectors": list(self.detectors), "names": list(self.names),
                "fluors": self.fluors.tolist(), "af": self.af.tolist()}

    @classmethod
    def from_dict(cls, d):
        n = len(d["detectors"])
        return cls(Path(d["file"]), list(d["detectors"]), list(d["names"]),
                   np.asarray(d["fluors"], dtype=float).reshape(-1, n),
                   np.asarray(d["af"], dtype=float).reshape(-1, n))

    def signature(self, name):
        """The imported spectrum for a fluor name (case and punctuation ignored), or None."""
        key = _key(name)
        for n, sig in zip(self.names, self.fluors):
            if _key(n) == key:
                return sig
        return None

    def on(self, detectors):
        """A copy with columns in the order of detectors; ValueError if any is missing."""
        missing = [d for d in detectors if d not in self.detectors]
        if missing:
            raise ValueError(f"{self.path.name} has no {', '.join(missing[:5])}"
                             + (f" and {len(missing) - 5} more" if len(missing) > 5 else "")
                             + "; it was made on another instrument configuration.")
        idx = [self.detectors.index(d) for d in detectors]
        return Spectra(self.path, list(detectors), list(self.names), self.fluors[:, idx], self.af[:, idx])


def _key(name):
    return re.sub(r"[\s\-_.]", "", str(name)).upper()


def _is_af(name, path):
    """An autofluorescence row: "AF" or "AF1".."AF99" as AutoSpectral names them (Alexa
    Fluor dyes have three digits), "Autofluorescence...", or any row of an
    autofluorescence spectra file."""
    key = _key(name)
    return (bool(re.fullmatch(r"AF\d{0,2}", key)) or key.startswith("AUTOFLUORESCENCE")
            or "autofluorescence" in path.name.lower())


def read_spectra(path):
    """Read an AutoSpectral spectra (or autofluorescence spectra) CSV."""
    path = Path(path)
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.reader(f) if r and any(c.strip() for c in r)]
    if len(rows) < 2 or len(rows[0]) < 3:
        raise ValueError(f"{path.name} isn't an AutoSpectral spectra file (no detector columns).")
    detectors = [c.strip() for c in rows[0][1:]]
    names, fluors, af = [], [], []
    for r in rows[1:]:
        try:
            values = np.array([float(v) if v.strip() not in ("", "NA") else 0.0
                               for v in r[1:len(detectors) + 1]])
        except ValueError:
            raise ValueError(f"{path.name}: row {r[0]!r} has values that aren't numbers.") from None
        if len(values) != len(detectors) or values.max() <= 0:
            continue
        sig = spectral.normalise(values)
        if _is_af(r[0], path):
            af.append(sig)
        else:
            names.append(r[0].strip())
            fluors.append(sig)
    if not names and not af:
        raise ValueError(f"{path.name} has no spectra.")
    d = len(detectors)
    return Spectra(path, detectors, names, np.array(fluors).reshape(-1, d), np.array(af).reshape(-1, d))


def write_spectra(path, names, detectors, spectra):
    """Write spectra as AutoSpectral reads them (read.spectra): rows named by fluor."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([""] + list(detectors))
        for name, sig in zip(names, spectra):
            w.writerow([name] + [f"{float(v):.6g}" for v in sig])
    return Path(path)


def _bool(v):
    return "TRUE" if v else "FALSE"


def control_rows(specs, results):
    """Rows of an AutoSpectral control file for the reference controls in specs."""
    refs = [s for s in specs if s.is_reference]
    by_path = {r.spec.path: r for r in results or ()}
    unstained = {c: next((s for s in refs if s.role == "unstained" and s.carrier == c), None)
                 for c in ("cells", "beads")}
    rows = []
    for s in refs:
        if s.role == "unstained":
            fluor = "AF" if s.carrier == "cells" else "Negative"
            neg = s.path.name if s.carrier == "cells" else ""
            channel, marker = "", ""
        else:
            fluor, marker = s.name, s.marker
            r = by_path.get(s.path)
            channel = r.peak_detector if r is not None else (s.gate_detector or "")
            if s.negative == "auto":
                neg = unstained[s.carrier].path.name if unstained[s.carrier] is not None else ""
            elif s.negative == "internal":
                neg = ""
            else:
                neg = Path(s.negative).name
        viability = s.role == "single_stain" and bool(_VIABILITY.search(f"{s.name} {s.marker}"))
        rows.append({"filename": s.path.name, "fluorophore": fluor, "marker": marker,
                     "channel": channel, "control.type": s.carrier, "universal.negative": neg,
                     "large.gate": _bool(False), "gate.name": "", "gate.define": _bool(True),
                     "is.viability": _bool(viability)})
    return rows


def write_control_file(path, specs, results=None):
    """An AutoSpectral control file for these controls (filenames as on disk)."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, CONTROL_COLUMNS)
        w.writeheader()
        w.writerows(control_rows(specs, results))
    return Path(path)


def export(folder, specs, unmixer, results=None, prefix="Sidescatter"):
    """Write <prefix>_spectra.csv (fluors, plus "AF" when AF is a single spectrum),
    <prefix>_Autofluorescence.csv (the AF library or AF signatures, when any) and
    <prefix>_control_file.csv into folder. Returns the paths written.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    fluors = [r for r in unmixer.references if not r.is_autofluorescence]
    afs = [r for r in unmixer.references if r.is_autofluorescence]
    names = [r.name for r in fluors]
    spectra = [r.signature for r in fluors]
    if len(afs) == 1 and not unmixer.per_cell_af:
        names.append("AF")
        spectra.append(afs[0].signature)
    written = [write_spectra(folder / f"{prefix}_spectra.csv", names, unmixer.detectors, spectra)]
    library = (unmixer.af_library if unmixer.per_cell_af
               else np.array([r.signature for r in afs]) if len(afs) > 1 else None)
    if library is not None and len(library):
        written.append(write_spectra(folder / f"{prefix}_Autofluorescence.csv",
                                     [f"AF{i + 1}" for i in range(len(library))], unmixer.detectors,
                                     library))
    written.append(write_control_file(folder / f"{prefix}_control_file.csv", specs, results))
    return written
