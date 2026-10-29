"""Known fluor spectra, for checking the signatures measured from controls.

A control that gives a spectrum unlike its fluor (the wrong tube, a dim or
degraded tandem, too few positive events) is the commonest cause of bad
unmixing, so each measured signature is compared with a reference by cosine
similarity, as AutoSpectral does (Burton et al. 2025). References come from:

- your own library: spectra you saved from good controls, kept per detector
  layout in a JSON file in the user's data folder. These are exact matches for
  your instrument, so the whole spectrum is compared.
- predictions from published dye spectra (FPbase excitation and emission,
  bundled in resources/spectra) and each detector's laser and band. Laser
  powers and detector gains differ between instruments, so only the detectors
  of the measured spectrum's peak laser are compared.
"""

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import classify, spectrum

THRESHOLD = 0.9   # cosine below which a control is flagged
DYES_FILE = Path(__file__).resolve().parents[1] / "resources" / "spectra" / "fpbase_dyes.json"

# Cytek Aurora / Northern Lights detector bands as (centre nm, width nm), per laser,
# from Cytek's published 5-laser configuration. Other configurations that keep
# these modules (3 and 4 lasers, Northern Lights) number their detectors the same.
CYTEK_BANDS = {
    "UV": [(373, 15), (388, 15), (428, 15), (443, 15), (458, 15), (473, 15), (514, 28), (542, 28),
           (582, 31), (613, 31), (664, 27), (692, 28), (720, 29), (750, 30), (780, 30), (812, 34)],
    "V": [(428, 15), (443, 15), (458, 15), (473, 15), (508, 20), (525, 17), (542, 17), (581, 19),
          (598, 20), (615, 20), (664, 27), (692, 28), (720, 29), (750, 30), (780, 30), (812, 34)],
    "B": [(508, 20), (525, 17), (542, 17), (581, 19), (598, 20), (615, 20), (661, 17), (679, 18),
          (697, 19), (717, 20), (738, 21), (760, 23), (783, 23), (812, 34)],
    "YG": [(577, 20), (598, 20), (615, 20), (661, 17), (679, 18), (697, 19), (720, 29), (750, 30),
           (780, 30), (812, 34)],
    "R": [(661, 17), (679, 18), (697, 19), (717, 20), (738, 21), (760, 23), (783, 23), (812, 34)],
}
LASER_NM = {key: nm for key, nm, _ in spectrum.LASERS}
DEFAULT_WIDTH = 20   # nm, for detectors named by their centre wavelength only


def canonical_fluor(name):
    """Sidescatter's standard name for a fluor label ("PE Cy7" -> "PE-Cy7"), or the label."""
    tokens = [t for t in re.split(r"[\s_]+", name.strip()) if t]
    found = classify.find_fluor(tokens, 0) if tokens else None
    return found[0] if found and found[1] == len(tokens) else name.strip()


def detector_bands(detectors):
    """(laser nm, centre nm, width nm) for each detector, or None where unknown.

    Cytek names (V7-A) are looked up in CYTEK_BANDS, but only for a laser whose
    detector count matches the table, since older layouts number them otherwise.
    BD (B510-A) and Sony (405-450) names carry the centre wavelength.
    """
    parsed = [spectrum.parse_detector(d) for d in detectors]
    counts = {}
    for laser, number in parsed:
        if laser is not None and number is not None and number < 100:
            counts[laser] = counts.get(laser, 0) + 1
    out = []
    for name, (laser, number) in zip(detectors, parsed):
        if laser is None:
            out.append(None)
        elif number >= 300:
            m = re.match(r"^(\d{3})[-_/ ]", name.strip())
            out.append((float(m.group(1)) if m else float(LASER_NM[laser]), float(number),
                        float(DEFAULT_WIDTH)))
        elif laser in CYTEK_BANDS and counts.get(laser) == len(CYTEK_BANDS[laser]) \
                and 1 <= number <= len(CYTEK_BANDS[laser]):
            centre, width = CYTEK_BANDS[laser][number - 1]
            out.append((float(LASER_NM[laser]), float(centre), float(width)))
        else:
            out.append(None)
    return out


@lru_cache(maxsize=1)
def _dyes():
    try:
        return json.loads(DYES_FILE.read_text())["dyes"]
    except (OSError, ValueError, KeyError):
        return {}


def known_dyes():
    return sorted(_dyes())


def _interp(curve, nm):
    x, y = curve[:, 0], curve[:, 1]
    return np.interp(nm, x, y, left=0.0, right=0.0)


def predict(fluor, detectors):
    """Predicted signature of a fluor on these detectors: (signature, FPbase name) or None.

    Each detector gets excitation at its laser times the mean emission across
    its band. Detectors with no known band are 0. None when the dye isn't
    bundled or no detector band is known.
    """
    dye = _dyes().get(canonical_fluor(fluor))
    bands = detector_bands(detectors)
    if dye is None or not any(bands):
        return None
    ex, em = np.asarray(dye["ex"], dtype=float), np.asarray(dye["em"], dtype=float)
    sig = np.zeros(len(detectors))
    for i, band in enumerate(bands):
        if band is None:
            continue
        laser, centre, width = band
        nm = np.linspace(centre - width / 2, centre + width / 2, 9)
        sig[i] = _interp(ex, laser) * _interp(em, nm).mean()
    if sig.max() <= 0:
        return None
    return sig / sig.max(), dye["fpbase"]


def cosine(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def peak_laser(signature, detectors):
    """Indices of the detectors on the same laser as the signature's peak."""
    laser = spectrum.parse_detector(detectors[int(np.argmax(signature))])[0]
    if laser is None:
        return list(range(len(detectors)))
    return [i for i, d in enumerate(detectors) if spectrum.parse_detector(d)[0] == laser]


# ----- your own library ------------------------------------------------------

def default_path():
    """The library file: $SIDESCATTER_DATA_DIR, else the platform's per-user data folder."""
    if os.environ.get("SIDESCATTER_DATA_DIR"):
        base = Path(os.environ["SIDESCATTER_DATA_DIR"])
    elif sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home())) / "Sidescatter"
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support" / "Sidescatter"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "sidescatter"
    return base / "spectra_library.json"


class SpectraLibrary:
    """Spectra saved from good controls, one per fluor, carrier and detector layout."""

    def __init__(self, path=None):
        self.path = Path(path) if path else default_path()
        self.entries = []
        try:
            self.entries = json.loads(self.path.read_text()).get("spectra", [])
        except (OSError, ValueError):
            pass

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"format": 1, "spectra": self.entries}, indent=1))
        tmp.replace(self.path)

    def find(self, fluor, detectors, carrier=None):
        """The newest saved entry for this fluor on exactly these detectors (same
        carrier preferred), or None."""
        key = canonical_fluor(fluor).upper()
        hits = [e for e in self.entries
                if e["fluor"].upper() == key and e["detectors"] == list(detectors)]
        hits.sort(key=lambda e: (e.get("carrier") == carrier, e.get("saved", "")))
        return hits[-1] if hits else None

    def add(self, fluor, detectors, signature, carrier="", source=""):
        """Save a spectrum, replacing the one for the same fluor, carrier and layout."""
        key = canonical_fluor(fluor)
        self.entries = [e for e in self.entries
                        if not (e["fluor"].upper() == key.upper() and e["detectors"] == list(detectors)
                                and e.get("carrier", "") == carrier)]
        entry = {"fluor": key, "carrier": carrier, "detectors": list(detectors),
                 "signature": [round(float(v), 6) for v in signature], "source": source,
                 "saved": time.strftime("%Y-%m-%d %H:%M")}
        self.entries.append(entry)
        return entry

    def remove(self, fluor, detectors, carrier=""):
        key = canonical_fluor(fluor).upper()
        before = len(self.entries)
        self.entries = [e for e in self.entries
                        if not (e["fluor"].upper() == key and e["detectors"] == list(detectors)
                                and e.get("carrier", "") == carrier)]
        return len(self.entries) < before


# ----- the check -------------------------------------------------------------

@dataclass
class Check:
    """How a measured signature compares with its reference."""

    cosine: float
    reference: np.ndarray       # the reference spectrum on the experiment's detectors
    kind: str                   # "saved" (your library) or "predicted" (FPbase)
    label: str                  # what the reference is, for display
    threshold: float = THRESHOLD

    @property
    def passed(self):
        return self.cosine >= self.threshold


def check(fluor, signature, detectors, library=None, carrier=None, threshold=THRESHOLD):
    """Compare a measured signature with your saved spectrum, else the predicted one.

    Returns a Check, or None when there is nothing to compare with.
    """
    detectors = list(detectors)
    saved = library.find(fluor, detectors, carrier) if library is not None else None
    if saved is not None:
        ref = np.asarray(saved["signature"], dtype=float)
        where = f" from {saved['source']}" if saved.get("source") else ""
        return Check(cosine(signature, ref), ref, "saved",
                     f"your saved {saved['fluor']}{where} ({saved.get('saved', '')})", threshold)
    predicted = predict(fluor, detectors)
    if predicted is None:
        return None
    ref, dye = predicted
    bands = detector_bands(detectors)
    idx = [i for i in peak_laser(signature, detectors) if bands[i] is not None]
    if not idx:
        return None   # no band known on the laser this control peaks on
    if not np.any(ref[idx] > 0):
        return Check(0.0, ref, "predicted", f"{dye} predicted from FPbase spectra", threshold)
    return Check(cosine(np.asarray(signature)[idx], ref[idx]), ref, "predicted",
                 f"{dye} predicted from FPbase spectra", threshold)
