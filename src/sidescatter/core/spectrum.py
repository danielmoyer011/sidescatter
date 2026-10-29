"""Laying out raw detectors the way full-spectrum cytometers show them.

Detectors are grouped by excitation laser from the shortest wavelength to the
longest (UV, violet, blue, yellow-green, red, infrared) and numbered within
each laser, whatever order the FCS file stores them in. Names are read from
the common schemes: Cytek "V7-A" / "YG3-A", BD "B510-A" / "UV379-A" (laser
letter plus emission wavelength) and Sony "405-450" (laser wavelength, then
emission). Detectors with other names still plot; they keep their file order
after the recognised ones and get a neutral colour.
"""

import re

import flowkit as fk
import numpy as np

# (key, laser wavelength nm, colour) from the shortest wavelength to the longest.
LASERS = (
    ("UV", 355, "#7C4DCC"),
    ("V", 405, "#5B5BD6"),
    ("B", 488, "#1F9BD1"),
    ("YG", 561, "#8DB33A"),
    ("R", 640, "#E0524A"),
    ("IR", 808, "#9E2B4F"),
)
UNKNOWN_COLOR = "#6A7187"
_ALIASES = {"BL": "B", "Y": "YG", "G": "YG", "RL": "R", "NIR": "IR"}
_RANK = {key: i for i, (key, _, _) in enumerate(LASERS)}
_COLOR = {key: color for key, _, color in LASERS}

_LETTERS = re.compile(r"^(UV|V|BL|B|YG|Y|G|RL|R|NIR|IR)\s?(\d{1,3})(?:/\d+)?(?:-[AH])?$", re.IGNORECASE)
_WAVELENGTHS = re.compile(r"^(\d{3})[-_/ ](\d{3})(?:/\d+)?(?:-[AH])?$")

# Intensity scale for spectrum plots: biexponential with a little room below
# zero, so dim detectors sit just above the floor as they do on the instrument.
DEFAULT_TOP = 4194304.0     # Cytek Aurora / Northern Lights full scale (2^22)
FLOOR = -1000.0


def _laser_for_wavelength(nm):
    return min(LASERS, key=lambda laser: abs(laser[1] - nm))[0]


def parse_detector(name):
    """(laser key, number) for a raw detector name, or (None, None) when unknown.

    The number is the detector's position (Cytek) or its emission wavelength
    (BD, Sony); either way it sorts the detectors of one laser in order.
    """
    m = _LETTERS.match(name.strip())
    if m:
        letters = m.group(1).upper()
        return _ALIASES.get(letters, letters), int(m.group(2))
    m = _WAVELENGTHS.match(name.strip())
    if m:
        return _laser_for_wavelength(int(m.group(1))), int(m.group(2))
    return None, None


def short_name(name):
    """The detector name as printed under a spectrum: "V7-A" -> "V7"."""
    return re.sub(r"-[AH]$", "", name)


def detector_order(names):
    """Indices that put detectors in laser order, then detector order.

    Unrecognised names keep their file order after the recognised ones, so a
    file with an unfamiliar naming scheme is shown exactly as stored.
    """
    def key(i):
        laser, number = parse_detector(names[i])
        if laser is None:
            return (1, 0, 0, i)
        return (0, _RANK[laser], number, i)

    return sorted(range(len(names)), key=key)


def laser_color(name):
    laser, _ = parse_detector(name)
    return _COLOR.get(laser, UNKNOWN_COLOR)


def _mix(hex_color, other, amount):
    a = np.array([int(hex_color[i:i + 2], 16) for i in (1, 3, 5)], dtype=float)
    b = np.array(other, dtype=float)
    return tuple(int(round(v)) for v in a + (b - a) * amount)


def detector_color(names, index):
    """RGB for one detector: its laser's colour, lighter for the first detectors
    of that laser and deeper for the last, so neighbouring fluors differ."""
    laser, _ = parse_detector(names[index])
    base = _COLOR.get(laser, UNKNOWN_COLOR)
    same = [i for i in detector_order(names) if parse_detector(names[i])[0] == laser]
    if laser is None or len(same) < 2:
        return _mix(base, (0, 0, 0), 0.0)
    f = same.index(index) / (len(same) - 1)
    if f < 0.5:
        return _mix(base, (255, 255, 255), 0.45 * (1 - 2 * f))
    return _mix(base, (0, 0, 0), 0.35 * (2 * f - 1))


def signature_color(signature, names):
    """Colour for a signature line: the colour of its peak detector."""
    return detector_color(names, int(np.argmax(signature)))


def laser_groups(names, order=None):
    """[(laser key or None, first position, last position)] along the ordered axis."""
    order = detector_order(names) if order is None else order
    groups = []
    for pos, i in enumerate(order):
        laser = parse_detector(names[i])[0]
        if groups and groups[-1][0] == laser:
            groups[-1] = (laser, groups[-1][1], pos)
        else:
            groups.append((laser, pos, pos))
    return groups


def intensity_transform(top=None):
    """The biexponential scale used for spectrum plot intensities."""
    return fk.transforms.LogicleTransform(param_t=float(top or DEFAULT_TOP), param_w=0.5,
                                          param_m=4.5, param_a=0.0)


def to_display(xform, values):
    values = np.asarray(values, dtype=float)
    return xform.apply(values.reshape(-1, 1)).reshape(values.shape)


def density(display, lo, hi, bins=200):
    """Per-detector histograms of display-scale values: shape (n_detectors, bins).

    display: (n_events, n_detectors) already on the display scale. Values
    outside [lo, hi] pile up in the first or last bin, as on the instrument.
    """
    display = np.clip(np.asarray(display, dtype=float), lo, hi)
    edges = np.linspace(lo, hi, bins + 1)
    return np.stack([np.histogram(display[:, j], bins=edges)[0] for j in range(display.shape[1])])


def column_scale(counts):
    """Scale each detector's histogram to 0..1 on its own (log of the count),
    so every detector shows its full distribution like a spectrum plot does."""
    counts = np.asarray(counts, dtype=float)
    top = np.log1p(counts.max(axis=1, keepdims=True))
    return np.log1p(counts) / np.where(top > 0, top, 1.0)
