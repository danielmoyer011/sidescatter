"""Guessing what each raw file in a spectral experiment is.

Every file gets a role (unstained reference, single-stain reference,
multicolor sample, or excluded), a carrier (beads or cells), and for single
stains a fluor and marker. Three kinds of evidence are used, strongest first:

  1. where the file sits and what it is called: SpectroFlo exports put
     references in a "Reference Group" folder, unmixed copies in "Unmixed",
     and name reference tubes like "BV421 (Beads)" or "Unstained (Cells)";
  2. FCS keywords such as the tube and group names;
  3. the events themselves: one dominant spectrum over background means a
     single stain, a tight scatter cluster means beads.

Each guess records why it was made, so the UI can show it and the user can
correct it.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import fcs

UNSTAINED, SINGLE_STAIN, SAMPLE, EXCLUDE = "unstained", "single_stain", "sample", "exclude"
ROLES = (UNSTAINED, SINGLE_STAIN, SAMPLE, EXCLUDE)
BEADS, CELLS = "beads", "cells"

# (canonical name, aliases). Matching ignores case, spaces, hyphens and dots,
# so "PE-Cy7", "PE Cy7" and "PECy7" are the same.
_FLUORS = [
    ("BUV395", []), ("BUV496", []), ("BUV563", []), ("BUV615", []), ("BUV661", []),
    ("BUV737", []), ("BUV805", []),
    ("BV421", ["Brilliant Violet 421"]), ("BV480", []), ("BV510", ["Brilliant Violet 510"]),
    ("BV570", []), ("BV605", ["Brilliant Violet 605"]), ("BV650", ["Brilliant Violet 650"]),
    ("BV711", ["Brilliant Violet 711"]), ("BV750", []), ("BV785", ["BV786", "Brilliant Violet 785"]),
    ("BB515", []), ("BB700", []), ("RB545", []), ("RB613", []), ("RB705", []), ("RB780", []),
    ("RY586", []), ("RY610", []), ("RY775", []),
    ("SB436", ["Super Bright 436"]), ("SB600", ["Super Bright 600"]),
    ("SB645", ["Super Bright 645"]), ("SB702", ["Super Bright 702"]), ("SB780", ["Super Bright 780"]),
    ("eF450", ["eFluor 450", "eFluor450"]), ("eF506", ["eFluor 506"]), ("eF660", ["eFluor 660"]),
    ("eF710", ["eFluor 710"]),
    ("PB", ["Pacific Blue", "PacBlue"]), ("PO", ["Pacific Orange", "PacOrange"]),
    ("V450", []), ("V500", []),
    ("FITC", []), ("GFP", ["EGFP"]), ("YFP", ["EYFP"]), ("mCherry", []), ("tdTomato", []),
    ("AF350", ["Alexa Fluor 350", "A350"]), ("AF405", ["Alexa Fluor 405", "A405"]),
    ("AF430", ["Alexa Fluor 430"]), ("AF488", ["Alexa Fluor 488", "A488"]),
    ("AF532", ["Alexa Fluor 532", "A532"]), ("AF555", ["Alexa Fluor 555", "A555"]),
    ("AF568", ["Alexa Fluor 568"]), ("AF594", ["Alexa Fluor 594", "A594"]),
    ("AF647", ["Alexa Fluor 647", "A647"]), ("AF660", ["Alexa Fluor 660"]),
    ("AF700", ["Alexa Fluor 700", "A700"]), ("AF750", ["Alexa Fluor 750"]),
    ("PE", ["R-PE"]), ("PE-CF594", ["CF594"]), ("PE-Dazzle594", ["PE-Dazzle 594", "PE Dazzle"]),
    ("PE-Texas Red", ["PE-TR", "ECD"]), ("PE-AF610", ["PE-Alexa Fluor 610"]),
    ("PE-Cy5", []), ("PE-Cy5.5", []), ("PE-Cy7", []),
    ("PE-Fire 640", []), ("PE-Fire 700", []), ("PE-Fire 744", []), ("PE-Fire 780", []),
    ("PE-Fire 810", []), ("PE-eF610", ["PE-eFluor 610"]),
    ("PerCP", []), ("PerCP-Cy5.5", []), ("PerCP-eF710", ["PerCP-eFluor 710"]),
    ("PerCP-Fire 780", []), ("PerCP-Fire 806", []),
    ("APC", []), ("APC-Cy7", []), ("APC-H7", []), ("APC-R700", []), ("APC-eF780", ["APC-eFluor 780"]),
    ("APC-Fire 750", []), ("APC-Fire 810", []), ("APC-AF700", []), ("APC-Vio770", []),
    ("Cy5", []), ("Cy7", []),
    ("Zombie UV", []), ("Zombie Violet", []), ("Zombie Aqua", []), ("Zombie Green", []),
    ("Zombie Yellow", []), ("Zombie Red", []), ("Zombie NIR", []),
    ("LIVE/DEAD Blue", ["LD Blue", "Live Dead Blue"]), ("LIVE/DEAD Aqua", ["LD Aqua"]),
    ("LIVE/DEAD NIR", ["LD NIR"]), ("eF780 Viability", ["FVD780", "FVD eF780"]),
    ("DAPI", []), ("7-AAD", ["7AAD"]), ("PI", ["Propidium Iodide"]), ("Hoechst", []),
]
# Families with many numbered members: "Spark Blue 550", "cFluor V420", "NovaFluor Red 660".
_FAMILY = re.compile(
    r"^(SPARK(UV|VIOLET|BLUE|YG|RED|NIR)?|CFLUOR(UV|V|B|YG|R)?|NOVAFLUOR(UV|BLUE|YELLOW|RED)?"
    r"|REALBLUE|REALYELLOW|VIOBRIGHT|VIOGREEN|VIOBLUE|STARBRIGHT(UV|VIOLET|BLUE|YELLOW|RED)?)(\d{3})$")
_FAMILY_NAMES = {"SPARK": "Spark", "CFLUOR": "cFluor", "NOVAFLUOR": "NovaFluor", "REALBLUE": "RealBlue",
                 "REALYELLOW": "RealYellow", "VIOBRIGHT": "VioBright", "VIOGREEN": "VioGreen",
                 "VIOBLUE": "VioBlue", "STARBRIGHT": "StarBright"}


def _norm(text):
    return re.sub(r"[\s\-_.]", "", text).upper()


_ALIASES = {}
for _name, _more in _FLUORS:
    for _a in [_name, *_more]:
        _ALIASES.setdefault(_norm(_a), _name)
_ALIASES.setdefault("APCCY7", "APC-Cy7")

_SPLIT = re.compile(r"[\s_\-()\[\],;+]+")
_UNSTAINED_WORDS = {"UNSTAINED", "UNSTAIN", "US", "UNS", "UNSTN", "NEG", "NEGATIVE", "BLANK",
                    "AUTOFLUORESCENCE", "AF", "NOSTAIN"}
_BEAD_WORDS = {"BEAD", "BEADS", "ULTRACOMP", "ULTRACOMPEBEADS", "ONECOMP", "COMPBEADS", "VERSACOMP",
               "ABC", "CAPTURE"}
_CELL_WORDS = {"CELL", "CELLS", "PBMC", "PBMCS", "BLOOD", "WB", "SPLEEN", "SPLENOCYTES", "THYMUS",
               "LN", "LYMPH", "BM", "DEAD", "LIVE", "HK", "HEATKILLED", "KILLED", "CELLLINE", "JURKAT"}
_SAMPLE_WORDS = {"SAMPLE", "SAMPLES", "FULL", "PANEL", "FULLSTAIN", "FULLYSTAINED", "MULTICOLOR",
                 "MULTICOLOUR", "MC", "FMO", "TEST", "SPECIMEN", "STIM", "UNSTIM", "DONOR", "PATIENT",
                 "TREATED"}
_REFERENCE_WORDS = {"REFERENCE", "REF", "REFERENCES", "SINGLE", "SINGLESTAIN", "SS", "COMP", "COMPENSATION",
                    "CTRL", "SC", "SCC"}
_FILLER = re.compile(r"^([A-P]\d{1,2}|\d+|TUBE|GROUP|WELL|EXP|EXPT|STAIN(ED)?|ONLY|AND|THE|OF|WITH)$",
                     re.IGNORECASE)
# FCS keyword names (FlowKit lowercases them and drops "$") that hold tube or group names.
GROUP_KEYWORDS = ("groupname", "group name", "group", "plate name", "platename")
TUBE_KEYWORDS = ("tubename", "tube name", "src", "sample id", "sampleid", "well id", "wellid", "smno")


@dataclass
class NameInfo:
    """What a file or tube name says."""

    fluors: list = field(default_factory=list)
    marker: str = ""
    unstained: bool = False
    carrier: str = None      # BEADS, CELLS or None
    sample_hint: bool = False
    reference_hint: bool = False
    words: list = field(default_factory=list)  # the evidence, for display


def find_fluor(tokens, start):
    """Longest fluorochrome name starting at tokens[start]: (canonical, n_tokens) or None."""
    for n in range(min(4, len(tokens) - start), 0, -1):
        key = _norm("".join(tokens[start:start + n]))
        if key in _ALIASES:
            return _ALIASES[key], n
        m = _FAMILY.match(key)
        if m:
            fam = next(k for k in _FAMILY_NAMES if key.startswith(k))
            rest, num = key[len(fam):-3], key[-3:]
            fam = _FAMILY_NAMES[fam]
            if not rest:
                return f"{fam} {num}", n
            if len(rest) <= 2:
                return f"{fam} {rest}{num}", n
            return f"{fam} {rest if rest in ('NIR', 'YG', 'UV') else rest.title()} {num}", n
    return None


def parse_name(name):
    """Read roles, carrier, fluor and marker out of a file or tube name."""
    tokens = [t for t in _SPLIT.split(name) if t]
    info = NameInfo()
    leftover = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        up = _norm(tok)
        hit = find_fluor(tokens, i)
        # "AF" alone and "US" are role words; "AF647" is a fluor.
        if hit and up not in _UNSTAINED_WORDS:
            info.fluors.append(hit[0])
            info.words.append(" ".join(tokens[i:i + hit[1]]))
            i += hit[1]
            continue
        if up in _UNSTAINED_WORDS or up.startswith("UNSTAIN"):
            info.unstained = True
            info.words.append(tok)
        elif up in _BEAD_WORDS or up.endswith("BEADS"):
            info.carrier = info.carrier or BEADS
            info.words.append(tok)
        elif up in _CELL_WORDS or up.endswith("CELLS"):
            info.carrier = info.carrier or CELLS
            info.words.append(tok)
        elif up in _SAMPLE_WORDS:
            info.sample_hint = True
            info.words.append(tok)
        elif up in _REFERENCE_WORDS:
            info.reference_hint = True
            info.words.append(tok)
        elif not _FILLER.match(tok):
            leftover.append(tok)
        i += 1
    info.marker = " ".join(leftover)
    return info


@dataclass
class Guess:
    role: str
    carrier: str
    fluor: str = ""
    marker: str = ""
    reasons: list = field(default_factory=list)
    source: str = "name"  # strongest evidence used: "folder", "name", "keywords", "data" or "default"
    evidence: "DataEvidence" = None


def _keyword(meta, keys):
    for k in keys:
        v = meta.get(k)
        if v not in (None, ""):
            return str(v)
    return ""


@dataclass
class DataEvidence:
    single_stain_like: bool
    rank1: float        # share of spectral variance in the brightest events' main shape
    brightness: float   # 99th percentile of each event's brightest detector
    carrier: str = None


def data_evidence(sample):
    """What the events suggest, or None when there is too little to go on.

    Single stains: the brightest events share one spectrum shape (the first
    principal component of background-subtracted, length-normalised spectra
    carries most of the variance). Beads: side scatter is a tight cluster.
    Unstained files are told apart by brightness relative to the other files
    (see classify_files), since one file alone can't say what "dim" is.
    """
    dets = fcs.detector_channels(sample)
    if len(dets) < 3 or sample.event_count < 50:
        return None
    raw = sample.get_events(source="raw")
    rows = np.sort(np.random.default_rng(0).choice(len(raw), min(len(raw), 20_000), replace=False))
    Y = raw[rows][:, [c.index for c in dets]].astype(np.float64)
    total = Y.sum(axis=1)
    bg = np.median(Y[total <= np.quantile(total, 0.3)], axis=0)
    diff = np.clip(Y[total >= np.quantile(total, 0.9)] - bg, 0, None)
    shapes = diff / np.maximum(np.linalg.norm(diff, axis=1, keepdims=True), 1e-9)
    sv = np.linalg.svd(shapes, compute_uv=False)
    rank1 = float(sv[0] ** 2 / max((sv ** 2).sum(), 1e-12))

    carrier = None
    ssc = [c for c in fcs.channels(sample) if c.pnn.upper() in ("SSC-A", "SSC-H")]
    if ssc:
        side = np.log10(np.clip(raw[rows, ssc[0].index], 1, None))
        carrier = BEADS if (np.percentile(side, 84) - np.percentile(side, 16)) / 2 < 0.12 else CELLS
    return DataEvidence(rank1 >= 0.85, rank1, float(np.percentile(Y.max(axis=1), 99)), carrier)


# A file counts as unstained when it is within this factor of the dimmest file
# of the same carrier.
UNSTAINED_BRIGHTNESS_FACTOR = 3.0


def classify(path, sample=None, use_data=True):
    """Guess the role, carrier, fluor and marker of one raw FCS file."""
    path = Path(path)
    reasons = []
    folders = [p.upper() for p in path.parts[:-1]]

    if sample is None:
        sample = fcs.load_sample(path)
    if "UNMIXED" in folders or not fcs.is_spectral(sample):
        why = "in an Unmixed folder" if "UNMIXED" in folders else "not raw full-spectrum data"
        return Guess(EXCLUDE, CELLS, reasons=[why], source="folder")

    meta = fcs.keywords(sample)
    group = _keyword(meta, GROUP_KEYWORDS)
    tube = _keyword(meta, TUBE_KEYWORDS)
    in_reference_folder = any("REFERENCE" in f for f in folders[-2:])
    in_group_keyword = "REFERENCE" in group.upper()

    info = parse_name(path.stem)
    kw = parse_name(tube) if tube and _norm(tube) != _norm(path.stem) else NameInfo()

    fluors = info.fluors or kw.fluors
    unstained = info.unstained or (not info.fluors and kw.unstained)
    carrier = info.carrier or kw.carrier
    marker = info.marker if info.fluors else (kw.marker if kw.fluors else "")
    if info.words:
        reasons.append("name: " + ", ".join(f"'{w}'" for w in info.words))
    if kw.words:
        reasons.append(f"tube keyword '{tube}'")

    reference = None  # True: reference group, False: sample group, None: unknown
    source = "name"
    if in_reference_folder:
        reference, source = True, "folder"
        reasons.insert(0, "in a Reference Group folder")
    elif in_group_keyword:
        reference, source = True, "keywords"
        reasons.insert(0, f"group keyword '{group}'")
    elif group:
        reference = False
        reasons.append(f"group keyword '{group}'")
    elif any(f in ("SAMPLES", "SAMPLE") for f in folders[-2:]):
        reference = False
        reasons.append("in a Samples folder")
    elif info.reference_hint or kw.reference_hint:
        reference = True

    if unstained and not (info.sample_hint and info.fluors):
        role = UNSTAINED
    elif len(fluors) == 1 and not info.sample_hint and reference is not False:
        role = SINGLE_STAIN
    elif fluors or info.sample_hint or reference is False:
        role = SAMPLE
        if len(fluors) > 1:
            reasons.append(f"{len(fluors)} fluors in the name")
    elif reference:
        role = SINGLE_STAIN
    else:
        role = None

    evidence = data_evidence(sample) if use_data else None
    if role is None:
        if evidence is None:
            role, source = SAMPLE, "default"
            reasons.append("nothing in the name; assumed a sample")
        else:
            source = "data"
            role = SINGLE_STAIN if evidence.single_stain_like else SAMPLE
            reasons.append(f"data: {'one dominant spectrum' if evidence.single_stain_like else 'mixed spectra'}"
                           f" ({evidence.rank1:.2f})")
    if carrier is None:
        if evidence is not None and evidence.carrier is not None:
            carrier = evidence.carrier
            reasons.append(f"data: scatter looks like {carrier}")
        else:
            carrier = CELLS
    fluor = fluors[0] if role == SINGLE_STAIN and fluors else ""
    if role == UNSTAINED:
        marker = ""
    return Guess(role, carrier, fluor, marker, reasons, source, evidence)


def classify_files(paths, samples=None, use_data=True):
    """Classify a batch of files: classify() on each, then compare brightness.

    Files whose role came only from the data are called unstained when they
    are about as dim as the dimmest file of the same carrier.
    samples: optional {path: loaded Sample} to avoid loading files twice.
    """
    samples = samples or {}
    guesses = [classify(p, samples.get(str(p)) or samples.get(p), use_data) for p in paths]
    for carrier in (BEADS, CELLS):
        group = [g for g in guesses if g.carrier == carrier and g.evidence is not None
                 and g.role != EXCLUDE]
        if len(group) < 2:
            continue
        dimmest = min(g.evidence.brightness for g in group)
        if dimmest * 20 > max(g.evidence.brightness for g in group):
            continue  # no clearly dim file, so probably no unstained one
        for g in group:
            if g.source == "data" and g.evidence.brightness <= UNSTAINED_BRIGHTNESS_FACTOR * dimmest:
                g.role = UNSTAINED
                g.reasons[0] = (f"data: about as dim as the dimmest {carrier} file "
                                f"({g.evidence.brightness / max(dimmest, 1e-9):.1f}x)")
    return guesses
