"""Loading FCS files and describing their channels."""

import io
import re
from dataclasses import dataclass
from pathlib import Path

import flowkit as fk
import numpy as np

# Full-spectrum instruments have 20+ fluorescence detectors: Cytek Aurora and
# Northern Lights 14-64 (1-5 lasers), BD FACSDiscover and Symphony A5 SE 48-78,
# Sony ID7000 up to 184. Nothing below assumes a laser or detector count:
# detectors are whatever fluorescence channels the file contains.
SPECTRAL_MIN_DETECTORS = 20
# Raw detector names: laser letter(s) plus a detector number (Cytek "UV7-A",
# "YG3-A") or a centre wavelength (BD "B510-A", "UV379-A", Sony "405-450").
# Unmixed or conventional files use fluorochrome names ("BV421-A", "PE-Cy7-A").
_LASER = r"(UV|V|B|BL|YG|Y|G|R|RL|IR|NIR)"
DETECTOR_NAME = re.compile(
    rf"^({_LASER}\s?\d{{1,2}}|{_LASER}\s?\d{{3}}(/\d+)?|\d{{3}}[-_/ ]\d{{3}}(/\d+)?|CH\s?\d+)(-[AH])?$",
    re.IGNORECASE)
_SCATTER = ("FSC", "SSC", "BSC", "FS ", "SS ", "FS-", "SS-", "LIGHTLOSS", "LIGHT LOSS")
_OTHER = ("EVENT", "WIDTH", "SORT", "INDEX", "SAMPLE", "TRIGGER", "BEAD", "GATE",
          "LASER", "DROP", "WELL", "PLATE", "IMAGE", "DIAMETER")


@dataclass(frozen=True)
class Channel:
    index: int
    pnn: str
    pns: str
    kind: str  # "time", "scatter", "fluorescence"

    @property
    def label(self):
        return f"{self.pns} :: {self.pnn}" if self.pns else self.pnn


def _channel_kind(pnn):
    name = pnn.upper()
    if name.startswith("TIME"):
        return "time"
    if name.startswith(_SCATTER) or "SCATTER" in name:
        return "scatter"
    if name.startswith(_OTHER) or name.endswith("-W"):
        return "other"
    return "fluorescence"


def find_fcs_files(paths):
    """Expand dropped files and folders into FCS paths.

    Folders are searched recursively. Returns (fcs_paths, skipped): FCS files
    in a stable order with duplicates removed, and the names of anything else.
    Hidden files (e.g. macOS "._" resource forks) inside folders are ignored.
    """
    found, skipped = [], []
    for p in map(Path, paths):
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                rel = f.relative_to(p).parts
                if f.is_file() and f.suffix.lower() == ".fcs" and not any(x.startswith(".") for x in rel):
                    found.append(f)
        elif p.is_file() and p.suffix.lower() == ".fcs":
            found.append(p)
        else:
            skipped.append(p.name)
    unique = list(dict.fromkeys(str(f.resolve()) for f in found))
    return unique, skipped


def load_sample(path_or_data, sample_id=None, channel_labels=None, pns_labels=None, keywords=None):
    """Load an FCS file (or an events array with channel_labels) as a FlowKit Sample.

    Full-resolution events are kept; subsampling is only used for plotting.
    An array goes through an in-memory FCS file, which keeps $PnS descriptions
    and keywords and avoids FlowKit's slow list conversion.
    """
    if isinstance(path_or_data, (str, Path)):
        path = Path(path_or_data)
        return fk.Sample(
            str(path),
            sample_id=sample_id or path.name,
            ignore_offset_error=True,
            subsample=20000,
        )
    fh = io.BytesIO()
    write_fcs(fh, path_or_data, channel_labels, pns_labels, keywords)
    fh.seek(0)
    return fk.Sample(fh, sample_id=sample_id, subsample=20000)


def channels(sample):
    return [
        Channel(i, pnn, sample.pns_labels[i], _channel_kind(pnn))
        for i, pnn in enumerate(sample.pnn_labels)
    ]


def detector_channels(sample):
    """Fluorescence detector channels, preferring area (-A) parameters."""
    fluor = [c for c in channels(sample) if c.kind == "fluorescence"]
    area = [c for c in fluor if c.pnn.upper().endswith("-A")]
    return area or fluor


def scatter_channels(sample):
    """The (x, y) channels for a forward/side scatter gate, or None.

    FSC-A against SSC-A when both exist (Cytek also has SSC-B-A, which comes
    second); otherwise the first two scatter area channels.
    """
    scatter = [c.pnn for c in channels(sample) if c.kind == "scatter"]
    if "FSC-A" in scatter and "SSC-A" in scatter:
        return "FSC-A", "SSC-A"
    area = [n for n in scatter if n.upper().endswith("-A")] or scatter
    return tuple(area[:2]) if len(area) >= 2 else None


def is_spectral(sample):
    """Raw full-spectrum data (as opposed to conventional or already-unmixed files)."""
    dets = detector_channels(sample)
    if len(dets) < SPECTRAL_MIN_DETECTORS:
        return False
    named = sum(bool(DETECTOR_NAME.match(c.pnn)) for c in dets)
    # Unknown naming schemes still count when there are clearly too many
    # channels to be a fluorochrome panel exported after unmixing.
    return named >= 0.8 * len(dets) or len(dets) >= 45


def raw_events(sample, channel_list=None):
    """Raw (uncompensated, untransformed) events, optionally restricted to channels."""
    events = sample.get_events(source="raw")
    if channel_list is None:
        return events
    return events[:, [c.index for c in channel_list]]


def keywords(sample):
    return dict(sample.get_metadata())


# Keywords that describe the file's own layout; write_fcs sets these itself.
_LAYOUT_KEYWORDS = re.compile(
    r"^\$?(BEGIN|END)(ANALYSIS|DATA|STEXT)$|^\$?(BYTEORD|DATATYPE|MODE|NEXTDATA|PAR|TOT)$"
    r"|^\$?P\d+[A-Z]+$|^\$?(SPILL|SPILLOVER|COMP)$|^\$?FIL$", re.IGNORECASE)
_BLOCK = 100_000


def write_fcs(target, events, pnn, pns=None, keywords=None, ranges=None):
    """Write an FCS 3.1 file of 32-bit float events.

    target: a path or a binary file handle. keywords: extra TEXT keywords
    (for example the source file's $DATE and $CYT); layout keywords in it are
    ignored. ranges: optional $PnR per channel (default 262144).
    Events are written in blocks, so a large array is never copied whole.
    """
    events = np.asarray(events)
    n, k = events.shape
    if len(pnn) != k:
        raise ValueError(f"{len(pnn)} channel names for {k} columns")
    pns = list(pns) if pns is not None else [""] * k
    ranges = list(ranges) if ranges is not None else [262144] * k

    text = {}
    for key, value in (keywords or {}).items():
        key = str(key).upper()
        key = key if key.startswith("$") or not _is_standard(key) else "$" + key
        if not _LAYOUT_KEYWORDS.match(key) and str(value) != "":
            text[key] = str(value)
    for i in range(k):
        text[f"$P{i + 1}N"] = pnn[i]
        text[f"$P{i + 1}B"] = "32"
        text[f"$P{i + 1}E"] = "0,0"
        text[f"$P{i + 1}R"] = str(int(ranges[i]))
        if pns[i]:
            text[f"$P{i + 1}S"] = pns[i]
    text.update({"$BYTEORD": "1,2,3,4", "$DATATYPE": "F", "$MODE": "L", "$NEXTDATA": "0",
                 "$PAR": str(k), "$TOT": str(n), "$BEGINANALYSIS": "0", "$ENDANALYSIS": "0",
                 "$BEGINSTEXT": "0", "$ENDSTEXT": "0"})

    # The delimiter is a character no keyword or value uses. FCS allows doubling
    # it inside values, but not every reader undoes that.
    used = set("".join(text) + "".join(text.values()))
    delim = next((c for c in "|/\\\x0c\x1e" if c not in used), "|")

    def body(begin, end):
        pairs = dict(text, **{"$BEGINDATA": f"{begin:020d}", "$ENDDATA": f"{end:020d}"})
        esc = lambda v: v.replace(delim, delim * 2)  # noqa: E731
        return (delim + "".join(f"{esc(a)}{delim}{esc(b)}{delim}" for a, b in pairs.items())).encode("utf-8")

    text_start = 58
    text_len = len(body(0, 0))  # the offsets are fixed width, so the length is known now
    data_start = text_start + text_len
    data_end = data_start + 4 * n * k - 1 if n * k else data_start
    header = b"FCS3.1    " + b"".join(f"{v:>8}".encode() for v in (
        text_start, data_start - 1,
        data_start if data_end <= 99_999_999 else 0, data_end if data_end <= 99_999_999 else 0,
        0, 0))

    def write(fh):
        fh.write(header)
        fh.write(body(data_start, data_end))
        for start in range(0, n, _BLOCK):
            fh.write(np.ascontiguousarray(events[start:start + _BLOCK], dtype="<f4").tobytes())

    if isinstance(target, (str, Path)):
        with open(target, "wb") as fh:
            write(fh)
    else:
        write(target)


_STANDARD = {"DATE", "CYT", "CYTSN", "BTIM", "ETIM", "TIMESTEP", "OP", "SRC", "SMNO", "INST",
             "PROJ", "CELLS", "COM", "EXP", "FIL", "LAST_MODIFIED", "LAST_MODIFIER", "ORIGINALITY",
             "PLATEID", "PLATENAME", "WELLID", "VOL", "SYS", "ABRT", "LOST", "FLOWRATE"}


def _is_standard(key):
    """FlowKit drops the '$' from keywords; put it back on the standard FCS ones."""
    return key in _STANDARD or bool(re.match(r"^P\d+[A-Z]+$", key))
