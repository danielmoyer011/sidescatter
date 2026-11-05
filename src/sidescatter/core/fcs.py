"""Loading FCS files and describing their channels."""

import io
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import flowkit as fk
import numpy as np
import pandas as pd

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
# Channels Sidescatter computes itself (UMAP, tSNE, PCA, FlowSOM); see core.highdim.
DERIVED_PREFIXES = ("UMAP-", "TSNE-", "PCA-", "FLOWSOM")
_OTHER = ("EVENT", "WIDTH", "SORT", "INDEX", "SAMPLE", "TRIGGER", "BEAD", "GATE",
          "LASER", "DROP", "WELL", "PLATE", "IMAGE", "DIAMETER")


@dataclass(frozen=True)
class Channel:
    index: int
    pnn: str
    pns: str
    kind: str  # "time", "scatter", "fluorescence", "derived" or "other"

    @property
    def label(self):
        return f"{self.pns} :: {self.pnn}" if self.pns else self.pnn


def _channel_kind(pnn):
    name = pnn.upper()
    if name.startswith("TIME"):
        return "time"
    if name.startswith(DERIVED_PREFIXES):
        return "derived"
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


def load_sample(path_or_data, sample_id=None, channel_labels=None, pns_labels=None, keywords=None,
                ranges=None, keep_spillover=False):
    """Load an FCS file (or an events array with channel_labels) as a FlowKit Sample.

    Full-resolution events are kept; subsampling is only used for plotting.
    An array goes through an in-memory FCS file, which keeps $PnS descriptions
    and keywords and avoids FlowKit's slow list conversion.

    Files are checked first (see read_fcs): one that can't be read raises
    UnreadableFCS saying why in plain words, and one that reads but looks wrong
    gets its warnings in sample.load_warnings.
    """
    if isinstance(path_or_data, (str, Path)):
        return read_fcs(path_or_data, sample_id)
    fh = io.BytesIO()
    write_fcs(fh, path_or_data, channel_labels, pns_labels, keywords, ranges, keep_spillover)
    fh.seek(0)
    sample = fk.Sample(fh, sample_id=sample_id, subsample=20000)
    sample.load_warnings = []
    return sample


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


def append_channels(sample, names, columns, pns=None):
    """Add computed channels to a loaded sample, in place.

    columns: one array of sample.event_count values per name. The values go
    into the raw events and, when the sample is compensated, the compensated
    events too (computed channels are never compensated).
    """
    pns = list(pns) if pns is not None else [""] * len(names)
    block = np.column_stack([np.asarray(c, dtype=np.float64) for c in columns])
    sample._raw_events = np.hstack([sample._raw_events, block])
    if sample._comp_events is not None:
        sample._comp_events = np.hstack([sample._comp_events, block])
    sample._transformed_events = None
    start = len(sample.pnn_labels)
    rows = [{"channel_number": start + i + 1, "pnn": n, "pns": s, "pne": (0.0, 0.0), "png": 1.0,
             "pnr": float(np.nanmax(np.abs(c))) if np.isfinite(c).any() else 1.0}
            for i, (n, s, c) in enumerate(zip(names, pns, block.T))]
    sample.channels = pd.concat([sample.channels, pd.DataFrame(rows, index=range(start, start + len(rows)))])
    sample.pnn_labels = list(sample.pnn_labels) + list(names)
    sample.pns_labels = list(sample.pns_labels) + list(pns)


def drop_channels(sample, names):
    """Remove channels (added by append_channels) from a loaded sample, in place."""
    drop = [i for i, n in enumerate(sample.pnn_labels) if n in set(names)]
    if not drop:
        return
    if any(_channel_kind(sample.pnn_labels[i]) != "derived" for i in drop):
        raise ValueError("only computed channels can be removed")
    keep = [i for i in range(len(sample.pnn_labels)) if i not in drop]
    sample._raw_events = sample._raw_events[:, keep]
    if sample._comp_events is not None:
        sample._comp_events = sample._comp_events[:, keep]
    sample._transformed_events = None
    sample.channels = sample.channels.iloc[keep].reset_index(drop=True)
    sample.channels["channel_number"] = range(1, len(keep) + 1)
    sample.pnn_labels = [sample.pnn_labels[i] for i in keep]
    sample.pns_labels = [sample.pns_labels[i] for i in keep]


def keywords(sample):
    return dict(sample.get_metadata())


# Keywords that describe the file's own layout; write_fcs sets these itself.
_LAYOUT_KEYWORDS = re.compile(
    r"^\$?(BEGIN|END)(ANALYSIS|DATA|STEXT)$|^\$?(BYTEORD|DATATYPE|MODE|NEXTDATA|PAR|TOT)$"
    r"|^\$?P\d+[A-Z]+$|^\$?(SPILL|SPILLOVER|COMP)$|^\$?FIL$", re.IGNORECASE)
_BLOCK = 100_000


def write_fcs(target, events, pnn, pns=None, keywords=None, ranges=None, keep_spillover=False):
    """Write an FCS 3.1 file of 32-bit float events.

    target: a path or a binary file handle. keywords: extra TEXT keywords
    (for example the source file's $DATE and $CYT); layout keywords in it are
    ignored, and so is any spillover matrix unless keep_spillover (it describes
    the source file's detectors). ranges: optional $PnR per channel (default 262144).
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
        spill = keep_spillover and key.lstrip("$") in ("SPILL", "SPILLOVER")
        if (spill or not _LAYOUT_KEYWORDS.match(key)) and str(value) != "":
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


# ----- reading odd and broken files ------------------------------------------
#
# FlowKit reads well-formed files. Before handing a file over, read_fcs checks
# its header and keywords (cheap: the events aren't read), so a broken file is
# refused with a reason a person can act on, and the quirks real instruments
# produce are worked around. tests/data/fcs_corpus has one file per quirk.


class UnreadableFCS(ValueError):
    """An FCS file Sidescatter can't read. The message says why, in plain words."""

    def __init__(self, path, reason):
        self.path, self.reason = Path(path), reason
        super().__init__(f"{Path(path).name}: {reason}")


_RESOURCE_FORK = b"\x00\x05\x16\x07"


def read_fcs(path, sample_id=None):
    path = Path(path)
    info = inspect_fcs(path)
    if info["fallback"]:
        sample = _decode(path, info, sample_id or path.name)
    else:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # quirks are reported in load_warnings instead
                sample = fk.Sample(str(path), sample_id=sample_id or path.name, ignore_offset_error=True,
                                   ignore_offset_discrepancy=info["offsets_only_in_text"], subsample=20000)
        except Exception:  # noqa: BLE001 - anything FlowKit trips on, read it ourselves
            sample = _decode(path, info, sample_id or path.name)
    sample.load_warnings = info["warnings"]
    return sample


def inspect_fcs(path):
    """Check an FCS file's header and keywords without reading its events.

    Raises UnreadableFCS for a file that can't be read. Otherwise returns what
    reading it needs: the keywords, where the events are, warnings for the
    user, and whether FlowKit can read it as it is ("fallback" if not).
    """
    path = Path(path)
    size = path.stat().st_size
    if size == 0:
        raise UnreadableFCS(path, "the file is empty (0 bytes), probably an interrupted copy")
    with open(path, "rb") as fh:
        head = fh.read(58)
        if head.startswith(_RESOURCE_FORK) or (path.name.startswith("._") and not head.startswith(b"FCS")):
            raise UnreadableFCS(path, "not an FCS file: it's a hidden \"._\" file macOS makes when "
                                      "copying to a USB stick or network drive. The real file has the "
                                      "same name without \"._\".")
        if not re.match(rb"FCS\d\.\d", head):
            raise UnreadableFCS(path, "not an FCS file (it doesn't start with an FCS header)")
        if len(head) < 58:
            raise UnreadableFCS(path, "the file ends before its header does, probably an interrupted copy")
        try:
            offsets = [int(head[10 + 8 * i:18 + 8 * i].strip() or 0) for i in range(6)]
        except ValueError:
            raise UnreadableFCS(path, "the file's header is damaged") from None
        text_start, text_end = offsets[0], offsets[1]
        if text_end >= size:
            raise UnreadableFCS(path, f"the file ends {_size(text_end + 1 - size)} early, before its "
                                      "keywords do: probably an interrupted copy or download")
        fh.seek(text_start)
        text = parse_text(fh.read(text_end - text_start + 1))

    version = head[3:6].decode()
    for key in ("PAR", "TOT", "DATATYPE"):
        if key not in text:
            raise UnreadableFCS(path, f"the file is missing the ${key} keyword, so its events can't be read")
    try:
        par, tot = int(text["PAR"]), int(text["TOT"])
    except ValueError:
        raise UnreadableFCS(path, "the file's $PAR or $TOT keyword isn't a number") from None
    if tot == 0:
        raise UnreadableFCS(path, "the file contains no events")
    datatype = text["DATATYPE"].upper()
    if datatype == "A":
        raise UnreadableFCS(path, "the events are stored as ASCII text, an old format Sidescatter can't "
                                  "read yet")
    if datatype not in ("I", "F", "D"):
        raise UnreadableFCS(path, f"unknown event data type \"{datatype}\"")
    bits = []
    for i in range(1, par + 1):
        try:
            bits.append(int(text.get(f"P{i}B", "0")))
        except ValueError:
            bits.append(0)
    allowed = {"I": (8, 16, 32), "F": (32,), "D": (64,)}[datatype]
    odd = sorted({b for b in bits if b not in allowed})
    if odd:
        raise UnreadableFCS(path, f"the events are stored as {odd[0]}-bit values, which Sidescatter "
                                  "can't read yet")

    begin = int(text.get("BEGINDATA", "0") or 0) or offsets[2]
    end = int(text.get("ENDDATA", "0") or 0) or offsets[3]
    expected = tot * sum(bits) // 8
    if size - begin < expected:
        raise UnreadableFCS(path, f"the file ends {_size(expected - (size - begin))} early: probably an "
                                  "interrupted copy or download")

    warn, fallback = [], False
    names = [text.get(f"P{i}N", f"P{i}") for i in range(1, par + 1)]
    if end - begin + 1 != expected:
        if end - begin + 1 != expected + 1:  # one byte over is common, and FlowKit handles it
            fallback = True
        warn.append("The file's data offsets are slightly off (a known instrument quirk); "
                    f"Sidescatter read the {tot:,} events it says it has. Check them before trusting it.")
    if int(text.get("NEXTDATA", "0") or 0):
        fallback = True
        warn.append("The file holds more than one dataset; only the first is loaded.")
    if "MODE" not in text or any(f"P{i}R" not in text for i in range(1, par + 1)):
        fallback = True  # FCS 3.2 dropped $MODE; some writers leave out $PnR
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        fallback = True
        warn.append("Several channels have the same name ("
                    + ", ".join(dupes) + "); the later ones are shown as e.g. "
                    + f"\"{dupes[0]} (2)\".")
    spill = _spill_keyword(text)
    if spill is not None:
        spill_names = _spill_names(text[spill])
        missing = [n for n in spill_names if n not in names]
        if spill_names is None or missing:
            fallback = True
            text.pop(spill)
            warn.append("The spillover matrix names channels that aren't in the file ("
                        + ", ".join((missing or ["unreadable"])[:4])
                        + "), so it can't be used: the file opens uncompensated.")
    return {"version": version, "text": text, "begin": begin, "par": par, "tot": tot, "bits": bits,
            "datatype": datatype, "names": names, "warnings": warn, "fallback": fallback,
            "offsets_only_in_text": offsets[2] == 0 and begin != 0}


def parse_text(raw):
    """The TEXT segment's keywords, upper-case and without "$". The first byte is
    the delimiter; a doubled delimiter inside a value stands for itself."""
    try:
        s = raw.decode("utf-8")
    except UnicodeDecodeError:
        s = raw.decode("latin-1")  # older instruments write their local code page
    if not s:
        return {}
    d, out, buf, i = s[0], [], [], 1
    while i < len(s):
        c = s[i]
        if c == d:
            if i + 1 < len(s) and s[i + 1] == d and buf:  # escaped (keywords and values are never empty)
                buf.append(d)
                i += 2
                continue
            out.append("".join(buf))
            buf = []
        else:
            buf.append(c)
        i += 1
    if buf and "".join(buf).strip(" \x00\r\n"):
        out.append("".join(buf))
    return {k.strip().upper().lstrip("$"): v for k, v in zip(out[::2], out[1::2])}


def _spill_keyword(text):
    for key in ("SPILLOVER", "SPILL", "$SPILLOVER", "$SPILL", "spillover", "spill", "$spillover", "$spill"):
        if key in text:
            return key
    return None


def _spill_names(value):
    parts = [p.strip() for p in str(value).split(",")]
    try:
        n = int(parts[0])
    except (ValueError, IndexError):
        return None
    return parts[1:n + 1] if len(parts) >= 1 + n + n * n else None


def _decode(path, info, sample_id):
    """Read the events ourselves, for files FlowKit can't take as they are."""
    text, par, tot = info["text"], info["par"], info["tot"]
    order = text.get("BYTEORD", "1,2,3,4").replace(" ", "")
    endian = ">" if order.startswith(("4,3", "2,1", "8,7")) else "<"
    bits = info["bits"]
    if info["datatype"] == "I" and len(set(bits)) > 1:
        raise UnreadableFCS(path, "the events mix integer sizes in a way Sidescatter can't read yet")
    kind = {"I": "u", "F": "f", "D": "f"}[info["datatype"]]
    dtype = np.dtype(f"{endian}{kind}{bits[0] // 8}")
    events = np.fromfile(path, dtype=dtype, count=tot * par, offset=info["begin"]).reshape(tot, par)
    events = events.astype(np.float64)

    ranges = []
    for i in range(par):
        try:
            r = float(text.get(f"P{i + 1}R", "0"))
        except ValueError:
            r = 0.0
        if r <= 0:
            r = float(2 ** bits[i]) if kind == "u" else 262144.0
        ranges.append(r)
        if kind == "u" and 2 ** bits[i] > r:  # FCS: bits above the range are ignored
            events[:, i] = events[:, i].astype(np.uint64) & (int(2 ** np.ceil(np.log2(r))) - 1)
        decades, log0 = _pne(text.get(f"P{i + 1}E", "0,0"))
        if decades > 0:
            events[:, i] = 10 ** (decades * events[:, i] / r) * (log0 or 1.0)
        try:
            gain = float(text.get(f"P{i + 1}G", "1") or 1)
        except ValueError:
            gain = 1.0
        if gain not in (0.0, 1.0):
            events[:, i] /= gain

    names, seen = [], {}
    for n in info["names"]:
        seen[n] = seen.get(n, 0) + 1
        names.append(n if seen[n] == 1 else f"{n} ({seen[n]})")
    pns = [text.get(f"P{i}S", "") for i in range(1, par + 1)]
    keywords = {k: v for k, v in text.items() if not re.match(r"^(P\d+[A-Z]|NEXTDATA)", k)}
    return load_sample(events, sample_id, names, pns, keywords, ranges,
                       keep_spillover=_spill_keyword(text) is not None)


def _pne(value):
    try:
        a, b = (float(x) for x in str(value).split(",")[:2])
        return a, b
    except ValueError:
        return 0.0, 0.0


def _size(n):
    for unit in ("bytes", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:,.0f} {unit}" if unit == "bytes" else f"{n:.1f} {unit}"
        n /= 1024
