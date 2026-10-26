"""Loading FCS files and describing their channels."""

from dataclasses import dataclass
from pathlib import Path

import re

import flowkit as fk

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


def load_sample(path_or_data, sample_id=None, channel_labels=None):
    """Load an FCS file (or an events array with channel_labels) as a FlowKit Sample.

    Full-resolution events are kept; subsampling is only used for plotting.
    """
    if isinstance(path_or_data, (str, Path)):
        path = Path(path_or_data)
        return fk.Sample(
            str(path),
            sample_id=sample_id or path.name,
            ignore_offset_error=True,
            subsample=20000,
        )
    return fk.Sample(
        path_or_data, sample_id=sample_id, channel_labels=channel_labels, subsample=20000
    )


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
