"""Loading FCS files and describing their channels."""

from dataclasses import dataclass
from pathlib import Path

import flowkit as fk

# A sample with at least this many fluorescence detectors is treated as
# full-spectrum data (Cytek Aurora has 38-64, Sony ID7000 up to 184).
SPECTRAL_MIN_DETECTORS = 20


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
    if name.startswith(("FSC", "SSC")) or "SCATTER" in name:
        return "scatter"
    return "fluorescence"


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
    return len(detector_channels(sample)) >= SPECTRAL_MIN_DETECTORS


def raw_events(sample, channel_list=None):
    """Raw (uncompensated, untransformed) events, optionally restricted to channels."""
    events = sample.get_events(source="raw")
    if channel_list is None:
        return events
    return events[:, [c.index for c in channel_list]]


def keywords(sample):
    return dict(sample.get_metadata())
