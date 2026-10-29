"""Cleaner single-stain controls, following AutoSpectral (Burton et al. 2025).

A signature taken as median(positive) - median(whole negative) carries every
difference between the positive cells and the average unstained cell: bigger
cells have more autofluorescence, saturated events clip the peak, and doublets
add a second cell's background. Before a signature is calculated:

- events at a detector's full scale (saturated) and doublets (high FSC-A/FSC-H)
  are dropped;
- only the brightest positive events on the peak detector are kept (500 by
  default), since bright events carry the most reliable spectral shape;
- for cell controls with an unstained cells file, each positive event's
  background is the average of its nearest unstained cells in forward/side
  scatter ("scatter-matched negatives"), not the median of all unstained cells;
- optionally, positive events with unusually high autofluorescence for their
  fluor brightness are dropped (cells only; slow, so off by default).

Beads keep the plain median of their negative.
"""

from dataclasses import asdict, dataclass

import numpy as np

SATURATION = 0.999   # share of a detector's range that counts as saturated


@dataclass
class Settings:
    enabled: bool = True
    saturation: bool = True
    singlets: bool = True
    brightest: int = 500        # positive events kept per control (0: all)
    scatter_match: bool = True
    neighbours: int = 2

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        if not d:
            return cls(enabled=False)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def not_saturated(Y, top):
    """Events with no detector at (or within 0.1% of) full scale.

    Y: (n, d) detector events; top: (d,) or scalar range of each detector.
    """
    top = np.broadcast_to(np.asarray(top, dtype=float), (Y.shape[1],))
    return ~(Y >= SATURATION * top).any(axis=1)


def singlets(area, height, spread=4.0):
    """Events whose FSC-A/FSC-H ratio isn't far above the typical single cell's.

    Doublets pass the laser as one longer pulse: their area grows more than
    their height. Events above median + spread x robust SD (MAD) of the ratio
    are called doublets.
    """
    area, height = np.asarray(area, dtype=float), np.asarray(height, dtype=float)
    ok = (height > 0) & (area > 0)
    keep = np.ones(len(area), dtype=bool)
    if ok.sum() < 50:
        return keep
    ratio = np.full(len(area), np.nan)
    ratio[ok] = area[ok] / height[ok]
    med = np.median(ratio[ok])
    mad = 1.4826 * np.median(np.abs(ratio[ok] - med))
    if mad <= 0:
        return keep
    keep[ok] = ratio[ok] <= med + spread * mad
    return keep


def brightest(values, mask, n):
    """mask limited to its n events with the highest values (all of them when n <= 0)."""
    idx = np.flatnonzero(mask)
    if n <= 0 or len(idx) <= n:
        return mask
    keep = idx[np.argsort(values[idx])[-n:]]
    out = np.zeros_like(mask)
    out[keep] = True
    return out


def _scatter_scale(xy):
    """Scatter values on a log scale, so neighbours are judged on relative size."""
    return np.log10(np.clip(np.asarray(xy, dtype=float), 1.0, None))


def matched_background(pos_scatter, neg_scatter, neg_Y, k=2):
    """Each positive event's background: the mean of its k nearest negatives in scatter.

    pos_scatter (n, 2), neg_scatter (m, 2): forward/side scatter; neg_Y (m, d)
    the negatives' detector events. Distances use log scatter divided by the
    negatives' spread on each axis, so FSC and SSC count equally.
    """
    from scipy.spatial import cKDTree

    neg = _scatter_scale(neg_scatter)
    pos = _scatter_scale(pos_scatter)
    lo, hi = np.percentile(neg, [15.87, 84.13], axis=0)
    scale = np.where(hi - lo > 0, (hi - lo) / 2, 1.0)
    k = max(1, min(int(k), len(neg)))
    _, idx = cKDTree(neg / scale).query(pos / scale, k=k)
    idx = np.asarray(idx).reshape(len(pos), k)
    return neg_Y[idx].mean(axis=1)


def high_af(pos_Y, fluor_signature, af_signature, unstained_Y, quantile=99.0):
    """Positive events with more autofluorescence than almost all unstained cells.

    Each event is fitted as fluor + AF (two-component least squares); events
    whose AF amount is above the given percentile of the unstained cells' AF
    amount are flagged.
    """
    M = np.vstack([fluor_signature, af_signature])
    pinv = np.linalg.pinv(M @ M.T) @ M
    af_pos = pos_Y @ pinv[1]
    af_neg = unstained_Y @ pinv[1]
    return af_pos > np.percentile(af_neg, quantile)


def signature(pos_Y, background):
    """Peak-normalised median of the background-subtracted positive events.

    background: (n, d) per-event backgrounds, or (d,) one shared background.
    """
    from .spectral import normalise

    return normalise(np.median(np.asarray(pos_Y, dtype=float) - background, axis=0))
