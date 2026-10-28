"""Spectral unmixing: reference signatures, unmixing, and quality metrics.

Conventions used throughout:
  Y  raw detector events, shape (n_events, n_detectors)
  M  reference signature matrix, shape (n_fluors, n_detectors), each row
     normalised so its peak detector is 1
  A  unmixed abundances, shape (n_events, n_fluors), so that Y ~= A @ M
"""

from dataclasses import dataclass, field

import numpy as np

# Robust SD from the 15.87th-84.13th percentile span, as used for spread metrics.
_RSD_LOW, _RSD_HIGH = 15.87, 84.13


def robust_sd(x, axis=0):
    lo, hi = np.percentile(x, [_RSD_LOW, _RSD_HIGH], axis=axis)
    return (hi - lo) / 2.0


def normalise(signature):
    signature = np.clip(np.asarray(signature, dtype=np.float64), 0.0, None)
    peak = signature.max()
    if peak <= 0:
        raise ValueError("signature has no positive signal")
    return signature / peak


def extract_signature(positive, negative):
    """Signature = median(positive) - median(negative), clipped at 0, peak-normalised.

    positive, negative: (n, n_detectors) raw detector events.
    """
    return normalise(np.median(positive, axis=0) - np.median(negative, axis=0))


def peak_detector(events, negative=None):
    """Index of the detector where the control is brightest above background."""
    background = np.median(negative, axis=0) if negative is not None else 0.0
    return int(np.argmax(np.percentile(events, 99, axis=0) - background))


def auto_gate_control(events, negative=None, positive_fraction=0.1, negative_fraction=0.3):
    """Pick positive (and, if no negative is given, internal negative) events.

    Mirrors the usual manual approach on the peak detector. With a separate
    unstained control, positives are the events brighter than 99.5% of the
    unstained ones. Without one (a control containing both populations, as
    with compensation beads or stained cells), the brightest fraction is
    positive and the dimmest fraction is negative.
    Returns (positive_mask, negative_mask_or_None, peak_index).
    """
    peak = peak_detector(events, negative)
    values = events[:, peak]
    if negative is not None:
        pos = values > np.percentile(negative[:, peak], 99.5)
        if pos.sum() >= 20:
            return pos, None, peak
    pos = values >= np.quantile(values, 1.0 - positive_fraction)
    neg = None
    if negative is None:
        neg = values <= np.quantile(values, negative_fraction)
    return pos, neg, peak


def autofluorescence_signatures(unstained, n_signatures=1, n_iter=50, seed=0):
    """Autofluorescence signature(s) from unstained cells.

    One signature is the median spectrum. Several are found by k-means on
    peak-normalised spectra of the brighter half of events, which separates
    cell types with different autofluorescence (e.g. lymphocytes vs myeloid).
    """
    if n_signatures == 1:
        return normalise(np.median(unstained, axis=0))[None, :]

    bright = unstained[unstained.sum(axis=1) >= np.median(unstained.sum(axis=1))]
    shapes = np.clip(bright, 0, None)
    shapes = shapes / np.maximum(shapes.max(axis=1, keepdims=True), 1e-12)
    rng = np.random.default_rng(seed)
    centres = shapes[rng.choice(len(shapes), n_signatures, replace=False)]
    for _ in range(n_iter):
        dist = ((shapes[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
        labels = dist.argmin(axis=1)
        new = np.array(
            [shapes[labels == k].mean(axis=0) if np.any(labels == k) else centres[k]
             for k in range(n_signatures)]
        )
        if np.allclose(new, centres):
            break
        centres = new
    return np.array([normalise(c) for c in centres])


def unmix_ols(Y, M):
    """Ordinary least squares: A = Y @ pinv(M)."""
    return np.asarray(Y, dtype=np.float64) @ np.linalg.pinv(M)


def unmix_wls(Y, M, background_var=None, chunk=50_000):
    """Weighted least squares with per-event, per-detector weights.

    Weight = 1 / (max(signal, 0) + background variance), i.e. Poisson-like
    photon noise plus detector background, so bright detectors don't dominate.
    """
    Y = np.asarray(Y, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)
    if background_var is None:
        background_var = np.ones(M.shape[1])
    background_var = np.maximum(np.asarray(background_var, dtype=np.float64), 1e-9)

    out = np.empty((Y.shape[0], M.shape[0]))
    for start in range(0, Y.shape[0], chunk):
        y = Y[start:start + chunk]
        w = 1.0 / (np.clip(y, 0, None) + background_var)       # (n, d)
        mw = M[None, :, :] * w[:, None, :]                     # (n, k, d)
        lhs = mw @ M.T                                          # (n, k, k)
        rhs = np.einsum("nkd,nd->nk", mw, y)                    # (n, k)
        out[start:start + chunk] = np.linalg.solve(lhs, rhs[..., None])[..., 0]
    return out


def similarity_matrix(M):
    """Cosine similarity between every pair of signatures (1 = identical)."""
    unit = M / np.linalg.norm(M, axis=1, keepdims=True)
    return unit @ unit.T


def complexity_index(M):
    """Condition number of the signature matrix; higher means harder to unmix."""
    return float(np.linalg.cond(M))


def residuals(Y, A, M):
    """Per-event, per-detector residual Y - A @ M."""
    return np.asarray(Y, dtype=np.float64) - A @ M


def apply_fine_tune(A, spill):
    """Correct unmixed data with a fluor x fluor spillover matrix (1 on the diagonal).

    spill[i, j] is the fraction of fluor i that still appears in fluor j.
    """
    return A @ np.linalg.inv(spill)


def spillover_spread_matrix(controls_unmixed):
    """Spillover spread matrix from unmixed single-stain controls.

    controls_unmixed: list of (fluor_index, positive A, negative A).
    SSM[i, j] = (rSD_pos_j - rSD_neg_j) / sqrt(median_pos_i - median_neg_i)
    """
    k = controls_unmixed[0][1].shape[1]
    ssm = np.full((k, k), np.nan)
    for i, pos, neg in controls_unmixed:
        delta = np.median(pos[:, i]) - np.median(neg[:, i])
        if delta <= 0:
            continue
        ssm[i] = (robust_sd(pos) - robust_sd(neg)) / np.sqrt(delta)
        ssm[i, i] = np.nan
    return np.clip(ssm, 0, None)


@dataclass
class Reference:
    """One component of the unmixing model."""

    name: str
    signature: np.ndarray
    is_autofluorescence: bool = False
    source: str = ""  # file the signature came from
    marker: str = ""  # antigen or target, written as the unmixed channel's $PnS


@dataclass
class Unmixer:
    detectors: list
    references: list = field(default_factory=list)
    method: str = "ols"  # "ols" or "wls"
    background_var: np.ndarray = None
    fine_tune: np.ndarray = None
    # "spectroflo": unmixed channels are named like SpectroFlo's ("BV421-A",
    # autofluorescence "AF-A") with the marker as the description. "name":
    # channels are the bare reference names, as in workspaces saved before.
    channel_style: str = "spectroflo"

    @property
    def names(self):
        return [r.name for r in self.references]

    def channel_labels(self):
        """(PnN, PnS) of each unmixed channel, one per reference, never repeated."""
        n_af = sum(r.is_autofluorescence for r in self.references)
        labels, seen, af = [], set(), 0
        for r in self.references:
            if self.channel_style != "spectroflo":
                pnn, pns = r.name, r.marker
            elif r.is_autofluorescence:
                af += 1
                pnn, pns = ("AF-A" if n_af == 1 else f"AF{af}-A"), r.marker
            else:
                pnn = r.name if r.name.upper().endswith("-A") else f"{r.name}-A"
                pns = r.marker
            base, k = pnn, 2
            while pnn in seen:
                pnn, k = f"{base} {k}", k + 1
            seen.add(pnn)
            labels.append((pnn, pns))
        return labels

    @property
    def matrix(self):
        return np.vstack([r.signature for r in self.references])

    def unmix(self, Y):
        M = self.matrix
        if self.method == "wls":
            A = unmix_wls(Y, M, self.background_var)
        elif self.method == "ols":
            A = unmix_ols(Y, M)
        else:
            raise ValueError(f"unknown unmixing method {self.method!r}")
        if self.fine_tune is not None:
            A = apply_fine_tune(A, self.fine_tune)
        return A

    def to_dict(self):
        return {
            "detectors": list(self.detectors),
            "method": self.method,
            "references": [
                {"name": r.name, "signature": r.signature.tolist(),
                 "is_autofluorescence": r.is_autofluorescence, "source": r.source,
                 "marker": r.marker}
                for r in self.references
            ],
            "background_var": None if self.background_var is None
            else np.asarray(self.background_var).tolist(),
            "fine_tune": None if self.fine_tune is None else self.fine_tune.tolist(),
            "channel_style": self.channel_style,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            detectors=d["detectors"],
            method=d["method"],
            references=[
                Reference(r["name"], np.asarray(r["signature"]),
                          r.get("is_autofluorescence", False), r.get("source", ""),
                          r.get("marker", ""))
                for r in d["references"]
            ],
            background_var=None if d.get("background_var") is None
            else np.asarray(d["background_var"]),
            fine_tune=None if d.get("fine_tune") is None else np.asarray(d["fine_tune"]),
            channel_style=d.get("channel_style", "name"),
        )
