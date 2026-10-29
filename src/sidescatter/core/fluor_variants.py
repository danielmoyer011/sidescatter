"""Per-cell fluor variation, following AutoSpectral (Burton et al. 2025).

A fluor's spectrum isn't quite the same on every cell: tandems degrade
unevenly, and the local environment shifts emission. One signature per fluor
leaves that variation in the residual, where it spreads into other fluors'
channels. Here:

1. variants: each single-stain control's background-subtracted positive
   events are clustered with a self-organising map (10 x 10 by default), and
   node spectra close to the fluor's signature (cosine 0.98 or more) become
   that fluor's variants;
2. per cell, each fluor the cell clearly carries is given the variant that
   best explains what the plain unmixing left over (the residual), and the
   abundances are then corrected in one more unmixing step ("fast" in
   AutoSpectral's terms).

hotspots() turns the variants into an error-propagation matrix: how much
spread in one fluor each other fluor's variation is expected to cause.
"""

from dataclasses import asdict, dataclass

import numpy as np

from . import spectral

MIN_COSINE = 0.98
MAX_VARIANTS = 100


@dataclass
class Settings:
    enabled: bool = False          # use the variants when unmixing
    max_variants: int = MAX_VARIANTS
    min_cosine: float = MIN_COSINE
    channels: bool = False         # add a "<fluor> Variant" channel per fluor

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__})


_cache = {}


def cached_variants(key, spectra, signature, max_variants=MAX_VARIANTS, min_cosine=MIN_COSINE):
    """variants(), remembered for the same control events, so dragging another
    control's gate doesn't retrain every map."""
    spectra = np.ascontiguousarray(spectra, dtype=np.float64)
    full = (key, spectra.shape, hash(spectra.tobytes()), max_variants, min_cosine,
            hash(np.asarray(signature, dtype=np.float64).tobytes()))
    if full not in _cache:
        if len(_cache) > 256:
            _cache.clear()
        _cache[full] = variants(spectra, signature, max_variants, min_cosine)
    return _cache[full]


def variants(spectra, signature, max_variants=MAX_VARIANTS, min_cosine=MIN_COSINE, seed=0):
    """Variant spectra of one fluor, peak-normalised: (v, d), at most max_variants rows.

    spectra: (n, d) background-subtracted positive events of its control.
    """
    from .flowsom import train_som

    spectra = np.asarray(spectra, dtype=np.float64)
    signature = spectral.normalise(signature)
    peak = int(np.argmax(signature))
    bright = spectra[spectra[:, peak] > 0]
    side = int(np.sqrt(max(1, min(max_variants, len(bright) // 5))))
    if side < 2:
        return np.empty((0, len(signature)))
    shapes = bright / bright[:, [peak]]           # on the signature's scale: 1 at its peak
    codes = train_som(shapes, side, side, rlen=10, seed=seed)
    out = []
    for c in codes:
        c = np.clip(c, 0, None)
        if c.max() <= 0:
            continue
        c = c / c.max()
        if spectral.cosine(c[None], signature[None])[0, 0] >= min_cosine:
            out.append(c)
    if not out:
        return np.empty((0, len(signature)))
    out = np.array(out)
    keep = spectral.dedupe_spectra(out, 0.9999)
    return out[keep]


def positive_threshold(spread):
    """Abundance above which a cell counts as carrying a fluor: 3 x its spread on negatives."""
    return 3.0 * np.asarray(spread, dtype=np.float64)


def refine(Y, A, M, fluor_cols, variant_sets, thresholds, unmix_fn, fixed=None):
    """Per-cell variant choice and corrected abundances.

    Y (n, d) raw events; A (n, k) abundances from the plain model; M (k', d)
    the fluor signatures, row i belonging to column fluor_cols[i] of A;
    variant_sets[i] the (v_i, d) variants of fluor i (may be empty);
    thresholds[i] the abundance above which variants are tried; unmix_fn(R)
    unmixes a residual with the plain model's fluor matrix; fixed (n, d) the
    part of the model that isn't fluors (per-cell AF), or None.
    Returns (corrected A, chosen variant index per fluor (n, k'), 0 = none).
    """
    Y = np.asarray(Y, dtype=np.float64)
    n = len(Y)
    A = np.array(A, dtype=np.float64)
    a = A[:, fluor_cols]
    model = a @ M if fixed is None else a @ M + fixed
    R = Y - model
    chosen = np.zeros((n, len(fluor_cols)), dtype=np.int16)
    for i, V in enumerate(variant_sets):
        if len(V) == 0:
            continue
        rows = np.flatnonzero(a[:, i] > thresholds[i])
        if not len(rows):
            continue
        D = V - M[i]                                  # (v, d) change from the signature
        ai = a[rows, i][:, None]
        # ||R - a d||^2 = ||R||^2 - 2 a R.d + a^2 ||d||^2; keep the signature (0) unless a variant fits better.
        score = -2 * ai * (R[rows] @ D.T) + ai ** 2 * (D ** 2).sum(axis=1)[None, :]
        best = score.argmin(axis=1)
        better = score[np.arange(len(rows)), best] < 0
        rows, best = rows[better], best[better]
        R[rows] -= a[rows, i][:, None] * D[best]
        chosen[rows, i] = best + 1
    # One more unmixing step on what is left, with the variants in place.
    A[:, fluor_cols] = a + unmix_fn(R)
    return A, chosen


def hotspots(M, names, variant_sets):
    """Expected spread in each fluor (columns) from each fluor's variation (rows).

    Each variant's difference from its fluor's signature is unmixed with M;
    the RMS over variants of what lands in fluor j, per unit of fluor i, is
    entry (i, j). The diagonal is NaN.
    """
    M = np.asarray(M, dtype=np.float64)
    pinv = np.linalg.pinv(M)
    k = len(names)
    out = np.full((k, k), np.nan)
    for i, V in enumerate(variant_sets):
        if len(V) == 0:
            continue
        leak = (np.asarray(V) - M[i]) @ pinv        # (v, k)
        out[i] = np.sqrt((leak ** 2).mean(axis=0))
        out[i, i] = np.nan
    return out
