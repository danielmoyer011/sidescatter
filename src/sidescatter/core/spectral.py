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


def _kmeans(X, k, n_iter=30, seed=0):
    """Plain k-means with k-means++ starts. Returns (centres, labels)."""
    rng = np.random.default_rng(seed)
    k = min(k, len(X))
    centres = [X[rng.integers(len(X))]]
    dist = ((X - centres[0]) ** 2).sum(axis=1)
    for _ in range(1, k):
        if dist.sum() <= 0:
            break
        centres.append(X[rng.choice(len(X), p=dist / dist.sum())])
        dist = np.minimum(dist, ((X - centres[-1]) ** 2).sum(axis=1))
    centres = np.array(centres)
    sq = (X ** 2).sum(axis=1)[:, None]
    for _ in range(n_iter):
        labels = (sq - 2 * X @ centres.T + (centres ** 2).sum(axis=1)[None, :]).argmin(axis=1)
        new = np.array([X[labels == j].mean(axis=0) if np.any(labels == j) else centres[j]
                        for j in range(len(centres))])
        if np.allclose(new, centres):
            break
        centres = new
    labels = (sq - 2 * X @ centres.T + (centres ** 2).sum(axis=1)[None, :]).argmin(axis=1)
    return centres, labels


def cosine(a, b):
    """Cosine similarity of every row of a with every row of b."""
    a = np.atleast_2d(a) / np.maximum(np.linalg.norm(np.atleast_2d(a), axis=1, keepdims=True), 1e-12)
    b = np.atleast_2d(b) / np.maximum(np.linalg.norm(np.atleast_2d(b), axis=1, keepdims=True), 1e-12)
    return a @ b.T


def dedupe_spectra(spectra, threshold=0.99):
    """Indices of the spectra to keep: each one is dropped when its cosine similarity
    to an earlier kept one is above threshold, so put the most important first."""
    keep = []
    for i, s in enumerate(spectra):
        if not keep or cosine(s, spectra[keep]).max() <= threshold:
            keep.append(i)
    return keep


def af_library(unstained, max_spectra=30, fluor_signatures=None, duplicate_threshold=0.99,
               max_events=20_000, seed=0):
    """A library of autofluorescence spectra from unstained cells, for per-cell matching.

    Events are clustered by spectral shape (peak-normalised) with k-means into
    up to max_spectra groups; each group's median spectrum is one library
    entry. The first entry is the median of all events. Entries closer than
    duplicate_threshold (cosine similarity) to an earlier one are dropped, as
    are entries that look like one of fluor_signatures (contamination from a
    stained tube). Returns (library, n_events): (m, d) peak-normalised spectra,
    largest group first, and how many clustered events each one stands for.
    """
    Y = np.asarray(unstained, dtype=np.float64)
    if len(Y) > max_events:
        Y = Y[np.sort(np.random.default_rng(seed).choice(len(Y), max_events, replace=False))]
    # The dimmest events are mostly detector noise and have no usable shape.
    total = Y.sum(axis=1)
    Y = Y[total >= np.quantile(total, 0.1)]
    spectra, sizes = [normalise(np.median(Y, axis=0))], [len(Y)]
    if max_spectra > 1 and len(Y) >= 2 * max_spectra:
        shapes = np.clip(Y, 0, None)
        shapes = shapes / np.maximum(shapes.max(axis=1, keepdims=True), 1e-12)
        _, labels = _kmeans(shapes, max_spectra - 1, seed=seed)
        groups = sorted((np.flatnonzero(labels == j) for j in np.unique(labels)), key=len, reverse=True)
        for g in groups:
            if len(g) >= 5:
                try:
                    spectra.append(normalise(np.median(Y[g], axis=0)))
                    sizes.append(len(g))
                except ValueError:
                    continue
    spectra, sizes = np.array(spectra), np.array(sizes)
    if fluor_signatures is not None and len(fluor_signatures):
        clean = cosine(spectra, fluor_signatures).max(axis=1) < 0.99
        clean[0] = True
        spectra, sizes = spectra[clean], sizes[clean]
    keep = dedupe_spectra(spectra, duplicate_threshold)
    return spectra[keep], sizes[keep]


def unmix_af_per_cell(Y, F, library, method="ols", background_var=None, allowed=None):
    """Unmix with one autofluorescence spectrum per cell, picked from a library.

    For every event and every library spectrum L_j, the model is
    y = a @ F + c * L_j; the j that leaves the smallest least-squares residual
    wins (AutoSpectral's approach). That is found in one pass: after removing
    the part of L_j the fluors can explain (L_j_perp), the residual drops by
    (y . L_j_perp)^2 / |L_j_perp|^2. Only positive AF amounts count.

    F: (k, d) fluor signatures; library: (m, d). allowed: optional (m,) bool,
    the library entries this sample may use.
    Returns (A, af, index): fluor abundances (n, k), fitted AF amount (n,), and
    the 1-based library entry used (0 when no AF fitted).
    """
    Y = np.asarray(Y, dtype=np.float64)
    F = np.asarray(F, dtype=np.float64)
    L = np.asarray(library, dtype=np.float64)
    P = np.linalg.pinv(F)                      # (d, k)
    L_perp = L - (L @ P) @ F                   # parts of each AF spectrum the fluors can't fit
    norm2 = (L_perp ** 2).sum(axis=1)
    usable = norm2 > 1e-8 * (L ** 2).sum(axis=1)
    if allowed is not None:
        usable &= np.asarray(allowed, bool)
    proj = Y @ L_perp.T                        # (n, m), equal to y_perp . L_perp
    gain = np.where((proj > 0) & usable, proj ** 2 / np.where(usable, norm2, 1.0), 0.0)
    best = gain.argmax(axis=1)
    rows = np.arange(len(Y))
    fitted = gain[rows, best] > 0
    af = np.where(fitted, proj[rows, best] / np.where(usable, norm2, 1.0)[best], 0.0)
    Y_fluor = Y - af[:, None] * L[best]
    if method == "wls":
        # Photon noise comes from all the light, AF included, so weight by the raw event.
        A = unmix_wls(Y_fluor, F, background_var, weights_from=Y)
        # Refit the AF amount on what the weighted fit leaves, so it stays consistent.
        R = Y - A @ F
        L_best = L[best]
        af = np.where(fitted, np.clip((R * L_best).sum(axis=1)
                                      / np.maximum((L_best ** 2).sum(axis=1), 1e-12), 0, None), 0.0)
    else:
        A = Y_fluor @ P
    return A, af, np.where(fitted, best + 1, 0)


def unmix_ols(Y, M):
    """Ordinary least squares: A = Y @ pinv(M)."""
    return np.asarray(Y, dtype=np.float64) @ np.linalg.pinv(M)


def unmix_wls(Y, M, background_var=None, chunk=50_000, weights_from=None):
    """Weighted least squares with per-event, per-detector weights.

    Weight = 1 / (max(signal, 0) + background variance), i.e. Poisson-like
    photon noise plus detector background, so bright detectors don't dominate.
    weights_from: events to take the signal from, when not Y itself.
    """
    Y = np.asarray(Y, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)
    if background_var is None:
        background_var = np.ones(M.shape[1])
    background_var = np.maximum(np.asarray(background_var, dtype=np.float64), 1e-9)

    k = M.shape[0]
    # lhs[n] = M diag(w[n]) M.T for every event at once, as one matrix product:
    # w (n, d) @ G (d, k*k), where G[:, i*k + j] = M[i] * M[j].
    G = (M[:, None, :] * M[None, :, :]).reshape(k * k, -1).T
    out = np.empty((Y.shape[0], k))
    for start in range(0, Y.shape[0], chunk):
        y = Y[start:start + chunk]
        signal = y if weights_from is None else np.asarray(weights_from[start:start + chunk], np.float64)
        w = 1.0 / (np.clip(signal, 0, None) + background_var)  # (n, d)
        lhs = (w @ G).reshape(-1, k, k)                         # (n, k, k)
        rhs = (w * y) @ M.T                                     # (n, k)
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
    # Per-cell autofluorescence: (m, d) library of AF spectra, one of which is
    # fitted to each cell. When set, the single autofluorescence reference is
    # the AF-A channel and an extra AF Index channel says which entry matched.
    af_library: np.ndarray = None
    af_sources: list = None  # file each library entry came from
    # Per-cell fluor variation (fluor_variants): per reference, its (v, d)
    # variant spectra (empty for autofluorescence), and the abundance above
    # which a cell's variant is chosen. variant_channels adds one "<fluor>
    # Variant" channel per fluor saying which variant each cell used (0: none).
    variants: list = None
    variant_thresholds: np.ndarray = None
    variant_channels: bool = False

    @property
    def names(self):
        return [r.name for r in self.references]

    @property
    def per_cell_af(self):
        return self.af_library is not None

    @property
    def per_cell_variants(self):
        return self.variants is not None and any(len(v) for v in self.variants)

    def _variant_refs(self):
        """Indices of the references that can have variants (all but per-cell AF's)."""
        return [i for i, r in enumerate(self.references)
                if not (self.per_cell_af and r.is_autofluorescence)]

    def _variant_names(self):
        if not (self.per_cell_variants and self.variant_channels):
            return []
        return [f"{self.references[i].name} Variant" for i in self._variant_refs()
                if not self.references[i].is_autofluorescence]

    @property
    def output_names(self):
        """Names of the columns unmix() returns."""
        return self.names + (["AF Index"] if self.per_cell_af else []) + self._variant_names()

    def af_allowed(self, af_source=None):
        """Library entries a sample may use: all of them, or those from one unstained file."""
        if not self.per_cell_af or not af_source or not self.af_sources:
            return None
        allowed = np.array([s == af_source for s in self.af_sources])
        return allowed if allowed.any() else None

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
        if self.per_cell_af:
            labels.append(("AF Index", "AF library entry"))
        labels += [(n, "fluor variant") for n in self._variant_names()]
        return labels

    @property
    def matrix(self):
        return np.vstack([r.signature for r in self.references])

    def unmix(self, Y, af_source=None):
        """Abundances, one column per output_names entry.

        af_source: with per-cell AF, the unstained file whose library entries
        this sample may use (None: the whole shared library).
        """
        if self.method not in ("ols", "wls"):
            raise ValueError(f"unknown unmixing method {self.method!r}")
        if self.per_cell_af:
            fluor = [i for i, r in enumerate(self.references) if not r.is_autofluorescence]
            af_col = next(i for i, r in enumerate(self.references) if r.is_autofluorescence)
            A_f, af, index = unmix_af_per_cell(Y, self.matrix[fluor], self.af_library, self.method,
                                               self.background_var, self.af_allowed(af_source))
            A = np.empty((len(A_f), len(self.references) + 1))
            A[:, fluor] = A_f
            A[:, af_col] = af
            A[:, -1] = index
        else:
            M = self.matrix
            A = unmix_wls(Y, M, self.background_var) if self.method == "wls" else unmix_ols(Y, M)
        if self.per_cell_variants:
            A = self._apply_variants(Y, A)
        if self.fine_tune is not None:
            n = len(self.references)
            A[:, :n] = apply_fine_tune(A[:, :n], self.fine_tune)
        return A

    def _apply_variants(self, Y, A):
        """Per-cell fluor variants on top of the plain abundances A (see fluor_variants.refine)."""
        from . import fluor_variants

        Y = np.asarray(Y, dtype=np.float64)
        cols = self._variant_refs()
        M = self.matrix[cols]
        fixed = None
        if self.per_cell_af:
            af_col = next(i for i, r in enumerate(self.references) if r.is_autofluorescence)
            index = A[:, len(self.references)].astype(int)
            spectra = np.vstack([np.zeros(self.af_library.shape[1]), self.af_library])[index]
            fixed = A[:, [af_col]] * spectra

        def unmix_fn(R):
            if self.method == "wls":
                return unmix_wls(R, M, self.background_var, weights_from=Y)
            return unmix_ols(R, M)

        sets = [np.asarray(self.variants[i]) for i in cols]
        thresholds = (np.zeros(len(cols)) if self.variant_thresholds is None
                      else np.asarray(self.variant_thresholds, dtype=float)[cols])
        A, chosen = fluor_variants.refine(Y, A, M, cols, sets, thresholds, unmix_fn, fixed)
        if self.variant_channels:
            keep = [k for k, i in enumerate(cols) if not self.references[i].is_autofluorescence]
            A = np.hstack([A, chosen[:, keep].astype(np.float64)])
        return A

    def reconstruct(self, A):
        """The detector signal the model predicts from unmixed abundances A."""
        n = len(self.references)
        if not self.per_cell_af:
            return A[:, :n] @ self.matrix
        fluor = [i for i, r in enumerate(self.references) if not r.is_autofluorescence]
        af_col = next(i for i, r in enumerate(self.references) if r.is_autofluorescence)
        index = A[:, n].astype(int)
        spectra = np.vstack([np.zeros(self.af_library.shape[1]), self.af_library])[index]
        return A[:, fluor] @ self.matrix[fluor] + A[:, [af_col]] * spectra

    def residuals(self, Y, A=None):
        """Per-event, per-detector residual Y - model(A)."""
        A = self.unmix(Y) if A is None else A
        return np.asarray(Y, dtype=np.float64) - self.reconstruct(A)

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
            "af_library": None if self.af_library is None else np.asarray(self.af_library).tolist(),
            "af_sources": self.af_sources,
            "variants": None if self.variants is None
            else [np.asarray(v).tolist() for v in self.variants],
            "variant_thresholds": None if self.variant_thresholds is None
            else np.asarray(self.variant_thresholds).tolist(),
            "variant_channels": self.variant_channels,
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
            af_library=None if d.get("af_library") is None else np.asarray(d["af_library"]),
            af_sources=d.get("af_sources"),
            variants=None if d.get("variants") is None
            else [np.asarray(v, dtype=float).reshape(-1, len(d["detectors"])) for v in d["variants"]],
            variant_thresholds=None if d.get("variant_thresholds") is None
            else np.asarray(d["variant_thresholds"]),
            variant_channels=d.get("variant_channels", False),
        )
