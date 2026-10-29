"""Synthetic spectral data with mixed autofluorescence and a known answer.

A made-up 5-laser instrument (64 detectors), smooth fluor signatures, and
unstained cells drawn from several AF types with very different spectra (as in
tissue: lymphocytes, macrophages, granulocytes, debris), each with its own
brightness. Every event's true fluor abundances are known, so unmixing
methods can be scored directly.
"""

import numpy as np

LASERS = [("UV", 16), ("V", 16), ("B", 14), ("YG", 10), ("R", 8)]
DETECTORS = [f"{laser}{i}-A" for laser, n in LASERS for i in range(1, n + 1)]
D = len(DETECTORS)


def _bumps(*peaks, floor=0.01):
    """A spectrum from (centre, width, height) bumps over the detector index."""
    x = np.arange(D, dtype=float)
    s = sum(h * np.exp(-0.5 * ((x - c) / w) ** 2) for c, w, h in peaks) + floor
    return s / s.max()


def fluor_signatures(k=16, seed=0):
    """k distinct fluor signatures, each with a main peak and a cross-laser echo."""
    rng = np.random.default_rng(seed)
    centres = np.linspace(2, D - 3, k)
    return np.array([_bumps((c, rng.uniform(1.2, 2.5), 1.0),
                            ((c + 16 * rng.integers(1, 3)) % D, 2.0, rng.uniform(0.05, 0.3)))
                     for c in centres])


# (name, spectrum, share of cells, median brightness)
AF_TYPES = [
    ("lymphocyte", _bumps((4, 5, 1.0), (20, 6, 0.6), (34, 4, 0.2)), 0.45, 600.0),
    ("macrophage", _bumps((18, 7, 0.7), (35, 6, 1.0), (50, 5, 0.6), (58, 4, 0.3)), 0.25, 6000.0),
    ("granulocyte", _bumps((22, 4, 1.0), (36, 3, 0.9), (45, 6, 0.2)), 0.2, 2500.0),
    ("debris", _bumps((8, 20, 1.0), floor=0.2), 0.1, 1500.0),
]


def cells(n, abundances=None, F=None, seed=1, background=40.0):
    """n events: AF from a random AF type plus abundances @ F, with photon and detector noise.

    Returns (Y, af_type) with Y (n, D) and af_type (n,) the index into AF_TYPES.
    """
    rng = np.random.default_rng(seed)
    shares = np.array([t[2] for t in AF_TYPES])
    af_type = rng.choice(len(AF_TYPES), n, p=shares / shares.sum())
    spectra = np.array([t[1] for t in AF_TYPES])
    bright = np.array([t[3] for t in AF_TYPES])[af_type] * rng.lognormal(0, 0.35, n)
    # Each cell's AF shape wobbles a little around its type's.
    wobble = 1 + 0.03 * rng.standard_normal((n, D))
    mean = bright[:, None] * spectra[af_type] * wobble
    if abundances is not None:
        mean = mean + abundances @ F
    Y = rng.poisson(np.clip(mean, 0, None)).astype(float) + rng.normal(0, background, (n, D))
    return Y, af_type
