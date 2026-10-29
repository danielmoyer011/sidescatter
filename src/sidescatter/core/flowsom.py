"""FlowSOM: a self-organising map, then consensus metaclustering of its nodes.

Follows the R FlowSOM package (Van Gassen et al., Cytometry A 2015) and its
Python port by the same group (github.com/saeyslab/FlowSOM_Python, GPL-3):
a 10×10 SOM trained for rlen passes with the learning rate falling from 0.05
to 0.01 and the neighbourhood shrinking from the 67th percentile of grid
distances to the winning node alone, then ConsensusClusterPlus-style
hierarchical metaclustering of the node codes. Written here rather than
imported because the port pulls in scanpy, anndata and matplotlib.
"""

from dataclasses import dataclass

import numba
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


@numba.njit(cache=True)
def _train(data, codes, nhbrdist, alpha_start, alpha_end, radius_start, radius_end, rlen, order):
    n, d = data.shape
    ncodes = codes.shape[0]
    niter = rlen * n
    threshold = radius_start
    step = (radius_start - radius_end) / niter
    dists = np.empty(ncodes)
    for k in range(niter):
        i = order[k]
        nearest = 0
        for c in range(ncodes):
            s = 0.0
            for j in range(d):
                t = data[i, j] - codes[c, j]
                s += t * t
            dists[c] = s
            if s < dists[nearest]:
                nearest = c
        if threshold < 1.0:
            threshold = 0.5
        alpha = alpha_start - (alpha_start - alpha_end) * k / niter
        for c in range(ncodes):
            if nhbrdist[c, nearest] > threshold:
                continue
            for j in range(d):
                codes[c, j] += (data[i, j] - codes[c, j]) * alpha
        threshold -= step
    return codes


@numba.njit(cache=True)
def _nearest(data, codes):
    n, d = data.shape
    out = np.empty(n, dtype=np.int32)
    for i in range(n):
        best, best_d = 0, np.inf
        for c in range(codes.shape[0]):
            s = 0.0
            for j in range(d):
                t = data[i, j] - codes[c, j]
                s += t * t
            if s < best_d:
                best, best_d = c, s
        out[i] = best
    return out


def grid_distances(xdim, ydim):
    """Chebyshev distance between SOM nodes on the grid, as R FlowSOM uses."""
    grid = np.array([(x, y) for y in range(ydim) for x in range(xdim)], dtype=float)
    return np.abs(grid[:, None, :] - grid[None, :, :]).max(axis=2)


def train_som(data, xdim=10, ydim=10, rlen=10, alpha=(0.05, 0.01), seed=42):
    """Train a SOM on data (events × channels); returns the node codes."""
    data = np.ascontiguousarray(data, dtype=np.float64)
    rng = np.random.default_rng(seed)
    ncodes = xdim * ydim
    if len(data) < ncodes:
        raise ValueError(f"FlowSOM needs at least {ncodes} events for a {xdim}×{ydim} grid; "
                         f"this population has {len(data)}.")
    codes = data[rng.choice(len(data), ncodes, replace=False)].copy()
    nhbrdist = grid_distances(xdim, ydim)
    order = rng.integers(0, len(data), size=rlen * len(data))
    return _train(data, codes, nhbrdist, alpha[0], alpha[1], float(np.quantile(nhbrdist, 0.67)),
                  0.0, rlen, order)


def map_to_nodes(data, codes):
    """0-based SOM node of each event."""
    return _nearest(np.ascontiguousarray(data, dtype=np.float64), np.ascontiguousarray(codes))


def metacluster(codes, k, reps=100, fraction=0.9, seed=42):
    """Consensus hierarchical clustering of SOM codes into k metaclusters (1..k).

    As ConsensusClusterPlus with FlowSOM's settings: average-linkage
    hierarchical clustering of resampled codes, then average linkage on
    1 − consensus. Metaclusters are numbered in the order they first appear
    along the SOM grid, so the numbering is stable for a given seed.
    """
    n = len(codes)
    if not 2 <= k <= n:
        raise ValueError(f"The number of metaclusters must be between 2 and {n}.")
    rng = np.random.default_rng(seed)
    together = np.zeros((n, n))
    sampled = np.zeros((n, n))
    m = max(k, int(round(n * fraction)))
    for _ in range(reps):
        idx = np.sort(rng.choice(n, m, replace=False))
        labels = fcluster(linkage(codes[idx], "average"), k, criterion="maxclust")
        same = labels[:, None] == labels[None, :]
        together[np.ix_(idx, idx)] += same
        sampled[np.ix_(idx, idx)] += 1
    consensus = together / np.maximum(sampled, 1)
    np.fill_diagonal(consensus, 1.0)
    distance = squareform(1.0 - consensus, checks=False)
    labels = fcluster(linkage(distance, "average"), k, criterion="maxclust")
    # Renumber 1..k in order of first appearance along the SOM grid.
    order = {lab: i + 1 for i, lab in enumerate(dict.fromkeys(labels))}
    return np.array([order[lab] for lab in labels], dtype=np.int32)


@dataclass
class FlowSOM:
    """A trained FlowSOM model: node codes and each node's metacluster."""

    codes: np.ndarray
    node_meta: np.ndarray   # metacluster (1..k) of each node
    xdim: int
    ydim: int

    @classmethod
    def fit(cls, data, xdim=10, ydim=10, n_meta=10, rlen=10, seed=42):
        codes = train_som(data, xdim, ydim, rlen, seed=seed)
        return cls(codes, metacluster(codes, n_meta, seed=seed), xdim, ydim)

    def predict(self, data):
        """(cluster, metacluster) of each event, both numbered from 1."""
        nodes = map_to_nodes(data, self.codes)
        return nodes + 1, self.node_meta[nodes]
