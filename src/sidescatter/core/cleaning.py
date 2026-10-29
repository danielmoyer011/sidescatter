"""Automated data cleaning: margin events, flow rate, and signal stability over time.

The signal check follows PeacoQC (Emmaneel et al., Cytometry A 2022; R package
saeyslab/PeacoQC, GPL-3): the run is cut into overlapping windows of events,
the density peaks of every channel are found in each window, and windows are
removed when their peaks stand apart from the rest (an isolation tree) or
drift too far from the median (MAD on a smoothed trace). Short good stretches
caught between removed ones are removed too. Each step is a port of the R code
so results can be compared with PeacoQC on the same file.

Two things differ from PeacoQC on purpose:

- Margins: PeacoQC always drops the single highest and lowest event of every
  channel. Here only events piled up on the edge (at least MARGIN_PILEUP
  events on exactly the same extreme value, or at the top of the detector's
  $PnR) count, so a clean file loses nothing.
- Flow rate: PeacoQC has no flow-rate check. Windows whose event rate jumps
  away from the running median (clogs, bubbles, pauses) are flagged on the same
  windows, with the same MAD setting.

Removed stretches are kept as ranges of event ranks in time order, so they
can be shown and edited on a Time axis and saved without storing per-event masks.
"""

import base64
import re
from dataclasses import asdict, dataclass, field

import numpy as np

MARGIN_PILEUP = 3
MIN_WINDOWS = 10  # fewer windows than this and the signal and flow-rate checks are skipped

PRESETS = {
    # Higher isolation tree limits make a split harder, so fewer windows are removed.
    "Lenient": {"mad": 8.0, "it_limit": 0.65},
    "Standard": {"mad": 6.0, "it_limit": 0.6},  # PeacoQC's defaults
    "Strict": {"mad": 4.0, "it_limit": 0.55},
}

_AF_NAME = re.compile(r"^AF\d{0,2}(-A)?$", re.IGNORECASE)


@dataclass
class Settings:
    preset: str = "Standard"
    mad: float = 6.0
    it_limit: float = 0.6
    consecutive_bins: int = 5
    events_per_bin: int = 0          # 0: chosen from the file size, as PeacoQC does
    remove_zeros: bool = False       # for mass cytometry
    peak_removal: float = 1 / 3      # peaks lower than this fraction of the highest are ignored
    min_bins_peakdetection: float = 10.0  # percent of windows that must share a peak count
    force_it: int = 150              # fewer windows than this: no isolation tree
    margins: bool = True
    flow_rate: bool = True
    signal: bool = True
    channels: list = None            # None: the default channels of each sample

    @classmethod
    def from_preset(cls, name, **kw):
        return cls(preset=name, **{**PRESETS[name], **kw})

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})

    def same_method(self, other):
        """Whether two settings would clean the same way (the preset label aside)."""
        a, b = self.to_dict(), other.to_dict()
        a.pop("preset"), b.pop("preset")
        return a == b


@dataclass
class Result:
    """The cleaning of one sample."""

    settings: dict
    n_events: int
    time_channel: str = None         # None: event order stood in for time
    channels: list = field(default_factory=list)
    margin: np.ndarray = None        # True for margin events
    margin_counts: dict = field(default_factory=dict)  # channel -> events at its edges
    stretches: list = field(default_factory=list)       # [start, end) ranks, as shown and edited
    auto_stretches: list = field(default_factory=list)  # as the algorithm found them
    edited: bool = False
    removed_by: dict = field(default_factory=dict)       # step -> % of events
    bins: dict = field(default_factory=dict)             # window data for the diagnostic plots
    warnings: list = field(default_factory=list)
    trend: str = "No increasing or decreasing effect"

    def removed(self, time):
        """Boolean mask of removed events (margins plus stretches)."""
        order = time_order(time, self.n_events)
        in_stretch = np.zeros(self.n_events, dtype=bool)
        for s, e in self.stretches:
            in_stretch[int(s):int(e)] = True
        out = np.empty(self.n_events, dtype=bool)
        out[order] = in_stretch
        if self.margin is not None:
            out |= self.margin
        return out

    def percent_removed(self, time):
        return 100.0 * self.removed(time).sum() / max(self.n_events, 1)

    def to_dict(self):
        d = asdict(self)
        d["margin"] = None if self.margin is None else _pack(self.margin)
        return d

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        n = d["n_events"]
        d["margin"] = None if d.get("margin") is None else _unpack(d["margin"], n)
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


def _pack(mask):
    return base64.b64encode(np.packbits(np.asarray(mask, dtype=bool)).tobytes()).decode("ascii")


def _unpack(text, n):
    return np.unpackbits(np.frombuffer(base64.b64decode(text), dtype=np.uint8))[:n].astype(bool)


def time_order(time, n):
    """Event indices in acquisition order (stable, so ties keep file order)."""
    if time is None:
        return np.arange(n)
    return np.argsort(np.asarray(time), kind="stable")


def is_af_channel(pnn, pns=""):
    return bool(_AF_NAME.match(pnn)) or "autofluor" in (pns or "").lower()


def default_channels(chans):
    """All fluorescence channels plus FSC and SSC (area); time and autofluorescence left out.

    chans: core.fcs.Channel list.
    """
    fluor = [c for c in chans if c.kind == "fluorescence" and not is_af_channel(c.pnn, c.pns)]
    area = [c for c in fluor if c.pnn.upper().endswith("-A")]
    fluor = area or fluor
    scatter = [c for c in chans if c.kind == "scatter"]
    sc_area = [c for c in scatter if c.pnn.upper().endswith("-A")]
    return [c.pnn for c in (sc_area or scatter) + fluor]


# ----- margins ---------------------------------------------------------------

def find_margins(raw, ranges, names):
    """Events piled up on the bottom or top edge of any channel.

    raw: (n, k) untransformed values; ranges: $PnR per column.
    Returns (mask, {channel: count}).
    """
    n = raw.shape[0]
    mask = np.zeros(n, dtype=bool)
    counts = {}
    for j, name in enumerate(names):
        col = raw[:, j]
        if n == 0:
            break
        lo, hi = np.nanmin(col), np.nanmax(col)
        if lo == hi:
            continue  # a constant channel says nothing
        hit = np.zeros(n, dtype=bool)
        top = ranges[j] if ranges is not None and ranges[j] else None
        if top is not None and hi >= top:
            hit |= col >= top
        else:
            at_hi = col >= hi
            if at_hi.sum() >= MARGIN_PILEUP:
                hit |= at_hi
        at_lo = col <= lo
        if at_lo.sum() >= MARGIN_PILEUP:
            hit |= at_lo
        if hit.any():
            counts[name] = int(hit.sum())
            mask |= hit
    return mask, counts


# ----- windows ---------------------------------------------------------------

def events_per_bin(n, min_cells=150, max_bins=500, step=500):
    """PeacoQC's FindEventsPerBin."""
    max_cells = int(np.ceil(n / max_bins * 2))
    max_cells = (max_cells // step) * step + step
    return max(min_cells, max_cells)


def make_breaks(n, epb):
    """PeacoQC's MakeBreaks: windows of epb events overlapping by half. (start, end) pairs."""
    stride = epb - int(np.ceil(epb / 2))
    starts = np.arange(0, n, stride)
    ends = np.minimum(starts + epb, n)
    return np.column_stack([starts, ends])


# ----- density peaks ---------------------------------------------------------

def _bw_nrd0(x):
    """R's bw.nrd0."""
    n = len(x)
    hi = np.std(x, ddof=1) if n > 1 else 0.0
    q75, q25 = np.percentile(x, [75, 25])
    lo = min(hi, (q75 - q25) / 1.34)
    if not lo > 0:
        lo = hi or abs(x[0]) or 1.0
    return 0.9 * lo * n ** -0.2


def density(x, n_points=512, cut=3):
    """R's density() with its defaults (Gaussian kernel, bw.nrd0, FFT on a binned grid)."""
    x = np.asarray(x, dtype=float)
    nx = len(x)
    bw = _bw_nrd0(x)
    n = n_points
    lo_x, hi_x = x.min() - cut * bw, x.max() + cut * bw
    lo, up = lo_x - 4 * bw, hi_x + 4 * bw
    # BinDist: linear binning onto 2n cells; only the first n can be hit
    y = np.zeros(2 * n)
    delta = (up - lo) / (n - 1)
    pos = (x - lo) / delta
    ix = np.floor(pos).astype(int)
    fx = pos - ix
    w = 1.0 / nx
    inside = (ix >= 0) & (ix <= n - 2)
    np.add.at(y, ix[inside], w * (1 - fx[inside]))
    np.add.at(y, ix[inside] + 1, w * fx[inside])
    left = ix == -1
    np.add.at(y, np.zeros(left.sum(), dtype=int), w * fx[left])
    right = ix == n - 1
    np.add.at(y, ix[right], w * (1 - fx[right]))
    kords = np.linspace(0, 2 * (up - lo), 2 * n)
    kords[n + 1:] = -kords[n - 1:0:-1]
    kords = np.exp(-0.5 * (kords / bw) ** 2) / (bw * np.sqrt(2 * np.pi))
    conv = np.fft.ifft(np.fft.fft(y) * np.conj(np.fft.fft(kords))).real
    conv = np.maximum(conv[:n], 0)
    xords = np.linspace(lo, up, n)
    xs = np.linspace(lo_x, hi_x, n_points)
    return xs, np.interp(xs, xords, conv)


def find_peaks(x, remove_zeros=False, peak_removal=1 / 3):
    """PeacoQC's FindThemPeaks: local maxima of the density above peak_removal x the highest."""
    x = np.asarray(x, dtype=float)
    if remove_zeros:
        x = x[x != 0]
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return None
    xs, ys = density(x)
    mid = ys[1:-1]
    sel = (mid > ys[:-2]) & (mid > ys[2:]) & (mid > peak_removal * ys.max())
    peaks = xs[1:-1][sel]
    if len(peaks) == 0:
        peaks = np.array([xs[np.argmax(ys)]])
    return peaks


def channel_peaks(values, breaks, remove_zeros=False, peak_removal=1 / 3, min_bins_pct=10.0):
    """PeacoQC's DetermineAllPeaks + ExtractPeakValues for one channel.

    Returns a list of per-window peak traces (one per peak cluster), or [].
    """
    if find_peaks(values, remove_zeros, peak_removal) is None:
        return []
    per_bin = [find_peaks(values[s:e], remove_zeros, peak_removal) for s, e in breaks]
    found = [(b, p) for b, p in enumerate(per_bin) if p is not None]
    if not found:
        return []
    lengths = [len(p) for _, p in found]
    uniq = list(dict.fromkeys(lengths))
    tally = [lengths.count(u) for u in uniq]
    often = [u for u, t in zip(uniq, tally) if t > (min_bins_pct / 100) * len(lengths)]
    common = max(often) if often else uniq[int(np.argmax(tally))]
    limited = np.vstack([p for (_, p), ln in zip(found, lengths) if ln == common])
    medians = np.median(limited, axis=0)

    rows = []  # (bin, peak, cluster)
    if len(medians) > 1:
        for b, p in found:
            for v in p:
                rows.append((b, v, int(np.argmin(np.abs(v - medians)))))
        final = medians
    else:
        for b, p in found:
            for v in p:
                rows.append((b, v, 0))
        final = np.array([np.median([r[1] for r in rows])])

    # DuplicatePeaks: one peak per cluster per window, dropping the farthest from the median
    by_bin = {}
    for r in rows:
        by_bin.setdefault(r[0], []).append(r)
    kept = []
    for b in dict.fromkeys(r[0] for r in rows):
        use = by_bin[b]
        clusters = [r[2] for r in use]
        dups = [c for i, c in enumerate(clusters) if c in clusters[:i]]
        if dups:
            for c in dups:
                idx = [i for i, r in enumerate(use) if r[2] == c]
                diff = [abs(use[i][1] - final[c]) for i in idx]
                del use[idx[int(np.argmax(diff))]]
            use = sorted(use, key=lambda r: r[2])
        kept.extend(use)

    # TooSmallClusters: drop clusters seen in fewer than half the windows
    nr_bins = max(r[0] for r in kept) + 1
    sizes = {}
    for r in kept:
        sizes[r[2]] = sizes.get(r[2], 0) + 1
    kept = [r for r in kept if sizes[r[2]] >= 0.5 * nr_bins]

    traces = []
    for c in dict.fromkeys(r[2] for r in kept):
        pts = [(r[0], r[1]) for r in kept if r[2] == c]
        trace = np.full(len(breaks), np.median([v for _, v in pts]))
        for b, v in pts:
            trace[b] = v
        traces.append(trace)
    return traces


# ----- isolation tree --------------------------------------------------------

def _avg_path_length(n):
    if n - 1 == 0:
        return 0.0
    return 2 * (np.log(n - 1) + 0.5772156649) - (2 * (n - 1)) / n


def _best_split(col):
    """Best SD-gain split of one column: (gain, split value), as PeacoQC's isolationTreeSD."""
    xs = np.sort(col)
    m = len(xs)
    base = np.std(xs, ddof=1)
    i = np.arange(1, m)  # left part is xs[:i]
    c1, c2 = np.cumsum(xs), np.cumsum(xs ** 2)
    s1, q1 = c1[i - 1], c2[i - 1]
    s2, q2 = c1[-1] - s1, c2[-1] - q1
    n1, n2 = i.astype(float), (m - i).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        sd1 = np.sqrt(np.maximum(q1 - s1 ** 2 / n1, 0) / (n1 - 1))
        sd2 = np.sqrt(np.maximum(q2 - s2 ** 2 / n2, 0) / (n2 - 1))
        sd1[i == 1] = 0
        sd2[i == m - 1] = 0
        gain = (base - (sd1 + sd2) / 2) / base
    ok = ~np.isnan(gain)
    if not ok.any():
        return 0.0, None
    g = np.where(ok, gain, -np.inf)
    best = g.max()
    if best < 0:
        return 0.0, None  # R starts from 0 and keeps the last split reaching it
    last = len(g) - 1 - int(np.argmax(g[::-1] >= best))
    return float(best), float(xs[last])


def isolation_tree(x, gain_limit):
    """PeacoQC's isolationTreeSD. x: (bins, peak traces). Returns a mask of the bins kept."""
    n_rows = x.shape[0]
    max_depth = int(np.ceil(np.log2(n_rows)))
    nodes = [{"depth": 0, "to_split": True, "path_length": None, "rows": np.ones(n_rows, bool),
              "leaf": False}]
    while True:
        todo = [i for i, nd in enumerate(nodes) if nd["to_split"]]
        if not todo:
            break
        node = nodes[todo[0]]
        rows = np.flatnonzero(node["rows"])
        node["to_split"] = False
        if len(rows) > 3 and node["depth"] < max_depth:
            gain_max, split_value, split_col = gain_limit, None, None
            for col in range(x.shape[1]):
                g, v = _best_split(x[rows, col])
                if v is not None and g > gain_max:
                    gain_max, split_value, split_col = g, v, col
            if split_value is not None:
                left = node["rows"] & (x[:, split_col] <= split_value)
                right = node["rows"] & (x[:, split_col] > split_value)
                if left.sum() == len(rows) or right.sum() == len(rows):
                    node["path_length"] = _avg_path_length(len(rows)) + node["depth"]
                    continue
                gain_limit = gain_max
                for part in (left, right):
                    nodes.append({"depth": node["depth"] + 1, "to_split": True, "path_length": None,
                                  "rows": part})
                continue
        node["path_length"] = _avg_path_length(len(rows)) + node["depth"]
    leaves = [nd for nd in nodes if nd["path_length"] is not None]
    biggest = max(leaves, key=lambda nd: nd["rows"].sum())  # first of equals, as R's which.max
    return biggest["rows"].copy()


# ----- MAD on a smoothed trace -----------------------------------------------

def _nknots(n):
    """R's .nknots.smspl."""
    if n < 50:
        return n
    a1, a2, a3, a4 = np.log2(50), np.log2(100), np.log2(140), np.log2(200)
    if n < 200:
        return int(2 ** (a1 + (a2 - a1) * (n - 50) / 150))
    if n < 800:
        return int(2 ** (a2 + (a3 - a2) * (n - 200) / 600))
    if n < 3200:
        return int(2 ** (a3 + (a4 - a3) * (n - 800) / 2400))
    return int(200 + (n - 3200) ** 0.2)


def smooth_spline(y, spar=0.5):
    """Fitted values of R's smooth.spline(seq_along(y), y, spar=spar)."""
    from scipy.interpolate import BSpline

    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 4:
        return y.copy()
    xbar = np.linspace(0, 1, n)
    nk_in = _nknots(n)
    inner = xbar[np.round(np.linspace(0, n - 1, nk_in)).astype(int)]
    knots = np.concatenate([[0.0] * 3, inner, [1.0] * 3])
    nk = len(knots) - 4
    X = BSpline.design_matrix(xbar, knots, 3).toarray()
    # penalty: integral of products of second derivatives (piecewise linear, so 2-point Gauss is exact)
    g = np.array([-1, 1]) / np.sqrt(3)
    pts, wts = [], []
    for a, b in zip(inner[:-1], inner[1:]):
        if b > a:
            pts.extend((a + b) / 2 + (b - a) / 2 * g)
            wts.extend([(b - a) / 2] * 2)
    pts, wts = np.array(pts), np.array(wts)
    D2 = np.column_stack([BSpline(knots, np.eye(nk)[j], 3).derivative(2)(pts) for j in range(nk)])
    omega = (D2 * wts[:, None]).T @ D2
    xtx = X.T @ X
    # R's sslvrg: lambda = ratio * 256^(3 spar - 1), ratio of traces over the inner basis
    ratio = np.trace(xtx[3:nk - 3, 3:nk - 3]) / np.trace(omega[3:nk - 3, 3:nk - 3])
    lam = ratio * 256.0 ** (3 * spar - 1)
    coef = np.linalg.solve(xtx + lam * omega, X.T @ y)
    return X @ coef


def _mad(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def mad_outliers(trace, n_mad):
    smooth = smooth_spline(trace, 0.5)
    med, spread = np.median(smooth), _mad(smooth)
    return (smooth > med + n_mad * spread) | (smooth < med - n_mad * spread)


# ----- consecutive windows, trends, flow rate --------------------------------

def remove_short_regions(good, consecutive_bins):
    """PeacoQC's RemoveShortRegions: runs shorter than consecutive_bins become bad."""
    out = good.copy()
    n = len(good)
    start = 0
    for i in range(1, n + 1):
        if i == n or good[i] != good[start]:
            if i - start < consecutive_bins:
                out[start:i] = False
            start = i
    return out


def _ksmooth_box(y, bandwidth=50):
    x = np.arange(1, len(y) + 1, dtype=float)
    out = np.empty(len(y))
    half = 0.5 * bandwidth
    for i, x0 in enumerate(x):
        sel = np.abs(x - x0) < half
        out[i] = y[sel].mean()
    return out


def find_trend(values, breaks, names):
    """PeacoQC's FindIncreasingDecreasingChannels on window medians."""
    inc, dec = [], []
    for j, name in enumerate(names):
        med = np.array([np.median(values[s:e, j]) for s, e in breaks])
        sm = _ksmooth_box(med)
        if (np.maximum.accumulate(sm) == sm).sum() > 0.75 * len(sm):
            inc.append(name)
        elif (np.minimum.accumulate(sm) == sm).sum() > 0.75 * len(sm):
            dec.append(name)
    if inc and dec:
        return "Increasing and decreasing channel", inc + dec
    if inc:
        return "Increasing channel", inc
    if dec:
        return "Decreasing channel", dec
    return "No increasing or decreasing effect", []


def _running_median(y, window):
    half = window // 2
    pad = np.pad(y, half, mode="edge")
    from numpy.lib.stride_tricks import sliding_window_view

    return np.median(sliding_window_view(pad, 2 * half + 1), axis=1)


def flow_rate_outliers(time_sorted, breaks, n_mad, window=51, floor=0.05):
    """Windows whose event rate jumps away from the running median.

    Rate is events per unit time in each window, on a log scale. The running
    median follows slow changes (the tube emptying); clogs, bubbles and pauses
    show up as sudden jumps. Returns (bad mask, rate per window).
    """
    span = np.array([time_sorted[e - 1] - time_sorted[s] for s, e in breaks], dtype=float)
    count = (breaks[:, 1] - breaks[:, 0]).astype(float)
    with np.errstate(divide="ignore"):
        rate = np.where(span > 0, count / np.where(span > 0, span, 1), np.nan)
    finite = np.isfinite(rate) & (rate > 0)
    if finite.sum() < 10:
        return np.zeros(len(breaks), bool), rate
    log_rate = np.log(np.where(finite, rate, np.nanmedian(rate[finite])))
    # windows with fewer events than usual (the last ones) are noisier; leave them to the signal check
    full = count >= np.median(count)
    trend = _running_median(log_rate, min(window, len(log_rate) | 1))
    resid = log_rate - trend
    spread = max(_mad(resid[full & finite]), floor)
    bad = full & finite & (np.abs(resid) > n_mad * spread)
    return bad, rate


# ----- the whole step --------------------------------------------------------

def _bins_to_ranks(bad_bins, breaks, n):
    """Event positions (in the cleaned order) covered by any bad window."""
    hit = np.zeros(n + 1, dtype=int)
    for s, e in breaks[bad_bins]:
        hit[s] += 1
        hit[e] -= 1
    return np.cumsum(hit[:-1]) > 0


def _runs(mask):
    """[start, end) of each run of True."""
    if not mask.any():
        return []
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return [[int(s), int(e)] for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]


def clean(values, names, time=None, settings=None, margin_raw=None, margin_ranges=None,
          margin_names=None, time_channel=None):
    """Clean one sample.

    values: (n, k) events of the checked channels on their display (transformed) scale.
    time: raw Time values, or None to use event order.
    margin_raw/margin_ranges/margin_names: untransformed values and $PnR of the
    channels checked for margin events (often the raw detectors).
    """
    settings = settings or Settings()
    values = np.asarray(values, dtype=float)
    n = values.shape[0]
    res = Result(settings=settings.to_dict(), n_events=n, channels=list(names),
                 time_channel=time_channel if time is not None else None)
    if time is None:
        res.warnings.append("No Time channel: event order is used as time, and the flow rate isn't checked.")
    elif np.any(np.diff(np.asarray(time, dtype=float)) < 0):
        res.warnings.append("Events aren't stored in time order; they were sorted by Time first.")

    order = time_order(time, n)
    margin = np.zeros(n, dtype=bool)
    if settings.margins and margin_raw is not None and n:
        margin, res.margin_counts = find_margins(np.asarray(margin_raw), margin_ranges, margin_names)
        if margin.mean() > 0.1:
            res.warnings.append(f"{100 * margin.mean():.1f}% of events are margin events. Check the "
                                "detector voltages or the channel ranges.")
    res.margin = margin
    res.removed_by["Margins"] = 100.0 * margin.sum() / max(n, 1)

    keep_pos = np.flatnonzero(~margin[order])     # positions (time order) of non-margin events
    ordered = values[order][keep_pos]
    m = len(ordered)
    epb = int(settings.events_per_bin) or events_per_bin(m)
    breaks = make_breaks(m, epb) if m else np.zeros((0, 2), int)
    nb = len(breaks)
    res.bins = {"events_per_bin": epb, "n_bins": nb}
    if nb < max(MIN_WINDOWS, settings.consecutive_bins):
        if settings.signal or settings.flow_rate:
            res.warnings.append(f"Only {m} events ({nb} windows of {epb}): too few to check the signal "
                                "or the flow rate, so only margin events were removed.")
        return res

    good = np.ones(nb, dtype=bool)
    removed = {}
    if settings.signal and values.shape[1]:
        res.trend, trend_channels = find_trend(ordered, breaks, names)
        if trend_channels:
            res.warnings.append(f"{res.trend} ({', '.join(trend_channels[:4])}"
                                f"{'…' if len(trend_channels) > 4 else ''}): check the Time plots.")
        traces, labels = [], []
        for j, name in enumerate(names):
            for c, tr in enumerate(channel_peaks(ordered[:, j], breaks, settings.remove_zeros,
                                                 settings.peak_removal, settings.min_bins_peakdetection)):
                traces.append(tr)
                labels.append(f"{name}.{c + 1}")
        if traces:
            peaks = np.column_stack(traces)
            if nb >= settings.force_it:
                good = isolation_tree(peaks, settings.it_limit)
                removed["Isolation tree"] = good.copy()
            else:
                res.warnings.append(f"Only {nb} windows: too few for the isolation tree "
                                    f"(needs {settings.force_it}), so only the MAD check ran.")
            idx = np.flatnonzero(good)
            mad_bad = np.zeros(len(idx), dtype=bool)
            for j in range(peaks.shape[1]):
                mad_bad |= mad_outliers(peaks[idx, j], settings.mad)
            before_mad = good.copy()
            good[idx[mad_bad]] = False
            removed["MAD"] = ~(before_mad & ~good)
            res.bins["peaks"] = {lab: peaks[:, j].tolist() for j, lab in enumerate(labels)}

    if settings.flow_rate and time is not None:
        t_sorted = np.asarray(time, dtype=float)[order][keep_pos]
        rate_bad, rate = flow_rate_outliers(t_sorted, breaks, settings.mad)
        removed["Flow rate"] = ~rate_bad
        res.bins["rate"] = [None if not np.isfinite(r) else float(r) for r in rate]
        good &= ~rate_bad

    before_short = good.copy()
    good = remove_short_regions(good, settings.consecutive_bins)
    removed["Consecutive windows"] = ~(before_short & ~good)

    for step, keep in removed.items():
        res.removed_by[step] = 100.0 * _bins_to_ranks(~keep, breaks, m).sum() / max(n, 1)

    bad_pos = _bins_to_ranks(~good, breaks, m)
    # back to ranks among all events: a removed run spans from its first to its last event
    rank_bad = np.zeros(n, dtype=bool)
    for s, e in _runs(bad_pos):
        rank_bad[keep_pos[s]:keep_pos[e - 1] + 1] = True
    res.stretches = _runs(rank_bad)
    res.auto_stretches = [list(s) for s in res.stretches]
    res.bins["mids"] = [float(keep_pos[min((s + e) // 2, m - 1)]) for s, e in breaks]
    res.bins["good"] = good.tolist()
    return res


def stretch_times(result, time):
    """Removed stretches as (start time, end time) on the Time axis."""
    t = sorted_time(result, time)
    return [(float(t[s]), float(t[e - 1])) for s, e in result.stretches]


def sorted_time(result, time):
    if time is None:
        return np.arange(result.n_events, dtype=float)
    return np.asarray(time, dtype=float)[time_order(time, result.n_events)]


def set_stretch_times(result, time, spans):
    """Replace the stretches with (start, end) Time spans, e.g. after the user dragged them."""
    t = sorted_time(result, time)
    out = []
    for a, b in sorted((min(a, b), max(a, b)) for a, b in spans):
        s = int(np.searchsorted(t, a, side="left"))
        e = int(np.searchsorted(t, b, side="right"))
        if e > s:
            if out and s <= out[-1][1]:
                out[-1][1] = max(out[-1][1], e)
            else:
                out.append([s, e])
    result.stretches = out
    result.edited = out != result.auto_stretches


# ----- diagnostic plots ------------------------------------------------------

def plot_data(result, time, values, names, time_step=None, nx=240, ny=90):
    """What the Time plots need, small enough to keep for a report.

    values: (n, k) display-scale events of the channels in names.
    Returns {"x_label", "x_range", "stretches", "rate": (x, y), "channels": [
    {"name", "hist" (nx, ny), "y_range", "peaks": [(x, y), ...]}]}; x in seconds
    when time_step is known, else in Time units (or events with no Time channel).
    """
    t = sorted_time(result, time)
    scale = time_step if (time is not None and time_step) else 1.0
    label = "Time (s)" if (time is not None and time_step) else ("Time" if time is not None else "Event")
    order = time_order(time, result.n_events)
    x = t * scale
    x0, x1 = (float(x[0]), float(x[-1])) if len(x) else (0.0, 1.0)
    if x1 <= x0:
        x1 = x0 + 1.0
    edges = np.linspace(x0, x1, nx + 1)
    keep_pos = ~result.removed(time)[order]
    counts, _ = np.histogram(x, edges)
    width = (x1 - x0) / nx
    out = {"x_label": label, "x_range": (x0, x1), "channels": [],
           "stretches": [(float(x[s]), float(x[e - 1])) for s, e in result.stretches],
           "rate": ((edges[:-1] + width / 2).tolist(), (counts / width).tolist()),
           "removed_pct": 100.0 * (1 - keep_pos.mean()) if len(keep_pos) else 0.0}
    mids = np.asarray(result.bins.get("mids", []), dtype=int)
    peaks = result.bins.get("peaks", {})
    for j, name in enumerate(names):
        v = np.asarray(values[order, j], dtype=float)
        ok = np.isfinite(v)
        lo, hi = (np.percentile(v[ok], [0.2, 99.8]) if ok.any() else (0.0, 1.0))
        if hi <= lo:
            hi = lo + 1.0
        h, _, _ = np.histogram2d(x[ok], v[ok], bins=[edges, np.linspace(lo, hi, ny + 1)])
        traces = []
        for lab, tr in peaks.items():
            if lab.rsplit(".", 1)[0] == name and len(mids):
                traces.append((x[np.clip(mids, 0, len(x) - 1)].tolist(), list(tr)))
        out["channels"].append({"name": name, "hist": h, "y_range": (float(lo), float(hi)),
                                "peaks": traces})
    return out
