"""Axis scales: the transforms a channel can be shown on, and their settings.

Every scale maps the visible data range onto 0..1 on screen. Each one is
described by a small settings dict (Min, Max and the shape sliders shown in
the Transform window) so the UI can switch scales without knowing FlowKit's
parameter names.
"""

import flowkit as fk
import numpy as np

LN10 = np.log(10.0)
BIEX_CHANNELS = 4096.0  # FlowJo's biex lookup table runs 0..4096


class BiexTransform(fk.transforms.WSPBiexTransform):
    """FlowJo's biexponential scale, rescaled from FlowJo's 0..4096 channels to 0..1."""

    def apply(self, events):
        return super().apply(events) / BIEX_CHANNELS

    def inverse(self, events):
        return super().inverse(np.asarray(events, dtype=float) * BIEX_CHANNELS)


class SymLogTransform(fk.transforms._base_transform.Transform):
    """Negative log: a log scale mirrored below zero, with a linear band of
    half-width C around zero so values near zero don't blow up.

    s(x) = sign(x) * log10(1 + |x| / C), mapped so param_n (the most negative
    value shown) lands at 0 and param_t (the top) at 1.
    """

    def __init__(self, param_t, param_n, param_c):
        super().__init__()
        self.param_t = param_t
        self.param_n = param_n
        self.param_c = param_c

    def __repr__(self):
        return f"{type(self).__name__}(t: {self.param_t}, n: {self.param_n}, c: {self.param_c})"

    def _s(self, x):
        x = np.asarray(x, dtype=float)
        return np.sign(x) * np.log10(1.0 + np.abs(x) / self.param_c)

    def apply(self, events):
        lo, hi = self._s(self.param_n), self._s(self.param_t)
        return (self._s(events) - lo) / (hi - lo)

    def inverse(self, events):
        lo, hi = self._s(self.param_n), self._s(self.param_t)
        v = np.asarray(events, dtype=float) * (hi - lo) + lo
        return np.sign(v) * self.param_c * (10.0 ** np.abs(v) - 1.0)


# kind -> (menu label, short button tag)
KINDS = {
    "linear": ("Linear", "Lin"),
    "log": ("Log", "Log"),
    "biex": ("Biex", "Biex"),
    "logicle": ("Logicle", "Logicle"),
    "arcsinh": ("Arcsinh", "Asinh"),
    "hyperlog": ("Hyperlog", "Hyper"),
    "neglog": ("Negative log", "NegLog"),
}

# The shape sliders each scale shows: (key, label, lo, hi, step).
SLIDERS = {
    "linear": [],
    "log": [],
    "biex": [("neg", "Extra Neg. Decades", 0.0, 1.0, 0.05),
             ("width", "Width Basis", 0.0, 3.0, 0.05),
             ("pos", "Positive Decades", 2.0, 7.0, 0.05)],
    "logicle": [("a", "Extra Neg. Decades (A)", 0.0, 2.0, 0.05),
                ("w", "Linear Width (W)", 0.0, 3.0, 0.05),
                ("m", "Positive Decades (M)", 2.0, 7.5, 0.05)],
    "hyperlog": [("a", "Extra Neg. Decades (A)", 0.0, 2.0, 0.05),
                 ("w", "Linear Width (W)", 0.0, 3.0, 0.05),
                 ("m", "Positive Decades (M)", 2.0, 7.5, 0.05)],
    "arcsinh": [("a", "Extra Neg. Decades", 0.0, 2.0, 0.05),
                ("cofactor", "Cofactor (log₁₀)", 0.0, 4.0, 0.05)],
    "neglog": [("band", "Linear Band (log₁₀)", 0.0, 4.0, 0.05)],
}

# Scales the user saved as the default for a parameter (pnn -> {"kind", "settings"}),
# used for channels new to a workspace. The app fills this from its settings.
saved_defaults = {}

# Which ends of the scale the user sets directly; the others follow from the sliders.
EDITABLE_MIN = {"linear", "log", "neglog"}


def kind_of(xform):
    if isinstance(xform, BiexTransform):
        return "biex"
    if isinstance(xform, fk.transforms.WSPBiexTransform):
        return "biex"
    if isinstance(xform, SymLogTransform):
        return "neglog"
    for kind, cls in ((("linear", fk.transforms.LinearTransform)),
                      ("log", fk.transforms.LogTransform),
                      ("logicle", fk.transforms.LogicleTransform),
                      ("arcsinh", fk.transforms.AsinhTransform),
                      ("hyperlog", fk.transforms.HyperlogTransform)):
        if isinstance(xform, cls):
            return kind
    return None


def label(xform):
    kind = kind_of(xform)
    return KINDS[kind][0] if kind else type(xform).__name__


def tag(xform):
    kind = kind_of(xform)
    return KINDS[kind][1] if kind else "?"


def is_linear(xform):
    return xform is None or isinstance(xform, fk.transforms.LinearTransform)


def display_span(xform):
    """Where the scale's visible range sits on screen: (0, 1), except FlowJo
    biex scales imported as they are, which run 0..4096."""
    if isinstance(xform, fk.transforms.WSPBiexTransform) and not isinstance(xform, BiexTransform):
        return 0.0, BIEX_CHANNELS
    return 0.0, 1.0


def raw_range(xform):
    """(min, max) data values at the ends of the scale."""
    lo, hi = display_span(xform)
    out = xform.inverse(np.array([[lo], [hi]], dtype=float)).ravel()
    return float(out[0]), float(out[1])


def settings(xform):
    """The scale's kind and settings dict (min, max and its sliders)."""
    kind = kind_of(xform)
    lo, hi = raw_range(xform)
    s = {"min": lo, "max": hi}
    if kind == "biex":
        s.update(neg=float(xform.negative), width=float(np.log10(-xform.width)),
                 pos=float(xform.positive), max=float(xform.max_value))
    elif kind in ("logicle", "hyperlog"):
        s.update(a=float(xform.param_a), w=float(xform.param_w), m=float(xform.param_m),
                 max=float(xform.param_t))
    elif kind == "arcsinh":
        cof = xform.param_t / np.sinh(xform.param_m * LN10)
        s.update(a=float(xform.param_a), cofactor=float(np.log10(cof)), max=float(xform.param_t))
    elif kind == "neglog":
        s.update(band=float(np.log10(xform.param_c)), min=float(xform.param_n), max=float(xform.param_t))
    return kind, s


def make(kind, s):
    """Build a transform from a kind and settings dict (missing keys get defaults)."""
    d = {**defaults(kind, s.get("max", 262144.0)), **{k: v for k, v in s.items() if v is not None}}
    top = float(d["max"])
    if kind == "linear":
        lo = float(d["min"])
        if lo >= top:
            raise ValueError("Min must be below Max.")
        return fk.transforms.LinearTransform(param_t=top, param_a=-lo)
    if kind == "log":
        lo = float(d["min"])
        if lo <= 0:
            raise ValueError("A log scale can't show zero or negatives: Min must be above 0.")
        if lo >= top:
            raise ValueError("Min must be below Max.")
        return fk.transforms.LogTransform(param_t=top, param_m=float(np.log10(top / lo)))
    if kind == "biex":
        return BiexTransform(negative=float(d["neg"]), width=-float(10.0 ** d["width"]),
                             positive=float(d["pos"]), max_value=top)
    if kind == "logicle":
        return fk.transforms.LogicleTransform(param_t=top, param_w=float(d["w"]),
                                              param_m=float(d["m"]), param_a=float(d["a"]))
    if kind == "hyperlog":
        return fk.transforms.HyperlogTransform(param_t=top, param_w=float(d["w"]),
                                               param_m=float(d["m"]), param_a=float(d["a"]))
    if kind == "arcsinh":
        cof = 10.0 ** float(d["cofactor"])
        return fk.transforms.AsinhTransform(param_t=top, param_m=float(np.arcsinh(top / cof) / LN10),
                                            param_a=float(d["a"]))
    if kind == "neglog":
        lo = float(d["min"])
        if lo >= top:
            raise ValueError("Min must be below Max.")
        return SymLogTransform(param_t=top, param_n=lo, param_c=float(10.0 ** d["band"]))
    raise ValueError(f"unknown scale {kind}")


def defaults(kind, top):
    """Starting settings for a scale whose top is `top` (FlowJo's defaults where it has them)."""
    top = float(top)
    decades = float(np.log10(top))
    return {
        "linear": {"min": 0.0, "max": top},
        "log": {"min": top / 10.0 ** min(decades, 4.5), "max": top},
        # FlowJo: width basis -10, no extra negative decades, 4.42 decades for 2^18
        "biex": {"neg": 0.0, "width": 1.0, "pos": round(decades - 1.0, 2), "max": top},
        "logicle": {"a": 0.0, "w": 0.5, "m": 4.5, "max": top},
        "hyperlog": {"a": 0.0, "w": 0.5, "m": 4.5, "max": top},
        "arcsinh": {"a": 0.0, "cofactor": float(np.log10(150.0 * top / 262144.0)), "max": top},
        "neglog": {"min": -top / 100.0, "band": float(np.log10(top * 1e-4)), "max": top},
    }[kind]


def convert(xform, kind):
    """The same channel on another scale, keeping its top (and bottom, where
    the new scale lets you set it)."""
    _, current = settings(xform)
    lo, hi = current["min"], current["max"]
    s = dict(defaults(kind, hi))
    if kind == "linear":
        s["min"] = lo
    return make(kind, s)


def suggest_width(kind, s, values):
    """A width for biex/logicle-type scales from the data, after Parks et al.
    (2006): wide enough that the 5th percentile of the negative events sits
    inside the linear region. Returns the new settings."""
    values = np.asarray(values, dtype=float)
    neg = values[values < 0]
    s = dict(s)
    if len(neg) < 10:
        r = 0.0
    else:
        r = abs(float(np.percentile(neg, 5)))
    top = float(s["max"])
    if kind == "biex":
        # FlowJo's width basis is about the size of the linear region near zero
        s["width"] = float(np.clip(np.log10(max(r, 1.0)), 0.0, 3.0))
    elif kind in ("logicle", "hyperlog"):
        w = 0.0 if r <= 0 else (s["m"] - np.log10(top / r)) / 2.0
        s["w"] = float(np.clip(w, 0.0, 3.0))
    return s
