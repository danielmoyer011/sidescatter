"""Plot axes that are drawn on the transformed (display) scale but labelled in data units."""

import flowkit as fk
import numpy as np
import pyqtgraph as pg

_SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def _decade_label(raw):
    if raw == 0:
        return "0"
    exp = int(round(np.log10(abs(raw))))
    sign = "-" if raw < 0 else ""
    return f"{sign}10{str(exp).translate(_SUPERSCRIPT)}"


def _si_label(raw):
    a = abs(raw)
    if a >= 1e6:
        return f"{raw / 1e6:g}M"
    if a >= 1e3:
        return f"{raw / 1e3:g}K"
    return f"{raw:g}"


def to_display(xform, raw):
    return xform.apply(np.asarray(raw, dtype=float).reshape(-1, 1)).ravel()


def to_raw(xform, display):
    return xform.inverse(np.asarray(display, dtype=float).reshape(-1, 1)).ravel()


class TransformAxis(pg.AxisItem):
    """Axis whose ticks sit at round data values: decades for log-like scales."""

    def __init__(self, orientation, **kwargs):
        super().__init__(orientation, **kwargs)
        self.xform = None
        self._labels = {}

    def set_transform(self, xform):
        self.xform = xform
        self._labels = {}
        self.picture = None
        self.update()

    def _is_linear(self):
        return self.xform is None or isinstance(self.xform, fk.transforms.LinearTransform)

    def tickValues(self, minVal, maxVal, size):
        if self.xform is None:
            return super().tickValues(minVal, maxVal, size)
        if self._is_linear():
            lo, hi = to_raw(self.xform, [minVal, maxVal])
            levels = super().tickValues(lo, hi, size)
            self._labels = {}
            out = []
            for spacing, raws in levels:
                ds = [float(d) for d in to_display(self.xform, raws)] if len(raws) else []
                if spacing == levels[0][0]:
                    self._labels = {round(d, 9): _si_label(r) for d, r in zip(ds, raws)}
                out.append((spacing, ds))
            return out
        majors, minors, self._labels = [], [], {}
        for exp in range(0, 9):
            for sign in (1, -1):
                raw = sign * 10.0 ** exp if exp else 0.0
                d = float(to_display(self.xform, [raw])[0])
                if minVal <= d <= maxVal and np.isfinite(d):
                    majors.append(d)
                    self._labels[round(d, 9)] = _decade_label(raw)
                if exp:
                    for m in range(2, 10):
                        dm = float(to_display(self.xform, [sign * m * 10.0 ** exp])[0])
                        if minVal <= dm <= maxVal and np.isfinite(dm):
                            minors.append(dm)
        # Drop decade labels that would overlap near zero on biexponential scales.
        majors.sort()
        spaced = []
        min_gap = (maxVal - minVal) * 0.035
        for d in majors:
            if not spaced or d - spaced[-1] >= min_gap:
                spaced.append(d)
        return [(1.0, spaced), (0.5, minors)]

    def tickStrings(self, values, scale, spacing):
        if self.xform is None:
            return super().tickStrings(values, scale, spacing)
        return [self._labels.get(round(v, 9), "") for v in values]


def nice_step(span, target=5):
    """A 1/2/5 x 10^n step that cuts span into about target pieces."""
    if span <= 0 or not np.isfinite(span):
        return 1.0
    raw = span / target
    mag = 10.0 ** np.floor(np.log10(raw))
    for m in (1, 2, 5, 10):
        if raw <= m * mag:
            return float(m * mag)
    return float(10 * mag)


def ticks(xform, lo=0.0, hi=1.0):
    """Ticks for a static plot axis: ([(display, label)], [minor display values]).

    Same placement as TransformAxis: decades on log-like scales, round data
    values on linear ones. xform None means the axis is already in data units
    (histogram counts), running from lo to hi.
    """
    if xform is None or isinstance(xform, fk.transforms.LinearTransform):
        rlo, rhi = (lo, hi) if xform is None else to_raw(xform, [lo, hi])
        step = nice_step(rhi - rlo)
        raws = np.arange(np.ceil(rlo / step) * step, rhi + step * 1e-6, step)
        ds = raws if xform is None else to_display(xform, raws)
        return [(float(d), _si_label(round(float(r), 9))) for d, r in zip(ds, raws)], []
    majors, minors = [], []
    for exp in range(0, 9):
        for sign in (1, -1):
            raw = sign * 10.0 ** exp if exp else 0.0
            d = float(to_display(xform, [raw])[0])
            if lo <= d <= hi and np.isfinite(d):
                majors.append((d, _decade_label(raw)))
            if exp:
                for m in range(2, 10):
                    dm = float(to_display(xform, [sign * m * 10.0 ** exp])[0])
                    if lo <= dm <= hi and np.isfinite(dm):
                        minors.append(dm)
    majors.sort()
    spaced = []
    for d, label in majors:
        if not spaced or d - spaced[-1][0] >= (hi - lo) * 0.035:
            spaced.append((d, label))
    return spaced, minors
