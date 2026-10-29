"""The Identify step of the Unmixing tab: one row per reference control.

Each row shows the control's forward/side scatter with an editable gate, the
peak detector inside that gate with Positive and Negative ranges to drag, and
the spectrum of the positive events across every detector. Unstained controls
get the scatter gate and the spectrum of everything inside it. Every edit is
stored on the control's ControlSpec, in raw units, so it is saved with the
experiment and used when the matrix is rebuilt.
"""

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core import fcs, spectrum, unmixing
from . import theme
from .axes import _si_label, to_raw
from .spectrum_plot import PSEUDOCOLOR, IntensityAxis, SpectrumPlot, _plot, subsample

SCATTER_BINS = 128
HIST_BINS = 128
ROW_HEIGHT = 300
POSITIVE = theme.TEAL
NEGATIVE = theme.CORAL


class SIAxis(pg.AxisItem):
    """Linear axis labelled 1.0M, 2.0M... like the instrument's scatter plots."""

    def tickStrings(self, values, scale, spacing):
        return [_si_label(v) for v in values]


def _region_label(text, color):
    label = pg.TextItem(text, color=color, anchor=(0.5, 0))
    label.setZValue(20)
    return label


class ControlRow(QtWidgets.QFrame):
    """Scatter gate, peak-detector ranges and spectrum for one reference control."""

    edited = QtCore.Signal(object)   # the ControlSpec whose gates changed

    def __init__(self, spec, sample, detectors, result=None, negative=None, parent=None):
        """detectors: raw detector names in file order (the unmixer's).
        negative: (ControlSpec, Sample) of the file used as this control's negative, if any.
        """
        super().__init__(parent)
        self.setObjectName("ControlRow")
        self.setFixedHeight(ROW_HEIGHT)
        self.spec, self.sample, self.detectors = spec, sample, list(detectors)
        self.result, self.negative = result, negative

        raw = sample.get_events(source="raw")
        self.sub = subsample(len(raw))
        self.events = raw[self.sub][:, unmixing._detector_index(sample, self.detectors)]
        self.pair = fcs.scatter_channels(sample)
        if self.pair is not None:
            self.sx = raw[self.sub, sample.get_channel_index(self.pair[0])]
            self.sy = raw[self.sub, sample.get_channel_index(self.pair[1])]
        pnr = sample.channels["pnr"].tolist()
        self.top = max(pnr[sample.get_channel_index(d)] for d in self.detectors)
        self.xform = spectrum.intensity_transform(self.top)
        self.lo = float(spectrum.to_display(self.xform, spectrum.FLOOR))

        self.title = QtWidgets.QLabel()
        self.title.setObjectName("ControlRowTitle")
        self.subtitle = QtWidgets.QLabel()
        self.subtitle.setObjectName("Hint")

        self.scatter = _plot({"bottom": SIAxis("bottom"), "left": SIAxis("left")})
        self.hist = _plot({"bottom": IntensityAxis("bottom"), "left": SIAxis("left")})
        self.hist.getPlotItem().getAxis("bottom").set_transform(self.xform)
        self.spectrum = SpectrumPlot()
        self.scatter_title = self._plot_title()
        self.hist_title = self._plot_title()

        head = QtWidgets.QHBoxLayout()
        head.addWidget(self.title)
        head.addSpacing(10)
        head.addWidget(self.subtitle, 1)
        plots = QtWidgets.QHBoxLayout()
        plots.setSpacing(12)
        for title, plot, stretch in ((self.scatter_title, self.scatter, 2), (self.hist_title, self.hist, 2)):
            col = QtWidgets.QVBoxLayout()
            col.setSpacing(2)
            col.addWidget(title)
            col.addWidget(plot, 1)
            plots.addLayout(col, stretch)
        plots.addWidget(self.spectrum, 5)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.addLayout(head)
        layout.addLayout(plots, 1)

        self._roi = None
        self._gate_label = None
        self.draw_scatter()
        self.update_result(result)

    @staticmethod
    def _plot_title():
        label = QtWidgets.QLabel()
        label.setObjectName("PlotTitle")
        label.setAlignment(QtCore.Qt.AlignCenter)
        return label

    # ----- what the row shows ---------------------------------------------

    @property
    def is_single(self):
        return self.spec.role == "single_stain"

    def in_gate(self):
        """Displayed events inside the scatter gate."""
        if self.pair is None or self.spec.scatter_gate is None:
            return np.ones(len(self.sub), dtype=bool)
        return unmixing.in_polygon(self.sx, self.sy, self.spec.scatter_gate)

    def peak_index(self):
        if self.result is not None and self.result.peak_detector in self.detectors:
            return self.detectors.index(self.result.peak_detector)
        if self.spec.gate_detector in self.detectors:
            return self.detectors.index(self.spec.gate_detector)
        return int(np.argmax(np.median(self.events, axis=0)))

    def _display(self, raw_values):
        return spectrum.to_display(self.xform, raw_values)

    # ----- scatter gate ----------------------------------------------------

    def draw_scatter(self):
        self.scatter.clear()
        if self.pair is None:
            self.scatter_title.setText("No scatter channels")
            return
        self.scatter_title.setText(f"{self.pair[0]} / {self.pair[1]}")
        xmax = float(np.percentile(self.sx, 99.9)) * 1.08 or 1.0
        ymax = float(np.percentile(self.sy, 99.9)) * 1.08 or 1.0
        xmin = min(0.0, float(np.percentile(self.sx, 0.1)))
        ymin = min(0.0, float(np.percentile(self.sy, 0.1)))
        h, _, _ = np.histogram2d(self.sx, self.sy, bins=SCATTER_BINS, range=[[xmin, xmax], [ymin, ymax]])
        dens = np.log1p(h)
        rgba = PSEUDOCOLOR.map(dens.ravel() / (dens.max() or 1), mode="byte").reshape(*h.shape, 4)
        rgba[h == 0] = (0, 0, 0, 0)
        img = pg.ImageItem(rgba, axisOrder="col-major")
        img.setRect(QtCore.QRectF(xmin, ymin, xmax - xmin, ymax - ymin))
        self.scatter.addItem(img)
        self.scatter.setXRange(xmin, xmax, padding=0)
        self.scatter.setYRange(ymin, ymax, padding=0)

        vertices = self.spec.scatter_gate
        if vertices is None:
            # No gate drawn yet: a box around everything, so it already matches "all events".
            dx, dy = (xmax - xmin) * 0.01, (ymax - ymin) * 0.01
            vertices = [[xmin + dx, ymin + dy], [xmax - dx, ymin + dy], [xmax - dx, ymax - dy],
                        [xmin + dx, ymax - dy]]
        self._roi = pg.PolyLineROI([list(map(float, v)) for v in vertices], closed=True,
                                   pen=pg.mkPen(theme.GATE, width=1.6),
                                   hoverPen=pg.mkPen(theme.ACCENT, width=2.2), movable=True)
        self._roi.sigRegionChangeFinished.connect(self._scatter_edited)
        self.scatter.addItem(self._roi)
        self._gate_label = pg.TextItem("", color=theme.INK, anchor=(0, 1),
                                       fill=theme.qcolor(theme.SURFACE, 210))
        self.scatter.addItem(self._gate_label)
        self._update_gate_label()

    def _roi_vertices(self):
        return [[float(p.x()), float(p.y())] for p in
                (self._roi.mapToParent(pos) for _, pos in self._roi.getLocalHandlePositions())]

    def _update_gate_label(self):
        if self._roi is None:
            return
        v = np.array(self._roi_vertices())
        pct = 100.0 * self.in_gate().mean() if len(self.sub) else 0.0
        self._gate_label.setText(f"Gate {pct:.1f}%")
        self._gate_label.setPos(float(v[:, 0].min()), float(v[:, 1].max()))

    def _scatter_edited(self):
        self.spec.scatter_gate = self._roi_vertices()
        self._update_gate_label()
        self._emit_later()

    # ----- peak detector and spectrum ---------------------------------------

    def update_result(self, result):
        """Redraw the histogram and spectrum for a freshly built result."""
        self.result = result
        self.draw_histogram()
        self.draw_spectrum()
        self._update_titles()

    def _update_titles(self):
        carrier = self.spec.carrier.capitalize()
        if self.is_single:
            self.title.setText(f"{self.spec.name} ({carrier})")
            if self.result is None:
                self.subtitle.setText(self.spec.path.name)
            else:
                self.subtitle.setText(
                    f"{self.spec.path.name}  ·  {self.result.n_positive:,} positive events  ·  "
                    f"negative: {self.result.negative_source}"
                    + ("" if self.result.negative_mask is not None else " (dashed)"))
        else:
            self.title.setText(f"Unstained ({carrier})")
            self.subtitle.setText(self.spec.path.name)

    def _positive_sub(self, gate):
        """Displayed positive events, from the last build when there is one."""
        if self.result is not None:
            return self.result.positive_mask[self.sub]
        return gate

    def _negative_sub(self):
        if self.result is None or self.result.negative_mask is None:
            return None
        return self.result.negative_mask[self.sub]

    def _range_of(self, values, mask):
        if mask is None or not mask.any():
            return None
        return float(values[mask].min()), float(values[mask].max())

    def draw_histogram(self):
        self.hist.clear()
        gate = self.in_gate()
        peak = self.peak_index()
        name = spectrum.short_name(self.detectors[peak])
        values = self.events[:, peak]
        shown = self._display(values[gate])
        self.hist_title.setText(f"{name} in gate")
        counts, edges = np.histogram(np.clip(shown, self.lo, 1.0), bins=HIST_BINS, range=(self.lo, 1.0))
        top = max(int(counts.max()) if len(counts) else 0, 1)
        self.hist.addItem(pg.PlotDataItem(edges, counts, stepMode="center", fillLevel=0,
                                          brush=theme.qcolor(theme.ACCENT, 60),
                                          pen=pg.mkPen(theme.ACCENT_DARK, width=1.2)))
        self.hist.setXRange(self.lo, 1.0, padding=0)
        self.hist.setYRange(0, top * 1.18, padding=0)
        self._regions = {}
        self._negative_file_range = None
        if not self.is_single:
            return
        if self.negative is not None and self.result is not None and self.result.negative_mask is None:
            self._draw_negative_file(peak, top)
        pos = self.spec.positive_range or self._range_of(values, self._positive_sub(gate))
        if pos is not None:
            self._add_region("Positive", pos, POSITIVE, top)
        if self.result is not None and self.result.negative_mask is not None:
            neg = self.spec.negative_range or self._range_of(values, self._negative_sub())
            if neg is not None:
                self._add_region("Negative", neg, NEGATIVE, top)

    def _draw_negative_file(self, peak, top):
        """Outline of the unstained file used as the negative, scaled to this plot."""
        spec, sample = self.negative
        raw = sample.get_events(source="raw")
        sub = subsample(len(raw))
        try:
            col = unmixing._detector_index(sample, [self.detectors[peak]])[0]
        except ValueError:
            return
        vals = raw[sub, col]
        pair = fcs.scatter_channels(sample)
        if spec.scatter_gate is not None and pair is not None:
            keep = unmixing.in_polygon(raw[sub, sample.get_channel_index(pair[0])],
                                       raw[sub, sample.get_channel_index(pair[1])], spec.scatter_gate)
            vals = vals[keep]
        counts, edges = np.histogram(np.clip(self._display(vals), self.lo, 1.0), bins=HIST_BINS,
                                     range=(self.lo, 1.0))
        if counts.max() > 0:
            counts = counts * (top / counts.max())
        self.hist.addItem(pg.PlotDataItem(edges, counts, stepMode="center",
                                          pen=pg.mkPen(NEGATIVE, width=1.2, style=QtCore.Qt.DashLine)))
        if len(vals):
            self._negative_file_range = tuple(float(v) for v in np.percentile(vals, [0.5, 99.5]))

    def _add_region(self, kind, bounds, color, top):
        lo, hi = (float(v) for v in self._display(np.array(bounds, dtype=float)))
        region = pg.LinearRegionItem((lo, hi), brush=theme.qcolor(color, 45),
                                     hoverBrush=theme.qcolor(color, 80),
                                     pen=pg.mkPen(color, width=1.4), bounds=(self.lo, 1.0))
        region.setZValue(10)
        label = _region_label(kind, color)
        self.hist.addItem(region)
        self.hist.addItem(label)

        def place():
            a, b = region.getRegion()
            label.setPos((a + b) / 2, top * 1.17)

        def done():
            a, b = region.getRegion()
            raw = to_raw(self.xform, [a, b])
            setattr(self.spec, "positive_range" if kind == "Positive" else "negative_range",
                    (float(raw[0]), float(raw[1])))
            self.spec.gate_detector = self.detectors[self.peak_index()]
            self._emit_later()

        place()
        region.sigRegionChanged.connect(place)
        region.sigRegionChangeFinished.connect(done)
        self._regions[kind] = region

    def draw_spectrum(self):
        gate = self.in_gate()
        if not self.is_single:
            self.spectrum.set_title("Events in gate")
            self.spectrum.set_events(self.events[gate], self.detectors, self.top)
            return
        positives = self._positive_sub(gate)
        markers = []
        peak_values = self.events[:, self.peak_index()]
        for kind, color, mask in (("Positive", POSITIVE, positives), ("Negative", NEGATIVE, self._negative_sub())):
            bounds = getattr(self.spec, f"{kind.lower()}_range") or self._range_of(peak_values, mask)
            if kind == "Negative" and self.result is not None and self.result.negative_mask is None:
                bounds = self._negative_file_range
            if bounds is not None:
                markers.append((bounds[0], bounds[1], color))
        self.spectrum.set_title(f"{self.spec.name} positives")
        self.spectrum.set_events(self.events[positives], self.detectors, self.top, markers)

    def _emit_later(self):
        # Let the dragged item finish its own signal before the row is redrawn.
        QtCore.QTimer.singleShot(0, lambda: self.edited.emit(self.spec))
