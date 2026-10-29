"""Spectrum plots: every raw detector side by side, as full-spectrum cytometers show them.

SpectrumPlot draws each detector as a column holding the density of all events
on a biexponential intensity scale, scaled column by column so each detector's
spread is visible. SignaturesView draws normalised reference signatures, all
together or one small plot per fluor, coloured by the laser of their peak.
Both lay detectors out in laser order (UV, V, B, YG, R, IR) for any instrument.
"""

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from ..core import spectrum
from . import theme
from .axes import TransformAxis, _decade_label

PSEUDOCOLOR = pg.ColorMap(*zip(*theme.PSEUDOCOLOR_STOPS))
DISPLAY_EVENTS = 20_000
BINS = 200
LABEL_ANGLE = 45


def _style_axes(plot):
    item = plot.getPlotItem()
    for name in ("bottom", "left"):
        ax = item.getAxis(name)
        ax.setPen(pg.mkPen(theme.BORDER_STRONG))
        ax.setTextPen(pg.mkPen(theme.MUTED))


def subsample(n, limit=DISPLAY_EVENTS, seed=0):
    """Sorted indices of at most limit events, the same ones every time."""
    if n <= limit:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, limit, replace=False))


class DetectorAxis(pg.AxisItem):
    """Detector names under a spectrum, one label per detector, slanted like the instrument.

    Labels are drawn at an angle (upright in compact plots) and the font
    shrinks when detectors are packed tightly, so long runs of names never
    clash.
    """

    def __init__(self, orientation="bottom", compact=False, **kwargs):
        super().__init__(orientation, **kwargs)
        self.names = []
        self.compact = compact
        self.angle = 90 if compact else LABEL_ANGLE
        self.setStyle(tickLength=-5)
        self.setHeight(46)

    def set_names(self, names):
        """names: the detectors in plotting order."""
        self.names = [spectrum.short_name(n) for n in names]
        font = self._font(QtGui.QFont())
        widest = max((QtGui.QFontMetrics(font).horizontalAdvance(n) for n in self.names), default=0)
        drop = widest * np.sin(np.radians(self.angle)) + QtGui.QFontMetrics(font).height()
        self.setHeight(int(drop) + 10)
        self.picture = None
        self.update()

    def _font(self, base, spacing=None):
        """The tick font, shrunk when detectors are closer together than a line of text."""
        font = QtGui.QFont(self.style["tickFont"] or base)
        size = font.pointSizeF() if font.pointSizeF() > 0 else 9.0
        if self.compact:
            size = min(size, 7.5)
        if spacing is not None:
            room = spacing * np.sin(np.radians(self.angle))
            height = QtGui.QFontMetricsF(font).height() / (font.pointSizeF() if font.pointSizeF() > 0 else 9.0)
            size = max(5.5, min(size, room / max(height, 1e-6) * 0.95))
        font.setPointSizeF(size)
        return font

    def tickValues(self, minVal, maxVal, size):
        if not self.names:
            return []
        return [(1.0, [float(i) for i in range(len(self.names))])]

    def tickStrings(self, values, scale, spacing):
        return ["" for _ in values]

    def drawPicture(self, p, axisSpec, tickSpecs, textSpecs):
        super().drawPicture(p, axisSpec, tickSpecs, [])
        view = self.linkedView()
        if view is None or not self.names:
            return
        lo, hi = view.viewRange()[0]
        bounds = view.mapRectToItem(self, view.boundingRect())
        if hi <= lo:
            return
        spacing = bounds.width() / (hi - lo)
        p.setClipping(False)
        p.setPen(self.textPen())
        font = self._font(p.font(), spacing)
        p.setFont(font)
        fm = QtGui.QFontMetricsF(font)
        for i, text in enumerate(self.names):
            if not lo <= i <= hi:
                continue
            x = bounds.left() + (i - lo) / (hi - lo) * bounds.width()
            w = fm.horizontalAdvance(text)
            p.save()
            # Rotate about the tick so each label ends right under its detector.
            p.translate(x + fm.height() * 0.35, 4)
            p.rotate(-self.angle)
            p.drawText(QtCore.QRectF(-w - 2, 0, w + 2, fm.height()),
                       int(QtCore.Qt.AlignRight | QtCore.Qt.AlignTop), text)
            p.restore()


class IntensityAxis(TransformAxis):
    """Biexponential intensity axis labelled 0, 10^3, 10^4... as on the instrument.

    The crowded decades around zero (-10^3 to 10^2) get no labels.
    """

    def tickValues(self, minVal, maxVal, size):
        if self._is_linear():
            return super().tickValues(minVal, maxVal, size)
        majors, minors, self._labels = [], [], {}
        for raw in [0.0] + [10.0 ** e for e in range(3, 9)]:
            d = float(spectrum.to_display(self.xform, raw))
            if minVal <= d <= maxVal:
                majors.append(d)
                self._labels[round(d, 9)] = _decade_label(raw)
        for e in range(2, 9):
            for m in range(2, 10):
                d = float(spectrum.to_display(self.xform, m * 10.0 ** e))
                if minVal <= d <= maxVal:
                    minors.append(d)
        return [(1.0, majors), (0.5, minors)]


def _plot(axis_items, background=None):
    plot = pg.PlotWidget(axisItems=axis_items, background=background or theme.SURFACE)
    plot.setMenuEnabled(False)
    plot.getPlotItem().hideButtons()
    plot.setMouseEnabled(x=False, y=False)
    _style_axes(plot)
    return plot


class SpectrumPlot(QtWidgets.QWidget):
    """Density of every event in every raw detector, detectors in laser order."""

    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.det_axis = DetectorAxis("bottom")
        self.int_axis = IntensityAxis("left")
        self.plot = _plot({"bottom": self.det_axis, "left": self.int_axis})
        self.plot.setLabel("left", "Intensity")
        self.plot.setLabel("bottom", "Detector")
        self.title = QtWidgets.QLabel(title)
        self.title.setObjectName("PlotTitle")
        self.title.setAlignment(QtCore.Qt.AlignCenter)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.title)
        layout.addWidget(self.plot, 1)
        self.names, self.order, self.counts = [], [], None
        self.xform = spectrum.intensity_transform()
        self.lo = float(spectrum.to_display(self.xform, spectrum.FLOOR))

    def set_title(self, text):
        self.title.setText(text)
        self.title.setVisible(bool(text))

    def set_events(self, events, names, top=None, markers=()):
        """Plot raw detector events (n_events, n_detectors) named by names.

        markers: [(low, high, colour)] raw-unit ranges drawn as bars left of
        the first detector, e.g. the Positive and Negative gates of a control.
        """
        self.plot.clear()
        self.names = list(names)
        self.order = spectrum.detector_order(self.names)
        self.xform = spectrum.intensity_transform(top)
        self.lo = float(spectrum.to_display(self.xform, spectrum.FLOOR))
        self.int_axis.set_transform(self.xform)
        self.det_axis.set_names([self.names[i] for i in self.order])
        n_det = len(self.order)
        events = np.asarray(events)
        if events.ndim != 2 or not n_det:
            self.counts = None
            return
        events = events[subsample(len(events))][:, self.order]
        display = spectrum.to_display(self.xform, events)
        self.counts = spectrum.density(display, self.lo, 1.0, BINS)
        if len(events):
            scaled = spectrum.column_scale(self.counts)
            rgba = PSEUDOCOLOR.map(scaled.ravel(), mode="byte").reshape(n_det, BINS, 4)
            rgba[self.counts == 0] = (0, 0, 0, 0)
            img = pg.ImageItem(rgba, axisOrder="col-major")
            img.setRect(QtCore.QRectF(-0.5, self.lo, n_det, 1.0 - self.lo))
            self.plot.addItem(img)
        left = -0.5
        for k, (lo, hi, color) in enumerate(markers):
            x = -0.9 - 0.55 * k
            y0, y1 = spectrum.to_display(self.xform, [lo, hi])
            bar = pg.ErrorBarItem(x=np.array([x]), y=np.array([(y0 + y1) / 2]),
                                  top=np.array([(y1 - y0) / 2]), bottom=np.array([(y1 - y0) / 2]),
                                  beam=0.3, pen=pg.mkPen(color, width=1.6))
            self.plot.addItem(bar)
            left = x - 0.4
        self.plot.setXRange(left, n_det - 0.5, padding=0)
        self.plot.setYRange(self.lo, 1.0, padding=0)


class SignaturesView(QtWidgets.QWidget):
    """Normalised signatures, overlaid or one small plot per fluor."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.together = QtWidgets.QToolButton()
        self.together.setText("All together")
        self.each = QtWidgets.QToolButton()
        self.each.setText("One per fluor")
        group = QtWidgets.QButtonGroup(self)
        for b in (self.together, self.each):
            b.setCheckable(True)
            b.setObjectName("Segment")
            group.addButton(b)
        self.together.setChecked(True)
        self.log_check = QtWidgets.QCheckBox("Log scale")
        bar = QtWidgets.QHBoxLayout()
        bar.setSpacing(0)
        bar.addWidget(self.together)
        bar.addWidget(self.each)
        bar.addSpacing(16)
        bar.addWidget(self.log_check)
        bar.addStretch()

        self.overlay = _plot({"bottom": DetectorAxis("bottom")})
        self.legend = self.overlay.addLegend(offset=(-10, 10), labelTextColor=theme.INK,
                                             brush=theme.qcolor(theme.SURFACE, 225))
        self.overlay.setLabel("left", "Normalised intensity")
        self.grid_box = QtWidgets.QWidget()
        self.grid = QtWidgets.QGridLayout(self.grid_box)
        self.grid.setSpacing(10)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.grid_box)
        self.stack = QtWidgets.QStackedWidget()
        self.stack.addWidget(self.overlay)
        self.stack.addWidget(scroll)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 6, 4, 4)
        layout.addLayout(bar)
        layout.addWidget(self.stack, 1)
        self.small = []
        self.references, self.names = [], []
        group.buttonClicked.connect(lambda b: self.stack.setCurrentIndex(int(b is self.each)))
        self.log_check.toggled.connect(lambda _: self.redraw())

    def clear(self):
        self.set_signatures([], [])

    def set_signatures(self, references, names):
        """references: spectral.Reference list; names: their detectors, in file order."""
        self.references, self.names = list(references), list(names)
        self.redraw()

    def _values(self, sig, order):
        y = np.asarray(sig, dtype=float)[order]
        return np.clip(y, 1e-3, None) if self.log_check.isChecked() else y

    def _pen(self, ref):
        if ref.is_autofluorescence:
            return pg.mkPen(theme.MUTED, width=2, style=QtCore.Qt.DashLine)
        return pg.mkPen(spectrum.signature_color(ref.signature, self.names), width=2)

    def redraw(self):
        log = self.log_check.isChecked()
        order = spectrum.detector_order(self.names)
        ordered = [self.names[i] for i in order]
        self.overlay.clear()
        self.legend.clear()
        self.overlay.getPlotItem().getAxis("bottom").set_names(ordered)
        self.overlay.setLogMode(y=log)
        for ref in self.references:
            self.overlay.plot(self._values(ref.signature, order), pen=self._pen(ref), name=ref.name)
        if self.names:
            self.overlay.setXRange(-0.5, len(order) - 0.5, padding=0)

        for w in self.small:
            w.setParent(None)
            w.deleteLater()
        self.small = []
        cols = 3
        for k, ref in enumerate(self.references):
            box = QtWidgets.QWidget()
            v = QtWidgets.QVBoxLayout(box)
            v.setContentsMargins(0, 0, 0, 0)
            v.setSpacing(0)
            label = QtWidgets.QLabel(ref.name)
            label.setObjectName("PlotTitle")
            label.setAlignment(QtCore.Qt.AlignCenter)
            plot = _plot({"bottom": DetectorAxis("bottom", compact=True)})
            plot.getPlotItem().getAxis("bottom").set_names(ordered)
            plot.setLogMode(y=log)
            pen = self._pen(ref)
            color = QtGui.QColor(pen.color())
            color.setAlpha(45)
            y = self._values(ref.signature, order)
            plot.plot(y, pen=pen, fillLevel=None if log else 0, brush=None if log else color)
            plot.setXRange(-0.5, len(order) - 0.5, padding=0)
            if not log:
                plot.setYRange(0, 1.05, padding=0)
            plot.setMinimumHeight(170)
            v.addWidget(label)
            v.addWidget(plot)
            self.grid.addWidget(box, k // cols, k % cols)
            self.small.append(box)
