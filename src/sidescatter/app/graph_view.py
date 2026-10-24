"""The graph window: plot a population, draw and edit gates on it."""

import flowkit as fk
import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from ..core.workspace import ROOT
from .axes import TransformAxis

BINS = 256
# Blue -> cyan -> green -> yellow -> red, the familiar cytometry pseudocolour ramp.
PSEUDOCOLOR = pg.ColorMap(
    [0.0, 0.25, 0.5, 0.75, 1.0],
    [(20, 40, 200), (0, 190, 230), (40, 200, 60), (250, 220, 0), (230, 30, 20)],
)
GATE_PEN = pg.mkPen((0, 0, 0), width=1.5)
GATE_PEN_HOVER = pg.mkPen((220, 0, 120), width=2)
PLOT_TYPES = ("Pseudocolor", "Dot plot", "Contour", "Histogram")


class GraphView(QtWidgets.QWidget):
    """Shows one population of one sample; child gates are drawn and editable."""

    gatesChanged = QtCore.Signal()

    def __init__(self, workspace, parent=None):
        super().__init__(parent)
        self.ws = workspace
        self.sample_id = None
        self.population = (None, ROOT)  # (name, parent_path); name None = all events
        self._rois = []
        self._polygon_points = []
        self._draw_mode = None

        self.x_combo = QtWidgets.QComboBox()
        self.y_combo = QtWidgets.QComboBox()
        self.type_combo = QtWidgets.QComboBox()
        self.type_combo.addItems(PLOT_TYPES)
        for combo in (self.x_combo, self.y_combo, self.type_combo):
            combo.currentIndexChanged.connect(self.refresh)
            combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)

        self.title = QtWidgets.QLabel()
        self.title.setStyleSheet("font-weight: 600;")

        self.x_axis = TransformAxis("bottom")
        self.y_axis = TransformAxis("left")
        self.plot = pg.PlotWidget(axisItems={"bottom": self.x_axis, "left": self.y_axis})
        self.plot.setBackground("w")
        self.plot.setMenuEnabled(False)
        self.plot.getPlotItem().hideButtons()
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.scene().sigMouseClicked.connect(self._on_click)

        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("Y"))
        bar.addWidget(self.y_combo)
        bar.addWidget(QtWidgets.QLabel("X"))
        bar.addWidget(self.x_combo)
        bar.addWidget(self.type_combo)
        bar.addStretch()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.title)
        layout.addLayout(bar)
        layout.addWidget(self.plot, 1)

        self.status = QtWidgets.QLabel(
            "Select a sample or population in the workspace to plot it.")
        self.status.setStyleSheet("color: #666;")
        layout.addWidget(self.status)

    # ----- what to show ----------------------------------------------------

    def show_population(self, sample_id, name=None, parent_path=ROOT):
        new_sample = sample_id != self.sample_id
        self.sample_id, self.population = sample_id, (name, tuple(parent_path))
        if new_sample:
            self._fill_channel_combos()
        self.refresh()

    def _fill_channel_combos(self):
        chans = self.ws.channels(self.sample_id)
        keep_x, keep_y = self.x_combo.currentData(), self.y_combo.currentData()
        for combo in (self.x_combo, self.y_combo):
            combo.blockSignals(True)
            combo.clear()
            for c in chans:
                combo.addItem(c.label, c.pnn)
        names = [c.pnn for c in chans]
        fluor = [c.pnn for c in chans if c.kind == "fluorescence"]
        x = keep_x if keep_x in names else _first(names, "FSC-A", fluor)
        y = keep_y if keep_y in names else _first(names, "SSC-A", fluor[1:] or fluor)
        self.x_combo.setCurrentIndex(names.index(x))
        self.y_combo.setCurrentIndex(names.index(y))
        for combo in (self.x_combo, self.y_combo):
            combo.blockSignals(False)

    def set_axes(self, x, y):
        for combo, ch in ((self.x_combo, x), (self.y_combo, y)):
            i = combo.findData(ch)
            if i >= 0:
                combo.setCurrentIndex(i)

    @property
    def is_histogram(self):
        return self.type_combo.currentText() == "Histogram"

    # ----- drawing ---------------------------------------------------------

    def refresh(self):
        self.plot.clear()
        self._rois = []
        if self.sample_id is None or self.x_combo.count() == 0:
            return
        name, parent = self.population
        try:
            mask = self.ws.membership(self.sample_id, name, parent)
        except KeyError:
            self.status.setText("This population doesn't apply to this sample.")
            return
        xch, ych = self.x_combo.currentData(), self.y_combo.currentData()
        self.y_combo.setEnabled(not self.is_histogram)
        x = self.ws.display_values(self.sample_id, xch, mask)
        self.x_axis.set_transform(self.ws.transform(xch))

        pop_label = name or "All events"
        self.title.setText(f"{self.sample_id}  ›  {pop_label}   ({int(mask.sum()):,} events)")

        if self.is_histogram:
            self.y_axis.set_transform(None)
            self._draw_histogram(x)
        else:
            y = self.ws.display_values(self.sample_id, ych, mask)
            self.y_axis.set_transform(self.ws.transform(ych))
            self._draw_bivariate(x, y)
        self.plot.setXRange(0, 1, padding=0.0)
        if not self.is_histogram:
            self.plot.setYRange(0, 1, padding=0.0)
        self._draw_gates(xch, None if self.is_histogram else ych)
        self.status.setText(
            "Draw a gate with the toolbar: rectangle, polygon (click points, double-click to "
            "finish), or range on a histogram. Drag a gate to edit it.")

    def _draw_bivariate(self, x, y):
        kind = self.type_combo.currentText()
        if kind == "Dot plot":
            n = min(len(x), 50_000)
            idx = np.random.default_rng(0).choice(len(x), n, replace=False) if len(x) > n else slice(None)
            self.plot.addItem(pg.ScatterPlotItem(x[idx], y[idx], size=2, pen=None,
                                                 brush=pg.mkBrush(30, 30, 30, 160)))
            return
        h, _, _ = np.histogram2d(x, y, bins=BINS, range=[[0, 1], [0, 1]])
        if kind == "Contour":
            from scipy.ndimage import gaussian_filter

            smooth = gaussian_filter(h, 2.0)
            top = smooth.max() or 1
            for level in (0.02, 0.05, 0.1, 0.2, 0.4, 0.7):
                c = pg.IsocurveItem(smooth, level=level * top, pen=pg.mkPen((40, 40, 40), width=1))
                c.setTransform(QtGui.QTransform.fromScale(1 / BINS, 1 / BINS))
                self.plot.addItem(c)
            return
        density = np.log1p(h)
        density = density / (density.max() or 1)
        rgba = PSEUDOCOLOR.map(density, mode="byte")
        rgba[h == 0] = (0, 0, 0, 0)
        img = pg.ImageItem(rgba)
        img.setRect(QtCore.QRectF(0, 0, 1, 1))
        self.plot.addItem(img)

    def _draw_histogram(self, x):
        counts, edges = np.histogram(x, bins=BINS, range=(0, 1))
        self.plot.addItem(pg.PlotDataItem(edges, counts, stepMode="center", fillLevel=0,
                                          brush=(90, 120, 200, 120), pen=pg.mkPen((40, 60, 140))))
        self.plot.setYRange(0, max(counts.max() * 1.1, 1), padding=0)

    # ----- gates on this plot ----------------------------------------------

    def _child_gates(self):
        name, parent = self.population
        path = parent if name is None else parent + (name,)
        return [(child, path, self.ws.get_gate(child, path)) for child, _ in self.ws.children(path)]

    def _draw_gates(self, xch, ych):
        for child, path, gate in self._child_gates():
            dims = [d.id for d in getattr(gate, "dimensions", []) if hasattr(d, "id")]
            if isinstance(gate, fk.gates.RectangleGate) and ych is None and dims == [xch]:
                self._add_range_roi(child, path, gate)
            elif ych is not None and dims == [xch, ych]:
                if isinstance(gate, fk.gates.RectangleGate):
                    self._add_rect_roi(child, path, gate)
                elif isinstance(gate, fk.gates.PolygonGate):
                    self._add_polygon_roi(child, path, gate)

    def _label(self, child, path, anchor, inside=False):
        try:
            freq = self.ws.statistic(self.sample_id, "Freq. of Parent", child, path)
            text = f"{child}\n{freq:.2f}%"
        except KeyError:
            text = child
        label = pg.TextItem(text, color=(0, 0, 0), anchor=(0, 0) if inside else (0, 1))
        label.setPos(*anchor)
        self.plot.addItem(label)

    def _add_rect_roi(self, child, path, gate):
        d0, d1 = gate.dimensions
        x0, x1 = _bounds(d0)
        y0, y1 = _bounds(d1)
        roi = pg.RectROI([x0, y0], [x1 - x0, y1 - y0], pen=GATE_PEN, hoverPen=GATE_PEN_HOVER,
                         movable=True, removable=False)

        def done():
            pos, size = roi.pos(), roi.size()
            d0.min, d0.max = float(pos.x()), float(pos.x() + size.x())
            d1.min, d1.max = float(pos.y()), float(pos.y() + size.y())
            self._gate_edited()

        roi.sigRegionChangeFinished.connect(done)
        self.plot.addItem(roi)
        self._rois.append(roi)
        self._label(child, path, (x0, min(y1, 0.98)), inside=True)

    def _add_polygon_roi(self, child, path, gate):
        roi = pg.PolyLineROI([list(map(float, v)) for v in gate.vertices], closed=True,
                             pen=GATE_PEN, hoverPen=GATE_PEN_HOVER, movable=True)

        def done():
            pts = [roi.mapToParent(h["item"].pos()) for h in roi.handles]
            gate.vertices = [[float(p.x()), float(p.y())] for p in pts]
            self._gate_edited()

        roi.sigRegionChangeFinished.connect(done)
        self.plot.addItem(roi)
        self._rois.append(roi)
        top = max(gate.vertices, key=lambda v: v[1])
        self._label(child, path, (float(top[0]), float(top[1])))

    def _add_range_roi(self, child, path, gate):
        d0 = gate.dimensions[0]
        lo, hi = _bounds(d0)
        region = pg.LinearRegionItem((lo, hi), brush=(220, 0, 120, 30),
                                     pen=pg.mkPen((220, 0, 120), width=1.5))

        def done():
            d0.min, d0.max = map(float, region.getRegion())
            self._gate_edited()

        region.sigRegionChangeFinished.connect(done)
        self.plot.addItem(region)
        self._rois.append(region)
        self._label(child, path, (lo, self.plot.getViewBox().viewRange()[1][1] * 0.98), inside=True)

    def _gate_edited(self):
        self.ws.invalidate()
        self.gatesChanged.emit()
        QtCore.QTimer.singleShot(0, self.refresh)

    # ----- creating gates --------------------------------------------------

    def start_gate(self, mode):
        """mode: 'rectangle', 'polygon' or 'range'."""
        if self.sample_id is None:
            return
        if mode == "range" and not self.is_histogram:
            self.type_combo.setCurrentText("Histogram")
        if mode in ("rectangle", "polygon") and self.is_histogram:
            self.type_combo.setCurrentText("Pseudocolor")
        self._draw_mode = mode
        self._polygon_points = []
        hints = {
            "rectangle": "Click two opposite corners of the rectangle.",
            "polygon": "Click to add points; double-click to close the polygon.",
            "range": "Click the start and end of the range.",
        }
        self.status.setText(hints[mode])

    def _on_click(self, event):
        if self._draw_mode is None or event.button() != QtCore.Qt.LeftButton:
            return
        p = self.plot.getViewBox().mapSceneToView(event.scenePos())
        pt = [float(np.clip(p.x(), 0, 1)), float(np.clip(p.y(), 0, 1))]
        if event.double() and self._draw_mode == "polygon":
            if len(self._polygon_points) >= 3:
                self._finish_gate(self._polygon_points)
            return
        self._polygon_points.append(pt)
        self.plot.addItem(pg.ScatterPlotItem([pt[0]], [pt[1]], size=6, brush=(220, 0, 120)))
        if self._draw_mode in ("rectangle", "range") and len(self._polygon_points) == 2:
            self._finish_gate(self._polygon_points)

    def _finish_gate(self, pts):
        mode, self._draw_mode = self._draw_mode, None
        name, ok = QtWidgets.QInputDialog.getText(self, "New gate", "Population name:")
        if not ok or not name.strip():
            self.refresh()
            return
        name = name.strip()
        sid = self.sample_id
        xch, ych = self.x_combo.currentData(), self.y_combo.currentData()
        (ax, ay), (bx, by) = pts[0], pts[-1]
        if mode == "range":
            dims = [self.ws.make_dimension(sid, xch, min(ax, bx), max(ax, bx))]
            gate = fk.gates.RectangleGate(name, dims)
        elif mode == "rectangle":
            dims = [self.ws.make_dimension(sid, xch, min(ax, bx), max(ax, bx)),
                    self.ws.make_dimension(sid, ych, min(ay, by), max(ay, by))]
            gate = fk.gates.RectangleGate(name, dims)
        else:
            dims = [self.ws.make_dimension(sid, xch), self.ws.make_dimension(sid, ych)]
            gate = fk.gates.PolygonGate(name, dims, [list(p) for p in pts])
        pop, parent = self.population
        path = parent if pop is None else parent + (pop,)
        try:
            self.ws.add_gate(gate, path)
        except Exception as exc:  # duplicate name and similar user errors
            QtWidgets.QMessageBox.warning(self, "Couldn't add gate", str(exc))
        self.gatesChanged.emit()
        self.refresh()

    def export_image(self, path):
        import pyqtgraph.exporters as exporters

        item = self.plot.getPlotItem()
        exporter = exporters.SVGExporter(item) if str(path).lower().endswith(".svg") \
            else exporters.ImageExporter(item)
        if not str(path).lower().endswith(".svg"):
            exporter.parameters()["width"] = 1200
        exporter.export(str(path))


def _bounds(dim):
    lo = 0.0 if dim.min is None else float(dim.min)
    hi = 1.0 if dim.max is None else float(dim.max)
    return lo, hi


def _first(names, preferred, fallback):
    if preferred in names:
        return preferred
    return fallback[0] if fallback else names[0]
