"""The graph window: plot a population, draw and edit gates on it.

Laid out like FlowJo 10's graph window: sample navigation and plot type on
top, the gating tools in a strip right above the plot, the Y parameter
picker running up the left axis and the X picker under the plot.
"""

import math
import time

import flowkit as fk
import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from . import icons, theme
from ..core.workspace import ALL_SAMPLES, ROOT
from .axes import TransformAxis

BINS = 256
PSEUDOCOLOR = pg.ColorMap(*zip(*theme.PSEUDOCOLOR_STOPS))
GATE_PEN = pg.mkPen(theme.GATE, width=1.6)
GATE_PEN_HOVER = pg.mkPen(theme.ACCENT, width=2.2)
DRAW_PEN = pg.mkPen(theme.ACCENT, width=1.6, style=QtCore.Qt.DashLine)
CLOSE_PEN = pg.mkPen(QtGui.QColor(theme.ACCENT).lighter(150), width=1.2, style=QtCore.Qt.DotLine)
SNAP_RADIUS_PX = 10  # a click this close to a polygon's first corner closes it
SNAP_FILL = QtGui.QColor(theme.ACCENT)
SNAP_FILL.setAlpha(60)
PLOT_TYPES = ("Pseudocolor", "Dot plot", "Contour", "Histogram")
DENSITY = "Density"
# Colour scale for "colour by channel" (low to high), as on FlowSOM/UMAP heat plots.
HEAT = pg.ColorMap([0.0, 0.35, 0.65, 1.0], ["#2B2D6E", "#3E7FC1", "#7CCBA2", "#F8E16C"])

# (mode, tooltip, needs a histogram). None separates the 2D and 1D tools.
GATE_TOOLS = (
    ("pointer", "Select and edit gates (Esc)", None),
    ("rectangle", "Rectangle gate: drag a box", False),
    ("polygon", "Polygon gate: click the corners, then click the first corner, "
                "double-click, right-click or press Enter to close", False),
    ("ellipse", "Ellipse gate: drag its bounding box, then rotate with the handle", False),
    ("quadrant", "Quadrant gate: click where the quadrants meet", False),
    ("freehand", "Freehand gate: hold the mouse button and trace around the cells", False),
    None,
    ("range", "Range gate: drag across the histogram", True),
    ("bisector", "Bisector: click where to split the histogram in two", True),
)
HINTS = {
    "pointer": "Pick a gating tool above the plot. Drag a gate to move it, double-click inside "
               "a gate to open that population, right-click a gate to rename or delete it.",
    "rectangle": "Drag a box around the population.",
    "polygon": "Click to add corners. Close it by clicking the first corner, double-clicking, "
               "right-clicking or pressing Enter. Backspace removes the last corner; Esc cancels.",
    "ellipse": "Drag the ellipse's bounding box. Rotate it afterwards with its round handle.",
    "quadrant": "Click where the four quadrants should meet.",
    "freehand": "Hold the mouse button down and trace around the population.",
    "range": "Drag across the histogram to set the range.",
    "bisector": "Click where the histogram should be split in two.",
}


class GraphView(QtWidgets.QWidget):
    """Shows one population of one sample; child gates are drawn and editable."""

    gatesChanged = QtCore.Signal()
    populationShown = QtCore.Signal(str, object, object)  # sample_id, name, parent_path
    transformRequested = QtCore.Signal(object)  # this view

    def __init__(self, workspace, parent=None):
        super().__init__(parent)
        self.ws = workspace
        self.sample_id = None
        self.population = (None, ROOT)  # (name, parent_path); name None = all events
        self.group = ALL_SAMPLES        # the sample arrows step through this group
        self._rois = []
        self._points = []
        self._preview = None
        self._close_preview = None  # dotted edge from the mouse back to the first corner
        self._snap_ring = None      # ring on the first corner while a click would close
        self._mouse_pt = None
        self._closed_at = 0.0
        self._dragging = False
        self._draw_mode = None

        # ----- header: population, sample arrows, plot type, transform ------
        self.title = QtWidgets.QLabel()
        self.title.setObjectName("GraphTitle")
        self.count_label = QtWidgets.QLabel()
        self.count_label.setObjectName("GraphCount")
        self.prev_button = self._nav_button("‹", "Previous sample in the group", -1)
        self.next_button = self._nav_button("›", "Next sample in the group", +1)
        self.type_combo = QtWidgets.QComboBox()
        self.type_combo.addItems(PLOT_TYPES)
        self.transform_button = QtWidgets.QToolButton()
        self.transform_button.setObjectName("GateTool")
        self.transform_button.setIcon(icons.icon("axis_transform", 20))
        self.transform_button.setIconSize(QtCore.QSize(20, 20))
        self.transform_button.setToolTip("Axis transform (biexponential, log, linear…)")
        self.transform_button.clicked.connect(lambda: self.transformRequested.emit(self))
        self.colour_combo = QtWidgets.QComboBox()
        self.colour_combo.setToolTip("Colour events by density, or by another channel's intensity "
                                     "(median on its display scale), as on a UMAP heat plot")
        self.colour_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.colour_combo.setMinimumContentsLength(8)

        header = QtWidgets.QWidget()
        header.setObjectName("GraphHeader")
        header.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        hl = QtWidgets.QHBoxLayout(header)
        hl.setContentsMargins(10, 6, 10, 6)
        hl.addWidget(self.prev_button)
        hl.addWidget(self.next_button)
        hl.addSpacing(6)
        hl.addWidget(self.title)
        hl.addSpacing(6)
        hl.addWidget(self.count_label)
        hl.addStretch()
        hl.addWidget(self.transform_button)
        hl.addWidget(self.colour_combo)
        hl.addWidget(self.type_combo)

        # ----- gating tools, right above the plot ---------------------------
        self.tool_group = QtWidgets.QButtonGroup(self)
        self.tool_group.setExclusive(True)
        self.tools = {}
        gate_bar = QtWidgets.QWidget()
        gate_bar.setObjectName("GateBar")
        gate_bar.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        gl = QtWidgets.QHBoxLayout(gate_bar)
        gl.setContentsMargins(8, 3, 8, 3)
        gl.setSpacing(2)
        for spec in GATE_TOOLS:
            if spec is None:
                sep = QtWidgets.QFrame()
                sep.setFrameShape(QtWidgets.QFrame.VLine)
                sep.setStyleSheet(f"color: {theme.BORDER};")
                gl.addSpacing(4)
                gl.addWidget(sep)
                gl.addSpacing(4)
                continue
            mode, tip, _ = spec
            b = QtWidgets.QToolButton()
            b.setObjectName("GateTool")
            b.setCheckable(True)
            b.setIcon(icons.icon(mode, 26))
            b.setIconSize(QtCore.QSize(26, 26))
            b.setToolTip(tip)
            b.clicked.connect(lambda _=False, m=mode: self.start_gate(m))
            self.tool_group.addButton(b)
            self.tools[mode] = b
            gl.addWidget(b)
        gl.addStretch()
        self.tools["pointer"].setChecked(True)

        # ----- plot with axis pickers ---------------------------------------
        self.x_combo = QtWidgets.QComboBox()
        self.y_combo = VerticalComboBox()
        for combo in (self.x_combo, self.y_combo):
            combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
            combo.setMinimumContentsLength(10)
        for combo in (self.x_combo, self.y_combo, self.type_combo, self.colour_combo):
            combo.currentIndexChanged.connect(self.refresh)

        self.x_axis = TransformAxis("bottom")
        self.y_axis = TransformAxis("left")
        self.plot = pg.PlotWidget(axisItems={"bottom": self.x_axis, "left": self.y_axis})
        self.plot.setBackground(theme.SURFACE)
        self.plot.setMenuEnabled(False)
        self.plot.getPlotItem().hideButtons()
        self.plot.setMouseEnabled(x=False, y=False)
        for ax in (self.x_axis, self.y_axis):
            ax.setPen(pg.mkPen(theme.BORDER_STRONG))
            ax.setTextPen(pg.mkPen(theme.MUTED))
        self.plot.scene().sigMouseClicked.connect(self._on_scene_click)
        self.plot.viewport().installEventFilter(self)
        self.plot.viewport().setMouseTracking(True)

        canvas = QtWidgets.QWidget()
        canvas.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        canvas.setStyleSheet(f"background: {theme.SURFACE};")
        grid = QtWidgets.QGridLayout(canvas)
        grid.setContentsMargins(6, 8, 12, 6)
        grid.setSpacing(2)
        grid.addWidget(self.y_combo, 0, 0, QtCore.Qt.AlignCenter)
        grid.addWidget(self.plot, 0, 1)
        grid.addWidget(self.x_combo, 1, 1, QtCore.Qt.AlignHCenter)
        grid.setColumnStretch(1, 1)
        grid.setRowStretch(0, 1)

        self.status = QtWidgets.QLabel("Select a sample or population in the workspace to plot it.")
        self.status.setObjectName("GraphHint")
        self.status.setWordWrap(True)
        self.status.setContentsMargins(10, 4, 10, 6)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(header)
        layout.addWidget(gate_bar)
        layout.addWidget(canvas, 1)
        layout.addWidget(self.status)

        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Escape), self, self.cancel_gate)
        # Only live while a polygon is being drawn, so they never steal keys from text fields.
        self._polygon_keys = [
            QtGui.QShortcut(QtGui.QKeySequence(key), self, slot)
            for key, slot in ((QtCore.Qt.Key_Return, self.close_polygon),
                              (QtCore.Qt.Key_Enter, self.close_polygon),
                              (QtCore.Qt.Key_Backspace, self.undo_polygon_point))]
        for sc in self._polygon_keys:
            sc.setEnabled(False)

    def _nav_button(self, text, tip, step):
        b = QtWidgets.QToolButton()
        b.setObjectName("NavButton")
        b.setText(text)
        b.setToolTip(tip)
        b.clicked.connect(lambda: self.step_sample(step))
        return b

    # ----- what to show ----------------------------------------------------

    def show_population(self, sample_id, name=None, parent_path=ROOT):
        new_sample = sample_id != self.sample_id
        self.sample_id, self.population = sample_id, (name, tuple(parent_path))
        if new_sample:
            self._fill_channel_combos()
        self.refresh()
        self.populationShown.emit(sample_id, name, tuple(parent_path))

    def step_sample(self, step):
        """Show the same population in the previous/next sample of the group."""
        order = [s for s in self.ws.groups.get(self.group) or self.ws.groups.get(ALL_SAMPLES, [])
                 if s in self.ws.samples]
        if self.sample_id not in order or len(order) < 2:
            return
        sid = order[(order.index(self.sample_id) + step) % len(order)]
        name, parent = self.population
        if not self.ws.applies_to(sid, name, parent):
            name, parent = None, ROOT
        self.show_population(sid, name, parent)

    def _fill_channel_combos(self):
        chans = self.ws.channels(self.sample_id)
        keep_x, keep_y = self.x_combo.currentData(), self.y_combo.currentData()
        keep_colour = self.colour_combo.currentData()
        self.colour_combo.blockSignals(True)
        self.colour_combo.clear()
        self.colour_combo.addItem(DENSITY, None)
        for c in chans:
            self.colour_combo.addItem(c.label, c.pnn)
        self.colour_combo.setCurrentIndex(max(0, self.colour_combo.findData(keep_colour)))
        self.colour_combo.blockSignals(False)
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

    def set_colour(self, channel):
        """Colour by a channel's intensity; None for density."""
        self.colour_combo.setCurrentIndex(max(0, self.colour_combo.findData(channel)))

    def set_axes(self, x, y):
        for combo, ch in ((self.x_combo, x), (self.y_combo, y)):
            i = combo.findData(ch)
            if i >= 0:
                combo.setCurrentIndex(i)

    @property
    def is_histogram(self):
        return self.type_combo.currentText() == "Histogram"

    @property
    def population_path(self):
        name, parent = self.population
        return parent if name is None else parent + (name,)

    # ----- drawing ---------------------------------------------------------

    def refresh(self):
        self.plot.clear()
        self._rois = []
        self._preview = self._close_preview = self._snap_ring = None
        if self.sample_id is not None and self.sample_id not in self.ws.samples:
            self.sample_id = None
            self.title.setText("")
            self.count_label.setText("")
            self.status.setText("This sample is no longer in the workspace.")
        if self.sample_id is None or self.x_combo.count() == 0:
            return
        name, parent = self.population
        try:
            mask = self.ws.membership(self.sample_id, name, parent)
        except KeyError:
            self.status.setText("This population doesn't apply to this sample.")
            return
        xch, ych = self.x_combo.currentData(), self.y_combo.currentData()
        self.y_combo.setVisible(not self.is_histogram)
        self.colour_combo.setVisible(self.type_combo.currentText() in ("Pseudocolor", "Dot plot"))
        for mode, b in self.tools.items():
            needs_hist = next(s[2] for s in GATE_TOOLS if s and s[0] == mode)
            b.setEnabled(needs_hist is None or needs_hist == self.is_histogram)
        x = self.ws.display_values(self.sample_id, xch, mask)
        self.x_axis.set_transform(self.ws.transform(xch))

        trail = " › ".join([*parent[1:], name]) if name else "All events"
        self.title.setText(f"{self.sample_id}  ›  {trail}" if name else self.sample_id)
        self.count_label.setText(f"{int(mask.sum()):,} events")
        win = self.window()
        if win is not self and isinstance(win, GraphWindow):
            win.setWindowTitle(f"{self.sample_id}: {name or 'All events'}")

        if self.is_histogram:
            self.y_axis.set_transform(None)
            self._draw_histogram(x[np.isfinite(x)])
        else:
            y = self.ws.display_values(self.sample_id, ych, mask)
            self.y_axis.set_transform(self.ws.transform(ych))
            colour = self.colour_combo.currentData() if self.colour_combo.isVisible() else None
            c = self.ws.display_values(self.sample_id, colour, mask) if colour else None
            # Computed channels (UMAP...) are empty outside the population they were run on.
            ok = np.isfinite(x) & np.isfinite(y)
            if not ok.all():
                x, y, c = x[ok], y[ok], (None if c is None else c[ok])
            if c is not None:
                self._draw_colour_by(x, y, c)
            else:
                self._draw_bivariate(x, y)
        self.plot.setXRange(0, 1, padding=0.0)
        if not self.is_histogram:
            self.plot.setYRange(0, 1, padding=0.0)
        self._draw_gates(xch, None if self.is_histogram else ych)
        self.status.setText(HINTS[self._draw_mode or "pointer"])

    def _draw_bivariate(self, x, y):
        kind = self.type_combo.currentText()
        if kind == "Dot plot":
            n = min(len(x), 50_000)
            idx = np.random.default_rng(0).choice(len(x), n, replace=False) if len(x) > n else slice(None)
            self.plot.addItem(pg.ScatterPlotItem(x[idx], y[idx], size=2, pen=None,
                                                 brush=theme.qcolor(theme.INK, 150)))
            return
        h, _, _ = np.histogram2d(x, y, bins=BINS, range=[[0, 1], [0, 1]])
        if kind == "Contour":
            from scipy.ndimage import gaussian_filter

            smooth = gaussian_filter(h, 2.0)
            top = smooth.max() or 1
            for i, level in enumerate((0.02, 0.05, 0.1, 0.2, 0.4, 0.7)):
                color = PSEUDOCOLOR.map(0.15 + i * 0.16, mode="qcolor")
                c = pg.IsocurveItem(smooth, level=level * top, pen=pg.mkPen(color, width=1.2))
                c.setTransform(QtGui.QTransform.fromScale(1 / BINS, 1 / BINS))
                self.plot.addItem(c)
            return
        density = np.log1p(h)
        density = density / (density.max() or 1)
        rgba = PSEUDOCOLOR.map(density, mode="byte")
        rgba[h == 0] = (0, 0, 0, 0)
        img = pg.ImageItem(rgba, axisOrder="col-major")
        img.setRect(QtCore.QRectF(0, 0, 1, 1))
        self.plot.addItem(img)

    def _draw_colour_by(self, x, y, c):
        """Each bin (or dot) coloured by the median of another channel."""
        good = np.isfinite(c)
        x, y, c = x[good], y[good], np.clip(c[good], 0, 1)
        if self.type_combo.currentText() == "Dot plot":
            n = min(len(x), 50_000)
            idx = np.random.default_rng(0).choice(len(x), n, replace=False) if len(x) > n else slice(None)
            lo, hi = (np.percentile(c, [1, 99]) if len(c) else (0, 1))
            colours = HEAT.map((c[idx] - lo) / ((hi - lo) or 1), mode="qcolor")
            self.plot.addItem(pg.ScatterPlotItem(x[idx], y[idx], size=2.5, pen=None, brush=colours))
            return
        bins = BINS // 2  # coarser bins keep the median steady
        ix = np.clip((x * bins).astype(int), 0, bins - 1)
        iy = np.clip((y * bins).astype(int), 0, bins - 1)
        cell = ix * bins + iy
        order = np.argsort(cell, kind="stable")
        cell, cs = cell[order], c[order]
        starts = np.flatnonzero(np.r_[True, cell[1:] != cell[:-1]])
        medians = np.array([np.median(part) for part in np.split(cs, starts[1:])]) if len(cs) else []
        grid = np.full(bins * bins, np.nan)
        if len(cs):
            grid[cell[starts]] = medians
        grid = grid.reshape(bins, bins)
        finite = grid[np.isfinite(grid)]
        lo, hi = (np.percentile(finite, [1, 99]) if finite.size else (0, 1))
        rgba = HEAT.map(np.nan_to_num((grid - lo) / ((hi - lo) or 1)).clip(0, 1), mode="byte")
        rgba[~np.isfinite(grid)] = (0, 0, 0, 0)
        img = pg.ImageItem(rgba, axisOrder="col-major")
        img.setRect(QtCore.QRectF(0, 0, 1, 1))
        self.plot.addItem(img)

    def _draw_histogram(self, x):
        counts, edges = np.histogram(x, bins=BINS, range=(0, 1))
        self.plot.addItem(pg.PlotDataItem(edges, counts, stepMode="center", fillLevel=0,
                                          brush=theme.qcolor(theme.TEAL, 70),
                                          pen=pg.mkPen(theme.TEAL, width=1.4)))
        self.plot.setYRange(0, max(counts.max() * 1.1, 1), padding=0)

    # ----- gates on this plot ----------------------------------------------

    def _child_gates(self):
        path = self.population_path
        return [(child, path, self.ws.get_gate(child, path)) for child, _ in self.ws.children(path)]

    def _draw_gates(self, xch, ych):
        quads, bisects = {}, {}
        for child, path, gate in self._child_gates():
            dims = [d.id for d in getattr(gate, "dimensions", []) if hasattr(d, "id")]
            if isinstance(gate, fk.gates.RectangleGate) and ych is None and dims == [xch]:
                split = _half_open(gate.dimensions[0])
                if split is not None:
                    bisects.setdefault(round(split, 9), []).append((child, path, gate))
                else:
                    self._add_range_roi(child, path, gate)
            elif ych is not None and dims == [xch, ych]:
                if isinstance(gate, fk.gates.RectangleGate):
                    sx, sy = (_half_open(d) for d in gate.dimensions)
                    if sx is not None and sy is not None:
                        quads.setdefault((round(sx, 9), round(sy, 9)), []).append((child, path, gate))
                    else:
                        self._add_rect_roi(child, path, gate)
                elif isinstance(gate, fk.gates.PolygonGate):
                    self._add_polygon_roi(child, path, gate)
                elif isinstance(gate, fk.gates.EllipsoidGate):
                    self._add_ellipse_roi(child, path, gate)
        for members in quads.values():
            if len(members) > 1:
                self._add_quadrant(members)
            else:
                self._add_rect_roi(*members[0])
        for members in bisects.values():
            if len(members) > 1:
                self._add_bisector(members)
            else:
                self._add_range_roi(*members[0])

    def _freq_text(self, child, path, short=False):
        try:
            freq = self.ws.statistic(self.sample_id, "Freq. of Parent", child, path)
        except KeyError:
            return child
        name = child.split(":")[0] if short else child
        return f"{name}\n{freq:.2f}%"

    def _label(self, child, path, anchor, align=(0, 1), short=False):
        label = pg.TextItem(self._freq_text(child, path, short), color=theme.INK, anchor=align,
                            fill=theme.qcolor(theme.SURFACE, 210), border=None)
        label.setPos(*anchor)
        self.plot.addItem(label)
        return label

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
        self._label(child, path, (x0, min(y1, 0.99)), align=(0, 0))

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
        self._label(child, path, (float(top[0]), min(float(top[1]), 0.92)))

    def _add_ellipse_roi(self, child, path, gate):
        (cx, cy), a, b, angle = ellipse_shape(gate)
        roi = pg.EllipseROI([0, 0], [2 * a, 2 * b], angle=math.degrees(angle), pen=GATE_PEN,
                            hoverPen=GATE_PEN_HOVER, movable=True)
        centre = roi.mapToParent(QtCore.QPointF(a, b))
        roi.setPos(QtCore.QPointF(roi.pos().x() + cx - centre.x(), roi.pos().y() + cy - centre.y()),
                   update=False, finish=False)

        def done():
            size = roi.size()
            c = roi.mapToParent(QtCore.QPointF(size.x() / 2, size.y() / 2))
            mean, cov = ellipse_gate_params((c.x(), c.y()), size.x() / 2, size.y() / 2,
                                            math.radians(roi.angle()))
            gate.coordinates, gate.covariance_matrix, gate.distance_square = mean, cov, 1.0
            self._gate_edited()

        roi.sigRegionChangeFinished.connect(done)
        self.plot.addItem(roi)
        self._rois.append(roi)
        self._label(child, path, (cx - a * 0.5, min(cy + max(a, b), 0.95)))

    def _add_range_roi(self, child, path, gate):
        d0 = gate.dimensions[0]
        lo, hi = _bounds(d0)
        region = pg.LinearRegionItem((lo, hi), brush=theme.qcolor(theme.ACCENT, 30),
                                     hoverBrush=theme.qcolor(theme.ACCENT, 50),
                                     pen=pg.mkPen(theme.ACCENT, width=1.6))

        def done():
            d0.min, d0.max = map(float, region.getRegion())
            self._gate_edited()

        region.sigRegionChangeFinished.connect(done)
        self.plot.addItem(region)
        self._rois.append(region)
        self._label(child, path, (lo, self.plot.getViewBox().viewRange()[1][1] * 0.98), align=(0, 0))

    def _add_quadrant(self, members):
        """Four (or more) half-open rectangles sharing one corner: one draggable crosshair."""
        d0 = members[0][2].dimensions
        sx, sy = _half_open(d0[0]), _half_open(d0[1])
        vline = pg.InfiniteLine(sx, angle=90, movable=True, bounds=(0, 1), pen=GATE_PEN,
                                hoverPen=GATE_PEN_HOVER)
        hline = pg.InfiniteLine(sy, angle=0, movable=True, bounds=(0, 1), pen=GATE_PEN,
                                hoverPen=GATE_PEN_HOVER)

        def done():
            for _, _, g in members:
                _move_split(g.dimensions[0], float(vline.value()))
                _move_split(g.dimensions[1], float(hline.value()))
            self._gate_edited()

        for line in (vline, hline):
            line.sigPositionChangeFinished.connect(done)
            self.plot.addItem(line)
            self._rois.append(line)
        for child, path, g in members:
            right = g.dimensions[0].max is None
            top = g.dimensions[1].max is None
            pos = (0.985 if right else 0.015, 0.985 if top else 0.015)
            self._label(child, path, pos, align=(1 if right else 0, 0 if top else 1), short=True)

    def _add_bisector(self, members):
        split = _half_open(members[0][2].dimensions[0])
        line = pg.InfiniteLine(split, angle=90, movable=True, bounds=(0, 1),
                               pen=pg.mkPen(theme.ACCENT, width=1.8),
                               hoverPen=pg.mkPen(theme.ACCENT_DARK, width=2.4))

        def done():
            for _, _, g in members:
                _move_split(g.dimensions[0], float(line.value()))
            self._gate_edited()

        line.sigPositionChangeFinished.connect(done)
        self.plot.addItem(line)
        self._rois.append(line)
        top = self.plot.getViewBox().viewRange()[1][1] * 0.98
        for child, path, g in members:
            right = g.dimensions[0].max is None
            self._label(child, path, (split + (0.01 if right else -0.01), top), align=(0 if right else 1, 0))

    def _gate_edited(self):
        self.ws.invalidate()
        self.gatesChanged.emit()
        QtCore.QTimer.singleShot(0, self.refresh)

    # ----- clicking on gates (pointer tool) ---------------------------------

    def gate_at(self, x, y=None):
        """The (name, parent_path) of the smallest child gate containing a point."""
        xch, ych = self.x_combo.currentData(), self.y_combo.currentData()
        hits = []
        for child, path, gate in self._child_gates():
            dims = [d.id for d in getattr(gate, "dimensions", []) if hasattr(d, "id")]
            if self.is_histogram:
                if isinstance(gate, fk.gates.RectangleGate) and dims == [xch]:
                    lo, hi = _open_bounds(gate.dimensions[0])
                    if lo <= x <= hi:
                        hits.append((min(hi, 1) - max(lo, 0), child, path))
            elif dims == [xch, ych] and _contains(gate, x, y):
                hits.append((_area(gate), child, path))
        if not hits:
            return None
        _, child, path = min(hits, key=lambda h: h[0])
        return child, path

    def _on_scene_click(self, event):
        if self._draw_mode is not None or self.sample_id is None:
            return
        vb = self.plot.getViewBox()
        if not vb.sceneBoundingRect().contains(event.scenePos()):
            return
        p = vb.mapSceneToView(event.scenePos())
        hit = self.gate_at(p.x(), p.y())
        if hit is None:
            return
        child, path = hit
        if event.button() == QtCore.Qt.LeftButton and event.double():
            QtCore.QTimer.singleShot(0, lambda: self.show_population(self.sample_id, child, path))
        elif event.button() == QtCore.Qt.RightButton:
            self._gate_menu(child, path, event.screenPos())

    def _gate_menu(self, child, path, screen_pos):
        menu = QtWidgets.QMenu(self)
        menu.addAction(f"Open {child}", lambda: self.show_population(self.sample_id, child, path))
        menu.addSeparator()
        menu.addAction(icons.icon("rename", 16), "Rename…", lambda: self._rename_gate(child, path))
        menu.addAction(icons.icon("delete", 16), "Delete gate", lambda: self._delete_gate(child, path))
        menu.exec(QtCore.QPoint(int(screen_pos.x()), int(screen_pos.y())))

    def _rename_gate(self, child, path):
        new, ok = QtWidgets.QInputDialog.getText(self, "Rename population", "New name:", text=child)
        if ok and new.strip() and new.strip() != child:
            self.ws.rename_gate(child, new.strip(), path)
            self.gatesChanged.emit()
            self.refresh()

    def _delete_gate(self, child, path):
        answer = QtWidgets.QMessageBox.question(
            self, "Delete gate", f"Delete {child} and every population inside it, in all samples?")
        if answer == QtWidgets.QMessageBox.Yes:
            self.ws.remove_gate(child, path)
            self.gatesChanged.emit()
            self.refresh()

    # ----- creating gates --------------------------------------------------

    def start_gate(self, mode):
        """mode: a GATE_TOOLS name, e.g. 'rectangle', 'polygon', 'ellipse', 'quadrant',
        'freehand', 'range' or 'bisector'. 'pointer' stops drawing."""
        self._clear_preview()
        self._points, self._dragging = [], False
        for sc in self._polygon_keys:
            sc.setEnabled(mode == "polygon" and self.sample_id is not None)
        if mode == "pointer" or self.sample_id is None:
            self._draw_mode = None
            self.tools["pointer"].setChecked(True)
            self.plot.viewport().unsetCursor()
            self.status.setText(HINTS["pointer"])
            self._set_rois_enabled(True)
            return
        needs_hist = next(s[2] for s in GATE_TOOLS if s and s[0] == mode)
        if needs_hist and not self.is_histogram:
            self.type_combo.setCurrentText("Histogram")
        if not needs_hist and self.is_histogram:
            self.type_combo.setCurrentText("Pseudocolor")
        self._draw_mode = mode
        self.tools[mode].setChecked(True)
        self.plot.viewport().setCursor(QtCore.Qt.CrossCursor)
        self.status.setText(HINTS[mode])
        self._set_rois_enabled(False)

    def cancel_gate(self):
        if self._draw_mode is not None:
            self.start_gate("pointer")

    def close_polygon(self):
        """Join the last corner to the first and make the gate. Needs 3 or more corners."""
        if self._draw_mode == "polygon" and len(self._points) >= 3:
            self._closed_at = time.monotonic()
            self.create_gate("polygon", self._points)
            return True
        return False

    def undo_polygon_point(self):
        if self._draw_mode != "polygon" or not self._points:
            return
        self._points.pop()
        if self._points and self._mouse_pt is not None:
            self._update_preview(self._mouse_pt)
        else:
            self._clear_preview()

    def _near_first_point(self, widget_pos):
        """True when the mouse is within SNAP_RADIUS_PX screen pixels of the first corner."""
        if len(self._points) < 3:
            return False
        vb = self.plot.getViewBox()
        first = self.plot.mapFromScene(vb.mapViewToScene(QtCore.QPointF(*self._points[0])))
        return math.hypot(widget_pos.x() - first.x(), widget_pos.y() - first.y()) <= SNAP_RADIUS_PX

    def _set_rois_enabled(self, on):
        for roi in self._rois:
            roi.setAcceptedMouseButtons(QtCore.Qt.LeftButton if on else QtCore.Qt.NoButton)
            roi.setAcceptHoverEvents(on)

    def _view_point(self, widget_pos):
        vb = self.plot.getViewBox()
        p = vb.mapSceneToView(self.plot.mapToScene(widget_pos.toPoint()))
        return [float(np.clip(p.x(), 0, 1)), float(np.clip(p.y(), 0, 1))]

    def eventFilter(self, obj, event):
        if obj is not self.plot.viewport():
            return super().eventFilter(obj, event)
        t = event.type()
        if (t == QtCore.QEvent.MouseButtonDblClick and self._draw_mode is None and time.monotonic()
                - self._closed_at < QtWidgets.QApplication.doubleClickInterval() / 1000):
            return True  # the second half of a click that closed a polygon; don't open the gate
        if self._draw_mode is None:
            return super().eventFilter(obj, event)
        mode = self._draw_mode
        if t == QtCore.QEvent.MouseButtonDblClick and event.button() == QtCore.Qt.LeftButton:
            if mode == "polygon":
                self.close_polygon()
            return True
        if t == QtCore.QEvent.MouseButtonPress:
            if event.button() == QtCore.Qt.RightButton:
                if not (mode == "polygon" and self.close_polygon()):
                    self.cancel_gate()
                return True
            if event.button() != QtCore.Qt.LeftButton:
                return True
            pt = self._view_point(event.position())
            if mode in ("quadrant", "bisector"):
                self.create_gate(mode, [pt])
            elif mode == "polygon":
                if self._near_first_point(event.position()):
                    self.close_polygon()
                    return True
                if not self._points or self._points[-1] != pt:
                    self._points.append(pt)
                self._update_preview(pt, event.position())
            else:
                self._points, self._dragging = [pt], True
            return True
        if t == QtCore.QEvent.MouseMove:
            pt = self._view_point(event.position())
            if self._dragging and mode == "freehand":
                self._points.append(pt)
            self._mouse_pt = pt
            if self._dragging or (mode == "polygon" and self._points):
                self._update_preview(pt, event.position())
            return True
        if t == QtCore.QEvent.MouseButtonRelease and event.button() == QtCore.Qt.LeftButton:
            if not self._dragging:
                return True
            self._dragging = False
            pt = self._view_point(event.position())
            start = self._points[0]
            if mode == "freehand":
                self._points.append(pt)
                if len(self._points) >= 3:
                    self.create_gate("freehand", self._points)
                return True
            if abs(pt[0] - start[0]) < 0.005 or (mode != "range" and abs(pt[1] - start[1]) < 0.005):
                self._clear_preview()
                self._points = []
                return True  # a click, not a drag
            self.create_gate(mode, [start, pt])
            return True
        return super().eventFilter(obj, event)

    def _clear_preview(self):
        for attr in ("_preview", "_close_preview", "_snap_ring"):
            item = getattr(self, attr)
            if item is not None:
                self.plot.removeItem(item)
                setattr(self, attr, None)

    def _update_polygon_extras(self, pt, widget_pos):
        """The dotted closing edge and, when a click would close, the ring on the first corner."""
        if self._close_preview is None:
            self._close_preview = pg.PlotDataItem(pen=CLOSE_PEN)
            self._snap_ring = pg.ScatterPlotItem(size=2 * SNAP_RADIUS_PX, symbol="o",
                                                 pen=pg.mkPen(theme.ACCENT, width=2),
                                                 brush=pg.mkBrush(SNAP_FILL))
            self.plot.addItem(self._close_preview)
            self.plot.addItem(self._snap_ring)
        first = self._points[0]
        snap = widget_pos is not None and self._near_first_point(widget_pos)
        if len(self._points) >= 2 and not snap:
            self._close_preview.setData([pt[0], first[0]], [pt[1], first[1]])
        else:
            self._close_preview.setData([], [])
        self._snap_ring.setData([first[0]] if snap else [], [first[1]] if snap else [])

    def _update_preview(self, pt, widget_pos=None):
        mode, pts = self._draw_mode, self._points
        if mode == "polygon":
            self._update_polygon_extras(pt, widget_pos)
            xs, ys = zip(*(pts + [pt]))
        elif mode == "freehand":
            xs, ys = zip(*pts)
        elif mode == "range":
            lo, hi = sorted((pts[0][0], pt[0]))
            top = self.plot.getViewBox().viewRange()[1][1]
            xs, ys = (lo, lo, hi, hi, lo), (0, top, top, 0, 0)
        elif mode == "rectangle":
            (ax, ay), (bx, by) = pts[0], pt
            xs, ys = (ax, bx, bx, ax, ax), (ay, ay, by, by, ay)
        elif mode == "ellipse":
            (ax, ay), (bx, by) = pts[0], pt
            t = np.linspace(0, 2 * np.pi, 80)
            xs = (ax + bx) / 2 + abs(bx - ax) / 2 * np.cos(t)
            ys = (ay + by) / 2 + abs(by - ay) / 2 * np.sin(t)
        else:
            return
        if self._preview is None:
            self._preview = pg.PlotDataItem(pen=DRAW_PEN)
            self.plot.addItem(self._preview)
        self._preview.setData(np.asarray(xs, float), np.asarray(ys, float))

    def create_gate(self, mode, pts, name=None):
        """Make a gate from drawn points (display units). Asks for a name unless given;
        quadrants and bisectors are named automatically. Returns the new population names."""
        self._clear_preview()
        self.start_gate("pointer")
        sid = self.sample_id
        if sid is None:
            return []
        xch, ych = self.x_combo.currentData(), self.y_combo.currentData()
        path = self.population_path
        taken = {n for n, _ in self.ws.children(path)}
        xl, yl = self.x_combo.currentText(), self.y_combo.currentText()

        def dim(ch, lo=None, hi=None):
            return self.ws.make_dimension(sid, ch, lo, hi)

        if mode == "quadrant":
            sx, sy = pts[0]
            specs = [(f"Q1: {xl}-, {yl}+", (None, sx), (sy, None)),
                     (f"Q2: {xl}+, {yl}+", (sx, None), (sy, None)),
                     (f"Q3: {xl}+, {yl}-", (sx, None), (None, sy)),
                     (f"Q4: {xl}-, {yl}-", (None, sx), (None, sy))]
            gates = [fk.gates.RectangleGate(_unique(n, taken), [dim(xch, *bx), dim(ych, *by)])
                     for n, bx, by in specs]
        elif mode == "bisector":
            s = pts[0][0]
            gates = [fk.gates.RectangleGate(_unique(f"{xl}-", taken), [dim(xch, None, s)]),
                     fk.gates.RectangleGate(_unique(f"{xl}+", taken), [dim(xch, s, None)])]
        else:
            if name is None:
                name, ok = QtWidgets.QInputDialog.getText(self, "New gate", "Population name:",
                                                          text=_unique("Gate", taken, always=True))
                if not ok or not name.strip():
                    self.refresh()
                    return []
            name = name.strip()
            (ax, ay), (bx, by) = pts[0], pts[-1]
            if mode == "range":
                gate = fk.gates.RectangleGate(name, [dim(xch, min(ax, bx), max(ax, bx))])
            elif mode == "rectangle":
                gate = fk.gates.RectangleGate(name, [dim(xch, min(ax, bx), max(ax, bx)),
                                                     dim(ych, min(ay, by), max(ay, by))])
            elif mode == "ellipse":
                mean, cov = ellipse_gate_params(((ax + bx) / 2, (ay + by) / 2),
                                                abs(bx - ax) / 2, abs(by - ay) / 2, 0.0)
                gate = fk.gates.EllipsoidGate(name, [dim(xch), dim(ych)], mean, cov, 1.0)
            else:  # polygon, freehand
                verts = _simplify(pts) if mode == "freehand" else pts
                gate = fk.gates.PolygonGate(name, [dim(xch), dim(ych)], [list(p) for p in verts])
            gates = [gate]
        made = []
        try:
            for gate in gates:
                self.ws.add_gate(gate, path)
                made.append(gate.gate_name)
        except Exception as exc:  # duplicate name and similar user errors
            QtWidgets.QMessageBox.warning(self, "Couldn't add gate", str(exc))
        self.gatesChanged.emit()
        self.refresh()
        return made

    def export_image(self, path):
        import pyqtgraph.exporters as exporters

        item = self.plot.getPlotItem()
        exporter = exporters.SVGExporter(item) if str(path).lower().endswith(".svg") \
            else exporters.ImageExporter(item)
        if not str(path).lower().endswith(".svg"):
            exporter.parameters()["width"] = 1200
        exporter.export(str(path))


class GraphWindow(QtWidgets.QWidget):
    """A floating graph window, opened by double-clicking a population (as in FlowJo)."""

    closed = QtCore.Signal(object)

    def __init__(self, workspace, parent=None):
        super().__init__(parent, QtCore.Qt.Window)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.view = GraphView(workspace, self)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.view)
        self.resize(720, 760)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        main = self.parent()
        if main is not None and hasattr(main, "dragEnterEvent"):
            main.dragEnterEvent(event)

    def dragMoveEvent(self, event):
        self.dragEnterEvent(event)

    def dropEvent(self, event):
        main = self.parent()
        if main is not None and hasattr(main, "dropEvent"):
            main.dropEvent(event)

    def closeEvent(self, event):
        self.closed.emit(self)
        super().closeEvent(event)


class VerticalComboBox(QtWidgets.QComboBox):
    """A parameter picker that reads bottom-to-top along the Y axis, as in FlowJo."""

    def sizeHint(self):
        s = super().sizeHint()
        return QtCore.QSize(s.height(), s.width())

    def minimumSizeHint(self):
        s = super().minimumSizeHint()
        return QtCore.QSize(s.height(), s.width())

    def paintEvent(self, event):
        p = QtWidgets.QStylePainter(self)
        p.translate(0, self.height())
        p.rotate(-90)
        opt = QtWidgets.QStyleOptionComboBox()
        self.initStyleOption(opt)
        opt.rect = QtCore.QRect(0, 0, self.height(), self.width())
        p.drawComplexControl(QtWidgets.QStyle.CC_ComboBox, opt)
        p.drawControl(QtWidgets.QStyle.CE_ComboBoxLabel, opt)


# ----- gate geometry (display units) -----------------------------------------

def ellipse_gate_params(centre, a, b, angle):
    """Mean and covariance (distance_square = 1) for an ellipse with semi-axes a, b."""
    a, b = max(a, 1e-4), max(b, 1e-4)
    c, s = math.cos(angle), math.sin(angle)
    rot = np.array([[c, -s], [s, c]])
    cov = rot @ np.diag([a * a, b * b]) @ rot.T
    return [float(centre[0]), float(centre[1])], cov.tolist()


def ellipse_shape(gate):
    """(centre, semi-axis a, semi-axis b, angle in radians) of an EllipsoidGate."""
    cov = np.asarray(gate.covariance_matrix, float) * float(gate.distance_square)
    vals, vecs = np.linalg.eigh(cov)
    a, b = math.sqrt(max(vals[1], 0)), math.sqrt(max(vals[0], 0))
    angle = math.atan2(vecs[1, 1], vecs[0, 1])
    cx, cy = (float(v) for v in gate.coordinates)
    return (cx, cy), a, b, angle


def _contains(gate, x, y):
    if isinstance(gate, fk.gates.RectangleGate):
        (x0, x1), (y0, y1) = (_open_bounds(d) for d in gate.dimensions)
        return x0 <= x <= x1 and y0 <= y <= y1
    if isinstance(gate, fk.gates.PolygonGate):
        poly = QtGui.QPolygonF([QtCore.QPointF(float(vx), float(vy)) for vx, vy in gate.vertices])
        return poly.containsPoint(QtCore.QPointF(x, y), QtCore.Qt.OddEvenFill)
    if isinstance(gate, fk.gates.EllipsoidGate):
        d = np.array([x, y]) - np.asarray(gate.coordinates, float)
        inv = np.linalg.inv(np.asarray(gate.covariance_matrix, float))
        return float(d @ inv @ d) <= float(gate.distance_square)
    return False


def _area(gate):
    if isinstance(gate, fk.gates.RectangleGate):
        (x0, x1), (y0, y1) = (_open_bounds(d) for d in gate.dimensions)
        return (min(x1, 1) - max(x0, 0)) * (min(y1, 1) - max(y0, 0))
    if isinstance(gate, fk.gates.PolygonGate):
        v = np.asarray(gate.vertices, float)
        return 0.5 * abs(np.dot(v[:, 0], np.roll(v[:, 1], 1)) - np.dot(v[:, 1], np.roll(v[:, 0], 1)))
    if isinstance(gate, fk.gates.EllipsoidGate):
        _, a, b, _ = ellipse_shape(gate)
        return math.pi * a * b
    return 1.0


def _simplify(pts, min_step=0.012):
    """Thin a traced freehand path so the polygon stays easy to edit."""
    out = [pts[0]]
    for p in pts[1:]:
        if math.dist(p, out[-1]) >= min_step:
            out.append(p)
    return out if len(out) >= 3 else pts


def _bounds(dim):
    lo = 0.0 if dim.min is None else float(dim.min)
    hi = 1.0 if dim.max is None else float(dim.max)
    return lo, hi


def _open_bounds(dim):
    lo = -np.inf if dim.min is None else float(dim.min)
    hi = np.inf if dim.max is None else float(dim.max)
    return lo, hi


def _half_open(dim):
    """The split value of a dimension bounded on exactly one side, else None."""
    if (dim.min is None) == (dim.max is None):
        return None
    return float(dim.min if dim.max is None else dim.max)


def _move_split(dim, value):
    if dim.max is None:
        dim.min = value
    else:
        dim.max = value


def _unique(name, taken, always=False):
    if not always and name not in taken:
        taken.add(name)
        return name
    n = 1 if always else 2
    while f"{name} {n}" in taken if always else f"{name} ({n})" in taken:
        n += 1
    out = f"{name} {n}" if always else f"{name} ({n})"
    taken.add(out)
    return out


def _first(names, preferred, fallback):
    if preferred in names:
        return preferred
    return fallback[0] if fallback else names[0]
