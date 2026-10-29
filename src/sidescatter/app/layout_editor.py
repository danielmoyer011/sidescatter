"""The Layout Editor: arrange plots and text on pages, batch them across a group, export.

Modelled on FlowJo 10's Layout Editor: a list of layouts on the left, the page
canvas in the middle (drag populations onto it from the workspace), and the
selected item's settings on the right. Plots stay linked to the workspace and
redraw when gates change.
"""

import json

from PySide6 import QtCore, QtGui, QtWidgets

from . import icons, theme
from .gating import service as gating
from .layout_render import PlotCache, default_channels, export_job, paint_item, print_layout
from .ribbon import Ribbon
from .workspace_panel import POPULATION_MIME
from ..core.layout import (GRID, MARGIN, PAGE_SIZES, PLOT_TYPES, TEXT_FIELDS, Layout, LayoutItem,
                           batch, new_id, snap, unique_name)
from ..core.workspace import ALL_SAMPLES, ROOT

MIN_SIZE = 36.0
PAGE_PAD = 40.0  # grey border around the pages in the editor
EXPORT_FILTER = "PDF document (*.pdf);;SVG vector image (*.svg);;PNG image (*.png);;TIFF image (*.tif)"


class SnapshotCommand(QtGui.QUndoCommand):
    """Undo step holding the layout before and after an edit."""

    def __init__(self, editor, index, before, after, label, merge_key=None):
        super().__init__(label)
        self.editor, self.index = editor, index
        self.before, self.after = before, after
        self.merge_key = merge_key
        self._first = True

    def id(self):
        return 1 if self.merge_key else -1

    def mergeWith(self, other):
        if other.merge_key != self.merge_key or other.index != self.index:
            return False
        self.after = other.after
        return True

    def redo(self):
        if self._first:  # the edit has already been made
            self._first = False
            return
        self.editor.restore(self.index, self.after)

    def undo(self):
        self.editor.restore(self.index, self.before)


class ItemGraphic(QtWidgets.QGraphicsObject):
    """One plot or text box on the canvas: drag to move, drag the corner to resize."""

    HANDLE = 8.0

    def __init__(self, editor, item):
        super().__init__()
        self.editor, self.item = editor, item
        self._resize_from = None
        self.setFlags(QtWidgets.QGraphicsItem.ItemIsMovable | QtWidgets.QGraphicsItem.ItemIsSelectable
                      | QtWidgets.QGraphicsItem.ItemSendsGeometryChanges)
        self.setAcceptHoverEvents(True)
        self.setPos(item.x, item.y)

    def boundingRect(self):
        h = self.HANDLE
        return QtCore.QRectF(-1, -1, self.item.w + h / 2 + 2, self.item.h + h / 2 + 2)

    def _handle_rect(self):
        h = self.HANDLE
        return QtCore.QRectF(self.item.w - h / 2, self.item.h - h / 2, h, h)

    def paint(self, p, option, widget=None):
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.save()
        paint_item(p, self.editor.ws, self.item, self.editor.cache)
        p.restore()
        rect = QtCore.QRectF(0, 0, self.item.w, self.item.h)
        if self.isSelected():
            pen = QtGui.QPen(QtGui.QColor(theme.ACCENT), 0)
            pen.setStyle(QtCore.Qt.DashLine)
            p.setPen(pen)
            p.setBrush(QtCore.Qt.NoBrush)
            p.drawRect(rect)
            p.setPen(QtGui.QPen(QtGui.QColor(theme.ACCENT), 0))
            p.setBrush(QtGui.QColor(theme.SURFACE))
            p.drawRect(self._handle_rect())
        elif self.item.kind == "text" and not self.item.text.strip():
            pen = QtGui.QPen(QtGui.QColor(theme.FAINT), 0, QtCore.Qt.DotLine)
            p.setPen(pen)
            p.setBrush(QtCore.Qt.NoBrush)
            p.drawRect(rect)

    def itemChange(self, change, value):
        if change == QtWidgets.QGraphicsItem.ItemPositionChange and self._resize_from is None:
            x, y = max(0.0, value.x()), max(0.0, value.y())
            if self.editor.grid_on:
                x, y = snap(x), snap(y)
            return QtCore.QPointF(x, y)
        if change == QtWidgets.QGraphicsItem.ItemSelectedHasChanged:
            self.editor.selection_changed()
        return super().itemChange(change, value)

    def hoverMoveEvent(self, event):
        on_handle = self.isSelected() and self._handle_rect().contains(event.pos())
        self.setCursor(QtCore.Qt.SizeFDiagCursor if on_handle else QtCore.Qt.SizeAllCursor)
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self.isSelected() \
                and self._handle_rect().contains(event.pos()):
            self._resize_from = (event.pos(), self.item.w, self.item.h)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._resize_from is None:
            super().mouseMoveEvent(event)
            return
        start, w0, h0 = self._resize_from
        w = w0 + event.pos().x() - start.x()
        h = h0 + event.pos().y() - start.y()
        if self.editor.grid_on:
            w, h = snap(self.item.x + w) - self.item.x, snap(self.item.y + h) - self.item.y
        self.prepareGeometryChange()
        self.item.w, self.item.h = max(MIN_SIZE, w), max(MIN_SIZE, h)
        self.update()

    def mouseReleaseEvent(self, event):
        if self._resize_from is not None:
            self._resize_from = None
            self.editor.commit("Resize")
            self.editor.inspector.load()
            return
        super().mouseReleaseEvent(event)
        self.editor.positions_changed()

    def mouseDoubleClickEvent(self, event):
        if self.item.kind == "plot":
            self.editor.open_graph(self.item)
        else:
            self.editor.inspector.focus_text()

    def sync(self):
        """Match the model after a change made elsewhere (inspector, align, undo)."""
        self.prepareGeometryChange()
        self.setPos(self.item.x, self.item.y)
        self.update()


class LayoutScene(QtWidgets.QGraphicsScene):
    """Draws the pages, margins and grid behind the items."""

    def __init__(self, editor):
        super().__init__()
        self.editor = editor

    def drawBackground(self, p, rect):
        p.fillRect(rect, QtGui.QColor(theme.CANVAS))
        lay = self.editor.layout
        if lay is None:
            return
        pw, ph = lay.page_width, lay.page_height
        pages = self.editor.shown_pages()
        p.setRenderHint(QtGui.QPainter.Antialiasing, False)
        p.fillRect(QtCore.QRectF(3, 3, pw, ph * pages), theme.qcolor(theme.INK, 25))  # shadow
        p.fillRect(QtCore.QRectF(0, 0, pw, ph * pages), QtGui.QColor("#FFFFFF"))
        if self.editor.grid_on:
            p.setPen(QtGui.QPen(QtGui.QColor("#EEF0F6"), 0))
            step = GRID * 2
            x = step
            while x < pw:
                p.drawLine(QtCore.QPointF(x, 0), QtCore.QPointF(x, ph * pages))
                x += step
            y = step
            while y < ph * pages:
                p.drawLine(QtCore.QPointF(0, y), QtCore.QPointF(pw, y))
                y += step
        margin_pen = QtGui.QPen(theme.qcolor(theme.ACCENT, 90), 0, QtCore.Qt.DashLine)
        for k in range(pages):
            p.setPen(margin_pen)
            p.setBrush(QtCore.Qt.NoBrush)
            p.drawRect(QtCore.QRectF(MARGIN, k * ph + MARGIN, pw - 2 * MARGIN, ph - 2 * MARGIN))
            if k:
                p.setPen(QtGui.QPen(QtGui.QColor(theme.CORAL), 0, QtCore.Qt.DashLine))
                p.drawLine(QtCore.QPointF(-12, k * ph), QtCore.QPointF(pw + 12, k * ph))
            p.setPen(QtGui.QColor(theme.MUTED))
            font = QtGui.QFont()
            font.setPixelSize(9)
            p.setFont(font)
            p.drawText(QtCore.QRectF(pw + 8, k * ph, PAGE_PAD, 14), QtCore.Qt.AlignLeft,
                       f"p. {k + 1}")
        p.setPen(QtGui.QPen(QtGui.QColor(theme.BORDER_STRONG), 0))
        p.setBrush(QtCore.Qt.NoBrush)
        p.drawRect(QtCore.QRectF(0, 0, pw, ph * pages))


class LayoutView(QtWidgets.QGraphicsView):
    def __init__(self, editor):
        super().__init__(editor.scene)
        self.editor = editor
        self.setAcceptDrops(True)
        self.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.TextAntialiasing)
        self.setDragMode(QtWidgets.QGraphicsView.RubberBandDrag)
        self.setViewportUpdateMode(QtWidgets.QGraphicsView.FullViewportUpdate)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)

    def wheelEvent(self, event):
        if event.modifiers() & QtCore.Qt.ControlModifier:
            self.editor.zoom_by(1.15 if event.angleDelta().y() > 0 else 1 / 1.15)
            return
        super().wheelEvent(event)

    def mousePressEvent(self, event):
        if self.editor.tool == "text" and event.button() == QtCore.Qt.LeftButton:
            pos = self.mapToScene(event.position().toPoint())
            self.editor.add_text(pos.x(), pos.y())
            return
        super().mousePressEvent(event)

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat(POPULATION_MIME):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        self.dragEnterEvent(event)

    def dropEvent(self, event):
        mime = event.mimeData()
        if not mime.hasFormat(POPULATION_MIME):
            event.ignore()
            return
        pops = json.loads(bytes(mime.data(POPULATION_MIME)).decode())
        pos = self.mapToScene(event.position().toPoint())
        event.acceptProposedAction()
        self.editor.add_plots(pops, pos.x(), pos.y())


class Inspector(QtWidgets.QWidget):
    """Settings for the selected plot or text box."""

    def __init__(self, editor):
        super().__init__()
        self.editor = editor
        self.setObjectName("LayoutInspector")
        self.setMinimumWidth(270)
        self._loading = False
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.stack = QtWidgets.QStackedWidget()

        empty = QtWidgets.QLabel("Drag populations from the workspace onto the page, or pick "
                                 "Text to add a caption.\n\nClick an item to change it here.")
        empty.setWordWrap(True)
        empty.setAlignment(QtCore.Qt.AlignTop)
        empty.setObjectName("GraphHint")
        empty.setContentsMargins(12, 12, 12, 12)
        self.stack.addWidget(empty)
        self.many = QtWidgets.QLabel()
        self.many.setObjectName("GraphHint")
        self.many.setWordWrap(True)
        self.many.setAlignment(QtCore.Qt.AlignTop)
        self.many.setContentsMargins(12, 12, 12, 12)
        self.stack.addWidget(self.many)

        # ----- shared fields --------------------------------------------------
        self.sample = QtWidgets.QComboBox()
        self.population = QtWidgets.QComboBox()
        self.sample.currentIndexChanged.connect(self._sample_changed)
        self.population.currentIndexChanged.connect(lambda: self._set("population", self.population.currentData()))
        self.font_size = QtWidgets.QDoubleSpinBox()
        self.font_size.setRange(4, 48)
        self.font_size.setSingleStep(0.5)
        self.font_size.setSuffix(" pt")
        self.font_size.valueChanged.connect(lambda v: self._set("font_size", float(v)))
        for combo in (self.sample, self.population):
            _compact(combo)
        self.geometry = {}
        for key in ("x", "y", "w", "h"):
            box = QtWidgets.QDoubleSpinBox()
            box.setRange(0.25 if key in "wh" else 0, 200)
            box.setDecimals(2)
            box.setSingleStep(0.125)
            box.setSuffix(" in")
            box.setMinimumWidth(60)
            box.valueChanged.connect(lambda v, k=key: self._set_geometry(k, v))
            self.geometry[key] = box

        # ----- plot page ------------------------------------------------------
        self.x_channel = QtWidgets.QComboBox()
        self.y_channel = QtWidgets.QComboBox()
        self.x_channel.currentIndexChanged.connect(lambda: self._set("x_channel", self.x_channel.currentData()))
        self.y_channel.currentIndexChanged.connect(lambda: self._set("y_channel", self.y_channel.currentData()))
        _compact(self.x_channel)
        _compact(self.y_channel)
        self.plot_type = QtWidgets.QComboBox()
        self.plot_type.addItems(PLOT_TYPES)
        self.plot_type.currentTextChanged.connect(lambda t: self._set("plot_type", t))
        self.title = QtWidgets.QLineEdit()
        self.title.setPlaceholderText("No title")
        self.title.textEdited.connect(lambda t: self._set("title", t))
        self.checks = {}
        for key, label in (("show_gates", "Gates"), ("show_gate_labels", "Gate names and %"),
                           ("show_axis_labels", "Axis titles")):
            cb = QtWidgets.QCheckBox(label)
            cb.toggled.connect(lambda on, k=key: self._set(k, on))
            self.checks[key] = cb

        # ----- text page ------------------------------------------------------
        self.text = QtWidgets.QPlainTextEdit()
        self.text.setMinimumHeight(90)
        self.text.textChanged.connect(lambda: self._set("text", self.text.toPlainText()))
        self.bold = QtWidgets.QCheckBox("Bold")
        self.italic = QtWidgets.QCheckBox("Italic")
        self.bold.toggled.connect(lambda on: self._set("bold", on))
        self.italic.toggled.connect(lambda on: self._set("italic", on))

        self.form_page = QtWidgets.QWidget()
        self.form = QtWidgets.QFormLayout(self.form_page)
        self.form.setContentsMargins(12, 10, 12, 10)
        self.form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        self.heading = QtWidgets.QLabel()
        self.heading.setStyleSheet("font-weight: 600; font-size: 11pt;")
        self.form.addRow(self.heading)
        self.rows = {}

        def row(key, label, widget):
            self.form.addRow(label, widget)
            self.rows[key] = widget

        row("sample", "Sample", self.sample)
        row("population", "Population", self.population)
        row("x_channel", "X axis", self.x_channel)
        row("y_channel", "Y axis", self.y_channel)
        row("plot_type", "Plot type", self.plot_type)
        row("title", "Title", self.title)
        row("text", "Text", self.text)
        styles = QtWidgets.QWidget()
        sl = QtWidgets.QHBoxLayout(styles)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.addWidget(self.bold)
        sl.addWidget(self.italic)
        sl.addStretch()
        row("style", "", styles)
        row("font_size", "Font size", self.font_size)
        shows = QtWidgets.QWidget()
        cl = QtWidgets.QVBoxLayout(shows)
        cl.setContentsMargins(0, 0, 0, 0)
        for cb in self.checks.values():
            cl.addWidget(cb)
        row("shows", "Show", shows)
        self.fields_hint = QtWidgets.QLabel(
            "Fields: " + "  ".join(TEXT_FIELDS) + "\nfilled from the sample and population above.")
        self.fields_hint.setObjectName("GraphHint")
        self.fields_hint.setWordWrap(True)
        self.form.addRow(self.fields_hint)
        geo = QtWidgets.QWidget()
        gl = QtWidgets.QGridLayout(geo)
        gl.setContentsMargins(0, 0, 0, 0)
        gl.setHorizontalSpacing(4)
        for i, (key, label) in enumerate((("x", "X"), ("y", "Y"), ("w", "W"), ("h", "H"))):
            gl.addWidget(QtWidgets.QLabel(label), i // 2, (i % 2) * 2)
            gl.addWidget(self.geometry[key], i // 2, (i % 2) * 2 + 1)
        gl.setColumnStretch(1, 1)
        gl.setColumnStretch(3, 1)
        self.form.addRow("Position", geo)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setWidget(self.form_page)
        self.stack.addWidget(scroll)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.stack)

    @property
    def item(self):
        sel = self.editor.selected_items()
        return sel[0] if len(sel) == 1 else None

    def _show_row(self, key, on):
        w = self.rows[key]
        w.setVisible(on)
        label = self.form.labelForField(w)
        if label is not None:
            label.setVisible(on)

    def load(self):
        sel = self.editor.selected_items()
        if not sel:
            self.stack.setCurrentIndex(0)
            return
        if len(sel) > 1:
            self.many.setText(f"{len(sel)} items selected.\n\nUse Arrange to line them up, "
                              "space them evenly or make them the same size.")
            self.stack.setCurrentIndex(1)
            return
        it = sel[0]
        plot = it.kind == "plot"
        ws = self.editor.ws
        self._loading = True
        self.heading.setText("Plot" if plot else "Text box")
        for key in ("x_channel", "y_channel", "plot_type", "title", "shows"):
            self._show_row(key, plot)
        for key in ("text", "style"):
            self._show_row(key, not plot)
        self.fields_hint.setVisible(not plot)

        self.sample.clear()
        if not plot:
            self.sample.addItem("(none)", None)
        for sid in ws.samples:
            self.sample.addItem(sid, sid)
        if it.sample_id and it.sample_id not in ws.samples:
            self.sample.addItem(f"{it.sample_id} (missing)", it.sample_id)
        self.sample.setCurrentIndex(max(0, self.sample.findData(it.sample_id)))
        self._fill_sample_fields(it)
        self.plot_type.setCurrentText(it.plot_type)
        self.title.setText(it.title)
        for key, cb in self.checks.items():
            cb.setChecked(getattr(it, key))
        if self.text.toPlainText() != it.text:
            self.text.setPlainText(it.text)
        self.bold.setChecked(it.bold)
        self.italic.setChecked(it.italic)
        self.font_size.setValue(it.font_size)
        for key, box in self.geometry.items():
            box.setValue(getattr(it, key) / 72.0)
        self._loading = False
        self.stack.setCurrentIndex(2)

    def _fill_sample_fields(self, it):
        ws = self.editor.ws
        self.population.clear()
        self.x_channel.clear()
        self.y_channel.clear()
        sid = it.sample_id
        if sid not in ws.samples:
            self.population.addItem("/".join(it.population[1:]) or "All events", list(it.population))
            return
        self.population.addItem("All events", ["root"])
        for name, parent in ws.population_paths():
            if ws.applies_to(sid, name, parent):
                self.population.addItem("    " * (len(parent) - 1) + name, list(parent) + [name])
        i = self.population.findData(list(it.population))
        if i < 0:
            self.population.addItem("/".join(it.population[1:]) + " (missing)", list(it.population))
            i = self.population.count() - 1
        self.population.setCurrentIndex(i)
        if it.kind == "plot":
            dx, dy = default_channels(ws, sid)
            for c in ws.channels(sid):
                self.x_channel.addItem(c.label, c.pnn)
                self.y_channel.addItem(c.label, c.pnn)
            self.x_channel.setCurrentIndex(max(0, self.x_channel.findData(it.x_channel or dx)))
            self.y_channel.setCurrentIndex(max(0, self.y_channel.findData(it.y_channel or dy)))

    def _sample_changed(self):
        it = self.item
        if self._loading or it is None:
            return
        sid = self.sample.currentData()
        if sid == it.sample_id:
            return
        it.sample_id = sid
        if sid is not None and sid in self.editor.ws.samples:
            name = it.population[-1] if len(it.population) > 1 else None
            if name and not self.editor.ws.applies_to(sid, name, tuple(it.population[:-1])):
                it.population = ["root"]
        self._loading = True
        self._fill_sample_fields(it)
        self._loading = False
        self.editor.item_edited(it, "sample")

    def _set(self, key, value):
        it = self.item
        if self._loading or it is None or value is None or getattr(it, key) == value:
            return
        setattr(it, key, value)
        self.editor.item_edited(it, key)

    def _set_geometry(self, key, inches):
        it = self.item
        if self._loading or it is None:
            return
        value = inches * 72.0
        if abs(getattr(it, key) - value) < 1e-6:
            return
        setattr(it, key, value)
        self.editor.item_edited(it, key)

    def focus_text(self):
        self.text.setFocus()
        self.text.selectAll()


class LayoutWindow(QtWidgets.QWidget):
    """The Layout Editor window."""

    changed = QtCore.Signal()                       # a layout was edited (workspace is dirty)
    graphRequested = QtCore.Signal(str, object, object)  # sample_id, name, parent_path

    def __init__(self, workspace, parent=None, current_population=None, current_group=None):
        super().__init__(parent, QtCore.Qt.Window)
        self.setWindowTitle("Layout Editor — Sidescatter")
        self.ws = workspace
        self.current_population = current_population or (lambda: None)  # () -> (sid, name, parent)
        self.current_group = current_group or (lambda: ALL_SAMPLES)
        self.axes_for = lambda sid, name, parent: None  # main window fills this in
        self.cache = PlotCache(lazy=True)
        self._repaint_timer = QtCore.QTimer(self, singleShot=True, interval=60)
        self._repaint_timer.timeout.connect(lambda: self.scene.update())
        gating().ready.connect(self._results_ready)
        self.index = None
        self.graphics = {}
        self.tool = "select"
        self.grid_on = True
        self.zoom = 1.0
        self._before = None
        self._order = []  # selection order, for "same size"
        self.undo = QtGui.QUndoStack(self)

        self.scene = LayoutScene(self)
        self.scene.selectionChanged.connect(self.selection_changed)
        self.view = LayoutView(self)
        self.inspector = Inspector(self)

        self.list = QtWidgets.QListWidget()
        self.list.setObjectName("LayoutList")
        self.list.setMinimumWidth(170)
        self.list.currentRowChanged.connect(self.show_layout)
        self.list.itemChanged.connect(self._renamed)
        list_box = QtWidgets.QWidget()
        lb = QtWidgets.QVBoxLayout(list_box)
        lb.setContentsMargins(0, 0, 0, 0)
        lb.setSpacing(0)
        caption = QtWidgets.QLabel("Layouts")
        caption.setObjectName("PanelCaption")
        caption.setContentsMargins(10, 6, 10, 6)
        lb.addWidget(caption)
        lb.addWidget(self.list)

        split = QtWidgets.QSplitter()
        split.addWidget(list_box)
        split.addWidget(self.view)
        split.addWidget(self.inspector)
        split.setStretchFactor(1, 1)
        split.setSizes([180, 900, 290])
        split.setChildrenCollapsible(False)

        self._build_ribbon()
        self.status = QtWidgets.QLabel()
        self.status.setObjectName("GraphHint")
        self.status.setContentsMargins(10, 4, 10, 4)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.ribbon)
        layout.addWidget(split, 1)
        layout.addWidget(self.status)
        self.resize(1360, 900)

        self.rebuild_list()
        if not self.ws.layouts:
            self.new_layout()
        else:
            self.list.setCurrentRow(0)

    # ----- ribbon ------------------------------------------------------------

    def _action(self, text, icon, slot, shortcut=None, checkable=False, tip=None):
        a = QtGui.QAction(icons.icon(icon), text.replace("\n", " "), self)
        a.setIconText(text)
        a.setToolTip(tip or text.replace("\n", " "))
        a.setCheckable(checkable)
        if shortcut:
            a.setShortcut(QtGui.QKeySequence(shortcut))
            a.setShortcutContext(QtCore.Qt.WindowShortcut)
        if checkable:
            a.toggled.connect(slot)
        else:
            a.triggered.connect(lambda checked=False: slot())
        self.addAction(a)
        return a

    def _build_ribbon(self):
        A = self._action
        undo = self.undo.createUndoAction(self, "Undo")
        undo.setIcon(icons.icon("undo"))
        undo.setShortcut(QtGui.QKeySequence.Undo)
        redo = self.undo.createRedoAction(self, "Redo")
        redo.setIcon(icons.icon("redo"))
        redo.setShortcut(QtGui.QKeySequence.Redo)
        for a in (undo, redo):
            a.setShortcutContext(QtCore.Qt.WindowShortcut)
            self.addAction(a)
        self.acts = acts = {
            "new": A("New\nLayout", "new_layout", self.new_layout, tip="New layout"),
            "duplicate_layout": A("Duplicate\nLayout", "duplicate", self.duplicate_layout),
            "delete_layout": A("Delete\nLayout", "delete", self.delete_layout),
            "select": A("Select", "pointer", lambda on: on and self.set_tool("select"), "Esc", True),
            "text": A("Text", "text", lambda on: on and self.set_tool("text"), "T", True,
                      tip="Text box: click on the page where it should go"),
            "add_plot": A("Add\nPlot", "add_plot", self.add_selected_population,
                          tip="Add the population selected in the workspace"),
            "batch": A("Batch…", "batch", self.batch_dialog,
                       tip="Repeat this layout for every sample in a group"),
            "page": A("Page\nSetup…", "page_setup", self.page_setup),
            "grid": A("Grid", "grid", self.set_grid, checkable=True, tip="Show the grid and snap to it"),
            "export": A("Export…", "export", self.export_dialog, "Ctrl+E"),
            "print": A("Print…", "print", self.print_dialog, QtGui.QKeySequence.Print),
            "align_left": A("Left", "align_left", lambda: self.align("left")),
            "align_right": A("Right", "align_right", lambda: self.align("right")),
            "align_top": A("Top", "align_top", lambda: self.align("top")),
            "align_bottom": A("Bottom", "align_bottom", lambda: self.align("bottom")),
            "dist_h": A("Across", "distribute_h", lambda: self.distribute("h"),
                        tip="Space evenly from left to right"),
            "dist_v": A("Down", "distribute_v", lambda: self.distribute("v"),
                        tip="Space evenly from top to bottom"),
            "same_size": A("Same\nSize", "same_size", self.same_size,
                           tip="Make the selection the size of the item picked first"),
            "front": A("Bring to\nFront", "bring_front", lambda: self.restack(True)),
            "back": A("Send to\nBack", "send_back", lambda: self.restack(False)),
            "duplicate": A("Duplicate", "duplicate", self.duplicate_items, "Ctrl+D"),
            "delete": A("Delete", "delete", self.delete_items, QtGui.QKeySequence.Delete),
            "zoom_in": A("Zoom\nIn", "zoom_in", lambda: self.zoom_by(1.25), QtGui.QKeySequence.ZoomIn),
            "zoom_out": A("Zoom\nOut", "zoom_out", lambda: self.zoom_by(0.8), QtGui.QKeySequence.ZoomOut),
            "zoom_fit": A("Fit\nPage", "zoom_fit", self.zoom_fit, "Ctrl+0"),
            "undo": undo, "redo": redo,
        }
        tools = QtGui.QActionGroup(self)
        for k in ("select", "text"):
            tools.addAction(acts[k])
        acts["select"].setChecked(True)
        acts["grid"].setChecked(True)
        backspace = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Backspace), self, self.delete_items)
        backspace.setContext(QtCore.Qt.WindowShortcut)

        self.ribbon = r = Ribbon()
        menu = QtWidgets.QMenu(self)
        for k in ("new", "export", "print", None):
            menu.addSeparator() if k is None else menu.addAction(acts[k])
        menu.addAction("Close", self.close)
        r.app_button.setMenu(menu)
        for k in ("undo", "redo", "export", "print"):
            r.add_quick(acts[k])

        page = r.add_tab("Layout")
        page.add_group("Layouts", [acts["new"], acts["duplicate_layout"], acts["delete_layout"]])
        page.add_group("Tools", [acts["select"], acts["text"], acts["add_plot"]])
        page.add_group("Batch", [acts["batch"]])
        page.add_group("Page", [acts["page"], acts["grid"]])
        page.add_group("Output", [acts["export"], acts["print"]])

        page = r.add_tab("Arrange")
        page.add_group("Align", [acts["align_left"], acts["align_right"], acts["align_top"],
                                 acts["align_bottom"]])
        page.add_group("Distribute", [acts["dist_h"], acts["dist_v"], acts["same_size"]])
        page.add_group("Order", [acts["front"], acts["back"]])
        page.add_group("Edit", [acts["undo"], acts["redo"], acts["duplicate"], acts["delete"]])

        page = r.add_tab("View")
        page.add_group("Zoom", [acts["zoom_in"], acts["zoom_out"], acts["zoom_fit"]])
        page.add_group("Page", [acts["grid"]])

    # ----- layouts -------------------------------------------------------------

    @property
    def layout(self):
        if self.index is None or self.index >= len(self.ws.layouts):
            return None
        return self.ws.layouts[self.index]

    def set_workspace(self, ws):
        self.ws = ws
        self.cache.clear()
        self.undo.clear()
        self.index = None
        self.rebuild_list()
        if not ws.layouts:
            self.new_layout()
        else:
            self.list.setCurrentRow(0)

    def rebuild_list(self):
        self.list.blockSignals(True)
        self.list.clear()
        for lay in self.ws.layouts:
            it = QtWidgets.QListWidgetItem(icons.icon("layout_editor", 18), lay.name)
            it.setFlags(it.flags() | QtCore.Qt.ItemIsEditable)
            self.list.addItem(it)
        if self.index is not None and self.index < self.list.count():
            self.list.setCurrentRow(self.index)
        self.list.blockSignals(False)

    def _renamed(self, list_item):
        row = self.list.row(list_item)
        name = list_item.text().strip()
        if 0 <= row < len(self.ws.layouts) and name and name != self.ws.layouts[row].name:
            self.ws.layouts[row].name = name
            self.changed.emit()

    def _add_layout(self, lay):
        self.ws.layouts.append(lay)
        self.undo.clear()
        self.rebuild_list()
        self.list.setCurrentRow(len(self.ws.layouts) - 1)
        self.changed.emit()

    def new_layout(self):
        self._add_layout(Layout(unique_name("Layout", self.ws.layouts)))

    def duplicate_layout(self):
        if self.layout is None:
            return
        copy_ = Layout.from_dict(self.layout.to_dict())
        for it in copy_.items:
            it.id = new_id()
        copy_.name = unique_name(self.layout.name, self.ws.layouts)
        self._add_layout(copy_)

    def delete_layout(self):
        if self.layout is None:
            return
        answer = QtWidgets.QMessageBox.question(self, "Delete layout", f"Delete {self.layout.name}?")
        if answer != QtWidgets.QMessageBox.Yes:
            return
        del self.ws.layouts[self.index]
        self.undo.clear()
        self.index = None
        self.rebuild_list()
        if not self.ws.layouts:
            self.new_layout()
        else:
            self.list.setCurrentRow(0)
        self.changed.emit()

    def show_layout(self, row):
        if row < 0 or row >= len(self.ws.layouts):
            return
        self.index = row
        self._before = self.layout.to_dict()
        self.rebuild_scene()
        self.zoom_fit()

    def shown_pages(self):
        """Pages drawn in the editor: those in use plus a blank one to drag onto."""
        lay = self.layout
        if lay is None:
            return 1
        used = lay.page_count()
        bottom = max((it.y + it.h for it in lay.items), default=0.0)
        return used + (1 if bottom > 0 else 0)

    def _update_scene_rect(self):
        lay = self.layout
        if lay is None:
            return
        self.scene.setSceneRect(-PAGE_PAD, -PAGE_PAD, lay.page_width + 2 * PAGE_PAD,
                                lay.page_height * self.shown_pages() + 2 * PAGE_PAD)
        self.scene.update()

    def rebuild_scene(self):
        selected = {it.id for it in self.selected_items()}
        self.scene.blockSignals(True)
        self.scene.clear()
        self.graphics = {}
        lay = self.layout
        if lay is not None:
            for z, item in enumerate(lay.items):
                g = ItemGraphic(self, item)
                g.setZValue(z)
                self.scene.addItem(g)
                self.graphics[item.id] = g
                if item.id in selected:
                    g.setSelected(True)
        self.scene.blockSignals(False)
        self._update_scene_rect()
        self.selection_changed()
        self._update_status()

    def _results_ready(self, ws, sample_id):
        if ws is self.ws:
            self._repaint_timer.start()  # plots that read "Calculating…" can draw now

    def refresh(self):
        """Redraw plots after the workspace changed (gates edited, samples added)."""
        self.cache.clear()
        if self.layout is not None:
            self.scene.update()
        self.inspector.load()

    def restore(self, index, state):
        """Put a layout back to a saved state (undo/redo)."""
        if index >= len(self.ws.layouts):
            return
        self.ws.layouts[index] = Layout.from_dict(state)
        if index != self.index:
            self.index = index
            self.list.blockSignals(True)
            self.list.setCurrentRow(index)
            self.list.blockSignals(False)
        self._before = state
        self.rebuild_list()
        self.rebuild_scene()
        self.inspector.load()
        self.changed.emit()

    def commit(self, label, merge_key=None):
        """Record the edit just made to the current layout as an undo step."""
        lay = self.layout
        if lay is None:
            return
        after = lay.to_dict()
        if after == self._before:
            return
        self.undo.push(SnapshotCommand(self, self.index, self._before, after, label, merge_key))
        self._before = after
        self._update_scene_rect()
        self._update_status()
        self.changed.emit()

    def _update_status(self):
        lay = self.layout
        if lay is None:
            return
        n = lay.page_count()
        orient = "landscape" if lay.landscape else "portrait"
        self.status.setText(f"{lay.name}: {len(lay.items)} item(s) on {n} page{'s' if n > 1 else ''}, "
                            f"{lay.page_size} {orient}. Drag populations from the workspace onto the "
                            "page; double-click a plot to open its graph window.")

    # ----- selection -------------------------------------------------------------

    def selected_items(self):
        if self.layout is None:
            return []
        try:
            graphics = self.scene.selectedItems()
        except RuntimeError:
            return []
        return [g.item for g in graphics if isinstance(g, ItemGraphic)]

    def selection_changed(self):
        ids = [it.id for it in self.selected_items()]
        self._order = [i for i in self._order if i in ids] + [i for i in ids if i not in self._order]
        self.inspector.load()

    def select(self, items):
        self.scene.clearSelection()
        for it in items:
            g = self.graphics.get(it.id)
            if g is not None:
                g.setSelected(True)

    def positions_changed(self):
        moved = False
        for item_id, g in self.graphics.items():
            pos = g.pos()
            if abs(g.item.x - pos.x()) > 1e-6 or abs(g.item.y - pos.y()) > 1e-6:
                g.item.x, g.item.y = pos.x(), pos.y()
                moved = True
        if moved:
            self.commit("Move")
            self.inspector.load()

    def item_edited(self, item, key):
        g = self.graphics.get(item.id)
        if g is not None:
            g.sync()
        self.commit(f"Change {key.replace('_', ' ')}", merge_key=(item.id, key))

    # ----- adding items ----------------------------------------------------------

    def set_tool(self, tool):
        self.tool = tool
        self.acts[tool].setChecked(True)
        self.view.setDragMode(QtWidgets.QGraphicsView.RubberBandDrag if tool == "select"
                              else QtWidgets.QGraphicsView.NoDrag)
        self.view.viewport().setCursor(QtCore.Qt.IBeamCursor if tool == "text" else QtCore.Qt.ArrowCursor)

    def _free_spot(self, w, h):
        """Somewhere on the pages that doesn't overlap existing items, scanning in grid steps."""
        lay = self.layout
        taken = [QtCore.QRectF(it.x, it.y, it.w, it.h) for it in lay.items]
        pw, ph = lay.page_width, lay.page_height
        if w > pw - 2 * MARGIN or h > ph - 2 * MARGIN:
            return MARGIN, lay.page_count() * ph + MARGIN
        page = 0
        while True:
            y = page * ph + MARGIN
            while y + h <= (page + 1) * ph - MARGIN + 1e-6:
                x = MARGIN
                while x + w <= pw - MARGIN + 1e-6:
                    r = QtCore.QRectF(x, y, w, h).adjusted(1, 1, -1, -1)
                    if not any(r.intersects(t) for t in taken):
                        return x, y
                    x += GRID * 2
                y += GRID * 2
            page += 1

    def add_plots(self, pops, x=None, y=None):
        """Add one plot per {sample_id, population} entry, in a row from (x, y)."""
        lay = self.layout
        if lay is None or not pops:
            return []
        size = 216.0  # 3 inches
        gap = GRID * 2
        made = []
        for i, pop in enumerate(pops):
            sid, path = pop["sample_id"], list(pop.get("population") or ["root"])
            item = LayoutItem("plot", w=size, h=size, sample_id=sid, population=path)
            name = path[-1] if len(path) > 1 else None
            axes = self.axes_for(sid, name, tuple(path[:-1]) if name else ROOT)
            if axes:
                item.x_channel, item.y_channel, item.plot_type = axes
            elif sid in self.ws.samples:
                item.x_channel, item.y_channel = default_channels(self.ws, sid)
            if x is None:
                item.x, item.y = self._free_spot(size, size)
            else:
                item.x = max(0.0, snap(x - size / 2 + i * (size + gap)))
                item.y = max(0.0, snap(y - size / 2))
            lay.items.append(item)
            made.append(item)
        self.rebuild_scene()
        self.select(made)
        self.commit(f"Add {len(made)} plot(s)")
        return made

    def add_selected_population(self):
        cur = self.current_population()
        if not cur:
            QtWidgets.QMessageBox.information(
                self, "Add plot", "Select a sample or population in the workspace first, "
                "or drag it onto the page.")
            return
        sid, name, parent = cur
        path = list(parent) + ([name] if name else [])
        self.add_plots([{"sample_id": sid, "population": path}])

    def add_text(self, x, y, text="Text"):
        lay = self.layout
        if lay is None:
            return None
        item = LayoutItem("text", x=max(0.0, snap(x)), y=max(0.0, snap(y)), w=180.0, h=36.0,
                          text=text, font_size=12.0, title="")
        lay.items.append(item)
        self.rebuild_scene()
        self.select([item])
        self.commit("Add text")
        self.set_tool("select")
        self.inspector.focus_text()
        return item

    def duplicate_items(self):
        sel = self.selected_items()
        if not sel:
            return
        made = []
        for it in sel:
            new = LayoutItem.from_dict({**it.to_dict(), "id": new_id()})
            new.x += GRID * 2
            new.y += GRID * 2
            self.layout.items.append(new)
            made.append(new)
        self.rebuild_scene()
        self.select(made)
        self.commit("Duplicate")

    def delete_items(self):
        typing = (QtWidgets.QLineEdit, QtWidgets.QPlainTextEdit, QtWidgets.QAbstractSpinBox)
        if isinstance(QtWidgets.QApplication.focusWidget(), typing):
            return  # Delete/Backspace while typing in the inspector
        sel = {it.id for it in self.selected_items()}
        if not sel:
            return
        self.layout.items = [it for it in self.layout.items if it.id not in sel]
        self.rebuild_scene()
        self.commit("Delete")

    # ----- arranging -------------------------------------------------------------

    def _sync_all(self):
        for g in self.graphics.values():
            g.sync()
        self.scene.update()

    def align(self, edge):
        sel = self.selected_items()
        if len(sel) < 2:
            return
        if edge == "left":
            v = min(it.x for it in sel)
            for it in sel:
                it.x = v
        elif edge == "right":
            v = max(it.x + it.w for it in sel)
            for it in sel:
                it.x = v - it.w
        elif edge == "top":
            v = min(it.y for it in sel)
            for it in sel:
                it.y = v
        else:
            v = max(it.y + it.h for it in sel)
            for it in sel:
                it.y = v - it.h
        self._sync_all()
        self.commit(f"Align {edge}")
        self.inspector.load()

    def distribute(self, axis):
        sel = self.selected_items()
        if len(sel) < 3:
            return
        pos, size = ("x", "w") if axis == "h" else ("y", "h")
        sel.sort(key=lambda it: getattr(it, pos))
        start = getattr(sel[0], pos)
        end = getattr(sel[-1], pos) + getattr(sel[-1], size)
        gap = (end - start - sum(getattr(it, size) for it in sel)) / (len(sel) - 1)
        cur = start
        for it in sel:
            setattr(it, pos, cur)
            cur += getattr(it, size) + gap
        self._sync_all()
        self.commit("Distribute")
        self.inspector.load()

    def same_size(self):
        sel = self.selected_items()
        if len(sel) < 2:
            return
        first = self.layout.item(self._order[0]) if self._order else sel[0]
        for it in sel:
            it.w, it.h = first.w, first.h
        self._sync_all()
        self.commit("Same size")
        self.inspector.load()

    def restack(self, front):
        sel = {it.id for it in self.selected_items()}
        if not sel:
            return
        rest = [it for it in self.layout.items if it.id not in sel]
        moved = [it for it in self.layout.items if it.id in sel]
        self.layout.items = rest + moved if front else moved + rest
        self.rebuild_scene()
        self.commit("Bring to front" if front else "Send to back")

    # ----- view ---------------------------------------------------------------------

    def set_grid(self, on):
        self.grid_on = bool(on)
        self.scene.update()

    def zoom_by(self, factor):
        self.zoom = min(8.0, max(0.15, self.zoom * factor))
        self.view.setTransform(QtGui.QTransform.fromScale(self.zoom, self.zoom))

    def zoom_fit(self):
        lay = self.layout
        if lay is None:
            return
        vp = self.view.viewport().size()
        if vp.width() < 50 or vp.height() < 50:
            vp = QtCore.QSize(900, 700)
        sx = vp.width() / (lay.page_width + 2 * PAGE_PAD)
        sy = vp.height() / (lay.page_height + 2 * PAGE_PAD)
        self.zoom = max(0.15, min(sx, sy, 4.0))
        self.view.setTransform(QtGui.QTransform.fromScale(self.zoom, self.zoom))
        self.view.centerOn(lay.page_width / 2, lay.page_height / 2)

    # ----- page, batch, output ------------------------------------------------------

    def page_setup(self):
        lay = self.layout
        if lay is None:
            return
        dlg = PageSetupDialog(lay, self)
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            lay.page_size, lay.landscape = dlg.values()
            self.rebuild_scene()
            self.commit("Page setup")
            self.zoom_fit()

    def batch_dialog(self):
        lay = self.layout
        if lay is None:
            return
        if not any(it.sample_id for it in lay.items):
            QtWidgets.QMessageBox.information(
                self, "Batch", "Add at least one plot first. Batch repeats this layout once per "
                "sample in a group.")
            return
        dlg = BatchDialog(self.ws, self.current_group(), self)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        group, direction, per_page = dlg.values()
        samples = [s for s in self.ws.groups.get(group, []) if s in self.ws.samples]
        if not samples:
            QtWidgets.QMessageBox.information(self, "Batch", f"{group} has no samples.")
            return
        name = unique_name(f"{lay.name} – {group}", self.ws.layouts)
        self._add_layout(batch(lay, samples, direction, per_page, name))
        self.status.setText(f"Made {name}: {len(samples)} sample(s) by {direction}.")

    def export_dialog(self):
        lay = self.layout
        if lay is None:
            return None
        path, chosen = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export layout", f"{lay.name}.pdf", EXPORT_FILTER)
        if not path:
            return None
        ext = {"PDF": ".pdf", "SVG": ".svg", "PNG": ".png", "TIFF": ".tif"}
        if "." not in path.rsplit("/", 1)[-1]:
            path += next((v for k, v in ext.items() if chosen.startswith(k)), ".pdf")
        dpi = 300
        if path.lower().endswith((".png", ".tif", ".tiff")):
            dpi, ok = QtWidgets.QInputDialog.getInt(self, "Export layout", "Resolution (DPI):",
                                                    300, 72, 1200, 50)
            if not ok:
                return None
        self.status.setText(f"Exporting {lay.name}… (see the status bar)")

        def done(files):
            self.status.setText(f"Exported {len(files)} file(s): " + ", ".join(f.name for f in files))

        return export_job(lay, self.ws, path, dpi=dpi, on_done=done)

    def print_dialog(self):
        from PySide6 import QtPrintSupport

        lay = self.layout
        if lay is None:
            return
        printer = QtPrintSupport.QPrinter(QtPrintSupport.QPrinter.HighResolution)
        printer.setDocName(lay.name)
        dlg = QtPrintSupport.QPrintDialog(printer, self)
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            print_layout(lay, self.ws, printer, self.cache)

    def open_graph(self, item):
        if item.sample_id not in self.ws.samples:
            return
        path = item.population_path
        name = path[-1] if len(path) > 1 else None
        self.graphRequested.emit(item.sample_id, name, tuple(path[:-1]) if name else ROOT)

    def showEvent(self, event):
        super().showEvent(event)
        QtCore.QTimer.singleShot(0, self.zoom_fit)


class PageSetupDialog(QtWidgets.QDialog):
    def __init__(self, layout, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Page setup")
        self.size = QtWidgets.QComboBox()
        self.size.addItems(list(PAGE_SIZES))
        self.size.setCurrentText(layout.page_size)
        self.portrait = QtWidgets.QRadioButton("Portrait")
        self.landscape = QtWidgets.QRadioButton("Landscape")
        (self.landscape if layout.landscape else self.portrait).setChecked(True)
        orient = QtWidgets.QHBoxLayout()
        orient.addWidget(self.portrait)
        orient.addWidget(self.landscape)
        form = QtWidgets.QFormLayout(self)
        form.addRow("Paper", self.size)
        form.addRow("Orientation", orient)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def values(self):
        return self.size.currentText(), self.landscape.isChecked()


class BatchDialog(QtWidgets.QDialog):
    """Pick the group to batch over, the direction, and how many samples per page."""

    def __init__(self, workspace, group, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch layout")
        self.group = QtWidgets.QComboBox()
        for name, members in workspace.groups.items():
            self.group.addItem(f"{name} ({len(members)})", name)
        self.group.setCurrentIndex(max(0, self.group.findData(group)))
        self.rows = QtWidgets.QRadioButton("Rows: one sample per row")
        self.columns = QtWidgets.QRadioButton("Columns: one sample per column")
        self.rows.setChecked(True)
        self.per_page = QtWidgets.QSpinBox()
        self.per_page.setRange(0, 100)
        self.per_page.setSpecialValueText("As many as fit")
        note = QtWidgets.QLabel("Every plot and text box is repeated for each sample in the group, "
                                "in a new layout. The plots stay linked to the gates.")
        note.setObjectName("GraphHint")
        note.setWordWrap(True)
        form = QtWidgets.QFormLayout(self)
        form.addRow(note)
        form.addRow("Group", self.group)
        form.addRow("Direction", self.rows)
        form.addRow("", self.columns)
        form.addRow("Samples per page", self.per_page)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def values(self):
        return (self.group.currentData(), "columns" if self.columns.isChecked() else "rows",
                self.per_page.value())


def _compact(combo):
    """Keep long sample names from pushing the inspector wider than the window."""
    combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
    combo.setMinimumContentsLength(12)
