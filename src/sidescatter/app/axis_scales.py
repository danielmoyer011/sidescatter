"""Axis scale controls: the small scale button beside each axis picker, its
menu, and the Transform window (laid out like FlowJo 10's)."""

import json

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from . import theme
from ..core import scales
from .axes import TransformAxis

PREVIEW_EVENTS = 100_000
SLIDER_STEPS = 1000
SETTINGS_KEY = "axisScales"


# ----- saved defaults --------------------------------------------------------

def _settings():
    return QtCore.QSettings("Sidescatter", "Sidescatter")


def load_saved_scales():
    """Fill scales.saved_defaults from the app settings."""
    try:
        saved = json.loads(_settings().value(SETTINGS_KEY, "{}") or "{}")
    except (TypeError, ValueError):
        saved = {}
    scales.saved_defaults.clear()
    scales.saved_defaults.update(saved)


def save_scale_default(channel, xform):
    """Use this scale for `channel` whenever it's new to a workspace (the top
    of the scale still comes from each file)."""
    kind, s = scales.settings(xform)
    s.pop("max", None)
    if kind not in scales.EDITABLE_MIN:
        s.pop("min", None)
    scales.saved_defaults[channel] = {"kind": kind, "settings": s}
    _settings().setValue(SETTINGS_KEY, json.dumps(scales.saved_defaults))


# ----- the button beside each axis picker ------------------------------------

class ScaleButton(QtWidgets.QAbstractButton):
    """A small tag showing the axis scale ("Biex", "Log"...); click for the menu.
    Vertical buttons read bottom-to-top, like the Y parameter picker."""

    def __init__(self, vertical=False, parent=None):
        super().__init__(parent)
        self.vertical = vertical
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setAttribute(QtCore.Qt.WA_Hover, True)
        f = self.font()
        f.setPointSizeF(8.5)
        f.setWeight(QtGui.QFont.DemiBold)
        self.setFont(f)

    def _text_size(self):
        fm = QtGui.QFontMetrics(self.font())
        return QtCore.QSize(fm.horizontalAdvance(self.text()) + 22, fm.height() + 6)

    def sizeHint(self):
        s = self._text_size()
        return s.transposed() if self.vertical else s

    def minimumSizeHint(self):
        return self.sizeHint()

    def enterEvent(self, event):
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        w, h = self.width(), self.height()
        if self.vertical:
            p.translate(0, h)
            p.rotate(-90)
            w, h = h, w
        hover = self.underMouse() and self.isEnabled()
        rect = QtCore.QRectF(0.5, 0.5, w - 1, h - 1)
        p.setPen(QtGui.QPen(QtGui.QColor(theme.ACCENT if hover else theme.BORDER_STRONG), 1))
        p.setBrush(QtGui.QColor(theme.ACCENT_SOFT if hover else theme.SURFACE))
        p.drawRoundedRect(rect, 4, 4)
        colour = theme.ACCENT_DARK if self.isEnabled() else theme.FAINT
        p.setPen(QtGui.QColor(colour))
        p.drawText(QtCore.QRectF(6, 0, w - 18, h), QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft,
                   self.text())
        # a small down-pointing caret, as on a drop-down
        cx, cy = w - 8.5, h / 2
        caret = QtGui.QPolygonF([QtCore.QPointF(cx - 3, cy - 1.5), QtCore.QPointF(cx + 3, cy - 1.5),
                                 QtCore.QPointF(cx, cy + 2)])
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(QtGui.QColor(colour))
        p.drawPolygon(caret)


def scale_menu(parent, ws, channel, on_kind, on_previous, on_customize):
    """The scale menu for one channel: every scale (current one ticked),
    Previous scale, Customize…"""
    menu = QtWidgets.QMenu(parent)
    current = scales.kind_of(ws.transform(channel))
    for kind, (label, _) in scales.KINDS.items():
        a = menu.addAction(label)
        a.setCheckable(True)
        a.setChecked(kind == current)
        a.triggered.connect(lambda _=False, k=kind: on_kind(channel, k))
    menu.addSeparator()
    prev = ws.previous_transform(channel)
    a = menu.addAction(f"Previous scale ({scales.label(prev)})" if prev is not None else "Previous scale")
    a.setEnabled(prev is not None)
    a.triggered.connect(lambda: on_previous(channel))
    menu.addSeparator()
    menu.addAction("Customize…").triggered.connect(lambda: on_customize(channel))
    return menu


# ----- Transform window --------------------------------------------------------

class _SliderBox(QtWidgets.QFrame):
    """One FlowJo-style transform setting: a slider with ticks, a label and a number box."""

    changed = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("SliderBox")
        self.setStyleSheet(f"#SliderBox {{ border: 1px solid {theme.BORDER}; border-radius: 4px; "
                           f"background: {theme.SURFACE}; }}")
        self.lo, self.hi = 0.0, 1.0
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, SLIDER_STEPS)
        self.slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.ends = QtWidgets.QLabel()
        self.ends.setStyleSheet(f"color: {theme.MUTED}; font-size: 8pt; border: none;")
        self.label = QtWidgets.QLabel()
        self.label.setStyleSheet("border: none;")
        self.box = QtWidgets.QDoubleSpinBox()
        self.box.setDecimals(2)
        self.box.setKeyboardTracking(False)
        self.box.setFixedWidth(72)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.label, 1)
        row.addWidget(self.box)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(2)
        lay.addWidget(self.slider)
        lay.addWidget(self.ends)
        lay.addLayout(row)
        self.slider.valueChanged.connect(self._from_slider)
        self.box.valueChanged.connect(self._from_box)
        self.key = None

    def configure(self, spec, value):
        self.blockSignals(True)
        if spec is None:
            self.key = None
            self.label.setText("")
            self.ends.setText("")
            self.box.clear()
            self.setEnabled(False)
            self.slider.setValue(0)
        else:
            self.key, text, self.lo, self.hi, step = spec
            self.label.setText(text)
            self.ends.setText(f"{self.lo:g}{' ' * 40}{self.hi:g}")
            self.slider.setTickInterval(SLIDER_STEPS // max(1, round(self.hi - self.lo)))
            self.box.setRange(self.lo, self.hi)
            self.box.setSingleStep(step)
            self.setEnabled(True)
            self.set_value(value)
        self.blockSignals(False)

    def set_value(self, value):
        value = float(np.clip(value, self.lo, self.hi))
        for w in (self.slider, self.box):
            w.blockSignals(True)
        self.box.setValue(value)
        self.slider.setValue(round((value - self.lo) / (self.hi - self.lo) * SLIDER_STEPS))
        for w in (self.slider, self.box):
            w.blockSignals(False)

    def value(self):
        return self.box.value()

    def _from_slider(self, i):
        self.box.blockSignals(True)
        self.box.setValue(self.lo + (self.hi - self.lo) * i / SLIDER_STEPS)
        self.box.blockSignals(False)
        self.changed.emit()

    def _from_box(self, v):
        self.slider.blockSignals(True)
        self.slider.setValue(round((v - self.lo) / (self.hi - self.lo) * SLIDER_STEPS))
        self.slider.blockSignals(False)
        self.changed.emit()


def _section(title):
    box = QtWidgets.QGroupBox(title)
    box.setStyleSheet(f"QGroupBox {{ font-weight: 600; border: 1px solid {theme.BORDER}; "
                      f"border-radius: 5px; margin-top: 10px; padding-top: 6px; }} "
                      f"QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; }}")
    return box


def _number_box():
    b = QtWidgets.QDoubleSpinBox()
    b.setRange(-1e12, 1e12)
    b.setDecimals(1)
    b.setKeyboardTracking(False)
    b.setGroupSeparatorShown(True)
    return b


class TransformWindow(QtWidgets.QDialog):
    """Customize a channel's scale with a live preview, as in FlowJo's Transform window.

    Changes stay pending (for any number of parameters) until Apply.
    """

    def __init__(self, ws, sample_id, mask=None, channel=None, parent=None):
        super().__init__(parent)
        self.ws, self.sample_id, self.mask = ws, sample_id, mask
        self.setWindowTitle(f"Transform of {sample_id}")
        self.pending = {}   # channel -> new transform
        self._values = {}   # channel -> raw values for the preview (subsampled)
        chans = ws.channels(sample_id)
        self.labels = {c.pnn: c.label for c in chans}
        self.ch = channel if channel in self.labels else chans[0].pnn

        # ----- preview -----
        self.axis = TransformAxis("bottom")
        self.preview = pg.PlotWidget(axisItems={"bottom": self.axis})
        self.preview.setBackground(theme.SURFACE)
        self.preview.setMenuEnabled(False)
        self.preview.setMouseEnabled(x=False, y=False)
        self.preview.getPlotItem().hideButtons()
        self.preview.setMinimumSize(420, 260)
        for ax in ("bottom", "left"):
            self.preview.getAxis(ax).setPen(pg.mkPen(theme.BORDER_STRONG))
            self.preview.getAxis(ax).setTextPen(pg.mkPen(theme.MUTED))
        self.channel_label = QtWidgets.QLabel()
        self.channel_label.setAlignment(QtCore.Qt.AlignCenter)
        self.nudges = {}
        ends = QtWidgets.QHBoxLayout()
        for end in ("min", "max"):
            pair = QtWidgets.QHBoxLayout()
            pair.setSpacing(2)
            for step, text in ((-1, "−"), (1, "+")):
                b = QtWidgets.QToolButton()
                b.setObjectName("NavButton")
                b.setText(text)
                b.setToolTip(f"{'Lower' if step < 0 else 'Raise'} the {'bottom' if end == 'min' else 'top'}"
                             " of the scale")
                b.clicked.connect(lambda _=False, e=end, s=step: self.nudge(e, s))
                self.nudges[(end, step)] = b
                pair.addWidget(b)
            if end == "max":
                ends.addWidget(self.channel_label, 1)
            ends.addLayout(pair)

        # ----- scale -----
        self.kind = QtWidgets.QComboBox()
        for kind, (label, _) in scales.KINDS.items():
            self.kind.addItem(label, kind)
        self.min_box, self.max_box = _number_box(), _number_box()
        scale_box = _section("Scale")
        sl = QtWidgets.QHBoxLayout(scale_box)
        sl.addWidget(QtWidgets.QLabel("Scale"))
        sl.addWidget(self.kind)
        sl.addSpacing(8)
        sl.addWidget(QtWidgets.QLabel("Min"))
        sl.addWidget(self.min_box, 1)
        sl.addWidget(QtWidgets.QLabel("Max"))
        sl.addWidget(self.max_box, 1)

        # ----- transform sliders -----
        self.sliders = [_SliderBox() for _ in range(3)]
        self.suggest = QtWidgets.QPushButton("Suggest width")
        self.suggest.setToolTip("Pick a width from this population's negative events, so the "
                                "dimmest events come off the axis without wasting space")
        xbox = _section("Transforms")
        xl = QtWidgets.QVBoxLayout(xbox)
        row = QtWidgets.QHBoxLayout()
        for s in self.sliders:
            row.addWidget(s)
        xl.addLayout(row)
        srow = QtWidgets.QHBoxLayout()
        self.problem = QtWidgets.QLabel()
        self.problem.setStyleSheet(f"color: {theme.CORAL};")
        self.problem.setWordWrap(True)
        srow.addWidget(self.problem, 1)
        srow.addWidget(self.suggest)
        xl.addLayout(srow)

        # ----- parameters -----
        self.params = QtWidgets.QListWidget()
        for c in chans:
            item = QtWidgets.QListWidgetItem(c.label)
            item.setData(QtCore.Qt.UserRole, c.pnn)
            self.params.addItem(item)
        self.params.setMinimumWidth(190)
        apply_to = QtWidgets.QPushButton("Apply to Parameters…")
        apply_to.setToolTip("Copy this scale to other parameters")
        apply_to.clicked.connect(self.apply_to_parameters)
        right = QtWidgets.QVBoxLayout()
        right.addWidget(apply_to)
        right.addWidget(self.params, 1)

        # ----- buttons -----
        save = QtWidgets.QPushButton("Save")
        save.setToolTip("Use this scale for this parameter in new workspaces too")
        save.clicked.connect(self.save_default)
        cancel = QtWidgets.QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        self.apply_button = QtWidgets.QPushButton("Apply")
        self.apply_button.setDefault(True)
        self.apply_button.clicked.connect(self.apply)
        self.saved_note = QtWidgets.QLabel()
        self.saved_note.setStyleSheet(f"color: {theme.MUTED};")
        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(save)
        buttons.addWidget(self.saved_note, 1)
        buttons.addWidget(cancel)
        buttons.addWidget(self.apply_button)

        left = QtWidgets.QVBoxLayout()
        left.addWidget(self.preview, 1)
        left.addLayout(ends)
        left.addWidget(scale_box)
        left.addWidget(xbox)
        top = QtWidgets.QHBoxLayout()
        top.addLayout(left, 1)
        top.addLayout(right)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(top, 1)
        lay.addLayout(buttons)

        self.kind.activated.connect(self._kind_picked)
        for w in (self.min_box, self.max_box):
            w.valueChanged.connect(self._edited)
        for s in self.sliders:
            s.changed.connect(self._edited)
        self.suggest.clicked.connect(self.suggest_width)
        self.params.currentItemChanged.connect(self._param_picked)
        self.params.setCurrentRow(list(self.labels).index(self.ch))
        self.resize(820, 640)

    # ----- state -----

    def transform(self, channel=None):
        channel = channel or self.ch
        return self.pending.get(channel) or self.ws.transform(channel)

    def _param_picked(self, item, _previous=None):
        if item is None:
            return
        self.ch = item.data(QtCore.Qt.UserRole)
        self.channel_label.setText(self.labels[self.ch])
        self._load()

    def _load(self):
        """Show the current channel's scale in the controls."""
        xform = self.transform()
        kind, s = scales.settings(xform)
        self._show(kind, s)
        self._draw()

    def _show(self, kind, s):
        widgets = [self.kind, self.min_box, self.max_box]
        for w in widgets:
            w.blockSignals(True)
        self.kind.setCurrentIndex(max(0, self.kind.findData(kind)))
        self.min_box.setValue(s["min"])
        self.max_box.setValue(s["max"])
        self.min_box.setEnabled(kind in scales.EDITABLE_MIN)
        self.min_box.setToolTip("" if kind in scales.EDITABLE_MIN else
                                "Set by the sliders on this scale (Extra Neg. Decades and width)")
        for w in widgets:
            w.blockSignals(False)
        specs = scales.SLIDERS[kind]
        for i, box in enumerate(self.sliders):
            spec = specs[i] if i < len(specs) else None
            box.configure(spec, s.get(spec[0], 0.0) if spec else 0.0)
        self.suggest.setEnabled(kind in ("biex", "logicle", "hyperlog"))
        self.problem.setText("")

    def _current_settings(self):
        s = {"min": self.min_box.value(), "max": self.max_box.value()}
        for box in self.sliders:
            if box.key:
                s[box.key] = box.value()
        return self.kind.currentData(), s

    def _set_pending(self, kind, s, channel=None):
        channel = channel or self.ch
        try:
            new = scales.make(kind, s)
            lo, hi = scales.raw_range(new)
            if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
                raise ValueError("These settings don't give a usable scale.")
        except (ValueError, ZeroDivisionError, FloatingPointError) as exc:
            self.problem.setText(str(exc))
            return False
        self.problem.setText("")
        self.pending[channel] = new
        if channel == self.ch and kind not in scales.EDITABLE_MIN:
            # the bottom of these scales follows from the sliders
            self.min_box.blockSignals(True)
            self.min_box.setValue(lo)
            self.min_box.blockSignals(False)
        item = self.params.item(list(self.labels).index(channel))
        f = item.font()
        f.setBold(True)
        item.setFont(f)
        return True

    def _edited(self):
        kind, s = self._current_settings()
        if self._set_pending(kind, s):
            self._draw()

    def _kind_picked(self, _index=None):
        kind = self.kind.currentData()
        new = scales.convert(self.transform(), kind)
        _, s = scales.settings(new)
        self._show(kind, s)
        self._set_pending(kind, s)
        self._draw()

    def nudge(self, end, step):
        """The − / + buttons under the ends of the axis."""
        kind, s = self._current_settings()
        if end == "max":
            s["max"] = s["max"] * (1.25 if kind == "linear" else 2.0) ** step
        elif kind == "linear":
            s["min"] += step * 0.05 * (s["max"] - s["min"])
        elif kind == "log":
            s["min"] *= 10.0 ** (0.25 * step)
        elif kind == "neglog":
            s["min"] = s["min"] / 2.0 ** step if s["min"] < 0 else s["min"] + step * 0.05 * s["max"]
        else:  # the bottom of biex-type scales is set by the extra negative decades
            key = "neg" if kind == "biex" else "a"
            box = next(b for b in self.sliders if b.key == key)
            s[key] = float(np.clip(box.value() - 0.1 * step, box.lo, box.hi))
        self._show(kind, s)
        if self._set_pending(kind, s):
            self._show(kind, scales.settings(self.pending[self.ch])[1])
            self._draw()

    def suggest_width(self):
        kind, s = self._current_settings()
        s = scales.suggest_width(kind, s, self._raw(self.ch))
        self._show(kind, s)
        if self._set_pending(kind, s):
            self._draw()

    # ----- preview -----

    def _raw(self, channel):
        if channel not in self._values:
            v = self.ws.scale_values(self.sample_id, channel, self.mask)
            v = v[np.isfinite(v)]
            if len(v) > PREVIEW_EVENTS:
                v = np.random.default_rng(0).choice(v, PREVIEW_EVENTS, replace=False)
            self._values[channel] = v
        return self._values[channel]

    def _draw(self):
        xform = self.transform()
        self.axis.set_transform(xform)
        self.preview.clear()
        raw = self._raw(self.ch)
        with np.errstate(all="ignore"):
            d = xform.apply(raw.reshape(-1, 1)).ravel()
        d = np.clip(d[np.isfinite(d)], 0.0, 1.0)  # off-scale events pile up on the ends, as in FlowJo
        counts, edges = np.histogram(d, bins=256, range=(0, 1))
        self.preview.addItem(pg.PlotDataItem(edges, counts, stepMode="center", fillLevel=0,
                                             brush=theme.qcolor(theme.TEAL, 70),
                                             pen=pg.mkPen(theme.TEAL, width=1.4)))
        self.preview.setXRange(0, 1, padding=0)
        self.preview.setYRange(0, max(counts.max() * 1.1, 1) if len(counts) else 1, padding=0)

    # ----- buttons -----

    def apply_to_parameters(self):
        others = [ch for ch in self.labels if ch != self.ch]
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Apply to Parameters")
        lst = QtWidgets.QListWidget()
        for ch in others:
            item = QtWidgets.QListWidgetItem(self.labels[ch])
            item.setData(QtCore.Qt.UserRole, ch)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Unchecked)
            lst.addItem(item)
        kind, _ = self._current_settings()
        note = QtWidgets.QLabel(f"Copy this {scales.KINDS[kind][0]} scale from "
                                f"{self.labels[self.ch]} to (each keeps its own top of scale):")
        note.setWordWrap(True)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        lay = QtWidgets.QVBoxLayout(dlg)
        lay.addWidget(note)
        lay.addWidget(lst)
        lay.addWidget(bb)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        chosen = [lst.item(i).data(QtCore.Qt.UserRole) for i in range(lst.count())
                  if lst.item(i).checkState() == QtCore.Qt.Checked]
        self.copy_to(chosen)

    def copy_to(self, channels):
        """Give these channels the current channel's scale, each keeping its own top."""
        kind, s = self._current_settings()
        for ch in channels:
            self._set_pending(kind, {**s, "max": scales.settings(self.transform(ch))[1]["max"]}, ch)

    def save_default(self):
        kind, s = self._current_settings()
        try:
            save_scale_default(self.ch, scales.make(kind, s))
        except ValueError as exc:
            self.problem.setText(str(exc))
            return
        self.saved_note.setText(f"Saved: new workspaces open {self.labels[self.ch]} on {scales.KINDS[kind][0]}.")

    def apply(self):
        try:
            for ch, new in self.pending.items():
                self.ws.set_transform(ch, new)
        except Exception as exc:  # noqa: BLE001 - show any engine error instead of crashing
            QtWidgets.QMessageBox.warning(self, "Transform", str(exc))
            return
        self.accept()
