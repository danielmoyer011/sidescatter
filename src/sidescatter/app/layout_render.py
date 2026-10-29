"""Paints layout items with QPainter, so the screen, PDF, SVG, PNG and printer all match.

Plots are drawn the way the graph window draws them (same binning, density
colours, axis ticks and gate outlines) but as plain vector drawing, which
keeps text and lines sharp in exported figures. Units are points.
"""

import math

import flowkit as fk
import numpy as np
from PySide6 import QtCore, QtGui

from . import theme
from .axes import ticks
from .gating import service as gating
from .graph_view import BINS, PSEUDOCOLOR, _bounds, _half_open, ellipse_shape
from ..core.workspace import ROOT

DOT_LIMIT = 20_000
# Fonts are sized in pixels at this multiple and drawn under a matching scale-down,
# so text is measured in points on every device (screen, PDF, SVG, printer).
_TEXT_SCALE = 8.0


# ----- text ------------------------------------------------------------------

def _font(size, bold=False, italic=False):
    f = QtGui.QFont()
    f.setPixelSize(max(1, round(size * _TEXT_SCALE)))
    f.setBold(bold)
    f.setItalic(italic)
    return f


def text_size(text, size, bold=False):
    """(width, height) of one line of text, in points."""
    m = QtGui.QFontMetricsF(_font(size, bold))
    return m.horizontalAdvance(text) / _TEXT_SCALE, m.height() / _TEXT_SCALE


def draw_text(p, rect, text, size, flags=QtCore.Qt.AlignCenter, color=theme.INK,
              bold=False, italic=False, rotate=False):
    """Draw text inside rect (points). rotate=True reads bottom-to-top, for a Y axis title."""
    p.save()
    p.setFont(_font(size, bold, italic))
    p.setPen(QtGui.QColor(color))
    if rotate:
        p.translate(rect.left(), rect.bottom())
        p.rotate(-90)
        rect = QtCore.QRectF(0, 0, rect.height(), rect.width())
    else:
        p.translate(rect.left(), rect.top())
        rect = QtCore.QRectF(0, 0, rect.width(), rect.height())
    p.scale(1 / _TEXT_SCALE, 1 / _TEXT_SCALE)
    s = _TEXT_SCALE
    p.drawText(QtCore.QRectF(0, 0, rect.width() * s, rect.height() * s), int(flags), text)
    p.restore()


def fill_fields(text, ws, item, lazy=False):
    """Replace {sample}, {population}, {count} and {freq} with values for the item.
    lazy: show "…" for numbers whose gating is still being worked out."""
    if "{" not in text:
        return text
    path = item.population_path
    name = path[-1] if len(path) > 1 else None
    values = {"sample": item.sample_id or "", "population": name or "All events",
              "count": "", "freq": ""}
    sid = item.sample_id
    wants_numbers = "{count}" in text or "{freq}" in text
    if wants_numbers and lazy and sid in ws.samples and not gating().ensure(ws, [sid]):
        values["count"] = values["freq"] = "…"
    elif wants_numbers and sid in ws.samples and _population_ok(ws, sid, path):
        values["count"] = f"{ws.count(sid, name, path[:-1] if name else ROOT):,}"
        if name:
            values["freq"] = f"{ws.statistic(sid, 'Freq. of Parent', name, path[:-1]):.2f}%"
    for key, value in values.items():
        text = text.replace("{" + key + "}", value)
    return text


# ----- data --------------------------------------------------------------------

def _population_ok(ws, sid, path):
    if len(path) == 1:
        return True
    try:
        if (path[-1], tuple(path[:-1])) not in ws.population_paths():
            return False
    except Exception:
        return False
    return ws.applies_to(sid, path[-1], tuple(path[:-1]))


def default_channels(ws, sid):
    """FSC-A against SSC-A when the sample has them, else the first fluorescence channels."""
    chans = ws.channels(sid)
    names = [c.pnn for c in chans]
    fluor = [c.pnn for c in chans if c.kind == "fluorescence"]
    x = "FSC-A" if "FSC-A" in names else (fluor[0] if fluor else names[0])
    y = "SSC-A" if "SSC-A" in names else ((fluor[1:] or fluor or names)[0])
    return x, y


def resolve_channels(ws, item):
    names = [c.pnn for c in ws.channels(item.sample_id)]
    dx, dy = default_channels(ws, item.sample_id)
    x = item.x_channel if item.x_channel in names else dx
    y = item.y_channel if item.y_channel in names else dy
    return x, y


class PlotCache:
    """Binned plot data per item, recomputed only when the gating results change.

    lazy (the editor's canvas): never gate on the UI thread; items whose samples
    are still being gated in the background read "Calculating…" until they're in.
    Exports aren't lazy, and run export_job, which gates everything first."""

    def __init__(self, lazy=False):
        self._data = {}
        self.lazy = lazy

    def clear(self):
        self._data.clear()

    def get(self, ws, item):
        """Plot data for an item, or a string saying why it can't be drawn."""
        sid = item.sample_id
        if sid is None:
            return "Drop a population here"
        if sid not in ws.samples:
            return f"{sid} is not in the workspace"
        path = item.population_path
        xch, ych = resolve_channels(ws, item)
        hist = item.plot_type == "Histogram"
        key = (ws.revision, sid, path, xch, None if hist else ych, item.plot_type)
        if key in self._data:
            return self._data[key]
        if len(self._data) > 200:
            self._data.clear()
        if not _population_ok(ws, sid, path):
            self._data[key] = f"{path[-1]} doesn't apply to {sid}"
            return self._data[key]
        if self.lazy and not gating().ensure(ws, [sid]):
            return "Calculating…"
        name = path[-1] if len(path) > 1 else None
        mask = ws.membership(sid, name, tuple(path[:-1]) if name else ROOT)
        labels = {c.pnn: c.label for c in ws.channels(sid)}
        x = ws.display_values(sid, xch, mask)
        d = {"x_channel": xch, "y_channel": None if hist else ych, "x_label": labels[xch],
             "y_label": "Count" if hist else labels[ych], "x_xform": ws.transform(xch),
             "y_xform": None if hist else ws.transform(ych), "count": int(mask.sum())}
        if hist:
            counts, _ = np.histogram(x, bins=BINS, range=(0, 1))
            d["counts"] = counts
            d["y_top"] = float(max(counts.max() * 1.1, 1)) if len(counts) else 1.0
        else:
            y = ws.display_values(sid, ych, mask)
            if item.plot_type == "Dot plot":
                ok = np.isfinite(x) & np.isfinite(y)  # computed channels (UMAP...) can be empty
                x, y = x[ok], y[ok]
                n = min(len(x), DOT_LIMIT)
                idx = np.random.default_rng(0).choice(len(x), n, replace=False) \
                    if len(x) > n else slice(None)
                d["dots"] = (x[idx], y[idx])
            else:
                h, _, _ = np.histogram2d(x, y, bins=BINS, range=[[0, 1], [0, 1]])
                if item.plot_type == "Contour":
                    d["contours"] = _contours(h)
                else:
                    d["image"] = _density_image(h)
        self._data[key] = d
        return d


def _density_image(h):
    density = np.log1p(h)
    density = density / (density.max() or 1)
    rgba = PSEUDOCOLOR.map(density, mode="byte")
    rgba[h == 0] = (0, 0, 0, 0)
    rows = np.ascontiguousarray(rgba.transpose(1, 0, 2)[::-1])  # image rows run top-down
    img = QtGui.QImage(rows.data, BINS, BINS, BINS * 4, QtGui.QImage.Format_RGBA8888)
    return img.copy()  # own the pixels; rows goes away


def _contours(h):
    import pyqtgraph as pg
    from scipy.ndimage import gaussian_filter

    smooth = gaussian_filter(h, 2.0)
    top = smooth.max() or 1
    out = []
    for i, level in enumerate((0.02, 0.05, 0.1, 0.2, 0.4, 0.7)):
        color = PSEUDOCOLOR.map(0.15 + i * 0.16, mode="qcolor")
        lines = pg.functions.isocurve(smooth, level * top, connected=True)
        out.append((color, [[(float(px) / BINS, float(py) / BINS) for px, py in line]
                            for line in lines]))
    return out


# ----- plots -------------------------------------------------------------------

def paint_plot(p, w, h, ws, item, cache):
    """Draw a plot item into the rectangle (0, 0, w, h)."""
    fs = item.font_size
    data = cache.get(ws, item)
    title = fill_fields(item.title, ws, item, cache.lazy).strip() if item.title else ""
    top = 2.0
    if title:
        _, th = text_size(title, fs, bold=True)
        draw_text(p, QtCore.QRectF(0, 0, w, th), _elide(title, fs, w, True), fs,
                  QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter, bold=True)
        top = th + 3
    if isinstance(data, str):
        frame = QtCore.QRectF(1, top, w - 2, h - top - 1)
        p.setPen(QtGui.QPen(QtGui.QColor(theme.BORDER_STRONG), 0.8, QtCore.Qt.DashLine))
        p.setBrush(QtCore.Qt.NoBrush)
        p.drawRect(frame)
        draw_text(p, frame.adjusted(6, 6, -6, -6), data, fs, QtCore.Qt.AlignCenter | QtCore.Qt.TextWordWrap,
                  color=theme.MUTED)
        return

    tick_fs = fs * 0.85
    hist = "counts" in data
    x_ticks, x_minor = ticks(data["x_xform"])
    y_ticks, y_minor = ticks(data["y_xform"], 0.0, data["y_top"]) if hist else ticks(data["y_xform"])
    _, line_h = text_size("0", tick_fs)
    label_w = max((text_size(lbl, tick_fs)[0] for _, lbl in y_ticks), default=0.0)
    axis_title_h = text_size("X", fs)[1] if item.show_axis_labels else 0.0
    left = axis_title_h + label_w + 6
    bottom = line_h + axis_title_h + 5
    plot = QtCore.QRectF(left, top, max(w - left - 6, 10), max(h - top - bottom, 10))

    def X(v):
        return plot.left() + v * plot.width()

    def Y(v):
        return plot.bottom() - v * plot.height()

    y_scale = (1.0 / data["y_top"]) if hist else 1.0

    p.save()
    p.setPen(QtCore.Qt.NoPen)
    p.setBrush(QtGui.QColor(theme.SURFACE))
    p.drawRect(plot)
    p.setClipRect(plot)
    if "image" in data:
        p.drawImage(plot, data["image"])
    elif "dots" in data:
        xs, ys = data["dots"]
        pen = QtGui.QPen(theme.qcolor(theme.INK, 150), max(plot.width() / 300, 0.6))
        pen.setCapStyle(QtCore.Qt.RoundCap)
        p.setPen(pen)
        p.drawPoints(QtGui.QPolygonF([QtCore.QPointF(X(a), Y(b)) for a, b in zip(xs, ys)]))
    elif "contours" in data:
        for color, lines in data["contours"]:
            p.setPen(QtGui.QPen(color, 0.9))
            for line in lines:
                p.drawPolyline(QtGui.QPolygonF([QtCore.QPointF(X(a), Y(b)) for a, b in line]))
    elif hist:
        counts = data["counts"]
        path = QtGui.QPainterPath(QtCore.QPointF(X(0), Y(0)))
        for i, c in enumerate(counts):
            path.lineTo(X(i / BINS), Y(c * y_scale))
            path.lineTo(X((i + 1) / BINS), Y(c * y_scale))
        path.lineTo(X(1), Y(0))
        path.closeSubpath()
        p.setPen(QtGui.QPen(QtGui.QColor(theme.TEAL), 1.0))
        p.setBrush(theme.qcolor(theme.TEAL, 70))
        p.drawPath(path)
    if item.show_gates:
        _paint_gates(p, ws, item, data, X, Y, y_scale, plot, tick_fs)
    p.restore()

    # axes, ticks and labels
    axis_pen = QtGui.QPen(QtGui.QColor(theme.BORDER_STRONG), 0.8)
    p.setPen(axis_pen)
    p.setBrush(QtCore.Qt.NoBrush)
    p.drawRect(plot)
    tick = 3.0
    for v in x_minor:
        p.drawLine(QtCore.QPointF(X(v), plot.bottom()), QtCore.QPointF(X(v), plot.bottom() + tick * 0.6))
    for v in y_minor:
        p.drawLine(QtCore.QPointF(plot.left(), Y(v)), QtCore.QPointF(plot.left() - tick * 0.6, Y(v)))
    for v, lbl in x_ticks:
        p.setPen(axis_pen)
        p.drawLine(QtCore.QPointF(X(v), plot.bottom()), QtCore.QPointF(X(v), plot.bottom() + tick))
        tw, _ = text_size(lbl, tick_fs)
        draw_text(p, QtCore.QRectF(X(v) - tw, plot.bottom() + tick, 2 * tw, line_h), lbl, tick_fs,
                  QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop, color=theme.MUTED)
    for v, lbl in y_ticks:
        yv = Y(v * y_scale)
        p.setPen(axis_pen)
        p.drawLine(QtCore.QPointF(plot.left(), yv), QtCore.QPointF(plot.left() - tick, yv))
        draw_text(p, QtCore.QRectF(0, yv - line_h / 2, plot.left() - tick - 1, line_h), lbl, tick_fs,
                  QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter, color=theme.MUTED)
    if item.show_axis_labels:
        draw_text(p, QtCore.QRectF(plot.left(), h - axis_title_h, plot.width(), axis_title_h),
                  _elide(data["x_label"], fs, plot.width()), fs, QtCore.Qt.AlignCenter)
        draw_text(p, QtCore.QRectF(0, plot.top(), axis_title_h, plot.height()),
                  _elide(data["y_label"], fs, plot.height()), fs, QtCore.Qt.AlignCenter, rotate=True)


def _elide(text, size, width, bold=False):
    m = QtGui.QFontMetricsF(_font(size, bold))
    return m.elidedText(text, QtCore.Qt.ElideMiddle, width * _TEXT_SCALE)


def _paint_gates(p, ws, item, data, X, Y, y_scale, plot, fs):
    sid, path = item.sample_id, item.population_path
    xch, ych = data["x_channel"], data["y_channel"]
    pen = QtGui.QPen(QtGui.QColor(theme.GATE), 1.0)
    labels = []

    def label(child, at, align, short=False):
        if not item.show_gate_labels:
            return
        shown = child.split(":")[0] if short else child
        try:
            freq = ws.statistic(sid, "Freq. of Parent", child, path)
            labels.append((f"{shown}\n{freq:.2f}%", at, align))
        except KeyError:
            labels.append((shown, at, align))

    quads, bisects = {}, {}
    for child, _ in ws.children(path):
        if not ws.applies_to(sid, child, path):
            continue
        gate = ws.get_gate(child, path)
        dims = [d.id for d in getattr(gate, "dimensions", []) if hasattr(d, "id")]
        p.setPen(pen)
        p.setBrush(QtCore.Qt.NoBrush)
        if ych is None and isinstance(gate, fk.gates.RectangleGate) and dims == [xch]:
            split = _half_open(gate.dimensions[0])
            if split is not None:
                bisects.setdefault(round(split, 9), []).append((child, gate))
                continue
            lo, hi = _bounds(gate.dimensions[0])
            p.setPen(QtGui.QPen(QtGui.QColor(theme.ACCENT), 1.0))
            p.setBrush(theme.qcolor(theme.ACCENT, 30))
            p.drawRect(QtCore.QRectF(QtCore.QPointF(X(lo), plot.top()), QtCore.QPointF(X(hi), plot.bottom())))
            label(child, QtCore.QPointF(X(lo) + 2, plot.top() + 2), "tl")
        elif ych is not None and dims == [xch, ych]:
            if isinstance(gate, fk.gates.RectangleGate):
                sx, sy = (_half_open(d) for d in gate.dimensions)
                if sx is not None and sy is not None:
                    quads.setdefault((round(sx, 9), round(sy, 9)), []).append((child, gate))
                    continue
                (x0, x1), (y0, y1) = (_bounds(d) for d in gate.dimensions)
                p.drawRect(QtCore.QRectF(QtCore.QPointF(X(x0), Y(y1)), QtCore.QPointF(X(x1), Y(y0))))
                label(child, QtCore.QPointF(X(x0), Y(min(y1, 0.99)) - 1), "bl")
            elif isinstance(gate, fk.gates.PolygonGate):
                p.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(X(float(a)), Y(float(b)))
                                               for a, b in gate.vertices]))
                topv = max(gate.vertices, key=lambda v: v[1])
                label(child, QtCore.QPointF(X(float(topv[0])), Y(min(float(topv[1]), 0.92)) - 1), "bl")
            elif isinstance(gate, fk.gates.EllipsoidGate):
                (cx, cy), a, b, angle = ellipse_shape(gate)
                t = np.linspace(0, 2 * np.pi, 120)
                ex = cx + a * np.cos(t) * math.cos(angle) - b * np.sin(t) * math.sin(angle)
                ey = cy + a * np.cos(t) * math.sin(angle) + b * np.sin(t) * math.cos(angle)
                p.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(X(u), Y(v)) for u, v in zip(ex, ey)]))
                label(child, QtCore.QPointF(X(cx - a * 0.5), Y(min(cy + max(a, b), 0.95)) - 1), "bl")
    for (sx, sy), members in quads.items():
        p.setPen(pen)
        p.drawLine(QtCore.QPointF(X(sx), plot.top()), QtCore.QPointF(X(sx), plot.bottom()))
        p.drawLine(QtCore.QPointF(plot.left(), Y(sy)), QtCore.QPointF(plot.right(), Y(sy)))
        for child, g in members:
            right, up = g.dimensions[0].max is None, g.dimensions[1].max is None
            at = QtCore.QPointF(plot.right() - 2 if right else plot.left() + 2,
                                plot.top() + 2 if up else plot.bottom() - 2)
            label(child, at, ("t" if up else "b") + ("r" if right else "l"), short=True)
    for split, members in bisects.items():
        p.setPen(QtGui.QPen(QtGui.QColor(theme.ACCENT), 1.2))
        p.drawLine(QtCore.QPointF(X(split), plot.top()), QtCore.QPointF(X(split), plot.bottom()))
        for child, g in members:
            right = g.dimensions[0].max is None
            label(child, QtCore.QPointF(X(split) + (2 if right else -2), plot.top() + 2),
                  "tl" if right else "tr")
    for text, at, align in labels:
        _gate_label(p, text, at, align, fs)


def _gate_label(p, text, at, align, fs):
    lines = text.split("\n")
    w = max(text_size(s, fs)[0] for s in lines) + 4
    lh = text_size("0", fs)[1]
    h = lh * len(lines) + 2
    x = at.x() - w if "r" in align else at.x()
    y = at.y() if "t" in align else at.y() - h
    box = QtCore.QRectF(x, y, w, h)
    p.setPen(QtCore.Qt.NoPen)
    p.setBrush(theme.qcolor(theme.SURFACE, 210))
    p.drawRect(box)
    draw_text(p, box, text, fs, QtCore.Qt.AlignCenter)


# ----- text boxes ----------------------------------------------------------------

def paint_text(p, w, h, ws, item, lazy=False):
    flags = QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop | QtCore.Qt.TextWordWrap
    draw_text(p, QtCore.QRectF(2, 2, w - 4, h - 4), fill_fields(item.text, ws, item, lazy), item.font_size,
              flags, bold=item.bold, italic=item.italic)


def paint_item(p, ws, item, cache):
    if item.kind == "plot":
        paint_plot(p, item.w, item.h, ws, item, cache)
    else:
        paint_text(p, item.w, item.h, ws, item, cache.lazy)


# ----- whole pages (export and printing) ------------------------------------------

def render_page(p, layout, page, ws, cache, scale=1.0):
    """Draw one page; scale is device units per point."""
    pw, ph = layout.page_width, layout.page_height
    p.save()
    p.scale(scale, scale)
    p.fillRect(QtCore.QRectF(0, 0, pw, ph), QtGui.QColor("#FFFFFF"))
    p.setClipRect(QtCore.QRectF(0, 0, pw, ph))
    p.translate(0, -page * ph)
    for item in layout.items:
        if item.y + item.h <= page * ph or item.y >= (page + 1) * ph:
            continue
        p.save()
        p.translate(item.x, item.y)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        paint_item(p, ws, item, cache)
        p.restore()
    p.restore()


def export_job(layout, ws, path, dpi=300, on_done=None):
    """Gate every sample on the layout in the background (with progress and
    Cancel), then write the files. on_done(files) runs once they're written."""
    sids = list(dict.fromkeys(it.sample_id for it in layout.items if it.sample_id))
    return gating().prepare(ws, sids, f"Exporting {layout.name}",
                            lambda: (on_done or (lambda files: None))(export(layout, ws, path, dpi)),
                            key=("export-layout", id(layout)))


def export(layout, ws, path, dpi=300, cache=None):
    """Write the layout to PDF (all pages), SVG or PNG (one file per page). Returns the files."""
    from pathlib import Path

    cache = _eager(cache)
    path = Path(path)
    suffix = path.suffix.lower()
    pages = layout.page_count()
    pw, ph = layout.page_width, layout.page_height
    if suffix == ".pdf":
        writer = QtGui.QPdfWriter(str(path))
        writer.setResolution(int(dpi))
        writer.setPageLayout(_page_layout(layout))
        writer.setTitle(layout.name)
        writer.setCreator("Sidescatter")
        p = QtGui.QPainter(writer)
        for page in range(pages):
            if page:
                writer.newPage()
            render_page(p, layout, page, ws, cache, dpi / 72.0)
        p.end()
        return [path]
    out = []
    for page in range(pages):
        target = path if pages == 1 else path.with_name(f"{path.stem}-page{page + 1}{path.suffix}")
        if suffix == ".svg":
            from PySide6 import QtSvg

            gen = QtSvg.QSvgGenerator()
            gen.setFileName(str(target))
            gen.setSize(QtCore.QSize(round(pw), round(ph)))
            gen.setViewBox(QtCore.QRectF(0, 0, pw, ph))
            gen.setResolution(72)
            gen.setTitle(layout.name)
            p = QtGui.QPainter(gen)
            render_page(p, layout, page, ws, cache)
            p.end()
        elif suffix in (".png", ".tif", ".tiff", ".jpg", ".jpeg"):
            scale = dpi / 72.0
            img = QtGui.QImage(round(pw * scale), round(ph * scale), QtGui.QImage.Format_ARGB32)
            img.setDotsPerMeterX(round(dpi / 0.0254))
            img.setDotsPerMeterY(round(dpi / 0.0254))
            img.fill(QtGui.QColor("#FFFFFF"))
            p = QtGui.QPainter(img)
            p.setRenderHint(QtGui.QPainter.Antialiasing)
            p.setRenderHint(QtGui.QPainter.TextAntialiasing)
            render_page(p, layout, page, ws, cache, scale)
            p.end()
            if not img.save(str(target)):
                raise OSError(f"could not write {target}")
        else:
            raise ValueError(f"unsupported export format {suffix}")
        out.append(target)
    return out


def _eager(cache):
    """A cache that gates on demand, sharing what a lazy (on-screen) cache already binned."""
    if cache is None:
        return PlotCache()
    if not cache.lazy:
        return cache
    eager = PlotCache()
    eager._data = cache._data
    return eager


def _page_layout(layout):
    size = QtGui.QPageSize(QtGui.QPageSize.Letter if layout.page_size == "Letter" else QtGui.QPageSize.A4)
    orient = QtGui.QPageLayout.Landscape if layout.landscape else QtGui.QPageLayout.Portrait
    return QtGui.QPageLayout(size, orient, QtCore.QMarginsF(0, 0, 0, 0))


def print_layout(layout, ws, printer, cache=None):
    """Print every page through a QPrinter the user has set up."""
    cache = _eager(cache)
    printer.setPageLayout(_page_layout(layout))
    p = QtGui.QPainter(printer)
    rect = printer.pageLayout().paintRectPixels(printer.resolution())
    scale = min(rect.width() / layout.page_width, rect.height() / layout.page_height)
    for page in range(layout.page_count()):
        if page:
            printer.newPage()
        render_page(p, layout, page, ws, cache, scale)
    p.end()
