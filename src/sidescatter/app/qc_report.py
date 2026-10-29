"""The data-cleaning report: a summary table, then Time plots of every sample, as a PDF.

Each plot shows a channel (or the event rate) against Time as a density, with
the removed stretches shaded and PeacoQC-style peak traces on top, like
PeacoQC's own overview figure.
"""

import numpy as np
from PySide6 import QtCore, QtGui

from . import theme
from .graph_view import PSEUDOCOLOR
from .layout_render import draw_text

AMBER_PCT = 20.0          # samples losing more than this are flagged
AMBER = "#E8A33D"
REMOVED = QtGui.QColor(theme.CORAL)
PEAK = QtGui.QColor(theme.ACCENT_DARK)
PAGE_W, PAGE_H, MARGIN = 612.0, 792.0, 40.0  # US Letter, points
COLS, ROWS = 3, 5


def flagged(pct):
    return pct > AMBER_PCT


def density_image(h):
    """A (nx, ny) histogram as a QImage, x left to right and y bottom to top."""
    h = np.asarray(h, dtype=float)
    density = np.log1p(h)
    density = density / (density.max() or 1)
    rgba = PSEUDOCOLOR.map(density, mode="byte")
    rgba[h == 0] = (0, 0, 0, 0)
    rows = np.ascontiguousarray(rgba.transpose(1, 0, 2)[::-1])
    nx, ny = h.shape
    img = QtGui.QImage(rows.data, nx, ny, nx * 4, QtGui.QImage.Format_RGBA8888)
    return img.copy()


def summary_rows(pages):
    """(sample, events, removed %, step percentages, warnings) for the summary table."""
    out = []
    for page in pages:
        res = page["result"]
        out.append((page["sample"], res.n_events, page["plot"]["removed_pct"], res.removed_by,
                    res.warnings))
    return out


def paint_panel(p, rect, title, plot, channel=None):
    """One Time plot: a channel's density, or the event rate when channel is None."""
    x0, x1 = plot["x_range"]
    inner = rect.adjusted(0, 12, 0, -10)
    p.save()
    p.setPen(QtGui.QPen(QtGui.QColor(theme.BORDER_STRONG), 0.6))
    p.setBrush(QtGui.QColor(theme.SURFACE))
    p.drawRect(inner)

    def X(v):
        return inner.left() + (v - x0) / ((x1 - x0) or 1) * inner.width()

    p.setPen(QtCore.Qt.NoPen)
    p.setBrush(theme.qcolor(theme.CORAL, 60))
    for a, b in plot["stretches"]:
        p.drawRect(QtCore.QRectF(X(a), inner.top(), max(X(b) - X(a), 0.6), inner.height()))

    if channel is None:
        xs, ys = (np.asarray(v, float) for v in plot["rate"])
        top = float(ys.max()) if len(ys) and ys.max() > 0 else 1.0
        path = QtGui.QPainterPath()
        for i, (x, y) in enumerate(zip(xs, ys)):
            pt = QtCore.QPointF(X(x), inner.bottom() - y / top * inner.height() * 0.92)
            path.moveTo(pt) if i == 0 else path.lineTo(pt)
        p.setPen(QtGui.QPen(QtGui.QColor(theme.TEAL), 0.9))
        p.setBrush(QtCore.Qt.NoBrush)
        p.drawPath(path)
    else:
        p.drawImage(inner, density_image(channel["hist"]))
        lo, hi = channel["y_range"]
        p.setPen(QtGui.QPen(PEAK, 0.8))
        p.setBrush(QtCore.Qt.NoBrush)
        for xs, ys in channel["peaks"]:
            path = QtGui.QPainterPath()
            for i, (x, y) in enumerate(zip(xs, ys)):
                yy = min(max((y - lo) / ((hi - lo) or 1), 0), 1)
                pt = QtCore.QPointF(X(x), inner.bottom() - yy * inner.height())
                path.moveTo(pt) if i == 0 else path.lineTo(pt)
            p.drawPath(path)
    p.restore()
    draw_text(p, QtCore.QRectF(rect.left(), rect.top(), rect.width(), 11), title, 7.5,
              QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, bold=True)
    axis = QtCore.QRectF(inner.left(), inner.bottom() + 1, inner.width(), 9)
    draw_text(p, axis, f"{x0:,.0f}", 6, QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop, theme.MUTED)
    draw_text(p, axis, plot["x_label"], 6, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop, theme.MUTED)
    draw_text(p, axis, f"{x1:,.0f}", 6, QtCore.Qt.AlignRight | QtCore.Qt.AlignTop, theme.MUTED)


def _steps_text(removed_by):
    parts = [f"{k} {v:.1f}%" for k, v in removed_by.items() if v > 0.0005]
    return ", ".join(parts) if parts else "nothing removed"


def render(p, pages, title, scale=1.0, new_page=None):
    """Paint the report. new_page() starts the next page (PDF writers); pages: see write_pdf."""
    p.save()
    p.scale(scale, scale)
    width = PAGE_W - 2 * MARGIN
    y = MARGIN
    draw_text(p, QtCore.QRectF(MARGIN, y, width, 20), title, 14, QtCore.Qt.AlignLeft, bold=True)
    y += 24
    settings = pages[0]["result"].settings if pages else {}
    draw_text(p, QtCore.QRectF(MARGIN, y, width, 12),
              f"{settings.get('preset', 'Custom')} settings: MAD {settings.get('mad')}, isolation tree "
              f"limit {settings.get('it_limit')}, {settings.get('consecutive_bins')} consecutive windows. "
              f"Samples losing more than {AMBER_PCT:.0f}% are marked.",
              7.5, QtCore.Qt.AlignLeft, theme.MUTED)
    y += 20
    col = (0.0, 0.42, 0.56, 0.68)
    for i, head in enumerate(("Sample", "Events", "Removed", "Removed by")):
        draw_text(p, QtCore.QRectF(MARGIN + col[i] * width, y, width * 0.4, 12), head, 7.5,
                  QtCore.Qt.AlignLeft, bold=True)
    y += 14
    for sample, n, pct, by, _ in summary_rows(pages):
        if y > PAGE_H - MARGIN - 14:
            new_page and new_page()
            y = MARGIN
        if flagged(pct):
            p.fillRect(QtCore.QRectF(MARGIN - 4, y - 1, width + 8, 12), theme.qcolor(AMBER, 70))
        cells = (sample, f"{n:,}", f"{pct:.2f}%", _steps_text(by))
        for i, text in enumerate(cells):
            w = (col[i + 1] - col[i]) * width - 6 if i < 3 else (1 - col[i]) * width
            draw_text(p, QtCore.QRectF(MARGIN + col[i] * width, y, w, 12), text, 7,
                      QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        y += 13

    gap = 10.0
    pw = (width - (COLS - 1) * gap) / COLS
    ph = (PAGE_H - 2 * MARGIN - 60 - (ROWS - 1) * gap) / ROWS
    for page in pages:
        res, plot = page["result"], page["plot"]
        panels = [("Event rate", None)] + [(c["name"], c) for c in plot["channels"]]
        for start in range(0, len(panels), COLS * ROWS):
            new_page and new_page()
            head = f"{page['sample']}: {plot['removed_pct']:.2f}% removed"
            draw_text(p, QtCore.QRectF(MARGIN, MARGIN, width, 16), head, 11, QtCore.Qt.AlignLeft,
                      AMBER if flagged(plot["removed_pct"]) else theme.INK, bold=True)
            notes = _steps_text(res.removed_by) + "".join(f". {w}" for w in res.warnings)
            draw_text(p, QtCore.QRectF(MARGIN, MARGIN + 18, width, 36), notes, 7,
                      QtCore.Qt.AlignLeft | QtCore.Qt.TextWordWrap, theme.MUTED)
            for k, (name, ch) in enumerate(panels[start:start + COLS * ROWS]):
                r, c = divmod(k, COLS)
                rect = QtCore.QRectF(MARGIN + c * (pw + gap), MARGIN + 60 + r * (ph + gap), pw, ph)
                paint_panel(p, rect, name, plot, ch)
    p.restore()


def write_pdf(path, pages, title="Data cleaning report"):
    """pages: [{"sample": name, "result": cleaning.Result, "plot": cleaning.plot_data(...)}]."""
    writer = QtGui.QPdfWriter(str(path))
    dpi = 150
    writer.setResolution(dpi)
    writer.setPageLayout(QtGui.QPageLayout(QtGui.QPageSize(QtGui.QPageSize.Letter),
                                           QtGui.QPageLayout.Portrait, QtCore.QMarginsF(0, 0, 0, 0)))
    writer.setTitle(title)
    writer.setCreator("Sidescatter")
    p = QtGui.QPainter(writer)
    p.setRenderHint(QtGui.QPainter.Antialiasing)
    render(p, pages, title, dpi / 72.0, writer.newPage)
    p.end()
    return path
