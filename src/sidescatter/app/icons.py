"""Line icons for the ribbon and the gating tools, painted at runtime (no image assets)."""

import math

from PySide6 import QtCore, QtGui

from . import theme

_CACHE = {}
_DRAW = {}


def icon(name, size=32):
    """A QIcon drawn on a 32x32 grid, crisp at 1x and 2x screens."""
    key = (name, size)
    if key not in _CACHE:
        ic = QtGui.QIcon()
        for scale in (1, 2):
            ic.addPixmap(_pixmap(name, size, scale), QtGui.QIcon.Normal)
            ic.addPixmap(_pixmap(name, size, scale, disabled=True), QtGui.QIcon.Disabled)
        _CACHE[key] = ic
    return _CACHE[key]


def _pixmap(name, size, scale, disabled=False):
    pm = QtGui.QPixmap(size * scale, size * scale)
    pm.fill(QtCore.Qt.transparent)
    p = QtGui.QPainter(pm)
    p.setRenderHint(QtGui.QPainter.Antialiasing)
    p.scale(size * scale / 32, size * scale / 32)
    if disabled:
        p.setOpacity(0.35)
    _DRAW[name](p)
    p.end()
    pm.setDevicePixelRatio(scale)
    return pm


def _register(name):
    def deco(fn):
        _DRAW[name] = fn
        return fn
    return deco


def _pen(p, color=theme.INK, width=2.0):
    pen = QtGui.QPen(QtGui.QColor(color), width)
    pen.setCapStyle(QtCore.Qt.RoundCap)
    pen.setJoinStyle(QtCore.Qt.RoundJoin)
    p.setPen(pen)
    p.setBrush(QtCore.Qt.NoBrush)


def _fill(p, color, alpha=255):
    p.setBrush(theme.qcolor(color, alpha))


def _plus(p, x, y, r=4.5, color=theme.CORAL):
    _pen(p, color, 2.6)
    p.drawLine(QtCore.QPointF(x - r, y), QtCore.QPointF(x + r, y))
    p.drawLine(QtCore.QPointF(x, y - r), QtCore.QPointF(x, y + r))


def _tube(p, x, top, h=20, w=6, liquid=theme.TEAL):
    body = QtGui.QPainterPath()
    body.moveTo(x, top)
    body.lineTo(x, top + h - w / 2)
    body.arcTo(x, top + h - w, w, w, 180, 180)
    body.lineTo(x + w, top)
    _fill(p, liquid, 70)
    _pen(p, theme.INK, 1.8)
    p.drawPath(body)
    _pen(p, theme.INK, 2.2)
    p.drawLine(QtCore.QPointF(x - 1.5, top), QtCore.QPointF(x + w + 1.5, top))


def _scatter(p, pts, color=theme.TEAL, r=1.6):
    p.setPen(QtCore.Qt.NoPen)
    _fill(p, color)
    for x, y in pts:
        p.drawEllipse(QtCore.QPointF(x, y), r, r)


# ----- ribbon ----------------------------------------------------------------

@_register("new_workspace")
def _(p):
    _pen(p)
    _fill(p, theme.SURFACE)
    path = QtGui.QPainterPath()
    path.moveTo(7, 3); path.lineTo(19, 3); path.lineTo(25, 9); path.lineTo(25, 29); path.lineTo(7, 29)
    path.closeSubpath()
    p.drawPath(path)
    p.drawPolyline([QtCore.QPointF(19, 3), QtCore.QPointF(19, 9), QtCore.QPointF(25, 9)])
    _plus(p, 16, 19, 5)


@_register("open")
def _(p):
    _pen(p)
    _fill(p, theme.ACCENT, 40)
    path = QtGui.QPainterPath()
    path.moveTo(3, 8); path.lineTo(12, 8); path.lineTo(15, 11); path.lineTo(28, 11); path.lineTo(28, 26)
    path.lineTo(3, 26); path.closeSubpath()
    p.drawPath(path)


@_register("save")
def _(p):
    _pen(p)
    _fill(p, theme.ACCENT, 40)
    p.drawRoundedRect(QtCore.QRectF(5, 5, 22, 22), 3, 3)
    _fill(p, theme.SURFACE)
    p.drawRect(QtCore.QRectF(10, 5, 12, 7))
    p.drawRect(QtCore.QRectF(9, 17, 14, 10))


@_register("add_samples")
def _(p):
    _tube(p, 14, 6, liquid=theme.TEAL)
    _tube(p, 23, 6, liquid=theme.ACCENT)
    _plus(p, 7, 17, 4.5)


@_register("create_group")
def _(p):
    _pen(p, theme.INK, 1.8)
    _fill(p, theme.ACCENT, 30)
    p.drawRoundedRect(QtCore.QRectF(4, 9, 18, 14), 3, 3)
    _fill(p, theme.SURFACE)
    p.drawRoundedRect(QtCore.QRectF(9, 4, 18, 14), 3, 3)
    _plus(p, 18, 11, 4)
    _pen(p, theme.INK, 1.8)
    p.drawLine(QtCore.QPointF(7, 27), QtCore.QPointF(25, 27))


@_register("table_editor")
def _(p):
    _pen(p, theme.INK, 1.8)
    _fill(p, theme.SURFACE)
    p.drawRoundedRect(QtCore.QRectF(4, 5, 24, 22), 2, 2)
    p.setPen(QtCore.Qt.NoPen)
    _fill(p, theme.ACCENT, 60)
    p.drawRect(QtCore.QRectF(5, 6, 22, 5))
    _pen(p, theme.INK, 1.2)
    for y in (11, 16.5, 22):
        p.drawLine(QtCore.QPointF(4, y), QtCore.QPointF(28, y))
    for x in (12, 20):
        p.drawLine(QtCore.QPointF(x, 5), QtCore.QPointF(x, 27))


@_register("unmixing")
def _(p):
    for centre, color in ((10, theme.ACCENT), (16, theme.TEAL), (22, theme.CORAL)):
        path = QtGui.QPainterPath()
        path.moveTo(2, 26)
        for i in range(41):
            x = 2 + i * 0.7
            path.lineTo(x, 26 - 18 * math.exp(-((x - centre) / 3.4) ** 2))
        _pen(p, color, 2.0)
        _fill(p, color, 45)
        p.drawPath(path)
    _pen(p, theme.INK, 1.8)
    p.drawLine(QtCore.QPointF(2, 27), QtCore.QPointF(30, 27))


@_register("transform")
def _(p):
    _pen(p, theme.INK, 2)
    p.drawPolyline([QtCore.QPointF(5, 4), QtCore.QPointF(5, 27), QtCore.QPointF(28, 27)])
    path = QtGui.QPainterPath()
    path.moveTo(7, 24)
    path.cubicTo(14, 24, 13, 7, 27, 7)
    _pen(p, theme.ACCENT, 2.4)
    p.drawPath(path)


@_register("keyword")
def _(p):
    _pen(p, theme.INK, 1.8)
    _fill(p, theme.TEAL, 50)
    path = QtGui.QPainterPath()
    path.moveTo(4, 6); path.lineTo(16, 6); path.lineTo(28, 18); path.lineTo(18, 28); path.lineTo(4, 14)
    path.closeSubpath()
    p.drawPath(path)
    _fill(p, theme.SURFACE)
    p.drawEllipse(QtCore.QPointF(10, 12), 2.2, 2.2)


@_register("import_flowjo")
def _(p):
    _pen(p, theme.INK, 1.8)
    _fill(p, theme.SURFACE)
    p.drawRoundedRect(QtCore.QRectF(11, 4, 17, 24), 2, 2)
    _pen(p, theme.ACCENT, 2.4)
    p.drawLine(QtCore.QPointF(3, 16), QtCore.QPointF(18, 16))
    p.drawPolyline([QtCore.QPointF(13, 11), QtCore.QPointF(18, 16), QtCore.QPointF(13, 21)])


@_register("demo")
def _(p):
    _pen(p, theme.INK, 1.8)
    _fill(p, theme.TEAL, 60)
    path = QtGui.QPainterPath()
    path.moveTo(12, 4); path.lineTo(12, 13); path.lineTo(5, 26); path.quadTo(4, 28, 7, 28)
    path.lineTo(25, 28); path.quadTo(28, 28, 27, 26); path.lineTo(20, 13); path.lineTo(20, 4)
    p.drawPath(path)
    p.drawLine(QtCore.QPointF(10, 4), QtCore.QPointF(22, 4))
    _scatter(p, [(13, 22), (17, 24), (19, 20)], theme.SURFACE, 1.5)


@_register("export")
def _(p):
    _pen(p, theme.INK, 1.8)
    _fill(p, theme.SURFACE)
    p.drawRoundedRect(QtCore.QRectF(4, 8, 20, 18), 2, 2)
    _scatter(p, [(9, 20), (12, 17), (14, 21), (17, 14), (11, 13)], theme.TEAL)
    _pen(p, theme.ACCENT, 2.4)
    p.drawLine(QtCore.QPointF(19, 13), QtCore.QPointF(28, 4))
    p.drawPolyline([QtCore.QPointF(22, 4), QtCore.QPointF(28, 4), QtCore.QPointF(28, 10)])


@_register("about")
def _(p):
    _pen(p, theme.ACCENT, 2.2)
    p.drawEllipse(QtCore.QPointF(16, 16), 12, 12)
    p.drawLine(QtCore.QPointF(16, 14), QtCore.QPointF(16, 23))
    _scatter(p, [(16, 9.5)], theme.ACCENT, 1.7)


@_register("help")
def _(p):
    _pen(p, theme.ACCENT, 2.6)
    path = QtGui.QPainterPath()
    path.moveTo(10, 11)
    path.cubicTo(10, 3, 22, 3, 22, 11)
    path.cubicTo(22, 16, 16, 15, 16, 21)
    p.drawPath(path)
    _scatter(p, [(16, 27)], theme.ACCENT, 1.9)


@_register("website")
def _(p):
    _pen(p, theme.TEAL, 2)
    p.drawEllipse(QtCore.QPointF(16, 16), 12, 12)
    p.drawEllipse(QtCore.QRectF(11, 4, 10, 24))
    p.drawLine(QtCore.QPointF(4, 16), QtCore.QPointF(28, 16))


@_register("license")
def _(p):
    _pen(p, theme.INK, 1.8)
    _fill(p, theme.CORAL, 60)
    p.drawEllipse(QtCore.QPointF(16, 12), 8, 8)
    path = QtGui.QPainterPath()
    path.moveTo(11, 18); path.lineTo(8, 29); path.lineTo(16, 25); path.lineTo(24, 29); path.lineTo(21, 18)
    p.drawPath(path)


@_register("rename")
def _(p):
    _pen(p, theme.INK, 1.8)
    _fill(p, theme.ACCENT, 50)
    path = QtGui.QPainterPath()
    path.moveTo(6, 26); path.lineTo(7, 20); path.lineTo(21, 6); path.lineTo(26, 11); path.lineTo(12, 25)
    path.closeSubpath()
    p.drawPath(path)


@_register("delete")
def _(p):
    _pen(p, theme.INK, 1.8)
    _fill(p, theme.CORAL, 40)
    p.drawRoundedRect(QtCore.QRectF(8, 9, 16, 19), 2, 2)
    p.drawLine(QtCore.QPointF(5, 9), QtCore.QPointF(27, 9))
    p.drawLine(QtCore.QPointF(13, 5), QtCore.QPointF(19, 5))
    for x in (13, 19):
        p.drawLine(QtCore.QPointF(x, 13), QtCore.QPointF(x, 24))


@_register("remove_sample")
def _(p):
    _tube(p, 18, 6, liquid=theme.TEAL)
    _pen(p, theme.CORAL, 2.6)
    p.drawLine(QtCore.QPointF(3, 17), QtCore.QPointF(12, 17))


@_register("quit")
def _(p):
    _pen(p, theme.INK, 2.2)
    p.drawArc(QtCore.QRectF(5, 6, 22, 22), 120 * 16, 300 * 16)
    p.drawLine(QtCore.QPointF(16, 3), QtCore.QPointF(16, 15))


# ----- gating tools ----------------------------------------------------------

_CLOUD = [(9, 22), (12, 19), (14, 23), (16, 17), (19, 20), (13, 15), (21, 14), (18, 12), (11, 25), (22, 18)]


def _axes(p):
    _pen(p, theme.MUTED, 1.4)
    p.drawPolyline([QtCore.QPointF(4, 3), QtCore.QPointF(4, 28), QtCore.QPointF(29, 28)])


def _gate_pen(p):
    _pen(p, theme.ACCENT, 2.0)
    _fill(p, theme.ACCENT, 28)


@_register("pointer")
def _(p):
    _pen(p, theme.INK, 1.6)
    _fill(p, theme.SURFACE)
    path = QtGui.QPainterPath()
    path.moveTo(10, 5); path.lineTo(10, 25); path.lineTo(15, 20); path.lineTo(18.5, 27.5)
    path.lineTo(21.5, 26); path.lineTo(18, 18.5); path.lineTo(25, 18.5); path.closeSubpath()
    p.drawPath(path)


@_register("rectangle")
def _(p):
    _axes(p)
    _scatter(p, _CLOUD)
    _gate_pen(p)
    p.drawRect(QtCore.QRectF(9, 10, 15, 13))


@_register("polygon")
def _(p):
    _axes(p)
    _scatter(p, _CLOUD)
    _gate_pen(p)
    p.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(*v) for v in
                                   ((8, 20), (13, 8), (24, 11), (25, 21), (15, 26))]))


@_register("ellipse")
def _(p):
    _axes(p)
    _scatter(p, _CLOUD)
    _gate_pen(p)
    p.save()
    p.translate(16, 18)
    p.rotate(-30)
    p.drawEllipse(QtCore.QPointF(0, 0), 10, 6.5)
    p.restore()


@_register("quadrant")
def _(p):
    _axes(p)
    _scatter(p, [(9, 9), (11, 11), (22, 22), (24, 20), (21, 10), (10, 23)])
    _pen(p, theme.ACCENT, 2.0)
    p.drawLine(QtCore.QPointF(16, 3), QtCore.QPointF(16, 28))
    p.drawLine(QtCore.QPointF(4, 16), QtCore.QPointF(29, 16))
    _fill(p, theme.SURFACE)
    p.drawEllipse(QtCore.QPointF(16, 16), 2.4, 2.4)


@_register("freehand")
def _(p):
    _axes(p)
    _scatter(p, _CLOUD)
    _gate_pen(p)
    path = QtGui.QPainterPath()
    path.moveTo(9, 21)
    path.cubicTo(5, 12, 14, 6, 19, 10)
    path.cubicTo(27, 8, 27, 20, 22, 22)
    path.cubicTo(18, 28, 11, 27, 9, 21)
    p.drawPath(path)


def _histogram(p):
    path = QtGui.QPainterPath()
    path.moveTo(4, 28)
    for i in range(51):
        x = 4 + i * 0.5
        y = 28 - 9 * math.exp(-((x - 10) / 3.0) ** 2) - 17 * math.exp(-((x - 21) / 3.2) ** 2)
        path.lineTo(x, y)
    path.lineTo(29, 28)
    _pen(p, theme.TEAL, 1.6)
    _fill(p, theme.TEAL, 60)
    p.drawPath(path)
    _axes(p)


@_register("range")
def _(p):
    _histogram(p)
    _pen(p, theme.ACCENT, 2.0)
    p.drawLine(QtCore.QPointF(16, 6), QtCore.QPointF(27, 6))
    for x in (16, 27):
        p.drawLine(QtCore.QPointF(x, 3), QtCore.QPointF(x, 28))


@_register("bisector")
def _(p):
    _histogram(p)
    _pen(p, theme.ACCENT, 2.0)
    p.drawLine(QtCore.QPointF(15, 3), QtCore.QPointF(15, 28))
    p.drawLine(QtCore.QPointF(7, 6), QtCore.QPointF(26, 6))
    _fill(p, theme.SURFACE)
    p.drawEllipse(QtCore.QPointF(15, 6), 2.2, 2.2)


@_register("axis_transform")
def _(p):
    _pen(p, theme.ACCENT, 2.6)
    p.drawLine(QtCore.QPointF(8, 7), QtCore.QPointF(24, 7))
    p.drawLine(QtCore.QPointF(16, 7), QtCore.QPointF(16, 26))


@_register("app")
def _(p):
    """Fallback app badge for the ribbon (replaced by the window icon when one is set)."""
    grad = QtGui.QLinearGradient(4, 4, 28, 28)
    grad.setColorAt(0, QtGui.QColor(theme.ACCENT))
    grad.setColorAt(1, QtGui.QColor(theme.TEAL))
    p.setPen(QtCore.Qt.NoPen)
    p.setBrush(grad)
    p.drawEllipse(QtCore.QPointF(16, 16), 14, 14)
    _scatter(p, [(10, 20), (13, 17), (12, 22), (16, 14), (19, 12), (17, 18), (21, 15), (22, 10)],
             theme.SURFACE, 1.6)


@_register("suggest")
def _(p):
    """Suggestion box: a speech bubble with a spark of an idea inside."""
    bubble = QtGui.QPainterPath()
    bubble.addRoundedRect(QtCore.QRectF(3, 4, 26, 18), 6, 6)
    tail = QtGui.QPainterPath()
    tail.moveTo(9, 20)
    tail.lineTo(8, 28)
    tail.lineTo(16, 21)
    tail.closeSubpath()
    _pen(p, theme.ACCENT, 2.0)
    _fill(p, theme.ACCENT_SOFT)
    p.drawPath(bubble.united(tail))
    spark = QtGui.QPolygonF([QtCore.QPointF(x, y) for x, y in
                             ((18, 6.5), (11.5, 14), (15.5, 14), (13.5, 19.5), (20.5, 11.5),
                              (16.5, 11.5))])
    p.setPen(QtCore.Qt.NoPen)
    _fill(p, theme.CORAL)
    p.drawPolygon(spark)
