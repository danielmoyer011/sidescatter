"""App icon, bundled fonts and the animated splash screen.

The splash shows what the icon stands for: cells flow single file down a flow
cell, cross the five Aurora laser beams at the interrogation points, and throw
side scatter in each laser's colour. Every cell that clears the last laser
lands as events on a small dot plot.
"""

import math
import random
import threading
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from .. import __version__

RESOURCES = Path(__file__).resolve().parents[1] / "resources"
ICONS = RESOURCES / "icons"
TAGLINE = "Free spectral flow cytometry analysis"

# Aurora laser lines, shortest wavelength first. Same colours as the icon.
LASERS = [
    ("355", QtGui.QColor("#b44dff")),
    ("405", QtGui.QColor("#5b4dff")),
    ("488", QtGui.QColor("#159cff")),
    ("561", QtGui.QColor("#a8dc1e")),
    ("640", QtGui.QColor("#ff3d2e")),
]
# Pseudocolour ramp for the dot plot, low to high density.
DENSITY = [QtGui.QColor(c) for c in ("#2f5bff", "#18b6e0", "#3fd18a", "#e8d935", "#ff8a2a", "#ff3d3d")]

_font_family = None


def app_icon():
    icon = QtGui.QIcon()
    for px in (16, 24, 32, 48, 64, 128, 256, 512, 1024):
        icon.addFile(str(ICONS / f"sidescatter-{px}.png"), QtCore.QSize(px, px))
    return icon


def brand_font(point_size, weight=QtGui.QFont.DemiBold):
    """Sora (bundled, SIL OFL) for the wordmark, falling back to the system sans."""
    global _font_family
    if _font_family is None:
        font_id = QtGui.QFontDatabase.addApplicationFont(str(RESOURCES / "fonts" / "Sora.ttf"))
        families = QtGui.QFontDatabase.applicationFontFamilies(font_id) if font_id >= 0 else []
        _font_family = families[0] if families else ""
    font = QtGui.QFont(_font_family) if _font_family else QtGui.QFont()
    font.setPointSizeF(point_size)
    font.setWeight(weight)
    return font


class _Cell:
    __slots__ = ("y", "r", "population", "logged")

    def __init__(self, y, rnd):
        self.y = y
        self.r = rnd.uniform(6.0, 9.5)
        self.population = 0 if rnd.random() < 0.6 else 1
        self.logged = False


class SplashScreen(QtWidgets.QWidget):
    """Frameless animated splash. Click it to dismiss early."""

    W, H = 720, 400
    CHANNEL_X = 196  # centre of the flow cell
    BEAM_Y = [112, 160, 208, 256, 304]  # interrogation points, one per laser
    SPEED = 260.0  # px/s
    PLOT = QtCore.QRectF(500, 230, 180, 120)

    def __init__(self, min_seconds=1.6):
        super().__init__(None, QtCore.Qt.SplashScreen | QtCore.Qt.FramelessWindowHint
                         | QtCore.Qt.WindowStaysOnTopHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setFixedSize(self.W, self.H)
        self.min_seconds = min_seconds
        self.status = "Starting"
        self._rnd = random.Random(11)
        self._cells = []
        self._dots = []
        self._events = 0
        self._next_spawn = 0.0
        self._t = 0.0
        self._clock = QtCore.QElapsedTimer()
        self._timer = QtCore.QTimer(self, interval=33, timeout=self._tick)
        self._fade = None
        # Start mid-run so the first frame already has cells in the stream
        # and a population on the plot.
        for _ in range(40):
            self._log_event(_Cell(0, self._rnd))
        for _ in range(90):
            self._advance(1 / 60)

    # -- public -------------------------------------------------------------------

    def start(self):
        screen = QtGui.QGuiApplication.primaryScreen().availableGeometry()
        self.move(screen.center() - self.rect().center())
        self._clock.start()
        self._timer.start()
        self.show()
        QtWidgets.QApplication.processEvents()

    def run_while(self, fn):
        """Run fn() on a worker thread (e.g. heavy imports) and keep animating
        until it is done and the splash has shown for min_seconds."""
        result = {}

        def work():
            try:
                result["value"] = fn()
            except BaseException as exc:  # re-raised on the main thread
                result["error"] = exc

        worker = threading.Thread(target=work, daemon=True)
        worker.start()
        loop = QtCore.QEventLoop()
        poll = QtCore.QTimer(interval=30)
        poll.timeout.connect(lambda: (not worker.is_alive()
                                      and (not self.isVisible()
                                           or self._clock.elapsed() >= self.min_seconds * 1000)
                                      and loop.quit()))
        poll.start()
        loop.exec()
        poll.stop()
        if "error" in result:
            raise result["error"]
        return result.get("value")

    def set_status(self, text):
        self.status = text
        self.update()

    def finish(self, window):
        """Fade out once the main window is up."""
        window.raise_()
        self._fade = QtCore.QPropertyAnimation(self, b"windowOpacity", self)
        self._fade.setDuration(260)
        self._fade.setStartValue(1.0)
        self._fade.setEndValue(0.0)
        self._fade.finished.connect(self.close)
        self._fade.start()

    def mousePressEvent(self, event):
        self.hide()

    def closeEvent(self, event):
        self._timer.stop()
        super().closeEvent(event)

    # -- simulation ---------------------------------------------------------------

    def _tick(self):
        now = self._clock.elapsed() / 1000
        dt = min(0.05, now - getattr(self, "_last", now))
        self._last = now
        self._advance(dt)
        self.update()

    def _advance(self, dt):
        self._t += dt
        if self._t >= self._next_spawn:
            self._cells.append(_Cell(-12, self._rnd))
            self._next_spawn = self._t + self._rnd.uniform(0.07, 0.22)
        for cell in self._cells:
            cell.y += self.SPEED * dt
            if not cell.logged and cell.y > self.BEAM_Y[-1] + 14:
                cell.logged = True
                self._log_event(cell)
        self._cells = [c for c in self._cells if c.y < self.H - 30]

    def _log_event(self, cell):
        # One drawn cell stands for many acquired events.
        self._events += self._rnd.randint(380, 920)
        mx, my, sx, sy, rho = ((0.30, 0.68, 0.08, 0.08, 0.2), (0.70, 0.32, 0.09, 0.08, -0.4))[cell.population]
        for _ in range(7):
            u, v = self._rnd.gauss(0, 1), self._rnd.gauss(0, 1)
            x = mx + sx * u
            y = my + sy * (rho * u + math.sqrt(1 - rho * rho) * v)
            if 0.03 < x < 0.97 and 0.03 < y < 0.97:
                self._dots.append((x, y, math.exp(-0.5 * (u * u + v * v))))

    # -- drawing ------------------------------------------------------------------

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        # Static parts and the dot plot are cached so the heavy imports on the
        # worker thread get most of the GIL.
        p.drawPixmap(0, 0, self._static_layer())
        p.drawPixmap(self.PLOT.topLeft(), self._plot_layer())
        self._draw_cells(p)
        self._draw_live_text(p)
        p.end()

    def _layer(self, w, h):
        ratio = self.devicePixelRatioF()
        pixmap = QtGui.QPixmap(int(w * ratio), int(h * ratio))
        pixmap.setDevicePixelRatio(ratio)
        pixmap.fill(QtCore.Qt.transparent)
        return pixmap

    def _static_layer(self):
        if getattr(self, "_static", None) is None:
            self._static = self._layer(self.W, self.H)
            p = QtGui.QPainter(self._static)
            p.setRenderHint(QtGui.QPainter.Antialiasing)
            bg = QtGui.QLinearGradient(0, 0, 0, self.H)
            bg.setColorAt(0, QtGui.QColor("#1b3657"))
            bg.setColorAt(1, QtGui.QColor("#0b1626"))
            p.setPen(QtCore.Qt.NoPen)
            p.setBrush(bg)
            p.drawRoundedRect(QtCore.QRectF(0, 0, self.W, self.H), 18, 18)
            self._draw_flow_cell(p)
            self._draw_text(p)
            self._draw_axes(p)
            p.end()
        return self._static

    def _plot_layer(self):
        if getattr(self, "_plot", None) is None:
            self._plot = self._layer(self.PLOT.width(), self.PLOT.height())
            self._plotted = 0
        if self._plotted < len(self._dots):
            p = QtGui.QPainter(self._plot)
            p.setRenderHint(QtGui.QPainter.Antialiasing)
            p.setPen(QtCore.Qt.NoPen)
            w, h = self.PLOT.width(), self.PLOT.height()
            for x, y, d in self._dots[self._plotted:]:
                p.setBrush(DENSITY[min(5, int(d * 6))])
                p.drawEllipse(QtCore.QPointF(x * w, y * h), 1.7, 1.7)
            p.end()
            self._plotted = len(self._dots)
        return self._plot

    def _draw_flow_cell(self, p):
        cx = self.CHANNEL_X
        # sheath stream and cuvette walls
        stream = QtGui.QLinearGradient(cx - 22, 0, cx + 22, 0)
        stream.setColorAt(0, QtGui.QColor(160, 200, 255, 0))
        stream.setColorAt(0.5, QtGui.QColor(160, 200, 255, 34))
        stream.setColorAt(1, QtGui.QColor(160, 200, 255, 0))
        p.setBrush(stream)
        p.drawRect(QtCore.QRectF(cx - 22, 0, 44, self.H))
        p.setPen(QtGui.QPen(QtGui.QColor(190, 214, 245, 70), 1.5))
        for dx in (-30, 30):
            p.drawLine(QtCore.QPointF(cx + dx, 24), QtCore.QPointF(cx + dx, self.H - 24))

        # laser beams, focused on the stream
        for (nm, colour), y in zip(LASERS, self.BEAM_Y):
            grad = QtGui.QLinearGradient(28, 0, cx, 0)
            faint = QtGui.QColor(colour)
            faint.setAlpha(0)
            grad.setColorAt(0, faint)
            grad.setColorAt(1, colour)
            p.setPen(QtGui.QPen(QtGui.QBrush(grad), 3.0, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
            p.drawLine(QtCore.QPointF(36, y), QtCore.QPointF(cx, y))
            past = QtGui.QColor(colour)
            past.setAlpha(50)
            p.setPen(QtGui.QPen(past, 2.0))
            p.drawLine(QtCore.QPointF(cx, y), QtCore.QPointF(cx + 60, y))
            p.setPen(QtGui.QColor(150, 170, 200))
            p.setFont(brand_font(7.5, QtGui.QFont.Medium))
            p.drawText(QtCore.QRectF(0, y - 8, 30, 16), QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter, nm)

    def _draw_cells(self, p):
        """Cells light up and scatter as they cross each beam."""
        cx = self.CHANNEL_X
        for cell in self._cells:
            glow_colour, glow = None, 0.0
            for (_, colour), y in zip(LASERS, self.BEAM_Y):
                g = max(0.0, 1 - abs(cell.y - y) / 16)
                if g > glow:
                    glow, glow_colour = g, colour
            centre = QtCore.QPointF(cx, cell.y)
            if glow > 0:
                self._draw_scatter(p, centre, glow_colour, glow)
            p.setPen(QtCore.Qt.NoPen)
            body = QtGui.QColor(255, 255, 255)
            if glow > 0:
                body = _mix(body, glow_colour, 0.45 * glow)
            p.setBrush(body)
            p.drawEllipse(centre, cell.r, cell.r)
            p.setBrush(QtGui.QColor(201, 214, 232) if glow == 0 else _mix(QtGui.QColor(201, 214, 232), glow_colour, 0.6 * glow))
            p.drawEllipse(centre + QtCore.QPointF(cell.r * 0.25, -cell.r * 0.2), cell.r * 0.4, cell.r * 0.4)

    def _draw_scatter(self, p, centre, colour, strength):
        halo = QtGui.QRadialGradient(centre, 26)
        c = QtGui.QColor(colour)
        c.setAlphaF(0.55 * strength)
        halo.setColorAt(0, c)
        c2 = QtGui.QColor(colour)
        c2.setAlpha(0)
        halo.setColorAt(1, c2)
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(halo)
        p.drawEllipse(centre, 26, 26)
        # side scatter: rays leaving at about 90 degrees to the beam, towards the detectors
        for angle in (-24, -8, 8, 24):
            a = math.radians(angle)
            start = centre + QtCore.QPointF(14 * math.cos(a), 14 * math.sin(a))
            length = 30 + 70 * strength
            end = centre + QtCore.QPointF(length * math.cos(a), length * math.sin(a))
            grad = QtGui.QLinearGradient(start, end)
            head = QtGui.QColor(colour)
            head.setAlphaF(0.9 * strength)
            tail = QtGui.QColor(colour)
            tail.setAlpha(0)
            grad.setColorAt(0, head)
            grad.setColorAt(1, tail)
            p.setPen(QtGui.QPen(QtGui.QBrush(grad), 3.0, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
            p.drawLine(start, end)

    def _draw_text(self, p):
        left = 340
        p.setPen(QtGui.QColor("#ffffff"))
        p.setFont(brand_font(30))
        p.drawText(QtCore.QRectF(left, 74, 360, 52), QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, "Sidescatter")
        p.setPen(QtGui.QColor("#a9b8cf"))
        p.setFont(brand_font(11, QtGui.QFont.Normal))
        p.drawText(QtCore.QRectF(left + 2, 126, 360, 24), QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, TAGLINE)
        p.setPen(QtGui.QColor("#7f90aa"))
        p.setFont(brand_font(8.5, QtGui.QFont.Normal))
        p.drawText(QtCore.QRectF(left + 2, 230, 140, 20), QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
                   "Events")

    def _draw_live_text(self, p):
        left = 340
        p.setPen(QtGui.QColor("#7f90aa"))
        p.setFont(brand_font(8.5, QtGui.QFont.Normal))
        p.drawText(QtCore.QRectF(left + 2, self.H - 46, 300, 20), QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
                   f"Version {__version__}  ·  {self.status}…")
        p.setPen(QtGui.QColor("#ffffff"))
        p.setFont(brand_font(16, QtGui.QFont.Medium))
        p.drawText(QtCore.QRectF(left + 2, 250, 150, 30), QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
                   f"{self._events:,}")

    def _draw_axes(self, p):
        r = self.PLOT
        p.setPen(QtGui.QPen(QtGui.QColor(143, 163, 191, 150), 1.5))
        p.drawLine(r.bottomLeft(), r.topLeft())
        p.drawLine(r.bottomLeft(), r.bottomRight())
        p.setPen(QtGui.QColor("#7f90aa"))
        p.setFont(brand_font(7.5, QtGui.QFont.Medium))
        p.drawText(QtCore.QRectF(r.left(), r.bottom() + 4, r.width(), 14), QtCore.Qt.AlignRight, "SSC-A")


def _mix(a, b, t):
    return QtGui.QColor(
        int(a.red() + (b.red() - a.red()) * t),
        int(a.green() + (b.green() - a.green()) * t),
        int(a.blue() + (b.blue() - a.blue()) * t),
    )
