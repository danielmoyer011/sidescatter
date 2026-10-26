"""Draw the Sidescatter app icon and export it for every platform.

    python scripts/make_icons.py

Writes the SVG masters, the PNG sizes, sidescatter.ico (Windows) and
sidescatter.icns (macOS) into src/sidescatter/resources/icons/. Needs PySide6
(for SVG rendering) and Pillow (for .ico/.icns).

The mark: a laser beam crosses a cell and light leaves it at 90 degrees, split
into the five Cytek Aurora laser lines (355, 405, 488, 561, 640 nm).
"""

import io
import math
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "src" / "sidescatter" / "resources" / "icons"

# Display colours for the Aurora laser lines, shortest wavelength first.
LASERS = [
    ("355 nm UV", "#b44dff"),
    ("405 nm violet", "#5b4dff"),
    ("488 nm blue", "#159cff"),
    ("561 nm yellow-green", "#a8dc1e"),
    ("640 nm red", "#ff3d2e"),
]
TILE_TOP = "#1b3657"
TILE_BOTTOM = "#0b1626"


def mark(cx=512, cy=700, r=92):
    """The beam, scatter rays and cell, drawn on a 1024 canvas."""
    rays = []
    spread = 21  # degrees between rays, fanned around straight up (side scatter)
    for i, (_, colour) in enumerate(LASERS):
        a = math.radians(-90 + (i - 2) * spread)
        x1, y1 = cx + (r + 62) * math.cos(a), cy + (r + 62) * math.sin(a)
        x2, y2 = cx + 430 * math.cos(a), cy + 430 * math.sin(a)
        rays.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{colour}" stroke-width="50" stroke-linecap="round"/>'
        )
    return f"""<rect x="64" y="{cy - 13}" width="{cx - 64}" height="26" fill="url(#beam)"/>
<rect x="{cx}" y="{cy - 9}" width="{960 - cx}" height="18" fill="url(#forward)"/>
{chr(10).join(rays)}
<circle cx="{cx}" cy="{cy}" r="{r + 18}" fill="#ffffff" opacity="0.14"/>
<circle cx="{cx}" cy="{cy}" r="{r}" fill="#ffffff"/>
<circle cx="{cx + 24}" cy="{cy - 20}" r="{r * 0.38:.0f}" fill="#c9d6e8"/>"""


def icon_svg(inset, radius):
    """A full icon: the tile is a rounded square `inset` px in from each edge."""
    size = 1024 - 2 * inset
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1024 1024" width="1024" height="1024">
<defs>
<linearGradient id="tile" x1="0" y1="0" x2="0" y2="1">
<stop offset="0" stop-color="{TILE_TOP}"/><stop offset="1" stop-color="{TILE_BOTTOM}"/>
</linearGradient>
<linearGradient id="beam" x1="0" y1="0" x2="1" y2="0">
<stop offset="0" stop-color="#9fc2ff" stop-opacity="0"/><stop offset="1" stop-color="#e4eeff"/>
</linearGradient>
<linearGradient id="forward" x1="0" y1="0" x2="1" y2="0">
<stop offset="0" stop-color="#e4eeff" stop-opacity="0.55"/><stop offset="1" stop-color="#9fc2ff" stop-opacity="0"/>
</linearGradient>
</defs>
<rect x="{inset}" y="{inset}" width="{size}" height="{size}" rx="{radius}" fill="url(#tile)"/>
<g transform="translate(512 512) scale({size / 896:.4f}) translate(-512 -512)">
{mark()}
</g>
</svg>
"""


def render_png(svg_text, px):
    from PySide6 import QtCore, QtGui, QtSvg

    renderer = QtSvg.QSvgRenderer(QtCore.QByteArray(svg_text.encode()))
    image = QtGui.QImage(px, px, QtGui.QImage.Format_ARGB32)
    image.fill(QtCore.Qt.transparent)
    painter = QtGui.QPainter(image)
    painter.setRenderHint(QtGui.QPainter.Antialiasing)
    renderer.render(painter)
    painter.end()
    buf = QtCore.QBuffer()
    buf.open(QtCore.QIODevice.WriteOnly)
    image.save(buf, "PNG")
    return bytes(buf.data())


def main():
    from PIL import Image
    from PySide6 import QtGui

    _app = QtGui.QGuiApplication(sys.argv)  # noqa: F841 - needed for rendering
    OUT.mkdir(parents=True, exist_ok=True)

    # Windows and Linux: tile fills most of the canvas.
    full = icon_svg(inset=48, radius=200)
    # macOS: Apple's grid puts an 824 px rounded square inside the 1024 canvas.
    macos = icon_svg(inset=100, radius=185)
    (OUT / "sidescatter.svg").write_text(full)
    (OUT / "sidescatter-macos.svg").write_text(macos)

    for px in (16, 24, 32, 48, 64, 128, 256, 512, 1024):
        (OUT / f"sidescatter-{px}.png").write_bytes(render_png(full, px))

    # Render each .ico size separately so small sizes stay sharp.
    ico_sizes = [16, 24, 32, 48, 64, 128, 256]
    frames = [Image.open(io.BytesIO(render_png(full, px))) for px in ico_sizes]
    frames[-1].save(OUT / "sidescatter.ico", format="ICO", sizes=[(s, s) for s in ico_sizes],
                    append_images=frames[:-1])

    mac_big = Image.open(io.BytesIO(render_png(macos, 1024)))
    mac_big.save(OUT / "sidescatter.icns", format="ICNS")
    print(f"Icons written to {OUT}")


if __name__ == "__main__":
    main()
