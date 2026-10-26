"""Build a standalone app with PyInstaller: `python packaging/build.py`.

Produces dist/Sidescatter/ (Windows, Linux) or dist/Sidescatter.app (macOS).
"""

import shutil
import sys
from pathlib import Path

import PyInstaller.__main__

HERE = Path(__file__).resolve().parent
ICONS = HERE.parent / "src" / "sidescatter" / "resources" / "icons"

args = [
    str(HERE / "entry.py"),
    "--name", "Sidescatter",
    "--windowed",
    "--noconfirm",
    "--collect-data", "sidescatter",
    "--collect-data", "flowkit",
    "--collect-submodules", "flowkit",
    "--exclude-module", "PySide6.QtWebEngineCore",
    "--exclude-module", "PySide6.QtWebEngineWidgets",
    "--exclude-module", "PySide6.Qt3DCore",
    "--exclude-module", "tkinter",
    "--exclude-module", "pyarrow",  # optional for pandas, 160 MB
]
if sys.platform == "darwin":
    args += ["--osx-bundle-identifier", "org.sidescatter.app", "--icon", str(ICONS / "sidescatter.icns")]
elif sys.platform == "win32":
    args += ["--icon", str(ICONS / "sidescatter.ico")]

PyInstaller.__main__.run(args)

if sys.platform.startswith("linux"):
    # A desktop entry and icon next to the executable, for menus and docks.
    dist = Path.cwd() / "dist" / "Sidescatter"  # PyInstaller's default output folder
    shutil.copy(ICONS / "sidescatter-256.png", dist / "sidescatter.png")
    shutil.copy(HERE / "linux" / "sidescatter.desktop", dist / "sidescatter.desktop")
