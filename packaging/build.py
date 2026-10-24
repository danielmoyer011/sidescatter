"""Build a standalone app with PyInstaller: `python packaging/build.py`.

Produces dist/Sidescatter/ (Windows, Linux) or dist/Sidescatter.app (macOS).
"""

import sys
from pathlib import Path

import PyInstaller.__main__

HERE = Path(__file__).resolve().parent

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
    args += ["--osx-bundle-identifier", "org.sidescatter.app"]

PyInstaller.__main__.run(args)
