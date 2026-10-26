"""Sidescatter's colour palette and Qt stylesheet.

FlowJo 10's layout with a calmer palette: a soft grey-blue window, white work
surfaces, iris as the main accent, teal and coral as supporting accents.
"""

import tempfile
from pathlib import Path

import pyqtgraph as pg
from PySide6 import QtGui, QtWidgets

# ----- palette ---------------------------------------------------------------

INK = "#1D2130"          # main text
MUTED = "#6A7187"        # secondary text, captions
FAINT = "#A3A9BA"        # disabled text, watermark
WINDOW = "#F6F7FB"       # window and ribbon background
SURFACE = "#FFFFFF"      # tables, plots, dialogs
ALT_ROW = "#F9FAFD"      # alternating table rows
BORDER = "#E3E6EF"
BORDER_STRONG = "#CDD2DF"
ACCENT = "#5B5BD6"       # iris: selection, active tool, focus
ACCENT_DARK = "#4646B8"
ACCENT_SOFT = "#ECECFB"  # selected rows, hovered buttons
TEAL = "#14A3A3"
CORAL = "#F26D5B"        # "add" marks on icons
GATE = "#1D2130"         # gate outlines: dark ink reads on every density colour

# Density ramp for pseudocolour plots: the usual blue -> green -> yellow -> red
# order cytometrists expect, in softer, evenly bright steps.
PSEUDOCOLOR = pg.ColorMap(
    [0.0, 0.2, 0.45, 0.7, 0.88, 1.0],
    [(64, 70, 196), (40, 150, 230), (30, 190, 150), (170, 215, 60), (250, 190, 50), (232, 72, 64)],
)


def qcolor(hex_color, alpha=255):
    c = QtGui.QColor(hex_color)
    c.setAlpha(alpha)
    return c


STYLESHEET = f"""
QWidget {{
    color: {INK};
    font-size: 10pt;
}}
QMainWindow, QDialog {{
    background: {WINDOW};
}}
QToolTip {{
    color: {INK};
    background: {SURFACE};
    border: 1px solid {BORDER_STRONG};
    padding: 4px 6px;
}}

/* ----- ribbon ----- */
#QuickBar {{
    background: {SURFACE};
}}
#Ribbon {{
    background: {WINDOW};
    border-bottom: 1px solid {BORDER};
}}
#RibbonTabs {{
    background: {SURFACE};
}}
#RibbonTabs QTabBar::tab {{
    background: transparent;
    color: {INK};
    padding: 6px 18px 7px 18px;
    border: none;
    border-bottom: 2px solid transparent;
}}
#RibbonTabs QTabBar::tab:hover {{
    color: {ACCENT};
}}
#RibbonTabs QTabBar::tab:selected {{
    color: {ACCENT_DARK};
    border-bottom: 2px solid {ACCENT};
    font-weight: 600;
}}
#RibbonPage {{
    background: {WINDOW};
    border-top: 1px solid {BORDER};
}}
#RibbonButton, #QuickButton {{
    background: transparent;
    border: 1px solid transparent;
    border-radius: 6px;
    padding: 3px 6px;
}}
#RibbonButton:hover, #QuickButton:hover {{
    background: {ACCENT_SOFT};
    border-color: #D9D9F6;
}}
#RibbonButton:pressed, #QuickButton:pressed {{
    background: #DCDCF7;
}}
#RibbonButton:disabled {{
    color: {FAINT};
}}
#RibbonButton::menu-indicator {{
    subcontrol-position: bottom center;
    subcontrol-origin: padding;
    bottom: -2px;
}}
#RibbonCaption {{
    color: {MUTED};
    font-size: 9pt;
}}
#RibbonDivider {{
    background: {BORDER};
}}

/* ----- tables ----- */
QTreeView, QTableView, QListView {{
    background: {SURFACE};
    alternate-background-color: {ALT_ROW};
    border: 1px solid {BORDER};
    selection-background-color: {ACCENT_SOFT};
    selection-color: {INK};
    outline: none;
}}
QTreeView::item, QTableView::item {{
    padding: 2px 0px;
}}
QTreeView::item:selected, QTableView::item:selected {{
    background: {ACCENT_SOFT};
    color: {INK};
}}
QTreeView::item:hover {{
    background: #F2F2FD;
}}
QHeaderView::section {{
    background: {WINDOW};
    color: {MUTED};
    border: none;
    border-right: 1px solid {BORDER};
    border-bottom: 1px solid {BORDER};
    padding: 4px 8px;
}}
QSplitter::handle {{
    background: {WINDOW};
}}
QSplitter::handle:vertical {{
    height: 6px;
}}

/* ----- inputs ----- */
QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {{
    background: {SURFACE};
    border: 1px solid {BORDER_STRONG};
    border-radius: 5px;
    padding: 3px 8px;
    min-height: 18px;
}}
QComboBox:hover, QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover {{
    border-color: {ACCENT};
}}
QComboBox:focus, QLineEdit:focus {{
    border-color: {ACCENT};
}}
QComboBox::drop-down {{
    border: none;
    width: 18px;
}}
QComboBox::down-arrow {{
    image: url(@ARROW@);
    width: 10px;
    height: 6px;
    margin-right: 6px;
}}
QComboBox QAbstractItemView {{
    background: {SURFACE};
    border: 1px solid {BORDER_STRONG};
    selection-background-color: {ACCENT_SOFT};
    selection-color: {INK};
}}
QPushButton {{
    background: {SURFACE};
    border: 1px solid {BORDER_STRONG};
    border-radius: 6px;
    padding: 5px 14px;
}}
QPushButton:hover {{
    border-color: {ACCENT};
    color: {ACCENT_DARK};
}}
QPushButton:default {{
    background: {ACCENT};
    border-color: {ACCENT};
    color: white;
}}
QPushButton:default:hover {{
    background: {ACCENT_DARK};
}}
QPushButton:disabled {{
    color: {FAINT};
    border-color: {BORDER};
}}

/* ----- graph window ----- */
#GraphHeader {{
    background: {SURFACE};
    border-bottom: 1px solid {BORDER};
}}
#GraphTitle {{
    font-weight: 600;
}}
#GraphCount {{
    color: {MUTED};
}}
#GateBar {{
    background: {SURFACE};
    border-bottom: 1px solid {BORDER};
}}
#GateTool {{
    background: transparent;
    border: 1px solid transparent;
    border-radius: 6px;
    padding: 3px;
}}
#GateTool:hover {{
    background: {ACCENT_SOFT};
}}
#GateTool:checked {{
    background: {ACCENT_SOFT};
    border: 1px solid {ACCENT};
}}
#GateTool:disabled {{
    background: transparent;
}}
#NavButton {{
    background: transparent;
    border: 1px solid {BORDER_STRONG};
    border-radius: 5px;
    padding: 2px 6px;
}}
#NavButton:hover {{
    border-color: {ACCENT};
}}
#GraphHint {{
    color: {MUTED};
    font-size: 9pt;
}}

/* ----- status bar ----- */
QStatusBar {{
    background: {WINDOW};
    color: {MUTED};
    border-top: 1px solid {BORDER};
}}
QStatusBar::item {{
    border: none;
}}
#LicenseButton {{
    padding: 3px 14px;
    margin: 2px 4px;
}}
QMenu {{
    background: {SURFACE};
    border: 1px solid {BORDER_STRONG};
    padding: 4px;
}}
QMenu::item {{
    padding: 5px 22px 5px 22px;
    border-radius: 4px;
}}
QMenu::item:selected {{
    background: {ACCENT_SOFT};
    color: {INK};
}}
QMenu::separator {{
    height: 1px;
    background: {BORDER};
    margin: 4px 6px;
}}
"""


def apply(app):
    """Use the same look on Windows, macOS and Linux."""
    app.setStyle(QtWidgets.QStyleFactory.create("Fusion"))
    pal = app.palette()
    for role, color in (
        (QtGui.QPalette.Window, WINDOW),
        (QtGui.QPalette.Base, SURFACE),
        (QtGui.QPalette.AlternateBase, ALT_ROW),
        (QtGui.QPalette.Text, INK),
        (QtGui.QPalette.WindowText, INK),
        (QtGui.QPalette.ButtonText, INK),
        (QtGui.QPalette.Button, SURFACE),
        (QtGui.QPalette.Highlight, ACCENT),
        (QtGui.QPalette.HighlightedText, "#FFFFFF"),
        (QtGui.QPalette.ToolTipBase, SURFACE),
        (QtGui.QPalette.ToolTipText, INK),
        (QtGui.QPalette.PlaceholderText, FAINT),
    ):
        pal.setColor(role, QtGui.QColor(color))
    pal.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.Text, QtGui.QColor(FAINT))
    pal.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, QtGui.QColor(FAINT))
    app.setPalette(pal)
    app.setStyleSheet(STYLESHEET.replace("@ARROW@", _arrow_file()))
    pg.setConfigOptions(antialias=True, foreground=MUTED, background=SURFACE)


def _arrow_file():
    """Stylesheets can only load images from files, so write the combo-box arrow once."""
    path = Path(tempfile.gettempdir()) / "sidescatter-arrow.svg"
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="10" height="6" viewBox="0 0 10 6">'
           f'<path d="M1 1l4 4 4-4" fill="none" stroke="{MUTED}" stroke-width="1.6" '
           f'stroke-linecap="round" stroke-linejoin="round"/></svg>')
    try:
        if not path.exists() or path.read_text() != svg:
            path.write_text(svg)
    except OSError:
        return ""
    return path.as_posix()
