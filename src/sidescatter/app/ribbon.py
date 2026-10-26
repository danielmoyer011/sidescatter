"""A FlowJo 10-style ribbon: quick-access bar, tabs, and labelled groups of big buttons."""

from PySide6 import QtCore, QtGui, QtWidgets

from . import icons


class AppButton(QtWidgets.QToolButton):
    """The round badge at the top-left. Shows the window icon once the app has one."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(40, 40)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.setStyleSheet("QToolButton { border: none; background: transparent; }"
                           "QToolButton::menu-indicator { image: none; }")

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        app_icon = QtWidgets.QApplication.windowIcon()
        ic = app_icon if not app_icon.isNull() else icons.icon("app", 36)
        ic.paint(p, self.rect().adjusted(2, 2, -2, -2))


class Ribbon(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Ribbon")
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)

        self.app_button = AppButton()
        self.quick = QtWidgets.QHBoxLayout()
        self.quick.setSpacing(2)

        quick_bar = QtWidgets.QWidget()
        quick_bar.setObjectName("QuickBar")
        quick_bar.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        top = QtWidgets.QHBoxLayout(quick_bar)
        top.setContentsMargins(6, 4, 8, 0)
        top.addWidget(self.app_button)
        top.addSpacing(6)
        top.addLayout(self.quick)
        top.addStretch()

        self.tabs = QtWidgets.QTabBar()
        self.tabs.setExpanding(False)
        self.tabs.setDrawBase(False)
        self.corner = QtWidgets.QHBoxLayout()
        self.corner.setSpacing(2)
        tab_row = QtWidgets.QWidget()
        tab_row.setObjectName("RibbonTabs")
        tab_row.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        row = QtWidgets.QHBoxLayout(tab_row)
        row.setContentsMargins(52, 0, 8, 0)
        row.addWidget(self.tabs)
        row.addStretch()
        row.addLayout(self.corner)

        self.pages = QtWidgets.QStackedWidget()
        self.tabs.currentChanged.connect(self.pages.setCurrentIndex)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(quick_bar)
        layout.addWidget(tab_row)
        layout.addWidget(self.pages)

    def add_quick(self, action):
        b = QtWidgets.QToolButton()
        b.setObjectName("QuickButton")
        b.setDefaultAction(action)
        b.setIconSize(QtCore.QSize(20, 20))
        b.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        self.quick.addWidget(b)
        return b

    def add_corner(self, action):
        b = QtWidgets.QToolButton()
        b.setObjectName("QuickButton")
        b.setDefaultAction(action)
        b.setIconSize(QtCore.QSize(18, 18))
        b.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        self.corner.addWidget(b)
        return b

    def add_tab(self, title):
        page = RibbonPage()
        self.pages.addWidget(page)
        self.tabs.addTab(title)
        return page


class RibbonPage(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("RibbonPage")
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self._row = QtWidgets.QHBoxLayout(self)
        self._row.setContentsMargins(6, 4, 6, 2)
        self._row.setSpacing(0)
        self._row.addStretch()

    def add_group(self, caption, actions):
        """A column of big buttons with a caption underneath, then a divider."""
        if self._row.count() > 1:
            line = QtWidgets.QFrame()
            line.setObjectName("RibbonDivider")
            line.setFixedWidth(1)
            self._row.insertWidget(self._row.count() - 1, line)
        box = QtWidgets.QWidget()
        col = QtWidgets.QVBoxLayout(box)
        col.setContentsMargins(4, 0, 4, 0)
        col.setSpacing(0)
        buttons = QtWidgets.QHBoxLayout()
        buttons.setSpacing(2)
        made = []
        for action in actions:
            b = QtWidgets.QToolButton()
            b.setObjectName("RibbonButton")
            b.setDefaultAction(action)
            b.setIconSize(QtCore.QSize(32, 32))
            b.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
            b.setMinimumWidth(64)
            b.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
            if action.menu():
                b.setPopupMode(QtWidgets.QToolButton.InstantPopup)
            buttons.addWidget(b)
            made.append(b)
        col.addLayout(buttons, 1)
        label = QtWidgets.QLabel(caption)
        label.setObjectName("RibbonCaption")
        label.setAlignment(QtCore.Qt.AlignCenter)
        col.addWidget(label)
        self._row.insertWidget(self._row.count() - 1, box)
        return made
