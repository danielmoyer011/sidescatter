"""The workspace: a group table on top, and the samples with their gating trees below.

Laid out like FlowJo 10's workspace window: Group | Size | Role, then
Name | Statistic | #Cells, with a splitter between the two.
"""

import json

from PySide6 import QtCore, QtGui, QtWidgets

from . import icons, theme
from ..core.workspace import ALL_SAMPLES, ROOT
from .boolean_editor import BooleanDialog, confirm_delete
from .gating import service as gating

ROLE = QtCore.Qt.UserRole
# Dragging populations from the sample table (e.g. onto a layout) carries this JSON:
# a list of {"sample_id": ..., "population": [full gate path from "root"]}.
POPULATION_MIME = "application/x-sidescatter-populations"


class WorkspacePanel(QtWidgets.QWidget):
    populationSelected = QtCore.Signal(str, object, object)  # sample_id, name, parent_path
    populationActivated = QtCore.Signal(str, object, object)  # double-click: open a graph
    groupChanged = QtCore.Signal(str)
    changed = QtCore.Signal()

    def __init__(self, workspace, parent=None):
        super().__init__(parent)
        self.ws = workspace

        self.groups = QtWidgets.QTreeWidget()
        self.groups.setHeaderLabels(["Group", "Size", "Role"])
        self.groups.setRootIsDecorated(False)
        self.groups.setUniformRowHeights(True)
        self.groups.setIconSize(QtCore.QSize(18, 18))
        header = self.groups.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.Fixed)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.Fixed)
        header.resizeSection(1, 70)
        header.resizeSection(2, 160)
        self.groups.currentItemChanged.connect(self._group_picked)

        self.tree = SampleTree()
        self.tree.setHeaderLabels(["Name", "Statistic", "#Cells"])
        self.tree.setUniformRowHeights(True)
        self.tree.setAlternatingRowColors(True)
        self.tree.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.tree.setDragDropMode(QtWidgets.QAbstractItemView.DragOnly)
        header = self.tree.header()
        header.setStretchLastSection(False)
        header.setDefaultAlignment(QtCore.Qt.AlignCenter)
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.Interactive)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.Interactive)
        header.resizeSection(1, 220)
        header.resizeSection(2, 220)
        self.tree.itemSelectionChanged.connect(self._on_select)
        self.tree.itemDoubleClicked.connect(self._on_double_click)
        self.tree.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._context_menu)

        split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        split.addWidget(self.groups)
        split.addWidget(self.tree)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([170, 600])
        split.setChildrenCollapsible(False)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(split)
        # counts fill in as background gating finishes; one rebuild per burst
        self._rebuild_timer = QtCore.QTimer(self, singleShot=True, interval=60)
        self._rebuild_timer.timeout.connect(self.rebuild)
        gating().ready.connect(self._results_ready)
        self.rebuild_groups()

    def set_workspace(self, workspace):
        self.ws = workspace
        self.rebuild_groups()

    # ----- groups ----------------------------------------------------------

    def rebuild_groups(self):
        current = self.current_group()
        self.groups.blockSignals(True)
        self.groups.clear()
        for name, members in self.ws.groups.items():
            item = QtWidgets.QTreeWidgetItem([name, str(len(members)), _role(name)])
            item.setData(0, ROLE, name)
            item.setIcon(0, _group_icon(name))
            item.setTextAlignment(1, QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            item.setTextAlignment(2, QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            font = item.font(0)
            font.setBold(name == ALL_SAMPLES)
            item.setFont(0, font)
            self.groups.addTopLevelItem(item)
        self._select_group_item(current)
        self.groups.blockSignals(False)
        self.rebuild()

    def _select_group_item(self, name):
        for i in range(self.groups.topLevelItemCount()):
            item = self.groups.topLevelItem(i)
            if item.data(0, ROLE) == name:
                self.groups.setCurrentItem(item)
                return True
        if self.groups.topLevelItemCount():
            self.groups.setCurrentItem(self.groups.topLevelItem(0))
        return False

    def select_group(self, name):
        if name in self.ws.groups:
            self._select_group_item(name)

    def current_group(self):
        item = self.groups.currentItem() if hasattr(self, "groups") else None
        name = item.data(0, ROLE) if item is not None else None
        return name if name in getattr(self.ws, "groups", {}) else ALL_SAMPLES

    def _group_picked(self, *_):
        self.rebuild()
        self.groupChanged.emit(self.current_group())

    # ----- samples and populations -----------------------------------------

    def rebuild(self):
        selected = self.selected()
        expanded = {self._key(it) for it in self._items() if it.isExpanded()}
        self.tree.clear()
        for sid in self.ws.groups.get(self.current_group(), []):
            item = QtWidgets.QTreeWidgetItem([sid, "", f"{self.ws.samples[sid].event_count:,}"])
            item.setData(0, ROLE, (sid, None, ROOT))
            item.setTextAlignment(2, QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            font = item.font(0)
            font.setWeight(QtGui.QFont.DemiBold)
            item.setFont(0, font)
            self.tree.addTopLevelItem(item)
            self._add_children(item, sid, ROOT)
        for it in self._items():
            key = self._key(it)
            if key in expanded or key[1] is None:
                it.setExpanded(True)
            if selected and key == selected:
                it.setSelected(True)

    def _results_ready(self, ws, sid):
        if ws is self.ws and sid in self.ws.groups.get(self.current_group(), []):
            self._rebuild_timer.start()

    def _add_children(self, parent_item, sid, path):
        children = self.ws.children(path)
        if children and path == ROOT:
            gating().ensure(self.ws, [sid])
        for name, parent in children:
            if not self.ws.applies_to(sid, name, parent):
                continue
            if self.ws.has_results(sid):
                count = f"{self.ws.count(sid, name, parent):,}"
                freq = f"{self.ws.statistic(sid, 'Freq. of Parent', name, parent):.2f}"
            else:
                count = freq = "…"
            item = QtWidgets.QTreeWidgetItem([name, freq, count])
            item.setData(0, ROLE, (sid, name, parent))
            item.setTextAlignment(1, QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            item.setTextAlignment(2, QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            item.setForeground(1, QtGui.QColor(theme.ACCENT_DARK))
            if self.ws.is_boolean(name, parent):
                item.setIcon(0, icons.icon("boolean", 16))
                op, inputs = self.ws.boolean_inputs(name, parent)
                item.setToolTip(0, "Boolean gate: " + f" {op.upper()} ".join(
                    ("NOT " if neg else "") + " › ".join([*p[1:], n]) for n, p, neg in inputs))
            parent_item.addChild(item)
            self._add_children(item, sid, parent + (name,))

    def _items(self):
        it = QtWidgets.QTreeWidgetItemIterator(self.tree)
        while it.value():
            yield it.value()
            it += 1

    @staticmethod
    def _key(item):
        sid, name, parent = item.data(0, ROLE)
        return sid, name, tuple(parent)

    def selected(self):
        items = self.tree.selectedItems()
        return self._key(items[0]) if items else None

    def selected_populations(self):
        """(name, parent_path) of each selected gated population, once each, in tree order."""
        out = []
        for it in self._items():
            if it.isSelected():
                _, name, parent = self._key(it)
                if name is not None and (name, parent) not in out:
                    out.append((name, parent))
        return out

    def selected_samples(self):
        return list(dict.fromkeys(self._key(i)[0] for i in self.tree.selectedItems()))

    def _on_select(self):
        sel = self.selected()
        if sel:
            self.populationSelected.emit(*sel)

    def _on_double_click(self, item, _column):
        self.populationActivated.emit(*self._key(item))

    # ----- editing -----------------------------------------------------------

    def _context_menu(self, pos):
        item = self.tree.itemAt(pos)
        if item is None:
            return
        sid, name, parent = self._key(item)
        menu = QtWidgets.QMenu(self)
        menu.addAction("Open graph window", lambda: self.populationActivated.emit(sid, name, parent))
        menu.addSeparator()
        if name is not None:
            if self.ws.is_boolean(name, parent):
                menu.addAction(icons.icon("boolean", 16), "Edit Boolean gate…",
                               lambda: self.edit_boolean(name, parent, sid))
            menu.addAction(icons.icon("boolean", 16), "Boolean gate from selection…",
                           lambda: self.make_boolean())
            menu.addAction("Rename population…", lambda: self._rename(name, parent))
            menu.addAction("Delete gate", lambda: self._delete(name, parent))
        else:
            menu.addAction("Add to group…", lambda: self._add_to_group())
            menu.addAction("Remove sample", lambda: self._remove_sample(sid))
        menu.exec(self.tree.viewport().mapToGlobal(pos))

    def rename_selected(self):
        sel = self.selected()
        if sel and sel[1] is not None:
            self._rename(sel[1], sel[2])

    def delete_selected(self):
        sel = self.selected()
        if sel and sel[1] is not None:
            self._delete(sel[1], sel[2])

    def remove_selected_samples(self):
        samples = self.selected_samples()
        if not samples:
            return
        answer = QtWidgets.QMessageBox.question(
            self, "Remove samples", f"Remove {len(samples)} sample(s) from the workspace?")
        if answer == QtWidgets.QMessageBox.Yes:
            for sid in samples:
                self.ws.remove_sample(sid)
            self.changed.emit()

    def _rename(self, name, parent):
        new, ok = QtWidgets.QInputDialog.getText(self, "Rename population", "New name:", text=name)
        if ok and new.strip() and new.strip() != name:
            self.ws.rename_gate(name, new.strip(), parent)
            self.changed.emit()

    def _delete(self, name, parent):
        if confirm_delete(self, self.ws, name, parent):
            self.ws.remove_gate(name, parent)
            self.changed.emit()

    def make_boolean(self):
        """Open the Boolean gate editor with the selected populations filled in."""
        if not self.ws.population_paths():
            QtWidgets.QMessageBox.information(
                self, "Boolean gate", "Draw some gates first, then select the populations to combine.")
            return []
        sel = self.selected()
        dlg = BooleanDialog(self.ws, self.selected_populations(),
                            sample_id=sel[0] if sel else None, parent=self)
        if dlg.exec() == QtWidgets.QDialog.Accepted and dlg.made:
            self.changed.emit()
        return dlg.made

    def edit_boolean(self, name, parent, sample_id=None):
        dlg = BooleanDialog(self.ws, editing=(name, tuple(parent)), sample_id=sample_id, parent=self)
        if dlg.exec() == QtWidgets.QDialog.Accepted and dlg.made:
            self.changed.emit()
        return dlg.made

    def create_group(self):
        name, ok = QtWidgets.QInputDialog.getText(self, "Create group", "Group name:")
        if ok and name.strip():
            self.ws.add_group(name.strip(), self.selected_samples())
            self.rebuild_groups()
            self.select_group(name.strip())
            self.changed.emit()

    def _add_to_group(self):
        samples = self.selected_samples()
        name, ok = QtWidgets.QInputDialog.getText(self, "Add to group", "Group name:")
        if ok and name.strip():
            self.ws.add_group(name.strip(), samples)
            self.rebuild_groups()

    def _remove_sample(self, sid):
        self.ws.remove_sample(sid)
        self.changed.emit()


class SampleTree(QtWidgets.QTreeWidget):
    """The sample table, with Sidescatter's answer to FlowJo's licence watermark."""

    WATERMARK = ("No License, No Problem:", "Use Any Files You Like")

    def mimeTypes(self):
        return [POPULATION_MIME]

    def mimeData(self, items):
        pops = []
        for it in items:
            sid, name, parent = it.data(0, ROLE)
            path = list(parent) + ([name] if name is not None else [])
            pops.append({"sample_id": sid, "population": path})
        mime = QtCore.QMimeData()
        mime.setData(POPULATION_MIME, json.dumps(pops).encode())
        return mime

    def paintEvent(self, event):
        super().paintEvent(event)
        vp = self.viewport()
        if vp.height() < 160:
            return
        p = QtGui.QPainter(vp)
        p.setRenderHint(QtGui.QPainter.TextAntialiasing)
        font = QtGui.QFont("Georgia")
        font.setStyleHint(QtGui.QFont.Serif)
        font.setPixelSize(max(22, min(46, vp.width() // 22)))
        p.setFont(font)
        p.setPen(theme.qcolor(theme.FAINT, 150))
        line = QtGui.QFontMetrics(font).lineSpacing()
        x = int(vp.width() * 0.09)
        y = vp.height() - int(line * 1.4) - 24
        for i, text in enumerate(self.WATERMARK):
            p.drawText(x, y + i * int(line * 1.25), text)
        p.end()


def _role(group):
    if group == "Unmixed":
        return "Spectral"
    if "control" in group.lower() or "compensation" in group.lower():
        return "Compensation"
    return "Test"


def _group_icon(group):
    pm = QtGui.QPixmap(36, 36)
    pm.fill(QtCore.Qt.transparent)
    p = QtGui.QPainter(pm)
    p.setRenderHint(QtGui.QPainter.Antialiasing)
    color = {"Test": theme.ACCENT, "Spectral": theme.TEAL}.get(_role(group), theme.CORAL)
    p.setPen(QtGui.QPen(QtGui.QColor(color), 2.4))
    p.setBrush(theme.qcolor(color, 40))
    p.drawRoundedRect(QtCore.QRectF(5, 9, 26, 18), 4, 4)
    p.setPen(QtCore.Qt.NoPen)
    p.setBrush(QtGui.QColor(color))
    for x in (12, 18, 24):
        p.drawEllipse(QtCore.QPointF(x, 18), 2.2, 2.2)
    p.end()
    pm.setDevicePixelRatio(2)
    return QtGui.QIcon(pm)
