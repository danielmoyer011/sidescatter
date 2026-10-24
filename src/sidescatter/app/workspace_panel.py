"""The workspace panel: groups, samples, and each sample's gating tree with counts."""

from PySide6 import QtCore, QtWidgets

from ..core.workspace import ALL_SAMPLES, ROOT

ROLE = QtCore.Qt.UserRole


class WorkspacePanel(QtWidgets.QWidget):
    populationSelected = QtCore.Signal(str, object, object)  # sample_id, name, parent_path
    changed = QtCore.Signal()

    def __init__(self, workspace, parent=None):
        super().__init__(parent)
        self.ws = workspace

        self.group_combo = QtWidgets.QComboBox()
        self.group_combo.currentIndexChanged.connect(self.rebuild)

        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(["Name", "Count", "%"])
        self.tree.setColumnWidth(0, 250)
        self.tree.setColumnWidth(1, 80)
        self.tree.setColumnWidth(2, 60)
        self.tree.setAlternatingRowColors(True)
        self.tree.itemSelectionChanged.connect(self._on_select)
        self.tree.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._context_menu)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Group"))
        top.addWidget(self.group_combo, 1)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addLayout(top)
        layout.addWidget(self.tree)

    def set_workspace(self, workspace):
        self.ws = workspace
        self.rebuild_groups()

    def rebuild_groups(self):
        current = self.group_combo.currentData() or ALL_SAMPLES
        self.group_combo.blockSignals(True)
        self.group_combo.clear()
        for name, members in self.ws.groups.items():
            self.group_combo.addItem(f"{name} ({len(members)})", name)
        i = self.group_combo.findData(current)
        self.group_combo.setCurrentIndex(max(i, 0))
        self.group_combo.blockSignals(False)
        self.rebuild()

    def current_group(self):
        return self.group_combo.currentData() or ALL_SAMPLES

    def rebuild(self):
        selected = self.selected()
        expanded = {self._key(it) for it in self._items() if it.isExpanded()}
        self.tree.clear()
        for sid in self.ws.groups.get(self.current_group(), []):
            item = QtWidgets.QTreeWidgetItem([sid, f"{self.ws.samples[sid].event_count:,}", ""])
            item.setData(0, ROLE, (sid, None, ROOT))
            self.tree.addTopLevelItem(item)
            self._add_children(item, sid, ROOT)
        for it in self._items():
            key = self._key(it)
            if key in expanded or key[1] is None:
                it.setExpanded(True)
            if selected and key == selected:
                it.setSelected(True)

    def _add_children(self, parent_item, sid, path):
        for name, parent in self.ws.children(path):
            if not self.ws.applies_to(sid, name, parent):
                continue
            count = self.ws.count(sid, name, parent)
            freq = self.ws.statistic(sid, "Freq. of Parent", name, parent)
            item = QtWidgets.QTreeWidgetItem([name, f"{count:,}", f"{freq:.2f}"])
            item.setData(0, ROLE, (sid, name, parent))
            item.setTextAlignment(1, QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            item.setTextAlignment(2, QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
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

    def selected_samples(self):
        return list(dict.fromkeys(self._key(i)[0] for i in self.tree.selectedItems()))

    def _on_select(self):
        sel = self.selected()
        if sel:
            self.populationSelected.emit(*sel)

    def _context_menu(self, pos):
        item = self.tree.itemAt(pos)
        if item is None:
            return
        sid, name, parent = self._key(item)
        menu = QtWidgets.QMenu(self)
        if name is not None:
            menu.addAction("Rename population…", lambda: self._rename(name, parent))
            menu.addAction("Delete gate", lambda: self._delete(name, parent))
        else:
            menu.addAction("Add to group…", lambda: self._add_to_group())
            menu.addAction("Remove sample", lambda: self._remove_sample(sid))
        menu.exec(self.tree.viewport().mapToGlobal(pos))

    def _rename(self, name, parent):
        new, ok = QtWidgets.QInputDialog.getText(self, "Rename population", "New name:", text=name)
        if ok and new.strip() and new.strip() != name:
            self.ws.rename_gate(name, new.strip(), parent)
            self.changed.emit()

    def _delete(self, name, parent):
        answer = QtWidgets.QMessageBox.question(
            self, "Delete gate", f"Delete {name} and every population inside it, in all samples?")
        if answer == QtWidgets.QMessageBox.Yes:
            self.ws.remove_gate(name, parent)
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
