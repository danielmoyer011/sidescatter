"""Table editor: choose populations x statistics, compute across samples, export."""

from PySide6 import QtCore, QtWidgets

from ..core import stats
from ..core.workspace import ROOT
from .gating import service as gating
from .jobs import run_job


class TableEditor(QtWidgets.QDialog):
    def __init__(self, workspace, sample_ids, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Table editor")
        self.resize(1000, 600)
        self.ws = workspace
        self.sample_ids = sample_ids
        self.columns = []
        self.table_df = None

        self.pop_combo = QtWidgets.QComboBox()
        self.pop_combo.addItem("All events", (None, ROOT))
        for name, parent in workspace.population_paths():
            self.pop_combo.addItem("/".join([*parent[1:], name]), (name, parent))
        self.stat_combo = QtWidgets.QComboBox()
        self.stat_combo.addItems(stats.ALL_STATS)
        self.stat_combo.currentTextChanged.connect(self._stat_changed)
        self.chan_combo = QtWidgets.QComboBox()
        chans = {}
        for sid in sample_ids:
            for c in workspace.channels(sid):
                chans.setdefault(c.pnn, c.label)
        for pnn, label in chans.items():
            self.chan_combo.addItem(label, pnn)
        self.pct = QtWidgets.QDoubleSpinBox()
        self.pct.setRange(0, 100)
        self.pct.setValue(50)
        self.pct.setSuffix(" th")
        add = QtWidgets.QPushButton("Add column")
        add.clicked.connect(self._add_column)
        clear = QtWidgets.QPushButton("Clear")
        clear.clicked.connect(self._clear)

        row = QtWidgets.QHBoxLayout()
        for w in (QtWidgets.QLabel("Population"), self.pop_combo, QtWidgets.QLabel("Statistic"),
                  self.stat_combo, self.chan_combo, self.pct, add, clear):
            row.addWidget(w)
        row.addStretch()

        self.view = QtWidgets.QTableWidget()
        export = QtWidgets.QPushButton("Export…")
        export.clicked.connect(self._export)
        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(self.accept)
        self.status = QtWidgets.QLabel()
        self.status.setObjectName("GraphHint")
        bottom = QtWidgets.QHBoxLayout()
        bottom.addWidget(self.status)
        bottom.addStretch()
        bottom.addWidget(export)
        bottom.addWidget(close)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(row)
        layout.addWidget(self.view, 1)
        layout.addLayout(bottom)
        self._stat_changed(self.stat_combo.currentText())

    def _stat_changed(self, stat):
        self.chan_combo.setEnabled(stat in stats.CHANNEL_STATS)
        self.pct.setEnabled(stat == "Percentile")

    def _add_column(self):
        name, parent = self.pop_combo.currentData()
        stat = self.stat_combo.currentText()
        col = {"name": name, "parent_path": parent, "stat": stat}
        if stat in stats.CHANNEL_STATS:
            col["channel"] = self.chan_combo.currentData()
        if stat == "Percentile":
            col["percentile"] = self.pct.value()
        self.columns.append(col)
        self._recompute()

    def _clear(self):
        self.columns = []
        self._recompute()

    def _recompute(self):
        """Work out the table in the background; the dialog stays usable meanwhile."""
        ws, sids, columns = self.ws, list(self.sample_ids), [dict(c) for c in self.columns]
        self.status.setText("Calculating…")

        def compute():
            self.job = run_job("Statistics table", lambda job: ws.statistics_table(sids, columns),
                               on_done=self._show, key=("table", id(self)))

        self.job = gating().prepare(ws, sids, "Calculating gates for the table", compute,
                                    key=("table-gating", id(self)))
        return self.job

    def _show(self, df):
        self.status.setText("")
        self.table_df = df
        self.view.setRowCount(len(df))
        self.view.setColumnCount(len(df.columns))
        self.view.setHorizontalHeaderLabels(list(df.columns))
        for r in range(len(df)):
            for c, col in enumerate(df.columns):
                v = df.iat[r, c]
                text = v if isinstance(v, str) else _fmt(v)
                item = QtWidgets.QTableWidgetItem(text)
                if not isinstance(v, str):
                    item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                self.view.setItem(r, c, item)
        self.view.resizeColumnsToContents()

    def _export(self):
        if self.table_df is None or not self.columns:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export table", "statistics.xlsx", "Excel (*.xlsx);;CSV (*.csv)")
        if not path:
            return
        if path.lower().endswith(".csv"):
            self.table_df.to_csv(path, index=False)
        else:
            self.table_df.to_excel(path, index=False)


def _fmt(v):
    if isinstance(v, int):
        return f"{v:,}"
    if v != v:  # NaN
        return ""
    return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.4g}"
