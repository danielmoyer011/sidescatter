"""The Layout Editor: layout model, batching, export, and the editor window."""

import json
import re

import flowkit as fk
import pytest

from sidescatter.core.layout import Layout, LayoutItem, batch, rename_population
from sidescatter.app.jobs import wait_idle  # noqa: E402
from sidescatter.core.workspace import ROOT, Workspace

SID = "PBMC_full_panel.fcs"


def _plot(**kw):
    return LayoutItem("plot", sample_id=SID, **kw)


def test_layout_roundtrip_and_pages():
    lay = Layout("Fig 1", "A4", True, [_plot(x=36, y=36), LayoutItem("text", text="Hi", y=700)])
    again = Layout.from_dict(json.loads(json.dumps(lay.to_dict())))
    assert again == lay
    assert (again.page_width, again.page_height) == pytest.approx((841.89, 595.28))
    assert again.page_count() == 2  # the text box sits on the second landscape page
    assert Layout().page_count() == 1


def test_batch_rows_columns_and_per_page():
    template = Layout("T", items=[_plot(x=36, y=36, w=216, h=216), _plot(x=270, y=36, w=216, h=216),
                                  LayoutItem("text", text="note", x=36, y=260, w=100, h=20)])
    samples = [f"s{i}" for i in range(5)]

    rows = batch(template, samples, "rows")
    assert len(rows.items) == 15
    plots = [it for it in rows.items if it.kind == "plot"]
    assert [it.sample_id for it in plots[::2]] == samples
    assert all(it.sample_id is None for it in rows.items if it.kind == "text")
    tops = sorted({round(it.y) for it in plots})
    assert len(tops) == 5 and rows.page_count() == 3  # 2 rows of 244pt fit on a Letter page
    assert all(it.y + it.h <= (it.y // 792 + 1) * 792 - 36 + 1e-6 for it in rows.items)

    one_per_page = batch(template, samples, "rows", per_page=1)
    assert one_per_page.page_count() == 5

    narrow = Layout("N", items=[_plot(x=36, y=36, w=120, h=120)])
    cols = batch(narrow, samples, "columns")
    assert sorted({round(it.y) for it in cols.items}) == [36, 174]  # 4 across, then wraps
    assert [round(it.x) for it in cols.items[:4]] == [36, 174, 312, 450]
    assert len({it.id for it in cols.items}) == 5


def test_rename_population_follows_into_layouts(spectral_dir):
    ws = Workspace()
    ws.add_fcs([spectral_dir / SID])
    ws.add_gate(fk.gates.RectangleGate("Big", [ws.make_dimension(SID, "FSC-A", 0.3, None)]))
    ws.add_gate(fk.gates.RectangleGate("Bigger", [ws.make_dimension(SID, "FSC-A", 0.5, None)]),
                ("root", "Big"))
    ws.layouts.append(Layout(items=[_plot(population=["root", "Big", "Bigger"]),
                                    _plot(population=["root", "Bigish"])]))
    before = ws.revision
    ws.rename_gate("Big", "Large", ROOT)
    assert ws.revision > before
    assert ws.layouts[0].items[0].population == ["root", "Large", "Bigger"]
    assert ws.layouts[0].items[1].population == ["root", "Bigish"]  # only whole names match

    lays = [Layout(items=[_plot(population=["root", "A", "B"])])]
    rename_population(lays, ("root", "A"), "B", "C")
    assert lays[0].items[0].population == ["root", "A", "C"]


def test_layouts_are_saved_with_the_workspace(spectral_dir, tmp_path):
    ws = Workspace()
    ws.add_fcs([spectral_dir / SID])
    ws.layouts.append(Layout("Figure", items=[_plot(x_channel="FSC-A", y_channel="SSC-A")]))
    ws.save(tmp_path / "w.ssw")
    again = Workspace.load(tmp_path / "w.ssw")
    assert [lay.to_dict() for lay in again.layouts] == [lay.to_dict() for lay in ws.layouts]


# ----- rendering and export (needs Qt) -------------------------------------------

pytest.importorskip("pytestqt")

from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402

from sidescatter.app import layout_render  # noqa: E402
from sidescatter.app.main_window import DEMO_DIR, MainWindow  # noqa: E402
from sidescatter.app.workspace_panel import POPULATION_MIME  # noqa: E402


@pytest.fixture
def window(qtbot):
    w = MainWindow()
    w.ask_before_discard = False
    qtbot.addWidget(w)
    w.add_samples([str(DEMO_DIR / SID)])
    wait_idle()
    ws = w.ws
    ws.add_gate(fk.gates.RectangleGate("Big", [ws.make_dimension(SID, "FSC-A", 0.3, 0.9),
                                                ws.make_dimension(SID, "SSC-A", 0.05, 0.8)]))
    w._changed()
    return w


@pytest.fixture
def editor(window, qtbot):
    win = window.open_layout_editor()
    qtbot.addWidget(win)
    return win


def test_export_pdf_svg_png(window, tmp_path, qapp):
    ws = window.ws
    lay = Layout("Fig", items=[_plot(x=36, y=36, x_channel="FSC-A", y_channel="SSC-A"),
                               _plot(x=36, y=300, population=["root", "Big"], plot_type="Histogram"),
                               _plot(x=300, y=36, plot_type="Contour"),
                               _plot(x=300, y=300, plot_type="Dot plot"),
                               LayoutItem("text", text="{count} in {population}", x=36, y=900,
                                          sample_id=SID, population=["root", "Big"]),
                               _plot(x=300, y=560, population=["root", "Gone"])])
    assert lay.page_count() == 2

    pdf, = layout_render.export(lay, ws, tmp_path / "fig.pdf")
    data = pdf.read_bytes()
    assert data.startswith(b"%PDF") and len(re.findall(rb"/Type\s*/Page[^s]", data)) == 2

    svgs = layout_render.export(lay, ws, tmp_path / "fig.svg")
    assert [p.name for p in svgs] == ["fig-page1.svg", "fig-page2.svg"]
    assert "<svg" in svgs[0].read_text()

    pngs = layout_render.export(lay, ws, tmp_path / "fig.png", dpi=144)
    img = QtGui.QImage(str(pngs[0]))
    assert (img.width(), img.height()) == (1224, 1584)  # 8.5 x 11 in at 144 dpi
    assert img.pixelColor(10, 10) == QtGui.QColor("#FFFFFF")

    from PySide6 import QtPrintSupport

    printer = QtPrintSupport.QPrinter()
    printer.setOutputFormat(QtPrintSupport.QPrinter.PdfFormat)
    printer.setOutputFileName(str(tmp_path / "printed.pdf"))
    layout_render.print_layout(lay, ws, printer)
    assert (tmp_path / "printed.pdf").read_bytes().startswith(b"%PDF")

    big = ws.count(SID, "Big")
    assert layout_render.fill_fields("{count} in {population}", ws, lay.items[4]) == f"{big:,} in Big"
    assert "doesn't apply" in layout_render.PlotCache().get(ws, lay.items[5])


def _drop_populations(view, pops, at):
    mime = QtCore.QMimeData()
    mime.setData(POPULATION_MIME, json.dumps(pops).encode())
    pos = QtCore.QPointF(view.mapFromScene(QtCore.QPointF(*at)))
    target = view.viewport()
    enter = QtGui.QDragEnterEvent(pos.toPoint(), QtCore.Qt.CopyAction, mime, QtCore.Qt.LeftButton,
                                  QtCore.Qt.NoModifier)
    QtWidgets.QApplication.sendEvent(target, enter)
    assert enter.isAccepted()
    drop = QtGui.QDropEvent(pos, QtCore.Qt.CopyAction, mime, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier)
    QtWidgets.QApplication.sendEvent(target, drop)


def test_ribbon_opens_one_editor_with_a_blank_layout(window, editor):
    assert window.acts["layout"].iconText() == "Layout\nEditor"
    window.acts["layout"].trigger()
    assert window.layout_window is editor
    assert len(window.ws.layouts) == 1 and editor.layout.items == []
    tabs = [editor.ribbon.tabs.tabText(i) for i in range(editor.ribbon.tabs.count())]
    assert tabs == ["Layout", "Arrange", "View"]


def test_drag_populations_from_the_workspace(window, editor):
    tree = window.panel.tree
    sample_item = tree.topLevelItem(0)
    mime = tree.mimeData([sample_item, sample_item.child(0)])
    pops = json.loads(bytes(mime.data(POPULATION_MIME)).decode())
    assert pops == [{"sample_id": SID, "population": ["root"]},
                    {"sample_id": SID, "population": ["root", "Big"]}]

    editor.show()
    _drop_populations(editor.view, pops, (200, 200))
    items = editor.layout.items
    assert [it.population for it in items] == [["root"], ["root", "Big"]]
    assert items[0].x_channel == "FSC-A" and items[0].y_channel == "SSC-A"
    assert items[1].x > items[0].x and items[0].y == items[1].y  # laid out in a row
    assert all(v % 9 == 0 for it in items for v in (it.x, it.y))  # snapped to the grid
    assert window.dirty and len(editor.selected_items()) == 2


def test_new_plot_copies_the_open_graph_window(window, editor):
    v = window.open_graph(SID, "Big", ROOT)
    v.set_axes("SSC-A", "FSC-A")
    v.type_combo.setCurrentText("Contour")
    item, = editor.add_plots([{"sample_id": SID, "population": ["root", "Big"]}])
    assert (item.x_channel, item.y_channel, item.plot_type) == ("SSC-A", "FSC-A", "Contour")


def test_arrange_undo_and_redo(editor):
    a, b, c = editor.add_plots([{"sample_id": SID, "population": ["root"]}] * 3, 150, 150)
    b.y, c.y, c.w, c.h = 180, 360, 144, 108
    editor.select([a, b, c])
    editor.align("left")
    assert a.x == b.x == c.x
    editor.distribute("v")
    gaps = [b.y - (a.y + a.h), c.y - (b.y + b.h)]
    assert gaps[0] == pytest.approx(gaps[1])
    editor._order = [c.id, a.id, b.id]
    editor.same_size()
    assert (a.w, a.h) == (b.w, b.h) == (144, 108)

    editor.undo.undo()  # same size
    a, b, c = editor.layout.items
    assert (a.w, a.h) == (216, 216)
    editor.undo.redo()
    assert all((it.w, it.h) == (144, 108) for it in editor.layout.items)

    editor.select(editor.layout.items[:1])
    editor.delete_items()
    assert len(editor.layout.items) == 2
    editor.undo.undo()
    assert len(editor.layout.items) == 3


def test_inspector_edits_merge_into_one_undo_step(editor):
    item, = editor.add_plots([{"sample_id": SID, "population": ["root"]}])
    editor.select([item])
    insp = editor.inspector
    assert insp.stack.currentIndex() == 2
    steps = editor.undo.count()
    for text in ("C", "CD", "CD3"):
        insp.title.setText(text)
        insp.title.textEdited.emit(text)
    insp.plot_type.setCurrentText("Histogram")
    assert editor.layout.items[0].title == "CD3"
    assert editor.layout.items[0].plot_type == "Histogram"
    assert editor.undo.count() == steps + 2
    insp.population.setCurrentIndex(insp.population.findData(["root", "Big"]))
    assert editor.layout.items[0].population == ["root", "Big"]


def test_text_tool_and_fields(editor, qtbot):
    editor.show()
    qtbot.waitExposed(editor)
    editor.acts["text"].trigger()
    assert editor.tool == "text"
    vp = editor.view.viewport()
    pos = editor.view.mapFromScene(QtCore.QPointF(100, 400))
    qtbot.mouseClick(vp, QtCore.Qt.LeftButton, pos=pos)
    item, = editor.layout.items
    assert item.kind == "text" and editor.tool == "select"
    assert (item.x, item.y) == pytest.approx((99, 396), abs=9)
    editor.inspector.text.setPlainText("{sample}")
    assert item.text == "{sample}"
    assert layout_render.fill_fields(item.text, editor.ws, item) == ""  # no sample picked yet
    editor.inspector.sample.setCurrentIndex(editor.inspector.sample.findData(SID))
    assert layout_render.fill_fields(item.text, editor.ws, item) == SID


def test_batch_and_page_setup(window, editor, monkeypatch):
    from sidescatter.app import layout_editor

    window.add_samples([str(DEMO_DIR / "controls" / "Beads_unstained.fcs")])
    wait_idle()
    window.ws.add_group("Pair", [SID, "Beads_unstained.fcs"])
    editor.add_plots([{"sample_id": SID, "population": ["root", "Big"]}])
    monkeypatch.setattr(layout_editor.BatchDialog, "exec", lambda self: QtWidgets.QDialog.Accepted)
    monkeypatch.setattr(layout_editor.BatchDialog, "values", lambda self: ("Pair", "columns", 0))
    editor.batch_dialog()
    assert len(window.ws.layouts) == 2 and editor.index == 1
    assert editor.layout.name == "Layout – Pair"
    assert [it.sample_id for it in editor.layout.items] == [SID, "Beads_unstained.fcs"]
    assert editor.list.count() == 2

    monkeypatch.setattr(layout_editor.PageSetupDialog, "exec", lambda self: QtWidgets.QDialog.Accepted)
    monkeypatch.setattr(layout_editor.PageSetupDialog, "values", lambda self: ("A4", True))
    editor.page_setup()
    assert editor.layout.page_size == "A4" and editor.layout.landscape


def _plot_data(editor, ws, item):
    """The editor's plot data, once background gating has caught up."""
    data = editor.cache.get(ws, item)
    if data == "Calculating…":
        wait_idle()
        data = editor.cache.get(ws, item)
    return data


def test_plots_follow_gate_edits_and_workspace_changes(window, editor):
    item, = editor.add_plots([{"sample_id": SID, "population": ["root", "Big"]}])
    first = _plot_data(editor, window.ws, item)["count"]
    gate = window.ws.get_gate("Big", ROOT)
    gate.dimensions[0].min = 0.6
    window.ws.invalidate()
    window._changed()
    QtWidgets.QApplication.processEvents()
    assert _plot_data(editor, window.ws, item)["count"] < first

    window.ws.rename_gate("Big", "Large", ROOT)
    assert item.population == ["root", "Large"]

    window.new_workspace()
    assert editor.ws is window.ws and len(window.ws.layouts) == 1 and editor.layout.items == []
