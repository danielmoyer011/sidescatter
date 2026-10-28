"""Layouts: pages of plots and text arranged into a figure, like FlowJo 10's Layout Editor.

A layout is one continuous canvas cut into pages. Positions and sizes are in
points (1/72 inch) from the top-left of the first page, so page k covers
y in [k * page_height, (k + 1) * page_height). Plots reference a sample and a
population by name, not a copy of the data, so they follow gate edits.

The fields mirror what a FlowJo .wsp <Layout> stores for each graph (sample,
gate path, axis parameters, graph type, position and size) so FlowJo layouts
can be imported into this model later.
"""

import copy
import math
import uuid
from dataclasses import asdict, dataclass, field

PAGE_SIZES = {"Letter": (612.0, 792.0), "A4": (595.28, 841.89)}  # portrait, points
MARGIN = 36.0      # half an inch
GRID = 9.0         # snap step: 1/8 inch
BATCH_GAP = 18.0
PLOT_TYPES = ("Pseudocolor", "Dot plot", "Contour", "Histogram")

# Fields a text box (or a plot title) can contain; filled from its sample and population.
TEXT_FIELDS = ("{sample}", "{population}", "{count}", "{freq}")


def new_id():
    return uuid.uuid4().hex[:12]


@dataclass
class LayoutItem:
    kind: str                       # "plot" or "text"
    x: float = MARGIN
    y: float = MARGIN
    w: float = 216.0
    h: float = 216.0
    sample_id: str | None = None
    population: list = field(default_factory=lambda: ["root"])  # full gate path from root
    # plots
    x_channel: str | None = None
    y_channel: str | None = None
    plot_type: str = "Pseudocolor"
    show_gates: bool = True
    show_gate_labels: bool = True
    show_axis_labels: bool = True
    title: str = "{sample}: {population}"
    # text boxes
    text: str = ""
    bold: bool = False
    italic: bool = False
    font_size: float = 9.0
    id: str = field(default_factory=new_id)

    @property
    def population_path(self):
        return tuple(self.population)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Layout:
    name: str = "Layout"
    page_size: str = "Letter"
    landscape: bool = False
    items: list = field(default_factory=list)

    @property
    def page_width(self):
        w, h = PAGE_SIZES[self.page_size]
        return h if self.landscape else w

    @property
    def page_height(self):
        w, h = PAGE_SIZES[self.page_size]
        return w if self.landscape else h

    def page_count(self):
        """Pages with something on them (at least one)."""
        bottom = max((it.y + it.h for it in self.items), default=0.0)
        return max(1, math.ceil(bottom / self.page_height - 1e-6))

    def item(self, item_id):
        return next(it for it in self.items if it.id == item_id)

    def bounds(self, items=None):
        """(x, y, w, h) around the given items (default: all)."""
        items = self.items if items is None else items
        if not items:
            return (0.0, 0.0, 0.0, 0.0)
        x0 = min(it.x for it in items)
        y0 = min(it.y for it in items)
        x1 = max(it.x + it.w for it in items)
        y1 = max(it.y + it.h for it in items)
        return (x0, y0, x1 - x0, y1 - y0)

    def to_dict(self):
        return {"name": self.name, "page_size": self.page_size, "landscape": self.landscape,
                "items": [it.to_dict() for it in self.items]}

    @classmethod
    def from_dict(cls, d):
        return cls(d.get("name", "Layout"), d.get("page_size", "Letter"),
                   bool(d.get("landscape", False)),
                   [LayoutItem.from_dict(i) for i in d.get("items", [])])


def unique_name(name, layouts):
    taken = {lay.name for lay in layouts}
    if name not in taken:
        return name
    n = 2
    while f"{name} ({n})" in taken:
        n += 1
    return f"{name} ({n})"


def snap(value, step=GRID):
    return round(value / step) * step


def batch(layout, sample_ids, direction="rows", per_page=0, name=None):
    """Repeat a layout once per sample, as a new layout.

    Every plot and text box in the template is switched to each sample in turn.
    "rows" stacks one sample per row down the page; "columns" places one sample
    per column across the page and wraps onto a new band when the row is full.
    per_page limits how many samples go on one page (0 = as many as fit).
    """
    if direction not in ("rows", "columns"):
        raise ValueError("direction must be 'rows' or 'columns'")
    out = Layout(name or f"{layout.name} batch", layout.page_size, layout.landscape)
    if not layout.items or not sample_ids:
        return out
    bx, by, bw, bh = layout.bounds()
    pw, ph = layout.page_width, layout.page_height
    usable_w, usable_h = pw - 2 * MARGIN, ph - 2 * MARGIN
    across = max(1, int((usable_w + BATCH_GAP) // (bw + BATCH_GAP))) if direction == "columns" else 1
    down = max(1, int((usable_h + BATCH_GAP) // (bh + BATCH_GAP)))
    fit = across * down
    per_page = fit if per_page <= 0 else min(per_page, fit)

    for i, sid in enumerate(sample_ids):
        page, slot = divmod(i, per_page)
        row, col = divmod(slot, across)
        ox = MARGIN + col * (bw + BATCH_GAP) if direction == "columns" else bx
        oy = page * ph + MARGIN + row * (bh + BATCH_GAP)
        for it in layout.items:
            new = copy.deepcopy(it)
            new.id = new_id()
            new.x = ox + (it.x - bx)
            new.y = oy + (it.y - by)
            if new.sample_id is not None:
                new.sample_id = sid
            out.items.append(new)
    return out


def rename_population(layouts, parent_path, old, new):
    """Keep layout items pointing at a population after it (or an ancestor) is renamed."""
    prefix = list(parent_path) + [old]
    n = len(prefix)
    for lay in layouts:
        for it in lay.items:
            if it.population[:n] == prefix:
                it.population = list(parent_path) + [new] + it.population[n:]
