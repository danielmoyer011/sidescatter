"""High-dimensional analysis: PCA, UMAP, tSNE and FlowSOM on one population.

The chosen samples' events in that population are concatenated and
downsampled to an equal number per sample; the methods run on each channel's
display scale (logicle, arcsinh...), so they see what the plots show. Events
left out by downsampling are then placed onto the finished map, so every
event in the population gets coordinates and a cluster: each goes to the
median map position of its nearest sampled neighbours, which is how UMAP's
and openTSNE's own transform() start. Their refinement step after that
costs about 0.2 ms per event, half an hour for ten million, for little
visible change, so it is skipped.

Results become ordinary channels of each sample (UMAP-1, UMAP-2, FlowSOM,
FlowSOM meta...), so they can be plotted, gated and exported, and each
FlowSOM metacluster becomes a gate under the population.

gather() reads the workspace and must run on the GUI thread; compute() is
the slow part and can run in a worker thread; Workspace.add_analysis()
applies the result.
"""

import warnings
from dataclasses import asdict, dataclass, field

import numpy as np

SAMPLE_ID = "Sample ID"
METHODS = ("pca", "umap", "tsne", "flowsom")
METHOD_LABELS = {"pca": "PCA", "umap": "UMAP", "tsne": "tSNE", "flowsom": "FlowSOM"}
CHUNK = 50_000
NEIGHBOURS = 10  # sampled neighbours used to place each left-out event


class Cancelled(Exception):
    """Raised by a progress callback when the user cancels."""


@dataclass
class Settings:
    name: str = "Analysis 1"
    population: str = None              # population name; None means all events
    parent_path: tuple = ("root",)
    samples: list = field(default_factory=list)
    channels: list = field(default_factory=list)
    methods: list = field(default_factory=lambda: ["pca", "umap", "flowsom"])
    per_sample: int = 20_000
    max_total: int = 1_000_000
    seed: int = 42
    umap_neighbors: int = 15
    umap_min_dist: float = 0.1
    tsne_perplexity: float = 30.0
    som_x: int = 10
    som_y: int = 10
    n_meta: int = 10
    suffix: str = ""                    # added to channel names when several analyses exist

    def to_dict(self):
        d = asdict(self)
        d["parent_path"] = list(self.parent_path)
        return d

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        d["parent_path"] = tuple(d.get("parent_path", ("root",)))
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in known})

    @property
    def population_path(self):
        return self.parent_path if self.population is None else self.parent_path + (self.population,)

    def per_sample_target(self):
        """Events taken from each sample: equal numbers, within the total cap."""
        return max(1, min(self.per_sample, self.max_total // max(len(self.samples), 1)))

    # ----- the channels this analysis adds ---------------------------------

    def output_channels(self):
        """{method: [channel names]} in the order they are added to samples."""
        s = self.suffix
        out = {}
        for m in METHODS:
            if m not in self.methods:
                continue
            if m == "flowsom":
                out[m] = [f"FlowSOM{s}", f"FlowSOM meta{s}"]
            else:
                out[m] = [f"{METHOD_LABELS[m].upper()}-{i}{s}" for i in (1, 2)]
        return out

    def all_output_channels(self):
        return [c for chans in self.output_channels().values() for c in chans]

    @property
    def meta_channel(self):
        return f"FlowSOM meta{self.suffix}" if "flowsom" in self.methods else None


@dataclass
class Inputs:
    """What compute() needs, read from the workspace on the GUI thread."""

    settings: Settings
    events: dict        # sample_id -> (population events × channels) on the display scale, float32
    indices: dict       # sample_id -> event indices of the population in the full sample
    sizes: dict         # sample_id -> total events in the sample


@dataclass
class Result:
    settings: Settings
    values: dict        # sample_id -> {channel: full-length float32 array}
    sampled: dict       # sample_id -> event indices (in the full sample) used to build the map
    meta_medians: np.ndarray = None  # metacluster × channel medians, display scale
    meta_counts: np.ndarray = None   # events per metacluster over all samples


def default_channels(channels):
    """Channels ticked by default: fluorescence, except autofluorescence."""
    import re

    af = re.compile(r"^AF\d{0,2}(-[AH])?$", re.IGNORECASE)
    return [c.pnn for c in channels
            if c.kind == "fluorescence" and not af.match(c.pnn)
            and "autofluor" not in (c.pns or "").lower()]


def gather(ws, settings):
    events, indices, sizes = {}, {}, {}
    for sid in settings.samples:
        mask = ws.membership(sid, settings.population, settings.parent_path)
        idx = np.flatnonzero(mask)
        cols = [ws.display_values(sid, ch, mask).astype(np.float32) for ch in settings.channels]
        events[sid] = np.column_stack(cols) if cols else np.empty((len(idx), 0), np.float32)
        indices[sid] = idx
        sizes[sid] = ws.samples[sid].event_count
    return Inputs(settings, events, indices, sizes)


def _quiet(*_):
    pass


def compute(inputs, progress=_quiet):
    """Run the chosen methods. progress(fraction, text) may raise Cancelled."""
    st = inputs.settings
    if not st.channels:
        raise ValueError("Tick at least two channels.")
    if not st.methods:
        raise ValueError("Pick at least one method.")
    rng = np.random.default_rng(st.seed)
    per = st.per_sample_target()

    # ----- downsample and concatenate --------------------------------------
    picks, blocks = {}, []
    for sid in st.samples:
        n = len(inputs.indices[sid])
        pick = np.sort(rng.choice(n, per, replace=False)) if n > per else np.arange(n)
        picks[sid] = pick
        blocks.append(inputs.events[sid][pick])
    X = np.vstack(blocks).astype(np.float64)
    if len(X) < 50:
        raise ValueError(f"Only {len(X)} events in this population; at least 50 are needed.")
    X = np.nan_to_num(X)
    offsets = np.cumsum([0] + [len(picks[s]) for s in st.samples])

    out_names = st.output_channels()
    # Per sample, values for every population event (sampled ones first filled from the fit).
    pop_values = {sid: {} for sid in st.samples}

    weights = {"pca": 1, "umap": 6, "tsne": 6, "flowsom": 2}
    total = sum(weights[m] for m in st.methods)
    done = 0

    def stage(method, frac, text):
        progress((done + weights[method] * frac) / total, text)

    index = []

    def neighbours(data):
        """Indices of each event's nearest sampled events (an index built once, on demand)."""
        if not index:
            from pynndescent import NNDescent

            index.append(NNDescent(X, n_neighbors=max(NEIGHBOURS, 15), random_state=st.seed,
                                   low_memory=True))
            index[0].prepare()
        return index[0].query(data, k=NEIGHBOURS)[0]

    def place(method, coords_fit, mapper=None):
        """Fill pop_values with coords for sampled events, then map the rest.

        mapper maps events straight to coordinates (PCA); without one, each
        event goes to the median position of its nearest sampled neighbours.
        """
        if mapper is None:
            mapper = lambda d: np.median(coords_fit[neighbours(d)], axis=1)  # noqa: E731
        names = out_names[method]
        n_rest = sum(len(inputs.indices[s]) - len(picks[s]) for s in st.samples)
        placed = 0
        for k, sid in enumerate(st.samples):
            ev = inputs.events[sid]
            arr = np.full((len(ev), coords_fit.shape[1]), np.nan, dtype=np.float32)
            arr[picks[sid]] = coords_fit[offsets[k]:offsets[k + 1]]
            rest = np.setdiff1d(np.arange(len(ev)), picks[sid], assume_unique=True)
            for start in range(0, len(rest), CHUNK):
                chunk = rest[start:start + CHUNK]
                arr[chunk] = mapper(np.nan_to_num(ev[chunk].astype(np.float64)))
                placed += len(chunk)
                stage(method, 0.6 + 0.4 * placed / max(n_rest, 1),
                      f"{METHOD_LABELS[method]}: placing the other events "
                      f"({placed:,} of {n_rest:,})")
            for i, name in enumerate(names):
                pop_values[sid][name] = arr[:, i]

    if "pca" in st.methods:
        from sklearn.decomposition import PCA

        stage("pca", 0, "PCA")
        pca = PCA(n_components=2, random_state=st.seed).fit(X)
        place("pca", pca.transform(X), pca.transform)
        done += weights["pca"]

    if "umap" in st.methods:
        stage("umap", 0.02, f"UMAP: building the map from {len(X):,} events (the first run after "
                            "starting Sidescatter takes about half a minute longer)")
        import umap

        model = umap.UMAP(n_neighbors=min(st.umap_neighbors, len(X) - 1), min_dist=st.umap_min_dist,
                          random_state=st.seed, low_memory=True)
        with warnings.catch_warnings():  # "n_jobs overridden by random_state": the seed is the point
            warnings.filterwarnings("ignore", message=".*n_jobs.*", category=UserWarning)
            coords = model.fit_transform(X)
        stage("umap", 0.6, "UMAP: placing the other events")
        place("umap", coords)
        done += weights["umap"]

    if "tsne" in st.methods:
        import openTSNE

        n_iter, every, ran = 750, 25, [0]  # 250 early-exaggeration iterations, then 500

        def tick(iteration, error, embedding):
            ran[0] = min(ran[0] + every, n_iter)
            stage("tsne", 0.05 + 0.55 * ran[0] / n_iter, f"tSNE: iteration {ran[0]} of {n_iter}")
            return False

        stage("tsne", 0.02, f"tSNE: finding neighbours of {len(X):,} events")
        tsne = openTSNE.TSNE(perplexity=min(st.tsne_perplexity, (len(X) - 1) / 3),
                             n_iter=n_iter - 250, random_state=st.seed, n_jobs=-1,
                             callbacks=tick, callbacks_every_iters=every)
        emb = tsne.fit(X)
        place("tsne", np.asarray(emb))
        done += weights["tsne"]

    meta_medians = meta_counts = None
    if "flowsom" in st.methods:
        from .flowsom import FlowSOM

        stage("flowsom", 0.05, "FlowSOM: training the self-organising map")
        som = FlowSOM.fit(X, st.som_x, st.som_y, st.n_meta, seed=st.seed)
        stage("flowsom", 0.6, "FlowSOM: assigning events to clusters")
        cl_name, meta_name = out_names["flowsom"]
        meta_counts = np.zeros(st.n_meta, dtype=np.int64)
        for sid in st.samples:
            cl, meta = som.predict(np.nan_to_num(inputs.events[sid].astype(np.float64)))
            pop_values[sid][cl_name] = cl.astype(np.float32)
            pop_values[sid][meta_name] = meta.astype(np.float32)
            meta_counts += np.bincount(meta, minlength=st.n_meta + 1)[1:]
        all_meta = np.concatenate([pop_values[s][meta_name][picks[s]] for s in st.samples])
        meta_medians = np.array([np.median(X[all_meta == k], axis=0) if (all_meta == k).any()
                                 else np.full(X.shape[1], np.nan) for k in range(1, st.n_meta + 1)])
        done += weights["flowsom"]

    # Shift map coordinates so they start just above 0, for a linear scale.
    for method, names in out_names.items():
        if method == "flowsom":
            continue
        for name in names:
            vals = np.concatenate([pop_values[s][name] for s in st.samples])
            lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
            for sid in st.samples:
                pop_values[sid][name] = pop_values[sid][name] - lo + 0.05 * max(hi - lo, 1e-9)

    # ----- full-length channels: NaN (or cluster 0) outside the population --
    values = {}
    for sid in st.samples:
        idx = inputs.indices[sid]
        values[sid] = {}
        for name in st.all_output_channels():
            fill = 0.0 if name.startswith("FlowSOM") else np.nan
            full = np.full(inputs.sizes[sid], fill, dtype=np.float32)
            full[idx] = pop_values[sid][name]
            values[sid][name] = full
    sampled = {sid: inputs.indices[sid][picks[sid]] for sid in st.samples}
    progress(1.0, "Done")
    return Result(st, values, sampled, meta_medians, meta_counts)


class Analysis:
    """A finished analysis as kept in the workspace: settings and computed channels."""

    def __init__(self, settings, values, sampled, meta_medians=None, meta_counts=None):
        self.settings = settings
        self.values = values          # sample_id -> {channel: full-length float32 array}
        self.sampled = sampled        # sample_id -> event indices the map was built from
        self.meta_medians = meta_medians
        self.meta_counts = meta_counts

    @property
    def concatenated_id(self):
        return f"Concatenated ({self.settings.name})"

    def transforms(self):
        """Linear display scales for the added channels."""
        import flowkit as fk

        st = self.settings
        out = {}
        for method, names in st.output_channels().items():
            for i, name in enumerate(names):
                if method == "flowsom":
                    top = (st.som_x * st.som_y if i == 0 else st.n_meta) + 1
                else:
                    tops = [np.nanmax(v[name]) for v in self.values.values()
                            if np.isfinite(v[name]).any()]
                    top = max(tops, default=1.0) * 1.1 / 1.05
                out[name] = fk.transforms.LinearTransform(param_t=float(top), param_a=0.0)
        return out

    def metacluster_gate(self, k, xform, name=None):
        """A gate holding metacluster k, on the metacluster channel."""
        import flowkit as fk

        ch = self.settings.meta_channel
        lo, hi = xform.apply(np.array([[k - 0.5], [k + 0.5]])).ravel()
        dim = fk.Dimension(ch, compensation_ref="uncompensated", transformation_ref=ch,
                           range_min=float(lo), range_max=float(hi))
        return fk.gates.RectangleGate(name or f"Metacluster {k}", [dim])

    def metacluster_of(self, gate, xform):
        """Which metacluster a gate on the metacluster channel holds, or None."""
        ch = self.settings.meta_channel
        dims = getattr(gate, "dimensions", [])
        if not ch or len(dims) != 1 or getattr(dims[0], "id", None) != ch:
            return None
        xform_lo, xform_hi = dims[0].min, dims[0].max
        if xform_lo is None or xform_hi is None:
            return None
        centre = xform.inverse(np.array([[(xform_lo + xform_hi) / 2]])).ravel()[0]
        k = int(round(float(centre)))
        return k if 1 <= k <= self.settings.n_meta else None

    # ----- saving ------------------------------------------------------------

    def to_dict(self):
        d = {"settings": self.settings.to_dict(), "samples": list(self.values)}
        if self.meta_medians is not None:
            d["meta_medians"] = np.where(np.isfinite(self.meta_medians), self.meta_medians, None).tolist()
            d["meta_counts"] = [int(x) for x in self.meta_counts]
        return d

    def arrays_bytes(self):
        """Computed channels and sampled events, as an .npz file's bytes."""
        import io

        arrays = {}
        for i, sid in enumerate(self.values):
            for j, name in enumerate(self.settings.all_output_channels()):
                v = self.values[sid][name]
                arrays[f"s{i}_c{j}"] = v.astype(np.uint16) if name.startswith("FlowSOM") else v
            arrays[f"s{i}_sampled"] = self.sampled[sid]
        fh = io.BytesIO()
        np.savez_compressed(fh, **arrays)
        return fh.getvalue()

    @classmethod
    def from_dict(cls, d, npz_bytes):
        import io

        st = Settings.from_dict(d["settings"])
        z = np.load(io.BytesIO(npz_bytes))
        values, sampled = {}, {}
        for i, sid in enumerate(d["samples"]):
            values[sid] = {name: z[f"s{i}_c{j}"].astype(np.float32)
                           for j, name in enumerate(st.all_output_channels())}
            sampled[sid] = z[f"s{i}_sampled"]
        medians = d.get("meta_medians")
        medians = None if medians is None else np.array(medians, dtype=float)
        counts = d.get("meta_counts")
        return cls(st, values, sampled, medians, None if counts is None else np.array(counts))
