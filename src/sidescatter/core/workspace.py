"""The workspace: samples, groups, transforms, compensation, and the gating tree.

Gating is done by a FlowKit GatingStrategy (GatingML 2.0 semantics). Gates
live in transformed ("display") space: every dimension references the
transform registered under its channel name, as FlowJo workspaces do.
"""

import copy
import json
import zipfile
from pathlib import Path

import flowkit as fk
from flowkit._models.gates._base_gate import Gate
import numpy as np
import pandas as pd

from . import cleaning, fcs, serialize, stats
from .layout import Layout, rename_population
from .spectral import Unmixer

ROOT = ("root",)
ALL_SAMPLES = "All Samples"
FORMAT_VERSION = 1

# Default display range for full-spectrum instruments (Cytek Aurora is 2^22).
SPECTRAL_RANGE = 4194304.0
CONVENTIONAL_RANGE = 262144.0


QC_GATE = "QC passed"


class QCGate(Gate):
    """The events that passed data cleaning: no shape, just one sample's mask.

    The template gate keeps every event; each cleaned sample gets a custom
    gate holding its own mask, so samples that were never cleaned are unaffected.
    """

    def __init__(self, gate_name=QC_GATE, keep=None):
        super().__init__(gate_name, [])
        self.gate_type = "QCGate"
        self.keep = keep

    def apply(self, df_events):
        n = len(df_events)
        if self.keep is None or len(self.keep) != n:
            return np.ones(n, dtype=bool)
        return self.keep.copy()


class Strategy(fk.GatingStrategy):
    """FlowKit's gating strategy, minus copying every event to apply a QC gate."""

    def _preprocess_sample_events(self, sample, gate, cache_events=False):
        if isinstance(gate, QCGate):
            return pd.DataFrame(index=pd.RangeIndex(sample.event_count))
        return super()._preprocess_sample_events(sample, gate, cache_events)


def default_transform(channel, top):
    if channel.kind in ("time", "scatter"):
        return fk.transforms.LinearTransform(param_t=top, param_a=0.0)
    return fk.transforms.LogicleTransform(param_t=top, param_w=0.5, param_m=4.5, param_a=0.0)


class Workspace:
    def __init__(self):
        self.strategy = Strategy()
        self.samples = {}        # sample_id -> fk.Sample
        # sample_id -> {"path": str}, {"unmixed_from": id, "unmixer": name}, or
        # {"unmixed_from_path": raw file not in the workspace, "unmixer": name}
        self.sources = {}
        self.groups = {ALL_SAMPLES: []}
        self.unmixers = {}       # name -> Unmixer
        self.experiments = []    # Unmixing tab setups, as experiment.Setup.to_dict() dicts
        self.compensation = {}   # sample_id -> "FCS" (acquisition spillover) or "uncompensated"
        self.layouts = []        # Layout Editor pages (core.layout.Layout)
        self.qc = {}             # sample_id -> cleaning.Result
        self.qc_settings = cleaning.Settings()  # what the Clean data dialog last used
        self.revision = 0        # bumped whenever gating results may have changed
        self._results = {}       # sample_id -> fk GatingResults
        self._display_cache = {}

    # ----- samples ---------------------------------------------------------

    def add_fcs(self, paths, group=None):
        ids = []
        for p in map(Path, paths):
            sample = fcs.load_sample(p, sample_id=self._unique_id(p.name))
            self.add_sample(sample, {"path": str(p.resolve())}, group)
            ids.append(sample.id)
        return ids

    def add_sample(self, sample, source, group=None):
        self.samples[sample.id] = sample
        self.sources[sample.id] = source
        self.groups[ALL_SAMPLES].append(sample.id)
        if group:
            self.groups.setdefault(group, []).append(sample.id)
        has_spill = sample.compensation is not None or "spill" in sample.get_metadata() \
            or "spillover" in sample.get_metadata()
        self.compensation[sample.id] = "FCS" if has_spill else "uncompensated"
        self._ensure_transforms(sample)
        return sample.id

    def remove_sample(self, sample_id):
        for members in self.groups.values():
            if sample_id in members:
                members.remove(sample_id)
        for d in (self.samples, self.sources, self.compensation, self._results, self.qc):
            d.pop(sample_id, None)
        qc = self.qc_gate()
        if qc is not None:
            self.strategy.remove_gate(qc, gate_path=ROOT, sample_id=sample_id)
        self._display_cache = {k: v for k, v in self._display_cache.items() if k[0] != sample_id}

    def _unique_id(self, name):
        sample_id, n = name, 2
        while sample_id in self.samples:
            sample_id, n = f"{name} ({n})", n + 1
        return sample_id

    def _ensure_transforms(self, sample):
        spectral = fcs.is_spectral(sample) or sample.id.endswith("(unmixed)")
        default_top = SPECTRAL_RANGE if spectral else CONVENTIONAL_RANGE
        raw = sample.get_events(source="raw")
        for ch in fcs.channels(sample):
            if ch.pnn in self.strategy.transformations:
                continue
            values = raw[:, ch.index]
            if ch.kind == "time":
                top = float(np.nanmax(values)) or 1.0
            else:
                # $PnR is often wrong in exported files, so never let data run off-scale
                pnr = float(sample.channels.loc[ch.index, "pnr"] or 0)
                top = max(pnr if pnr > 1 else default_top, _next_pow2(np.nanpercentile(values, 99.9)))
            self.strategy.add_transform(ch.pnn, default_transform(ch, top))

    def channels(self, sample_id):
        return fcs.channels(self.samples[sample_id])

    # ----- groups ----------------------------------------------------------

    def add_group(self, name, sample_ids=()):
        self.groups.setdefault(name, [])
        for sid in sample_ids:
            if sid not in self.groups[name]:
                self.groups[name].append(sid)

    def group_by_keyword(self, keyword):
        """Create one group per distinct value of an FCS keyword."""
        made = []
        for sid, sample in self.samples.items():
            value = sample.get_metadata().get(keyword.lower())
            if value:
                name = f"{keyword}={value}"
                self.add_group(name, [sid])
                made.append(name)
        return sorted(set(made))

    # ----- transforms ------------------------------------------------------

    def transform(self, channel):
        return self.strategy.transformations[channel]

    def set_transform(self, channel, new):
        """Change a channel's transform; gate coordinates are re-mapped so gates
        keep selecting the same events."""
        old = self.strategy.transformations[channel]
        for gate in self._all_gates():
            for dim in getattr(gate, "dimensions", []):
                if getattr(dim, "transformation_ref", None) != channel:
                    continue
                if isinstance(gate, fk.gates.RectangleGate):
                    dim.min = _remap(dim.min, old, new)
                    dim.max = _remap(dim.max, old, new)
            if isinstance(gate, fk.gates.PolygonGate):
                for i, dim in enumerate(gate.dimensions):
                    if dim.transformation_ref == channel:
                        for v in gate.vertices:
                            v[i] = _remap(v[i], old, new)
            if isinstance(gate, fk.gates.EllipsoidGate):
                # Scales aren't linear, so move the centre and stretch the
                # ellipse by the local slope of the new scale.
                cov = np.asarray(gate.covariance_matrix, dtype=float)
                for i, dim in enumerate(gate.dimensions):
                    if dim.transformation_ref == channel:
                        c, h = float(gate.coordinates[i]), 1e-3
                        slope = (_remap(c + h, old, new) - _remap(c - h, old, new)) / (2 * h)
                        gate.coordinates[i] = _remap(c, old, new)
                        cov[i, :] *= slope
                        cov[:, i] *= slope
                gate.covariance_matrix = cov.tolist()
        self.strategy.transformations[channel] = new
        self.invalidate()

    # ----- gating ----------------------------------------------------------

    def _all_gates(self):
        seen = []
        for node in self.strategy._dag.nodes:
            if node == ROOT:
                continue
            gate = self.strategy.get_gate(node[-1], node[:-1])
            seen.append(gate)
        return seen

    def make_dimension(self, sample_id, channel, range_min=None, range_max=None):
        ch = next(c for c in self.channels(sample_id) if c.pnn == channel)
        comp = self.compensation.get(sample_id, "uncompensated") \
            if ch.kind == "fluorescence" else "uncompensated"
        return fk.Dimension(channel, compensation_ref=comp, transformation_ref=channel,
                            range_min=range_min, range_max=range_max)

    def add_gate(self, gate, parent_path=ROOT, sample_id=None):
        self.strategy.add_gate(gate, tuple(parent_path), sample_id=sample_id)
        self.invalidate()

    def remove_gate(self, name, parent_path):
        if tuple(parent_path) == ROOT and name == self.qc_gate():
            self.remove_qc()  # keeps the gates inside it
            return
        self.strategy.remove_gate(name, gate_path=tuple(parent_path))
        self.invalidate()

    def rename_gate(self, name, new_name, parent_path):
        self.strategy.rename_gate(name, new_name, gate_path=tuple(parent_path))
        rename_population(self.layouts, parent_path, name, new_name)
        self.invalidate()

    def get_gate(self, name, parent_path, sample_id=None):
        return self.strategy.get_gate(name, tuple(parent_path), sample_id=sample_id)

    def children(self, parent_path=ROOT):
        """(name, path) of each child population of the population at parent_path."""
        parent_path = tuple(parent_path)
        if parent_path == ROOT:
            names = [g.gate_name for g in self.strategy.get_root_gates()]
        else:
            names = self.strategy.get_child_gate_ids(parent_path[-1], parent_path[:-1])
            names = [n[0] if isinstance(n, tuple) else n for n in names]
        return [(n, parent_path) for n in names]

    def population_paths(self):
        """Every population as (name, parent_path), depth first."""
        out = []

        def walk(path):
            for name, parent in self.children(path):
                out.append((name, parent))
                walk(parent + (name,))

        walk(ROOT)
        return out

    def invalidate(self, sample_id=None):
        self.revision += 1
        if sample_id is None:
            self._results.clear()
        else:
            self._results.pop(sample_id, None)
        self.strategy.clear_cache()

    def results(self, sample_id):
        if sample_id not in self._results:
            sample = self.samples[sample_id]
            self._results[sample_id] = self._strategy_for(sample).gate_sample(sample)
        return self._results[sample_id]

    def applies_to(self, sample_id, name, parent_path=ROOT):
        """Whether a population can be computed for a sample (it has the gate's channels)."""
        if name is None:
            return True
        path = tuple(parent_path) + (name,)
        have = set(self.samples[sample_id].pnn_labels)
        for i in range(1, len(path)):
            if not _gate_channels(self.get_gate(path[i], path[:i])) <= have:
                return False
        return True

    def _strategy_for(self, sample):
        """The gating strategy, minus gates on channels this sample doesn't have.

        Raw spectral files and their unmixed versions share a workspace, so a
        gate drawn on an unmixed channel must simply not apply to raw files.
        """
        have = set(sample.pnn_labels)
        drop = [(n, p) for n, p in self.population_paths()
                if not _gate_channels(self.get_gate(n, p)) <= have]
        if not drop:
            return self.strategy
        pruned = copy.deepcopy(self.strategy)
        for name, parent in drop:
            try:
                pruned.remove_gate(name, gate_path=parent)
            except (KeyError, ValueError):
                pass  # already removed with its parent
        return pruned

    def membership(self, sample_id, name=None, parent_path=ROOT):
        """Boolean event mask for a population; name=None means all events."""
        sample = self.samples[sample_id]
        if name is None:
            return np.ones(sample.event_count, dtype=bool)
        if not self.applies_to(sample_id, name, parent_path):
            raise KeyError(f"{name} does not apply to {sample_id}")
        return self.results(sample_id).get_gate_membership(name, gate_path=tuple(parent_path))

    def count(self, sample_id, name=None, parent_path=ROOT):
        return int(self.membership(sample_id, name, parent_path).sum())

    # ----- events for plotting and statistics ------------------------------

    def compensated_events(self, sample_id):
        """Events after compensation (if the sample has a spillover matrix), untransformed."""
        key = (sample_id, "comp")
        if key not in self._display_cache:
            sample = self.samples[sample_id]
            if self.compensation.get(sample_id) == "FCS" and sample.compensation is None:
                spill = sample.get_metadata().get("spill") or sample.get_metadata().get("spillover")
                sample.apply_compensation(spill)
            source = "comp" if sample.compensation is not None else "raw"
            self._display_cache[key] = sample.get_events(source=source)
        return self._display_cache[key]

    def display_values(self, sample_id, channel, mask=None):
        """Values for one channel on the display (transformed) scale."""
        sample = self.samples[sample_id]
        idx = sample.get_channel_index(channel)
        values = self.compensated_events(sample_id)[:, idx]
        if mask is not None:
            values = values[mask]
        xform = self.strategy.transformations[channel]
        return xform.apply(values.reshape(-1, 1)).ravel()

    def scale_values(self, sample_id, channel, mask=None):
        """Values for one channel on the linear scale (used for MFI and other stats)."""
        sample = self.samples[sample_id]
        values = self.compensated_events(sample_id)[:, sample.get_channel_index(channel)]
        return values if mask is None else values[mask]

    # ----- statistics ------------------------------------------------------

    def statistic(self, sample_id, stat, name=None, parent_path=ROOT, channel=None, percentile=50.0):
        parent_path = tuple(parent_path)
        mask = self.membership(sample_id, name, parent_path)
        count = int(mask.sum())
        if stat == "Count":
            return count
        if stat == "Freq. of Parent":
            parent = self.count(sample_id, *(_split(parent_path)))
            return stats.frequency(count, parent)
        if stat == "Freq. of Grandparent":
            gp = parent_path[:-1] if len(parent_path) > 1 else ROOT
            return stats.frequency(count, self.count(sample_id, *(_split(gp))))
        if stat == "Freq. of Total":
            return stats.frequency(count, self.samples[sample_id].event_count)
        if channel is None:
            raise ValueError(f"{stat} needs a channel")
        return stats.channel_stat(self.scale_values(sample_id, channel, mask), stat, percentile)

    def statistics_table(self, sample_ids, columns):
        """A pandas table: one row per sample, one column per requested statistic.

        columns: list of dicts with keys name, parent_path, stat, channel (optional).
        """
        import pandas as pd

        rows = []
        for sid in sample_ids:
            row = {"Sample": sid}
            for col in columns:
                label = _column_label(col)
                try:
                    row[label] = self.statistic(sid, col["stat"], col.get("name"),
                                                col.get("parent_path", ROOT), col.get("channel"),
                                                col.get("percentile", 50.0))
                except KeyError:
                    row[label] = float("nan")  # population missing for this sample
            rows.append(row)
        return pd.DataFrame(rows)

    # ----- data cleaning ---------------------------------------------------

    def qc_gate(self):
        """Name of the QC passed population, if there is one."""
        for gate in self.strategy.get_root_gates():
            if isinstance(gate, QCGate):
                return gate.gate_name
        return None

    def time_values(self, sample_id):
        """Raw Time values, or None when the sample has no Time channel."""
        ch = next((c for c in self.channels(sample_id) if c.kind == "time"), None)
        if ch is None:
            return None, None
        return self.samples[sample_id].get_events(source="raw")[:, ch.index], ch.pnn

    def time_step(self, sample_id):
        """Seconds per Time unit ($TIMESTEP), or None."""
        try:
            return float(self.samples[sample_id].get_metadata().get("timestep"))
        except (TypeError, ValueError):
            return None

    def default_qc_channels(self, sample_id):
        return cleaning.default_channels(self.channels(sample_id))

    def _margin_input(self, sample_id, names, raw=None):
        """Raw values, $PnR and names of the channels checked for margin events.

        Unmixed samples are checked on their raw detectors, where saturation
        happens: raw, if given, or the raw file they were unmixed from.
        """
        src = self.sources.get(sample_id, {})
        sample = self.samples[sample_id]
        if raw is None and ("unmixed_from" in src or "unmixed_from_path" in src):
            try:
                raw = self._raw_for(src)
            except (FileNotFoundError, KeyError, OSError, ValueError):
                raw = None
        if raw is not None:
            if raw.event_count == sample.event_count:
                chans = [c for c in fcs.channels(raw) if c.kind in ("scatter", "fluorescence")]
                return (raw.get_events(source="raw")[:, [c.index for c in chans]],
                        [float(raw.channels.loc[c.index, "pnr"] or 0) for c in chans],
                        [c.pnn for c in chans])
        idx = [sample.get_channel_index(n) for n in names]
        return (sample.get_events(source="raw")[:, idx],
                [float(sample.channels.loc[i, "pnr"] or 0) for i in idx], list(names))

    def clean(self, sample_ids, settings=None, progress=None, raw=None):
        """Clean samples with the same settings and put them under QC passed.

        A sample whose stretches were edited by hand keeps the edits when it is
        cleaned again with the same settings. progress(done, total, sample_id)
        is called before each sample. raw: optional {sample_id: raw Sample} for
        unmixed samples' margin check. Returns {sample_id: cleaning.Result}.
        """
        raw = raw or {}
        settings = settings or self.qc_settings
        self.qc_settings = settings
        done = {}
        for i, sid in enumerate(sample_ids):
            if progress:
                progress(i, len(sample_ids), sid)
            have = set(self.samples[sid].pnn_labels)
            names = [c for c in (settings.channels or self.default_qc_channels(sid)) if c in have]
            values = np.column_stack([self.display_values(sid, c) for c in names]) if names \
                else np.zeros((self.samples[sid].event_count, 0))
            time, time_name = self.time_values(sid)
            mraw, ranges, mnames = self._margin_input(sid, names, raw.get(sid))
            res = cleaning.clean(values, names, time, settings, mraw, ranges, mnames, time_name)
            old = self.qc.get(sid)
            if old is not None and old.edited and \
                    cleaning.Settings.from_dict(old.settings).same_method(settings):
                res.stretches, res.edited = [list(x) for x in old.stretches], True
            self.qc[sid] = res
            done[sid] = res
        self._sync_qc()
        return done

    def qc_keep(self, sample_id):
        """Events of a sample that passed cleaning (all events if it wasn't cleaned)."""
        res = self.qc.get(sample_id)
        if res is None:
            return np.ones(self.samples[sample_id].event_count, dtype=bool)
        return ~res.removed(self.time_values(sample_id)[0])

    def qc_plot_data(self, sample_id, channels=None):
        """Time plots of a cleaned sample (see cleaning.plot_data)."""
        res = self.qc[sample_id]
        names = [c for c in (channels or res.channels) if c in self.samples[sample_id].pnn_labels]
        values = np.column_stack([self.display_values(sample_id, c) for c in names]) if names \
            else np.zeros((res.n_events, 0))
        return cleaning.plot_data(res, self.time_values(sample_id)[0], values, names,
                                  self.time_step(sample_id))

    def set_qc_stretches(self, sample_id, spans):
        """Replace a sample's removed stretches with (start, end) spans in Time units."""
        res = self.qc[sample_id]
        cleaning.set_stretch_times(res, self.time_values(sample_id)[0], spans)
        self._sync_qc()

    def reset_qc_stretches(self, sample_id):
        res = self.qc[sample_id]
        res.stretches, res.edited = [list(x) for x in res.auto_stretches], False
        self._sync_qc()

    def remove_qc(self, sample_ids=None):
        """Undo cleaning. With no samples left cleaned, QC passed goes and its gates move back up."""
        for sid in list(self.qc) if sample_ids is None else sample_ids:
            self.qc.pop(sid, None)
        self._sync_qc()

    def _sync_qc(self):
        qc = self.qc_gate()
        if not self.qc:
            if qc is not None:
                self.strategy.remove_gate(qc, gate_path=ROOT, keep_children=True)
                _move_layouts(self.layouts, (ROOT[0], qc), ROOT)
            self.invalidate()
            return
        if qc is None:
            qc = _unique(QC_GATE, [n for n, _ in self.children(ROOT)])
            self._nest_root_gates(qc)
        node = self.strategy._get_gate_node(qc, ROOT)
        node.custom_gates.clear()
        for sid in self.qc:
            if self.samples.get(sid) is not None:  # an export drops events once a sample is written
                node.custom_gates[sid] = QCGate(qc, self.qc_keep(sid))
        self.invalidate()

    def _nest_root_gates(self, qc):
        """Put a new QC gate at the top and every existing gate inside it."""
        old = self.strategy
        new = Strategy()
        for k, v in old.transformations.items():
            new.add_transform(k, v)
        for k, v in old.comp_matrices.items():
            new.add_comp_matrix(k, v)
        new.add_gate(QCGate(qc), ROOT)
        top = (ROOT[0], qc)

        def moved(path):
            path = tuple(path)
            return top + path[1:] if path[:1] == ROOT else path

        for name, parent in self.population_paths():
            node = old._get_gate_node(name, parent)
            if isinstance(node.gate, fk.gates.Quadrant):
                continue  # added with its QuadrantGate
            gate = node.gate
            if isinstance(gate, fk.gates.BooleanGate):
                gate = copy.deepcopy(gate)
                for ref in gate.gate_refs:
                    ref["path"] = moved(ref["path"])
            new.add_gate(gate, moved(parent))
            for sid, custom in node.custom_gates.items():
                new.add_gate(custom, moved(parent), sample_id=sid)
        self.strategy = new
        _move_layouts(self.layouts, ROOT, top)

    def export_cleaned(self, sample_id, path):
        """Write the events that passed cleaning (all channels, raw values) as an FCS file."""
        sample = self.samples[sample_id]
        keep = self.qc_keep(sample_id)
        meta = {k: v for k, v in fcs.keywords(sample).items() if not k.lower().startswith("p")
                or not k[1:2].isdigit()}
        meta["SIDESCATTER_QC"] = f"{int(keep.sum())} of {len(keep)} events passed"
        fcs.write_fcs(path, sample.get_events(source="raw")[keep], list(sample.pnn_labels),
                      list(sample.pns_labels), meta, sample.channels["pnr"].tolist())
        return int(keep.sum())

    # ----- spectral --------------------------------------------------------

    def add_unmixer(self, name, unmixer):
        self.unmixers[name] = unmixer

    def unmix(self, sample_id, unmixer_name, group=None):
        """Unmix a raw spectral sample into a new sample in this workspace."""
        from .unmixing import unmix_sample

        unmixed = unmix_sample(self.samples[sample_id], self.unmixers[unmixer_name])
        new_id = self._unique_id(f"{sample_id} (unmixed)")
        unmixed.id = new_id
        self.add_sample(unmixed, {"unmixed_from": sample_id, "unmixer": unmixer_name}, group)
        return new_id

    def unmix_file(self, path, unmixer_name, group=None, raw_sample=None):
        """Unmix a raw file that stays out of the workspace (the Unmixing tab's files).

        Only the unmixed sample is added; reopening the workspace unmixes the
        raw file again from its path.
        """
        from .unmixing import unmix_sample

        path = Path(path)
        raw = raw_sample if raw_sample is not None else fcs.load_sample(path)
        unmixed = unmix_sample(raw, self.unmixers[unmixer_name])
        unmixed.id = self._unique_id(f"{path.name} (unmixed)")
        return self.add_sample(unmixed, {"unmixed_from_path": str(path.resolve()), "unmixer": unmixer_name},
                               group)

    def _raw_for(self, src, search_dirs=()):
        if "unmixed_from" in src:
            return self.samples[src["unmixed_from"]]
        return fcs.load_sample(_find_file(src["unmixed_from_path"], search_dirs))

    def reunmix(self, unmixer_name):
        """Recompute every sample unmixed with this unmixer (after controls change)."""
        from .unmixing import unmix_sample

        for sid, src in list(self.sources.items()):
            if src.get("unmixer") == unmixer_name:
                fresh = unmix_sample(self._raw_for(src), self.unmixers[unmixer_name])
                fresh.id = sid
                self.samples[sid] = fresh
                self._display_cache = {k: v for k, v in self._display_cache.items() if k[0] != sid}
                self.invalidate(sid)
        if self.qc:
            self._sync_qc()

    # ----- save / load -----------------------------------------------------

    def save(self, path):
        doc = {
            "format": "sidescatter-workspace",
            "version": FORMAT_VERSION,
            "samples": [
                {"id": sid, "source": self.sources[sid], "compensation": self.compensation[sid]}
                for sid in self.samples
            ],
            "groups": self.groups,
            "transforms": {k: serialize.transform_to_dict(v)
                           for k, v in self.strategy.transformations.items()},
            "gates": [_gate_dict(self.get_gate(n, p), p) for n, p in self.population_paths()],
            "unmixers": {k: v.to_dict() for k, v in self.unmixers.items()},
            "layouts": [lay.to_dict() for lay in self.layouts],
            "experiments": self.experiments,
            "qc": {sid: r.to_dict() for sid, r in self.qc.items() if sid in self.samples},
            "qc_settings": self.qc_settings.to_dict(),
        }
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("workspace.json", json.dumps(doc, indent=1))

    @classmethod
    def load(cls, path, search_dirs=()):
        with zipfile.ZipFile(path) as zf:
            doc = json.loads(zf.read("workspace.json"))
        if doc.get("format") != "sidescatter-workspace":
            raise ValueError("not a Sidescatter workspace")

        ws = cls()
        for name, d in doc["transforms"].items():
            ws.strategy.add_transform(name, serialize.transform_from_dict(d))
        ws.unmixers = {k: Unmixer.from_dict(v) for k, v in doc["unmixers"].items()}

        pending = []
        for s in doc["samples"]:
            src = s["source"]
            if "path" in src:
                p = _find_file(src["path"], [Path(path).parent, *map(Path, search_dirs)])
                sample = fcs.load_sample(p, sample_id=s["id"])
                ws.add_sample(sample, {"path": str(p)})
                ws.compensation[s["id"]] = s["compensation"]
            else:
                pending.append(s)
        for s in pending:  # unmixed samples are rebuilt from their raw source
            from .unmixing import unmix_sample

            src = s["source"]
            raw = ws._raw_for(src, [Path(path).parent, *map(Path, search_dirs)])
            sample = unmix_sample(raw, ws.unmixers[src["unmixer"]])
            sample.id = s["id"]
            ws.add_sample(sample, src)

        ws.groups = {k: list(v) for k, v in doc["groups"].items()}
        ws.experiments = doc.get("experiments", [])
        for g in doc["gates"]:
            if g["type"] == "QCGate":
                ws.strategy.add_gate(QCGate(g["name"]), ROOT)
            else:
                ws.strategy.add_gate(serialize.gate_from_dict(g), tuple(g["parent_path"]))
        ws.layouts = [Layout.from_dict(d) for d in doc.get("layouts", [])]
        ws.qc_settings = cleaning.Settings.from_dict(doc.get("qc_settings", {}))
        ws.qc = {sid: cleaning.Result.from_dict(r) for sid, r in doc.get("qc", {}).items()
                 if sid in ws.samples}
        if ws.qc or ws.qc_gate():
            ws._sync_qc()
        return ws

    # ----- FlowJo import ---------------------------------------------------

    @classmethod
    def from_flowjo(cls, wsp_path, fcs_dir):
        """Import a FlowJo 10 workspace: samples, groups, compensation, transforms, gates."""
        fj = fk.Workspace(str(wsp_path), fcs_samples=str(fcs_dir))
        ws = cls()
        sample_ids = fj.get_sample_ids()
        if not sample_ids:
            raise ValueError("no samples in the workspace matched files in " + str(fcs_dir))

        first_strategy = fj.get_gating_strategy(sample_ids[0])
        ws.strategy = copy.deepcopy(first_strategy)
        for sid in sample_ids:
            sample = fj.get_sample(sid)
            ws.samples[sid] = sample
            ws.sources[sid] = {"path": str(Path(fcs_dir) / sid)}
            ws.groups[ALL_SAMPLES].append(sid)
            ws.compensation[sid] = "FCS"
            strategy = fj.get_gating_strategy(sid)
            for node in strategy._dag.nodes:
                if node == ROOT:
                    continue
                gate = strategy.get_gate(node[-1], node[:-1])
                template = first_strategy.get_gate(node[-1], node[:-1]) \
                    if node in first_strategy._dag.nodes else None
                if template is None or not serialize.same_gate(gate, template):
                    ws.strategy.add_gate(copy.deepcopy(gate), node[:-1], sample_id=sid)
            for name, xform in strategy.transformations.items():
                ws.strategy.transformations.setdefault(name, xform)
            for comp_id, matrix in strategy.comp_matrices.items():
                ws.strategy.comp_matrices.setdefault(comp_id, matrix)
        for name in fj.get_sample_groups():
            if name != ALL_SAMPLES:
                ws.groups[name] = list(fj.get_sample_ids(name))
        return ws


def _gate_dict(gate, parent_path):
    if isinstance(gate, QCGate):
        return {"name": gate.gate_name, "parent_path": list(parent_path), "type": "QCGate"}
    return serialize.gate_to_dict(gate, parent_path)


def _move_layouts(layouts, old_prefix, new_prefix):
    """Point layout items at their populations' new paths after gates were nested or un-nested.

    Items showing all events stay on all events.
    """
    old_prefix, n = list(old_prefix), len(old_prefix)
    for lay in layouts:
        for it in lay.items:
            if it.population[:n] == old_prefix and len(it.population) > 1:
                it.population = list(new_prefix) + it.population[n:]


def _unique(name, taken):
    out, k = name, 2
    while out in taken:
        out, k = f"{name} {k}", k + 1
    return out


def _gate_channels(gate):
    if isinstance(gate, fk.gates.BooleanGate):
        return set()
    return {d.dimension_ref if hasattr(d, "dimension_ref") else d.id for d in gate.dimensions}


def _next_pow2(x):
    return float(2 ** int(np.ceil(np.log2(max(float(x), 1.0)))))


def _remap(value, old, new):
    if value is None:
        return None
    raw = old.inverse(np.array([[value]], dtype=float))
    return float(new.apply(raw).ravel()[0])


def _split(path):
    """Population path -> (name, parent_path) arguments; ROOT means all events."""
    path = tuple(path)
    if path == ROOT:
        return (None, ROOT)
    return (path[-1], path[:-1])


def _column_label(col):
    pop = "/".join([*col.get("parent_path", ROOT)[1:], col.get("name") or "All events"])
    label = f"{pop} | {col['stat']}"
    if col.get("channel"):
        label += f" ({col['channel']})"
    return label


def _find_file(original, search_dirs):
    p = Path(original)
    if not p.is_absolute():  # relative to the workspace, as in exported experiments
        for d in search_dirs:
            if (Path(d) / p).exists():
                return Path(d) / p
    if p.exists():
        return p
    for d in search_dirs:
        candidate = Path(d) / p.name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"cannot find {p.name}; looked in {[str(d) for d in search_dirs]}")
