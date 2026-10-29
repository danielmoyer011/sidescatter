"""The workspace: samples, groups, transforms, compensation, and the gating tree.

Gating is done by a FlowKit GatingStrategy (GatingML 2.0 semantics). Gates
live in transformed ("display") space: every dimension references the
transform registered under its channel name, as FlowJo workspaces do.
"""

import copy
import itertools
import json
import zipfile
from pathlib import Path

import flowkit as fk
import numpy as np

from . import fcs, highdim, serialize, stats
from .layout import Layout, rename_population
from .spectral import Unmixer

ROOT = ("root",)
ALL_SAMPLES = "All Samples"
FORMAT_VERSION = 1

# Default display range for full-spectrum instruments (Cytek Aurora is 2^22).
SPECTRAL_RANGE = 4194304.0
CONVENTIONAL_RANGE = 262144.0


def default_transform(channel, top):
    if channel.kind in ("time", "scatter"):
        return fk.transforms.LinearTransform(param_t=top, param_a=0.0)
    return fk.transforms.LogicleTransform(param_t=top, param_w=0.5, param_m=4.5, param_a=0.0)


class Workspace:
    def __init__(self):
        self.strategy = fk.GatingStrategy()
        self.samples = {}        # sample_id -> fk.Sample
        # sample_id -> {"path": str}, {"unmixed_from": id, "unmixer": name}, or
        # {"unmixed_from_path": raw file not in the workspace, "unmixer": name}
        self.sources = {}
        self.groups = {ALL_SAMPLES: []}
        self.unmixers = {}       # name -> Unmixer
        self.experiments = []    # Unmixing tab setups, as experiment.Setup.to_dict() dicts
        self.compensation = {}   # sample_id -> "FCS" (acquisition spillover) or "uncompensated"
        self.layouts = []        # Layout Editor pages (core.layout.Layout)
        self.analyses = []       # UMAP/tSNE/FlowSOM runs (core.highdim.Analysis)
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
        for d in (self.samples, self.sources, self.compensation, self._results):
            d.pop(sample_id, None)
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
        """Remove a gate, everything inside it, and the Boolean gates that use any of those."""
        _remove_with_booleans(self.strategy, name, tuple(parent_path))
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

    # ----- Boolean gates ---------------------------------------------------

    def is_boolean(self, name, parent_path):
        return isinstance(self.get_gate(name, parent_path), fk.gates.BooleanGate)

    def boolean_inputs(self, name, parent_path):
        """(op, [(name, parent_path, negated), ...]) of a Boolean gate, as the editor shows it."""
        gate = self.get_gate(name, parent_path)
        refs = [(r["ref"], tuple(r["path"]), bool(r.get("complement", False))) for r in gate.gate_refs]
        if gate.type == "not":  # GatingML NOT: one input, negated
            return "and", [(n, p, True) for n, p, _ in refs[:1]]
        return gate.type, refs

    def boolean_parent(self, inputs):
        """Where a Boolean gate on these inputs goes: the deepest population that
        contains all of them (the top level if they share nothing)."""
        parents = [tuple(p) for _, p, *_ in inputs]
        common = parents[0]
        for p in parents[1:]:
            n = 0
            while n < min(len(common), len(p)) and common[n] == p[n]:
                n += 1
            common = common[:n]
        return common or ROOT

    def boolean_candidates(self, exclude=None):
        """Populations a Boolean gate may use as inputs. With exclude=(name, parent),
        leaves out that gate, everything inside it, and every Boolean gate built on it
        (so editing a gate can't make it depend on itself)."""
        pops = self.population_paths()
        if exclude is None:
            return pops
        own = tuple(exclude[1]) + (exclude[0],)
        banned = [own] + [p + (n,) for n, p in _boolean_dependents(self.strategy, *exclude)]
        return [(n, p) for n, p in pops if not any((p + (n,))[:len(b)] == b for b in banned)]

    def add_boolean(self, inputs, op="and", name=None):
        """Add a Boolean gate. inputs: [(name, parent_path, negated), ...]; op: "and" or "or".
        Returns (name, parent_path) of the new population."""
        inputs = _check_inputs(inputs, op)
        parent = self.boolean_parent(inputs)
        taken = {n for n, _ in self.children(parent)}
        name = (name or "").strip() or boolean_name(inputs, op)
        if name in taken:
            raise ValueError(f"there is already a population called {name} there")
        self.add_gate(_boolean_gate(name, inputs, op), parent)
        return name, parent

    def edit_boolean(self, name, parent_path, inputs, op="and", new_name=None):
        """Change a Boolean gate's inputs, operator and name. Returns its new (name, parent_path)."""
        parent_path = tuple(parent_path)
        inputs = _check_inputs(inputs, op)
        allowed = set(self.boolean_candidates(exclude=(name, parent_path)))
        for n, p, _ in inputs:
            if (n, tuple(p)) not in allowed:
                raise ValueError(f"{name} can't use {n}: it is {name} itself, inside it, or built on it")
        new_parent = self.boolean_parent(inputs)
        new_name = (new_name or "").strip() or name
        if new_parent == parent_path:
            gate = self.get_gate(name, parent_path)
            replacement = _boolean_gate(name, inputs, op)
            gate.type, gate.gate_refs = replacement.type, replacement.gate_refs
            self.strategy._rebuild_dag()
            self.invalidate()
        else:
            if self.children(parent_path + (name,)):
                raise ValueError(f"{name} has populations inside it, so its inputs must still "
                                 "share the same parent population")
            if name in {n for n, _ in self.children(new_parent)}:
                raise ValueError(f"there is already a population called {name} there")
            users = _boolean_dependents(self.strategy, name, parent_path)
            if users:
                raise ValueError(f"other Boolean gates use {name}, so its inputs must still "
                                 "share the same parent population")
            self.strategy.remove_gate(name, gate_path=parent_path)
            self.add_gate(_boolean_gate(name, inputs, op), new_parent)
        if new_name != name:
            self.rename_gate(name, new_name, new_parent)
        return new_name, new_parent

    def boolean_combinations(self, inputs, include_all_negative=False):
        """One AND gate for every +/- combination of the inputs (FlowJo's "make all
        combinations"), e.g. IFNg+ TNF- IL-2+. Returns the new (name, parent_path)s."""
        inputs = [(n, tuple(p)) for n, p, *_ in inputs]
        if len(inputs) < 2:
            raise ValueError("pick at least two populations to combine")
        parent = self.boolean_parent(inputs)
        taken = {n for n, _ in self.children(parent)}
        made = []
        for signs in itertools.product((False, True), repeat=len(inputs)):
            if all(signs) and not include_all_negative:
                continue
            combo = [(n, p, neg) for (n, p), neg in zip(inputs, signs)]
            name = _unique(combination_name(combo), taken)
            self.strategy.add_gate(_boolean_gate(name, combo, "and"), parent)
            made.append((name, parent))
        self.invalidate()
        return made

    def boolean_dependents(self, name, parent_path):
        """Boolean gates outside this population that use it or anything inside it,
        and so would be deleted with it."""
        own = tuple(parent_path) + (name,)
        return [(n, p) for n, p in _boolean_dependents(self.strategy, name, tuple(parent_path))
                if (p + (n,))[:len(own)] != own]

    def plot_axes(self, name, parent_path):
        """The channels a population's own gate is drawn on. For a Boolean gate, those of its
        first input (following nested Boolean gates)."""
        seen = set()
        while name is not None and (name, tuple(parent_path)) not in seen:
            seen.add((name, tuple(parent_path)))
            gate = self.get_gate(name, parent_path)
            if not isinstance(gate, fk.gates.BooleanGate):
                return [d.dimension_ref if hasattr(d, "dimension_ref") else d.id
                        for d in getattr(gate, "dimensions", [])]
            first = gate.gate_refs[0]
            name, parent_path = first["ref"], tuple(first["path"])
        return []

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
            gate = self.get_gate(path[i], path[:i])
            if not _gate_channels(gate) <= have:
                return False
            if isinstance(gate, fk.gates.BooleanGate) and not all(
                    self.applies_to(sample_id, r["ref"], r["path"]) for r in gate.gate_refs):
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
            _remove_with_booleans(pruned, name, parent)
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

    def unmix_file(self, path, unmixer_name, group=None, raw_sample=None, af_source=None, progress=None):
        """Unmix a raw file that stays out of the workspace (the Unmixing tab's files).

        Only the unmixed sample is added; reopening the workspace unmixes the
        raw file again from its path. af_source: the unstained file whose
        per-cell AF library entries this sample uses (None: all of them).
        """
        from .unmixing import unmix_sample

        path = Path(path)
        raw = raw_sample if raw_sample is not None else fcs.load_sample(path)
        unmixed = unmix_sample(raw, self.unmixers[unmixer_name], af_source=af_source, progress=progress)
        unmixed.id = self._unique_id(f"{path.name} (unmixed)")
        src = {"unmixed_from_path": str(path.resolve()), "unmixer": unmixer_name}
        if af_source and af_source != "shared":
            src["af_source"] = str(af_source)
        return self.add_sample(unmixed, src, group)

    def _raw_for(self, src, search_dirs=()):
        if "unmixed_from" in src:
            return self.samples[src["unmixed_from"]]
        return fcs.load_sample(_find_file(src["unmixed_from_path"], search_dirs))

    def reunmix(self, unmixer_name):
        """Recompute every sample unmixed with this unmixer (after controls change)."""
        from .unmixing import unmix_sample

        for sid, src in list(self.sources.items()):
            if src.get("unmixer") == unmixer_name:
                fresh = unmix_sample(self._raw_for(src), self.unmixers[unmixer_name],
                                     af_source=src.get("af_source"))
                fresh.id = sid
                self.samples[sid] = fresh
                self._attach_derived(sid)
                self._display_cache = {k: v for k, v in self._display_cache.items() if k[0] != sid}
                self.invalidate(sid)

    # ----- UMAP, tSNE, PCA and FlowSOM --------------------------------------

    def new_analysis_settings(self, population=None, parent_path=ROOT, samples=()):
        """Settings for a new analysis with a unique name and channel names."""
        n = len(self.analyses) + 1
        names = {a.settings.name for a in self.analyses}
        while f"Analysis {n}" in names:
            n += 1
        taken = {c for a in self.analyses for c in a.settings.all_output_channels()}
        st = highdim.Settings(name=f"Analysis {n}", population=population,
                              parent_path=tuple(parent_path), samples=list(samples))
        k = 1
        while taken & set(highdim.Settings(suffix=st.suffix, methods=list(highdim.METHODS))
                           .all_output_channels()):
            k += 1
            st.suffix = f" ({k})"
        if st.samples:
            st.channels = highdim.default_channels(self.channels(st.samples[0]))
        return st

    def analysis(self, name):
        return next((a for a in self.analyses if a.settings.name == name), None)

    def add_analysis(self, result):
        """Add a finished analysis: channels on each sample, metacluster gates, and
        a concatenated sample of the events the map was built from.

        Re-running an analysis (same name) replaces it.
        """
        st = result.settings
        if self.analysis(st.name) is not None:
            self.remove_analysis(st.name)
        record = highdim.Analysis(st, result.values, result.sampled, result.meta_medians,
                                  result.meta_counts)
        self.analyses.append(record)
        for name, xform in record.transforms().items():
            self.strategy.transformations[name] = xform
        for sid in st.samples:
            self._attach_derived(sid)
        if st.meta_channel:
            for k in range(1, st.n_meta + 1):
                self.strategy.add_gate(record.metacluster_gate(k, self.transform(st.meta_channel)),
                                       st.population_path)
        self._add_concatenated(record)
        self.add_group(st.name, [*st.samples, record.concatenated_id])
        self.invalidate()
        return record

    def remove_analysis(self, name):
        """Remove an analysis with its channels, gates (and anything gated under them),
        concatenated sample and group."""
        record = self.analysis(name)
        if record is None:
            return
        st = record.settings
        added = set(st.all_output_channels())
        for gname, parent in reversed(self.population_paths()):
            try:
                gate = self.get_gate(gname, parent)
            except Exception:
                continue  # removed with its parent already
            if _gate_channels(gate) & added:
                self.strategy.remove_gate(gname, gate_path=parent)
        if record.concatenated_id in self.samples:
            self.remove_sample(record.concatenated_id)
        for sid in st.samples:
            if sid in self.samples:
                fcs.drop_channels(self.samples[sid], added)
                self._display_cache = {k: v for k, v in self._display_cache.items() if k[0] != sid}
        for ch in added:
            self.strategy.transformations.pop(ch, None)
        if self.groups.get(name) is not None:
            del self.groups[name]
        self.analyses.remove(record)
        self.invalidate()

    def _attach_derived(self, sample_id):
        """Put every analysis's channels on a (re)loaded sample."""
        sample = self.samples[sample_id]
        for record in self.analyses:
            cols = record.values.get(sample_id)
            if not cols:
                continue
            names = [n for n in cols if n not in sample.pnn_labels]
            if names and len(cols[names[0]]) == sample.event_count:
                fcs.append_channels(sample, names, [cols[n] for n in names])
        self._display_cache = {k: v for k, v in self._display_cache.items() if k[0] != sample_id}

    def _add_concatenated(self, record):
        """A sample of the events each map was built from, with a Sample ID channel."""
        st = record.settings
        sids = [s for s in st.samples if s in self.samples]
        if not sids:
            return
        first = self.samples[sids[0]]
        common = [c for c in first.pnn_labels
                  if all(c in self.samples[s].pnn_labels for s in sids)]
        derived = set(st.all_output_channels())
        common = [c for c in common if c not in derived and fcs._channel_kind(c) != "derived"]
        blocks = []
        for i, sid in enumerate(sids):
            sample = self.samples[sid]
            rows = record.sampled[sid]
            ev = self.compensated_events(sid)
            cols = [ev[rows, sample.get_channel_index(c)] for c in common]
            cols.append(np.full(len(rows), i + 1, dtype=float))
            cols += [record.values[sid][c][rows] for c in st.all_output_channels()]
            blocks.append(np.column_stack(cols))
        pns = dict(zip(first.pnn_labels, first.pns_labels))
        names = common + [highdim.SAMPLE_ID] + st.all_output_channels()
        keywords = {k: v for k, v in first.get_metadata().items() if k.lower() not in ("spill", "spillover")}
        keywords["$FIL"] = record.concatenated_id
        sample = fcs.load_sample(np.vstack(blocks), record.concatenated_id, names,
                                 [pns.get(n, "") for n in common] + ["; ".join(sids)[:200]] +
                                 [""] * len(st.all_output_channels()), keywords)
        # Values are compensated already, and the gates' "FCS" compensation
        # means "none" for a file without a spillover keyword.
        self.strategy.transformations.setdefault(
            highdim.SAMPLE_ID, fk.transforms.LinearTransform(param_t=len(sids) + 1, param_a=0.0))
        self.add_sample(sample, {"concatenated": st.name})
        self.compensation[sample.id] = "uncompensated"

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
            "gates": [serialize.gate_to_dict(self.get_gate(n, p), p)
                      for n, p in self.population_paths()],
            "unmixers": {k: v.to_dict() for k, v in self.unmixers.items()},
            "layouts": [lay.to_dict() for lay in self.layouts],
            "experiments": self.experiments,
            "analyses": [a.to_dict() for a in self.analyses],
        }
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("workspace.json", json.dumps(doc, indent=1))
            for i, a in enumerate(self.analyses):
                zf.writestr(f"analyses/{i}.npz", a.arrays_bytes())

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

        pending, concatenated = [], []
        for s in doc["samples"]:
            src = s["source"]
            if "concatenated" in src:
                concatenated.append(s)
            elif "path" in src:
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
            sample = unmix_sample(raw, ws.unmixers[src["unmixer"]], af_source=src.get("af_source"))
            sample.id = s["id"]
            ws.add_sample(sample, src)

        with zipfile.ZipFile(path) as zf:
            ws.analyses = [highdim.Analysis.from_dict(d, zf.read(f"analyses/{i}.npz"))
                           for i, d in enumerate(doc.get("analyses", []))]
        for sid in ws.samples:
            ws._attach_derived(sid)
        for s in concatenated:
            record = ws.analysis(s["source"]["concatenated"])
            if record is not None:
                ws._add_concatenated(record)
        ws.groups = {k: list(v) for k, v in doc["groups"].items()}
        ws.experiments = doc.get("experiments", [])
        for g in doc["gates"]:
            ws.strategy.add_gate(serialize.gate_from_dict(g), tuple(g["parent_path"]))
        ws.layouts = [Layout.from_dict(d) for d in doc.get("layouts", [])]
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


def boolean_name(inputs, op="and"):
    """ "CD4+ AND NOT Treg", "NOT Dump" """
    word = f" {op.upper()} "
    return word.join(("NOT " if neg else "") + n for n, _, neg in inputs)


def combination_name(inputs):
    """ "IFNg+ TNF- IL-2+" for one +/- combination of the inputs."""
    return " ".join(n.rstrip("+-− ") + ("-" if neg else "+") for n, _, neg in inputs)


def _check_inputs(inputs, op):
    inputs = [(n, tuple(p), bool(neg)) for n, p, neg in inputs]
    if op not in ("and", "or"):
        raise ValueError("a Boolean gate is AND or OR")
    if not inputs:
        raise ValueError("pick the populations to combine")
    if len(inputs) == 1 and not inputs[0][2]:
        raise ValueError("pick at least two populations, or tick NOT for a single one")
    if len({(n, p) for n, p, _ in inputs}) < len(inputs):
        raise ValueError("each population can only be used once")
    return inputs


def _boolean_gate(name, inputs, op):
    if len(inputs) == 1:  # GatingML NOT takes exactly one input
        n, p, _ = inputs[0]
        return fk.gates.BooleanGate(name, "not", [{"ref": n, "path": p, "complement": False}])
    refs = [{"ref": n, "path": p, "complement": neg} for n, p, neg in inputs]
    return fk.gates.BooleanGate(name, op, refs)


def _boolean_dependents(strategy, name, parent_path):
    """Every Boolean gate that must go when this gate goes: those inside it, and those
    that use it, anything inside it, or another doomed Boolean gate (transitively)."""
    doomed = [tuple(parent_path) + (name,)]
    out = []

    def inside(path):
        return any(path[:len(d)] == d for d in doomed)

    changed = True
    while changed:
        changed = False
        for node in list(strategy._dag.nodes):
            if node in (ROOT, doomed[0]) or node in [p + (n,) for n, p in out]:
                continue
            try:
                gate = strategy.get_gate(node[-1], node[:-1])
            except Exception:
                continue  # quadrants, refs to gates not added yet
            if not isinstance(gate, fk.gates.BooleanGate):
                continue
            refs = [tuple(r["path"]) + (r["ref"],) for r in gate.gate_refs]
            if inside(node) or any(inside(r) for r in refs):
                out.append((node[-1], node[:-1]))
                doomed.append(node)
                changed = True
    return out


def _remove_with_booleans(strategy, name, parent_path):
    """Remove a gate from a strategy, first removing the Boolean gates that depend on it
    (FlowKit refuses to remove a gate a Boolean gate still uses)."""
    pending = sorted(_boolean_dependents(strategy, name, parent_path),
                     key=lambda np_: len(np_[1]), reverse=True)
    while pending:
        progress = False
        for n, p in list(pending):
            if p + (n,) not in strategy._dag.nodes:
                pending.remove((n, p))  # went with an earlier removal
                continue
            try:
                strategy.remove_gate(n, gate_path=p)
            except fk.exceptions.GateTreeError:
                continue  # still used by another Boolean gate; try again after it goes
            pending.remove((n, p))
            progress = True
        if not progress and pending:
            break
    try:
        strategy.remove_gate(name, gate_path=tuple(parent_path))
    except (KeyError, ValueError):
        pass  # already removed with its parent


def _unique(name, taken):
    out, n = name, 2
    while out in taken:
        out, n = f"{name} ({n})", n + 1
    taken.add(out)
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
