"""Packaging an unmixing experiment the way Cytek SpectroFlo exports one.

SpectroFlo's exported experiment is a folder (usually zipped) with a Raw and
an Unmixed tree that mirror each other: references in "Reference Group",
samples in their group folder, and the same file name for a tube's raw and
unmixed FCS files. Sidescatter writes the same trees, with all samples in
"Samples", and puts open files where SpectroFlo has its own (.expt, .wtml,
.ust): the unmixing matrix, a manifest of what every file is, a QC report,
and a Sidescatter workspace of the unmixed files.

    <experiment>/
        Raw/Reference Group/*.fcs        raw files, copied byte for byte
        Raw/Samples/*.fcs
        Unmixed/Reference Group/*.fcs    unmixed with the experiment's matrix
        Unmixed/Samples/*.fcs
        <experiment> unmixing matrix.csv
        manifest.csv
        QC report.txt
        <experiment>.ssw
"""

import csv
import io
import os
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import flowio
import numpy as np

from . import fcs, spectral, unmixing

RAW, UNMIXED = "Raw", "Unmixed"
REFERENCE_GROUP, SAMPLES = "Reference Group", "Samples"
ROLE_LABELS = {"single_stain": "Single color", "unstained": "Unstained",
               "sample": "Multicolor sample", "exclude": "Excluded"}
QC_EVENTS = 50_000


@dataclass
class ExportResult:
    path: Path            # the zip file or the experiment folder
    n_references: int
    n_samples: int
    warnings: list


def role_label(spec):
    label = ROLE_LABELS[spec.role]
    return f"{label} ({spec.carrier})" if spec.is_reference else label


def safe_name(name):
    """A file or folder name that works on Windows, macOS and Linux."""
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip().rstrip(".")
    return name or "Experiment"


def default_name(specs, samples=None):
    """Acquisition date plus the name the sample files share, e.g. "2026-10-24 PBMC panel"."""
    date = ""
    for spec in specs:
        sample = (samples or {}).get(spec.path)
        if sample is not None:
            raw = fcs.keywords(sample).get("date", "")
        else:
            try:
                raw = flowio.FlowData(str(spec.path), only_text=True).text.get("date", "")
            except Exception:
                continue
        for fmt in ("%d-%b-%Y", "%Y-%m-%d", "%d-%b-%y", "%m/%d/%Y", "%d.%m.%Y"):
            try:
                date = datetime.strptime(raw.strip(), fmt).strftime("%Y-%m-%d")
                break
            except ValueError:
                continue
        if date:
            break
    stems = [s.path.stem for s in specs if s.role == "sample"] or [s.path.stem for s in specs]
    prefix = os.path.commonprefix(stems).strip(" _-.") if len(stems) > 1 else (stems[0] if stems else "")
    if len(prefix) < 3:
        prefix = "Experiment"
    return safe_name(" ".join(x for x in (date, prefix.replace("_", " ")) if x))


def layout(specs):
    """Where each included file goes: {spec.path: (group folder, file name)}."""
    out, used = {}, set()
    for spec in specs:
        if spec.role == "exclude":
            continue
        group = REFERENCE_GROUP if spec.is_reference else SAMPLES
        name, n = spec.path.name, 2
        while (group, name.lower()) in used:  # same name from different folders
            name, n = f"{spec.path.stem} ({n}){spec.path.suffix}", n + 1
        used.add((group, name.lower()))
        out[spec.path] = (group, name)
    return out


def matrix_csv(unmixer):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    labels = unmixer.channel_labels()
    w.writerow(["Fluor", "Channel", "Marker", "Autofluorescence", "Source file", *unmixer.detectors])
    for ref, (pnn, _) in zip(unmixer.references, labels):
        w.writerow([ref.name, pnn, ref.marker, "yes" if ref.is_autofluorescence else "",
                    ref.source, *(f"{v:.6g}" for v in ref.signature)])
    if unmixer.per_cell_af:
        # Per-cell AF: the AF-A row above is the median; each cell used one of these.
        for i, sig in enumerate(unmixer.af_library):
            src = Path(unmixer.af_sources[i]).name if unmixer.af_sources else ""
            w.writerow([f"AF library {i + 1}", f"AF Index = {i + 1}", "", "library", src,
                        *(f"{v:.6g}" for v in sig)])
    return buf.getvalue()


def manifest_csv(specs, places):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["File", "Role", "Carrier", "Fluor", "Marker", "Negative", "Raw file", "Unmixed file",
                "Original location", "Guessed from", "Why", "AF source"])
    for spec in specs:
        group, name = places.get(spec.path, ("", ""))
        neg = spec.negative if spec.negative in ("auto", "internal") else Path(spec.negative).name
        w.writerow([
            spec.path.name, ROLE_LABELS[spec.role], spec.carrier if spec.is_reference else "",
            spec.name if spec.role == "single_stain" else "", spec.marker,
            neg if spec.role == "single_stain" else "",
            f"{RAW}/{group}/{name}" if group else "", f"{UNMIXED}/{group}/{name}" if group else "",
            str(spec.path), spec.guessed_by, "; ".join(spec.reasons),
            af_source_label(spec) if spec.role == "sample" else ""])
    return buf.getvalue()


def af_source_label(spec):
    return "Shared library" if spec.af_source in ("shared", "", None) else Path(spec.af_source).name


def residual_fraction(sample, unmixer, af_source=None):
    """Median per-event |residual| / |signal| on a fixed subset of events."""
    idx = unmixing._detector_index(sample, unmixer.detectors)
    raw = sample.get_events(source="raw")
    rows = np.sort(np.random.default_rng(0).choice(len(raw), min(len(raw), QC_EVENTS), replace=False))
    Y = raw[rows][:, idx].astype(np.float64)
    R = unmixer.residuals(Y, unmixer.unmix(Y, None if af_source == "shared" else af_source))
    return float(np.median(np.abs(R).sum(axis=1) / np.maximum(np.abs(Y).sum(axis=1), 1e-9)))


def qc_report(name, unmixer, specs, residuals, problems, af_check=None):
    names = unmixer.names
    sim = spectral.similarity_matrix(unmixer.matrix)
    lines = [f"Unmixing QC report: {name}",
             f"Written by Sidescatter on {datetime.now():%Y-%m-%d %H:%M}", "",
             f"Method: {'weighted' if unmixer.method == 'wls' else 'ordinary'} least squares "
             f"({unmixer.method.upper()})",
             f"Components: {len(names)} ({sum(r.is_autofluorescence for r in unmixer.references)} "
             "autofluorescence)",
             "Autofluorescence: " + (f"per-cell matching to a library of {len(unmixer.af_library)} spectra"
                                     if unmixer.per_cell_af else "signatures as fluors"),
             f"Detectors: {len(unmixer.detectors)}",
             f"Complexity index: {spectral.complexity_index(unmixer.matrix):.2f}", ""]
    pairs = sorted(((sim[i, j], names[i], names[j]) for i in range(len(names))
                    for j in range(i + 1, len(names))), reverse=True)
    lines.append("Most similar signatures (1 = identical; above 0.98 is very hard to separate):")
    lines += [f"  {a} / {b}: {v:.3f}" for v, a, b in pairs[:5]] or ["  (only one component)"]
    lines += ["", "Residual after unmixing (median |residual| / |signal|; lower is better):"]
    lines += [f"  {f}: {v:.3f}" for f, v in residuals.items()] or ["  (no files)"]
    if af_check is not None:
        lines += [""] + unmixing.af_check_text(af_check)
    if problems:
        lines += ["", "Notes:"] + [f"  {level}: {msg}" for level, msg in problems]
    lines += ["", "Files:"]
    lines += [f"  {s.path.name}: {role_label(s)}" + (f", {s.name}" if s.role == "single_stain" else "")
              for s in specs]
    width = max(len(n) for n in names) + 2
    lines += ["", "Similarity matrix:", " " * width + "".join(f"{n[:8]:>9}" for n in names)]
    lines += [f"{n:<{width}}" + "".join(f"{v:9.3f}" for v in row) for n, row in zip(names, sim)]
    return "\n".join(lines) + "\n"


def export_experiment(dest_dir, name, specs, unmixer, as_zip=True, samples=None,
                      unmixer_name="Unmixing", progress=None, af_settings=None):
    """Write the experiment into dest_dir as <name>.zip or a <name>/ folder.

    specs: every file of the experiment (excluded ones are skipped). samples:
    optional {path: loaded Sample}; files not in it are loaded one at a time.
    progress(done, total, text) is called before each file.
    af_settings: keyword arguments for unmixing.af_check (the AF comparison in
    the QC report), e.g. n_autofluorescence and af_library_size.
    Returns an ExportResult. Nothing is left behind if writing fails.
    """
    from .workspace import Workspace  # workspace imports unmixing; keep this module light

    problems = [p for p in unmixing.check_specs(specs) if p[0] == "error"]
    if problems:
        raise ValueError(" ".join(msg for _, msg in problems))
    name = safe_name(name)
    dest_dir = Path(dest_dir)
    final = dest_dir / (f"{name}.zip" if as_zip else name)
    if final.exists():
        raise FileExistsError(f"{final} already exists")

    places = layout(specs)
    included = [s for s in specs if s.path in places]
    samples = samples or {}
    ws = Workspace()
    ws.add_unmixer(unmixer_name, unmixer)
    residuals = {}
    work = Path(tempfile.mkdtemp(prefix="sidescatter-export-", dir=dest_dir))
    try:
        root = work / name
        for tree in (RAW, UNMIXED):
            for group in (REFERENCE_GROUP, SAMPLES):
                (root / tree / group).mkdir(parents=True)
        for i, spec in enumerate(included):
            group, fname = places[spec.path]
            if progress:
                progress(i, len(included), f"Unmixing {spec.path.name}")
            shutil.copy2(spec.path, root / RAW / group / fname)
            sample = samples.get(spec.path) or fcs.load_sample(spec.path)
            af_source = spec.af_source if spec.role == "sample" else None
            events, pnn, pns, ranges, meta = unmixing.unmixed_events(sample, unmixer, af_source=af_source)
            meta["fil"] = fname
            out = root / UNMIXED / group / fname
            fcs.write_fcs(out, events, pnn, pns, meta, ranges)
            del events
            residuals[fname] = residual_fraction(sample, unmixer, af_source)
            unmixed = fcs.load_sample(out, sample_id=ws._unique_id(fname))
            ws.add_sample(unmixed, {"path": f"{UNMIXED}/{group}/{fname}"},
                          "Unmixed" if group == SAMPLES else REFERENCE_GROUP)
            # The saved workspace only needs ids, sources and transforms, so the
            # events can go now rather than piling up over a big experiment.
            ws.samples[unmixed.id] = None
        if progress:
            progress(len(included), len(included), "Writing the experiment files")

        (root / f"{name} unmixing matrix.csv").write_text(matrix_csv(unmixer), encoding="utf-8")
        (root / "manifest.csv").write_text(manifest_csv(specs, places), encoding="utf-8")
        warnings = [msg for level, msg in unmixing.check_specs(specs) if level == "warning"]
        check = None
        if any(r.is_autofluorescence for r in unmixer.references):
            check = unmixing.af_check(included, unmixer, samples, **(af_settings or {}))
        (root / "QC report.txt").write_text(
            qc_report(name, unmixer, specs, residuals, [("warning", w) for w in warnings], check),
            encoding="utf-8")
        ws.save(root / f"{name}.ssw")

        if as_zip:
            tmp_zip = work / f"{name}.zip"
            with zipfile.ZipFile(tmp_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
                for f in sorted(root.rglob("*")):
                    zf.write(f, f.relative_to(work).as_posix())
            os.replace(tmp_zip, final)
        else:
            shutil.move(str(root), final)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    n_ref = sum(s.is_reference for s in included)
    return ExportResult(final, n_ref, len(included) - n_ref, warnings)


@dataclass
class Setup:
    """An unmixing experiment as the Unmixing tab edits it: files and settings."""

    name: str
    specs: list = field(default_factory=list)
    method: str = "wls"
    autofluorescence: bool = True
    n_autofluorescence: int = 1
    af_mode: str = "per_cell"      # "per_cell" library matching, or "fluors"
    af_library_size: int = 30      # most automatic AF spectra per unstained file
    af_auto: bool = True           # False: only the hand-drawn AF gates

    @property
    def af_settings(self):
        """The AF keyword arguments of unmixing.build_unmixer and af_check."""
        return {"n_autofluorescence": self.n_autofluorescence,
                "af_library_size": self.af_library_size, "af_auto": self.af_auto}

    def to_dict(self):
        return {
            "name": self.name, "method": self.method, "autofluorescence": self.autofluorescence,
            "n_autofluorescence": self.n_autofluorescence, "af_mode": self.af_mode,
            "af_library_size": self.af_library_size, "af_auto": self.af_auto,
            "files": [{"path": str(s.path), "name": s.name, "role": s.role, "carrier": s.carrier,
                       "negative": s.negative, "marker": s.marker, "reasons": list(s.reasons),
                       "guessed_by": s.guessed_by, "af_gates": list(s.af_gates),
                       "af_source": s.af_source} for s in self.specs],
        }

    @classmethod
    def from_dict(cls, d):
        specs = [unmixing.ControlSpec(Path(f["path"]), f["name"], f["role"], f["carrier"], f["negative"],
                                      marker=f.get("marker", ""), reasons=f.get("reasons", []),
                                      guessed_by=f.get("guessed_by", ""),
                                      af_gates=list(f.get("af_gates", [])),
                                      af_source=f.get("af_source", "shared"))
                 for f in d["files"]]
        # Setups saved before per-cell AF existed keep AF as fluors.
        return cls(d["name"], specs, d.get("method", "wls"), d.get("autofluorescence", True),
                   d.get("n_autofluorescence", 1), d.get("af_mode", "fluors"),
                   d.get("af_library_size", 30), d.get("af_auto", True))
