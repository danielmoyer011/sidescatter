<img src="src/sidescatter/resources/icons/sidescatter-128.png" width="96" alt="Sidescatter icon: a laser beam hits a cell and scatters light in the five Aurora laser colours">

# Sidescatter

Free, open-source flow cytometry analysis for Windows, macOS and Linux, built
for full-spectrum data. Load raw spectral files and your unmixing controls,
explore the unmixing interactively, then gate and export statistics with a
workflow familiar to FlowJo users.

> Early development. Numbers are tested against reference data (see
> [Accuracy](#accuracy)), but check results against your current software
> before relying on them.

## What works today

- **FCS files**: FCS 2.0, 3.0 and 3.1, conventional and full-spectrum. Add them
  with File › Add samples, or drag files or whole folders onto the window
  (folders are searched recursively; anything that isn't FCS is skipped with a warning).
- **Unmixing tab**, next to the workspace:
  - add or drop every raw file of an experiment (files or whole folders); each
    is shown as Unstained or Single color (beads or cells), Multicolor sample,
    or Exclude, with the reason, and any role can be changed (for many rows
    at once by selecting them first);
  - several experiments per workspace, saved with it;
  - single-stain controls on **beads or cells** plus unstained controls;
    files are classified from their folder (a SpectroFlo "Reference Group"),
    name (fluor, marker, beads/cells, unstained), FCS keywords, or the events
    themselves, and each stain is paired with the unstained control of the
    same kind, or with its own dim events;
  - smart control clean-up, after AutoSpectral: saturated events and
    doublets are dropped, only the brightest positives are used, and cell
    controls subtract scatter-matched unstained cells instead of the average
    unstained cell; every signature is checked against your own saved
    spectra or one predicted from published dye spectra, and a control that
    doesn't match is flagged;
  - per-cell fluor variation (also after AutoSpectral): each fluor's
    control gives a set of slightly different spectra, each cell uses the
    one that fits it, and a Hotspots table shows which fluors' variation
    spreads into which;
  - AutoSpectral files: unmix with spectra AutoSpectral made (its
    `table_spectra` CSVs, autofluorescence files included), or write this
    experiment's spectra plus an AutoSpectral control file to run both tools
    on the same controls;
  - signature viewer, similarity matrix and complexity index;
  - ordinary or weighted least squares, with autofluorescence handled per
    cell (each cell is matched to the best of a library of AF spectra from
    the unstained cells, AutoSpectral style, adding `AF-A` and `AF Index`
    channels) or as one or more AF signatures unmixed like fluors;
  - draw AF populations on the unstained cells to add them to the library,
    give each sample its own tissue's unstained, and compare both AF methods
    on the unstained cells in the AF check tab and the QC report;
  - drag the positive gate on any control and see signatures, unmixed plots
    and per-detector residuals update live;
  - unmix any number of samples into new, gateable samples, with channels
    named like SpectroFlo's (`BV421-A`, marker as the description, `AF-A`);
  - export the experiment as a SpectroFlo-style package: `Raw/` and
    `Unmixed/` trees with `Reference Group/` and `Samples/` folders and the
    same file names in both, plus the unmixing matrix, a file manifest, a QC
    report and a Sidescatter workspace of the unmixed files.
- **FlowJo 10 layout**: a ribbon, the group table above the
  Name / Statistic / #Cells sample table, and graph windows that open when you
  double-click a population.
- **Gating**: rectangle, polygon, ellipse, quadrant, freehand, range and
  bisector tools in a strip right above each plot, drawn on pseudocolor, dot,
  contour or histogram plots, nested to any depth, editable by dragging. Gates
  apply to every sample in the workspace, as FlowJo group gates do.
- **Scales**: linear, log, FlowJo-style biex (the default for fluorescence),
  logicle, arcsinh, hyperlog and negative log (log mirrored below zero), per
  channel. Pick one from the button beside each axis picker (or right-click
  the axis); Customize… opens a Transform window with a live preview, range
  and width sliders, and Apply to Parameters. Changing a scale keeps gates on
  the same events.
- **Data cleaning** (Tools › Clean Data): removes margin events, flow-rate
  jumps (clogs, bubbles, pauses) and unstable stretches of a run, PeacoQC
  style, for every sample of a group at once. Lenient, Standard (PeacoQC's
  defaults) and Strict presets, or set MAD, isolation tree limit and window
  size yourself. Cleaned events become a **QC passed** population at the top
  of each sample's gates, so nothing in the files changes and every gate below
  inherits the cleaning. The QC window lists the % removed per sample (amber
  above 20%) and shows Time plots with the removed stretches shaded; drag,
  add or delete stretches by hand. Save a PDF report, export cleaned FCS
  files, or clean single-stain controls before their signatures are
  calculated (Unmixing tab › Clean controls first).
- **Statistics**: count, frequency of parent/grandparent/total, mean, median,
  geometric mean, SD, rSD, CV, rCV and percentiles, in a table editor that
  exports CSV or Excel.
- **UMAP, tSNE, PCA and FlowSOM** (Tools › UMAP / tSNE / FlowSOM, or right-click a
  population): pick a population, the samples and the channels (fluorescence
  ticked, autofluorescence, scatter and Time not). The samples are concatenated
  and downsampled to an equal number each (20,000 by default, at most
  1,000,000 in total), the methods run on each channel's display scale, and
  every other event of the population is then placed on the finished map.
  UMAP-1/2, TSNE-1/2, PCA-1/2, FlowSOM and FlowSOM meta are added to each
  sample as channels you can plot and gate; each FlowSOM metacluster becomes
  a population (rename it to "NK cells" and the name sticks), and a
  Concatenated sample with a Sample ID channel shows the combined map. Colour
  any plot by another channel's intensity, and see a heatmap of each
  metacluster's medians (CSV export). Runs in the background with a progress
  bar and Cancel; the seed is saved, so re-running gives the same map.
- **Layout Editor** (Sidescatter › Layout Editor, Ctrl+L): drag populations
  from the sample table onto Letter or A4 pages to build figures from
  pseudocolor, dot, contour and histogram plots plus text boxes (with
  `{sample}`, `{population}`, `{count}` and `{freq}` fields). Plots stay linked
  to the gates. Align, distribute, same size, snap to grid, undo/redo. Batch
  repeats a layout for every sample in a group, in rows or columns. Export to
  multi-page PDF, SVG, PNG or TIFF at any DPI, or print. Layouts are saved in
  the workspace.
- **FlowJo 10 workspaces**: open a `.wsp` with its FCS files to bring in
  samples, groups, compensation, scales and gates.
- **Workspaces** saved as `.ssw` (a zip holding readable JSON).
- **Demo data**: File › Open demo spectral data loads a 12-colour Cytek Aurora
  PBMC sample and its 14 unmixing controls.

## Install

Download the build for your system from the
[Releases page](https://github.com/danielmoyer011/sidescatter/releases):
`Sidescatter-windows.zip`, `Sidescatter-macos-arm64.dmg` (Apple silicon),
`Sidescatter-macos-intel.dmg` or `Sidescatter-linux-x86_64.tar.gz`.

Beta builds are not code-signed yet, so the first launch needs one extra step:

- **Windows**: unzip, open `Sidescatter.exe`, and if SmartScreen warns, choose
  **More info › Run anyway**.
- **macOS**: drag Sidescatter to Applications and open it. When macOS says it
  can't verify the developer, open **System Settings › Privacy & Security**,
  scroll down and click **Open Anyway**. If macOS instead says the app is
  damaged, run `xattr -dr com.apple.quarantine /Applications/Sidescatter.app`
  in Terminal.

From source (Python 3.10+):

```
pip install .
sidescatter
```

## Accuracy

`pytest` runs the checks below (`QT_QPA_PLATFORM=offscreen` on headless
machines; clone [FlowKit](https://github.com/whitews/FlowKit) into
`.testdata/FlowKit` to include the FlowJo comparisons):

- Signatures derived from the Aurora demo controls match the ones published
  with flowSpecs (cosine similarity above 0.995 for every fluor).
- Data cleaning removes exactly the events PeacoQC's own R code removes from
  PeacoQC's example file, and finds clogs, bubbles, drift and flow pauses
  injected into synthetic runs.
- Unmixing recovers known abundances exactly on synthetic data, and WLS beats
  OLS under photon noise.
- On synthetic cell controls where the stained cells are bigger and brighter
  in autofluorescence than the average unstained cell, scatter-matched
  negatives get the signature at least 3 times closer to the true one.
- On synthetic data where a fluor comes in several spectral variants, per-cell
  fluor variation cuts the spread it causes in other fluors by at least 20%.
- On synthetic tissue-like data with four kinds of autofluorescence, per-cell
  AF matching leaves at least 4 times less spread on unstained cells than a
  single AF signature, and recovers stained cells at least twice as
  accurately.
- Gate counts on FlowJo 10 workspaces: populations above 1,000 events agree
  with FlowJo within 0.5%. Small populations gated on FlowJo's biexponential
  scale can still differ by up to about 15%; closing that gap is the current
  accuracy priority.

## Project layout

- `src/sidescatter/core`: analysis engine with no GUI code (FCS loading,
  spectral unmixing, workspace and gating, statistics, file formats). Usable
  from Python scripts and notebooks.
- `src/sidescatter/app`: the PySide6 desktop interface.
- `src/sidescatter/resources`: app icon and bundled font. Regenerate the icon
  files with `python scripts/make_icons.py`.
- `packaging/`: PyInstaller build script used by the release workflow.

## Licence

GPL-3.0-or-later. Built on [FlowKit](https://github.com/whitews/FlowKit)
(BSD-3), PySide6 (LGPL-3) and pyqtgraph (MIT). Demo data from
[flowSpecs](https://github.com/jtheorell/flowSpecs) (MIT). The Sora typeface
used in the logo and splash screen is under the SIL Open Font License 1.1.
Dye spectra for the signature check come from [FPbase](https://www.fpbase.org)
(free of copyright restrictions; refresh with `scripts/fetch_fpbase_spectra.py`).
The control clean-up and per-cell autofluorescence follow the method of
AutoSpectral (Burton et al. 2025, bioRxiv 10.1101/2025.10.27.684855),
written independently in Python.

Sidescatter is not affiliated with FlowJo, LLC, BD, or Cytek Biosciences.
FlowJo is a trademark of Becton, Dickinson and Company.
