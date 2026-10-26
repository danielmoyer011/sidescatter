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
- **Spectral unmixing workbench** (Analysis › Spectral unmixing):
  - add single-stain controls on **beads or cells** plus unstained controls;
    files are classified by name and each stain is paired with the unstained
    control of the same kind, or with its own dim events;
  - signature viewer, similarity matrix and complexity index;
  - ordinary or weighted least squares, with one or more autofluorescence
    signatures from unstained cells;
  - drag the positive gate on any control and see signatures, unmixed plots
    and per-detector residuals update live;
  - unmix any number of samples into new, gateable samples.
- **FlowJo 10 layout**: a ribbon, the group table above the
  Name / Statistic / #Cells sample table, and graph windows that open when you
  double-click a population.
- **Gating**: rectangle, polygon, ellipse, quadrant, freehand, range and
  bisector tools in a strip right above each plot, drawn on pseudocolor, dot,
  contour or histogram plots, nested to any depth, editable by dragging. Gates
  apply to every sample in the workspace, as FlowJo group gates do.
- **Scales**: logicle (biexponential), arcsinh, log and linear per channel;
  changing a scale keeps gates on the same events.
- **Statistics**: count, frequency of parent/grandparent/total, mean, median,
  geometric mean, SD, rSD, CV, rCV and percentiles, in a table editor that
  exports CSV or Excel.
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
- Unmixing recovers known abundances exactly on synthetic data, and WLS beats
  OLS under photon noise.
- Gate counts on FlowJo 10 workspaces: populations above 1,000 events agree
  with FlowJo within 0.5%. Small populations gated on FlowJo's biexponential
  scale can still differ by up to about 15%; closing that gap is the current
  accuracy priority.

## Project layout

- `src/sidescatter/core`: analysis engine with no GUI code (FCS loading,
  spectral unmixing, workspace and gating, statistics, file formats). Usable
  from Python scripts and notebooks.
- `src/sidescatter/app`: the PySide6 desktop interface.
- `packaging/`: PyInstaller build script used by the release workflow.

## Licence

GPL-3.0-or-later. Built on [FlowKit](https://github.com/whitews/FlowKit)
(BSD-3), PySide6 (LGPL-3) and pyqtgraph (MIT). Demo data from
[flowSpecs](https://github.com/jtheorell/flowSpecs) (MIT).

Sidescatter is not affiliated with FlowJo, LLC, BD, or Cytek Biosciences.
FlowJo is a trademark of Becton, Dickinson and Company.
