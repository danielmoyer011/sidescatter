# Demo spectral dataset

Cytek Aurora data (3 lasers, 44 detectors) from the
[flowSpecs](https://github.com/jtheorell/flowSpecs) R package by Jakob Theorell,
MIT licence (see `LICENSE-flowSpecs`), converted to FCS by
`scripts/make_spectral_demo.py`.

- `controls/`: 14 unmixing controls. 11 single-stained bead sets, unstained
  beads, a dead-cell marker (PO_DCM) on PBMCs, and unstained PBMCs used both as
  the dead-cell marker's negative and as the autofluorescence control.
- `PBMC_full_panel.fcs`: PBMCs stained with all 12 fluorochromes.
- `flowSpecs_reference_signatures.csv`: the signatures flowSpecs derived from
  these controls; the test suite checks Sidescatter's signatures against them.
