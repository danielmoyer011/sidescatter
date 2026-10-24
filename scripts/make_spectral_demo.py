"""Convert the flowSpecs example data (MIT licence, Jakob Theorell) into FCS files.

The flowSpecs R package ships a Cytek Aurora (3 lasers, 44 detectors) dataset:
14 unmixing controls and one 12-colour PBMC sample. This script turns them into
plain FCS 3.1 files so Sidescatter can use them as demo and test data.

Usage:
    git clone https://github.com/jtheorell/flowSpecs.git
    python scripts/make_spectral_demo.py flowSpecs/data src/sidescatter/demo_data/spectral
"""

import sys
import warnings
from pathlib import Path

import flowio
import numpy as np
import rdata


def _load(rda_path, name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = rdata.parser.parse_file(rda_path)
        return rdata.conversion.convert(parsed, default_encoding="utf8")[name]


def _write(frame, out_path, sample_name):
    events = np.asarray(frame.exprs.values, dtype=np.float32)
    names = [str(n) for n in frame.exprs.coords["dim_1"].values]
    meta = {"$CYT": "Aurora", "$FIL": out_path.name, "SAMPLE": sample_name}
    # Aurora detectors and scatter span 0..2^22; time keeps its own range.
    for i, name in enumerate(names, start=1):
        meta[f"$P{i}R"] = str(int(events[:, i - 1].max()) + 1) if name == "Time" else "4194304"
    with open(out_path, "wb") as fh:
        flowio.create_fcs(fh, events.flatten().tolist(), names, metadata_dict=meta)


def main(src_dir, out_dir):
    src_dir, out_dir = Path(src_dir), Path(out_dir)
    (out_dir / "controls").mkdir(parents=True, exist_ok=True)

    ctrls = _load(src_dir / "unmixCtrls.rda", "unmixCtrls")
    for fname, frame in ctrls.frames.items():
        _write(frame, out_dir / "controls" / str(fname), Path(str(fname)).stem)

    full = _load(src_dir / "fullPanel.rda", "fullPanel")
    _write(full, out_dir / "PBMC_full_panel.fcs", "PBMC_full_panel")
    print(f"wrote {len(ctrls.frames) + 1} FCS files to {out_dir}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
