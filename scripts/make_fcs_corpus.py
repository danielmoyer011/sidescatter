"""Make the library of odd and broken FCS files in tests/data/fcs_corpus/.

    python scripts/make_fcs_corpus.py

Every file is written byte by byte here, so the library is tiny, has no
licence questions, and each quirk is exactly what its name says. expected.json
records what Sidescatter should do with each: "ok" (loads), "warn" (loads,
with a warning containing the given words) or "fail" (refused, with a reason
containing them). tests/test_fcs_corpus.py checks every file against it.
"""

import json
import shutil
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "tests" / "data" / "fcs_corpus"
NAMES = ["FSC-A", "SSC-A", "FL1-A", "FL2-A", "Time"]
N = 200


def _events(n=N, k=len(NAMES), top=1000):
    rng = np.random.default_rng(len(NAMES) * n + k)
    ev = rng.uniform(1, top, size=(n, k)).round()
    ev[:, -1] = np.arange(n)  # Time
    return ev


def fcs_bytes(events=None, names=NAMES, version="3.1", datatype="F", bits=32, big_endian=False,
              keywords=None, drop=(), delim="/", encoding="utf-8", lower=False,
              enddata_shift=0, header_data_offsets=True, truncate=0, nextdata=0):
    """One FCS dataset. keywords: extra TEXT keywords (or overrides); drop: keywords to leave out."""
    events = _events() if events is None else np.asarray(events)
    n, k = events.shape
    dtype = {("F", 32): "f4", ("D", 64): "f8", ("I", 16): "u2", ("I", 32): "u4"}[(datatype, bits)]
    dtype = (">" if big_endian else "<") + dtype
    data = np.ascontiguousarray(events, dtype=dtype).tobytes()
    word = bits // 8
    order = list(range(1, 5 if word >= 4 else 3))
    byteord = ",".join(map(str, reversed(order) if big_endian else order))
    if bits == 16:  # FCS 3.x wants the 32-bit form whatever the width; 2.0 writers vary
        byteord = ("4,3,2,1" if big_endian else "1,2,3,4") if version != "2.0" else byteord

    text = {"$BYTEORD": byteord, "$DATATYPE": datatype, "$MODE": "L", "$PAR": str(k), "$TOT": str(n),
            "$NEXTDATA": "0"}
    for i, name in enumerate(names):
        text[f"$P{i + 1}N"] = name
        text[f"$P{i + 1}B"] = str(bits)
        text[f"$P{i + 1}E"] = "0,0"
        text[f"$P{i + 1}R"] = "1024"
    if version == "3.2":
        text.pop("$MODE")
    text.update(keywords or {})
    for key in drop:
        text.pop(key, None)
    if lower:
        text = {key.lower(): v for key, v in text.items()}

    def body(begin, end, nxt):
        pairs = dict(text)
        if version != "2.0":
            pairs.update({"$BEGINANALYSIS": "0", "$ENDANALYSIS": "0", "$BEGINSTEXT": "0", "$ENDSTEXT": "0",
                          "$BEGINDATA": f"{begin:012d}", "$ENDDATA": f"{end + enddata_shift:012d}"})
        if nxt:
            pairs["$NEXTDATA"] = f"{nxt:012d}"
        esc = lambda v: v.replace(delim, delim * 2)  # noqa: E731
        return (delim + "".join(f"{esc(a)}{delim}{esc(b)}{delim}" for a, b in pairs.items())).encode(encoding)

    text_start = 58
    text_len = len(body(0, 0, 1 if nextdata else 0))
    begin = text_start + text_len
    end = begin + len(data) - 1
    total = begin + len(data)
    head_begin, head_end = (begin, end + enddata_shift) if header_data_offsets else (0, 0)
    header = f"FCS{version}    ".encode() + b"".join(f"{v:>8}".encode() for v in (
        text_start, begin - 1, head_begin, head_end, 0, 0))
    out = header + body(begin, end, total if nextdata else 0) + data
    if truncate:
        out = out[:-truncate]
    return out


def _spill(names, matrix):
    return ",".join([str(len(names)), *names, *(f"{v:g}" for v in np.ravel(matrix))])


def cases():
    two = [[1, 0.1], [0.05, 1]]
    yield "fcs20_int16_little.fcs", fcs_bytes(version="2.0", datatype="I", bits=16), "ok", ""
    yield "fcs20_int16_big.fcs", fcs_bytes(version="2.0", datatype="I", bits=16, big_endian=True), "ok", ""
    yield "fcs30_int32_big.fcs", fcs_bytes(version="3.0", datatype="I", bits=32, big_endian=True), "ok", ""
    yield "fcs31_float_little.fcs", fcs_bytes(), "ok", ""
    yield "fcs31_float_big.fcs", fcs_bytes(big_endian=True), "ok", ""
    yield "fcs31_double.fcs", fcs_bytes(datatype="D", bits=64), "ok", ""
    yield "fcs31_double_big.fcs", fcs_bytes(datatype="D", bits=64, big_endian=True), "ok", ""
    yield "fcs31_int16_little.fcs", fcs_bytes(datatype="I", bits=16), "ok", ""
    high = _events().astype(np.uint32)
    high[:, :4] |= 1 << 12  # junk above $PnR's bits, which readers must ignore
    yield "int16_junk_above_range.fcs", fcs_bytes(high, datatype="I", bits=16), "ok", ""
    yield "fcs32_float.fcs", fcs_bytes(version="3.2"), "ok", ""
    yield "one_event.fcs", fcs_bytes(_events(n=1)), "ok", ""
    yield "log_amplified.fcs", fcs_bytes(datatype="I", bits=16, keywords={
        "$P3E": "4,1", "$P4G": "2"}), "ok", ""
    yield "lowercase_keywords.fcs", fcs_bytes(lower=True), "ok", ""
    yield "keywords_utf8.fcs", fcs_bytes(keywords={"$P3S": "CD3 – Spécial", "$SRC": "Zürich µ-tube"}), "ok", ""
    yield "keywords_latin1.fcs", fcs_bytes(keywords={"$P3S": "CD3 Spécial", "$SRC": "Zürich"},
                                           encoding="latin-1"), "ok", ""
    yield "delimiter_in_value.fcs", fcs_bytes(keywords={"$P3S": "CD4/CD8", "$SRC": "a/b/c"}), "ok", ""
    yield "backslash_delimiter.fcs", fcs_bytes(delim="\\"), "ok", ""
    yield "missing_pnr.fcs", fcs_bytes(drop=[f"$P{i}R" for i in range(1, 6)]), "ok", ""
    yield "names_with_spaces_and_slashes.fcs", fcs_bytes(
        names=["FSC A", "SSC/Height", "FL1 Log", "PE/Cy7 - A", "Time"]), "ok", ""
    yield "offsets_only_in_text.fcs", fcs_bytes(header_data_offsets=False), "ok", ""
    yield "spill.fcs", fcs_bytes(keywords={"$SPILL": _spill(["FL1-A", "FL2-A"], two)}), "ok", ""
    yield "spillover.fcs", fcs_bytes(keywords={"SPILLOVER": _spill(["FL1-A", "FL2-A"], two)}), "ok", ""
    yield "spillover_unknown_channels.fcs", fcs_bytes(
        keywords={"$SPILLOVER": _spill(["FITC-A", "PE-A"], two)}), "warn", "spillover"
    yield "two_datasets.fcs", fcs_bytes(nextdata=1) + fcs_bytes(_events(n=50)), "warn", "one dataset"
    yield "enddata_off_by_one.fcs", fcs_bytes(enddata_shift=1), "warn", "offsets"
    yield "duplicate_channel_names.fcs", fcs_bytes(
        names=["FSC-A", "SSC-A", "FL1-A", "FL1-A", "Time"]), "warn", "same name"
    yield "zero_events.fcs", fcs_bytes(_events(n=0)), "fail", "no events"
    yield "truncated.fcs", fcs_bytes(_events(n=2000), truncate=20_000), "fail", "ends"
    yield "empty.fcs", b"", "fail", "empty"
    yield "header_only.fcs", fcs_bytes()[:40], "fail", "ends"
    yield "csv_renamed.fcs", b"FSC-A,SSC-A\n1,2\n3,4\n", "fail", "not an FCS file"
    yield "._resource_fork.fcs", b"\x00\x05\x16\x07\x00\x02\x00\x00Mac OS X        " + bytes(64), \
        "fail", "not an FCS file"
    yield "unsupported_10bit.fcs", fcs_bytes(datatype="I", bits=16, keywords={
        f"$P{i}B": "10" for i in range(1, 6)}), "fail", "-bit values"


# Files whose events differ from the standard 200 (everything else must match
# fcs31_float_little.fcs exactly once read).
OWN_EVENTS = {"one_event.fcs", "log_amplified.fcs"}


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    expected = {}
    for name, data, outcome, words in cases():
        (OUT / name).write_bytes(data)
        expected[name] = {"outcome": outcome, "words": words}
        if outcome != "fail":
            expected[name]["standard_events"] = name not in OWN_EVENTS
    (OUT / "expected.json").write_text(json.dumps(expected, indent=1, sort_keys=True) + "\n")
    print(f"Wrote {len(expected)} files to {OUT}")


if __name__ == "__main__":
    main()
