"""Download the dye spectra Sidescatter uses to predict signatures, from FPbase.

FPbase (https://www.fpbase.org) states its data are free of copyright
restrictions and asks that the original authors be credited; the output keeps
FPbase's name for every dye so it can be looked up there. Run this to refresh
src/sidescatter/resources/spectra/fpbase_dyes.json:

    python scripts/fetch_fpbase_spectra.py
"""

import json
import sys
import urllib.request
from pathlib import Path

URL = "https://www.fpbase.org/graphql/"
OUT = Path(__file__).resolve().parents[1] / "src" / "sidescatter" / "resources" / "spectra" / "fpbase_dyes.json"

# Sidescatter's fluor name -> FPbase's name for the dye or protein.
NAMES = {
    "BUV395": "BD Horizon BUV395", "BUV496": "BD Horizon BUV496", "BUV563": "BD Horizon BUV563",
    "BUV661": "BD Horizon BUV661", "BUV737": "BD Horizon BUV737", "BUV805": "BD Horizon BUV805",
    "BV421": "Brilliant Violet 421", "BV480": "Brilliant Violet 480", "BV510": "Brilliant Violet 510",
    "BV570": "Brilliant Violet 570", "BV605": "Brilliant Violet 605", "BV650": "Brilliant Violet 650",
    "BV711": "Brilliant Violet 711", "BV750": "Brilliant Violet 750", "BV785": "Brilliant Violet 785",
    "BB515": "BD Horizon BB515", "BB700": "BD Horizon - BB700",
    "V450": "BD Horizon V450", "V500": "BD Horizon V500",
    "SB436": "Super Bright 436", "SB600": "Super Bright 600", "SB645": "Super Bright 645",
    "SB702": "Super Bright 702", "SB780": "Super Bright 780",
    "eF450": "eFluor 450", "eF506": "eFluor 506", "eF660": "eFluor 660", "eF710": "eFluor 710",
    "PB": "Pacific Blue", "PO": "Pacific Orange",
    "FITC": "Fluorescein (FITC)", "GFP": "EGFP", "YFP": "EYFP", "mCherry": "mCherry",
    "tdTomato": "tdTomato",
    "AF350": "Alexa Fluor 350", "AF405": "Alexa Fluor 405", "AF430": "Alexa Fluor 430",
    "AF488": "Alexa Fluor 488", "AF532": "Alexa Fluor 532", "AF555": "Alexa Fluor 555",
    "AF568": "Alexa Fluor 568", "AF594": "Alexa Fluor 594", "AF647": "Alexa Fluor 647",
    "AF660": "Alexa Fluor 660", "AF700": "Alexa Fluor 700", "AF750": "Alexa Fluor 750",
    "PE": "PE (R-PE / R-phycoerythrin)", "PE-CF594": "PE-CF594", "PE-Dazzle594": "PE/Dazzle 594",
    "PE-Texas Red": "PE-Texas Red", "PE-AF610": "PE-Alexa Fluor 610", "PE-Cy5": "PE-Cy5",
    "PE-Cy5.5": "PE-Cy5.5", "PE-Cy7": "PE-Cy7", "PE-Fire 780": "PE/Fire 780",
    "PE-eF610": "PE-eFluor 610",
    "PerCP": "PerCP", "PerCP-Cy5.5": "PerCP-Cy5.5", "PerCP-eF710": "PerCP-eFluor 710",
    "APC": "APC (allophycocyanin)", "APC-Cy7": "APC/Cy7", "APC-H7": "APC/H7",
    "APC-eF780": "APC-eFluor 780", "APC-Fire 750": "APC/Fire-750",
    "Cy5": "Cy5", "Cy7": "Cy7",
    "Zombie UV": "Zombie UV", "Zombie Violet": "Zombie Violet", "Zombie Aqua": "Zombie Aqua",
    "Zombie Green": "Zombie Green", "Zombie Yellow": "Zombie Yellow", "Zombie Red": "Zombie Red",
    "Zombie NIR": "Zombie NIR",
    "LIVE/DEAD Blue": "LIVE/DEAD Fixable Blue", "LIVE/DEAD Aqua": "LIVE/DEAD Fixable Aqua",
    "LIVE/DEAD NIR": "LIVE/DEAD Fixable Near-IR", "eF780 Viability": "Fixable Viability Dye eFluor 780",
    "DAPI": "DAPI", "7-AAD": "7-AAD", "PI": "Propidium Iodide",
    "Spark Blue 550": "Spark Blue 550", "Spark Violet 538": "Spark Violet 538",
    "RealBlue 670": "RealBlue 670", "StarBright Yellow 665": "StarBright Yellow 665",
}
STEP = 2  # nm between stored points


def query(text):
    req = urllib.request.Request(URL, json.dumps({"query": text}).encode(),
                                 {"Content-Type": "application/json",
                                  "User-Agent": "sidescatter-spectra-fetch"})
    with urllib.request.urlopen(req, timeout=120) as r:
        out = json.load(r)
    if "errors" in out:
        raise RuntimeError(out["errors"])
    return out["data"]


def resample(points):
    """[[nm, value], ...] on a STEP nm grid, peak 1, values rounded to 3 decimals."""
    points = sorted(points)
    lo, hi = int(points[0][0]) // STEP * STEP + STEP, int(points[-1][0])
    out, j = [], 0
    for nm in range(lo, hi + 1, STEP):
        while j + 1 < len(points) and points[j + 1][0] < nm:
            j += 1
        (x0, y0), (x1, y1) = points[j], points[min(j + 1, len(points) - 1)]
        y = y0 if x1 == x0 else y0 + (y1 - y0) * (nm - x0) / (x1 - x0)
        out.append([nm, max(float(y), 0.0)])
    peak = max(v for _, v in out) or 1.0
    return [[nm, round(v / peak, 3)] for nm, v in out]


def main():
    listing = query("{ spectra { id category subtype owner { name } } }")["spectra"]
    ids = {}
    for s in listing:
        ids.setdefault(s["owner"]["name"], {})[s["subtype"]] = s["id"]
    dyes, missing = {}, []
    for name, fpbase in NAMES.items():
        found = ids.get(fpbase, {})
        ex_id = found.get("EX") or found.get("AB")   # absorption stands in for excitation
        em_id = found.get("EM")
        if not (ex_id and em_id):
            missing.append(name)
            continue
        ex = query(f"{{ spectrum(id: {ex_id}) {{ data }} }}")["spectrum"]["data"]
        em = query(f"{{ spectrum(id: {em_id}) {{ data }} }}")["spectrum"]["data"]
        dyes[name] = {"fpbase": fpbase, "ex": resample(ex), "em": resample(em)}
        print(f"{name}: {fpbase}", file=sys.stderr)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "source": "FPbase (https://www.fpbase.org), data free of copyright restrictions; "
                  "credit the original authors listed on each dye's FPbase page.",
        "step_nm": STEP, "dyes": dyes}, separators=(",", ":")))
    if missing:
        print("Not on FPbase with both spectra:", ", ".join(missing), file=sys.stderr)


if __name__ == "__main__":
    main()
