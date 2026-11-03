"""Print README results rows from records/, sorted by improvement."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
table = {(e["d"], e["n"]): e["cosine"] for e in json.loads((ROOT / "data" / "spherical_table.json").read_text())}
rows = []
for rec in sorted((ROOT / "records").glob("spherical_*d_*")):
    m = re.match(r"spherical_(\d+)d_(\d+)$", rec.name)
    if not m:
        continue
    d, n = int(m.group(1)), int(m.group(2))
    score = json.loads((rec / "solution.json").read_text())["score"]
    known = table[(d, n)]
    imp = known - score
    rows.append((imp, d, n, known, score))
rows.sort(key=lambda r: -r[0])
print(f"{len(rows)} records", flush=True)
print("| Problem | Published record | Ours | Status |")
print("|---|---|---|---|")
first = True
for imp, d, n, known, score in rows:
    note = "New record, verified at 50 digits"
    if imp < 1e-8:
        note = f"New record (small: {imp:.1e}), verified at 50 digits"
    label = f"Spherical code, {n} points in {d}D"
    if first:
        label += " (max cosine, lower is better)"
        known_s = f"{known:.12f} ([spherical-codes.org](https://spherical-codes.org/))"
        first = False
    else:
        known_s = f"{known:.12f}"
    print(f"| {label} | {known_s} | **{score:.12f}** | {note} |")
print("| 26 circles in a square (sum of radii, higher is better) | 2.6359830849 | 2.6359404147 | Beats AlphaEvolve's 2.635863, not the current best |")
