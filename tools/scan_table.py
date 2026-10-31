"""Find table entries beaten by trivially derived codes (drop points, or embed a lower-dimensional code)."""
import json
import pathlib

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"
entries = json.loads((DATA / "spherical_table.json").read_text())
cos = {(e["d"], e["n"]): e["cosine"] for e in entries if e["cosine"] is not None}

hits = []
for (d, n), c in cos.items():
    best_src = None
    for m in range(n + 1, n + 400):
        if (d, m) in cos and cos[(d, m)] < c - 1e-11:
            if best_src is None or cos[(d, m)] < best_src[1]:
                best_src = ((d, m), cos[(d, m)])
    for e in range(3, d):
        for m in range(n, n + 400):
            if (e, m) in cos and cos[(e, m)] < c - 1e-11:
                if best_src is None or cos[(e, m)] < best_src[1]:
                    best_src = ((e, m), cos[(e, m)])
    if best_src:
        hits.append(((d, n), c, best_src))

print(len(hits), "entries beaten by a derived code")
for h in sorted(hits, key=lambda h: h[1] - h[2][1], reverse=True)[:30]:
    print(h)
