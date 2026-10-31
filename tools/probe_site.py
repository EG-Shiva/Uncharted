import collections
import json
import pathlib
import re
import urllib.request

data_dir = pathlib.Path(__file__).resolve().parent.parent / "data"
html = urllib.request.urlopen("https://spherical-codes.org/", timeout=60).read().decode("utf-8", "replace")
rows = re.findall(r"<tr>(.*?)</tr>", html, re.S)
entries = []
for row in rows:
    m = re.search(r'href="/data/(\d+)/(\d+)"', row)
    if not m:
        continue
    cells = re.findall(r"<td>(.*?)</td>", row, re.S)
    cosine = re.search(r"\d\.\d+", cells[2]) if len(cells) > 2 else None
    plain = re.sub(r"<[^>]+>", "", " ".join(cells))
    entries.append({
        "d": int(m.group(1)),
        "n": int(m.group(2)),
        "cosine": float(cosine.group()) if cosine else None,
        "needs_opt": "needs more optimization" in plain.lower(),
        "source": re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", cells[-1])).strip()[:160],
    })
(data_dir / "spherical_table.json").write_text(json.dumps(entries, indent=0))
needs = [e for e in entries if e["needs_opt"]]
print(len(entries), "entries,", len(needs), "need optimization")
print("by dim:", sorted(collections.Counter(e["d"] for e in needs).items()))
print("sources:", collections.Counter(e["source"][:70] for e in needs).most_common(8))
small = sorted(needs, key=lambda e: e["d"] * e["n"])[:15]
for e in small:
    print(e["d"], e["n"], e["cosine"], e["source"][:60])
