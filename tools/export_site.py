"""Bundle the web page plus selected runs into docs/ as a static site (GitHub Pages serves docs/ on main)."""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from discovery import site_data  # noqa: E402
from discovery.config import RUNS_DIR  # noqa: E402

OUT = ROOT / "docs"


def main():
    idx = site_data.index()
    names = sys.argv[1:] or [r["name"] for r in idx["runs"] if r["record_program"] is not None or r["name"] == "calib_cp26_v1"]
    if OUT.exists():
        shutil.rmtree(OUT)
    shutil.copytree(ROOT / "web", OUT)
    (OUT / ".nojekyll").write_text("")
    data = OUT / "data"
    data.mkdir()
    runs = site_data.index(names=names)["runs"]
    for r in runs:
        r.pop("cost", None)
    (data / "index.json").write_text(json.dumps({"runs": runs}))
    for name in names:
        src, dst = RUNS_DIR / name, data / name
        dst.mkdir()
        meta = json.loads((src / "run.json").read_text())
        meta.pop("budget", None)
        (dst / "run.json").write_text(json.dumps(meta, indent=2))
        events = [{k: v for k, v in e.items() if not k.startswith("cost")} for e in site_data.read_jsonl(src / "events.jsonl")]
        (dst / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
        progs = [{k: v for k, v in p.items() if k != "llm_cost"} for p in site_data.programs(src)]
        (dst / "programs.json").write_text(json.dumps(progs))
        shutil.copytree(src / "images", dst / "images")
    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 1e6
    print(f"exported {len(names)} runs to {OUT} ({size:.1f} MB): {', '.join(names)}")


if __name__ == "__main__":
    main()
