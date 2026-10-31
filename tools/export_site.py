"""Bundle the web page plus selected runs into docs/ as a static site (GitHub Pages serves docs/ on main)."""
import json
import shutil
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from discovery import site_data  # noqa: E402
from discovery.config import RUNS_DIR  # noqa: E402

OUT = ROOT / "docs"


def save_jpeg(src: Path, dst: Path, width: int, quality: int):
    with Image.open(src) as im:
        im = im.convert("RGB")
        if im.width > width:
            im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
        im.save(dst, "JPEG", quality=quality, optimize=True)


def main():
    idx = site_data.index()
    names = sys.argv[1:] or [r["name"] for r in idx["runs"] if r["record_program"] is not None or r["name"] == "calib_cp26_v1"]
    OUT.mkdir(exist_ok=True)
    for child in OUT.iterdir():
        shutil.rmtree(child) if child.is_dir() else child.unlink()
    shutil.copytree(ROOT / "web", OUT, dirs_exist_ok=True)
    (OUT / ".nojekyll").write_text("")
    data = OUT / "data"
    data.mkdir()
    runs = site_data.index(names=names)["runs"]
    for r in runs:
        r.pop("cost", None)
    (data / "index.json").write_text(json.dumps({"runs": runs}))
    (data / "records.json").write_text(json.dumps(site_data.records()))
    (data / "stats.json").write_text(json.dumps(site_data.model_stats()))
    (data / "ablation.json").write_text(json.dumps(site_data.ablation()))
    for name in names:
        src, dst = RUNS_DIR / name, data / name
        for sub in ("images", "thumbs", "code"):
            (dst / sub).mkdir(parents=True)
        meta = json.loads((src / "run.json").read_text())
        meta.pop("budget", None)
        (dst / "run.json").write_text(json.dumps(meta, indent=2))
        events = [{k: v for k, v in e.items() if not k.startswith("cost")} for e in site_data.read_jsonl(src / "events.jsonl")]
        (dst / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
        progs = []
        for p in site_data.programs(src):
            p.pop("llm_cost", None)
            (dst / "code" / f"{p['id']}.py").write_text(p.pop("code") or "", encoding="utf-8")
            if p.get("image") and (src / p["image"]).exists():
                stem = Path(p["image"]).stem
                save_jpeg(src / p["image"], dst / "images" / f"{stem}.jpg", 1100, 82)
                save_jpeg(src / p["image"], dst / "thumbs" / f"{stem}.jpg", 260, 70)
                p["image"], p["thumb"] = f"images/{stem}.jpg", f"thumbs/{stem}.jpg"
            progs.append(p)
        (dst / "programs.json").write_text(json.dumps(progs))
    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 1e6
    print(f"exported {len(names)} runs to {OUT} ({size:.1f} MB): {', '.join(names)}")


if __name__ == "__main__":
    main()
