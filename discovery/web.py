"""Local server for the Uncharted lab page. Reads runs/ live: python -m discovery.web"""
import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from . import site_data
from .config import ROOT, RUNS_DIR

WEB_DIR = ROOT / "web"


def safe_join(base: Path, rel: str) -> Path | None:
    path = (base / rel).resolve()
    return path if path.is_relative_to(base.resolve()) and path.is_file() else None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, body: bytes, ctype: str, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj):
        self._send(json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        if path == "/data/index.json":
            return self._json(site_data.index())
        if path == "/data/records.json":
            return self._json(site_data.records())
        if path == "/data/stats.json":
            return self._json(site_data.model_stats())
        if path == "/data/ablation.json":
            return self._json(site_data.ablation())
        if path.startswith("/data/"):
            run, _, rest = path[len("/data/"):].partition("/")
            run_dir = RUNS_DIR / run
            if rest == "programs.json" and run_dir.is_dir():
                return self._json(site_data.programs(run_dir))
            target = safe_join(RUNS_DIR, f"{run}/{rest}")
        else:
            target = safe_join(WEB_DIR, path.lstrip("/") or "index.html")
        if target is None:
            return self._send(b"not found", "text/plain", 404)
        ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if target.suffix in (".jsonl", ".md"):
            ctype = "text/plain; charset=utf-8"
        self._send(target.read_bytes(), ctype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Uncharted lab: http://127.0.0.1:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
