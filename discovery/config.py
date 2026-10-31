import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"

load_dotenv(ROOT / ".env")

API_BASE_URL = "https://api.meta.ai/v1"
API_KEY_ENV = "MODEL_API_KEY"

WORKER_MODEL = os.environ.get("WORKER_MODEL", "muse-spark-1.3-contributor")
LEAD_MODEL = os.environ.get("LEAD_MODEL", "muse-wonder-1.0")

# USD per 1M tokens: (input, cached input, output). Source: dev.meta.ai/docs/pricing-rate-limits
PRICES = {
    "muse-wonder-1.0": (4.00, 0.20, 20.00),
    "muse-spark-1.3": (1.25, 0.15, 4.25),
    "muse-spark-1.2": (1.25, 0.15, 4.25),
    "muse-spark-1.3-contributor": (0.10, 0.002, 0.20),
    "muse-spark-1.2-contributor": (0.10, 0.002, 0.20),
}


def api_key() -> str:
    key = os.environ.get(API_KEY_ENV, "")
    if not key or key == "paste-your-key-here":
        raise RuntimeError(f"Set {API_KEY_ENV} in {ROOT / '.env'}")
    return key
