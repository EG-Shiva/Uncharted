import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HARNESS = r'''
import importlib.util, sys, numpy as np
spec = importlib.util.spec_from_file_location("candidate", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
out = np.asarray(mod.solve(), dtype=float)
np.save(sys.argv[2], out, allow_pickle=False)
'''


@dataclass
class RunResult:
    ok: bool
    output: np.ndarray | None
    error: str
    seconds: float


def _child_env() -> dict:
    env = {k: v for k, v in os.environ.items() if "KEY" not in k.upper() and "TOKEN" not in k.upper() and "SECRET" not in k.upper()}
    env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "PYTHONHASHSEED": "0"})
    return env


def run_candidate(code: str, timeout: float) -> RunResult:
    """Run untrusted candidate code in a separate process with no API credentials and a hard timeout."""
    with tempfile.TemporaryDirectory(prefix="cand_") as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "harness.py").write_text(HARNESS)
        (tmp_path / "candidate.py").write_text(code, encoding="utf-8")
        out_file = tmp_path / "out.npy"
        start = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, "harness.py", "candidate.py", "out.npy"],
                cwd=tmp, env=_child_env(), capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return RunResult(False, None, f"Timed out after {timeout:.0f}s", time.time() - start)
        elapsed = time.time() - start
        if proc.returncode != 0 or not out_file.exists():
            err = (proc.stderr or proc.stdout or "no output")[-2000:]
            return RunResult(False, None, err, elapsed)
        try:
            return RunResult(True, np.load(out_file, allow_pickle=False), "", elapsed)
        except Exception as e:
            return RunResult(False, None, f"Could not read output: {e}", elapsed)
