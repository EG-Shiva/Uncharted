"""Stop all sweeps once total API spend across runs/ reaches a cap.

  python tools/spend_guard.py --cap 148
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def total_spend():
    total = 0.0
    for c in (ROOT / "runs").glob("*/cost.json"):
        try:
            total += json.loads(c.read_text())["total"]
        except (OSError, ValueError, KeyError):
            pass
    return total


def stop_sweeps():
    ps = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
          "Where-Object { $_.CommandLine -match 'sweep.py|discovery.run' } | "
          "ForEach-Object { Stop-Process -Id $_.ProcessId -Force }")
    subprocess.run(["powershell", "-NoProfile", "-Command", ps])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=float, default=148.0)
    ap.add_argument("--every", type=int, default=60)
    args = ap.parse_args()
    while True:
        spent = total_spend()
        print(f"{time.strftime('%H:%M:%S')} spent ${spent:.2f} / ${args.cap:.2f}", flush=True)
        if spent >= args.cap:
            stop_sweeps()
            print("cap reached, sweeps stopped", flush=True)
            return
        time.sleep(args.every)


if __name__ == "__main__":
    main()
