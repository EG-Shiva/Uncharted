"""Independent high-precision check of a spherical code: max pairwise cosine after exact normalisation."""
import json
import sys

import numpy as np
from mpmath import mp, mpf, sqrt

mp.dps = 50


def verify(x):
    rows = [[mpf(float(v)) for v in r] for r in x]
    rows = [[v / sqrt(sum(w * w for w in r)) for v in r] for r in rows]
    n = len(rows)
    worst, pair = mpf(-2), None
    for i in range(n):
        for j in range(i + 1, n):
            c = sum(a * b for a, b in zip(rows[i], rows[j]))
            if c > worst:
                worst, pair = c, (i, j)
    return worst, pair


if __name__ == "__main__":
    path = sys.argv[1]
    x = np.load(path) if path.endswith(".npy") else np.loadtxt(path, delimiter=",")
    record = float(sys.argv[2]) if len(sys.argv) > 2 else None
    worst, pair = verify(x)
    print("shape", x.shape, "max cosine (50 digits):", mp.nstr(worst, 20), "at pair", pair)
    if record is not None:
        print("improvement over record:", mp.nstr(mpf(record) - worst, 6))
    print("min angle (deg):", mp.nstr(mp.degrees(mp.acos(worst)), 15))
