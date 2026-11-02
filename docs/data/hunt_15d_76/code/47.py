import time
import numpy as np

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def solve():
    t0 = time.time()
    LIM = 158.0
    rng = np.random.default_rng(42)
    n, d = 76, 15
    try:
        Xb = np.load("best.npy").astype(np.float64)
    except Exception:
        Xb = None
    try:
        Xr = np.load("record.npy").astype(np.float64)
    except Exception:
        Xr = None
    cands = []
    for c in (Xb, Xr):
        if c is not None and c.shape == (76, 15):
            c = np.asarray(c, dtype=np.float64)
            c = _norm(c)
            if np.all(np.isfinite(c)):
                cands.append(c)
    if not cands:
        cands.append(_norm(rng.standard_normal((n, d))))
    vals = [_maxcos(c) for c in cands]
    Xbest = cands[int(np.argmin(vals))].copy()
    best = float(np.min(vals))

    def anneal(X, steps, T0, T1, lr0, lr1):
        # normalized soft-max tangential descent
        lT0, lT1 = np.log(T0), np.log(T1)
        ll0, ll1 = np.log(lr0), np.log(lr1)
        for k in range(steps):
            if (k & 63) == 0 and time.time() - t0 > LIM:
                break
            f = k / max(1, steps - 1)
            T = np.exp(lT0 + (lT1 - lT0) * f)
            lr = np.exp(ll0 + (ll1 - ll0) * f)
            G = X @ X.T
            np.fill_diagonal(G, -10.0)
            m = float(np.max(G))
            E = np.exp((G - m) / T)
            np.fill_diagonal(E, 0.0)
            S = float(np.sum(E))
            if not np.isfinite(S) or S <= 0:
                continue
            Gr = (E @ X) / S
            dot = np.sum(Gr * X, axis=1, keepdims=True)
            Gt = Gr - dot * X
            rms = float(np.sqrt(np.sum(Gt * Gt) / n)) + 1e-12
            if rms < 1e-15:
                continue
            Gt /= rms
            X = X - lr * Gt
            X = _norm(X)
        return X

    def refine_hole(v, Y):
        v = v / (np.linalg.norm(v) + 1e-12)
        # ~120 steps soft-max hole descent
        for k in range(120):
            f = k / 119.0
            Tk = 0.008 * (0.15 ** f)
            lrk = 0.10 * (0.15 ** f)
            s = Y @ v
            m = float(np.max(s))
            w = np.exp((s - m) / Tk)
            sw = float(np.sum(w))
            if not np.isfinite(sw) or sw <= 0:
                break
            w /= sw
            g = w @ Y
            g = g - float(np.dot(g, v)) * v
            ng = float(np.linalg.norm(g))
            if ng < 1e-14:
                break
            # scale to avoid overshoot
            g = g / (ng + 1e-12) * min(1.0, ng * 4.0)
            v = v - lrk * g
            nv = float(np.linalg.norm(v))
            if not np.isfinite(nv) or nv == 0:
                break
            v = v / nv
        return v

    def largest_hole(Y, ns=3000):
        C = _norm(rng.standard_normal((ns, d)))
        scores = np.max(C @ Y.T, axis=1)
        bi = int(np.argmin(scores))
        return refine_hole(C[bi], Y)

    # short initial squeeze (cheap)
    Xbest = anneal(Xbest.copy(), 600, 0.004, 0.0008, 0.6, 0.08)
    v = _maxcos(Xbest)
    if v < best:
        best = v

    sigmas = [0.10, 0.18, 0.07, 0.14, 0.25, 0.05, 0.12, 0.20]
    trial = 0
    # main basin-hopping
    while time.time() - t0 < LIM - 8:
        trial += 1
        sigma = sigmas[(trial - 1) % len(sigmas)]
        # occasional pure random restart to find new basin
        if trial % 9 == 0 and time.time() - t0 < LIM - 40:
            Xt = _norm(rng.standard_normal((n, d)))
            Xt = anneal(Xt, 1800, 0.020, 0.0012, 5.0, 0.12) if False else anneal(Xt, 1800, 0.02, 0.0012, 0.05, 0.004)
            # anneal above uses normalized steps; use larger lr for random start
            v = _maxcos(Xt)
            if v < best:
                best = v
                Xbest = Xt.copy()
                # extra polish
                if time.time() - t0 < LIM - 12:
                    Xp = anneal(Xbest.copy(), 1200, 0.002, 0.0003, 0.008, 0.001)
                    vp = _maxcos(Xp)
                    if vp < best:
                        best = vp
                        Xbest = Xp
            continue

        Xt = Xbest.copy()
        # contact counts to find overloaded core
        try:
            G = Xt @ Xt.T
            np.fill_diagonal(G, -2.0)
            m0 = float(np.max(G))
            cnt = np.sum(G >= m0 - 2e-3, axis=1)
            rowmax = np.max(G, axis=1)
            order = np.argsort(-rowmax)
        except Exception:
            cnt = np.zeros(n)
            order = rng.permutation(n)

        k_re = 2 + (trial % 4)  # 2..5
        idxs = []
        # take 1-2 overloaded
        idxs.append(int(order[(trial - 1) % n]))
        if k_re >= 3:
            idxs.append(int(order[(trial * 7 + 3) % n]))
        while len(idxs) < k_re:
            r = int(rng.integers(0, n))
            if r not in idxs:
                idxs.append(r)

        for idx in idxs:
            if time.time() - t0 > LIM - 8:
                break
            mask = np.ones(n, dtype=bool)
            mask[idx] = False
            Y = Xt[mask]
            try:
                Xt[idx] = largest_hole(Y, ns=3000)
            except Exception:
                vv = rng.standard_normal(d)
                Xt[idx] = vv / np.linalg.norm(vv)

        # gaussian shake on all points
        Xt = Xt + sigma * rng.standard_normal((n, d))
        Xt = _norm(Xt)

        # anneal: two stages (hot then cold) in one call
        if sigma > 0.13:
            Xt = anneal(Xt, 1600, 0.014, 0.0009, 0.030, 0.0025)
        else:
            Xt = anneal(Xt, 1400, 0.010, 0.0007, 0.022, 0.0020)
        if time.time() - t0 > LIM - 6:
            v = _maxcos(Xt)
            if v < best:
                best = v
                Xbest = Xt.copy()
            break
        v = _maxcos(Xt)
        if v < best - 1e-12:
            best = v
            Xbest = Xt.copy()
            # greedy extra polish on improvement
            if time.time() - t0 < LIM - 12:
                Xp = anneal(Xbest.copy(), 1000, 0.0015, 0.00025, 0.006, 0.0008)
                vp = _maxcos(Xp)
                if vp < best:
                    best = vp
                    Xbest = Xp

    Xbest = _norm(np.nan_to_num(Xbest))
    return Xbest
