import time
import math
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _build_X(seeds, freqs, m):
    ns = seeds.shape[0]
    angs = 2.0 * np.pi * (np.arange(m)[:, None] * freqs[None, :]) / float(m)
    C = np.cos(angs)
    S = np.sin(angs)
    Xr = np.empty((ns, m, 8), dtype=np.float64)
    for p in range(4):
        s0 = seeds[:, 2 * p]
        s1 = seeds[:, 2 * p + 1]
        c = C[:, p]
        s = S[:, p]
        Xr[:, :, 2 * p] = s0[:, None] * c[None, :] - s1[:, None] * s[None, :]
        Xr[:, :, 2 * p + 1] = s0[:, None] * s[None, :] + s1[:, None] * c[None, :]
    return Xr.reshape((ns * m, 8))

def _seed_quench(seeds0, freqs, m, steps, beta0, beta1, lr0, lr1):
    ns = seeds0.shape[0]
    seeds = seeds0.copy()
    seeds /= np.linalg.norm(seeds, axis=1, keepdims=True)
    angs = 2.0 * np.pi * (np.arange(m)[:, None] * freqs[None, :]) / float(m)
    C = np.cos(angs)
    S = np.sin(angs)
    mad = np.zeros_like(seeds)
    vad = np.zeros_like(seeds)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    lb0 = np.log(beta0); lb1 = np.log(beta1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    n = ns * m
    best_val = 1e9
    best_seeds = seeds.copy()
    for t in range(1, steps + 1):
        frac = t / steps
        beta = np.exp(lb0 + (lb1 - lb0) * frac)
        lr = np.exp(ll0 + (ll1 - ll0) * frac)
        Xr = np.empty((ns, m, 8), dtype=np.float64)
        for p in range(4):
            s0 = seeds[:, 2 * p]
            s1 = seeds[:, 2 * p + 1]
            c = C[:, p]; s = S[:, p]
            Xr[:, :, 2 * p] = s0[:, None] * c[None, :] - s1[:, None] * s[None, :]
            Xr[:, :, 2 * p + 1] = s0[:, None] * s[None, :] + s1[:, None] * c[None, :]
        X = Xr.reshape((n, 8))
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        gmax = float(G.max())
        if gmax < best_val:
            best_val = gmax
            best_seeds = seeds.copy()
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        Rs = W.sum(axis=1, keepdims=True) + 1e-18
        GradX = (W @ X) / Rs
        dotx = (GradX * X).sum(axis=1, keepdims=True)
        GtX = GradX - dotx * X
        Gtr = GtX.reshape((ns, m, 8))
        Gs = np.empty_like(seeds)
        for p in range(4):
            g0 = Gtr[:, :, 2 * p]
            g1 = Gtr[:, :, 2 * p + 1]
            c = C[:, p]; s = S[:, p]
            Gs[:, 2 * p] = ((c[None, :] * g0 + s[None, :] * g1).sum(axis=1)) / float(m)
            Gs[:, 2 * p + 1] = ((-s[None, :] * g0 + c[None, :] * g1).sum(axis=1)) / float(m)
        dots = (Gs * seeds).sum(axis=1, keepdims=True)
        Gt = Gs - dots * seeds
        mad *= b1; mad += (1.0 - b1) * Gt
        vad *= b2; vad += (1.0 - b2) * (Gt * Gt)
        bc1 = 1.0 - b1 ** t; bc2 = 1.0 - b2 ** t
        step = (mad / bc1) / (np.sqrt(vad / bc2) + eps)
        seeds = seeds - lr * step
        seeds /= np.linalg.norm(seeds, axis=1, keepdims=True)
    return best_seeds, best_val

def _adam_full(X, steps, beta0, beta1, lr0, lr1, t_lim=1e18):
    n, d = X.shape
    m = np.zeros_like(X); v = np.zeros_like(X)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    lb0 = np.log(beta0); lb1 = np.log(beta1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    best = X.copy(); best_m = _max_cos(X)
    for t in range(1, steps + 1):
        if (t & 63) == 0 and time.time() > t_lim:
            break
        frac = t / steps
        beta = np.exp(lb0 + (lb1 - lb0) * frac)
        lr = np.exp(ll0 + (ll1 - ll0) * frac)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        gmax = float(G.max())
        if gmax < best_m:
            best_m = gmax; best = X.copy()
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        Rs = W.sum(axis=1, keepdims=True) + 1e-18
        Grad = (W @ X) / Rs
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        m *= b1; m += (1.0 - b1) * Gt
        v *= b2; v += (1.0 - b2) * (Gt * Gt)
        bc1 = 1.0 - b1 ** t; bc2 = 1.0 - b2 ** t
        step = (m / bc1) / (np.sqrt(v / bc2) + eps)
        X = X - lr * step
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return best

def _hard_polish(X, steps=250, lr0=0.0032, lr1=0.00012):
    n, d = X.shape
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        tol = 3e-4 + (2e-6 - 3e-4) * frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        mm = float(G.max())
        M = (G > (mm - tol)).astype(np.float64)
        np.fill_diagonal(M, 0.0)
        cnt = M.sum(axis=1, keepdims=True) + 1e-12
        Grad = (M @ X) / cnt
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        nn = np.sqrt((Gt * Gt).sum(axis=1, keepdims=True)) + 1e-12
        active = (cnt[:, 0] > 0.5)
        step = np.zeros_like(Gt)
        step[active] = Gt[active] / nn[active]
        X = X - lr * step
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        mm = float(c.max())
        sel = c > mm - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > 2800:
            idx = np.argsort(c[sel])[-2800:]
            I, J = I[idx], J[idx]
            cc = c[sel][idx]
        else:
            cc = c[sel]
            p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cc, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        vv = _max_cos(y)
        if vv < best:
            x, best = y, vv
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _rand_freqs(rng, m):
    f = np.empty(4, dtype=np.int64)
    for p in range(4):
        for _ in range(100):
            a = int(rng.randint(1, m))
            if math.gcd(a, m) == 1:
                f[p] = a
                break
        else:
            f[p] = 1
    return f

def solve():
    t0 = time.time()
    BUDGET = 175.0
    n, d = 168, 8
    rng = np.random.RandomState(20261103)
    best = None; best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape != (n, d):
                continue
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            if not np.all(np.isfinite(a)):
                continue
            va = _max_cos(a)
            if va < best_val:
                best_val = va; best = a.copy()
        except Exception:
            pass
    if best is None:
        best = rng.randn(n, d)
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    incumbent = best.copy(); inc_val = best_val

    ms = [84, 56, 42, 28, 24, 21]
    candidates = []
    # Phase 1: short races over symmetry codes
    t_race_end = t0 + 70.0
    races_per_m = 12
    for m in ms:
        ns = n // m
        for r in range(races_per_m):
            if time.time() > t_race_end or time.time() > t0 + BUDGET - 80:
                break
            freqs = _rand_freqs(rng, m)
            seeds0 = rng.randn(ns, d)
            seeds0 /= np.linalg.norm(seeds0, axis=1, keepdims=True)
            try:
                bseeds, bval = _seed_quench(seeds0, freqs, m, steps=320,
                                            beta0=30.0, beta1=600.0,
                                            lr0=0.012, lr1=0.0012)
            except Exception:
                continue
            Xc = _build_X(bseeds, freqs, m)
            Xc /= np.linalg.norm(Xc, axis=1, keepdims=True)
            vc = _max_cos(Xc)
            candidates.append((vc, bseeds, freqs.copy(), m))
            if vc < best_val - 1e-12:
                best_val = vc; best = Xc.copy()
        if time.time() > t_race_end or time.time() > t0 + BUDGET - 80:
            break
    try:
        candidates.sort(key=lambda e: e[0])
    except Exception:
        pass
    topk = candidates[:6] if len(candidates) >= 6 else candidates
    # Phase 2: full seed quench + symmetry breaking + full quench
    for (vc, bseeds, freqs, m) in list(topk):
        if time.time() > t0 + BUDGET - 45:
            break
        ns = n // m
        try:
            eseeds, eval_ = _seed_quench(bseeds, freqs, m, steps=900,
                                         beta0=30.0, beta1=600.0,
                                         lr0=0.008, lr1=0.0008)
        except Exception:
            continue
        Xsym = _build_X(eseeds, freqs, m)
        Xsym /= np.linalg.norm(Xsym, axis=1, keepdims=True)
        vsym = _max_cos(Xsym)
        if vsym < best_val:
            best_val = vsym; best = Xsym.copy()
        for sigma in (1e-3, 1e-2):
            if time.time() > t0 + BUDGET - 40:
                break
            Xb = Xsym + rng.randn(n, d) * sigma
            Xb /= np.linalg.norm(Xb, axis=1, keepdims=True)
            t_lim = t0 + BUDGET - 25
            try:
                Xq = _adam_full(Xb, steps=900, beta0=30.0, beta1=550.0,
                                lr0=0.006, lr1=0.0004, t_lim=t_lim)
                Xq = _hard_polish(Xq, steps=220, lr0=0.0030, lr1=0.00012)
            except Exception:
                continue
            vq = _max_cos(Xq)
            if vq < best_val:
                best_val = vq; best = Xq.copy()
    # Phase 3: polish current best with short full quench from small noise (diversify basin tail)
    if time.time() < t0 + BUDGET - 30:
        for sigma in (0.004, 0.012):
            if time.time() > t0 + BUDGET - 28:
                break
            Xb = incumbent + rng.randn(n, d) * sigma
            Xb /= np.linalg.norm(Xb, axis=1, keepdims=True)
            try:
                Xq = _adam_full(Xb, steps=700, beta0=25.0, beta1=500.0,
                                lr0=0.006, lr1=0.0004, t_lim=t0 + BUDGET - 22)
                Xq = _hard_polish(Xq, steps=200, lr0=0.0030, lr1=0.00012)
            except Exception:
                continue
            vq = _max_cos(Xq)
            if vq < best_val:
                best_val = vq; best = Xq.copy()
    # Final LP squeeze on best
    remain = (t0 + BUDGET - 4) - time.time()
    if remain > 4:
        try:
            Xlp = _lp_polish(best, seconds=min(remain, 18.0))
            vlp = _max_cos(Xlp)
            if vlp < best_val and np.all(np.isfinite(Xlp)):
                best_val = vlp; best = Xlp.copy()
        except Exception:
            pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    # safety: never return worse than incumbent
    try:
        if _max_cos(best) > inc_val and np.all(np.isfinite(incumbent)):
            best = incumbent.copy()
    except Exception:
        pass
    return best
