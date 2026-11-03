import time
import math
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _per_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def _adam(X, steps, b0, b1, lr0, lr1, t_lim=1e18):
    n, d = X.shape
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    m = np.zeros_like(X); v = np.zeros_like(X)
    b1a = 0.9; b2a = 0.999; eps = 1e-8
    lb0 = math.log(b0); lb1 = math.log(b1)
    ll0 = math.log(lr0); ll1 = math.log(lr1)
    best = X.copy(); best_m = _max_cos(X)
    for t in range(1, steps + 1):
        if (t & 63) == 0 and time.time() > t_lim:
            break
        frac = t / steps
        beta = math.exp(lb0 + (lb1 - lb0) * frac)
        lr = math.exp(ll0 + (ll1 - ll0) * frac)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        gmax = float(G.max())
        if gmax < best_m:
            best_m = gmax; best = X.copy()
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        Rs = W.sum(axis=1, keepdims=True) + 1e-30
        Grad = (W @ X) / Rs
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        m *= b1a; m += (1.0 - b1a) * Gt
        v *= b2a; v += (1.0 - b2a) * (Gt * Gt)
        bc1 = 1.0 - b1a ** t; bc2 = 1.0 - b2a ** t
        step = (m / bc1) / (np.sqrt(v / bc2) + eps)
        X = X - lr * step
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return best

def _refine_hole(X, v0, steps=120, beta=60.0, lr0=0.05, lr1=0.002):
    v = v0.copy() / np.linalg.norm(v0)
    ll0 = math.log(lr0); ll1 = math.log(lr1)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        lr = math.exp(ll0 + (ll1 - ll0) * frac)
        g = X @ v
        gm = float(g.max())
        w = np.exp(beta * (g - gm))
        w /= (w.sum() + 1e-30)
        grad = w @ X
        gt = grad - float(grad @ v) * v
        v = v - lr * gt
        nv = np.linalg.norm(v)
        if nv < 1e-12 or not np.isfinite(nv):
            return v0 / np.linalg.norm(v0)
        v /= nv
    return v

def _deepest_hole(X, rng, n_cand=6000, n_ref=8):
    d = X.shape[1]
    C = rng.randn(n_cand, d)
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    S = (X @ C.T).max(axis=0)
    idx = np.argsort(S)[:n_ref]
    best_v = None; best_s = 1e9
    for k in idx:
        v = _refine_hole(X, C[k])
        s = float((X @ v).max())
        if s < best_s:
            best_s = s; best_v = v.copy()
    return best_v

def _lp_polish(x, seconds, r0=1.1e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = r0
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
            kk = np.argsort(c[sel])[-2800:]
            I, J = I[kk], J[kk]
            cc = c[sel][kk]
        else:
            cc = c[sel]
            p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
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

def solve():
    t0 = time.time()
    BUDGET = 175.0
    n, d = 168, 8
    rng = np.random.RandomState(917421)
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
    glob_best = best.copy(); glob_val = best_val

    # three ladders: different N0 and removal masks
    ladders = [
        {"N0": 132, "mode": "random", "seed": 11},
        {"N0": 136, "mode": "mixed", "seed": 22},
        {"N0": 128, "mode": "worst", "seed": 33},
    ]

    for li, cfg in enumerate(ladders):
        # reserve time for final polish
        if time.time() > t0 + BUDGET - 28:
            break
        N0 = int(cfg["N0"]); mode = cfg["mode"]
        r2 = np.random.RandomState(cfg["seed"])
        # removal mask
        try:
            per = _per_max(incumbent)
            order = np.argsort(per)
        except Exception:
            order = np.arange(n)
        if mode == "random":
            keep = np.sort(r2.choice(n, N0, replace=False))
        elif mode == "mixed":
            nworst = (n - N0) // 2
            worst = np.argsort(per)[-nworst:] if (n - N0) > 0 else np.zeros(0, dtype=int)
            mask_w = np.ones(n, dtype=bool); mask_w[worst] = False
            rest = np.where(mask_w)[0]
            need = N0 - (len(rest) - (len(rest) - (N0 - 0)) if False else N0
            # simply: keep all non-worst minus random to reach N0
            # non-worst count = n - nworst > N0, drop random from non-worst
            cand = np.where(mask_w)[0]
            drop2 = r2.choice(len(cand), len(cand) - N0, replace=False) if len(cand) > N0 else []
            kmask = np.ones(len(cand), dtype=bool); kmask[drop2] = False
            keep = np.sort(cand[kmask])
        else:  # worst: drop worst points (keep best N0)
            keep = np.sort(order[:N0])
        X = incumbent[keep].copy()
        # initial slack optimisation at small N
        t_lim = t0 + BUDGET - 22
        try:
            X = _adam(X, steps=900, b0=20.0, b1=500.0, lr0=0.008, lr1=0.0006, t_lim=t_lim)
        except Exception:
            continue
        # regrow to 168
        while X.shape[0] < n:
            if time.time() > t0 + BUDGET - 24:
                break
            # batch size 1-2 (1 near top)
            cur = X.shape[0]
            batch = 1 if cur >= 160 else 2
            batch = min(batch, n - cur)
            for _ in range(batch):
                try:
                    v = _deepest_hole(X, r2, n_cand=5000, n_ref=6)
                except Exception:
                    v = r2.randn(d); v /= np.linalg.norm(v)
                X = np.vstack([X, v[None, :]])
            # tiny jitter to avoid exact symmetry lock
            try:
                X = X + r2.randn(*X.shape) * 2.5e-4
                X /= np.linalg.norm(X, axis=1, keepdims=True)
            except Exception:
                pass
            steps = 650 if X.shape[0] >= 158 else 480
            try:
                X = _adam(X, steps=steps, b0=25.0, b1=550.0, lr0=0.006, lr1=0.0004, t_lim=t0 + BUDGET - 22)
            except Exception:
                pass
            # occasional worst-point ejection to prevent greedy lock-in
            cur2 = X.shape[0]
            if cur2 in (144, 152, 160, 164) or (cur2 % 16 == 0 and cur2 < n):
                try:
                    pm = _per_max(X)
                    kw = int(3 if cur2 < 160 else 4)
                    worst_idx = np.argsort(pm)[-kw:]
                    keepm = np.ones(cur2, dtype=bool); keepm[worst_idx] = False
                    X = X[keepm]
                    for _ in range(kw):
                        v = _deepest_hole(X, r2, n_cand=5000, n_ref=6)
                        X = np.vstack([X, v[None, :]])
                    X /= np.linalg.norm(X, axis=1, keepdims=True)
                    X = _adam(X, steps=400, b0=25.0, b1=500.0, lr0=0.005, lr1=0.0004, t_lim=t0 + BUDGET - 22)
                except Exception:
                    pass
        if X.shape[0] != n:
            continue
        try:
            val = _max_cos(X)
        except Exception:
            continue
        if val < glob_val and np.all(np.isfinite(X)):
            glob_val = val; glob_best = X.copy()

    # extra jamming on global best (short) before final LP
    if time.time() < t0 + BUDGET - 20:
        try:
            Xq = _adam(glob_best, steps=800, b0=30.0, b1=600.0, lr0=0.005, lr1=0.00035, t_lim=t0 + BUDGET - 18)
            vq = _max_cos(Xq)
            if vq < glob_val:
                glob_val = vq; glob_best = Xq.copy()
        except Exception:
            pass
    # final long LP polish only at N=168
    remain = (t0 + BUDGET - 3) - time.time()
    if remain > 5:
        try:
            Xlp = _lp_polish(glob_best, seconds=min(remain, 20.0))
            vlp = _max_cos(Xlp)
            if vlp < glob_val and np.all(np.isfinite(Xlp)):
                glob_val = vlp; glob_best = Xlp.copy()
        except Exception:
            pass
    glob_best = glob_best / np.linalg.norm(glob_best, axis=1, keepdims=True)
    try:
        if _max_cos(glob_best) > inc_val:
            glob_best = incumbent.copy()
    except Exception:
        glob_best = incumbent.copy()
    return glob_best
