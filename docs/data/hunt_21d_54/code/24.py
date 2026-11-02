import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _quench(y, steps, T0, T1, lr0, lr1):
    # normalized soft-max tangential descent, fast matrix ops only
    n, d = y.shape
    for s in range(steps):
        frac = s / max(1, steps - 1)
        T = T0 + (T1 - T0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        G = y @ y.T
        np.fill_diagonal(G, -1.0)
        m = float(G.max())
        E = (G - m) / max(T, 1e-12)
        np.clip(E, -15.0, 0.0, out=E)
        W = np.exp(E)
        np.fill_diagonal(W, 0.0)
        grad = W @ y
        dot = np.einsum('ij,ij->i', grad, y)[:, None]
        grad -= dot * y
        nr = np.linalg.norm(grad, axis=1, keepdims=True) + 1e-12
        grad /= nr
        y = y - lr * grad
        # renormalize
        nn = np.linalg.norm(y, axis=1, keepdims=True)
        nn[nn == 0] = 1.0
        y /= nn
    return y

def _squeeze(y, steps, window0=5e-3, window1=5e-4, lr0=5e-3, lr1=5e-4):
    n, d = y.shape
    for s in range(steps):
        frac = s / max(1, steps - 1)
        window = window0 + (window1 - window0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        G = y @ y.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        thr = m - window
        W = G - thr
        W[W < 0] = 0.0
        np.fill_diagonal(W, 0.0)
        if np.count_nonzero(W) == 0:
            continue
        grad = W @ y
        dot = np.einsum('ij,ij->i', grad, y)[:, None]
        grad -= dot * y
        nr = np.linalg.norm(grad, axis=1, keepdims=True) + 1e-12
        grad /= nr
        y = y - lr * grad
        nn = np.linalg.norm(y, axis=1, keepdims=True)
        nn[nn == 0] = 1.0
        y /= nn
    return y

def _lp_polish(x, seconds, r0=1.5e-3, cap=800):
    x = _norm(np.asarray(x, dtype=float))
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        remain = seconds - (time.time() - start)
        if remain < 0.3:
            break
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > cap:
            # keep largest
            part = np.argpartition(-c, cap)[:cap]
            # build mask from top-cap among all? simpler: top cap overall
            # c argsort top cap
            idx = np.argsort(-c)[:cap]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        csv = c[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        Ccol = np.empty((p, 2 * d + 1), dtype=int)
        Ccol[:, :d] = I[:, None] * d + np.arange(d)
        Ccol[:, d:2 * d] = J[:, None] * d + np.arange(d)
        Ccol[:, 2 * d] = n * d
        cols = Ccol.ravel()
        Arow = np.empty((p, 2 * d + 1))
        Arow[:, :d] = x[J]
        Arow[:, d:2 * d] = x[I]
        Arow[:, 2 * d] = -1.0
        vals = Arow.ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csv, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": min(2.0, max(0.3, remain - 0.1))})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    LIMIT = 168.0
    rng = np.random.RandomState(7)
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.randn(54, 21)
    xb = _norm(np.asarray(xb, dtype=float))
    if xb.shape != (54, 21) or not np.all(np.isfinite(xb)):
        xb = _norm(rng.randn(54, 21))
    try:
        xr = _norm(np.asarray(np.load("record.npy"), dtype=float))
        if xr.shape == (54, 21) and np.all(np.isfinite(xr)):
            if _max_cos(xr) < _max_cos(xb):
                xb = xr
    except Exception:
        pass
    best = xb.copy()
    best_val = _max_cos(best)
    n, d = 54, 21

    def tl():
        return LIMIT - (time.time() - t0)

    # Phase 1: cheap basin hopping, no LP, reserve ~30s for final
    sigmas = [0.05, 0.02, 0.08, 0.012, 0.035, 0.10, 0.015, 0.06]
    trial = 0
    # precompute hole sample pool reuse? generate on fly
    while tl() > 32.0:
        trial += 1
        if tl() < 33.0:
            break
        sigma = sigmas[(trial - 1) % len(sigmas)]
        mode = trial % 4
        cand = best.copy()
        if mode == 1:
            # global jitter + quench
            cand = cand + sigma * rng.randn(n, d)
            cand = _norm(cand)
            cand = _quench(cand, steps=700, T0=0.015, T1=0.002, lr0=0.035, lr1=0.004)
        elif mode == 2:
            # worst-point big jitter
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            rm = G.max(axis=1)
            iw = int(np.argmax(rm))
            cand[iw] = cand[iw] + (sigma * 2.5) * rng.randn(d)
            cand = _norm(cand)
            cand = cand + (sigma * 0.2) * rng.randn(n, d)
            cand = _norm(cand)
            cand = _quench(cand, steps=600, T0=0.012, T1=0.0015, lr0=0.03, lr1=0.003)
        elif mode == 3:
            # kill worst -> best hole
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            rm = G.max(axis=1)
            order = np.argsort(-rm)
            iw = int(order[0] if (trial % 8 != 0) else order[int(rng.randint(0, 5))])
            keep = np.delete(best, iw, axis=0)
            K = 2500
            S = _norm(rng.randn(K, d))
            dots = S @ keep.T
            md = dots.max(axis=1)
            hole = S[int(np.argmin(md))]
            cand[iw] = hole + 0.005 * rng.randn(d)
            cand = _norm(cand)
            cand = _quench(cand, steps=600, T0=0.008, T1=0.0012, lr0=0.025, lr1=0.003)
        else:
            # subset shake 6 points
            idx = rng.choice(n, size=6, replace=False)
            cand[idx] = cand[idx] + sigma * 1.5 * rng.randn(6, d)
            cand = _norm(cand)
            cand = _quench(cand, steps=800, T0=0.018, T1=0.002, lr0=0.04, lr1=0.004)
        # light hinge squeeze
        if tl() < 32.5:
            break
        cand = _squeeze(cand, steps=120, window0=4e-3, window1=5e-4, lr0=3e-3, lr1=5e-4)
        v = _max_cos(cand)
        if v < best_val - 1e-12:
            best_val = v
            best = cand.copy()
        # safety: abort if too close to deadline
        if trial > 400:
            break

    # Phase 2: intensify best with longer quench + squeeze
    if tl() > 28.0:
        c2 = _quench(best.copy(), steps=1500, T0=0.006, T1=0.0008, lr0=0.02, lr1=0.002)
        c2 = _squeeze(c2, steps=300, window0=3e-3, window1=2e-4, lr0=2e-3, lr1=2e-4)
        v2 = _max_cos(c2)
        if v2 < best_val:
            best_val = v2
            best = c2

    # Phase 3: single short LP polish
    rem = tl()
    if rem > 3.0:
        budget = min(rem - 1.0, 24.0)
        best = _lp_polish(best, seconds=budget, r0=1.2e-3, cap=800)
        best_val = _max_cos(best)

    best = _norm(best)
    return best
