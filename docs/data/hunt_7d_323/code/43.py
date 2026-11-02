import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _smooth_anneal(x, steps, T0, T1, e0, e1, rng=None):
    # vectorized soft-max tangential descent, max step e per iter
    n, d = x.shape
    # log schedule for T, linear for eta
    lT0, lT1 = np.log(T0), np.log(T1)
    for s in range(steps):
        frac = s / max(1, steps - 1)
        T = float(np.exp(lT0 + (lT1 - lT0) * frac))
        eta = float(e0 + (e1 - e0) * frac)
        G = x @ x.T
        # current max (off diag) for shift
        # cheap max
        m = float(np.max(G - 2.0 * np.eye(n, dtype=G.dtype)))
        # weights
        # clip exponent for stability
        E = (G - m) / T
        # avoid overflow: E<=0, exp ok; very negative -> 0
        E[E < -30.0] = -30.0
        np.fill_diagonal(E, -30.0)
        W = np.exp(E)
        # tangential gradient of LogSumExp
        WX = W @ x  # n x d
        srow = np.sum(W * G, axis=1, keepdims=True)  # n x 1
        Tgrad = WX - srow * x
        mn = float(np.sqrt((Tgrad * Tgrad).sum(axis=1).max()))
        if not np.isfinite(mn) or mn < 1e-18:
            continue
        step = eta / mn
        # optional cap to avoid huge jumps
        if step > 1.0:
            step = 1.0
        y = x - step * Tgrad
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        x = y
    return x


def _greedy_cap_reinsert(y, idx, rng, n_cand=7000):
    n, d = y.shape
    in_set = np.zeros(n, dtype=bool)
    in_set[idx] = True
    X_fixed_base = y[~in_set]  # (n-K) x d
    new_pts = []
    order = idx.copy()
    rng.shuffle(order)
    # pre-generate candidates in chunks to save time? one batch per point
    for k in order:
        cand = rng.standard_normal(size=(n_cand, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        if new_pts:
            Xa = np.vstack([X_fixed_base] + new_pts)
        else:
            Xa = X_fixed_base
        # worst cosine to existing set
        dots = cand @ Xa.T
        worst = dots.max(axis=1)
        bi = int(np.argmin(worst))
        bp = cand[bi]
        y[k] = bp
        new_pts.append(bp[None, :])
    return y


def _lp_polish_fast(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 5e-3
    start = time.time()
    iuI, iuJ = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start >= seconds:
            break
        G = x @ x.T
        c = G[iuI, iuJ]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > 3200:
            idxw = np.where(sel)[0]
            cw = c[idxw]
            part = np.argpartition(cw, -3200)[-3200:]
            sel = np.zeros_like(sel, dtype=bool)
            sel[idxw[part]] = True
            cnt = 3200
        I = iuI[sel]
        J = iuJ[sel]
        csel = c[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        vals = np.empty(p * (2 * d + 1), dtype=np.float64)
        Cb = cols.reshape(p, 2 * d + 1)
        Vb = vals.reshape(p, 2 * d + 1)
        ad = np.arange(d)
        Cb[:, :d] = I[:, None] * d + ad
        Cb[:, d:2 * d] = J[:, None] * d + ad
        Cb[:, 2 * d] = n * d
        Vb[:, :d] = x[J]
        Vb[:, d:2 * d] = x[I]
        Vb[:, 2 * d] = -1.0
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        v = _max_cos(y)
        if v < best:
            x = y
            best = v
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    HARD = 172.0  # leave margin for return
    rng = np.random.default_rng(0)
    n, d = 323, 7
    x_best = None
    b_best = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(a)
                if v < b_best:
                    b_best = v
                    x_best = a.copy()
        except Exception:
            pass
    if x_best is None:
        a = rng.standard_normal(size=(n, d))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        x_best = a
        b_best = _max_cos(a)

    # exploration until ~135s, reserve rest for LP
    EXPL_END = t0 + 135.0
    trial = 0
    # quick pre-anneal params
    while time.time() - t0 < EXPL_END - t0 and time.time() < t0 + 135.0:
        trial += 1
        # time guard: need at least ~8s for final polish
        if time.time() > t0 + 132.0:
            break
        y = x_best.copy()
        mode = trial % 4
        if mode == 0:
            # global jitter + anneal
            sig = 0.04 + 0.02 * rng.random()
            y = y + sig * rng.standard_normal(size=y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth_anneal(y, steps=700, T0=0.015, T1=0.0025, e0=0.04, e1=0.004)
        elif mode == 1:
            K = int(rng.integers(18, 30))
            c0 = int(rng.integers(n))
            dots = y @ y[c0]
            idx = np.argpartition(-dots, K)[:K]
            y = _greedy_cap_reinsert(y, idx, rng, n_cand=6000)
            y = _smooth_anneal(y, steps=900, T0=0.02, T1=0.002, e0=0.05, e1=0.003)
        elif mode == 2:
            K = int(rng.integers(28, 42))
            c0 = int(rng.integers(n))
            dots = y @ y[c0]
            idx = np.argpartition(-dots, K)[:K]
            y = _greedy_cap_reinsert(y, idx, rng, n_cand=8000)
            y = _smooth_anneal(y, steps=1100, T0=0.02, T1=0.002, e0=0.05, e1=0.003)
        else:
            # hub-targeted: worst points
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            w = G.max(axis=1)
            K = int(rng.integers(15, 25))
            idx = np.argpartition(-w, K)[:K]
            # expand each hub to small neighbourhood for coordination
            # add a few nearest of first hub
            c0 = int(idx[0])
            dots = y @ y[c0]
            extra = np.argpartition(-dots, 10)[:10]
            idx = np.unique(np.concatenate([idx, extra]))[:35]
            y = _greedy_cap_reinsert(y, idx, rng, n_cand=6000)
            y = _smooth_anneal(y, steps=900, T0=0.018, T1=0.002, e0=0.05, e1=0.003)
        v = _max_cos(y)
        if v < b_best - 1e-12:
            x_best = y
            b_best = v
        # early exit if big win
        if time.time() > t0 + 133.0:
            break
        # stop after many trials to guarantee polish time
        if trial >= 25:
            break

    remain = HARD - (time.time() - t0)
    # reserve 2s for return
    lp_time = float(min(30.0, max(5.0, remain - 2.0)))
    if lp_time > 3.0:
        try:
            x_best = _lp_polish_fast(x_best, lp_time)
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
