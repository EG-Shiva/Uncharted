import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def _smooth_polish(x, iters, T0, T1, lr0, lr1):
    n, d = x.shape
    for it in range(iters):
        frac = it / max(iters, 1)
        T = T0 * (T1 / T0) ** frac if T0 > 0 and T1 > 0 else T0
        lr = lr0 + (lr1 - lr0) * frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        D = (G - m) / T
        # threshold exp(-7)
        W = np.zeros_like(G)
        mk = D > -7.0
        W[mk] = np.exp(D[mk])
        rs = W.sum(axis=1)
        Wx = W @ x
        wgs = (W * G).sum(axis=1)
        inv = np.where(rs > 1e-12, 1.0 / (rs + 1e-12), 0.0)
        Gdir = (Wx - wgs[:, None] * x) * inv[:, None]
        x = x - lr * Gdir
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x = x / nr
    return x


def _lp_polish(x, seconds):
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
            idx = np.where(sel)[0]
            cidx = c[idx]
            part = np.argpartition(cidx, -3200)[-3200:]
            sel = np.zeros_like(sel, dtype=bool)
            sel[idx[part]] = True
            cnt = 3200
        I = iuI[sel]; J = iuJ[sel]; csel = c[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        vals = np.empty(p * (2 * d + 1), dtype=np.float64)
        Cb = cols.reshape(p, 2 * d + 1)
        Vb = vals.reshape(p, 2 * d + 1)
        ar = np.arange(d)
        Cb[:, :d] = I[:, None] * d + ar
        Cb[:, d:2 * d] = J[:, None] * d + ar
        Cb[:, 2 * d] = n * d
        Vb[:, :d] = x[J]
        Vb[:, d:2 * d] = x[I]
        Vb[:, 2 * d] = -1.0
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
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
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    rng = np.random.default_rng(0)
    # load start
    xb = None
    best = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (323, 7):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(a)
                if v < best:
                    best = v
                    xb = a.copy()
        except Exception:
            pass
    if xb is None:
        xb = rng.standard_normal(size=(323, 7))
        xb /= np.linalg.norm(xb, axis=1, keepdims=True)
        best = _max_cos(xb)
    n, d = xb.shape

    def worst_idx(x, k):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        w = G.max(axis=1)
        return np.argsort(-w)[:k], w, G

    # basin hopping with cheap smoother; reserve ~30s for final LP
    # total budget 180, keep safe margin: hopping until 140s
    HOP_END = 140.0
    trial = 0
    while time.time() - t0 < HOP_END:
        trial += 1
        y = xb.copy()
        r = rng.random()
        if r < 0.62:
            # cluster shake
            # choose center: mostly worst, sometimes random
            if rng.random() < 0.75:
                Gtmp = y @ y.T
                np.fill_diagonal(Gtmp, -2.0)
                wtmp = Gtmp.max(axis=1)
                order = np.argsort(-wtmp)
                # pick among top 30
                ci = int(order[rng.integers(0, min(30, n))])
                row = Gtmp[ci].copy()
                row[ci] = -2.0
                nn = np.argsort(-row)[:rng.integers(4, 12)]
                idx = np.concatenate([[ci], nn])
                # occasionally add rattler-like low-contact points
                if rng.random() < 0.3:
                    cnt = (Gtmp > wtmp.max() - 1e-9).sum(axis=1)
                    rattle = np.where(cnt <= 2)[0]
                    if len(rattle) > 0:
                        idx = np.concatenate([idx, rng.choice(rattle, size=min(len(rattle), 2), replace=False)])
                idx = np.unique(idx)
            else:
                ci = int(rng.integers(0, n))
                Gtmp = y @ y.T
                np.fill_diagonal(Gtmp, -2.0)
                row = Gtmp[ci]
                nn = np.argsort(-row)[:rng.integers(5, 12)]
                idx = np.unique(np.concatenate([[ci], nn]))
            sigma = float(rng.uniform(0.008, 0.07))
            # anneal sigma slightly as trials grow
            y[idx] += sigma * rng.standard_normal(size=(len(idx), d))
            y[idx] /= np.linalg.norm(y[idx], axis=1, keepdims=True)
            iters = int(rng.integers(700, 1600))
            T0 = float(rng.uniform(0.006, 0.016))
            T1 = float(rng.uniform(0.0008, 0.003))
            lr0 = float(rng.uniform(0.015, 0.035))
            lr1 = float(rng.uniform(0.002, 0.008))
            y = _smooth_polish(y, iters, T0, T1, lr0, lr1)
        else:
            # greedy hole reinsertion of 1-3 worst points
            k = int(rng.integers(1, 4))
            widx, wv, Gc = worst_idx(y, k)
            # for each worst, sample candidates and pick max-min
            for kk in widx:
                nc = 6000
                cand = rng.standard_normal(size=(nc, d))
                cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                others = np.delete(y, kk, axis=0)
                dots = cand @ others.T
                wm = dots.max(axis=1)
                bi = int(np.argmin(wm))
                y[kk] = cand[bi]
            # small local hill climb on reinserted
            for _ in range(1):
                for kk in widx:
                    cur = y[kk].copy()
                    others = np.delete(y, kk, axis=0)
                    cw = float((others @ cur).max())
                    tr = rng.standard_normal(size=(300, d)) * 0.03 + cur[None, :]
                    tr /= np.linalg.norm(tr, axis=1, keepdims=True)
                    dw = (tr @ others.T).max(axis=1)
                    bi = int(np.argmin(dw))
                    if dw[bi] < cw:
                        y[kk] = tr[bi]
            iters = int(rng.integers(800, 1600))
            T0 = float(rng.uniform(0.005, 0.012))
            T1 = float(rng.uniform(0.0007, 0.0025))
            lr0 = float(rng.uniform(0.015, 0.03))
            lr1 = float(rng.uniform(0.002, 0.007))
            y = _smooth_polish(y, iters, T0, T1, lr0, lr1)
        v = _max_cos(y)
        if v < best - 1e-12:
            # extra refine
            y2 = _smooth_polish(y.copy(), 1200, 0.004, 0.0006, 0.012, 0.003)
            v2 = _max_cos(y2)
            if v2 < v:
                y, v = y2, v2
            xb, best = y, v
        # time guard: if very little time, break
        if time.time() - t0 >= HOP_END:
            break

    # final snap with LP if time remains
    remain = 178.0 - (time.time() - t0)
    if remain > 8.0:
        # short extra smooth to tighten before LP
        xb = _smooth_polish(xb, 800, 0.003, 0.0005, 0.008, 0.002)
        lp_time = min(remain - 3.0, 28.0)
        if lp_time > 5.0:
            xb = _lp_polish(xb, lp_time)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    return xb
