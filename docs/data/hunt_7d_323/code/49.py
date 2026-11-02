import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _anneal(x, steps, tau0, tau1, lr0, lr1, mu, seed_jit=0.0, rng=None):
    n, d = x.shape
    vel = np.zeros_like(x)
    # track best via m already computed
    best_m = np.inf
    best_x = x.copy()
    # initial m
    for s in range(steps):
        frac = s / max(1, steps - 1)
        tau = tau0 + (tau1 - tau0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        if m < best_m:
            best_m = m
            best_x = x.copy()
        # softmax weights dense, vectorized
        # exp in place to save alloc
        E = np.exp((G - m) / tau)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum())
        if not np.isfinite(S) or S <= 0:
            break
        E /= S
        Wx = E @ x
        dot = np.sum(Wx * x, axis=1, keepdims=True)
        Gt = Wx - dot * x
        vel = mu * vel + Gt
        x = x - lr * vel
        # renormalize
        xn = np.sqrt(np.sum(x * x, axis=1, keepdims=True))
        xn[xn == 0] = 1.0
        x /= xn
    # final check
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    if m < best_m:
        best_m = m
        best_x = x.copy()
    return best_x

def _void_reinsert(y, to_replace, rng, n_cand=12000):
    n, d = y.shape
    for k in to_replace:
        cand = rng.standard_normal(size=(n_cand, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        others = np.delete(y, k, axis=0)
        # chunk to bound memory
        best_w = np.inf
        best_c = None
        cs = 4000
        for a in range(0, n_cand, cs):
            b = min(n_cand, a + cs)
            dots = cand[a:b] @ others.T
            worst = dots.max(axis=1)
            bi = int(np.argmin(worst))
            if worst[bi] < best_w:
                best_w = float(worst[bi])
                best_c = cand[a + bi].copy()
        y[k] = best_c
    return y

def _cap_shift(x, rng):
    # coherent translation of a random spherical cap
    n, d = x.shape
    u = rng.standard_normal(size=(d,))
    u /= np.linalg.norm(u)
    dots = x @ u
    thresh = float(rng.uniform(0.78, 0.90))
    idx = np.where(dots > thresh)[0]
    if len(idx) < 2 or len(idx) > 80:
        return x
    dv = rng.standard_normal(size=(d,))
    dv /= np.linalg.norm(dv)
    alpha = float(rng.uniform(0.04, 0.14))
    # common direction projected to each tangent plane
    proj = x[idx] @ dv
    x[idx] = x[idx] + alpha * (dv[None, :] - proj[:, None] * x[idx])
    x[idx] /= np.linalg.norm(x[idx], axis=1, keepdims=True)
    return x

def _lp_polish(x, deadline, init_radius=3e-3, max_iters=40):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = float(init_radius)
    iuI, iuJ = np.triu_indices(n, 1)
    it = 0
    while time.time() < deadline and radius > 1e-14 and it < max_iters:
        it += 1
        G = x @ x.T
        c = G[iuI, iuJ]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > 2200:
            idx = np.where(sel)[0]
            cidx = c[idx]
            part = np.argpartition(cidx, -2200)[-2200:]
            sel = np.zeros_like(sel, dtype=bool)
            sel[idx[part]] = True
            cnt = 2200
        I = iuI[sel]; J = iuJ[sel]
        csel = c[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        vals = np.empty(p * (2 * d + 1), dtype=np.float64)
        Cm = cols.reshape(p, 2 * d + 1)
        Vm = vals.reshape(p, 2 * d + 1)
        ar = np.arange(d)
        Cm[:, :d] = I[:, None] * d + ar
        Cm[:, d:2 * d] = J[:, None] * d + ar
        Cm[:, 2 * d] = n * d
        Vm[:, :d] = x[J]
        Vm[:, d:2 * d] = x[I]
        Vm[:, 2 * d] = -1.0
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
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
        if not np.all(np.isfinite(y)):
            radius *= 0.5
            continue
        v = _max_cos(y)
        if v < best:
            x = y; best = v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
        if time.time() >= deadline:
            break
    return x

def solve():
    t0 = time.time()
    HARD = t0 + 172.0
    EXPLORE_END = t0 + 143.0
    rng = np.random.default_rng(20260711)
    x0 = None; b0 = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (323, 7):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    v = _max_cos(a)
                    if v < b0:
                        b0 = v; x0 = a.copy()
        except Exception:
            pass
    if x0 is None:
        x0 = rng.standard_normal(size=(323, 7))
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        b0 = _max_cos(x0)
    x_best = x0.copy(); best = float(b0)
    n, d = x_best.shape
    it = 0
    while time.time() < EXPLORE_END:
        remain = HARD - 6.0 - time.time()
        if remain <= 0:
            break
        mode = it % 5
        x = x_best.copy()
        if mode == 0:
            sig = 0.022 if (it % 10 == 0) else 0.012
            x = x + rng.standard_normal(size=x.shape) * sig
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal(x, steps=900, tau0=0.016, tau1=0.0025, lr0=4.0, lr1=0.5, mu=0.82)
        elif mode == 1:
            # cap surgery + anneal
            x = _cap_shift(x, rng)
            x = x + rng.standard_normal(size=x.shape) * 0.006
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal(x, steps=900, tau0=0.013, tau1=0.0022, lr0=3.5, lr1=0.5, mu=0.80)
        elif mode == 2:
            # worst-point kick
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            w = G.max(axis=1)
            k = 20
            idx = np.argpartition(w, -k)[-k:]
            x[idx] += rng.standard_normal(size=(k, d)) * 0.09
            x[idx] /= np.linalg.norm(x[idx], axis=1, keepdims=True)
            x = _anneal(x, steps=800, tau0=0.011, tau1=0.002, lr0=3.0, lr1=0.4, mu=0.78)
        elif mode == 3:
            # void reinsertion of worst few
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            w = G.max(axis=1)
            k = 3
            idx = np.argpartition(w, -k)[-k:].tolist()
            # also occasionally include a random point
            if rng.random() < 0.4:
                idx[0] = int(rng.integers(0, n))
            x = _void_reinsert(x, [int(v) for v in idx], rng, n_cand=12000)
            x = x + rng.standard_normal(size=x.shape) * 0.004
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal(x, steps=800, tau0=0.010, tau1=0.002, lr0=3.0, lr1=0.4, mu=0.80)
        else:
            # explicit worst-pair split + anneal
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            iuI, iuJ = np.triu_indices(n, 1)
            c = G[iuI, iuJ]
            top = np.argpartition(c, -10)[-10:]
            for t in top:
                i = int(iuI[t]); j = int(iuJ[t])
                di = x[i] - x[j]
                di = di - np.dot(di, x[i]) * x[i]
                dj = -di
                # for j use its own tangent
                dj2 = x[j] - x[i]
                dj2 = dj2 - np.dot(dj2, x[j]) * x[j]
                ni = np.linalg.norm(di); nj = np.linalg.norm(dj2)
                if ni > 1e-12:
                    x[i] = x[i] + 0.05 * di / ni
                if nj > 1e-12:
                    x[j] = x[j] + 0.05 * dj2 / nj
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal(x, steps=700, tau0=0.009, tau1=0.0018, lr0=2.5, lr1=0.35, mu=0.75)
        v = _max_cos(x)
        if v < best - 1e-12 and np.all(np.isfinite(x)):
            best = v; x_best = x.copy()
        it += 1
        # safety: if each trial ~0.4s, loop naturally respects deadline
    # tighten with small-tau descent (no LP) first
    if time.time() < HARD - 22.0:
        try:
            x_best = _anneal(x_best, steps=1200, tau0=0.004, tau1=0.0009, lr0=0.9, lr1=0.12, mu=0.6)
            bv = _max_cos(x_best)
            best = bv
        except Exception:
            pass
    # single final LP snap with bounded budget
    if time.time() < HARD - 3.0:
        try:
            dl = min(HARD - 1.5, time.time() + 16.0)
            x_best = _lp_polish(x_best, deadline=dl, init_radius=2.5e-3, max_iters=30)
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    x_best[~np.isfinite(x_best)] = 0.0
    nn = np.linalg.norm(x_best, axis=1, keepdims=True)
    for i in range(n):
        if nn[i, 0] < 1e-12 or not np.isfinite(nn[i, 0]):
            vv = rng.standard_normal(size=(d,))
            vv /= np.linalg.norm(vv)
            x_best[i] = vv
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
