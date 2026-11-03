import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _adam_smooth(X, steps, k0, k1, lr0, lr1, rng=None, deadline=None):
    n, d = X.shape
    m1 = np.zeros_like(X)
    m2 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for s in range(steps):
        if deadline is not None and (s & 255) == 0 and time.time() > deadline:
            break
        frac = s / max(1, steps - 1)
        k = k0 * (k1 / k0) ** frac if k0 != k1 else k0
        lr = lr0 * (lr1 / lr0) ** frac if lr0 != lr1 else lr0
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        mv = G.max()
        W = np.exp(k * (G - mv))
        np.fill_diagonal(W, 0.0)
        sw = W.sum()
        if not np.isfinite(sw) or sw == 0:
            break
        grad = W @ X / sw
        dot = (grad * X).sum(axis=1, keepdims=True)
        gt = grad - dot * X
        # clip extreme
        # adam
        m1 = b1 * m1 + (1 - b1) * gt
        m2 = b2 * m2 + (1 - b2) * (gt * gt)
        bc1 = 1.0 - b1 ** (s + 1)
        bc2 = 1.0 - b2 ** (s + 1)
        upd = lr * (m1 / bc1) / (np.sqrt(m2 / bc2) + eps)
        X = X - upd
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X = X / nr
        if not np.all(np.isfinite(X)):
            break
    return X

def _lp_polish(x, seconds, t_end):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < t_end and time.time() < t_end and radius > 1e-14:
        # time check via t_end only; seconds handled by caller
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        if p > 2500:
            # keep tightest 2500
            idx = np.argpartition(-c[sel], 2500)[:2500]
            I, J = I[idx], J[idx]
            p = 2500
            bv = np.empty(p)
            # need corresponding b_ub values
            cc = c[sel][idx]
        else:
            cc = c[sel]
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cc, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() >= t_end:
            break
    return x, best

def _best_single(others, rng, n_cand=3000):
    d = others.shape[1]
    cand = rng.standard_normal(size=(n_cand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    sc = (cand @ others.T).max(axis=1)
    return cand[np.argmin(sc)]

def _refine_single(c, others, steps=250):
    for it in range(steps):
        dots = others @ c
        m = dots.max()
        w = np.exp(60.0 * (dots - m))
        s = w.sum()
        if s == 0 or not np.isfinite(s):
            break
        w /= s
        grad = w @ others
        gt = grad - (grad @ c) * c
        step = 0.2 * (1.0 - 0.6 * it / steps)
        c = c - step * gt
        nrm = np.linalg.norm(c)
        if nrm == 0 or not np.isfinite(nrm):
            break
        c /= nrm
    return c

def solve():
    T0 = time.time()
    BUDGET = 172.0
    T_END = T0 + BUDGET
    rng = np.random.default_rng(0)
    # load
    best = None; best_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn).astype(float)
            a /= np.linalg.norm(a, axis=1, keepdims=True)
            if a.shape == (94, 13) and np.all(np.isfinite(a)):
                v = _max_cos(a)
                if v < best_val:
                    best_val = v; best = a.copy()
        except Exception:
            pass
    if best is None:
        best = rng.standard_normal(size=(94, 13))
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    n, d = best.shape
    cur = best.copy(); cur_val = best_val
    HOP_END = T0 + 118.0
    trial = 0
    # precompute iu not needed
    while time.time() < HOP_END and time.time() < T_END - 55:
        trial += 1
        elapsed = time.time() - T0
        Temp = 0.0012 * max(0.08, 1.0 - elapsed / 120.0)
        y = cur.copy()
        r = rng.random()
        # contact info of cur (cheap)
        Gc = y @ y.T
        np.fill_diagonal(Gc, -2.0)
        mx = Gc.max()
        worst = Gc.max(axis=1)
        deg = (Gc > mx - 2e-4).sum(axis=1)
        if r < 0.30:
            # reinsert 1-2 worst points
            k = 1 if rng.random() < 0.7 else 2
            # order by worst then degree
            order = np.argsort(-(worst + 0.0005 * deg))
            victims = order[:k].copy()
            # sometimes pick random low-degree (near rattler)
            if rng.random() < 0.3:
                victims[0] = int(np.argmin(deg + rng.random(n) * 0.5))
            for vi in victims:
                others = np.delete(y, vi, axis=0)
                c = _best_single(others, rng, n_cand=3000)
                c = _refine_single(c, others, steps=250)
                y[vi] = c
        elif r < 0.48:
            # block shake of overloaded neighbourhood
            hub = int(np.argmax(deg + rng.random(n) * 2.0))
            dots = y @ y[hub]
            neigh = np.where(dots > 0.1)[0]
            if len(neigh) < 3:
                neigh = np.argsort(-dots)[:6]
            k = min(len(neigh), int(rng.integers(3, 7)))
            sel = rng.choice(neigh, size=k, replace=False)
            sigma = float(rng.choice([0.03, 0.06, 0.10, 0.16]))
            y[sel] += sigma * rng.standard_normal(size=(k, d))
            y[sel] /= np.linalg.norm(y[sel], axis=1, keepdims=True)
        elif r < 0.66:
            # global small shake
            sigma = float(rng.choice([0.004, 0.008, 0.015, 0.030]))
            if rng.random() < 0.4:
                k = int(rng.integers(2, 6))
                sel = rng.choice(n, size=k, replace=False)
                y[sel] += sigma * 2.0 * rng.standard_normal(size=(k, d))
                y[sel] /= np.linalg.norm(y[sel], axis=1, keepdims=True)
            else:
                y = y + sigma * rng.standard_normal(size=y.shape)
                y /= np.linalg.norm(y, axis=1, keepdims=True)
        elif r < 0.82:
            # shake high-degree + low-degree mix
            o_lo = np.argsort(deg); o_hi = np.argsort(-deg)
            k = int(rng.integers(2, 6))
            victims = set()
            for _ in range(k):
                if rng.random() < 0.5:
                    victims.add(int(o_lo[rng.integers(0, 10)]))
                else:
                    victims.add(int(o_hi[rng.integers(0, 10)]))
            victims = list(victims)
            sigma = float(rng.choice([0.02, 0.05, 0.09]))
            y[victims] += sigma * rng.standard_normal(size=(len(victims), d))
            y[victims] /= np.linalg.norm(y[victims], axis=1, keepdims=True)
        else:
            # randomize 2-3 points fully
            k = int(rng.integers(2, 4))
            sel = rng.choice(n, size=k, replace=False)
            fresh = rng.standard_normal(size=(k, d))
            fresh /= np.linalg.norm(fresh, axis=1, keepdims=True)
            # keep best of 2 tries per point greedily? just assign
            y[sel] = fresh
        # smooth relax
        # choose schedule by mode
        if r < 0.30:
            steps, k0, k1, lr0, lr1 = 1200, 30.0, 350.0, 0.008, 0.0015
        else:
            steps, k0, k1, lr0, lr1 = 900, 25.0, 250.0, 0.008, 0.002
        # shorten if time pressure
        if T_END - time.time() < 80:
            steps = min(steps, 600)
        y = _adam_smooth(y, steps, k0, k1, lr0, lr1, deadline=HOP_END + 5)
        v = _max_cos(y)
        # promising? do short LP
        do_lp = (v < cur_val + 0.004) or (v < best_val + 0.002)
        if do_lp and time.time() < HOP_END:
            y2, v2 = _lp_polish(y, 0, min(time.time() + 1.2, HOP_END, T_END - 40))
            # keep polished
            y, v = y2, v2
        # acceptance
        if v < cur_val:
            cur, cur_val = y, v
            if v < best_val:
                best, best_val = y.copy(), v
        elif rng.random() < np.exp((cur_val - v) / max(Temp, 1e-6)):
            cur, cur_val = y, v
        # also if best improved a lot, sync cur
        if best_val < cur_val - 0.002:
            cur, cur_val = best.copy(), best_val
    # final intensive polish of best with restarts
    # try a few micro-kicks around best
    if time.time() < T_END - 5:
        # long polish best
        best, best_val = _lp_polish(best, 0, min(T_END - 8, time.time() + 22))
    # micro search
    while time.time() < T_END - 2:
        sigma = 0.002
        y = best + sigma * rng.standard_normal(size=best.shape)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        y = _adam_smooth(y, 400, 40.0, 400.0, 0.004, 0.001, deadline=T_END - 2)
        y, v = _lp_polish(y, 0, min(T_END - 1, time.time() + 2.5))
        if v < best_val:
            best, best_val = y, v
        else:
            # try even smaller kick + pure LP
            if time.time() > T_END - 6:
                break
    best /= np.linalg.norm(best, axis=1, keepdims=True)
    return best
