import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _worst(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def _contacts(X, m, tol=1e-9):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return np.sum(G > m - tol, axis=1)

def _softmax_relax(X, steps, a0, a1, e0, e1, rng=None, jitter=0.0, time_lim=None, t0=None):
    n, d = X.shape
    for it in range(steps):
        if time_lim is not None and t0 is not None:
            if time.time() - t0 > time_lim:
                break
        f = it / max(1, steps - 1)
        alpha = a0 + (a1 - a0) * f
        eta = e0 + (e1 - e0) * f
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        # stable softmax weights
        E = np.exp(alpha * (G - m))
        np.fill_diagonal(E, 0.0)
        s = float(E.sum())
        if not np.isfinite(s) or s == 0:
            continue
        Gr = (E @ X) / s
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        med = float(np.median(np.linalg.norm(Gt, axis=1))) + 1e-18
        step = Gt / med * eta
        # clip extreme steps
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-18
        maxn = eta * 3.0 + 1e-18
        sc = np.minimum(1.0, maxn / sn)
        step = step * sc
        if jitter > 0 and rng is not None:
            step = step + rng.standard_normal(size=X.shape) * jitter
        X = X - step
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _best_void(X, rng, n_cand=20000):
    d = X.shape[1]
    cand = rng.standard_normal(size=(n_cand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ X.T
    worst = dots.max(axis=1)
    bi = int(np.argmin(worst))
    return cand[bi], float(worst[bi])

def _refine_void(p0, X, rng, rounds=24, batch=512, sig0=0.06):
    best_p = p0.copy()
    best_w = float((X @ best_p).max())
    sig = float(sig0)
    for _ in range(rounds):
        prop = rng.standard_normal(size=(batch, X.shape[1])) * sig + best_p[None, :]
        prop /= np.linalg.norm(prop, axis=1, keepdims=True)
        w = (prop @ X.T).max(axis=1)
        bi = int(np.argmin(w))
        if float(w[bi]) < best_w:
            best_w = float(w[bi])
            best_p = prop[bi].copy()
        sig *= 0.88
    return best_p, best_w

def _lp_polish(X, seconds, t0, init_radius=4e-3, cap=3500):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = float(init_radius)
    iuI, iuJ = np.triu_indices(n, 1)
    while True:
        if time.time() - t0 >= seconds:
            break
        if radius < 1e-14:
            break
        G = X @ X.T
        c = G[iuI, iuJ]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > cap:
            idx = np.where(sel)[0]
            cidx = c[idx]
            part = np.argpartition(cidx, -cap)[-cap:]
            sel = np.zeros_like(sel, dtype=bool)
            sel[idx[part]] = True
            cnt = cap
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
        Vb[:, :d] = X[J]
        Vb[:, d:2 * d] = X[I]
        Vb[:, 2 * d] = -1.0
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            X = y; best = v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def _regrow_route(x_start, k_remove, rng, t0, time_lim, n_cand=18000):
    n, d = x_start.shape
    m = _max_cos(x_start)
    cc = _contacts(x_start, m, tol=1e-9)
    ratt = np.where(cc == 0)[0].tolist()
    order_desc = list(np.argsort(-cc))
    # choose removal: rattlers + hubs (+ random to fill)
    torep = []
    for r in ratt:
        torep.append(int(r))
    for o in order_desc:
        if len(torep) >= k_remove:
            break
        if int(o) not in torep:
            torep.append(int(o))
    # if still short (no rattlers), add random
    while len(torep) < k_remove:
        c = int(rng.integers(0, n))
        if c not in torep:
            torep.append(c)
    torep = torep[:k_remove]
    keep = [i for i in range(n) if i not in set(torep)]
    Y = x_start[keep].copy()
    # annealed repulsion on shrunken code: aim well below record
    # budget split: 60% shrink-opt, 40% regrow relaxations
    t_shrink = time_lim * 0.45
    t0s = time.time()
    # two-stage anneal
    Y = _softmax_relax(Y, steps=2200, a0=60.0, a1=550.0, e0=0.035, e1=0.006,
                       rng=rng, jitter=0.0, time_lim=t_shrink, t0=t0s)
    # regrow one by one with void sampling + all-move relaxation
    for _ in range(k_remove):
        if time.time() - t0 > time_lim:
            break
        p0, _ = _best_void(Y, rng, n_cand=n_cand)
        p1, _ = _refine_void(p0, Y, rng, rounds=22, batch=512, sig0=0.06)
        Y = np.vstack([Y, p1[None, :]])
        # short global relaxation after each insertion
        Y = _softmax_relax(Y, steps=220, a0=250.0, a1=550.0, e0=0.012, e1=0.003,
                           rng=None, jitter=0.0)
    return Y

def _overfull_route(x_start, k_add, rng, time_lim, t0):
    n, d = x_start.shape
    Y = x_start.copy()
    # add k_add voids greedily
    for _ in range(k_add):
        p0, _ = _best_void(Y, rng, n_cand=15000)
        p1, _ = _refine_void(p0, Y, rng, rounds=18, batch=384, sig0=0.06)
        Y = np.vstack([Y, p1[None, :]])
    # optimise overfull N=330
    Y = _softmax_relax(Y, steps=1400, a0=80.0, a1=500.0, e0=0.030, e1=0.005,
                       rng=rng, jitter=0.0, time_lim=time_lim * 0.6, t0=t0)
    if Y.shape[0] != n + k_add:
        return x_start.copy()
    # delete k_add worst-contact points
    w = _worst(Y)
    # delete worst 7 (largest worst-dot)
    idx_del = list(np.argpartition(w, -k_add)[-k_add:])
    keep = [i for i in range(Y.shape[0]) if i not in set(idx_del)]
    Z = Y[keep].copy()
    # short relaxation
    Z = _softmax_relax(Z, steps=600, a0=200.0, a1=550.0, e0=0.012, e1=0.003,
                       rng=None, jitter=0.0)
    return Z

def solve():
    t0 = time.time()
    HARD = 170.0  # leave margin under 180
    rng = np.random.default_rng(20261101)
    x0 = None; b0 = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (323, 7):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.any(~np.isfinite(a)):
                    continue
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

    cands = []

    # Route A: shrink 323 -> ~311-313 then regrow (cardinality homotopy)
    # two k values if time allows
    for k_remove, nc in ((10, 18000), (14, 14000)):
        if time.time() - t0 > 105:
            break
        try:
            Y = _regrow_route(x_best, k_remove, rng, t0, time_lim=105 - (time.time() - t0), n_cand=nc)
            if Y.shape != (323, 7):
                continue
            Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
            v = _max_cos(Y)
            # retain only if competitive unpolished (within 6e-3 of best)
            if v < best + 6e-3:
                cands.append((v, Y.copy()))
                if v < best - 1e-12:
                    best = v; x_best = Y.copy()
        except Exception:
            continue

    # Route B: reverse - overfull N=330 then delete 7 worst
    if time.time() - t0 < 130:
        try:
            remain_b = 130 - (time.time() - t0)
            t0b = time.time()
            Z = _overfull_route(x_best, 7, rng, time_lim=remain_b, t0=t0b)
            if Z.shape == (323, 7):
                Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
                v = _max_cos(Z)
                if v < best + 6e-3:
                    cands.append((v, Z.copy()))
                    if v < best - 1e-12:
                        best = v; x_best = Z.copy()
        except Exception:
            pass

    # pick best candidate (lowest unpolished that is competitive)
    if cands:
        cands.sort(key=lambda t: t[0])
        # prefer the absolute best competitive one
        v0, Y0 = cands[0]
        # if it beats or is close, make it the base for final polish
        x_best = Y0.copy(); best = float(v0)

    # One bounded LP polish before global deadline on the retained regrown code
    remain = HARD - (time.time() - t0)
    if remain > 4:
        try:
            # polish the current best; cap time
            lp_time = min(remain, 38.0)
            lp_start = time.time()
            # use absolute deadline HARD
            abs_dead = HARD - (time.time() - t0)
            # _lp_polish uses t0 as start reference with seconds budget; create local t0
            lt0 = time.time()
            x_best = _lp_polish(x_best, seconds=lt0 + lp_time - 0.5 - lt0 + lt0 if False else lp_time, t0=lt0, init_radius=4e-3)
            # fix: _lp_polish expects (X, seconds, t0) with check time.time()-t0 >= seconds
            best = _max_cos(x_best)
        except Exception:
            pass

    # ensure against regression: if polish made it worse than original, revert to original best file
    try:
        if best > b0 + 1e-12:
            # keep original record-best instead
            x_best = x0.copy(); best = float(b0)
            remain2 = HARD - (time.time() - t0)
            if remain2 > 4:
                lt0 = time.time()
                x_best = _lp_polish(x_best, seconds=min(remain2, 12.0), t0=lt0, init_radius=2e-3)
    except Exception:
        pass

    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    # final safety: replace non-finite
    if np.any(~np.isfinite(x_best)):
        x_best = x0.copy()
    return x_best
