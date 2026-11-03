import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _descend(y, steps, T0, T1, lr0, lr1):
    n = y.shape[0]
    for s in range(steps):
        f = s / max(steps - 1, 1)
        T = T0 + (T1 - T0) * f
        lr = lr0 + (lr1 - lr0) * f
        G = y @ y.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        # softmax weights
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        rs = E.sum(axis=1, keepdims=True) + 1e-20
        A = E @ y
        S = (A * y).sum(axis=1, keepdims=True)
        F = (A - S * y) / rs
        y = y - lr * F
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y /= yn
    return y

def _deepest_hole(Xr, rng, n_cand=6000, jitter=0.0):
    d = Xr.shape[1]
    Y = rng.normal(size=(n_cand, d))
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    dots = Y @ Xr.T
    md = dots.max(axis=1)
    # randomized among top few for diversity
    k = min(5, n_cand)
    idx = np.argpartition(md, k - 1)[:k]
    # pick random among top-k weighted to best
    pick = idx[int(rng.integers(0, k))]
    v = Y[pick]
    if jitter > 0:
        v = v + jitter * rng.normal(size=(d,))
        v /= np.linalg.norm(v)
    return v

def _lp_polish(x, seconds, p_cap=1500):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    best_x = x.copy()
    radius = 1e-3
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        window = max(20.0 * radius, 1e-9)
        sel = c_all > m - window
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > p_cap:
            idx = np.where(sel)[0]
            part = np.argpartition(-c_all[idx], p_cap - 1)[:p_cap]
            keep = np.zeros_like(sel)
            keep[idx[part]] = True
            sel = keep
        I = iu0[sel]; J = iu1[sel]
        csel = c_all[sel]
        p = int(sel.sum())
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([(I[:, None] * d + np.arange(d)).reshape(p, d),
                               (J[:, None] * d + np.arange(d)).reshape(p, d),
                               np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
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
        y /= yn
        v = _max_cos(y)
        if v < best:
            best = v; best_x = y.copy(); x = y
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
            if radius < 1e-7:
                x = best_x.copy()
        if time.time() - start >= seconds:
            break
    return best_x

def solve():
    t0 = time.time()
    HARD = 160.0  # wall cutoff for building, reserve rest for final polish
    TL = 178.0
    n_full, d = 123, 13
    rng_master = np.random.default_rng(7)
    # load baseline
    base = None; base_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n_full, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(a)
                if v < base_val:
                    base_val = v; base = a.copy()
        except Exception:
            pass
    if base is None:
        base = rng_master.normal(size=(n_full, d))
        base /= np.linalg.norm(base, axis=1, keepdims=True)
        base_val = _max_cos(base)
    best123 = base.copy(); best123_val = float(base_val)
    best110 = None; best110_val = 1e9
    best120 = None; best120_val = 1e9

    def grow_once(X_init, rng, stage_steps=130, stage_T0=0.016, stage_T1=0.004, stage_lr0=0.35, stage_lr1=0.05):
        y = X_init.copy()
        n_cur = y.shape[0]
        while n_cur < n_full:
            if time.time() - t0 > HARD:
                break
            vnew = _deepest_hole(y, rng, n_cand=6000, jitter=0.003)
            y = np.vstack([y, vnew.reshape(1, d)])
            n_cur += 1
            # brief re-opt: fewer steps right after insertion, cap 2-3s (here ~0.1s)
            try:
                y = _descend(y, steps=stage_steps, T0=stage_T0, T1=stage_T1, lr0=stage_lr0, lr1=stage_lr1)
            except Exception:
                pass
            # small global jitter every few insertions to avoid immediate rejamming
            if n_cur % 7 == 0:
                y = y + 0.002 * rng.normal(size=y.shape)
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                try:
                    y = _descend(y, steps=60, T0=0.010, T1=0.003, lr0=0.25, lr1=0.04)
                except Exception:
                    pass
        return y

    # build multiple chains with randomized orders/sizes
    chain_id = 0
    # pre-generate seeds deterministically
    seeds = [11, 22, 33, 44, 55, 66, 77, 88, 99, 123]
    N0_list = [85, 90, 95, 80, 100, 88, 92, 86, 98, 90]
    for ci in range(len(seeds)):
        if time.time() - t0 > HARD - 8:
            break
        seed = seeds[ci]; N0 = N0_list[ci % len(N0_list)]
        rng = np.random.default_rng(seed)
        if ci % 2 == 0:
            # subset of base
            perm = rng.permutation(n_full)[:N0]
            X0 = base[perm].copy()
            # perturb to leave basin
            X0 = X0 + 0.01 * rng.normal(size=X0.shape)
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
            try:
                X0 = _descend(X0, steps=200, T0=0.018, T1=0.004, lr0=0.35, lr1=0.05)
            except Exception:
                pass
        else:
            # optimized random start
            X0 = rng.normal(size=(N0, d))
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
            try:
                X0 = _descend(X0, steps=400, T0=0.025, T1=0.004, lr0=0.45, lr1=0.05)
            except Exception:
                pass
        # grow, track intermediates
        y = X0.copy()
        n_cur = y.shape[0]
        # incremental growth with intermediate capture
        while n_cur < n_full:
            if time.time() - t0 > HARD:
                break
            vnew = _deepest_hole(y, rng, n_cand=6000, jitter=0.003)
            y = np.vstack([y, vnew.reshape(1, d)])
            n_cur += 1
            try:
                y = _descend(y, steps=130, T0=0.016, T1=0.004, lr0=0.35, lr1=0.05)
            except Exception:
                pass
            if n_cur == 110:
                v110 = _max_cos(y)
                if v110 < best110_val:
                    best110_val = v110; best110 = y.copy()
            if n_cur == 120:
                v120 = _max_cos(y)
                if v120 < best120_val:
                    best120_val = v120; best120 = y.copy()
            if n_cur % 7 == 0 and n_cur < n_full:
                y = y + 0.002 * rng.normal(size=y.shape)
                y /= np.linalg.norm(y, axis=1, keepdims=True)
        if y.shape[0] == n_full:
            # final quick tightening of completed config
            try:
                y = _descend(y, steps=250, T0=0.012, T1=0.0025, lr0=0.30, lr1=0.03)
            except Exception:
                pass
            v = _max_cos(y)
            if v < best123_val:
                best123_val = v; best123 = y.copy()
        chain_id += 1

    # branch from promising intermediates at N=110 and N=120 with new orders
    for tag, Xmid in (("110", best110), ("120", best120)):
        if Xmid is None:
            continue
        if time.time() - t0 > HARD - 4:
            break
        for bi in range(2):
            if time.time() - t0 > HARD:
                break
            rng = np.random.default_rng(1000 + (110 if tag == "110" else 120) + bi)
            y = Xmid.copy()
            # shake intermediate slightly for new order
            y = y + 0.004 * rng.normal(size=y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = grow_once(y, rng, stage_steps=120)
            if y.shape[0] == n_full:
                try:
                    y = _descend(y, steps=250, T0=0.012, T1=0.0025, lr0=0.30, lr1=0.03)
                except Exception:
                    pass
                v = _max_cos(y)
                if v < best123_val:
                    best123_val = v; best123 = y.copy()

    # finish with one LP polish of best completed config
    remaining = TL - (time.time() - t0) - 1.0
    remaining = max(1.0, min(remaining, 30.0))
    # if our grown best is worse than base, polish base instead (or both if time)
    target = best123
    # quick choice: polish whichever is better
    try:
        polished = _lp_polish(target, seconds=remaining)
        pv = _max_cos(polished)
        if pv < best123_val:
            best123 = polished; best123_val = pv
        else:
            # if no improvement and target was not base, try base briefly if time left
            if time.time() - t0 < TL - 6 and _max_cos(target) > base_val + 1e-12:
                r2 = max(1.0, TL - (time.time() - t0) - 1.0)
                p2 = _lp_polish(base, seconds=min(6.0, r2))
                if _max_cos(p2) < best123_val:
                    best123 = p2; best123_val = _max_cos(p2)
    except Exception:
        pass
    best123 = best123 / np.linalg.norm(best123, axis=1, keepdims=True)
    return best123
