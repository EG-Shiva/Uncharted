import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

_N = 123
_D = 13
_IU0 = None
_IU1 = None

def _get_iu(n):
    global _IU0, _IU1
    if _IU0 is None:
        iu = np.triu_indices(n, 1)
        _IU0 = iu[0]
        _IU1 = iu[1]
    return _IU0, _IU1

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _contact_counts(x, thresh=1e-3):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = G.max()
    mask = G > m - thresh
    # counts per point (symmetric, count both sides)
    cnt = mask.sum(axis=1)
    return cnt, m, G

def _lp_polish(x, seconds, radius_init=0.005, p_cap=2500):
    x = np.asarray(x, dtype=float)
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    iu0, iu1 = _get_iu(n)
    best = _max_cos(x)
    best_x = x.copy()
    radius = radius_init
    start = time.time()
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0, iu1]
        m = c_all.max()
        window = max(20.0 * radius, 1e-9)
        sel = c_all > m - window
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > p_cap:
            idx = np.where(sel)[0]
            # keep tightest p_cap
            part = np.argpartition(-c_all[idx], p_cap - 1)[:p_cap]
            keep = np.zeros_like(sel)
            keep[idx[part]] = True
            sel = keep
            cnt = p_cap
        I = iu0[sel]
        J = iu1[sel]
        csel = c_all[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([ (I[:, None] * d + np.arange(d)),
                                (J[:, None] * d + np.arange(d)),
                                np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            erows = np.repeat(np.arange(n), d)
            ecols = np.arange(n * d)
            A_eq = coo_matrix((x.ravel(), (erows, ecols)), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
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
            best = v
            best_x = y.copy()
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() - start >= seconds:
            break
    return best_x

def _smooth_run(y, steps, T0, T1, smax0, smax1):
    n, d = y.shape
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        T = T0 + (T1 - T0) * frac
        smax = smax0 + (smax1 - smax0) * frac
        G = y @ y.T
        np.fill_diagonal(G, -10.0)
        m = G.max()
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        row = E.sum(axis=1, keepdims=True) + 1e-20
        Avg = (E @ y) / row
        dot = (Avg * y).sum(axis=1, keepdims=True)
        Gt = Avg - dot * y
        # clip step to smax
        gn = np.sqrt((Gt ** 2).sum(axis=1, keepdims=True)) + 1e-20
        scale = np.minimum(1.0, smax / gn)
        step = Gt * scale  # length at most smax, direction of Gt
        # we want to move away from weighted neighbours: y -= step
        y = y - step
        y /= np.linalg.norm(y, axis=1, keepdims=True)
    return y

def _find_hole(X_keep, rng, n_cand=20000, refine_steps=250):
    # X_keep: (m,d) fixed points, find point minimizing max dot
    m, d = X_keep.shape
    cand = rng.normal(size=(n_cand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    S = cand @ X_keep.T
    mc = S.max(axis=1)
    order = np.argsort(mc)
    # refine top 3
    best_h = cand[order[0]]
    best_v = mc[order[0]]
    for k in range(min(3, n_cand)):
        h = cand[order[k]].copy()
        # anneal T 0.01 -> 0.003, steps refine_steps
        for s in range(refine_steps):
            frac = s / max(refine_steps - 1, 1)
            T = 0.012 + (0.003 - 0.012) * frac
            smax = 0.02 + (0.004 - 0.02) * frac
            g = X_keep @ h
            mx = g.max()
            w = np.exp((g - mx) / T)
            wsum = w.sum() + 1e-20
            grad = (w @ X_keep) / wsum
            # move away
            tg = grad - (grad @ h) * h
            nrm = float(np.sqrt((tg ** 2).sum())) + 1e-20
            st = min(1.0, smax / nrm)
            h = h - tg * st
            h /= np.linalg.norm(h)
        v = float((X_keep @ h).max())
        if v < best_v:
            best_v = v
            best_h = h.copy()
    return best_h

def solve():
    t0 = time.time()
    TIME_LIMIT = 178.0
    rng = np.random.default_rng(0)
    n, d = _N, _D
    # load start
    X0 = None
    best_val0 = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    v = _max_cos(a)
                    if v < best_val0:
                        best_val0 = v
                        X0 = a.copy()
        except Exception:
            pass
    if X0 is None:
        r = rng.normal(size=(n, d))
        r /= np.linalg.norm(r, axis=1, keepdims=True)
        X0 = r
        best_val0 = _max_cos(X0)
    cur = X0.copy()
    cur_val = float(best_val0)
    best = X0.copy()
    best_val = float(best_val0)
    # quick initial squeeze
    if time.time() - t0 < TIME_LIMIT - 25:
        try:
            cand = _lp_polish(best, 2.0, radius_init=0.003)
            v = _max_cos(cand)
            if v < best_val:
                best_val = v
                best = cand.copy()
                cur = cand.copy()
                cur_val = v
        except Exception:
            pass
    it = 0
    temp = 0.0004
    # reserve ~18s final
    while time.time() - t0 < TIME_LIMIT - 18:
        it += 1
        mode = it % 7
        y = cur.copy()
        try:
            cnt, mval, Gfull = _contact_counts(cur, thresh=0.002)
        except Exception:
            cnt = np.zeros(n)
        if mode == 0:
            sig = 0.025
            y = cur + sig * rng.normal(size=(n, d))
        elif mode == 1:
            # targeted hubs
            K = 12
            idx = np.argsort(-cnt)[:K]
            noise = rng.normal(size=(n, d)) * 0.008
            noise[idx] = rng.normal(size=(K, d)) * 0.12
            y = cur + noise
        elif mode == 2:
            # rattler reinsertion: 2 lowest
            K = 2
            idx = np.argsort(cnt)[:K]
            y = cur.copy()
            for j in idx:
                mask = np.ones(n, dtype=bool)
                mask[j] = False
                Xk = y[mask]
                h = _find_hole(Xk, rng, n_cand=15000, refine_steps=180)
                y[j] = h
            y = y + 0.01 * rng.normal(size=(n, d))
        elif mode == 3:
            K = 18
            idx = rng.choice(n, size=K, replace=False)
            noise = rng.normal(size=(n, d)) * 0.005
            noise[idx] = rng.normal(size=(K, d)) * 0.20
            y = cur + noise
        elif mode == 4:
            # worst pairs
            try:
                iu0, iu1 = _get_iu(n)
                c_all = Gfull[iu0, iu1]
                top = np.argpartition(-c_all, 5)[:5]
                uniq = set()
                for t in top:
                    uniq.add(int(iu0[t])); uniq.add(int(iu1[t]))
                uniq = np.array(list(uniq), dtype=int)
                noise = rng.normal(size=(n, d)) * 0.008
                noise[uniq] = rng.normal(size=(len(uniq), d)) * 0.10
                y = cur + noise
            except Exception:
                y = cur + 0.04 * rng.normal(size=(n, d))
        elif mode == 5:
            # move one hub to hole
            j = int(np.argmax(cnt))
            mask = np.ones(n, dtype=bool)
            mask[j] = False
            Xk = cur[mask]
            h = _find_hole(Xk, rng, n_cand=15000, refine_steps=180)
            y = cur.copy()
            y[j] = h
            y = y + 0.008 * rng.normal(size=(n, d))
        else:
            y = cur + 0.05 * rng.normal(size=(n, d))
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        # smooth
        try:
            # vary smooth length a bit
            steps = 350 if (it % 2 == 0) else 500
            y = _smooth_run(y, steps, 0.018, 0.003, 0.015, 0.003)
        except Exception:
            pass
        # short LP
        remain = TIME_LIMIT - 18 - (time.time() - t0)
        if remain < 0.5:
            break
        lp_sec = min(2.0, remain)
        # use smaller cap for speed when time low
        try:
            y2 = _lp_polish(y, lp_sec, radius_init=0.004, p_cap=2500)
        except Exception:
            y2 = y
        try:
            v = _max_cos(y2)
        except Exception:
            continue
        if v < best_val - 1e-12:
            best_val = v
            best = y2.copy()
            cur = y2.copy()
            cur_val = v
        elif v < cur_val:
            cur = y2.copy()
            cur_val = v
        else:
            # metropolis uphill to change basin
            if v < cur_val + 0.003:
                prob = float(np.exp(-(v - cur_val) / temp)) if v >= cur_val else 1.0)
                if rng.random() < prob:
                    cur = y2.copy()
                    cur_val = v
            # occasional restart to best if wandering too high
            if cur_val > best_val + 0.002 and (it % 10 == 0):
                cur = best.copy()
                cur_val = best_val
        # safety time check
        if time.time() - t0 >= TIME_LIMIT - 18:
            break
    # final polish on best
    remain = TIME_LIMIT - (time.time() - t0)
    remain = min(max(remain - 0.5, 1.0), 18.0)
    if remain > 0.5:
        try:
            # light smooth before final LP to settle
            b2 = _smooth_run(best.copy(), 200, 0.008, 0.002, 0.006, 0.002)
            bv = _max_cos(b2)
            if bv < best_val:
                best = b2
                best_val = bv
        except Exception:
            pass
        try:
            best = _lp_polish(best, remain, radius_init=0.003, p_cap=3000)
        except Exception:
            pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
