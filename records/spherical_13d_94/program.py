import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _adam_quantile(x, steps, M, lr0, lr1, iu0, iu1, deadline=1e18):
    n, d = x.shape
    m1 = np.zeros_like(x)
    v1 = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for s in range(steps):
        if s % 80 == 0 and time.time() > deadline:
            break
        frac = s / max(1, steps - 1)
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        vals = G[iu0, iu1]
        if M >= vals.size:
            t = float(vals.min())
        else:
            t = float(np.partition(vals, -M)[-M])
        Mm = G - t
        Mm[Mm < 0] = 0.0
        np.fill_diagonal(Mm, 0.0)
        cnt = float((Mm > 0).sum() * 0.5)
        if cnt < 1.0:
            cnt = 1.0
        Grad = (Mm @ x) / cnt
        dot = (Grad * x).sum(axis=1, keepdims=True)
        Gt = Grad - dot * x
        m1 = b1 * m1 + (1 - b1) * Gt
        v1 = b2 * v1 + (1 - b2) * (Gt * Gt)
        bc1 = 1.0 - b1 ** (s + 1)
        bc2 = 1.0 - b2 ** (s + 1)
        upd = lr * (m1 / bc1) / (np.sqrt(v1 / bc2) + eps)
        x = x - upd
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x = x / nr
        if not np.all(np.isfinite(x)):
            break
    return x

def _adam_lse(x, steps, k0, k1, lr0, lr1, deadline=1e18):
    n, d = x.shape
    m1 = np.zeros_like(x)
    v1 = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    eye = np.eye(n)
    for s in range(steps):
        if s % 100 == 0 and time.time() > deadline:
            break
        frac = s / max(1, steps - 1)
        kk = k0 * (k1 / k0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        mval = float(np.max(G - eye * 2.0))
        W = np.exp(kk * (G - mval))
        np.fill_diagonal(W, 0.0)
        S = W.sum() * 0.5
        if S <= 0 or not np.isfinite(S):
            break
        Grad = (W @ x) / S
        dot = (Grad * x).sum(axis=1, keepdims=True)
        Gt = Grad - dot * x
        m1 = b1 * m1 + (1 - b1) * Gt
        v1 = b2 * v1 + (1 - b2) * (Gt * Gt)
        bc1 = 1.0 - b1 ** (s + 1)
        bc2 = 1.0 - b2 ** (s + 1)
        upd = lr * (m1 / bc1) / (np.sqrt(v1 / bc2) + eps)
        x = x - upd
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x = x / nr
        if not np.all(np.isfinite(x)):
            break
    return x

def _lp_polish_deadline(x, deadline, max_pairs=1200, radius0=3e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = radius0
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        cfull = G[iu]
        m = float(cfull.max())
        slack = m - cfull
        order = np.argsort(slack)
        # wide active set: always take top max_pairs to equalize plateau
        thresh = max(8.0 * radius, 2.5e-4)
        cnt_thresh = int(np.count_nonzero(slack < thresh))
        qwant = max(400, min(max_pairs, cnt_thresh))
        qwant = min(qwant, len(order))
        # ensure at least 400, at most max_pairs
        if qwant < 400:
            qwant = min(400, len(order))
        sel_order = order[:qwant]
        # if radius very small, shrink set to tight part for exactness
        if radius < 2e-5:
            tight = sel_order[slack[sel_order] < max(20 * radius, 2e-9)]
            if len(tight) >= 10:
                sel_order = tight
        I = iu[0][sel_order]
        J = iu[1][sel_order]
        cact = cfull[sel_order]
        p = len(sel_order)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cact, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0 or res.x is None or not np.all(np.isfinite(res.x)):
            radius *= 0.5
            continue
        dx = res.x[:-1].reshape(n, d)
        dx = np.clip(dx, -radius, radius)
        improved = False
        # line search on segment to handle curvature
        for alpha in (1.0, 0.5, 0.25, 0.125):
            y = x + alpha * dx
            nr = np.linalg.norm(y, axis=1, keepdims=True)
            nr[nr == 0] = 1.0
            y = y / nr
            v = _max_cos(y)
            if v < best - 1e-15:
                x, best = y, v
                improved = True
                break
        if improved:
            radius = min(radius * 1.4, 0.02)
        else:
            radius *= 0.5
    return x, best

def solve():
    T0 = time.time()
    BUDGET = 168.0
    END = T0 + BUDGET
    rng = np.random.default_rng(7)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.standard_normal(size=(94, 13))
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    try:
        xr = np.load("record.npy").astype(float)
        xr /= np.linalg.norm(xr, axis=1, keepdims=True)
        if _max_cos(xr) < _max_cos(xb):
            xb = xr.copy()
    except Exception:
        pass
    n, d = xb.shape
    state_x = xb.copy()
    state_v = _max_cos(state_x)
    init_v = state_v
    iu0, iu1 = np.triu_indices(n, 1)
    sigmas = [0.03, 0.07, 0.13, 0.22, 0.34, 0.50, 0.75, 1.0]
    it = 0
    # Phase 1: diverse basin hopping + plateau crush
    while time.time() < END - 34.0 and it < 500:
        rmod = it % 20
        if rmod == 19:
            # fresh random basin
            xc = rng.standard_normal((n, d))
            xc /= np.linalg.norm(xc, axis=1, keepdims=True)
            sig = 2.0
        elif rmod == 13 or rmod == 7:
            # worst-point ejection: replace 2-3 points with largest row-max
            xc = state_x.copy()
            G = xc @ xc.T
            np.fill_diagonal(G, -2.0)
            rowmax = G.max(axis=1)
            krep = 3 if rmod == 13 else 2
            idx = np.argpartition(-rowmax, krep)[:krep]
            for ii in idx:
                v = rng.standard_normal(d)
                # try to be far from all: pick best of 8 random tries
                best_v = None
                best_m = 1e9
                for _ in range(8):
                    cand = rng.standard_normal(d)
                    cand /= np.linalg.norm(cand)
                    mm = float(np.max(xc @ cand))
                    if mm < best_m:
                        best_m = mm
                        best_v = cand
                v = best_v
                xc[ii] = v
            # small jitter rest
            xc = xc + rng.standard_normal((n, d)) * 0.01
            xc /= np.linalg.norm(xc, axis=1, keepdims=True)
            sig = 0.5
        else:
            sig = sigmas[it % len(sigmas)]
            xc = state_x + rng.standard_normal((n, d)) * sig
            xc /= np.linalg.norm(xc, axis=1, keepdims=True)
        try:
            if sig >= 0.5 or rmod == 19:
                xc = _adam_quantile(xc, 260, 900, 0.030, 0.003, iu0, iu1, deadline=END - 30.0)
                xc = _adam_quantile(xc, 150, 250, 0.015, 0.002, iu0, iu1, deadline=END - 30.0)
                xc = _adam_lse(xc, 260, 40.0, 900.0, 0.012, 0.002, deadline=END - 30.0)
            elif sig >= 0.2:
                xc = _adam_quantile(xc, 240, 700, 0.025, 0.003, iu0, iu1, deadline=END - 30.0)
                xc = _adam_lse(xc, 240, 60.0, 1100.0, 0.010, 0.0015, deadline=END - 30.0)
            else:
                xc = _adam_quantile(xc, 180, 400, 0.018, 0.002, iu0, iu1, deadline=END - 30.0)
                xc = _adam_lse(xc, 260, 90.0, 1300.0, 0.008, 0.001, deadline=END - 30.0)
        except Exception:
            it += 1
            continue
        try:
            v = _max_cos(xc)
        except Exception:
            it += 1
            continue
        if np.isfinite(v) and v < state_v - 1e-12:
            state_x = xc.copy()
            state_v = v
        it += 1
        if time.time() > END - 34.0:
            break
    # Phase 2: intensify with alternating wide/narrow crush
    for rep in range(7):
        if time.time() > END - 27.0:
            break
        try:
            y = state_x.copy()
            # tiny ejection of single worst point every other rep to change graph
            if rep % 3 == 2:
                G = y @ y.T
                np.fill_diagonal(G, -2.0)
                rm = G.max(axis=1)
                wi = int(np.argmax(rm))
                cand = rng.standard_normal(d)
                cand /= np.linalg.norm(cand)
                y[wi] = (y[wi] * 0.3 + cand * 0.7)
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                y = _adam_quantile(y, 200, 800, 0.020, 0.002, iu0, iu1, deadline=END - 23.0)
            else:
                M = 800 if rep % 2 == 0 else 250
                y = _adam_quantile(y, 320, M, 0.012, 0.0012, iu0, iu1, deadline=END - 23.0)
            y = _adam_lse(y, 420, 80.0, 1600.0, 0.006, 0.0007, deadline=END - 23.0)
            v = _max_cos(y)
            if np.isfinite(v) and v < state_v - 1e-13:
                state_x = y.copy()
                state_v = v
        except Exception:
            continue
    # Final wide LP polish
    if END - time.time() > 7.0:
        try:
            px, pv = _lp_polish_deadline(state_x.copy(), END - 1.0, max_pairs=1200, radius0=3e-3)
            if np.all(np.isfinite(px)) and pv < state_v:
                state_x, state_v = px, pv
        except Exception:
            pass
    state_x /= np.linalg.norm(state_x, axis=1, keepdims=True)
    try:
        x0 = np.load("best.npy").astype(float)
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        v0 = _max_cos(x0)
        vc = _max_cos(state_x)
        if vc > v0 + 1e-12 and state_v > init_v + 1e-12:
            state_x = x0
    except Exception:
        pass
    return state_x
