import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def adam_quench(X, steps, b0, b1, lr0, lr1):
    n, d = X.shape
    m1 = np.zeros_like(X)
    m2 = np.zeros_like(X)
    b_1 = 0.9; b_2 = 0.999; eps = 1e-8
    lb0, lb1 = np.log(b0), np.log(b1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        f = s / max(1, steps - 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        sw = float(W.sum())
        if not np.isfinite(sw) or sw == 0:
            break
        W /= sw
        Grad = W @ X
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        m1 = b_1 * m1 + (1 - b_1) * Gt
        m2 = b_2 * m2 + (1 - b_2) * (Gt * Gt)
        c1 = 1.0 - b_1 ** (s + 1)
        c2 = 1.0 - b_2 ** (s + 1)
        step = lr * (m1 / c1) / (np.sqrt(m2 / c2) + eps)
        sdot = np.sum(step * X, axis=1, keepdims=True)
        step = step - sdot * X
        sn = np.linalg.norm(step, axis=1, keepdims=True)
        over = sn > 0.05
        scale = np.ones_like(sn)
        scale[over] = 0.05 / (sn[over] + 1e-12)
        step = step * scale
        X = X - step
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X = X / nr
    return X

def lp_polish(X, seconds, t_deadline=None):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = max_cos(X)
    radius = 5e-4
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    it = 0
    while True:
        if time.time() - start >= seconds:
            break
        if t_deadline is not None and time.time() >= t_deadline:
            break
        if radius < 1e-13:
            break
        it += 1
        if it > 25:
            break
        G = X @ X.T
        c = G[iu0]
        m = float(c.max())
        window = max(20.0 * radius, 4e-5)
        order = np.argsort(-c)
        sel_list = []
        for idx in order:
            if c[idx] > m - window and len(sel_list) < 500:
                sel_list.append(int(idx))
            elif c[idx] <= m - window:
                break
            if len(sel_list) >= 500:
                break
        if len(sel_list) == 0:
            break
        sel = np.array(sel_list)
        I = iu0[0][sel]; J = iu0[1][sel]
        p = len(sel); nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                          options={"time_limit": 2.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        y = y / nr
        v = max_cos(y)
        if v < best - 1e-15:
            X = y; best = v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return X

def gram_ap_to_vecs(G0, t, rng, iters=450, sigma=0.02):
    n = G0.shape[0]
    if sigma > 0:
        R = rng.standard_normal((n, n))
        R = (R + R.T) * 0.5
        np.fill_diagonal(R, 0.0)
        G = G0 + sigma * R
        G = (G + G.T) * 0.5
        np.fill_diagonal(G, 1.0)
    else:
        G = G0.copy()
        np.fill_diagonal(G, 1.0)
    best_score = 1e9
    best_G = None
    best_V = None
    best_w = None
    for k in range(iters):
        Gc = np.minimum(G, t)
        np.fill_diagonal(Gc, 1.0)
        # symmetrize (minimum preserves symmetry)
        Gs = (Gc + Gc.T) * 0.5
        try:
            w, V = np.linalg.eigh(Gs)
        except Exception:
            break
        V22 = V[:, -22:]
        w22 = w[-22:]
        w22p = np.maximum(w22, 0.0)
        # avoid all-zero
        w22p = np.maximum(w22p, 1e-12)
        G = (V22 * w22p) @ V22.T
        G = (G + G.T) * 0.5
        # normalized max for tracking
        dg = np.diag(G).copy()
        dg = np.maximum(dg, 1e-6)
        D = np.sqrt(dg)
        # off-max normalized
        Gn = G / D[:, None] / D[None, :]
        np.fill_diagonal(Gn, -2.0)
        off = float(Gn.max())
        derr = float(np.max(np.abs(dg - 1.0)))
        score = off + 1.0 * derr
        if score < best_score:
            best_score = score
            best_G = G.copy()
            best_V = V22.copy()
            best_w = w22p.copy()
    outs = []
    for (V22, w22p) in [(best_V, best_w), (V22, np.maximum(w22, 0.0))]:
        if V22 is None or w22p is None:
            continue
        sw = np.maximum(w22p, 1e-12)
        X = V22 * np.sqrt(sw)[None, :]
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X = X / nr
        outs.append(X)
    return outs

def solve():
    t0 = time.time()
    TIME_LIMIT = 165.0
    DEADLINE = t0 + TIME_LIMIT
    rng = np.random.default_rng(12345)
    n, d = 58, 22
    best = None; best_val = 1e9
    for f in ("best.npy", "record.npy"):
        try:
            v = np.load(f).astype(float)
            if v.shape != (58, 22):
                continue
            v = v / np.linalg.norm(v, axis=1, keepdims=True)
            c = max_cos(v)
            if c < best_val:
                best_val = c; best = v.copy()
        except Exception:
            continue
    if best is None:
        z = rng.standard_normal((n, d)); z /= np.linalg.norm(z, axis=1, keepdims=True)
        best = z; best_val = max_cos(z)
    incumbent = best.copy()
    rec = 0.066624028492
    targets = [rec - 1e-5, rec - 2e-5, rec - 5e-5, rec - 1e-4]
    sigmas = [0.0, 0.015, 0.035]
    G0 = incumbent @ incumbent.T
    cands = []
    # Gram alternating projections: cheap, run first
    for t in targets:
        for sg in sigmas:
            if time.time() > t0 + 45.0:
                break
            try:
                outs = gram_ap_to_vecs(G0, t, rng, iters=400, sigma=sg)
            except Exception:
                continue
            for Xc in outs:
                try:
                    v = max_cos(Xc)
                except Exception:
                    continue
                cands.append((v, Xc))
        if time.time() > t0 + 45.0:
            break
    # add incumbent as candidate
    cands.append((best_val, incumbent.copy()))
    # quick quench screening of Gram candidates
    # sort by raw value, take top ~8 for quench
    cands.sort(key=lambda t: t[0])
    screened = []
    NQ = min(8, len(cands))
    for i in range(NQ):
        if time.time() > DEADLINE - 60.0:
            break
        _, Xc = cands[i]
        y = Xc.copy()
        try:
            y = adam_quench(y, 450, 25.0, 500.0, 0.005, 0.0006)
        except Exception:
            continue
        v = max_cos(y)
        screened.append((v, y))
    # also include incumbent quenched lightly if time
    screened.append((best_val, best.copy()))
    screened.sort(key=lambda t: t[0])
    # update best from screening (before LP)
    for v, y in screened:
        if v < best_val:
            best_val = v; best = y.copy()
    # medium quench + LP on top 3
    top = screened[:3]
    for idx in range(len(top)):
        if time.time() > DEADLINE - 25.0:
            break
        v0, y0 = top[idx]
        y = y0.copy()
        try:
            # sharp finish
            y = adam_quench(y, 350, 600.0, 2500.0, 0.0012, 0.00025)
        except Exception:
            pass
        v = max_cos(y)
        # LP polish with small budget
        try:
            budget = 4.0 if idx == 0 else 2.0
            budget = min(budget, max(0.5, DEADLINE - 5.0 - time.time()))
            if budget > 0.5:
                y2 = lp_polish(y, budget, DEADLINE - 3.0)
                v2 = max_cos(y2)
                if v2 < v:
                    y, v = y2, v2
        except Exception:
            pass
        if v < best_val - 1e-15:
            best_val = v; best = y.copy()
    # final safety: ensure unit norm, no nan
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    best = np.nan_to_num(best, nan=0.0, posinf=0.0, neginf=0.0)
    nr = np.linalg.norm(best, axis=1, keepdims=True)
    bad = (nr.squeeze() < 1e-8)
    if np.any(bad):
        best[bad] = incumbent[bad]
        best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
