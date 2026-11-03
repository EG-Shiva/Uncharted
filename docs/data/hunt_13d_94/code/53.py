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
        if s % 100 == 0 and time.time() > deadline:
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
        if s % 120 == 0 and time.time() > deadline:
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

def _lp_polish_deadline(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        cmat = (x @ x.T)[iu]
        m = cmat.max()
        sel = cmat > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
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
            res = linprog(obj, A_ub=A_ub, b_ub=-cmat[sel], A_eq=A_eq, b_eq=np.zeros(n),
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

    sigmas = [0.05, 0.10, 0.18, 0.30, 0.45, 0.65, 0.90]
    it = 0
    # Phase 1: diverse jitter + plateau crush + sharpen
    while time.time() < END - 32.0 and it < 400:
        sig = sigmas[it % len(sigmas)]
        # occasionally fresh random start to find new basin
        if it % 17 == 16:
            xc = rng.standard_normal((n, d))
            xc /= np.linalg.norm(xc, axis=1, keepdims=True)
        else:
            xc = state_x + rng.standard_normal((n, d)) * sig
            xc /= np.linalg.norm(xc, axis=1, keepdims=True)
        # crush wide plateau first
        try:
            if sig >= 0.3:
                xc = _adam_quantile(xc, 280, 650, 0.030, 0.003, iu0, iu1, deadline=END - 28.0)
                xc = _adam_quantile(xc, 160, 180, 0.015, 0.002, iu0, iu1, deadline=END - 28.0)
                xc = _adam_lse(xc, 280, 40.0, 900.0, 0.012, 0.002, deadline=END - 28.0)
            elif sig >= 0.15:
                xc = _adam_quantile(xc, 260, 550, 0.025, 0.003, iu0, iu1, deadline=END - 28.0)
                xc = _adam_lse(xc, 260, 60.0, 1000.0, 0.010, 0.0015, deadline=END - 28.0)
            else:
                xc = _adam_quantile(xc, 200, 350, 0.018, 0.002, iu0, iu1, deadline=END - 28.0)
                xc = _adam_lse(xc, 300, 90.0, 1200.0, 0.008, 0.001, deadline=END - 28.0)
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

    # Phase 2: intensify best with alternating crush/sharpen
    for rep in range(6):
        if time.time() > END - 26.0:
            break
        try:
            y = state_x.copy()
            M = 700 if rep % 2 == 0 else 250
            y = _adam_quantile(y, 350, M, 0.012, 0.0015, iu0, iu1, deadline=END - 22.0)
            y = _adam_lse(y, 450, 80.0, 1500.0, 0.006, 0.0008, deadline=END - 22.0)
            v = _max_cos(y)
            if np.isfinite(v) and v < state_v - 1e-13:
                state_x = y.copy()
                state_v = v
        except Exception:
            continue

    # Final LP polish
    if END - time.time() > 6.0:
        try:
            px, pv = _lp_polish_deadline(state_x.copy(), END - 1.0)
            if np.all(np.isfinite(px)) and pv < state_v:
                state_x, state_v = px, pv
        except Exception:
            pass
    state_x /= np.linalg.norm(state_x, axis=1, keepdims=True)
    try:
        x0 = np.load("best.npy").astype(float)
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        if _max_cos(state_x) > _max_cos(x0) + 1e-12 and state_v > init_v + 1e-12:
            state_x = x0
    except Exception:
        pass
    return state_x
