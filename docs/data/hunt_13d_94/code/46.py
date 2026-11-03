import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish_deadline(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        cmat = (x @ x.T)[iu]
        m = cmat.max()
        sel = cmat > m - max(20.0 * radius, 1e-9)
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

def _batch_adam(X, steps, k0, k1, lr0, lr1):
    B, N, D = X.shape
    m1 = np.zeros_like(X)
    v1 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    eye = np.eye(N, dtype=X.dtype)
    one_minus_eye = 1.0 - eye
    for s in range(steps):
        frac = s / max(1, steps - 1)
        kk = k0 * (k1 / k0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = np.matmul(X, np.transpose(X, (0, 2, 1)))
        Gm = G - eye[None, :, :] * 2.0
        mval = Gm.max(axis=(1, 2))
        W = np.exp(kk * (G - mval[:, None, None]))
        W = W * one_minus_eye[None, :, :]
        S = W.sum(axis=(1, 2)) * 0.5
        S[~(S > 0) | ~np.isfinite(S)] = 1.0
        Grad = np.matmul(W, X) / S[:, None, None]
        dot = (Grad * X).sum(axis=2, keepdims=True)
        Gt = Grad - dot * X
        m1 = b1 * m1 + (1 - b1) * Gt
        v1 = b2 * v1 + (1 - b2) * (Gt * Gt)
        bc1 = 1.0 - b1 ** (s + 1)
        bc2 = 1.0 - b2 ** (s + 1)
        upd = lr * (m1 / bc1) / (np.sqrt(v1 / bc2) + eps)
        X = X - upd
        nr = np.linalg.norm(X, axis=2, keepdims=True)
        nr[nr == 0] = 1.0
        X = X / nr
        if not np.all(np.isfinite(X)):
            break
    return X

def _single_adam(x, steps, k0, k1, lr0, lr1):
    return _batch_adam(x[None].copy(), steps, k0, k1, lr0, lr1)[0]

def solve():
    T0 = time.time()
    BUDGET = 168.0
    END = T0 + BUDGET
    rng = np.random.default_rng(0)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.standard_normal(size=(94, 13))
    try:
        xr = np.load("record.npy").astype(float)
        xr /= np.linalg.norm(xr, axis=1, keepdims=True)
        xn = xb / np.linalg.norm(xb, axis=1, keepdims=True)
        if _max_cos(xr) < _max_cos(xn):
            xb = xr.copy()
    except Exception:
        pass
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    n, d = xb.shape
    if n != 94 or d != 13:
        # fallback: reshape/truncate safely
        tmp = rng.standard_normal(size=(94, 13))
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        # copy what we can
        try:
            m = min(n, 94)
            tmp[:m, :min(d, 13)] = xb[:m, :min(d, 13)]
            tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
            xb = tmp
        except Exception:
            xb = tmp
        n, d = 94, 13
    state_x = xb.copy()
    state_v = _max_cos(state_x)
    init_v = state_v

    B = 8
    it = 0
    # main basin-hopping loop
    while time.time() < END - 16.0:
        remain = END - 16.0 - time.time()
        if remain <= 0:
            break
        # adapt steps to remaining time
        if remain < 30:
            steps = 600
        elif remain < 60:
            steps = 800
        else:
            steps = 950
        k0 = float(rng.uniform(20.0, 60.0))
        k1 = float(rng.uniform(500.0, 1500.0))
        lr0 = float(rng.uniform(0.015, 0.04))
        lr1 = float(rng.uniform(0.0008, 0.004))
        # build batch of starts from current best
        X0 = np.empty((B, n, d))
        for b in range(B):
            r = rng.random()
            if it % 25 == 24 and b == 0:
                # occasional fully random basin
                cand = rng.standard_normal(size=(n, d))
                cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                X0[b] = cand
            elif r < 0.45:
                sigma = float(rng.uniform(0.08, 0.38))
                cand = state_x + rng.standard_normal(size=(n, d)) * sigma
                cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                X0[b] = cand
            else:
                K = int(rng.integers(10, 30))
                idx = rng.choice(n, size=K, replace=False)
                cand = state_x + rng.standard_normal(size=(n, d)) * 0.02
                rv = rng.standard_normal(size=(K, d))
                rv /= np.linalg.norm(rv, axis=1, keepdims=True)
                cand[idx] = rv
                # extra jitter on a few fixed points to unlock plateau
                if rng.random() < 0.5:
                    j = rng.choice(n, size=int(rng.integers(5, 20)), replace=False)
                    cand[j] += rng.standard_normal(size=(len(j), d)) * 0.08
                cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                X0[b] = cand
        try:
            Xopt = _batch_adam(X0, steps, k0, k1, lr0, lr1)
        except Exception:
            it += 1
            continue
        # score
        for b in range(B):
            try:
                v = _max_cos(Xopt[b])
            except Exception:
                continue
            if v < state_v - 1e-15 and np.all(np.isfinite(Xopt[b])):
                state_x = (Xopt[b] / np.linalg.norm(Xopt[b], axis=1, keepdims=True)).copy()
                state_v = v
        it += 1
        # opportunistic quick single fine-tune if we just improved a lot
        # (skip to save time, rely on batch)

    # final single-config fine tune with high sharpness, low lr
    if time.time() < END - 10.0:
        try:
            rem = END - 10.0 - time.time()
            s2 = 1200 if rem > 12 else 500
            fx = _single_adam(state_x.copy(), s2, 100.0, 1600.0, 0.008, 0.0008)
            fv = _max_cos(fx)
            if fv < state_v and np.all(np.isfinite(fx)):
                state_x, state_v = fx, fv
        except Exception:
            pass
    # final LP polish
    if time.time() < END - 1.5:
        try:
            px, pv = _lp_polish_deadline(state_x.copy(), END - 1.0)
            if pv < state_v and np.all(np.isfinite(px)):
                state_x, state_v = px, pv
        except Exception:
            pass
    state_x /= np.linalg.norm(state_x, axis=1, keepdims=True)
    # safety: never worse than input
    try:
        x0 = np.load("best.npy").astype(float)
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        v0 = _max_cos(x0)
        if _max_cos(state_x) > min(v0, init_v) + 1e-12:
            # revert to whichever is better between x0 and initial
            if v0 <= init_v:
                state_x = x0
            else:
                state_x = xb
    except Exception:
        pass
    state_x /= np.linalg.norm(state_x, axis=1, keepdims=True)
    return state_x
