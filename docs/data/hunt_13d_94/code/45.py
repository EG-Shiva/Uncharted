import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _adam_lse(x, steps, k0, k1, lr0, lr1):
    n, d = x.shape
    m1 = np.zeros_like(x)
    v1 = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for s in range(steps):
        frac = s / max(1, steps - 1)
        kk = k0 * (k1 / k0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        mval = float(np.max(G - np.eye(n) * 2.0))
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

def _adam_quantile(x, steps, M, lr0, lr1):
    n, d = x.shape
    m1 = np.zeros_like(x)
    v1 = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for s in range(steps):
        frac = s / max(1, steps - 1)
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        # dynamic threshold = M-th largest upper-tri value
        iu = np.triu_indices(n, 1)
        vals = G[iu]
        if M >= vals.size:
            t = float(vals.min())
        else:
            t = float(np.partition(vals, -M)[-M])
        Mm = G - t
        Mm[Mm < 0] = 0.0
        np.fill_diagonal(Mm, 0.0)
        cnt = float((Mm > 0).sum() * 0.5)
        if cnt < 1:
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
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel() if False else np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
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
    rng = np.random.default_rng(1)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.standard_normal(size=(94, 13))
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    n, d = xb.shape
    state_x = xb.copy()
    state_v = _max_cos(state_x)
    init_v = state_v

    def remain():
        return END - time.time()

    # Phase 1: diverse full-config jitter + short anneal (new basins, not cluster-fixed)
    try:
        sigmas = [0.06, 0.12, 0.22, 0.35, 0.55]
        cands = []
        # include current state itself
        for rep in range(3):
            for sig in sigmas:
                if remain() < 30:
                    break
                xc = state_x + rng.standard_normal((n, d)) * sig
                xc /= np.linalg.norm(xc, axis=1, keepdims=True)
                xc = _adam_lse(xc, 500, 30.0, 350.0, 0.02, 0.003)
                v = _max_cos(xc)
                cands.append((v, xc))
                if v < state_v - 1e-12:
                    state_x = xc.copy(); state_v = v
            if remain() < 30:
                break
        # a few fully random starts for topology diversity
        for _ in range(4):
            if remain() < 28:
                break
            xc = rng.standard_normal((n, d))
            xc /= np.linalg.norm(xc, axis=1, keepdims=True)
            xc = _adam_lse(xc, 600, 25.0, 400.0, 0.025, 0.003)
            v = _max_cos(xc)
            if v < state_v - 1e-12:
                state_x = xc.copy(); state_v = v
    except Exception:
        pass

    # Phase 2: plateau crush on current best: jointly push top-M down, then sharpen
    try:
        if remain() > 25:
            xp = state_x.copy()
            # quantile phase targets ~600-pair plateau seen in diagnosis
            xp = _adam_quantile(xp, 700, 600, 0.02, 0.002)
            vp = _max_cos(xp)
            # sharpen with high-k LSE
            xp = _adam_lse(xp, 600, 80.0, 900.0, 0.008, 0.001)
            vp2 = _max_cos(xp)
            if vp2 < state_v - 1e-15 and np.all(np.isfinite(xp)):
                state_x = xp.copy(); state_v = vp2
            elif vp < state_v - 1e-15:
                # if sharpen hurt, keep quantile result + light sharpen
                xp2 = _adam_lse(xp, 250, 150.0, 900.0, 0.004, 0.001)
                vq = _max_cos(xp2)
                if vq < state_v - 1e-15:
                    state_x = xp2.copy(); state_v = vq
    except Exception:
        pass

    # Phase 3: basin-hopping kicks: random subset reinit + anneal
    try:
        it = 0
        while remain() > 20 and it < 40:
            it += 1
            K = int(rng.integers(12, 28))
            idx = rng.choice(n, size=K, replace=False)
            xc = state_x.copy()
            xc[idx] = rng.standard_normal((K, d))
            xc[idx] /= np.linalg.norm(xc[idx], axis=1, keepdims=True)
            # anneal from moderate k (keeps plateau together)
            xc = _adam_lse(xc, 550, 35.0, 500.0, 0.015, 0.002)
            # quantile squeeze
            xc = _adam_quantile(xc, 250, 500, 0.01, 0.002)
            # final sharpen
            xc = _adam_lse(xc, 300, 150.0, 1000.0, 0.005, 0.001)
            v = _max_cos(xc)
            if v < state_v - 1e-12:
                state_x = xc.copy(); state_v = v
                # extra exploitation around new basin
                try:
                    xp = _adam_quantile(state_x.copy(), 300, 700, 0.008, 0.0015)
                    xp = _adam_lse(xp, 300, 100.0, 1000.0, 0.005, 0.001)
                    vp = _max_cos(xp)
                    if vp < state_v - 1e-15:
                        state_x = xp.copy(); state_v = vp
                except Exception:
                    pass
    except Exception:
        pass

    # Phase 4: final polish of best with high-k + LP
    try:
        if remain() > 8:
            xf = _adam_lse(state_x.copy(), 400, 200.0, 1200.0, 0.004, 0.0008)
            vf = _max_cos(xf)
            if vf < state_v and np.all(np.isfinite(xf)):
                state_x, state_v = xf, vf
    except Exception:
        pass
    if remain() > 4:
        try:
            px, pv = _lp_polish_deadline(state_x.copy(), END - 1.0)
            if pv < state_v and np.all(np.isfinite(px)):
                state_x, state_v = px, pv
        except Exception:
            pass

    state_x /= np.linalg.norm(state_x, axis=1, keepdims=True)
    try:
        x0 = np.load("best.npy").astype(float)
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        v0 = _max_cos(x0)
        if _max_cos(state_x) > min(v0, init_v) + 1e-12:
            # revert to whichever is better between x0 and init
            if v0 <= init_v:
                state_x = x0
            else:
                state_x = xb
    except Exception:
        pass
    return state_x
