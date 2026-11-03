import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

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
        S = float(W.sum() * 0.5)
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
    iu = np.triu_indices(n, 1)
    for s in range(steps):
        frac = s / max(1, steps - 1)
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
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

def _lp_polish_deadline(x, deadline, max_pairs=1800):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() < deadline and radius > 1e-14:
        cmat = (x @ x.T)[iu]
        m = float(cmat.max())
        tol = max(20.0 * radius, 1e-9)
        sel = cmat > m - tol
        I = iu[0][sel]; J = iu[1][sel]
        csel = cmat[sel]
        p = len(I)
        if p == 0:
            break
        if p > max_pairs:
            order = np.argpartition(csel, -max_pairs)[-max_pairs:]
            I = I[order]; J = J[order]; csel = csel[order]
            p = max_pairs
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + ar, J[:, None] * d + ar, np.full((p, 1), n * d)], axis=1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x, best

def solve():
    T0 = time.time()
    END = T0 + 170.0
    PHASE1_END = T0 + 122.0
    rng = np.random.default_rng(12345)
    xb = None; bv = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn).astype(float)
            a /= np.linalg.norm(a, axis=1, keepdims=True)
            if a.shape == (94, 13) and np.all(np.isfinite(a)):
                v = _max_cos(a)
                if v < bv:
                    bv = v; xb = a.copy()
        except Exception:
            pass
    if xb is None:
        xb = rng.standard_normal(size=(94, 13))
        xb /= np.linalg.norm(xb, axis=1, keepdims=True)
        bv = _max_cos(xb)
    n, d = 94, 13
    best_x = xb.copy(); best_v = bv
    sigmas = [0.05, 0.10, 0.18, 0.30, 0.45, 0.22, 0.08, 0.60]
    trial = 0
    while time.time() < PHASE1_END:
        sig = sigmas[trial % len(sigmas)]
        trial += 1
        # jitter whole configuration to change topology
        xc = best_x + rng.standard_normal((n, d)) * sig
        xc /= np.linalg.norm(xc, axis=1, keepdims=True)
        # crush plateau: wide quantile then sharp LSE then narrow quantile
        try:
            # vary M slightly to diversify basins
            Mwide = 700 if (trial % 2 == 1) else 500
            xc = _adam_quantile(xc, steps=180, M=Mwide, lr0=0.045, lr1=0.005)
            if time.time() >= PHASE1_END:
                v = _max_cos(xc)
                if v < best_v:
                    best_v = v; best_x = xc.copy()
                break
            xc = _adam_lse(xc, steps=280, k0=200.0, k1=3000.0, lr0=0.022, lr1=0.0025)
            if time.time() >= PHASE1_END:
                v = _max_cos(xc)
                if v < best_v:
                    best_v = v; best_x = xc.copy()
                break
            Mnarr = 250 if (trial % 3 != 0) else 120
            xc = _adam_quantile(xc, steps=90, M=Mnarr, lr0=0.015, lr1=0.003)
            v = _max_cos(xc)
        except Exception:
            continue
        if v < best_v - 1e-12:
            best_v = v; best_x = xc.copy()
        # occasional extra sharpening on promising candidates
        if v < best_v + 0.002 and time.time() < PHASE1_END - 5:
            try:
                xc2 = _adam_lse(xc, steps=120, k0=800.0, k1=6000.0, lr0=0.008, lr1=0.002)
                v2 = _max_cos(xc2)
                if v2 < best_v - 1e-12:
                    best_v = v2; best_x = xc2.copy()
            except Exception:
                pass
    # final LP tying to create rigid contacts
    try:
        if time.time() < END - 3:
            best_x, best_v = _lp_polish_deadline(best_x, min(END - 8, time.time() + 30))
    except Exception:
        pass
    # final high-k smoothing + short LP
    try:
        if time.time() < END - 6:
            best_x = _adam_lse(best_x, steps=150, k0=1000.0, k1=8000.0, lr0=0.006, lr1=0.0015)
            bv2 = _max_cos(best_x)
            if bv2 < best_v:
                best_v = bv2
            if time.time() < END - 2:
                best_x, best_v = _lp_polish_deadline(best_x, END - 1.0, max_pairs=1500)
    except Exception:
        pass
    best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
    # safety: return incumbent if something went wrong
    try:
        fv = _max_cos(best_x)
        if not np.all(np.isfinite(best_x)) or best_x.shape != (94, 13):
            return xb
        # keep best known even if no improvement (never regress)
        if fv > bv + 1e-12:
            return xb
        return best_x
    except Exception:
        return xb
