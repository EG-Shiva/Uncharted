import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

# ---------- C13 cyclic shift family: 9x13+6=123 ----------
def _expand_c13(R, S):
    X = np.empty((123, 13), dtype=float)
    for o in range(9):
        r = R[o]
        base = o * 13
        # manual roll via indexing is fast
        for k in range(13):
            # roll(r,k)[j]=r[j-k]
            X[base + k] = np.roll(r, k)
    X[117:] = S
    return X

def _pullback_c13(Gt):
    Rg = np.zeros((9, 13), dtype=float)
    for o in range(9):
        base = o * 13
        acc = np.zeros(13, dtype=float)
        for k in range(13):
            acc += np.roll(Gt[base + k], -k)
        Rg[o] = acc / 13.0
    Sg = Gt[117:].copy()
    return Rg, Sg

def _sym_anneal_c13(R0, S0, steps, b0, b1, lr0, lr1, mu, t0, t_lim):
    R = _norm_rows(R0.copy())
    S = _norm_rows(S0.copy())
    vR = np.zeros_like(R)
    vS = np.zeros_like(S)
    for s in range(steps):
        if s % 100 == 0 and time.time() - t0 > t_lim:
            break
        f = s / max(steps - 1, 1)
        beta = b0 * (b1 / b0) ** f
        lr = lr0 * (lr1 / lr0) ** f
        X = _expand_c13(R, S)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        gmax = float(G.max())
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        row = W.sum(axis=1, keepdims=True) + 1e-12
        grad = W @ X
        dot = np.sum(grad * X, axis=1, keepdims=True)
        gt = (grad - dot * X) / row
        Rg, Sg = _pullback_c13(gt)
        vR = mu * vR + Rg
        vS = mu * vS + Sg
        R = R - lr * vR
        S = S - lr * vS
        R = _norm_rows(R)
        S = _norm_rows(S)
    X = _expand_c13(R, S)
    X = _norm_rows(X)
    return R, S, X

# ---------- C12 rotation family: 10x12+3=123, 6 planes + axis ----------
_C12 = np.pi * 2.0 / 12.0
_COS12 = np.cos(np.arange(12) * _C12)
_SIN12 = np.sin(np.arange(12) * _C12)

def _expand_c12(R, S):
    X = np.empty((123, 13), dtype=float)
    idx = 0
    for o in range(10):
        r = R[o]
        a = r[0]
        xs = r[1::2].copy()
        ys = r[2::2].copy()
        for m in range(12):
            c = _COS12[m]; sn = _SIN12[m]
            rp0 = a
            nx = c * xs - sn * ys
            ny = sn * xs + c * ys
            X[idx, 0] = rp0
            X[idx, 1::2] = nx
            X[idx, 2::2] = ny
            idx += 1
    X[120:] = S
    return X

def _pullback_c12(Gt):
    Rg = np.zeros((10, 13), dtype=float)
    for o in range(10):
        blk = Gt[o * 12:(o + 1) * 12]
        Rg[o, 0] = blk[:, 0].sum() / 12.0
        ax = np.zeros(6); ay = np.zeros(6)
        for m in range(12):
            c = _COS12[m]; s = _SIN12[m]
            gx = blk[m, 1::2]; gy = blk[m, 2::2]
            ax += c * gx + s * gy
            ay += -s * gx + c * gy
        Rg[o, 1::2] = ax / 12.0
        Rg[o, 2::2] = ay / 12.0
    Sg = Gt[120:].copy()
    return Rg, Sg

def _sym_anneal_c12(R0, S0, steps, b0, b1, lr0, lr1, mu, t0, t_lim):
    R = _norm_rows(R0.copy())
    S = _norm_rows(S0.copy())
    vR = np.zeros_like(R)
    vS = np.zeros_like(S)
    for s in range(steps):
        if s % 100 == 0 and time.time() - t0 > t_lim:
            break
        f = s / max(steps - 1, 1)
        beta = b0 * (b1 / b0) ** f
        lr = lr0 * (lr1 / lr0) ** f
        X = _expand_c12(R, S)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        gmax = float(G.max())
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        row = W.sum(axis=1, keepdims=True) + 1e-12
        grad = W @ X
        dot = np.sum(grad * X, axis=1, keepdims=True)
        gt = (grad - dot * X) / row
        Rg, Sg = _pullback_c12(gt)
        vR = mu * vR + Rg
        vS = mu * vS + Sg
        R -= lr * vR
        S -= lr * vS
        R = _norm_rows(R)
        S = _norm_rows(S)
    X = _expand_c12(R, S)
    return R, S, _norm_rows(X)

def _smooth_anneal(v0, steps, b0, b1, lr0, lr1, mu=0.85):
    v = v0.copy()
    n, d = v.shape
    vel = np.zeros_like(v)
    for s in range(steps):
        f = s / max(steps - 1, 1)
        beta = b0 * (b1 / b0) ** f
        lr = lr0 * (lr1 / lr0) ** f
        G = v @ v.T
        np.fill_diagonal(G, -10.0)
        gmax = float(G.max())
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        row = W.sum(axis=1, keepdims=True) + 1e-12
        grad = W @ v
        dot = np.sum(grad * v, axis=1, keepdims=True)
        gt = (grad - dot * v) / row
        vel = mu * vel + gt
        v = v - lr * vel
        v = _norm_rows(v)
    return v

def _lp_polish(x, seconds, p_cap=1600):
    x = _norm_rows(x)
    n, d = x.shape
    best = _max_cos(x)
    bx = x.copy()
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
            er = np.repeat(np.arange(n), d)
            ec = np.arange(n * d)
            A_eq = coo_matrix((x.ravel(), (er, ec)), shape=(n, nv)).tocsr()
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
        y = _norm_rows(y)
        v = _max_cos(y)
        if v < best:
            best = v
            bx = y.copy()
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return bx

def solve():
    t0 = time.time()
    rng = np.random.RandomState(0)
    n, d = 123, 13
    x_best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = _norm_rows(a)
                v = _max_cos(a)
                if v < best_val:
                    best_val = v
                    x_best = a.copy()
        except Exception:
            pass
    if x_best is None:
        x_best = _norm_rows(rng.randn(n, d))
        best_val = _max_cos(x_best)

    best_sym = None
    best_sym_val = 1e9

    # C13 symmetric search (reduced variables)
    try:
        for trial in range(3):
            if time.time() - t0 > 55:
                break
            R0 = rng.randn(9, 13); S0 = rng.randn(6, 13)
            R0 = _norm_rows(R0); S0 = _norm_rows(S0)
            if trial == 0:
                steps, b0, b1, l0, l1 = 500, 30.0, 400.0, 0.02, 0.002
            elif trial == 1:
                steps, b0, b1, l0, l1 = 500, 60.0, 600.0, 0.015, 0.0015
            else:
                steps, b0, b1, l0, l1 = 400, 20.0, 300.0, 0.025, 0.003
            R, S, X = _sym_anneal_c13(R0, S0, steps, b0, b1, l0, l1, 0.85, t0, 58)
            v = _max_cos(X)
            if v < best_sym_val:
                best_sym_val = v
                best_sym = X.copy()
    except Exception:
        pass

    # C12 symmetric search
    try:
        for trial in range(2):
            if time.time() - t0 > 85:
                break
            R0 = rng.randn(10, 13); S0 = rng.randn(3, 13)
            R0 = _norm_rows(R0); S0 = _norm_rows(S0)
            R, S, X = _sym_anneal_c12(R0, S0, 450, 30.0, 400.0, 0.02, 0.002, 0.85, t0, 88)
            v = _max_cos(X)
            if v < best_sym_val:
                best_sym_val = v
                best_sym = X.copy()
    except Exception:
        pass

    # symmetry breaking: perturb best symmetric and relax unrestricted
    try:
        if best_sym is not None:
            for pert, st in [(0.003, 600), (0.008, 600)]:
                if time.time() - t0 > 120:
                    break
                Xt = _norm_rows(best_sym + pert * rng.randn(n, d))
                Y = _smooth_anneal(Xt, st, 40.0, 500.0, 0.008, 0.0008, 0.85)
                v = _max_cos(Y)
                if v < best_sym_val:
                    best_sym_val = v
                    best_sym = Y.copy()
                if v < best_val:
                    best_val = v
                    x_best = Y.copy()
            # keep symmetric itself if it happens to be good
            if best_sym_val < best_val:
                best_val = best_sym_val
                x_best = best_sym.copy()
    except Exception:
        pass

    # small unrestricted shakes from record best to try other basin
    try:
        for k in range(2):
            if time.time() - t0 > 135:
                break
            Xt = _norm_rows(x_best + (0.005 + 0.005 * k) * rng.randn(n, d))
            Y = _smooth_anneal(Xt, 500, 40.0, 500.0, 0.007, 0.0007, 0.85)
            v = _max_cos(Y)
            if v < best_val:
                best_val = v
                x_best = Y.copy()
    except Exception:
        pass

    remain = 168 - (time.time() - t0)
    if remain > 6:
        try:
            x_best = _lp_polish(x_best, min(remain - 2, 28.0))
        except Exception:
            pass
    x_best = _norm_rows(x_best)
    return x_best
