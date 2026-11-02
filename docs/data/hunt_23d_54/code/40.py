import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _lp_polish(X, seconds, t0, hard):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - t0 > hard:
            break
        G = X @ X.T
        c_all = G[iu]
        m = float(c_all.max())
        sel = c_all > m - max(20 * radius, 1e-9)
        c = c_all[sel]
        I = iu[0][sel]
        J = iu[1][sel]
        p = len(I)
        if p == 0:
            break
        # cap problem size for speed
        if p > 2500:
            # keep most-violated
            idx = np.argpartition(-c, 2500)[:2500]
            I, J, c = I[idx], J[idx], c[idx]
            p = 2500
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols_blocks = np.empty((p, 2 * d + 1), dtype=int)
        cols_blocks[:, :d] = I[:, None] * d + np.arange(d)
        cols_blocks[:, d:2 * d] = J[:, None] * d + np.arange(d)
        cols_blocks[:, 2 * d] = n * d
        cols = cols_blocks.ravel()
        vals_blocks = np.empty((p, 2 * d + 1))
        vals_blocks[:, :d] = X[J]
        vals_blocks[:, d:2 * d] = X[I]
        vals_blocks[:, 2 * d] = -1.0
        vals = vals_blocks.ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            X, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def _lse_smooth(Y, iters, k0, k1, lr):
    n, d = Y.shape
    m1 = np.zeros_like(Y)
    m2 = np.zeros_like(Y)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t = 0
    for it in range(iters):
        k = k0 + (k1 - k0) * (it / max(1, iters - 1))
        G = Y @ Y.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        E = np.exp(k * (G - gmax))
        np.fill_diagonal(E, 0.0)
        s = float(E.sum() * 0.5)
        if s <= 0 or not np.isfinite(s):
            break
        W = E / s
        grad = W @ Y
        dot = (grad * Y).sum(axis=1, keepdims=True)
        g = grad - dot * Y
        t += 1
        m1 = b1 * m1 + (1 - b1) * g
        m2 = b2 * m2 + (1 - b2) * (g * g)
        m1h = m1 / (1 - b1 ** t)
        m2h = m2 / (1 - b2 ** t)
        step = lr * m1h / (np.sqrt(m2h) + eps)
        Y = Y - step
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return Y

def _bm_hinge(Y, t_cap, iters, lr):
    # Burer-Monteiro: minimize sum relu(dot-t)^2 with sphere constraints
    n, d = Y.shape
    m1 = np.zeros_like(Y)
    m2 = np.zeros_like(Y)
    b1, b2, eps = 0.9, 0.999, 1e-8
    st = 0
    for it in range(iters):
        G = Y @ Y.T
        np.fill_diagonal(G, -10.0)
        V = G - t_cap
        V[V < 0] = 0.0
        np.fill_diagonal(V, 0.0)
        # gradient of sum V^2
        grad = V @ Y
        # tangent projection
        dot = (grad * Y).sum(axis=1, keepdims=True)
        g = grad - dot * Y
        ng = float(np.sqrt((g * g).mean())) + 1e-12
        # normalize to keep Adam stable across scales
        if ng > 1e-12:
            g = g / max(ng, 1.0)
        st += 1
        m1 = b1 * m1 + (1 - b1) * g
        m2 = b2 * m2 + (1 - b2) * (g * g)
        m1h = m1 / (1 - b1 ** st)
        m2h = m2 / (1 - b2 ** st)
        step = lr * m1h / (np.sqrt(m2h) + eps)
        Y = Y - step
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    # infeasibility residual: max violation + rms violation
    G = Y @ Y.T
    np.fill_diagonal(G, -10.0)
    viol = G - t_cap
    viol[viol < 0] = 0.0
    maxv = float(viol.max())
    rmsv = float(np.sqrt((viol ** 2).sum() / (Y.shape[0] * (Y.shape[0] - 1) / 2)))
    return Y, maxv, rmsv

def _ap_dykstra(G0, t_cap, iters):
    # Dykstra-corrected alternating projections:
    # C1 = {sym, diag 1, offdiag <= t}, C2 = {PSD rank<=23}
    n = G0.shape[0]
    r = 23
    X = G0.copy()
    P = np.zeros_like(G0)
    Q = np.zeros_like(G0)
    diag_idx = np.diag_indices(n)
    for _ in range(iters):
        # proj C1 of X+P
        Y = X + P
        Y = 0.5 * (Y + Y.T)
        # clip off-diagonal
        # in-place for speed
        np.clip(Y, -1.0, t_cap, out=Y)
        # restore diagonal to 1 (diag clip would set to t)
        Y[diag_idx] = 1.0
        P = X + P - Y
        # proj C2 of Y+Q (rank-r PSD)
        Z_in = Y + Q
        Z_in = 0.5 * (Z_in + Z_in.T)
        w, V = np.linalg.eigh(Z_in)
        # keep largest r, clamped >=0
        idx = np.argsort(w)  # ascending
        wc = np.zeros_like(w)
        top = idx[-r:]
        wc[top] = np.maximum(w[top], 0.0)
        Z = (V * wc) @ V.T
        Z = 0.5 * (Z + Z.T)
        Q = Y + Q - Z
        X = Z
    # final symmetrize
    X = 0.5 * (X + X.T)
    # extract Y factor from top-23 eig of X
    w, V = np.linalg.eigh(X)
    idx = np.argsort(w)
    top = idx[-r:]
    # sort descending for stability
    order = np.argsort(-w[top])
    top = top[order]
    vals = np.maximum(w[top], 1e-12)
    Yf = V[:, top] * np.sqrt(vals)[None, :]
    # Yf is n x r
    # row normalize to sphere
    nr = np.linalg.norm(Yf, axis=1, keepdims=True)
    nr[nr < 1e-12] = 1.0
    Xn = Yf / nr
    # residuals
    off = X.copy()
    off[diag_idx] = -10.0
    maxv = float((off - t_cap).max())
    # rank residual: discarded positive mass + negative mass
    neg = float(np.maximum(-w, 0.0).sum())
    disc = float(np.maximum(w[idx[:-r]], 0.0).sum())
    diag_err = float(np.abs(np.diag(X) - 1.0).max())
    resid = maxv + 0.01 * (neg + disc) + diag_err
    return Xn, X, resid, maxv

def solve():
    import time as _tm
    t_start = _tm.time()
    HARD = 172.0  # leave margin under 180
    np.random.seed(0)
    rng = np.random.default_rng(0)
    n, d = 54, 23
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.standard_normal((n, d))
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (n, d):
        tmp = rng.standard_normal((n, d))
        try:
            flat = xb.ravel()
            tmp.ravel()[:min(flat.size, tmp.size)] = flat[:min(flat.size, tmp.size)]
        except Exception:
            pass
        xb = tmp
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = _max_cos(xb)
    x_best = xb.copy()

    G0 = xb @ xb.T

    # target caps below record: homotopy from near-record downward
    t_list = [best_val - 2.0e-05, 0.04755, 0.04745, 0.04730, 0.04700]
    # clamp to sensible range
    t_list = [float(t) for t in t_list if 0.03 < t < best_val + 1e-12]
    if len(t_list) == 0:
        t_list = [best_val - 2e-05]

    cands = []
    residuals = []

    for ti, t_cap in enumerate(t_list):
        if _tm.time() - t_start > HARD - 70:
            break
        # 1) alternating projections with Dykstra
        iters_ap = 600 if ti < 2 else 400
        try:
            Xn, Gf, resid, maxv = _ap_dykstra(G0, t_cap, iters_ap)
        except Exception:
            continue
        residuals.append((t_cap, resid, maxv))
        # spectral jitter for diversity: small random rotation in ambient space
        # generate 2 variants per t
        for rep in range(2):
            if _tm.time() - t_start > HARD - 65:
                break
            if rep == 0:
                Y0 = Xn.copy()
            else:
                Y0 = Xn + 0.01 * rng.standard_normal((n, d))
                Y0 /= np.linalg.norm(Y0, axis=1, keepdims=True)
            # 2) Burer-Monteiro hinge homotopy
            try:
                Y1, maxv1, rmsv1 = _bm_hinge(Y0, t_cap, iters=500, lr=0.01)
            except Exception:
                Y1 = Y0
                maxv1 = 1.0
            # 3) tangential LSE smoothing toward true min-max
            try:
                Y2 = _lse_smooth(Y1, iters=500, k0=80, k1=600, lr=0.008)
            except Exception:
                Y2 = Y1
            v2 = _max_cos(Y2)
            cands.append((v2, Y2.copy(), t_cap, maxv1))
            if v2 < best_val - 1e-12:
                best_val = v2
                x_best = Y2.copy()
                G0 = x_best @ x_best.T
        # also try pure LSE from perturbed best (keeps search grounded)
        if _tm.time() - t_start > HARD - 60:
            break
        try:
            Yp = x_best + 0.03 * rng.standard_normal((n, d))
            Yp /= np.linalg.norm(Yp, axis=1, keepdims=True)
            Yp = _lse_smooth(Yp, iters=400, k0=60, k1=500, lr=0.01)
            vp = _max_cos(Yp)
            cands.append((vp, Yp.copy(), t_cap, 0.0))
            if vp < best_val - 1e-12:
                best_val = vp
                x_best = Yp.copy()
                G0 = x_best @ x_best.T
        except Exception:
            pass

    # pick most promising candidates by value for short LP-assisted refinement
    # sort by max-cos
    cands.sort(key=lambda z: z[0])
    # refine top few with very short LP polish if time allows
    for k in range(min(3, len(cands))):
        if _tm.time() - t_start > HARD - 35:
            break
        v, Yc, tc, mv = cands[k]
        if v > best_val + 0.005:
            continue
        try:
            Yp = _lp_polish(Yc, seconds=2.0, t0=t_start, hard=HARD - 28)
            vp = _max_cos(Yp)
            if vp < best_val - 1e-12:
                best_val = vp
                x_best = Yp.copy()
        except Exception:
            continue

    # final long polish of global best
    rem = HARD - (_tm.time() - t_start) - 1.0
    if rem > 2:
        rem = min(rem, 24.0)
        try:
            x_best = _lp_polish(x_best, seconds=rem, t0=t_start, hard=HARD - 0.5)
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    # strict feasibility: shrink not needed (sphere), just ensure finite
    x_best = np.where(np.isfinite(x_best), x_best, 0.0)
    nr = np.linalg.norm(x_best, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    x_best = x_best / nr
    return x_best
