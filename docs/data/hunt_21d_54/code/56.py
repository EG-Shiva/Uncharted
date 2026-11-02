import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _proj(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _geodesic(X, D, s):
    # X (n,d) unit, D (n,d) tangent, s global scale
    # Dn scaled so Frobenius = sqrt(n) => avg per-point |d|=1
    fn = np.linalg.norm(D) + 1e-15
    n = X.shape[0]
    Dn = D / fn * np.sqrt(n)
    nr = np.linalg.norm(Dn, axis=1, keepdims=True) + 1e-15
    ang = s * nr
    ca = np.cos(ang)
    sa = np.sin(ang)
    U = Dn / nr
    Y = ca * X + sa * U
    return _proj(Y)

def _repair(Y, steps=350, K0=80.0, K1=800.0, lr0=0.08, mu=0.80):
    X = Y.copy()
    n, d = X.shape
    vel = np.zeros_like(X)
    for t in range(steps):
        frac = t / steps
        K = K0 * (K1 / K0) ** frac
        lr = lr0 * (1.0 - 0.5 * frac)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = (G - m) * K
        np.clip(E, -20.0, 0.0, out=E)
        W = np.exp(E)
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        grad = W @ X
        dot = np.einsum('ij,ij->i', grad, X)
        grad -= dot[:, None] * X
        grad /= rs
        vel = mu * vel + grad
        X = X - lr * vel
        X = _proj(X)
    return X

def _lp_polish(x, seconds, time_limit_per_solve=2.0):
    x = _proj(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 2500:
            idx = np.argpartition(-c[sel], 2500)[:2500]
            I, J = I[idx], J[idx]
            csv = c[sel][idx]
        else:
            csv = c[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csv, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs",
                          options={"time_limit": time_limit_per_solve})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _proj(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x

def solve():
    T0 = time.time()
    HARD = 172.0
    def left():
        return HARD - (time.time() - T0)
    rng = np.random.RandomState(12345)
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.randn(54, 21)
    xb = np.asarray(xb, dtype=np.float64)
    if xb.shape != (54, 21):
        flat = xb.ravel()
        need = 54 * 21
        if flat.size < need:
            extra = rng.randn(need - flat.size)
            flat = np.concatenate([flat, extra])
        xb = flat[:need].reshape(54, 21)
    xb = _proj(np.nan_to_num(xb))
    n, d = 54, 21
    best = xb.copy()
    best_val = _max_cos(best)
    # quick record fallback
    try:
        r = _proj(np.asarray(np.load("record.npy"), dtype=np.float64))
        if r.shape == (54, 21):
            rv = _max_cos(r)
            if rv < best_val:
                best, best_val = r.copy(), rv
    except Exception:
        pass

    X0 = best.copy()
    m0 = best_val
    # ---- build active-contact Jacobian, SVD capped ----
    nmodes = 15
    modes = None
    try:
        svd_deadline = 20.0
        s_mark = time.time()
        G0 = X0 @ X0.T
        iu_all = np.triu_indices(n, 1)
        c_all = G0[iu_all]
        # slack < 1e-4
        selA = c_all > m0 - 1e-4
        IA, JA = iu_all[0][selA], iu_all[1][selA]
        ca = c_all[selA]
        p = len(IA)
        if p > 0 and p < 5000 and (time.time() - s_mark) < svd_deadline:
            N = n * d
            A = np.zeros((p, N), dtype=np.float64)
            # rows: block i gets x_j - c*x_i
            for k in range(p):
                i = int(IA[k]); j = int(JA[k]); cc = float(ca[k])
                xi = X0[i]; xj = X0[j]
                A[k, i*d:(i+1)*d] = xj - cc * xi
                A[k, j*d:(j+1)*d] = xi - cc * xj
            MtM = A.T @ A
            lam = 10.0
            for i in range(n):
                xi = X0[i]
                MtM[i*d:(i+1)*d, i*d:(i+1)*d] += lam * np.outer(xi, xi)
                if time.time() - s_mark > svd_deadline:
                    break
            if time.time() - s_mark < svd_deadline:
                w, V = np.linalg.eigh(MtM)
                # V columns ascending
                modes = V[:, :nmodes]  # N x k
                del A, MtM
    except Exception:
        modes = None
    if modes is None:
        # fallback random tangent modes
        modes = rng.randn(n*d, nmodes)
        for k in range(nmodes):
            v = modes[:, k].reshape(n, d)
            v = v - (v * X0).sum(axis=1, keepdims=True) * X0
            modes[:, k] = v.ravel() / (np.linalg.norm(v) + 1e-12)

    # original contact mask at absolute level m0-1e-4 for Hamming
    Gref = X0 @ X0.T
    cref = Gref[iu_all]
    mask_ref = cref > m0 - 1e-4

    amps = [0.02, 0.06, 0.12, 0.22, 0.32]
    n_trials = 55
    repairs_done = 0
    max_repairs = 28
    # pre-reshape modes for speed
    M3 = [modes[:, k].reshape(n, d).copy() for k in range(nmodes)]
    # ensure tangent
    for k in range(nmodes):
        v = M3[k]
        v -= (v * X0).sum(axis=1, keepdims=True) * X0

    for trial in range(n_trials):
        if left() < 30:
            break
        if repairs_done >= max_repairs and left() < 40:
            break
        coeff = rng.randn(nmodes)
        coeff /= (np.linalg.norm(coeff) + 1e-12)
        D = np.zeros((n, d))
        for k in range(nmodes):
            D += coeff[k] * M3[k]
        # re-project to current best tangent? use X0 tangent then also project to best tangent later
        # use best as base for moves (update base when best improves)
        base = best
        # project D onto base tangent
        D = D - (D * base).sum(axis=1, keepdims=True) * base
        dn = np.linalg.norm(D) + 1e-15
        if dn < 1e-12:
            continue
        for s in amps:
            if left() < 28:
                break
            Y = _geodesic(base, D, s)
            vY = _max_cos(Y)
            if vY > 0.15:
                continue
            # Hamming distance at absolute level
            GY = Y @ Y.T
            cY = GY[iu_all]
            maskY = cY > m0 - 1e-4
            ham = int(np.count_nonzero(maskY != mask_ref))
            if ham < 40:
                continue
            # accept for repair
            if repairs_done >= max_repairs:
                continue
            # repair budget 200-500 steps depending on time
            steps = 350 if left() > 60 else 220
            Yr = _repair(Y, steps=steps)
            repairs_done += 1
            vr = _max_cos(Yr)
            if vr < best_val - 1e-13:
                # only LP-polish if already below 0.07
                if vr < 0.07 and left() > 20:
                    Yp = _lp_polish(Yr, min(4.0, left() - 12))
                    vp = _max_cos(Yp)
                    if vp < vr:
                        Yr, vr = Yp, vp
                if vr < best_val:
                    best, best_val = Yr, vr
                    # update reference? keep m0 fixed for Hamming, but update base automatically
                    # refresh modes tangent projection lazily (keep same modes)
            if left() < 25:
                break
        # occasional refresh of modes tangent to new best if improved a lot? skip for time

    # final squeeze with remaining time
    if left() > 9:
        try:
            best = _lp_polish(best, min(8.0, left() - 2.0))
        except Exception:
            pass
    best = _proj(best)
    return best
