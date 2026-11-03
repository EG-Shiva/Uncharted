import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _rowmax(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1), G

def _smooth_opt(X0, beta, maxiter, maxcor=10):
    n, d = X0.shape
    Y0 = X0.copy()
    def fg(Yflat):
        Y = Yflat.reshape(n, d)
        nrm = np.linalg.norm(Y, axis=1, keepdims=True)
        nrm = np.maximum(nrm, 1e-12)
        X = Y / nrm
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = float(np.max(G))
        D = beta * (G - M)
        np.clip(D, -30.0, 0.0, out=D)
        E = np.empty_like(D)
        np.exp(D, out=E)
        np.fill_diagonal(E, 0.0)
        S = float(np.sum(E))
        if S <= 0 or not np.isfinite(S):
            S = 1e-300
        loss = M + np.log(S) / beta
        W = E / S
        GX = 2.0 * (W @ X)
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GY = (GX - dot * X) / nrm
        return loss, GY.ravel()
    try:
        res = minimize(fg, Y0.ravel(), jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-20,
                                "gtol": 1e-12, "maxfun": maxiter * 5,
                                "maxcor": maxcor})
        Y = res.x.reshape(n, d)
    except Exception:
        Y = Y0
    Y = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
    return Y

def _lp_polish(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 2.0e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        cmat = x @ x.T
        iu_c = cmat[iu]
        m = float(iu_c.max())
        sel = iu_c > m - max(20.0 * radius, 8e-7)
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 4000:
            thr = np.sort(iu_c[sel])[-4000]
            sel = iu_c >= thr
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-iu_c[sel], A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
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
            radius = min(radius * 1.5, 0.012)
        else:
            radius *= 0.5
    return x

def _rand_unit(rng, m, d):
    A = rng.standard_normal((m, d))
    A /= np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    return A

def _void_reinsert(X, nr, rng):
    n, d = X.shape
    rm, G = _rowmax(X)
    worst = np.argsort(-rm)[:nr]
    keep = np.ones(n, dtype=bool); keep[worst] = False
    Xr = X[keep]
    new_pts = []
    for _ in range(nr):
        C = _rand_unit(rng, 2000, d)
        if len(Xr) > 0:
            D = C @ Xr.T
            s = D.max(axis=1)
        else:
            s = np.zeros(2000)
        bi = int(np.argmin(s))
        # tie-break among near-best
        thr = s[bi] + 1e-4
        cand = np.where(s <= thr)[0]
        if len(cand) > 1:
            bi = int(cand[rng.integers(len(cand))])
        bp = C[bi]
        new_pts.append(bp)
        Xr = np.vstack([Xr, bp[None, :]])
    Y = X.copy()
    Y[worst] = np.array(new_pts)
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return Y

def solve():
    t0 = time.time()
    BUDGET = 168.0
    def rem():
        return BUDGET - (time.time() - t0)
    rng = np.random.default_rng(20261103)
    n, d = 250, 6
    best = None; best_val = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            z = np.asarray(np.load(fn), dtype=np.float64)
            if z.shape == (n, d):
                z = z / np.linalg.norm(z, axis=1, keepdims=True)
                v = _max_cos(z)
                if v < best_val:
                    best_val = v; best = z.copy()
        except Exception:
            continue
    if best is None:
        best = _rand_unit(rng, n, d)
        best_val = _max_cos(best)
    # quick collective rebalance of plateau at low beta
    if rem() > 100:
        try:
            c = _smooth_opt(best, 2000.0, 80)
            v = _max_cos(c)
            if v < best_val:
                best = c; best_val = v
        except Exception:
            pass
    trial = 0
    stall = 0
    while rem() > 30 and trial < 60:
        trial += 1
        mode = trial % 5
        Y = best.copy()
        try:
            rm, G = _rowmax(Y)
        except Exception:
            rm = np.zeros(n); G = Y @ Y.T
        if mode == 0:
            K = int(8 + rng.integers(0, 8))  # 8-15 worst block
            idx = np.argsort(-rm)[:K]
            sig = 0.10 + 0.05 * rng.random()
            Y[idx] += rng.standard_normal((K, d)) * sig
        elif mode == 1:
            c0 = int(rng.integers(n))
            dots = Y @ Y[c0]
            dots[c0] = -10
            nb = np.argsort(-dots)[:10]
            Y[nb] += rng.standard_normal((len(nb), d)) * 0.15
            r5 = rng.choice(n, size=5, replace=False)
            Y[r5] += rng.standard_normal((5, d)) * 0.20
        elif mode == 2:
            nr = int(3 + rng.integers(0, 4))  # 3-6 voids
            try:
                Y = _void_reinsert(Y, nr, rng)
            except Exception:
                Y += rng.standard_normal((n, d)) * 0.05
            # small jitter on neighbours of voids
            Y += rng.standard_normal((n, d)) * 0.008
        elif mode == 3:
            sig = 0.025 + 0.025 * rng.random()
            Y += rng.standard_normal((n, d)) * sig
        else:
            # pair-split: push worst pair apart
            k = int(np.argmax(G.reshape(-1)))
            a = k // n; b = k % n
            direction = Y[a] - Y[b]
            dn = np.linalg.norm(direction)
            if dn < 1e-12:
                direction = rng.standard_normal(d)
                dn = np.linalg.norm(direction)
            direction /= dn
            step = 0.05 + 0.05 * rng.random()
            Y[a] += direction * step + rng.standard_normal(d) * 0.02
            Y[b] -= direction * step + rng.standard_normal(d) * 0.02
            # jitter their near shells
            for v in (a, b):
                dots = Y @ Y[v]
                dots[v] = -10
                nb = np.argsort(-dots)[:6]
                Y[nb] += rng.standard_normal((len(nb), d)) * 0.06
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        # anneal: low beta collective then high beta squeeze
        b1 = 1500.0 if (trial % 2 == 0) else 2500.0
        b2 = 8000.0 if (trial % 3 == 0) else 12000.0
        try:
            if rem() < 32:
                break
            Y = _smooth_opt(Y, b1, 70)
            if rem() < 30:
                v = _max_cos(Y)
                if v < best_val:
                    best = Y; best_val = v
                break
            Y = _smooth_opt(Y, b2, 70)
        except Exception:
            continue
        try:
            v = _max_cos(Y)
        except Exception:
            continue
        if v < best_val - 1e-12:
            best = Y.copy(); best_val = v
            stall = 0
        else:
            stall += 1
            # occasional large escape when stalled
            if stall >= 4 and rem() > 50:
                try:
                    Z = best.copy()
                    Z += rng.standard_normal((n, d)) * 0.08
                    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
                    Z = _smooth_opt(Z, 1200.0, 80)
                    Z = _smooth_opt(Z, 10000.0, 80)
                    vz = _max_cos(Z)
                    if vz < best_val:
                        best = Z; best_val = vz
                    stall = 0
                except Exception:
                    pass
    # final squeeze on best
    try:
        if rem() > 8:
            best = _lp_polish(best, time.time() + rem() - 1.0)
    except Exception:
        pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
