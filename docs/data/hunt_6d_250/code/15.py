import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _smooth_opt(X0, beta, maxiter):
    n, d = X0.shape
    Y0 = X0.copy()
    def fg(Yflat):
        Y = Yflat.reshape(n, d)
        nrm = np.linalg.norm(Y, axis=1, keepdims=True)
        nrm = np.maximum(nrm, 1e-12)
        X = Y / nrm
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = float(G.max())
        D = beta * (G - M)
        np.clip(D, -30.0, 0.0, out=D)
        E = np.empty_like(D)
        np.exp(D, out=E)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum())
        if S <= 0:
            S = 1e-300
        loss = M + np.log(S) / beta
        W = E / S
        GX = 2.0 * (W @ X)
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GY = (GX - dot * X) / nrm
        return loss, GY.ravel()
    res = minimize(fg, Y0.ravel(), jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter, "ftol": 1e-20, "gtol": 1e-12,
                            "maxfun": maxiter * 5, "maxcor": 12})
    Y = res.x.reshape(n, d)
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return Y

def _refine_hole(A, y0, beta=4000.0, maxiter=80):
    # A: (m,d) fixed points, minimize smooth max of A @ xhat
    def fg(Yflat):
        y = Yflat
        nrm = float(np.linalg.norm(y))
        nrm = max(nrm, 1e-12)
        xh = y / nrm
        s = A @ xh
        M = float(s.max())
        D = beta * (s - M)
        np.clip(D, -30.0, 0.0, out=D)
        E = np.exp(D)
        S = float(E.sum())
        if S <= 0:
            S = 1e-300
        loss = M + np.log(S) / beta
        W = E / S
        gx = A.T @ W  # d,
        dot = float(gx @ xh)
        gy = (gx - dot * xh) / nrm
        return loss, gy
    try:
        res = minimize(fg, y0.copy(), jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-20, "gtol": 1e-12,
                                "maxfun": maxiter * 5, "maxcor": 10})
        y = res.x
        y /= np.linalg.norm(y)
        return y
    except Exception:
        y = y0 / np.linalg.norm(y0)
        return y

def _lp_polish(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 3e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        cmat = x @ x.T
        iu_c = cmat[iu]
        m = float(iu_c.max())
        sel = iu_c > m - max(10.0 * radius, 2e-7)
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 6000:
            thr = np.sort(iu_c[sel])[-6000]
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
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
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
            x, best, radius = y, v, min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x

def solve():
    t_start = time.time()
    TIME_BUDGET = 168.0
    HOP_DEADLINE = t_start + 132.0
    LP_DEADLINE = t_start + 163.0
    rng = np.random.default_rng(42)
    n, d = 250, 6
    try:
        x0 = np.load("best.npy")
    except Exception:
        x0 = None
    try:
        rec = np.load("record.npy")
    except Exception:
        rec = None
    cands = []
    for z in (x0, rec):
        try:
            if z is not None:
                z = np.asarray(z, dtype=np.float64)
                if z.shape == (n, d):
                    z = z / np.linalg.norm(z, axis=1, keepdims=True)
                    if np.all(np.isfinite(z)):
                        cands.append(z)
        except Exception:
            pass
    if not cands:
        tmp = rng.standard_normal((n, d))
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        cands.append(tmp)
    best = min(cands, key=_max_cos).copy()
    best_val = _max_cos(best)

    # Stage 1: short anneal
    cur = best.copy()
    for beta, it in [(2000.0, 120), (8000.0, 150), (25000.0, 120)]:
        if time.time() > t_start + 35.0:
            break
        try:
            cur = _smooth_opt(cur, beta, it)
        except Exception:
            break
        v = _max_cos(cur)
        if v < best_val:
            best_val = v
            best = cur.copy()

    iu0, iu1 = np.triu_indices(n, 1)
    trial = 0
    sigmas = [0.012, 0.006, 0.0025, 0.001]
    while time.time() < HOP_DEADLINE and trial < 90:
        trial += 1
        if time.time() > HOP_DEADLINE:
            break
        # find worst structure of best
        try:
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            rowmax = G.max(axis=1)
            order = np.argsort(-rowmax)
        except Exception:
            order = np.arange(n)
        # every 3rd trial: void reinsertion of worst point
        if trial % 3 == 0:
            try:
                # worst point
                victim = int(order[0])
                # second max without victim
                mask = np.ones(n, dtype=bool)
                mask[victim] = False
                A = best[mask]  # (249,6)
                # sample holes
                Nc = 3500
                C = rng.standard_normal((Nc, d))
                C /= np.linalg.norm(C, axis=1, keepdims=True)
                S = C @ A.T  # (Nc,249)
                Sm = S.max(axis=1)
                bi = int(np.argmin(Sm))
                y0 = C[bi].copy()
                # refine top candidate
                yref = _refine_hole(A, y0, beta=4000.0, maxiter=60)
                # evaluate
                hole_val = float(np.max(A @ yref))
                # current m_without
                G2 = A @ A.T
                np.fill_diagonal(G2, -2.0)
                m_without = float(G2.max())
                newmax = max(m_without, hole_val)
                if newmax < best_val - 1e-12:
                    Y = best.copy()
                    Y[victim] = yref
                    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
                    # short settle
                    try:
                        Y2 = _smooth_opt(Y, 8000.0, 70)
                    except Exception:
                        Y2 = Y
                    v2 = _max_cos(Y2)
                    if v2 < best_val:
                        best_val = v2
                        best = Y2.copy()
                        cur = Y2.copy()
                        continue
                # if reinsert failed, fall through to kick
            except Exception:
                pass
        # kick-and-reoptimize
        try:
            sigma = sigmas[trial % len(sigmas)]
            k = 4 + (trial % 7)
            victims = list(order[:k])
            if trial % 2 == 1:
                victims += list(rng.integers(0, n, size=4))
            victims = list(set(victims))
            Y = best.copy()
            Y[victims] += rng.standard_normal((len(victims), d)) * sigma
            if trial % 7 == 0:
                Y += rng.standard_normal(Y.shape) * 0.0008
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            beta = 8000.0 if (trial % 2 == 0) else 3000.0
            try:
                Y2 = _smooth_opt(Y, beta, 90)
            except Exception:
                continue
            if time.time() < HOP_DEADLINE - 8 and trial % 4 == 0:
                try:
                    Y2 = _smooth_opt(Y2, 25000.0, 50)
                except Exception:
                    pass
            v2 = _max_cos(Y2)
            if v2 < best_val:
                best_val = v2
                best = Y2.copy()
        except Exception:
            continue

    # Stage 3: high-beta squeeze + LP
    if time.time() < LP_DEADLINE:
        try:
            if LP_DEADLINE - time.time() > 12:
                cur2 = _smooth_opt(best, 40000.0, 100)
                v = _max_cos(cur2)
                if v < best_val:
                    best_val = v
                    best = cur2.copy()
        except Exception:
            pass
        try:
            polished = _lp_polish(best, LP_DEADLINE)
            vp = _max_cos(polished)
            if vp < best_val:
                best = polished
                best_val = vp
        except Exception:
            pass

    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    best = np.where(np.isfinite(best), best, 0.0)
    best /= np.maximum(np.linalg.norm(best, axis=1, keepdims=True), 1e-12)
    return best
