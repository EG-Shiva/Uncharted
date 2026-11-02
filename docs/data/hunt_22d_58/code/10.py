import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def solve():
    t0 = time.time()
    HARD_END = t0 + 170.0
    SEARCH_END = t0 + 118.0
    rng = np.random.default_rng(0)

    # load start
    X0 = None
    try:
        X0 = np.load("best.npy")
    except Exception:
        pass
    Xr = None
    try:
        Xr = np.load("record.npy")
    except Exception:
        pass
    cands = []
    for a in (X0, Xr):
        try:
            a = np.asarray(a, dtype=float)
            if a.shape == (58, 22):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cands.append(((_max_cos(a)), a))
        except Exception:
            pass
    if len(cands) == 0:
        R = rng.standard_normal((58, 22))
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        cands.append((_max_cos(R), R))
    cands.sort(key=lambda t: t[0])
    X_best = cands[0][1].copy()
    V_best = float(cands[0][0])
    # if record better, use it
    for v, a in cands:
        if v < V_best:
            V_best, X_best = v, a.copy()

    n, d = (58, 22)

    def fast_quench(X, steps, lr0):
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        V = np.zeros_like(X)
        mu = 0.85
        for s in range(steps):
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            frac = s / steps
            delta = 0.012 * (1.0 - frac) + 0.002
            T = delta / 3.0
            if T < 0.0006:
                T = 0.0006
            thresh = m - 5.0 * T
            # sparse exponential weights
            W = np.zeros((n, n), dtype=float)
            # manual mask
            M = G > thresh
            np.fill_diagonal(M, False)
            if np.any(M):
                W[M] = np.exp((G[M] - m) / T)
            cnt = W.sum(axis=1, keepdims=True)
            WX = W @ X
            coeff = (W * G).sum(axis=1, keepdims=True)
            F = -WX + coeff * X
            Favg = F / np.maximum(cnt, 1.0)
            lr = lr0 * (1.0 - 0.7 * frac)
            V = mu * V + lr * Favg
            # limit step
            vn = np.linalg.norm(V, axis=1, keepdims=True)
            # clip to 0.03
            scale = np.ones_like(vn)
            over = vn > 0.03
            scale[over] = 0.03 / (vn[over] + 1e-12)
            X = X + V * scale
            X = X / np.linalg.norm(X, axis=1, keepdims=True)
            if time.time() > HARD_END - 35.0:
                break
        return X

    def refine_hole(Y):
        m1 = Y.shape[0]
        ns = 2500
        R = rng.standard_normal((ns, d))
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        # scores
        S = (Y @ R.T).max(axis=0)
        idx = np.argpartition(S, 5)[:5]
        best_x = R[idx[0]].copy()
        best_s = np.inf
        for k in idx:
            x = R[k].copy()
            for s in range(250):
                dv = Y @ x
                mm = float(np.max(dv))
                T = 0.004 * (1.0 - s / 250.0) + 0.0005
                w = np.exp((dv - mm) / T)
                w /= w.sum()
                grad = w @ Y
                gt = grad - float(grad @ x) * x
                lr = 0.05 * (1.0 - 0.8 * s / 250.0)
                x = x - lr * gt
                nx = np.linalg.norm(x)
                if nx == 0 or not np.isfinite(nx):
                    x = R[k] / np.linalg.norm(R[k])
                else:
                    x = x / nx
            sf = float(np.max(Y @ x))
            if sf < best_s:
                best_s = sf
                best_x = x.copy()
        return best_x

    # victim order: overloaded first
    try:
        G0 = X_best @ X_best.T
        np.fill_diagonal(G0, -2.0)
        m0 = float(np.max(G0))
        cnt0 = (G0 > m0 - 0.001).sum(axis=1)
        order = np.argsort(-cnt0)
    except Exception:
        order = np.arange(n)

    trial = 0
    # main basin hopping
    while time.time() < SEARCH_END and time.time() < HARD_END - 40.0:
        trial += 1
        try:
            if trial % 3 == 0:
                # pure noise perturbation
                sigma = 0.025 if (trial % 6 == 0) else 0.07
                Xt = X_best + rng.standard_normal((n, d)) * sigma
                Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
                Xt = fast_quench(Xt, steps=800, lr0=0.08)
            else:
                if trial < 60:
                    victim = int(order[(trial - 1) % n])
                else:
                    victim = int(rng.integers(0, n))
                Y = np.delete(X_best, victim, axis=0)
                xh = refine_hole(Y)
                Xt = np.vstack([Y, xh])
                # small noise to neighbours
                Xt = Xt + rng.standard_normal((n, d)) * 0.008
                Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
                Xt = fast_quench(Xt, steps=950, lr0=0.09)
            v = _max_cos(Xt)
            if v < V_best and np.isfinite(v):
                X_best = Xt / np.linalg.norm(Xt, axis=1, keepdims=True)
                V_best = float(v)
                # refresh order
                try:
                    G0 = X_best @ X_best.T
                    np.fill_diagonal(G0, -2.0)
                    m0 = float(np.max(G0))
                    cnt0 = (G0 > m0 - 0.001).sum(axis=1)
                    order = np.argsort(-cnt0)
                except Exception:
                    pass
        except Exception:
            continue

    # final trust-region LP polish to jammed optimum
    def lp_polish_deadline(x, deadline):
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        nn, dd = x.shape
        best = _max_cos(x)
        radius = 1e-3
        iu = np.triu_indices(nn, 1)
        while time.time() < deadline and radius > 1e-14:
            cmat = (x @ x.T)[iu]
            m = float(cmat.max())
            sel = cmat > m - max(20 * radius, 1e-9)
            # cap constraints for speed
            if int(np.sum(sel)) > 1200:
                # keep worst 1200
                thr = np.sort(cmat[sel])[-1200]
                sel = cmat >= thr
            I, J = iu[0][sel], iu[1][sel]
            p = len(I)
            if p == 0:
                radius *= 0.5
                continue
            nv = nn * dd + 1
            rows = np.repeat(np.arange(p), 2 * dd + 1)
            cols = np.concatenate([I[:, None] * dd + np.arange(dd), J[:, None] * dd + np.arange(dd), np.full((p, 1), nn * dd)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            try:
                A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
                A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(nn), dd), np.arange(nn * dd))), shape=(nn, nv)).tocsr()
            except Exception:
                radius *= 0.5
                continue
            obj = np.zeros(nv)
            obj[-1] = 1.0
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=-cmat[sel], A_eq=A_eq, b_eq=np.zeros(nn),
                              bounds=[(-radius, radius)] * (nn * dd) + [(None, None)], method="highs")
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            y = x + res.x[:-1].reshape(nn, dd)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            v = _max_cos(y)
            if v < best:
                x, best = y, v
                radius = min(radius * 1.5, 0.05)
            else:
                radius *= 0.5
        return x

    try:
        X_best = lp_polish_deadline(X_best, HARD_END - 2.0)
    except Exception:
        pass
    X_best = X_best / np.linalg.norm(X_best, axis=1, keepdims=True)
    return X_best
