import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix


def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def _row_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)


def _rand_sphere(rng, n, d):
    A = rng.standard_normal(size=(n, d))
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    return A


def _find_void(others, rng, n_cand=8000):
    d = others.shape[1]
    cand = _rand_sphere(rng, n_cand, d)
    dots = cand @ others.T
    m = dots.max(axis=1)
    v = cand[int(np.argmin(m))].copy()
    lr = 0.05
    tau = 0.01
    for _ in range(300):
        dots = others @ v
        m0 = float(dots.max())
        w = np.exp((dots - m0) / tau)
        s = w.sum()
        if s <= 0 or not np.isfinite(s):
            break
        w /= s
        g = w @ others
        gt = g - float(g @ v) * v
        nrm = float(np.linalg.norm(gt))
        if nrm < 1e-12 or not np.isfinite(nrm):
            break
        gt /= nrm
        v2 = v - lr * gt
        n2 = float(np.linalg.norm(v2))
        if n2 < 1e-12:
            lr *= 0.5
            continue
        v2 /= n2
        m2 = float(others @ v2 @ np.ones(1) if False else (others @ v2).max())
        if m2 < m0:
            v = v2
            lr = min(lr * 1.05, 0.1)
        else:
            lr *= 0.5
            if lr < 1e-6:
                break
    v /= np.linalg.norm(v)
    return v


def _fg(Yflat, n, d, tau):
    Y = Yflat.reshape(n, d)
    rn = np.linalg.norm(Y, axis=1, keepdims=True)
    rn = np.maximum(rn, 1e-12)
    X = Y / rn
    G = X @ X.T
    np.fill_diagonal(G, -10.0)
    m = float(G.max())
    E = np.exp((G - m) / tau)
    np.fill_diagonal(E, 0.0)
    S = float(E.sum() * 0.5)
    if S <= 0 or not np.isfinite(S):
        S = 1e-300
    f = m + tau * np.log(S + 1e-300)
    GX = E @ X
    grad_X = GX / S
    dot = np.sum(grad_X * X, axis=1, keepdims=True)
    grad_tan = grad_X - dot * X
    grad_Y = grad_tan / rn
    return float(f), grad_Y.ravel()


def _anneal(X0, taus, maxiters):
    n, d = X0.shape
    Y = X0.ravel().copy()
    for tau, mi in zip(taus, maxiters):
        try:
            res = minimize(_fg, Y, args=(n, d, float(tau)), jac=True,
                           method="L-BFGS-B",
                           options={"maxiter": int(mi), "ftol": 1e-14,
                                    "gtol": 1e-6, "maxcor": 20, "maxfun": 100000})
            Y = res.x
        except Exception:
            break
    X = Y.reshape(n, d)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def _lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        cfull = (x @ x.T)[iu]
        m = float(cfull.max())
        sel = cfull > m - max(20 * radius, 1e-9)
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
            res = linprog(obj, A_ub=A_ub, b_ub=-cfull[sel], A_eq=A_eq,
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
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    np.random.seed(0)
    rng = np.random.default_rng(0)
    t0 = time.time()
    BUDGET = 168.0

    # load starting point
    Xb = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            A = np.load(fn)
            A = np.asarray(A, dtype=float)
            if A.shape != (93, 13):
                continue
            A /= np.linalg.norm(A, axis=1, keepdims=True)
            v = _max_cos(A)
            if v < best_val:
                best_val = v
                Xb = A.copy()
        except Exception:
            pass
    if Xb is None:
        Xb = _rand_sphere(rng, 93, 13)
        best_val = _max_cos(Xb)
    Xb /= np.linalg.norm(Xb, axis=1, keepdims=True)
    x_best = Xb.copy()

    it = 0
    # phase 1: basin hopping until ~138s
    while time.time() - t0 < 128.0:
        it += 1
        if time.time() - t0 > BUDGET - 35:
            break
        cyc = it % 5
        Xt = None
        taus = [0.015, 0.007, 0.003]
        mis = [120, 150, 200]
        if cyc == 1:
            sig = 0.02
            Xt = x_best + sig * rng.standard_normal(size=x_best.shape)
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
        elif cyc == 2:
            rm = _row_max(x_best)
            k = 7
            worst = np.argpartition(-rm, k)[:k]
            Xt = x_best.copy()
            Xt[worst] = _rand_sphere(rng, k, 13)
        elif cyc == 3:
            rm = _row_max(x_best)
            wi = int(np.argmax(rm))
            others = np.delete(x_best, wi, axis=0)
            nc = 6000 if (it % 2 == 0) else 12000
            v = _find_void(others, rng, n_cand=nc)
            Xt = x_best.copy()
            Xt[wi] = v
            taus = [0.012, 0.006, 0.0025]
        elif cyc == 4:
            sig = 0.06
            # perturb only a random block to keep part of structure
            Xt = x_best.copy()
            nb = 20
            idx = rng.choice(93, size=nb, replace=False)
            Xt[idx] += sig * rng.standard_normal(size=(nb, 13))
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            taus = [0.02, 0.008, 0.003]
        else:
            # double void reinsert of two worst
            rm = _row_max(x_best)
            k = 2
            worst = np.argpartition(-rm, k)[:k]
            Xt = x_best.copy()
            for wi in worst:
                others_idx = [i for i in range(93) if i != wi]
                # use current Xt for others to avoid collision
                others = Xt[others_idx]
                v = _find_void(others, rng, n_cand=5000)
                Xt[wi] = v
            taus = [0.012, 0.006, 0.0025]
            mis = [100, 120, 150]

        try:
            Xo = _anneal(Xt, taus, mis)
        except Exception:
            continue
        v = _max_cos(Xo)
        if v < best_val and np.all(np.isfinite(Xo)):
            best_val = v
            x_best = Xo.copy()
        # opportunistic small LP touch if very close (cheap check every 10)
        if it % 12 == 0 and (time.time() - t0) < 110:
            try:
                Xp = _lp_polish(x_best, 2.0)
                vp = _max_cos(Xp)
                if vp < best_val:
                    best_val = vp
                    x_best = Xp
            except Exception:
                pass

    remain = BUDGET - (time.time() - t0)
    if remain > 3:
        try:
            x_best = _lp_polish(x_best, min(remain - 1.0, 30.0))
        except Exception:
            pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    # strict feasibility: replace any bad row
    for i in range(x_best.shape[0]):
        nrm = float(np.linalg.norm(x_best[i]))
        if not np.isfinite(nrm) or nrm < 1e-12:
            x_best[i] = _rand_sphere(rng, 1, 13)[0]
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
