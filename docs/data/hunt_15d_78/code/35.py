import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix


def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def _contact_counts(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = G.max()
    cnts = (G > m - 1e-9).sum(axis=1)
    return cnts, m


def _lbfgs_smooth(X0, betas, maxiters, frame_ws, t_deadline):
    n, d = X0.shape
    iu0, iu1 = np.triu_indices(n, 1)
    a = n / d
    Xcur = X0 / np.linalg.norm(X0, axis=1, keepdims=True)
    xf = Xcur.ravel().copy()
    for beta, mit, fw in zip(betas, maxiters, frame_ws):
        if time.time() > t_deadline:
            break

        def fun(g):
            Y0 = g.reshape(n, d)
            nr = np.linalg.norm(Y0, axis=1, keepdims=True)
            nr = np.maximum(nr, 1e-12)
            Y = Y0 / nr
            G = Y @ Y.T
            gv = G[iu0, iu1]
            m = gv.max()
            e = np.exp(beta * (gv - m))
            S = e.sum()
            L = m + np.log(S) / beta
            w = e / S
            W = np.zeros((n, n))
            W[iu0, iu1] = w
            W = W + W.T
            Gr = W @ Y
            val = float(L)
            gY = Gr.copy()
            if fw != 0.0:
                M = Y.T @ Y - a * np.eye(d)
                E = float((M * M).sum())
                val = val + fw * E
                gY = gY + fw * 4.0 * (Y @ M)
            dot = (gY * Y).sum(axis=1, keepdims=True)
            gX = (gY - dot * Y) / nr
            return val, gX.ravel().astype(np.float64)

        try:
            r = minimize(fun, xf, jac=True, method="L-BFGS-B",
                         options={"maxiter": mit, "ftol": 1e-14, "gtol": 1e-6})
            xf = r.x
        except Exception:
            break
    Xo = xf.reshape(n, d)
    Xo /= np.linalg.norm(Xo, axis=1, keepdims=True)
    return Xo


def _lp_polish(X, seconds):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    start = time.time()
    radius = 2e-3
    iu0, iu1 = np.triu_indices(n, 1)
    it = 0
    while time.time() - start < seconds and radius > 1e-13 and it < 60:
        it += 1
        if time.time() - start > seconds - 0.5:
            break
        Gf = X @ X.T
        c = Gf[iu0, iu1]
        m = c.max()
        sel = c > m - max(20 * radius, 2e-9)
        I = iu0[sel]; J = iu1[sel]
        cc = c[sel]
        # cap constraints to tightest pairs for speed
        if len(I) > 2500:
            k = np.argpartition(-cc, 2500)[:2500]
            I, J, cc = I[k], J[k], cc[k]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d),
                                      np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cc, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
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
            X, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return X


def _deep_voids(X, k, rng, nprobe=8000):
    n, d = X.shape
    P = rng.standard_normal((nprobe, d))
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    # max dot to X (closest existing point)
    S = P @ X.T
    best = S.max(axis=1)
    idx = np.argsort(best)  # smallest max-dot = deepest void first
    chosen = []
    selP = []
    for ii in idx:
        p = P[ii]
        ok = True
        for q in selP:
            if float(p @ q) > 0.7:  # keep voids separated
                ok = False
                break
        if ok:
            chosen.append(ii)
            selP.append(p)
            if len(chosen) >= k:
                break
    return P[np.array(chosen)]


def solve():
    t0 = time.time()
    LIMIT = 172.0
    rng = np.random.default_rng(7)
    n, d = 78, 15

    base = None
    baseV = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.asarray(np.load(fn), dtype=np.float64)
            if a.shape != (78, 15):
                continue
            a /= np.linalg.norm(a, axis=1, keepdims=True)
            v = _max_cos(a)
            if v < baseV:
                baseV = v
                base = a.copy()
        except Exception:
            continue
    if base is None:
        base = rng.standard_normal((n, d))
        base /= np.linalg.norm(base, axis=1, keepdims=True)
        baseV = _max_cos(base)

    cnts, _ = _contact_counts(base)
    order_desc = np.argsort(-cnts, kind="stable")

    cands = []
    # 3 fresh isotropic random starts
    for s in range(3):
        r = np.random.default_rng(100 + s).standard_normal((n, d))
        r /= np.linalg.norm(r, axis=1, keepdims=True)
        cands.append(r)
    # 2 topology-change starts: replace jammed core points by deep voids
    for kk, sd in [(14, 200), (20, 201)]:
        rr = np.random.default_rng(sd)
        voids = _deep_voids(base, kk, rr, nprobe=8000)
        Xt = base.copy()
        victims = order_desc[:kk].copy()
        Xt[victims] = voids
        # small global shake to unjam, larger on victims' neighbours
        Xt = Xt + 0.03 * rr.standard_normal((n, d))
        Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
        # push victims exactly to voids again after shake? keep shake
        cands.append(Xt)
    # 1 heavily shaken base (different basin attempt)
    rr = np.random.default_rng(202)
    Xs = base + 0.35 * rr.standard_normal((n, d))
    Xs /= np.linalg.norm(Xs, axis=1, keepdims=True)
    cands.append(Xs)

    betas = [70.0, 160.0, 400.0, 1000.0]
    maxiters = [120, 120, 160, 200]
    frame_ws = [8e-4, 4e-4, 0.0, 0.0]

    bestX = base.copy()
    bestV = baseV
    # budget smooth stage to leave ~30s for LP
    smooth_deadline = t0 + 130.0
    for Xi in cands:
        if time.time() > smooth_deadline:
            break
        # per-candidate deadline
        per_dead = min(smooth_deadline, time.time() + 28.0)
        try:
            Xo = _lbfgs_smooth(Xi, betas, maxiters, frame_ws, per_dead)
        except Exception:
            continue
        v = _max_cos(Xo)
        if v < bestV:
            bestV = v
            bestX = Xo.copy()
            # early exit if clearly new record by margin
            if bestV < 0.1675:
                break

    time_left = LIMIT - (time.time() - t0)
    # polish the new best (and base as fallback comparison)
    lp_time = max(5.0, min(30.0, time_left - 5.0))
    if time_left > 7.0:
        try:
            Xp = _lp_polish(bestX, lp_time * 0.75)
        except Exception:
            Xp = bestX
        vp = _max_cos(Xp)
        # also quick snap of base if new path failed
        if vp >= baseV and time.time() - t0 < LIMIT - 8:
            try:
                Xb = _lp_polish(base, min(6.0, LIMIT - (time.time() - t0) - 2.0))
                vb = _max_cos(Xb)
                if vb < vp:
                    Xp, vp = Xb, vb
            except Exception:
                pass
        bestX, bestV = Xp, vp
    bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX
