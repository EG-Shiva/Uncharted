import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _lp_polish(x, seconds, t_deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while True:
        if time.time() - start >= seconds:
            break
        if t_deadline is not None and time.time() >= t_deadline:
            break
        if radius < 1e-14:
            break
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
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
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
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


def _smooth_anneal(x_init, steps=2000, beta0=40.0, beta1=380.0,
                   lr0=0.015, lr1=0.002, mu=0.85):
    X = x_init / np.linalg.norm(x_init, axis=1, keepdims=True)
    n, d = X.shape
    V = np.zeros_like(X)
    for t in range(steps):
        frac = t / max(steps - 1, 1)
        beta = beta0 * (1 - frac) + beta1 * frac
        lr = lr0 * (1 - frac) + lr1 * frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        # exp in stable way
        E = np.exp(beta * (G - m))
        np.fill_diagonal(E, 0.0)
        S = E.sum() * 0.5
        if not np.isfinite(S) or S <= 0:
            break
        W = E / S
        grad = W @ X
        dot = np.sum(grad * X, axis=1, keepdims=True)
        gt = grad - dot * X
        # clip extreme gradients for stability
        # per-row clip at 5
        nr = np.sqrt(np.sum(gt * gt, axis=1, keepdims=True)) + 1e-12
        scale = np.minimum(1.0, 5.0 / nr)
        # scale array (n,1) broadcast
        gt = gt * scale
        V = mu * V + gt
        X = X - lr * V
        nrm = np.linalg.norm(X, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        X /= nrm
    return X


def _find_void(X, rng, n_sample=8000):
    n, d = X.shape
    C = rng.standard_normal((n_sample, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    dots = C @ X.T
    m = dots.max(axis=1)
    return C[int(np.argmin(m))]


def solve():
    t0 = time.time()
    DEADLINE = t0 + 176.0
    rng = np.random.default_rng(7)

    # load starting points
    best_X = None
    best_v = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (94, 13):
                a /= np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(a)
                if v < best_v:
                    best_v = v
                    best_X = a.copy()
        except Exception:
            pass
    if best_X is None:
        a = rng.standard_normal((94, 13))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        best_X = a
        best_v = _max_cos(a)
    n, d = 94, 13

    def row_maxima(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        return G.max(axis=1), G

    # deterministic basin-hopping schedule
    N_TRY = 48
    for it in range(N_TRY):
        if time.time() > DEADLINE - 14:
            break
        mode = it % 6
        cand = best_X.copy()
        if mode == 0:
            sigma = 0.008
            cand = cand + sigma * rng.standard_normal(cand.shape)
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            steps, b0, b1 = 1500, 50.0, 350.0
        elif mode == 1:
            sigma = 0.035
            cand = cand + sigma * rng.standard_normal(cand.shape)
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            steps, b0, b1 = 2000, 40.0, 400.0
        elif mode == 2:
            k = 3
            idx = rng.choice(n, size=k, replace=False)
            R = rng.standard_normal((k, d))
            R /= np.linalg.norm(R, axis=1, keepdims=True)
            cand[idx] = R
            steps, b0, b1 = 2200, 40.0, 420.0
        elif mode == 3:
            k = 6
            idx = rng.choice(n, size=k, replace=False)
            cand[idx] += 0.18 * rng.standard_normal((k, d))
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            steps, b0, b1 = 2000, 40.0, 400.0
        elif mode == 4:
            # void reinsertion of worst point
            rm, _ = row_maxima(cand)
            worst = int(np.argmax(rm))
            # second worst sometimes
            if (it // 6) % 2 == 1:
                order = np.argsort(rm)
                worst2 = int(order[-2])
                v1 = _find_void(cand, rng, 8000)
                cand[worst] = v1 + 1e-3 * rng.standard_normal(d)
                # remove second worst to random far point too
                v2 = _find_void(cand, rng, 8000)
                cand[worst2] = v2 + 1e-3 * rng.standard_normal(d)
            else:
                v = _find_void(cand, rng, 12000)
                cand[worst] = v + 1e-3 * rng.standard_normal(d)
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            steps, b0, b1 = 2400, 35.0, 450.0
        else:
            # targeted: move overloaded (many contacts) to void + small global shake
            G = cand @ cand.T
            np.fill_diagonal(G, -2.0)
            thresh = best_v - 0.002
            contacts = (G > thresh).sum(axis=1)
            worst = int(np.argmax(contacts))
            v = _find_void(cand, rng, 8000)
            cand[worst] = v + 1e-3 * rng.standard_normal(d)
            cand += 0.005 * rng.standard_normal(cand.shape)
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            steps, b0, b1 = 2200, 40.0, 420.0

        # vary lr slightly deterministically
        lr0 = 0.014 + 0.004 * ((it // 6) % 3) / 2.0
        cand2 = _smooth_anneal(cand, steps=steps, beta0=b0, beta1=b1,
                               lr0=lr0, lr1=0.002, mu=0.85)
        v = _max_cos(cand2)
        # accept without LP if clearly better (rare)
        if v < best_v - 1e-12 and v < best_v:
            # quick LP finish to lock it
            budget = min(2.5, max(0.5, (DEADLINE - 14 - time.time()) / 4))
            if budget > 0.4:
                cand2 = _lp_polish(cand2, seconds=budget, t_deadline=DEADLINE - 12)
                v = _max_cos(cand2)
            if v < best_v:
                best_X, best_v = cand2, v
                continue
        # LP finish promising candidates
        if v < best_v + 0.006:
            if time.time() > DEADLINE - 14:
                break
            budget = 2.0
            # give a bit more to very promising
            if v < best_v + 0.001:
                budget = 2.5
            cand2p = _lp_polish(cand2, seconds=budget, t_deadline=DEADLINE - 12)
            vp = _max_cos(cand2p)
            if vp < best_v - 1e-13:
                best_X, best_v = cand2p, vp
        # occasional random fresh restart to explore other basins
        if (it + 1) % 12 == 0 and time.time() < DEADLINE - 30:
            R = rng.standard_normal((n, d))
            R /= np.linalg.norm(R, axis=1, keepdims=True)
            R2 = _smooth_anneal(R, steps=1800, beta0=40.0, beta1=350.0,
                                lr0=0.015, lr1=0.002, mu=0.85)
            vr = _max_cos(R2)
            if vr < best_v + 0.008:
                R2p = _lp_polish(R2, seconds=1.5, t_deadline=DEADLINE - 12)
                vrp = _max_cos(R2p)
                if vrp < best_v:
                    best_X, best_v = R2p, vrp

    # final squeeze
    remain = DEADLINE - time.time() - 1.0
    if remain > 2.0:
        best_X = _lp_polish(best_X, seconds=min(remain, 12.0), t_deadline=DEADLINE - 0.5)
    best_X /= np.linalg.norm(best_X, axis=1, keepdims=True)
    return best_X
