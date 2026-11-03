import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def lp_polish(x, seconds, radius_init=1e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    t_end = time.time() + seconds
    radius = float(radius_init)
    iu = np.triu_indices(n, 1)
    while time.time() < t_end and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu]
        m = float(c_all.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = c_all > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 3000:
            idx = np.where(sel)[0]
            # keep largest c
            part = np.argpartition(-c_all[idx], 3000 - 1)[:3000]
            keep = np.zeros_like(sel, dtype=bool)
            keep[idx[part]] = True
            sel = keep
            cnt = 3000
        I = iu[0][sel]
        J = iu[1][sel]
        csel = c_all[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], 1
        ).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix(
            (x.ravel(),
             (np.repeat(np.arange(n), d), np.arange(n * d))),
            shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq,
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
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def fast_relax(x, steps=200, alpha=70.0, lr=0.03):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n = x.shape[0]
    for s in range(steps):
        G = x @ x.T
        # off-diagonal max
        m = float((G - 2.0 * np.eye(n)).max())
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        gd = (grad * x).sum(axis=1, keepdims=True)
        grad = grad - gd * x
        gn = np.linalg.norm(grad) / np.sqrt(n) + 1e-12
        step = lr * (1.0 - 0.7 * s / steps) / gn
        x = x - step * grad
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def solve():
    t0 = time.time()
    DEADLINE = t0 + 168.0
    rng = np.random.default_rng(0)
    # load start
    try:
        X0 = np.load("best.npy")
    except Exception:
        try:
            X0 = np.load("record.npy")
        except Exception:
            X0 = rng.standard_normal(size=(94, 13))
    X0 = np.asarray(X0, dtype=float)
    if X0.shape != (94, 13):
        # fallback reshape / random
        tmp = rng.standard_normal(size=(94, 13))
        X0 = tmp
    X0 = X0 / np.linalg.norm(X0, axis=1, keepdims=True)

    # initial squeeze
    init_sec = min(4.0, max(0.5, DEADLINE - 8.0 - time.time()))
    Xbest = lp_polish(X0, seconds=init_sec)
    best = max_cos(Xbest)
    Xcur = Xbest.copy()
    cur = best

    it = 0
    Temp = 0.0015
    while True:
        now = time.time()
        remain = DEADLINE - 8.0 - now
        if remain < 0.6:
            break
        it += 1
        quick = min(2.2, remain)
        mode = it % 6

        Y = Xcur.copy()
        n, d = Y.shape
        if mode == 1:
            sigma = 0.02
            Y = Y + sigma * rng.standard_normal(size=Y.shape)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        elif mode == 2:
            sigma = 0.05
            Y = Y + sigma * rng.standard_normal(size=Y.shape)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        elif mode == 3:
            # shake worst 8 points hard
            G = Y @ Y.T
            np.fill_diagonal(G, -2.0)
            mx = G.max(axis=1)
            worst = np.argsort(-mx)[:8]
            Y[worst] += 0.18 * rng.standard_normal(size=(8, d))
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        elif mode == 4:
            # single point re-insert with greedy search
            G = Y @ Y.T
            np.fill_diagonal(G, -2.0)
            mx = G.max(axis=1)
            w = int(np.argmax(mx))
            # candidate search: farthest from others
            Nc = 2500
            cand = rng.standard_normal(size=(Nc, d))
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            # dots to all except w
            M = Y.copy()
            # use Y without w for scoring
            dots = cand @ M.T
            # ignore self-column w by setting to -2
            dots[:, w] = -2.0
            scores = dots.max(axis=1)
            bi = int(np.argmin(scores))
            Y[w] = cand[bi]
        elif mode == 5:
            idx = rng.choice(n, 16, replace=False)
            Y[idx] += 0.06 * rng.standard_normal(size=(16, d))
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        else:  # mode 0
            # global small + subset medium mix
            Y = Y + 0.012 * rng.standard_normal(size=Y.shape)
            idx = rng.choice(n, 6, replace=False)
            Y[idx] += 0.10 * rng.standard_normal(size=(6, d))
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)

        # fast smooth relaxation before expensive LP
        try:
            Y = fast_relax(Y, steps=180, alpha=70.0, lr=0.03)
        except Exception:
            pass

        Yp = lp_polish(Y, seconds=quick)
        v = max_cos(Yp)

        if v < best - 1e-13:
            Xbest = Yp
            best = v
            Xcur = Yp.copy()
            cur = v
            # extra squeeze when improved
            now2 = time.time()
            rem2 = DEADLINE - 8.0 - now2
            if rem2 > 1.0:
                Xbest = lp_polish(Xbest, seconds=min(2.5, rem2))
                best = max_cos(Xbest)
                Xcur = Xbest.copy()
                cur = best
        else:
            # Metropolis move on current to keep exploring
            if v < cur or rng.random() < np.exp((cur - v) / Temp):
                Xcur = Yp
                cur = v
            # slow cooling, then reheat cyclically
            Temp *= 0.9995
            if Temp < 0.0003:
                Temp = 0.0015
            # occasional restart from best
            if it % 25 == 0 and rng.random() < 0.5:
                Xcur = Xbest.copy()
                cur = best

    # final polish with remaining time
    rem = DEADLINE - time.time()
    if rem > 0.5:
        Xbest = lp_polish(Xbest, seconds=rem - 0.2)
    Xbest /= np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest
