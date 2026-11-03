import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _norm_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def peak_shave(X, iters=1000):
    X = X.copy()
    n = X.shape[0]
    for _ in range(iters):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        flat = int(np.argmax(G))
        i = flat // n
        j = flat % n
        if i == j:
            break
        m = float(G[i, j])
        G[i, j] = -2.0
        G[j, i] = -2.0
        m2 = float(G.max())
        d = m - m2
        if d < 1e-12:
            break
        eta = min(d * 0.4, 1e-3)
        ui = X[j] - m * X[i]
        vj = X[i] - m * X[j]
        ni = float(np.linalg.norm(ui))
        nj = float(np.linalg.norm(vj))
        if ni < 1e-12 or nj < 1e-12:
            break
        ui /= ni
        vj /= nj
        improved = False
        for scale in (1.0, 0.25, 0.0625):
            e = eta * scale
            Xn = X.copy()
            Xn[i] -= e * ui
            Xn[j] -= e * vj
            Xn[i] /= max(np.linalg.norm(Xn[i]), 1e-12)
            Xn[j] /= max(np.linalg.norm(Xn[j]), 1e-12)
            G2 = Xn @ Xn.T
            np.fill_diagonal(G2, -2.0)
            mn = float(G2.max())
            if mn < m:
                X = Xn
                improved = True
                break
        if not improved:
            break
    return X


def adam_run(X0, schedule):
    X = X0.copy()
    m1 = np.zeros_like(X)
    m2 = np.zeros_like(X)
    t = 0
    b1, b2 = 0.9, 0.999
    eps = 1e-8
    for alpha, steps, lr in schedule:
        for _ in range(steps):
            t += 1
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            mmax = float(G.max())
            # G-mmax <= 0, safe exp
            W = np.exp(alpha * (G - mmax))
            np.fill_diagonal(W, 0.0)
            S = float(W.sum() * 0.5)
            if not np.isfinite(S) or S <= 0:
                break
            grad = (W @ X) / S
            dot = (grad * X).sum(axis=1, keepdims=True)
            gt = grad - dot * X
            m1 = b1 * m1 + (1 - b1) * gt
            m2 = b2 * m2 + (1 - b2) * (gt * gt)
            c1 = 1.0 - b1 ** t
            c2 = 1.0 - b2 ** t
            m1h = m1 / c1
            m2h = m2 / c2
            step = lr * m1h / (np.sqrt(m2h) + eps)
            # per-row clip for stability
            sn = np.linalg.norm(step, axis=1, keepdims=True)
            cap = 0.05
            fac = np.minimum(1.0, cap / np.maximum(sn, 1e-12))
            step = step * fac
            X = X - step
            X = _norm_rows(X)
    return X


def lp_polish(x, seconds):
    x = _norm_rows(x.copy())
    n, d = x.shape
    try:
        best = max_cos(x)
    except Exception:
        return x
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        Gfull = x @ x.T
        c = Gfull[iu0]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        if int(sel.sum()) < 1:
            break
        I = iu0[0][sel]
        J = iu0[1][sel]
        p = len(I)
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate(
                [I[:, None] * d + np.arange(d),
                 J[:, None] * d + np.arange(d),
                 np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix(
                (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
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
        y = _norm_rows(y)
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    np.random.seed(0)
    t0 = time.time()

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (219, 6):
                a = _norm_rows(a)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
        a = np.random.randn(219, 6)
        a = _norm_rows(a)
        cands.append(a)

    best = None
    bestv = 1e9
    for c in cands:
        v = max_cos(c)
        if v < bestv:
            bestv = v
            best = c.copy()

    # immediate shave of isolated peak (guaranteed tiny win if unique max)
    try:
        Xs = peak_shave(best, 2000)
        vs = max_cos(Xs)
        if vs < bestv:
            best, bestv = Xs, vs
    except Exception:
        pass

    # annealed smooth repulsion from current best
    try:
        Xa = adam_run(best, [(80, 500, 0.02), (250, 500, 0.01), (600, 700, 0.005)])
        Xa = peak_shave(Xa, 300)
        va = max_cos(Xa)
        if va < bestv:
            best, bestv = Xa, va
    except Exception:
        pass

    sigmas = [0.008, 0.02, 0.04, 0.004, 0.012, 0.06]
    trial = 0
    # basin hopping until ~135s
    while time.time() - t0 < 135:
        sigma = sigmas[trial % len(sigmas)]
        trial += 1
        try:
            P = best + sigma * np.random.randn(*best.shape)
            P = _norm_rows(P)
            # short anneal
            Q = adam_run(P, [(150, 350, 0.012), (500, 350, 0.006)])
            Q = peak_shave(Q, 400)
            vq = max_cos(Q)
            if vq < bestv:
                best, bestv = Q, vq
        except Exception:
            continue
        # safety: stop if time almost out
        if time.time() - t0 > 134:
            break

    remain = 178.0 - (time.time() - t0)
    if remain > 6:
        try:
            Xp = lp_polish(best, min(remain - 2.0, 25.0))
            vp = max_cos(Xp)
            if vp < bestv:
                best, bestv = Xp, vp
        except Exception:
            pass
    else:
        try:
            best = peak_shave(best, 500)
        except Exception:
            pass

    best = _norm_rows(best)
    return best
