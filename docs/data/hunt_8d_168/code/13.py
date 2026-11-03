import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _quench(X, steps, beta0, beta1, lr0, lr1):
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    lb0 = np.log(beta0)
    lb1 = np.log(beta1)
    ll0 = np.log(lr0)
    ll1 = np.log(lr1)
    for t in range(1, steps + 1):
        frac = t / steps
        beta = np.exp(lb0 + (lb1 - lb0) * frac)
        lr = np.exp(ll0 + (ll1 - ll0) * frac)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        gmax = G.max()
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        Z = W.sum()
        if Z == 0:
            continue
        Grad = (W @ X) / Z
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        m *= b1
        m += (1.0 - b1) * Gt
        v *= b2
        v += (1.0 - b2) * (Gt * Gt)
        bc1 = 1.0 - b1 ** t
        bc2 = 1.0 - b2 ** t
        mhat = m / bc1
        vhat = v / bc2
        step = mhat / (np.sqrt(vhat) + eps)
        X = X - lr * step
        nrm = np.sqrt((X * X).sum(axis=1, keepdims=True))
        nrm[nrm == 0] = 1.0
        X /= nrm
    return X


def _lp_polish(x, seconds, r0=1e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
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
        vv = _max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def _void_point(others, rng, ncand=4096, hill=120):
    d = others.shape[1]
    cand = rng.randn(ncand, d)
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ others.T
    mm = dots.max(axis=1)
    y = cand[np.argmin(mm)].copy()
    lr = 0.02
    for _ in range(hill):
        dts = others @ y
        # focus on worst few
        k = 4
        idx = np.argpartition(dts, -k)[-k:]
        g = others[idx].mean(axis=0)
        gt = g - (g @ y) * y
        nn = np.linalg.norm(gt)
        if nn < 1e-12:
            break
        gt /= nn
        y = y - lr * gt
        y /= np.linalg.norm(y)
        lr *= 0.994
    return y


def solve():
    t0 = time.time()
    BUDGET = 175.0
    PHASE1 = 133.0

    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape != (168, 8):
                continue
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            if not np.all(np.isfinite(a)):
                continue
            va = _max_cos(a)
            if va < best_val:
                best_val = va
                best = a.copy()
        except Exception:
            pass
    if best is None:
        rng0 = np.random.RandomState(7)
        best = rng0.randn(168, 8)
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)

    n, d = best.shape
    rng = np.random.RandomState(20261103)

    deadline1 = t0 + PHASE1
    it = 0
    stall = 0
    while time.time() < deadline1:
        if time.time() + 2.5 > t0 + BUDGET:
            break
        it += 1
        X = best.copy()
        r = rng.rand()

        if r < 0.33:
            # global Gaussian kick, sigma grows if stalled
            sigmas = (0.006, 0.012, 0.025, 0.05)
            sigma = sigmas[rng.randint(len(sigmas))]
            if stall > 25:
                sigma *= 1.6
            X = X + rng.randn(n, d) * sigma
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            beta0 = 30.0 + rng.rand() * 35.0
            beta1 = 250.0 + rng.rand() * 550.0
            steps = 900 + int(rng.randint(900))
            lr0 = 0.003 + rng.rand() * 0.005
            lr1 = lr0 * 0.1
            Xq = _quench(X, steps, beta0, beta1, lr0, lr1)
        elif r < 0.62:
            # void re-insertion of worst / rattler points
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            rowmax = G.max(axis=1)
            k = 1 + int(rng.randint(3))  # 1..3
            u = rng.rand()
            if u < 0.55:
                idx = np.argsort(rowmax)[-k:]  # worst
            elif u < 0.8:
                idx = np.argsort(rowmax)[:k]  # rattlers
                # sometimes mix one worst one rattler
                if k > 1 and rng.rand() < 0.5:
                    idx[0] = np.argmax(rowmax)
            else:
                idx = rng.choice(n, k, replace=False)
            for j in idx:
                others = np.delete(X, j, axis=0)
                y = _void_point(others, rng, ncand=4096, hill=100)
                X[j] = y
            beta0 = 40.0
            beta1 = 350.0 + rng.rand() * 450.0
            steps = 1100 + int(rng.randint(900))
            lr0 = 0.004 + rng.rand() * 0.004
            lr1 = lr0 * 0.12
            Xq = _quench(X, steps, beta0, beta1, lr0, lr1)
        elif r < 0.85:
            # hub + neighbours shake (attack overloaded point)
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            m = G.max()
            thr = m - 0.002
            deg = (G > thr).sum(axis=1)
            hub = int(np.argmax(deg))
            if deg[hub] < 3:
                hub = int(np.argmax(G.max(axis=1)))
            order = np.argsort(-G[hub])
            nb = int(min(n - 1, max(4, deg[hub] + 2)))
            idx = order[:nb]
            idx = np.concatenate([[hub], idx])
            idx = np.unique(idx)
            sigma = 0.03 + rng.rand() * 0.05
            noise = np.zeros((n, d))
            noise[idx] = rng.randn(len(idx), d) * sigma
            X = X + noise
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            beta0 = 35.0 + rng.rand() * 25.0
            beta1 = 300.0 + rng.rand() * 500.0
            steps = 1200 + int(rng.randint(800))
            lr0 = 0.004 + rng.rand() * 0.004
            lr1 = lr0 * 0.1
            Xq = _quench(X, steps, beta0, beta1, lr0, lr1)
        else:
            # worst-pair split: push the two endpoints apart tangentially
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            ij = int(np.argmax(G))
            ii, jj = ij // n, ij % n
            # tangent push
            for a, b in ((ii, jj), (jj, ii)):
                xa = X[a]
                xb = X[b]
                dot = float(xa @ xb)
                direction = -(xb - dot * xa)
                nn = np.linalg.norm(direction)
                if nn > 1e-12:
                    direction /= nn
                    amp = 0.015 + rng.rand() * 0.03
                    X[a] = X[a] + direction * amp + rng.randn(d) * 0.008
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            beta0 = 45.0
            beta1 = 400.0 + rng.rand() * 400.0
            steps = 1100 + int(rng.randint(800))
            lr0 = 0.004 + rng.rand() * 0.003
            lr1 = lr0 * 0.1
            Xq = _quench(X, steps, beta0, beta1, lr0, lr1)

        if not np.all(np.isfinite(Xq)):
            stall += 1
            continue
        vq = _max_cos(Xq)

        # short LP snap if close to best (topology lock-in)
        if vq < best_val + 2e-4 and vq > best_val - 5e-6:
            # only occasionally to save time
            if (it % 7 == 0) and (time.time() + 4.0 < t0 + BUDGET) and (time.time() < deadline1 - 5):
                try:
                    Xs = _lp_polish(Xq, 1.2, r0=5e-4)
                    vs = _max_cos(Xs)
                    if np.all(np.isfinite(Xs)) and vs < vq:
                        Xq, vq = Xs, vs
                except Exception:
                    pass

        if vq + 1e-12 < best_val:
            best = Xq
            best_val = vq
            stall = 0
        else:
            stall += 1

    remaining = (t0 + BUDGET) - time.time()
    if remaining > 7.0:
        secs = min(remaining - 2.0, 40.0)
        try:
            polished = _lp_polish(best, secs, r0=1e-3)
            vp = _max_cos(polished)
            if vp < best_val and np.all(np.isfinite(polished)):
                best = polished
                best_val = vp
        except Exception:
            pass

    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    bad = ~np.all(np.isfinite(best), axis=1)
    if np.any(bad):
        rr = np.random.RandomState(1).randn(int(bad.sum()), d)
        rr /= np.linalg.norm(rr, axis=1, keepdims=True)
        best[bad] = rr
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
