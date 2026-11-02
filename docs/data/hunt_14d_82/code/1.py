import time
import numpy as np

try:
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    _HAS_LINPROG = True
except Exception:
    _HAS_LINPROG = False


def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def per_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1), G


def smooth_polish(X, steps, beta0, beta1, lr0, lr1, mu=0.85):
    n, d = X.shape
    V = np.zeros_like(X)
    for s in range(steps):
        f = s / max(steps - 1, 1)
        beta = beta0 * (beta1 / beta0) ** f
        lr = lr0 * (lr1 / lr0) ** f
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        # weights exp(beta*(c-m)), <=1
        E = np.exp(beta * (G - m))
        np.fill_diagonal(E, 0.0)
        rs = E.sum(axis=1, keepdims=True) + 1e-12
        Gr = E @ X
        dot = (Gr * X).sum(axis=1, keepdims=True)
        Gt = (Gr - dot * X) / rs
        V = mu * V + (1.0 - mu) * Gt
        bc = 1.0 - mu ** (s + 1)
        Vhat = V / bc
        X = X - lr * Vhat
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X = X / nr
    return X


def lp_polish(x, seconds):
    if not _HAS_LINPROG:
        return x
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape

    def _mc(a):
        g = a @ a.T
        np.fill_diagonal(g, -2.0)
        return g.max()

    best = _mc(x)
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _mc(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    TIME_LIMIT = 178.0
    rng = np.random.RandomState(0)
    np.random.seed(0)
    n, d = 82, 14

    # load start
    Xinit = None
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            if a.shape == (n, d):
                Xinit = a.astype(float)
                break
        except Exception:
            continue
    if Xinit is None:
        Xinit = rng.randn(n, d)
    Xinit = Xinit / np.linalg.norm(Xinit, axis=1, keepdims=True)

    best = Xinit.copy()
    best_val = max_cos(best)

    # baseline medium polish of start (keep if better)
    try:
        Y0 = smooth_polish(best.copy(), steps=2500, beta0=80.0, beta1=1500.0, lr0=0.02, lr1=0.0006, mu=0.85)
        v0 = max_cos(Y0)
        if v0 < best_val:
            best, best_val = Y0, v0
    except Exception:
        pass

    sigmas = [0.004, 0.012, 0.035, 0.09, 0.18]
    NTRIAL = 60

    for trial in range(NTRIAL):
        if time.time() - t0 > 130.0:
            break
        mode = trial % 10
        Y = best.copy()
        if mode == 0 and trial > 0:
            # fresh random start exploration (1 in 10)
            Y = rng.randn(n, d)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            steps, b0, b1, l0, l1 = 1800, 60.0, 1200.0, 0.03, 0.0008
        elif mode == 1:
            # rattler teleport: replace loosest points by random
            pm, G = per_max(Y)
            m = pm.max()
            # loosest: small per-point max
            order = np.argsort(pm)
            # number to teleport: those with margin + at least 2
            loose = [i for i in order if pm[i] < m - 2e-4]
            if len(loose) < 2:
                loose = list(order[:5])
            loose = loose[:max(5, len(loose))]
            for i in loose:
                v = rng.randn(d)
                v /= np.linalg.norm(v)
                Y[i] = v
            steps, b0, b1, l0, l1 = 1500, 70.0, 1400.0, 0.025, 0.0006
        elif mode == 2:
            # worst-point greedy reinsertion
            pm, G = per_max(Y)
            w = int(np.argmax(pm))
            # occasionally do 2 points
            Ws = [w]
            if trial % 20 == 10:
                # second worst
                o = np.argsort(pm)
                Ws.append(int(o[-2]))
            Xrest_idx = [i for i in range(n) if i not in Ws]
            Xr = Y[Xrest_idx]
            for w_ in Ws:
                K = 3000
                C = rng.randn(K, d)
                C /= np.linalg.norm(C, axis=1, keepdims=True)
                scores = (C @ Xr.T).max(axis=1)
                bi = int(np.argmin(scores))
                # local hill-climb on new point alone (fixed rest)
                y = C[bi].copy()
                lr_y = 0.05
                for it in range(300):
                    dots = Xr @ y
                    j = int(np.argmax(dots))
                    # tangent gradient of max dot
                    g = Xr[j] - (Xr[j] @ y) * y
                    gn = np.linalg.norm(g)
                    if gn < 1e-12:
                        break
                    g /= gn
                    # decaying step, accept if improves
                    cur = dots[j]
                    yn = y - lr_y * g
                    yn /= np.linalg.norm(yn)
                    new = float((Xr @ yn).max())
                    if new < cur:
                        y = yn
                        lr_y *= 1.02
                    else:
                        lr_y *= 0.6
                    if lr_y < 1e-6:
                        break
                Y[w_] = y
            steps, b0, b1, l0, l1 = 1500, 70.0, 1400.0, 0.02, 0.0005
        else:
            sigma = sigmas[(trial // 10 + trial) % len(sigmas)]
            # global shake; sometimes subset stronger shake
            if trial % 3 == 0:
                Y = Y + sigma * rng.randn(n, d)
            else:
                # subset shake of jammed neighbourhood + a rattler
                pm, G = per_max(Y)
                # overloaded = largest per_max
                o = np.argsort(pm)
                subset = list(o[-8:]) + list(o[:2])
                Y[subset] += (sigma * 1.5) * rng.randn(len(subset), d)
                Y += (sigma * 0.3) * rng.randn(n, d)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            steps, b0, b1, l0, l1 = 1300, 60.0, 1300.0, 0.025, 0.0006

        try:
            Y = smooth_polish(Y, steps=steps, beta0=b0, beta1=b1, lr0=l0, lr1=l1, mu=0.85)
        except Exception:
            continue
        v = max_cos(Y)
        if v < best_val:
            best, best_val = Y, v

    # final long refinement on best
    try:
        if time.time() - t0 < TIME_LIMIT - 30:
            best = smooth_polish(best.copy(), steps=5000, beta0=150.0, beta1=3000.0, lr0=0.008, lr1=0.00015, mu=0.88)
            best_val = max_cos(best)
    except Exception:
        pass

    # final LP trust-region squeeze if time remains
    remain = TIME_LIMIT - (time.time() - t0) - 1.0
    if remain > 4 and _HAS_LINPROG:
        try:
            Ylp = lp_polish(best.copy(), seconds=min(remain, 25.0))
            vlp = max_cos(Ylp)
            if vlp <= best_val:
                best = Ylp
        except Exception:
            pass

    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
