import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))


def smooth_descend(X, iters, a0, a1, lr0, lr1):
    n, d = X.shape
    lr_ratio = lr1 / lr0 if lr0 != 0 else 1.0
    a_ratio = a1 / a0
    for it in range(iters):
        frac = it / max(iters - 1, 1)
        alpha = a0 * (a_ratio ** frac)
        lr = lr0 * (lr_ratio ** frac)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        E = np.exp(alpha * (G - m))
        np.fill_diagonal(E, 0.0)
        Grad = E @ X
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        rms = float(np.sqrt(np.mean(np.sum(Gt ** 2, axis=1)))) + 1e-12
        X = X - lr * (Gt / rms)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def refine_hole(y0, Xf, steps=300, beta=120.0):
    y = y0.copy()
    for it in range(steps):
        dots = Xf @ y
        m = float(np.max(dots))
        w = np.exp(beta * (dots - m))
        s = float(np.sum(w))
        if s == 0 or not np.isfinite(s):
            break
        w /= s
        g = w @ Xf
        gt = g - float(np.dot(g, y)) * y
        lr = 0.03 * (1.0 - it / steps) + 0.002 * (it / steps)
        y = y - lr * gt
        nrm = float(np.linalg.norm(y))
        if nrm == 0 or not np.isfinite(nrm):
            break
        y /= nrm
    return y


def best_hole(Xf, rng, n_cand=8000):
    d = Xf.shape[1]
    Y = rng.standard_normal((n_cand, d))
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    S = Y @ Xf.T
    scores = np.max(S, axis=1)
    k = min(5, n_cand)
    idx = np.argpartition(scores, k - 1)[:k]
    best_y = None
    best_s = 1e100
    for j in idx:
        y = refine_hole(Y[j], Xf, steps=300)
        s = float(np.max(Xf @ y))
        if s < best_s:
            best_s = s
            best_y = y
    return best_y, best_s


def lp_polish_budget(X, budget):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = max_cos(X)
    radius = 1e-3
    t_end = time.time() + budget
    iu = np.triu_indices(n, 1)
    while time.time() < t_end and radius > 1e-14:
        Gfull = X @ X.T
        c = Gfull[iu]
        m = float(np.max(c))
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], 1
        ).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix(
            (X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
            shape=(n, nv),
        ).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(
                obj, A_ub=A_ub, b_ub=-c[sel],
                A_eq=A_eq, b_eq=np.zeros(n),
                bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                method="highs",
            )
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() >= t_end:
            break
    return X


def solve():
    t0 = time.time()
    deadline = t0 + 174.0
    rng = np.random.default_rng(0)
    try:
        Xb = np.load("best.npy").astype(float)
    except Exception:
        Xb = np.load("record.npy").astype(float)
    Xb = Xb / np.linalg.norm(Xb, axis=1, keepdims=True)
    best = max_cos(Xb)
    Xbest = Xb.copy()

    trial = 0
    sigmas = [0.025, 0.06, 0.012, 0.09, 0.04]
    # exploration phase, reserve ~30s for final polish
    while time.time() < deadline - 30.0:
        trial += 1
        remaining = deadline - 30.0 - time.time()
        if remaining <= 2.0:
            break
        if trial % 3 == 0:
            # greedy hole re-insertion of worst offenders
            try:
                G = Xbest @ Xbest.T
                np.fill_diagonal(G, -2.0)
                rowmax = np.max(G, axis=1)
                worst = np.argsort(-rowmax)[:2]
                Xt = Xbest.copy()
                for idx in worst:
                    mask = np.ones(Xt.shape[0], dtype=bool)
                    mask[idx] = False
                    Xf = Xt[mask]
                    ynew, _ = best_hole(Xf, rng, n_cand=6000)
                    Xt[idx] = ynew
                Xt = smooth_descend(Xt, 1200, 40.0, 300.0, 0.012, 0.001)
                s = max_cos(Xt)
                if s < best + 0.0015 and time.time() < deadline - 30.0:
                    Xt2 = lp_polish_budget(Xt, min(4.0, deadline - 30.0 - time.time()))
                    s2 = max_cos(Xt2)
                    if s2 < best - 1e-12:
                        Xbest, best = Xt2, s2
                    elif s < best - 1e-12:
                        # keep smooth version and polish later
                        Xbest, best = Xt, s
                        Xbest = lp_polish_budget(Xbest, min(4.0, deadline - 30.0 - time.time()))
                        best = max_cos(Xbest)
            except Exception:
                pass
        else:
            sigma = sigmas[trial % len(sigmas)]
            Xn = Xbest + sigma * rng.standard_normal(Xbest.shape)
            Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
            # annealed softmax descent
            try:
                Xn = smooth_descend(Xn, 2500, 35.0, 350.0, 0.02, 0.0006)
            except Exception:
                continue
            s = max_cos(Xn)
            if s < best + 0.002:
                b = min(5.0, deadline - 30.0 - time.time())
                if b < 1.0:
                    if s < best - 1e-12:
                        Xbest, best = Xn, s
                    continue
                try:
                    Xp = lp_polish_budget(Xn, b)
                except Exception:
                    continue
                sp = max_cos(Xp)
                if sp < best - 1e-12:
                    Xbest, best = Xp, sp
        # safety: stop if time low
        if time.time() >= deadline - 30.0:
            break

    # final polish with remaining time
    try:
        fb = max(2.0, deadline - time.time() - 1.0)
        if fb > 0.5:
            Xbest = lp_polish_budget(Xbest, min(fb, 28.0))
    except Exception:
        pass
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    # strict finite check
    Xbest = np.where(np.isfinite(Xbest), Xbest, 0.0)
    nrm = np.linalg.norm(Xbest, axis=1, keepdims=True)
    nrm = np.where(nrm == 0, 1.0, nrm)
    Xbest = Xbest / nrm
    return Xbest
