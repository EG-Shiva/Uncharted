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
    a_ratio = a1 / a0 if a0 != 0 else 1.0
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
        nrm = np.linalg.norm(X, axis=1, keepdims=True)
        nrm = np.where(nrm == 0, 1.0, nrm)
        X = X / nrm
    return X


def refine_hole(y0, Xall, steps=250, beta=120.0):
    y = y0.copy()
    for it in range(steps):
        dots = Xall @ y
        m = float(np.max(dots))
        w = np.exp(beta * (dots - m))
        s = float(np.sum(w))
        if s == 0 or not np.isfinite(s):
            break
        w /= s
        g = w @ Xall
        gt = g - float(np.dot(g, y)) * y
        lr = 0.03 * (1.0 - it / steps) + 0.002 * (it / steps)
        y = y - lr * gt
        nrm = float(np.linalg.norm(y))
        if nrm == 0 or not np.isfinite(nrm):
            break
        y /= nrm
    return y


def joint_insert_opt(Y, Xf, iters=700, a0=60.0, a1=450.0, lr0=0.04, lr1=0.002):
    m = Y.shape[0]
    if m == 0:
        return Y
    lr_ratio = lr1 / lr0
    a_ratio = a1 / a0
    for it in range(iters):
        frac = it / max(iters - 1, 1)
        alpha = a0 * (a_ratio ** frac)
        lr = lr0 * (lr_ratio ** frac)
        D1 = Y @ Xf.T
        D2 = Y @ Y.T
        np.fill_diagonal(D2, -2.0)
        mx1 = float(np.max(D1)) if D1.size else -2.0
        mx2 = float(np.max(D2)) if D2.size else -2.0
        mall = mx1 if mx1 > mx2 else mx2
        W1 = np.exp(alpha * (D1 - mall))
        W2 = np.exp(alpha * (D2 - mall))
        np.fill_diagonal(W2, 0.0)
        Grad = W1 @ Xf + W2 @ Y
        dot = np.sum(Grad * Y, axis=1, keepdims=True)
        Gt = Grad - dot * Y
        rms = float(np.sqrt(np.mean(np.sum(Gt ** 2, axis=1)))) + 1e-12
        Y = Y - lr * (Gt / rms)
        nrm = np.linalg.norm(Y, axis=1, keepdims=True)
        nrm = np.where(nrm == 0, 1.0, nrm)
        Y = Y / nrm
    return Y


def build_insert(Xf, Ydone, rng, n_cand=16000):
    d = Xf.shape[1]
    Y = rng.standard_normal((n_cand, d))
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    S = Y @ Xf.T
    sc = np.max(S, axis=1)
    if Ydone is not None and len(Ydone) > 0:
        S2 = Y @ Ydone.T
        sc2 = np.max(S2, axis=1)
        sc = np.maximum(sc, sc2)
    k = min(5, n_cand)
    idx = np.argpartition(sc, k - 1)[:k]
    if Ydone is not None and len(Ydone) > 0:
        Xall = np.vstack([Xf, Ydone])
    else:
        Xall = Xf
    best_y = None
    best_s = 1e100
    for j in idx:
        y = refine_hole(Y[j], Xall, steps=250)
        s = float(np.max(Xall @ y))
        if s < best_s:
            best_s = s
            best_y = y
    return best_y


def reinsert_m(Xbest, worst_idx, rng):
    mask = np.ones(Xbest.shape[0], dtype=bool)
    mask[worst_idx] = False
    Xf = Xbest[mask]
    Ydone = []
    for _ in range(len(worst_idx)):
        Yarr = np.vstack(Ydone) if len(Ydone) > 0 else np.empty((0, Xf.shape[1]))
        ynew = build_insert(Xf, Yarr, rng, n_cand=16000)
        Ydone.append(ynew)
    Y0 = np.vstack(Ydone)
    Y0 = joint_insert_opt(Y0, Xf, iters=700)
    Xn = np.vstack([Xf, Y0])
    return Xn


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
        if p > 4000:
            # keep only tightest 4000 to keep HiGHS fast
            order = np.argpartition(-c[sel], 4000 - 1)[:4000]
            I, J = I[order], J[order]
            p = 4000
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
                obj, A_ub=A_ub, b_ub=-(X[I] * X[J]).sum(axis=1),
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
    rng = np.random.default_rng(7)
    try:
        Xb = np.load("best.npy").astype(float)
    except Exception:
        Xb = np.load("record.npy").astype(float)
    Xb = Xb / np.linalg.norm(Xb, axis=1, keepdims=True)
    best = max_cos(Xb)
    Xbest = Xb.copy()
    trial = 0
    while time.time() < deadline - 28.0:
        trial += 1
        if time.time() > deadline - 29.0:
            break
        mode = trial % 5
        try:
            if mode == 0:
                sigma = 0.028 if (trial % 20) < 14 else 0.055
                Xn = Xbest + sigma * rng.standard_normal(Xbest.shape)
                Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
                Xn = smooth_descend(Xn, 1700, 35.0, 380.0, 0.02, 0.0007)
            elif mode == 1:
                G = Xbest @ Xbest.T
                np.fill_diagonal(G, -2.0)
                rowmax = np.max(G, axis=1)
                w = int(np.argmax(rowmax))
                nbrs = np.argsort(-G[w])[:12]
                Xn = Xbest.copy()
                Xn[nbrs] += 0.32 * rng.standard_normal((len(nbrs), Xbest.shape[1]))
                Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
                Xn = smooth_descend(Xn, 1700, 40.0, 420.0, 0.018, 0.0006)
            elif mode == 2:
                G = Xbest @ Xbest.T
                np.fill_diagonal(G, -2.0)
                rowmax = np.max(G, axis=1)
                worst = np.argsort(-rowmax)[:4]
                Xn = reinsert_m(Xbest, worst, rng)
                Xn = smooth_descend(Xn, 1400, 40.0, 380.0, 0.015, 0.0007)
            elif mode == 3:
                sigma = 0.075
                Xn = Xbest + sigma * rng.standard_normal(Xbest.shape)
                Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
                Xn = smooth_descend(Xn, 2000, 30.0, 350.0, 0.022, 0.0006)
            else:
                G = Xbest @ Xbest.T
                np.fill_diagonal(G, -2.0)
                rowmax = np.max(G, axis=1)
                msz = 7 if (trial % 10) < 5 else 5
                worst = np.argsort(-rowmax)[:msz]
                Xn = reinsert_m(Xbest, worst, rng)
                Xn = smooth_descend(Xn, 1500, 45.0, 450.0, 0.015, 0.0006)
        except Exception:
            continue
        try:
            s = max_cos(Xn)
        except Exception:
            continue
        # LP only if promising to save time
        thresh = 0.0012
        if mode in (2, 4):
            thresh = 0.0016
        do_lp = (s < best + thresh) and (s < best + 0.002)
        # throttle LP: skip every other non-improving
        if do_lp and s >= best and (trial % 2 == 1):
            do_lp = False
        if do_lp:
            b = min(1.8, deadline - 28.0 - time.time())
            if b < 0.6:
                if s < best - 1e-12:
                    Xbest, best = Xn, s
                continue
            try:
                Xp = lp_polish_budget(Xn, b)
                sp = max_cos(Xp)
                if sp < best - 1e-12:
                    Xbest, best = Xp, sp
                elif s < best - 1e-12:
                    Xbest, best = Xn, s
            except Exception:
                if s < best - 1e-12:
                    Xbest, best = Xn, s
                continue
        else:
            if s < best - 1e-12:
                # quick squeeze with tiny LP if clearly better
                b = min(1.2, deadline - 28.0 - time.time())
                if b > 0.6:
                    try:
                        Xp = lp_polish_budget(Xn, b)
                        sp = max_cos(Xp)
                        if sp < s:
                            Xn, s = Xp, sp
                    except Exception:
                        pass
                Xbest, best = Xn, s
        if time.time() >= deadline - 28.0:
            break
    try:
        fb = max(2.0, deadline - time.time() - 1.0)
        if fb > 0.5:
            Xbest = lp_polish_budget(Xbest, min(fb, 26.0))
    except Exception:
        pass
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    Xbest = np.where(np.isfinite(Xbest), Xbest, 0.0)
    nrm = np.linalg.norm(Xbest, axis=1, keepdims=True)
    nrm = np.where(nrm == 0, 1.0, nrm)
    Xbest = Xbest / nrm
    return Xbest
