import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
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
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def _anneal(X, steps, a0, a1, lr0, lr1):
    X = X.copy()
    for s in range(steps):
        f = s / max(steps - 1, 1)
        alpha = a0 * (a1 / a0) ** f
        lr = lr0 * (lr1 / lr0) ** f
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        gmax = G.max()
        E = np.exp(alpha * (G - gmax))
        np.fill_diagonal(E, 0.0)
        rowsum = E.sum(axis=1, keepdims=True) + 1e-12
        grad = (E @ X) / rowsum
        dot = (grad * X).sum(axis=1, keepdims=True)
        gt = grad - dot * X
        X -= lr * gt
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def _contacts(X, tol=2e-4):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    mv = G.max()
    cnt = (G > mv - tol).sum(axis=1)
    return mv, cnt, G


def _reinsert(Xb, idx, rng, n_cand=4000):
    n, d = Xb.shape
    Y = Xb.copy()
    # fixed set = all except idx
    mask = np.ones(n, dtype=bool)
    mask[idx] = False
    fixed = Xb[mask]
    placed = []
    for _ in range(len(idx)):
        cand = rng.standard_normal((n_cand, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        mf = cand @ fixed.T
        m = mf.max(axis=1) if mf.shape[1] > 0 else np.full(n_cand, -1e9)
        if placed:
            P = np.stack(placed)
            mp = cand @ P.T
            m = np.maximum(m, mp.max(axis=1))
        bi = int(np.argmin(m))
        xn = cand[bi].copy()
        # quick single-point refinement
        for __ in range(150):
            if fixed.shape[0] + len(placed) == 0:
                break
            if placed:
                All = np.vstack([fixed, np.stack(placed)])
            else:
                All = fixed
            dots = All @ xn
            am = dots.max()
            w = np.exp(40.0 * (dots - am))
            g = (w @ All) / (w.sum() + 1e-12)
            gt = g - (g @ xn) * xn
            xn = xn - 0.06 * gt
            xn /= np.linalg.norm(xn)
        placed.append(xn)
    Y[np.asarray(idx)] = np.stack(placed)
    return Y


def solve():
    t0 = time.time()
    TIME_LIMIT = 176.0
    rng = np.random.default_rng(0)
    # load starting points
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (76, 15):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
        a = rng.standard_normal((76, 15))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    vals = [_max_cos(a) for a in cands]
    bi = int(np.argmin(vals))
    Xbest = cands[bi].copy()
    best_val = float(vals[bi])

    it = 0
    # main basin-hopping loop, reserve ~14s for final polish
    while time.time() - t0 < TIME_LIMIT - 14.0:
        it += 1
        elapsed = time.time() - t0
        remain = TIME_LIMIT - 14.0 - elapsed
        if remain <= 0:
            break
        strat = it % 6
        Y = Xbest.copy()
        try:
            mv, cnt, _ = _contacts(Xbest)
        except Exception:
            mv, cnt = best_val, np.zeros(76)
        if strat == 0:
            Y = Y + 0.015 * rng.standard_normal(Y.shape)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            Y = _anneal(Y, 700, 80, 450, 0.05, 0.005)
        elif strat == 1:
            Y = Y + 0.045 * rng.standard_normal(Y.shape)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            Y = _anneal(Y, 900, 60, 350, 0.06, 0.006)
        elif strat == 2:
            k = 5 + (it % 6)
            idx = rng.choice(76, k, replace=False)
            Y[idx] = rng.standard_normal((k, 15))
            Y[idx] /= np.linalg.norm(Y[idx], axis=1, keepdims=True)
            Y = _anneal(Y, 1000, 60, 400, 0.06, 0.005)
        elif strat == 3:
            ratt = np.where(cnt <= 1)[0]
            if len(ratt) < 2:
                # fallback: shake jammed
                order = np.argsort(-cnt)
                ratt = order[:6]
            # limit number reinserted
            if len(ratt) > 10:
                # pick worst rattlers randomly
                ratt = rng.choice(ratt, 10, replace=False)
            Y = _reinsert(Xbest, np.sort(ratt), rng)
            # small global jitter
            Y = Y + 0.008 * rng.standard_normal(Y.shape)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            Y = _anneal(Y, 700, 90, 500, 0.04, 0.004)
        elif strat == 4:
            order = np.argsort(-cnt)
            k = 8
            idx = order[:k]
            Y[idx] = Y[idx] + 0.09 * rng.standard_normal((k, 15))
            Y[idx] /= np.linalg.norm(Y[idx], axis=1, keepdims=True)
            # also jitter rattlers a bit
            ratt = np.where(cnt <= 1)[0]
            if len(ratt) > 0:
                Y[ratt] = Y[ratt] + 0.02 * rng.standard_normal((len(ratt), 15))
                Y[ratt] /= np.linalg.norm(Y[ratt], axis=1, keepdims=True)
            Y = _anneal(Y, 900, 70, 400, 0.055, 0.005)
        else:
            # random orthogonal-ish global resample with farthest greedy from scratch mix
            Y = Y + 0.025 * rng.standard_normal(Y.shape)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            Y = _anneal(Y, 600, 50, 300, 0.07, 0.008)

        v = _max_cos(Y)
        # skip hopeless trials to save LP time
        if v > best_val + 0.035:
            continue
        # short LP polish for evaluation
        Yp = _lp_polish(Y, 2.2)
        vp = _max_cos(Yp)
        if vp < best_val - 1e-12:
            Xbest = Yp
            best_val = vp
            # quick extra squeeze when improved
            if time.time() - t0 < TIME_LIMIT - 20:
                Xbest = _lp_polish(Xbest, 1.5)
                best_val = _max_cos(Xbest)
        # time guard
        if time.time() - t0 >= TIME_LIMIT - 14.0:
            break

    # final squeeze with remaining time
    rem = TIME_LIMIT - (time.time() - t0) - 1.0
    if rem > 2.0:
        Xbest = _lp_polish(Xbest, min(rem, 13.0))
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest
