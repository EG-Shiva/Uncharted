import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def lp_polish_deadline(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        iu_c = G[iu]
        m = float(np.max(iu_c))
        sel = iu_c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        # cap constraints for speed
        if p > 2500:
            # keep tightest 2500
            order = np.argsort(-iu_c[sel])[:2500]
            I, J = I[order], J[order]
            p = 2500
            csel = iu_c[sel][order]
        else:
            csel = iu_c[sel]
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
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
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
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() >= deadline:
            break
    return x


def smooth_refine(X, steps, a0, a1, lr0, deadline=1e30):
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    la0, la1 = np.log(a0), np.log(a1)
    for t in range(1, steps + 1):
        if t % 200 == 0 and time.time() > deadline:
            break
        frac = t / steps
        alpha = np.exp(la0 + (la1 - la0) * frac)
        lr = lr0 * (1.0 - 0.7 * frac)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = np.max(G)
        E = np.exp(alpha * (G - M))
        np.fill_diagonal(E, 0.0)
        S = np.sum(E)
        if not np.isfinite(S) or S == 0:
            break
        Grad = (2.0 / S) * (E @ X)
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        m *= b1
        m += (1.0 - b1) * Gt
        v *= b2
        v += (1.0 - b2) * (Gt * Gt)
        c1 = 1.0 - b1 ** t
        c2 = 1.0 - b2 ** t
        mhat = m / c1
        vhat = v / c2
        step = mhat / (np.sqrt(vhat) + eps)
        X = X - lr * step
        X /= np.sqrt(np.sum(X * X, axis=1, keepdims=True))
    return X


def greedy_refill(Xcur, idx_replace, rng, n_cand=6000):
    n, d = Xcur.shape
    Xtry = Xcur.copy()
    idx_set = list(idx_replace)
    # fixed mask evolves as we place
    remaining = set(idx_set)
    for pos in idx_set:
        remaining.discard(pos)
        # fixed indices = all not in remaining and not pos? include already placed
        keep = [j for j in range(n) if (j not in remaining and j != pos)]
        F = Xtry[keep]  # (nf, d)
        C = rng.standard_normal((n_cand, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        # max cosine to fixed set
        Mc = C @ F.T
        vals = np.max(Mc, axis=1)
        bi = int(np.argmin(vals))
        Xtry[pos] = C[bi]
    return Xtry


def solve():
    t0 = time.time()
    HOP_END = t0 + 143
    FINAL_END = t0 + 174
    rng = np.random.default_rng(7)
    n, d = 154, 10

    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    for c in (xb, xr):
        if c is not None and c.shape == (n, d):
            c = c / np.linalg.norm(c, axis=1, keepdims=True)
            if np.all(np.isfinite(c)):
                cands.append(c)
    if len(cands) == 0:
        Xbest = rng.standard_normal((n, d))
        Xbest /= np.linalg.norm(Xbest, axis=1, keepdims=True)
    else:
        vals = [max_cos(c) for c in cands]
        Xbest = cands[int(np.argmin(vals))].copy()
    best_val = max_cos(Xbest)
    Xcur = Xbest.copy()
    cur_val = best_val

    trial = 0
    stale = 0
    while time.time() < HOP_END:
        trial += 1
        # occasionally restart walker from best
        if stale > 25:
            Xcur = Xbest.copy()
            cur_val = best_val
            stale = 0
        # stats of walker
        Gc = Xcur @ Xcur.T
        np.fill_diagonal(Gc, -2.0)
        rowmax = np.max(Gc, axis=1)
        order = np.argsort(rowmax)  # loosest first
        deg = np.sum(Gc > cur_val - 0.005, axis=1)

        u = rng.random()
        if u < 0.30:
            # small global jitter of walker
            sigma = float(rng.choice([0.04, 0.06, 0.09]))
            Xtry = Xcur + sigma * rng.standard_normal((n, d))
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            steps, a0, a1, lr0 = 900, 30.0, 450.0, 0.015
        elif u < 0.55:
            # rattler void refill: replace loosest k
            k = int(rng.integers(4, 10))
            idx = order[:k]
            Xtry = greedy_refill(Xcur, idx, rng, n_cand=6000)
            # tiny shake to break symmetry
            Xtry += 0.008 * rng.standard_normal((n, d))
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            steps, a0, a1, lr0 = 1100, 25.0, 550.0, 0.014
        elif u < 0.75:
            # overloaded cluster shake
            k_top = 6
            idx_tight = np.argsort(-deg)[:k_top]
            # add neighbours of tightest point
            tight0 = idx_tight[0]
            neigh = np.argsort(-Gc[tight0])[:4]
            idx = np.unique(np.concatenate([idx_tight, neigh, order[:2]]))
            Xtry = Xcur.copy()
            sig = 0.18 + 0.1 * rng.random()
            Xtry[idx] += sig * rng.standard_normal((len(idx), d))
            Xtry += 0.008 * rng.standard_normal((n, d))
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            steps, a0, a1, lr0 = 1100, 25.0, 550.0, 0.014
        elif u < 0.92:
            # partial random reset biased to high degree
            prob = (deg + 1.0).astype(float)
            prob /= prob.sum()
            k = int(rng.integers(7, 15))
            idx = rng.choice(n, size=k, replace=False, p=prob)
            Xtry = Xcur.copy()
            Xtry[idx] = rng.standard_normal((k, d))
            Xtry += 0.01 * rng.standard_normal((n, d))
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            steps, a0, a1, lr0 = 1300, 22.0, 500.0, 0.015
        else:
            # full random fresh basin (rare)
            Xtry = rng.standard_normal((n, d))
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            steps, a0, a1, lr0 = 1700, 18.0, 500.0, 0.02

        dl = min(HOP_END, time.time() + 8)
        Xref = smooth_refine(Xtry, steps, a0, a1, lr0, deadline=dl)
        val = max_cos(Xref)

        if val < cur_val - 1e-12:
            Xcur, cur_val = Xref, val
            stale = 0
        elif val < cur_val + 0.004 and rng.random() < 0.15:
            Xcur, cur_val = Xref, val
            stale += 1
        else:
            stale += 1

        if val < best_val - 1e-12:
            # snap to local optimum
            tl = FINAL_END - (time.time())
            if tl < 2:
                Xbest, best_val = Xref, val
                Xcur, cur_val = Xref, val
            else:
                sec = min(3.5, max(1.2, tl / 15.0))
                Xpol = lp_polish_deadline(Xref, time.time() + sec)
                vpol = max_cos(Xpol)
                if vpol < best_val:
                    Xbest, best_val = Xpol, vpol
                    Xcur, cur_val = Xpol, vpol
                else:
                    # keep refined even if polish failed to beat (still better than best)
                    if val < best_val:
                        Xbest, best_val = Xref, val
        # opportunistic polish of near-best walker
        if trial % 25 == 0 and time.time() < HOP_END - 5:
            if cur_val < best_val + 0.002 and cur_val > best_val:
                Xpol = lp_polish_deadline(Xcur, time.time() + 1.5)
                vpol = max_cos(Xpol)
                if vpol < best_val:
                    Xbest, best_val = Xpol, vpol
                    Xcur, cur_val = Xpol, vpol
        if time.time() >= HOP_END:
            break

    if time.time() < FINAL_END - 2:
        Xbest = lp_polish_deadline(Xbest, FINAL_END - 0.5)
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest
