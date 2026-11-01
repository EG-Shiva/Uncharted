import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def lp_polish(x, deadline, radius0=1e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = radius0
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu]
        m = c_all.max()
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c_all > thresh
        # cap size for speed
        if np.count_nonzero(sel) > 2500:
            # keep tightest 2500
            idx = np.where(sel)[0]
            order = np.argsort(c_all[idx])[::-1][:2500]
            msel = np.zeros_like(sel, dtype=bool)
            msel[idx[order]] = True
            sel = msel
        I, J = iu[0][sel], iu[1][sel]
        csel = c_all[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
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
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best


def smooth_anneal(x, steps, T0, T1, step0, step1):
    n, d = x.shape
    x = x.copy()
    for s in range(steps):
        frac = s / max(1, steps - 1)
        T = T0 * ((T1 / T0) ** frac) if T0 != T1 else T0
        step = step0 + (step1 - step0) * frac
        G = x @ x.T
        np.fill_diagonal(G, -1e9)
        m = G.max()
        # weights, stable (<=1)
        W = np.exp((G - m) / T)
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        grad = (W @ x) / rs
        gdot = (grad * x).sum(axis=1, keepdims=True)
        gt = grad - gdot * x
        # clip extreme
        x = x - step * gt
        xn = np.linalg.norm(x, axis=1, keepdims=True)
        xn[xn == 0] = 1.0
        x = x / xn
    return x


def row_maxima(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1), G


def solve():
    t0 = time.time()
    BUDGET = 168.0
    rng = np.random.default_rng(12345)
    # load starting points
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (173, 5):
                nrm = np.linalg.norm(a, axis=1, keepdims=True)
                nrm[nrm == 0] = 1.0
                a = a / nrm
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if len(cands) == 0:
        a = rng.standard_normal((173, 5))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    x_best = None
    best = 2.0
    for a in cands:
        v = max_cos(a)
        if v < best:
            best = v
            x_best = a.copy()
    # short initial LP to ensure local optimum (limited)
    if time.time() - t0 < BUDGET - 10:
        x_best, best = lp_polish(x_best, time.time() + 4.0, radius0=1e-3)

    it = 0
    sigmas = [0.008, 0.02, 0.05, 0.12, 0.25, 0.45]
    while True:
        remain = BUDGET - (time.time() - t0)
        if remain < 6.0:
            break
        it += 1
        mode = it % 6
        x_try = x_best.copy()
        n, d = x_try.shape
        if mode == 0:
            # global tiny shake
            sigma = sigmas[it % len(sigmas)]
            x_try = x_try + sigma * rng.standard_normal(x_try.shape)
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            x_try = smooth_anneal(x_try, 400, 0.02, 0.003, 0.35, 0.15)
        elif mode == 1:
            # shake worst points strongly
            rm, G = row_maxima(x_try)
            k = 18
            worst = np.argsort(rm)[-k:]
            sigma = 0.18 + 0.1 * ((it // 6) % 3)
            x_try[worst] += sigma * rng.standard_normal((k, d))
            xn = np.linalg.norm(x_try[worst], axis=1, keepdims=True)
            xn[xn == 0] = 1.0
            x_try[worst] /= xn
            x_try = smooth_anneal(x_try, 600, 0.025, 0.002, 0.4, 0.12)
        elif mode == 2:
            # move overloaded into rattler hole: pick worst point, reinit near hole
            rm, G = row_maxima(x_try)
            worst = int(np.argmax(rm))
            # find hole by sampling
            S = 20000
            samp = rng.standard_normal((S, d))
            samp /= np.linalg.norm(samp, axis=1, keepdims=True)
            # score = max dot to others excluding worst
            others = np.delete(x_try, worst, axis=0)
            scores = samp @ others.T
            s_max = scores.max(axis=1)
            bi = int(np.argmin(s_max))
            x_try[worst] = samp[bi]
            # also jitter neighbours of worst
            G2 = x_try @ x_try.T
            np.fill_diagonal(G2, -2)
            nbrs = np.argsort(G2[worst])[::-1][:6]
            x_try[nbrs] += 0.08 * rng.standard_normal((len(nbrs), d))
            x_try[nbrs] /= np.linalg.norm(x_try[nbrs], axis=1, keepdims=True)
            x_try = smooth_anneal(x_try, 500, 0.015, 0.002, 0.3, 0.1)
        elif mode == 3:
            # medium global shake + coarse-fine smooth
            sigma = 0.06
            x_try = x_try + sigma * rng.standard_normal(x_try.shape)
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            x_try = smooth_anneal(x_try, 800, 0.03, 0.003, 0.4, 0.15)
        elif mode == 4:
            # random subset re-init (topology change)
            k = 8 + (it % 8)
            idx = rng.choice(n, size=k, replace=False)
            rnd = rng.standard_normal((k, d))
            rnd /= np.linalg.norm(rnd, axis=1, keepdims=True)
            # mix: half random, half jitter
            x_try[idx] = rnd
            x_try = smooth_anneal(x_try, 700, 0.03, 0.002, 0.45, 0.12)
        else:
            # large global kick, then strong anneal
            sigma = 0.3
            x_try = x_try + sigma * rng.standard_normal(x_try.shape)
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            x_try = smooth_anneal(x_try, 900, 0.04, 0.004, 0.5, 0.15)

        # quick check: if smooth result already much worse, still polish briefly
        # LP polish with small budget
        lp_budget = min(4.0, remain - 2.0)
        if lp_budget < 1.0:
            break
        x_pol, v_pol = lp_polish(x_try, time.time() + lp_budget, radius0=3e-3)
        if v_pol < best - 1e-12:
            x_best, best = x_pol, v_pol
        # also opportunistic: if equal within noise but different structure, keep occasionally to diversify
        elif v_pol < best + 1e-7 and (it % 11 == 0):
            # sideways move to escape basin without losing much
            if rng.random() < 0.15:
                x_best, best = x_pol, v_pol

    # final squeeze with remaining time
    remain = BUDGET - (time.time() - t0)
    if remain > 2.0:
        x_best, best = lp_polish(x_best, time.time() + remain - 0.5, radius0=5e-4)
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
