import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def row_max_deg(x, m=None, tol=1e-7):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    if m is None:
        m = float(G.max())
    deg = (G > m - tol).sum(axis=1)
    rm = G.max(axis=1)
    return rm, deg, G, m

def smooth_anneal(x, steps, T0, T1, step0, step1):
    n, d = x.shape
    x = x.copy()
    for s in range(steps):
        frac = s / max(1, steps - 1)
        # geometric T schedule
        if T0 == T1:
            T = T0
        else:
            T = T0 * ((T1 / T0) ** frac)
        step = step0 + (step1 - step0) * frac
        G = x @ x.T
        np.fill_diagonal(G, -1e9)
        m = G.max()
        W = np.exp((G - m) / T)
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        grad = (W @ x) / rs
        gdot = (grad * x).sum(axis=1, keepdims=True)
        gt = grad - gdot * x
        x = x - step * gt
        xn = np.linalg.norm(x, axis=1, keepdims=True)
        xn[xn == 0] = 1.0
        x = x / xn
    return x

def lp_polish(x, deadline, radius0=2e-3, cap=2000):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = radius0
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        m = c_all.max()
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > cap:
            idx = np.where(sel)[0]
            # keep tightest
            part = np.argpartition(c_all[idx], -cap)[-cap:]
            msel = np.zeros_like(sel, dtype=bool)
            msel[idx[part]] = True
            sel = msel
        I, J = iu0[0][sel], iu0[1][sel]
        csel = c_all[sel]
        p = len(I)
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

def best_hole(x, exclude=None, nsamp=20000, rng=None):
    n, d = x.shape
    samp = rng.standard_normal((nsamp, d))
    samp /= np.linalg.norm(samp, axis=1, keepdims=True)
    if exclude is None:
        others = x
    else:
        mask = np.ones(n, dtype=bool)
        mask[np.asarray(exclude)] = False
        others = x[mask]
    scores = samp @ others.T
    s_max = scores.max(axis=1)
    bi = int(np.argmin(s_max))
    return samp[bi], float(s_max[bi])

def solve():
    t0 = time.time()
    BUDGET = 168.0
    rng = np.random.default_rng(7)
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
    # ensure local opt briefly
    x_best, best = lp_polish(x_best, time.time() + 3.0, radius0=1e-3)
    x_cur = x_best.copy()
    cur = best
    n, d = x_best.shape
    it = 0
    while True:
        elapsed = time.time() - t0
        remain = BUDGET - elapsed
        if remain < 7.0:
            break
        it += 1
        # basin temperature annealed
        frac = elapsed / BUDGET
        Tbasin = 0.004 * (0.15 ** frac) + 1e-4
        x_try = x_cur.copy()
        r = rng.random()
        if r < 0.22:
            # global small-medium shake, wander
            sigma = float(rng.choice([0.03, 0.06, 0.10, 0.16]))
            x_try = x_try + sigma * rng.standard_normal(x_try.shape)
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            x_try = smooth_anneal(x_try, 450, 0.025, 0.003, 0.4, 0.12)
        elif r < 0.42:
            # attack overloaded points
            rm, deg, G, m = row_max_deg(x_try, tol=1e-7)
            k = int(rng.integers(8, 20))
            worst = np.argsort(rm)[-k:]
            sigma = 0.25 + 0.15 * rng.random()
            x_try[worst] += sigma * rng.standard_normal((k, d))
            xn = np.linalg.norm(x_try[worst], axis=1, keepdims=True)
            xn[xn == 0] = 1.0
            x_try[worst] /= xn
            x_try = smooth_anneal(x_try, 550, 0.03, 0.0025, 0.42, 0.12)
        elif r < 0.62:
            # relocate 1-3 worst points into largest voids
            rm, deg, G, m = row_max_deg(x_try, tol=1e-7)
            nw = int(rng.integers(1, 4))
            worst = np.argsort(rm)[-nw:]
            for w in worst:
                h, _ = best_hole(x_try, exclude=[w], nsamp=15000, rng=rng)
                # slight jitter around hole
                h = h + 0.02 * rng.standard_normal(d)
                h /= np.linalg.norm(h)
                x_try[w] = h
            # jitter neighbours of first worst to open topology
            try:
                G2 = x_try @ x_try.T
                np.fill_diagonal(G2, -2)
                nbrs = np.argsort(G2[worst[-1]])[::-1][:5]
                x_try[nbrs] += 0.1 * rng.standard_normal((len(nbrs), d))
                x_try[nbrs] /= np.linalg.norm(x_try[nbrs], axis=1, keepdims=True)
            except Exception:
                pass
            x_try = smooth_anneal(x_try, 550, 0.02, 0.002, 0.35, 0.10)
        elif r < 0.80:
            # random block reinit for topology change
            k = int(rng.integers(5, 16))
            idx = rng.choice(n, size=k, replace=False)
            # bias: include some worst points
            if rng.random() < 0.7:
                rm, deg, G, m = row_max_deg(x_try, tol=1e-7)
                worst5 = np.argsort(rm)[-5:]
                idx[:min(5, k)] = worst5[:min(5, k)]
            rnd = rng.standard_normal((k, d))
            rnd /= np.linalg.norm(rnd, axis=1, keepdims=True)
            x_try[idx] = rnd
            x_try = smooth_anneal(x_try, 650, 0.032, 0.0025, 0.45, 0.12)
        else:
            # equilibrate rattlers / low-degree points: move all deg<=2 to voids
            rm, deg, G, m = row_max_deg(x_try, tol=1e-7)
            low = np.where(deg <= 2)[0]
            if len(low) == 0:
                low = np.argsort(rm)[-3:]
            # limit to at most 8
            if len(low) > 8:
                # pick worst among low? pick lowest degree then highest rm
                order = np.argsort(rm[low])[::-1][:8]
                low = low[order]
            for w in low:
                h, _ = best_hole(x_try, exclude=[w], nsamp=12000, rng=rng)
                x_try[w] = h
            # light shake of overloaded to accommodate
            rm2, deg2, _, _ = row_max_deg(x_try, tol=1e-7)
            hi = np.argsort(rm2)[-6:]
            x_try[hi] += 0.06 * rng.standard_normal((len(hi), d))
            x_try[hi] /= np.linalg.norm(x_try[hi], axis=1, keepdims=True)
            x_try = smooth_anneal(x_try, 500, 0.018, 0.002, 0.32, 0.10)

        v_smooth = max_cos(x_try)
        # prune hopeless basins to save LP time
        if v_smooth > cur + 0.025 and v_smooth > best + 0.01:
            # still allow occasional wander
            if rng.random() > 0.05:
                continue
        lp_budget = min(1.6, BUDGET - (time.time() - t0) - 5.0)
        if lp_budget < 0.6:
            break
        x_pol, v_pol = lp_polish(x_try, time.time() + lp_budget, radius0=3e-3, cap=2000)
        if v_pol < best - 1e-12:
            x_best, best = x_pol, v_pol
            x_cur, cur = x_pol.copy(), v_pol
        else:
            # Metropolis acceptance for wandering (allow uphill)
            if v_pol < cur or rng.random() < np.exp((cur - v_pol) / max(Tbasin, 1e-9)):
                x_cur, cur = x_pol.copy(), v_pol
        # opportunistic extra equilibration: if we sit at jam, push rattlers without LP cost
        # (handled in proposal above)

    remain = BUDGET - (time.time() - t0)
    if remain > 2.0:
        x_best, best = lp_polish(x_best, time.time() + remain - 0.6, radius0=5e-4, cap=2500)
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
