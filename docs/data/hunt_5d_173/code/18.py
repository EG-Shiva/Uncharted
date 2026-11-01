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
        if np.count_nonzero(sel) > 2500:
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

def mom_anneal(x, steps, T0, T1, step0, step1, mom=0.85):
    n, d = x.shape
    x = x.copy()
    v = np.zeros_like(x)
    for s in range(steps):
        frac = s / max(1, steps - 1)
        T = T0 * ((T1 / T0) ** frac) if T0 != T1 else T0
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
        # clip huge
        nrm = np.linalg.norm(gt, axis=1, keepdims=True)
        # avoid blow up: scale down if very large (should be bounded ~2)
        v = mom * v + step * gt
        x = x - v
        xn = np.linalg.norm(x, axis=1, keepdims=True)
        xn[xn == 0] = 1.0
        x = x / xn
    return x

def refine_single(x, idx, steps=200, lr=0.05, topk=6):
    # optimize point idx alone, others fixed: minimize max dot
    n, d = x.shape
    others = np.delete(x, idx, axis=0)
    p = x[idx:idx+1].copy()
    for s in range(steps):
        dots = (others @ p[0])
        # active set: topk largest
        if topk < len(dots):
            part = np.argpartition(dots, -topk)[-topk:]
        else:
            part = np.arange(len(dots))
        g = others[part].mean(axis=0, keepdims=True)
        # project tangent at p
        gdot = (g * p).sum()
        gt = g - gdot * p
        st = lr * (0.2 + 0.8 * (1 - s / steps))
        p = p - st * gt
        p /= np.linalg.norm(p)
    x2 = x.copy()
    x2[idx] = p[0]
    return x2

def hole_sample(x, idx, rng, S=20000):
    n, d = x.shape
    samp = rng.standard_normal((S, d))
    samp /= np.linalg.norm(samp, axis=1, keepdims=True)
    others = np.delete(x, idx, axis=0)
    scores = samp @ others.T
    s_max = scores.max(axis=1)
    bi = int(np.argmin(s_max))
    return samp[bi]

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
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
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
    # ensure locally optimal start with short polish
    x_best, best = lp_polish(x_best, t0 + 4.0, radius0=1e-3)
    n, d = x_best.shape
    it = 0
    # cheap screening tolerance for LP
    while True:
        remain = BUDGET - (time.time() - t0)
        if remain < 8.0:
            break
        it += 1
        mode = it % 5
        x_try = x_best.copy()
        if mode == 0:
            # tiny global + momentum anneal (basin explore)
            sigma = 0.015 + 0.01 * ((it // 5) % 3)
            x_try = x_try + sigma * rng.standard_normal(x_try.shape)
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            x_try = mom_anneal(x_try, 500, 0.018, 0.0025, 0.30, 0.10, mom=0.82)
        elif mode == 1:
            # worst-cluster shake
            G = x_try @ x_try.T
            np.fill_diagonal(G, -2.0)
            rm = G.max(axis=1)
            k = 14 + (it % 7)
            worst = np.argsort(rm)[-k:]
            sigma = 0.22
            x_try[worst] += sigma * rng.standard_normal((k, d))
            x_try[worst] /= np.linalg.norm(x_try[worst], axis=1, keepdims=True)
            x_try = mom_anneal(x_try, 600, 0.022, 0.002, 0.35, 0.10, mom=0.87)
        elif mode == 2:
            # hole jump: move worst point to best hole + refine neighbours
            G = x_try @ x_try.T
            np.fill_diagonal(G, -2.0)
            rm = G.max(axis=1)
            worst = int(np.argmax(rm))
            hp = hole_sample(x_try, worst, rng, S=20000)
            x_try[worst] = hp
            # jitter its new neighbours slightly
            G2 = x_try @ x_try.T
            np.fill_diagonal(G2, -2.0)
            nbrs = np.argsort(G2[worst])[::-1][:5]
            x_try[nbrs] += 0.06 * rng.standard_normal((len(nbrs), d))
            x_try[nbrs] /= np.linalg.norm(x_try[nbrs], axis=1, keepdims=True)
            # single-point refine worst then global anneal
            x_try = refine_single(x_try, worst, steps=150, lr=0.08, topk=8)
            x_try = mom_anneal(x_try, 450, 0.012, 0.0018, 0.28, 0.08, mom=0.85)
        elif mode == 3:
            # subset reinit for topology change
            k = 6 + (it % 10)
            idx = rng.choice(n, size=k, replace=False)
            rnd = rng.standard_normal((k, d))
            rnd /= np.linalg.norm(rnd, axis=1, keepdims=True)
            x_try[idx] = rnd
            x_try = mom_anneal(x_try, 700, 0.028, 0.0022, 0.40, 0.10, mom=0.88)
        else:
            # coordinate sweep over worst points then anneal
            G = x_try @ x_try.T
            np.fill_diagonal(G, -2.0)
            rm = G.max(axis=1)
            order = np.argsort(rm)[::-1][:6]
            for w in order:
                # try hole sample if it improves that row
                hp = hole_sample(x_try, int(w), rng, S=8000)
                # evaluate row max for hp
                others = np.delete(x_try, int(w), axis=0)
                cur = float((x_try[int(w)] @ others.T).max())
                newv = float((hp @ others.T).max())
                if newv < cur:
                    x_try[int(w)] = hp
                    x_try = refine_single(x_try, int(w), steps=100, lr=0.06, topk=6)
            # small global noise then anneal
            x_try = x_try + 0.01 * rng.standard_normal(x_try.shape)
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            x_try = mom_anneal(x_try, 500, 0.015, 0.002, 0.30, 0.09, mom=0.84)

        v_try = max_cos(x_try)
        # accept directly if better (even unpolished)
        if v_try < best - 1e-12:
            # polish briefly to lock in
            lp_b = min(3.5, BUDGET - (time.time() - t0) - 6.0)
            if lp_b > 0.7:
                x_pol, v_pol = lp_polish(x_try, time.time() + lp_b, radius0=2e-3)
                if v_pol < best:
                    x_best, best = x_pol, v_pol
                    continue
            x_best, best = x_try, v_try
            continue
        # if close, polish opportunistically
        if v_try < best + 0.004:
            lp_b = min(2.5, BUDGET - (time.time() - t0) - 6.0)
            if lp_b < 0.7:
                continue
            # only polish every few to save time, or if very close
            if (v_try < best + 0.001) or (it % 3 == 0):
                x_pol, v_pol = lp_polish(x_try, time.time() + lp_b, radius0=2e-3)
                if v_pol < best - 1e-12:
                    x_best, best = x_pol, v_pol
                elif v_pol < best + 5e-8 and (it % 13 == 0) and rng.random() < 0.2:
                    x_best, best = x_pol, v_pol
        # occasional sideways diversification
        elif it % 17 == 0 and rng.random() < 0.05:
            # keep a slightly worse but different basin rarely
            pass

    remain = BUDGET - (time.time() - t0)
    if remain > 2.0:
        x_best, best = lp_polish(x_best, time.time() + remain - 0.6, radius0=5e-4)
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
