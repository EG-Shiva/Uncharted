import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _lp_polish(x, seconds, deadline=None, r0=1e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if deadline is not None and time.time() > deadline:
            break
        gfull = x @ x.T
        # max without diag
        # fast max
        np.fill_diagonal(gfull, -2.0)
        m = gfull.max()
        cfull = gfull[iu]
        thresh = m - max(20 * radius, 1e-9)
        sel = cfull > thresh
        if not np.any(sel):
            radius *= 0.5
            continue
        I = iu[0][sel]
        J = iu[1][sel]
        csel = cfull[sel]
        p = len(I)
        # cap constraints for speed
        if p > 2500:
            # keep tightest 2500
            idx = np.argpartition(csel, p - 2500)[p - 2500:]
            I = I[idx]
            J = J[idx]
            csel = csel[idx]
            p = 2500
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, n * d + 1)).tocsr()
        obj = np.zeros(n * d + 1)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def _find_void(Xrem, rng, n_sample=12000):
    # largest hole w.r.t Xrem
    m, d = Xrem.shape
    R = rng.normal(size=(n_sample, d))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    D = R @ Xrem.T
    mm = D.max(axis=1)
    y = R[int(np.argmin(mm))].copy()
    # local refine: minimise smooth max dot
    T = 0.006
    step0 = 0.05
    for it in range(200):
        dv = Xrem @ y
        md = dv.max()
        w = np.exp((dv - md) / T)
        s = w.sum()
        if s == 0:
            break
        w /= s
        gvec = Xrem.T @ w
        gt = gvec - float(gvec @ y) * y
        nrm = float(np.linalg.norm(gt))
        if nrm < 1e-12:
            break
        step = step0 * (0.985 ** it)
        y = y - step * gt / nrm
        y /= np.linalg.norm(y)
        # slight anneal
        if it % 80 == 79:
            T *= 0.7
    return y / np.linalg.norm(y)


def solve():
    t0 = time.time()
    HARD_DEADLINE = t0 + 168.0
    rng = np.random.default_rng(0)
    try:
        x_best = np.load("best.npy").astype(float)
    except Exception:
        x_best = np.load("record.npy").astype(float)
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    n, d = x_best.shape
    best_val = _max_cos(x_best)

    # contact counts for victim choice
    def contacts_of(x):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        C = G > m - 1e-9
        cnt = C.sum(axis=1) + C.sum(axis=0)
        # C symmetric, sum once
        cnt = C.sum(axis=1)
        return m, cnt

    trial = 0
    # exploration phase until ~135s
    EXP_END = t0 + 132.0
    while time.time() < EXP_END and time.time() < HARD_DEADLINE - 30:
        trial += 1
        _, cnt = contacts_of(x_best)
        cand = x_best.copy()
        mode = trial % 3
        if mode in (0, 1):
            # void insertion of 1-2 overloaded points
            w = (cnt.astype(float) + 1.0) ** 2
            w = w / w.sum()
            nv = 1 if rng.random() < 0.7 else 2
            # choose without replacement
            victims = rng.choice(n, size=nv, replace=False, p=w)
            for v in victims:
                mask = np.ones(n, dtype=bool)
                mask[v] = False
                # if 2 victims, keep other victim out too for hole search?
                # use current cand without v
                Xrem = np.delete(cand, v, axis=0)
                ynew = _find_void(Xrem, rng)
                cand[v] = ynew
            # tiny global jitter to break exact ties
            cand += rng.normal(scale=8e-4, size=cand.shape)
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            budget = 2.2
        else:
            # Gaussian shake of subset incl. rattlers re-init
            # reinit rattlers with prob
            ratt = np.where(cnt == 0)[0]
            if len(ratt) > 0 and rng.random() < 0.6:
                for v in ratt:
                    if rng.random() < 0.7:
                        rv = rng.normal(size=d)
                        rv /= np.linalg.norm(rv)
                        cand[v] = rv
            k = int(rng.integers(4, 12))
            # bias to high contacts
            w = (cnt.astype(float) + 0.5)
            w = w / w.sum()
            idx = rng.choice(n, size=k, replace=False, p=w)
            sigmas = np.array([0.004, 0.008, 0.015, 0.03])
            sig = sigmas[int(rng.integers(len(sigmas)))]
            cand[idx] += rng.normal(scale=sig, size=(k, d))
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            budget = 2.2

        # don't overrun exploration
        remain_exp = EXP_END - time.time()
        if remain_exp < 0.5:
            break
        budget = min(budget, remain_exp, HARD_DEADLINE - 30 - (time.time()))
        if budget < 0.4:
            break
        polished = _lp_polish(cand, seconds=budget, deadline=HARD_DEADLINE - 25)
        v = _max_cos(polished)
        if v < best_val - 1e-12:
            x_best = polished
            best_val = v
        # opportunistic extended polish if very close and promising
        # (skip to save time)

    # final high-precision polish of best with remaining time
    remain = HARD_DEADLINE - time.time() - 1.0
    if remain > 2:
        # two-stage: small radius then larger
        x_best = _lp_polish(x_best, seconds=remain, deadline=HARD_DEADLINE - 0.5, r0=8e-4)

    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
