import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _normalize(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    x = x / n
    x = np.where(np.isfinite(x), x, 0.0)
    n2 = np.linalg.norm(x, axis=1, keepdims=True)
    bad = (n2[:, 0] < 1e-12)
    if np.any(bad):
        x[bad] = np.array([1.0, 0.0, 0.0, 0.0])
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _personal_worst(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1)

def _best_void(others, rng, n_cand=12000):
    cand = rng.standard_normal((n_cand, 4))
    cand = cand / np.linalg.norm(cand, axis=1, keepdims=True)
    s = cand @ others.T
    scores = s.max(axis=1)
    bi = int(np.argmin(scores))
    b = cand[bi].copy()
    bs = float(scores[bi])
    step = 0.05
    for _ in range(250):
        prop = b + step * rng.standard_normal((32, 4))
        prop = prop / np.linalg.norm(prop, axis=1, keepdims=True)
        sc = (prop @ others.T).max(axis=1)
        j = int(np.argmin(sc))
        if sc[j] < bs:
            b = prop[j].copy()
            bs = float(sc[j])
            step *= 1.05
        else:
            step *= 0.97
        if step < 1e-5:
            break
    nrm = float(np.linalg.norm(b))
    if nrm < 1e-12 or not np.isfinite(nrm):
        b = np.array([1.0, 0.0, 0.0, 0.0])
    else:
        b = b / nrm
    return b

def _relax_adam(y_init, betas, steps_list, lrs):
    y = y_init.copy()
    n, d = y.shape
    for beta, steps, lr in zip(betas, steps_list, lrs):
        m = np.zeros_like(y)
        vv = np.zeros_like(y)
        b1, b2, eps = 0.9, 0.999, 1e-8
        for t in range(1, steps + 1):
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            mmax = float(G.max())
            E = np.exp((G - mmax) * beta)
            np.fill_diagonal(E, 0.0)
            S = float(E.sum())
            if not np.isfinite(S) or S <= 0:
                break
            W = E / S
            grad = 2.0 * (W @ y)
            proj = np.sum(grad * y, axis=1, keepdims=True)
            gt = grad - proj * y
            m = b1 * m + (1.0 - b1) * gt
            vv = b2 * vv + (1.0 - b2) * (gt * gt)
            mhat = m / (1.0 - b1 ** t)
            vhat = vv / (1.0 - b2 ** t)
            y = y - lr * mhat / (np.sqrt(vhat) + eps)
            y = _normalize(y)
    return y

def _lp_polish(x, deadline):
    x = _normalize(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        cmat = (x @ x.T)[iu]
        m = float(cmat.max())
        sel = cmat > m - max(20.0 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cmat[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _normalize(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    rng = np.random.default_rng(20261031)
    try:
        x0 = np.load("best.npy")
    except Exception:
        try:
            x0 = np.load("record.npy")
        except Exception:
            x0 = rng.standard_normal((272, 4))
    x0 = _normalize(np.asarray(x0, dtype=np.float64))
    if x0.shape != (272, 4):
        x0 = _normalize(rng.standard_normal((272, 4)))
    best = x0.copy()
    best_val = _max_cos(best)
    cur = best.copy()
    cur_val = best_val

    # hopping phase, reserve ~30s for final LP
    HOP_END = HARD_END - 30.0
    it = 0
    stall = 0
    # temperature schedule
    T0, T1 = 8e-4, 2e-5
    MAX_IT = 160
    while time.time() < HOP_END and it < MAX_IT:
        it += 1
        frac = it / float(MAX_IT)
        Temp = T0 * (1.0 - frac) + T1 * frac
        # victim selection
        pw = _personal_worst(cur)
        order = np.argsort(-pw)
        r = rng.random()
        if r < 0.65:
            pool = order[:8]
            vi = int(pool[rng.integers(0, len(pool))])
        elif r < 0.85:
            pool = order[:30]
            vi = int(pool[rng.integers(0, len(pool))])
        else:
            vi = int(rng.integers(0, 272))
        # occasionally double move
        double = (it % 11 == 0)
        y = cur.copy()
        if double:
            # pick two worst distinct
            vi2 = int(order[1 + rng.integers(0, 10)])
            if vi2 == vi:
                vi2 = int(order[10])
            others1 = np.delete(y, vi, axis=0)
            b1 = _best_void(others1, rng, n_cand=8000)
            y[vi] = b1
            others2 = np.delete(y, vi2, axis=0)
            b2 = _best_void(others2, rng, n_cand=8000)
            y[vi2] = b2
        else:
            others = np.delete(y, vi, axis=0)
            nc = 12000 if (it % 3 == 0) else 6000
            b = _best_void(others, rng, n_cand=nc)
            y[vi] = b
        # small global shake, decaying
        shake = 0.012 * (1.0 - frac) + 0.003
        if rng.random() < 0.8:
            y = y + shake * rng.standard_normal(y.shape)
            y = _normalize(y)
        # relax schedule: early broad, late sharp
        if frac < 0.5:
            betas = [45.0, 130.0]
            stepsl = [110, 110]
            lrsl = [0.008, 0.003]
        else:
            betas = [60.0, 160.0, 350.0]
            stepsl = [80, 80, 80]
            lrsl = [0.006, 0.003, 0.0015]
        y2 = _relax_adam(y, betas, stepsl, lrsl)
        v2 = _max_cos(y2)
        if v2 < cur_val:
            cur, cur_val = y2, v2
            stall = 0
            if v2 < best_val:
                best, best_val = y2.copy(), v2
        else:
            # SA acceptance
            if rng.random() < np.exp((cur_val - v2) / max(Temp, 1e-12)):
                cur, cur_val = y2, v2
                stall = 0
            else:
                stall += 1
        # restart from best if stalled
        if stall >= 18:
            stall = 0
            # kick best slightly and relax briefly
            yb = best + 0.02 * rng.standard_normal(best.shape)
            yb = _normalize(yb)
            yb = _relax_adam(yb, [50.0, 150.0], [60, 60], [0.006, 0.003])
            vb = _max_cos(yb)
            cur, cur_val = yb, vb
            if vb < best_val:
                best, best_val = yb.copy(), vb
        if time.time() > HOP_END - 2.0:
            break

    # final LP polish on best (and cur if different and competitive)
    cands = [(best_val, best.copy())]
    if cur_val < best_val + 5e-4:
        cands.append((cur_val, cur.copy()))
    cands.sort(key=lambda kv: kv[0])
    polished_x, polished_v = cands[0][1], cands[0][0]
    # use remaining time, at least a few seconds
    for _, xc in cands[:2]:
        if time.time() > HARD_END - 4.0:
            break
        deadline = min(HARD_END - 1.0, time.time() + max(10.0, (HARD_END - 1.0 - time.time()) / 2.0))
        xp, vp = _lp_polish(xc, deadline)
        if vp < polished_v:
            polished_v, polished_x = vp, xp.copy()

    out = _normalize(polished_x)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    nrm = np.linalg.norm(out, axis=1, keepdims=True)
    nrm[nrm < 1e-12] = 1.0
    out = out / nrm
    return out
