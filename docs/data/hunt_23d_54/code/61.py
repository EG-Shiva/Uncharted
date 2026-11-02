import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def _adam_anneal(x_init, steps, k0, k1, lr0, lr1):
    x = x_init.copy()
    n, d = x.shape
    b1, b2, eps = 0.9, 0.999, 1e-8
    m1 = np.zeros_like(x)
    m2 = np.zeros_like(x)
    lk0, lk1 = np.log(k0), np.log(k1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        f = s / max(steps - 1, 1)
        k = float(np.exp(lk0 + (lk1 - lk0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        # stable LogSumExp weights, max maps to 1
        W = np.exp(k * (G - m))
        np.fill_diagonal(W, 0.0)
        S = float(np.sum(W)) * 0.5
        if not np.isfinite(S) or S <= 0:
            S = 1.0
        F = (W @ x) / S
        dot = np.sum(F * x, axis=1, keepdims=True)
        Gt = F - dot * x
        m1 *= b1
        m1 += (1.0 - b1) * Gt
        m2 *= b2
        m2 += (1.0 - b2) * (Gt * Gt)
        c1 = 1.0 - b1 ** (s + 1)
        c2 = 1.0 - b2 ** (s + 1)
        step = (lr / c1) * m1 / (np.sqrt(m2 / c2) + eps)
        # clip extreme steps for stability
        sn = np.sqrt(np.sum(step * step))
        if sn > 0.1 * n:
            step *= (0.1 * n) / sn
        x = x - step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x /= nr
    return x


def _lp_polish(x, seconds, t0=None, budget=1e9):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if t0 is not None and (time.time() - t0) > budget:
            break
        if radius < 1e-14:
            break
        Gfull = x @ x.T
        c_all = Gfull[iu0]
        m = c_all.max()
        sel = c_all > m - max(20 * radius, 1e-9)
        if np.count_nonzero(sel) > 800:
            idx = np.where(sel)[0]
            order = np.argsort(c_all[idx])[::-1][:800]
            nsel = np.zeros_like(sel, dtype=bool)
            nsel[idx[order]] = True
            sel = nsel
        I, J = iu0[0][sel], iu0[1][sel]
        c = c_all[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix(
                (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                 shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq,
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    BUDGET = 172.0
    CHAIN_DEADLINE = 145.0  # chains must finish within 145s of start
    n_full, d = 54, 23
    n_start = 40

    # incumbent
    try:
        xb = np.load("best.npy").astype(float)
        xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    except Exception:
        rng0 = np.random.RandomState(0)
        xb = rng0.randn(n_full, d)
        xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    if xb.shape != (n_full, d):
        rng0 = np.random.RandomState(0)
        xb = rng0.randn(n_full, d)
        xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    try:
        xr = np.load("record.npy").astype(float)
        xr /= np.linalg.norm(xr, axis=1, keepdims=True)
        if xr.shape == (n_full, d) and _max_cos(xr) < _max_cos(xb):
            xb = xr.copy()
    except Exception:
        pass
    inc_val = _max_cos(xb)
    x_inc = xb.copy()

    seeds = [101, 202, 303, 404]
    best_chain_x = None
    best_chain_v = 1e9

    for ci, sd in enumerate(seeds):
        if time.time() - t0 > CHAIN_DEADLINE:
            break
        # leave time for at least LP polish (~10s) even for last chain
        if time.time() - t0 > CHAIN_DEADLINE - 5.0 and ci >= 3:
            break
        rng = np.random.RandomState(sd)
        try:
            x = rng.randn(n_start, d)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            # well-optimised 40-point code, k 30->500 over 2000 steps
            x = _adam_anneal(x, steps=2000, k0=30.0, k1=500.0, lr0=0.01, lr1=0.002)
            if time.time() - t0 > CHAIN_DEADLINE:
                # score partial chain if already at 54? else skip
                pass
            # insert one at a time up to 54
            ok = True
            for cur in range(n_start, n_full):
                if time.time() - t0 > CHAIN_DEADLINE:
                    ok = False
                    break
                # best of 50,000 random candidates
                cand = rng.randn(50000, d)
                cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                dots = cand @ x.T
                md = dots.max(axis=1)
                bj = int(np.argmin(md))
                new_pt = cand[bj]
                x = np.vstack([x, new_pt.reshape(1, d)])
                # global re-anneal, k 100->700, lr 0.01->0.002, 800 steps
                x = _adam_anneal(x, steps=800, k0=100.0, k1=700.0, lr0=0.01, lr1=0.002)
            if not ok:
                # incomplete chain, evaluate only if reached 54
                if x.shape[0] != n_full:
                    continue
            if x.shape[0] != n_full:
                continue
            v = _max_cos(x)
            # discard chains more than 0.01 behind incumbent
            if v > inc_val + 0.01:
                continue
            if v < best_chain_v:
                best_chain_v = v
                best_chain_x = x.copy()
        except Exception:
            continue

    # choose single best final configuration for LP polish
    if best_chain_x is None:
        cand_x = x_inc.copy()
        cand_v = inc_val
    else:
        if best_chain_v < inc_val:
            cand_x = best_chain_x.copy()
            cand_v = best_chain_v
        else:
            # chain exists but incumbent still better: polish incumbent
            # (keeps best final quality while respecting single-polish rule)
            cand_x = x_inc.copy()
            cand_v = inc_val
            # remember chain as fallback if polish fails
            pass

    # trust-region LP polish in remaining ~25s
    try:
        elapsed = time.time() - t0
        left = BUDGET - elapsed
        # reserve 1s for return
        lp_sec = min(max(left - 1.0, 0.0), 25.0)
        if lp_sec > 1.0:
            cand_x = _lp_polish(cand_x, lp_sec, t0, BUDGET - 0.5)
            cand_v = _max_cos(cand_x)
            # final safety: return best of polished candidate vs incumbent vs chain
            best_x = cand_x
            best_v = cand_v
            if inc_val < best_v:
                best_x, best_v = x_inc.copy(), inc_val
            if best_chain_x is not None and best_chain_v < best_v:
                best_x, best_v = best_chain_x.copy(), best_chain_v
            cand_x = best_x
    except Exception:
        pass

    cand_x /= np.linalg.norm(cand_x, axis=1, keepdims=True)
    # strict validity: no zero rows / NaN
    cand_x = np.nan_to_num(cand_x)
    nr = np.linalg.norm(cand_x, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    cand_x /= nr
    return cand_x
