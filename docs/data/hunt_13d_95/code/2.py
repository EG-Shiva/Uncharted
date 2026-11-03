import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def lp_polish(x, seconds, radius_init=1e-3):
    """Trust-region LP: move points tangentially to lower the largest cosine."""
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = float(radius_init)
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        # time guard for linprog overhead
        if time.time() - start > seconds - 0.05:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        thresh = m - max(20 * radius, 1e-9)
        sel = c > thresh
        # cap constraints for speed
        if int(sel.sum()) > 1500:
            # keep top 1500
            idx = np.argpartition(c, -1500)[-1500:]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
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
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def contact_counts(x, mc, tol=0.02):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return (g > mc - tol).sum(axis=1)


def best_hole(others, rng, n_samples=2000):
    n, d = others.shape
    cand = rng.normal(size=(n_samples, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    # max dot to others
    dots = cand @ others.T
    md = dots.max(axis=1)
    j = int(np.argmin(md))
    return cand[j]


def solve():
    t0 = time.time()
    BUDGET = 168.0
    EXPL_T = 122.0  # exploration seconds, reserve rest for final polish

    rng = np.random.default_rng(0)

    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None

    cands = []
    if xb is not None:
        xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
        cands.append(xb)
    if xr is not None:
        xr = xr / np.linalg.norm(xr, axis=1, keepdims=True)
        cands.append(xr)
    if len(cands) == 0:
        x0 = rng.normal(size=(95, 13))
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        cands.append(x0)

    vals = [max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    x_best = cands[bi].copy()
    best_val = float(vals[bi])

    cur = x_best.copy()
    cur_val = best_val

    n, d = x_best.shape

    sigmas = [0.005, 0.01, 0.02, 0.04, 0.08, 0.15, 0.30, 0.01, 0.03, 0.07]
    it = 0
    while time.time() - t0 < EXPL_T:
        it += 1
        remain_expl = EXPL_T - (time.time() - t0)
        if remain_expl < 1.0:
            break
        polish_sec = min(2.0, max(0.8, remain_expl * 0.15))
        # choose trial type
        u = rng.random()
        x_try = cur.copy()
        if u < 0.55:
            # gaussian shake all points
            sigma = sigmas[rng.integers(0, len(sigmas))]
            # occasionally shake only subset
            if rng.random() < 0.25:
                # subset shake: pick 5-15 points
                ksub = int(rng.integers(5, 16))
                idx = rng.choice(n, size=ksub, replace=False)
                noise = rng.normal(size=(ksub, d))
                x_try[idx] += sigma * 2.0 * noise
            else:
                x_try += sigma * rng.normal(size=(n, d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            rinit = 0.008 if sigma > 0.05 else 0.004
        else:
            # re-insertion of k points
            if rng.random() < 0.33:
                k = 1
            elif rng.random() < 0.6:
                k = 2
            else:
                k = int(rng.integers(3, 6))
            cc = contact_counts(cur, cur_val, tol=0.02)
            # pick indices: mostly overloaded, sometimes random / low-contact
            pick = rng.random()
            if pick < 0.6:
                idx = np.argsort(-cc)[:k]
                # add randomness: sometimes swap one
                if k > 1 and rng.random() < 0.5:
                    idx = idx.copy()
                    idx[-1] = rng.integers(0, n)
            elif pick < 0.8:
                idx = rng.choice(n, size=k, replace=False)
            else:
                idx = np.argsort(cc)[:k]  # rattlers
            # generate replacements
            use_hole = rng.random() < 0.7
            for j in idx:
                if use_hole:
                    # hole w.r.t. all other points
                    mask = np.ones(n, dtype=bool)
                    mask[j] = False
                    others = x_try[mask]
                    # more samples for k==1
                    ns = 3000 if k == 1 else 1500
                    x_try[j] = best_hole(others, rng, n_samples=ns)
                else:
                    v = rng.normal(size=(d,))
                    v /= np.linalg.norm(v)
                    x_try[j] = v
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            rinit = 0.01

        # quick screen: if unpolished is terrible, still polish (LP handles it)
        try:
            x_pol = lp_polish(x_try, seconds=polish_sec, radius_init=rinit)
        except Exception:
            continue
        v = max_cos(x_pol)

        if v < best_val - 1e-12:
            x_best = x_pol.copy()
            best_val = v
            cur = x_pol.copy()
            cur_val = v
        elif v < cur_val:
            cur = x_pol.copy()
            cur_val = v
        else:
            # occasional sideways move to change basin
            if rng.random() < 0.06 and v < cur_val + 0.01:
                cur = x_pol.copy()
                cur_val = v
            # occasionally restart from best
            if rng.random() < 0.12:
                cur = x_best.copy()
                cur_val = best_val

        # hard time check
        if time.time() - t0 > BUDGET - 35:
            break

    # final long polish of best
    remain = BUDGET - (time.time() - t0)
    # keep at least 5 sec, at most 45 sec
    final_sec = float(np.clip(remain - 2.0, 5.0, 45.0))
    if final_sec > 1.0:
        try:
            x_best = lp_polish(x_best, seconds=final_sec * 0.6, radius_init=1e-3)
            # second pass with smaller threshold behaviour via larger init if improved
            rest = BUDGET - (time.time() - t0) - 1.0
            if rest > 2.0:
                x_best = lp_polish(x_best, seconds=min(rest, 20.0), radius_init=5e-4)
        except Exception:
            pass

    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
