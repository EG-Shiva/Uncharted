import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _personal_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1), g


def _contacts(x, tol=1e-7):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    m = g.max()
    tight = g > m - tol
    # count per point (each contact counted once per endpoint)
    cnt = (tight.sum(axis=1)).astype(int)
    return cnt, m, g


def _lp_polish(x, deadline, r_init=0.01, r_max=0.05):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r_init
    iu0 = np.triu_indices(n, 1)
    # preallocate obj
    nv = n * d + 1
    obj = np.zeros(nv)
    obj[-1] = 1.0
    bnds = None
    while time.time() < deadline and radius > 1e-13:
        if time.time() + 0.4 > deadline:
            break
        G = x @ x.T
        # max over i<j
        # use triu for max
        c_all = G[iu0]
        m = c_all.max()
        thresh = max(20.0 * radius, 1e-9)
        sel = c_all > m - thresh
        # cap size for speed: keep tightest 2500
        if np.count_nonzero(sel) > 2500:
            # keep top 2500
            idx = np.where(sel)[0]
            order = np.argsort(-c_all[idx])[:2500]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I = iu0[0][sel]
        J = iu0[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        c_sel = c_all[sel]
        rows = np.repeat(np.arange(p), 2 * d + 1)
        # cols: I*d + k, J*d + k, last
        # build efficiently
        cols_I = (I[:, None] * d + np.arange(d)).ravel()
        cols_J = (J[:, None] * d + np.arange(d)).ravel()
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        cols[0::2 * d + 1] = 0  # placeholder, will overwrite in blocks? easier concat
        # simpler concat (as parent) - still fast for p~3000
        cols = np.concatenate([cols_I.reshape(p, d), cols_J.reshape(p, d), np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        bounds = [(-radius, radius)] * (n * d) + [(None, None)]
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=bounds, method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        if not np.all(np.isfinite(yn)) or np.any(yn < 1e-12):
            radius *= 0.5
            continue
        y = y / yn
        v = _max_cos(y)
        if v < best:
            x = y
            best = v
            radius = min(radius * 1.5, r_max)
        else:
            radius *= 0.5
    return x, best


def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    rng = np.random.default_rng(0)
    n, d = 78, 15
    # load starts
    cand = []
    try:
        b = np.load("best.npy")
        if b.shape == (n, d):
            cand.append(b.astype(float))
    except Exception:
        pass
    try:
        r = np.load("record.npy")
        if r.shape == (n, d):
            cand.append(r.astype(float))
    except Exception:
        pass
    if len(cand) == 0:
        x0 = rng.standard_normal((n, d))
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        cand.append(x0)
    best_x = None
    best_v = 1e9
    for c in cand:
        c = c / np.linalg.norm(c, axis=1, keepdims=True)
        v = _max_cos(c)
        if v < best_v:
            best_v = v
            best_x = c.copy()
    x_best = best_x.copy()
    v_best = best_v

    # short initial re-polish to squeeze record (8s)
    if HARD_END - time.time() > 12:
        x_best, v_best = _lp_polish(x_best, min(time.time() + 8.0, HARD_END - 4), r_init=0.002, r_max=0.02)
    # also try record if different (brief)
    # main basin hopping
    trial = 0
    # reserve ~25s for final polish
    while time.time() < HARD_END - 25:
        trial += 1
        remaining = HARD_END - 25 - time.time()
        if remaining < 4:
            break
        # budget per trial
        budget = min(9.0, remaining / 2.0, remaining - 1.0)
        if budget < 2.5:
            budget = remaining
        deadline = time.time() + budget
        mode = trial % 5
        x_try = x_best.copy()
        if mode == 1:
            sigma = 0.008
            x_try = x_try + sigma * rng.standard_normal((n, d))
        elif mode == 2:
            sigma = 0.025
            x_try = x_try + sigma * rng.standard_normal((n, d))
        elif mode == 3:
            # kick worst points hard, rest light
            pm, _ = _personal_max(x_try)
            order = np.argsort(-pm)
            k = 6
            worst = order[:k]
            noise = 0.005 * rng.standard_normal((n, d))
            noise[worst] += 0.06 * rng.standard_normal((k, d))
            x_try = x_try + noise
        elif mode == 4:
            # replace 2 lowest-contact points with random + jitter rest
            cnt, _, _ = _contacts(x_try, tol=1e-7)
            order = np.argsort(cnt)
            k = 2 if (trial % 10 != 4) else 3
            repl = order[:k]
            rand = rng.standard_normal((k, d))
            rand /= np.linalg.norm(rand, axis=1, keepdims=True)
            x_try[repl] = rand
            # small jitter to others to break symmetry
            mask = np.ones(n, dtype=bool)
            mask[repl] = False
            x_try[mask] += 0.004 * rng.standard_normal((np.count_nonzero(mask), d))
        else:  # mode 0: tail-focused
            cnt, _, _ = _contacts(x_try, tol=1e-7)
            order = np.argsort(cnt)  # low contacts first
            # take ~18 lowest
            tail = order[:18]
            noise = 0.004 * rng.standard_normal((n, d))
            noise[tail] += 0.05 * rng.standard_normal((len(tail), d))
            x_try = x_try + noise
        # renormalize
        x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        # polish attempt, start radius larger for larger kicks
        r_init = 0.015 if mode in (2, 3, 4, 0) else 0.008
        x_try, v_try = _lp_polish(x_try, deadline, r_init=r_init, r_max=0.05)
        if v_try < v_best - 1e-13:
            x_best = x_try
            v_best = v_try
        # opportunistic early exit if time low
        if time.time() >= HARD_END - 25:
            break

    # final intensive polish of best
    if time.time() < HARD_END - 1:
        x_best, v_best = _lp_polish(x_best, HARD_END - 0.5, r_init=0.003, r_max=0.02)
        # one more ultra-fine pass
        if time.time() < HARD_END - 0.5:
            x_best, v_best = _lp_polish(x_best, HARD_END - 0.3, r_init=0.0005, r_max=0.005)

    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    # safety: replace non-finite
    if not np.all(np.isfinite(x_best)):
        x_best = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return x_best
