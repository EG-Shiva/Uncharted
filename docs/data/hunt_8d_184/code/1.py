import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _norm(x):
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-300)
    return x


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _lp_polish(x, seconds, radius_init=3e-3):
    x = _norm(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    xb = x.copy()
    radius = radius_init
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    # preallocate not needed
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        cfull = G[iu0]
        m = float(cfull.max())
        thr = m - max(20.0 * radius, 2e-9)
        sel = cfull > thr
        I = iu0[0][sel]
        J = iu0[1][sel]
        csel = cfull[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        # cap constraints for speed
        if p > 4000:
            idx = np.argpartition(-csel, 4000)[:4000]
            I, J, csel = I[idx], J[idx], csel[idx]
            p = 4000
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        # cols: dx_I block, dx_J block, t
        # build cols vectorized
        ar = np.arange(d)
        cols_I = (I[:, None] * d + ar[None, :]).ravel()
        cols_J = (J[:, None] * d + ar[None, :]).ravel()
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        cols[0::(2 * d + 1)] = cols_I[0::d]  # placeholder, will overwrite below
        # simpler: construct via concatenate
        cols = np.concatenate([np.repeat(I * d, d) + np.tile(ar, p),
                               np.repeat(J * d, d) + np.tile(ar, p),
                               np.full(p, n * d)], axis=0)
        # rows need matching order: for each constraint, first d entries are I, next d are J, last is t
        # our cols above groups all I then all J then t, so rows must match
        rows_I = np.repeat(np.arange(p), d)
        rows_J = np.repeat(np.arange(p), d)
        rows_T = np.arange(p)
        rows = np.concatenate([rows_I, rows_J, rows_T])
        vals = np.concatenate([x[J].ravel(), x[I].ravel(), -np.ones(p)])
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
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            best = v
            xb = y.copy()
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            # try keeping move if very close? no, shrink
            radius *= 0.5
            # if stuck for long, break to save time
            if radius < 1e-7 and p > 1000:
                # already tight
                break
    return xb


def _smooth_polish(x, steps=700, T0=0.03, T1=0.0025, lr0=0.03, lr1=0.002):
    x = _norm(x.copy())
    n, d = x.shape
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        T = T0 * (T1 / T0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -1.0)
        m = float(G.max())
        # exp in stable way
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        rs = E.sum(axis=1, keepdims=True) + 1e-300
        D = (E @ x) / rs
        # tangential projection (move away from weighted neighbours)
        dot = (D * x).sum(axis=1, keepdims=True)
        Dt = D - dot * x
        # limit step
        x = x - lr * Dt
        x = _norm(x)
    return x


def solve():
    t0 = time.time()
    LIMIT = 172.0
    rng = np.random.RandomState(0)
    try:
        xb0 = np.load("best.npy")
    except Exception:
        xb0 = None
    try:
        xr0 = np.load("record.npy")
    except Exception:
        xr0 = None
    cands = []
    for c in (xb0, xr0):
        if c is not None:
            c = np.asarray(c, dtype=np.float64)
            if c.shape == (184, 8):
                c = _norm(c)
                cands.append(((_max_cos(c)), c))
    if len(cands) == 0:
        x_best = _norm(rng.randn(184, 8))
        best_val = _max_cos(x_best)
    else:
        cands.sort(key=lambda t: t[0])
        best_val, x_best = cands[0][0], cands[0][1].copy()
    n, d = 184, 8

    # quick contact analysis helper
    def contacts_of(x, tol=5e-9):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        C = (G > m - tol)
        cnt = C.sum(axis=1)
        return G, m, cnt

    trial = 0
    # reserve ~22s for final LP
    while time.time() - t0 < LIMIT - 25.0:
        trial += 1
        if time.time() - t0 > LIMIT - 26:
            break
        _, mcur, cnt = contacts_of(x_best, tol=2e-9)
        order_low = np.argsort(cnt, kind="stable")
        order_high = order_low[::-1]
        # kick size schedule
        if trial % 25 == 0:
            k = 14 + int(rng.randint(0, 6))
        elif trial % 7 == 0:
            k = 9 + int(rng.randint(0, 4))
        else:
            k = 3 + int(rng.randint(0, 4))
        victims = set()
        # 1-2 lowest
        nl = 1 + int(rng.randint(0, 2))
        # pick among bottom 12 randomly
        bot_pool = order_low[:12]
        perm = rng.permutation(len(bot_pool))
        for i in range(min(nl, len(bot_pool))):
            victims.add(int(bot_pool[perm[i]]))
        # 1 high to relieve jam
        if rng.rand() < 0.7:
            top_pool = order_high[:12]
            victims.add(int(top_pool[rng.randint(len(top_pool))]))
        while len(victims) < k:
            victims.add(int(rng.randint(n)))
        victims = list(victims)
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[victims] = False
        x_keep = x_best[keep_mask]
        # sequential greedy void filling
        x_new = x_best.copy()
        filled = list(x_keep)  # list of vectors
        filled_arr = x_keep.copy()
        for v in victims:
            # sample candidates, pick most isolated
            if trial % 3 == 0:
                ncand = 3000
            else:
                ncand = 5000
            cand = rng.randn(ncand, d)
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            # max dot to filled set
            dots = cand @ filled_arr.T
            md = dots.max(axis=1)
            bi = int(np.argmin(md))
            # occasionally add small noise to best hole for diversity
            if rng.rand() < 0.25:
                nb = cand[bi] + 0.02 * rng.randn(d)
                nb /= np.linalg.norm(nb)
                # keep if still good
                md2 = float(nb @ filled_arr.T.max()) if False else float((filled_arr @ nb).max())
                if md2 < md[bi] + 0.02:
                    cand_best = nb
                else:
                    cand_best = cand[bi]
            else:
                cand_best = cand[bi]
            x_new[v] = cand_best
            # update filled for next victim (replace: append)
            filled_arr = np.vstack([filled_arr, cand_best[None, :]])
        # local reconverge with smooth polish
        # vary smooth length
        if k >= 12:
            steps = 900
        else:
            steps = 600
        # randomize annealing slightly
        T0 = 0.025 + 0.015 * rng.rand()
        x_try = _smooth_polish(x_new, steps=steps, T0=T0, T1=0.002, lr0=0.03, lr1=0.002)
        v_try = _max_cos(x_try)
        # if promising (within 0.005 or better), refine a bit more
        if v_try < best_val - 1e-12:
            # short LP to snap to local optimum
            budget = min(3.0, max(0.5, (LIMIT - 25.0 - (time.time() - t0)) / 8.0))
            # only LP every promising case; cap frequency to save time
            x_try2 = _lp_polish(x_try, seconds=budget, radius_init=2e-3)
            v_try2 = _max_cos(x_try2)
            if v_try2 < best_val:
                x_best = x_try2
                best_val = v_try2
            else:
                # keep smooth version if it is still best
                if v_try < best_val:
                    x_best = _norm(x_try)
                    best_val = v_try
        elif v_try < best_val + 0.002 and rng.rand() < 0.05:
            # occasional second-stage shake from near-best to diversify without LP
            pass
        # global shake fallback every ~40 trials if no progress: perturb all slightly
        if trial % 40 == 0:
            # check time
            if time.time() - t0 > LIMIT - 30:
                break
            x_pert = x_best + 0.015 * rng.randn(n, d)
            x_pert = _norm(x_pert)
            x_pert = _smooth_polish(x_pert, steps=500, T0=0.02, T1=0.003, lr0=0.02, lr1=0.002)
            vp = _max_cos(x_pert)
            if vp < best_val:
                x_best = _lp_polish(x_pert, seconds=2.0, radius_init=2e-3)
                best_val = _max_cos(x_best)
        # safety: if time low break
        if time.time() - t0 > LIMIT - 26:
            break

    # final tightening on best
    remain = LIMIT - (time.time() - t0) - 1.0
    if remain > 3.0:
        # first a smooth touch then LP
        x_best = _smooth_polish(x_best, steps=400, T0=0.008, T1=0.0015, lr0=0.008, lr1=0.001)
        remain2 = LIMIT - (time.time() - t0) - 0.5
        if remain2 > 1.0:
            x_best = _lp_polish(x_best, seconds=min(remain2, 22.0), radius_init=1.5e-3)
    x_best = _norm(x_best)
    # final safety: remove NaN
    x_best = np.where(np.isfinite(x_best), x_best, 0.0)
    # if any zero row (should not happen), replace random
    nr = np.linalg.norm(x_best, axis=1)
    for i in range(n):
        if nr[i] < 1e-12 or not np.isfinite(nr[i]):
            v = rng.randn(d)
            v /= np.linalg.norm(v)
            x_best[i] = v
    x_best = _norm(x_best)
    return x_best
