import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def _norm(x):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x


def _lp_polish_minmax(x, seconds, t0, deadline):
    x = _norm(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu0 = np.triu_indices(n, 1)[0]
    iu1 = np.triu_indices(n, 1)[1]
    start = time.time()
    while time.time() - start < seconds and time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iu0, iu1]
        m = float(np.max(c))
        sel = c > m - max(20.0 * radius, 1e-9)
        I = iu0[sel]; J = iu1[sel]
        cs = c[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cc = np.empty((p, 2 * d + 1), dtype=float)
        # cols
        cols = np.empty((p, 2 * d + 1), dtype=int)
        ar = np.arange(d)
        # build row by row via vectorized concat
        cols[:, :d] = I[:, None] * d + ar[None, :]
        cols[:, d:2 * d] = J[:, None] * d + ar[None, :]
        cols[:, 2 * d] = n * d
        cc[:, :d] = x[J]
        cc[:, d:2 * d] = x[I]
        cc[:, 2 * d] = -1.0
        try:
            A_ub = coo_matrix((cc.ravel(), (rows, cols.ravel())), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
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
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best


def solve():
    t_start = time.time()
    HARD_END = t_start + 172.0
    SEARCH_END = t_start + 140.0
    POLISH_BUDGET = 25.0
    rng = np.random.RandomState(0)
    np.random.seed(0)
    n, d = 180, 9

    # load incumbent
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            c = np.asarray(np.load(fn), dtype=float)
            if c.shape == (180, 9):
                c = _norm(c)
                cands.append((_max_cos(c), c))
        except Exception:
            pass
    if len(cands) == 0:
        xb = rng.randn(n, d); xb = _norm(xb)
        bbest = _max_cos(xb)
    else:
        cands.sort(key=lambda t: t[0])
        bbest, xb = cands[0][0], cands[0][1].copy()
    best_x = xb.copy()
    best_val = float(bbest)

    iu0, iu1 = np.triu_indices(n, 1)
    targets = [0.4620, 0.4615, 0.4610]
    radii = [0.02, 0.03, 0.015, 0.025]

    outer = 0
    # preallocate eq pattern helper: row index for A_eq
    eq_rows = np.repeat(np.arange(n), d)
    eq_cols = np.arange(n * d)

    while time.time() < SEARCH_END and time.time() < HARD_END - 30:
        t_targ = targets[outer % len(targets)]
        r_base = radii[outer % len(radii)]
        # restart from best with tangent kick to change basin
        sigmas = [0.012, 0.022, 0.032]
        sigma = sigmas[outer % len(sigmas)]
        noise = rng.randn(n, d) * sigma
        dot = np.sum(noise * best_x, axis=1, keepdims=True)
        noise -= dot * best_x
        x_cur = best_x + noise
        x_cur = _norm(x_cur)

        # rotating excluded-contact memory: pairs excluded for a few iters
        # implement as set of encoded pairs with expiry
        excl = {}  # code -> expiry inner idx
        n_inner = 7
        r = r_base
        for inner in range(n_inner):
            if time.time() >= SEARCH_END or time.time() >= HARD_END - 30:
                break
            try:
                G = x_cur @ x_cur.T
                c_all = G[iu0, iu1]
            except Exception:
                break
            # candidate pool above t-0.01
            thr = t_targ - 0.01
            # arg where
            # for speed use where
            idx = np.where(c_all > thr)[0]
            if idx.size == 0:
                v = _max_cos(x_cur)
                if v < best_val:
                    best_val = v; best_x = x_cur.copy()
                break
            # cap at 800 tightest
            if idx.size > 800:
                # largest 800
                part = np.argpartition(c_all[idx], -800)[-800:]
                idx = idx[part]
            # order by tightness descending
            order = np.argsort(-c_all[idx])
            idx = idx[order]
            # drop expired
            # build exclusion mask: remove pairs whose code in excl and not expired
            if len(excl) > 0:
                keep = []
                for k in idx:
                    i = int(iu0[k]); j = int(iu1[k])
                    code = i * n + j
                    if code in excl and excl[code] > inner:
                        continue
                    keep.append(k)
                # if we excluded too many, keep at least half
                if len(keep) >= 200:
                    idx = np.asarray(keep, dtype=int)
                # purge expired
                for code in list(excl.keys()):
                    if excl[code] <= inner:
                        del excl[code]
            # deliberately break most stressed contacts:
            # choose rotating random subset among top 200 to exclude for next iters
            L = idx.size
            topK = int(min(220, L))
            n_ex = int(min(70, max(10, topK // 3)))
            if topK > 10 and n_ex > 0:
                pool = idx[:topK]
                # random subset (deterministic via rng)
                ch = rng.choice(pool, size=n_ex, replace=False)
                # these are excluded *next* iterations, but also drop a
                # rotating subset now: drop half of them now
                drop_now = ch[:n_ex // 2]
                if len(drop_now) > 0:
                    s = set(drop_now.tolist())
                    idx = np.asarray([k for k in idx if k not in s], dtype=int)
                # schedule full set for next 3 iters
                for k in ch:
                    i = int(iu0[k]); j = int(iu1[k])
                    excl[i * n + j] = inner + 3
            p = idx.size
            if p == 0:
                continue
            I = iu0[idx]; J = iu1[idx]
            cs = c_all[idx]
            b_ub = t_targ - cs  # <=0.01
            nv = n * d + p
            # build A_ub: p x (nd+p), each row has 2d+1 nonzeros
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.empty((p, 2 * d + 1), dtype=int)
            ar = np.arange(d)
            cols[:, :d] = I[:, None] * d + ar[None, :]
            cols[:, d:2 * d] = J[:, None] * d + ar[None, :]
            cols[:, 2 * d] = n * d + np.arange(p)
            vals = np.empty((p, 2 * d + 1), dtype=float)
            vals[:, :d] = x_cur[J]
            vals[:, d:2 * d] = x_cur[I]
            vals[:, 2 * d] = -1.0
            try:
                A_ub = coo_matrix((vals.ravel(), (rows, cols.ravel())), shape=(p, nv)).tocsr()
                A_eq = coo_matrix((x_cur.ravel(), (eq_rows, eq_cols)), shape=(n, nv)).tocsr()
            except Exception:
                continue
            obj = np.zeros(nv)
            obj[n * d:] = 1.0
            bounds = [(-r, r)] * (n * d) + [(0.0, None)] * p
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=bounds, method="highs")
            except Exception:
                r = max(r * 0.5, 0.005)
                continue
            if res.status != 0:
                r = max(r * 0.6, 0.005)
                # small random tangent nudge to escape infeasible vertex
                nz = rng.randn(n, d) * 0.003
                nz -= np.sum(nz * x_cur, axis=1, keepdims=True) * x_cur
                x_cur = _norm(x_cur + nz)
                continue
            dx = res.x[:n * d].reshape(n, d)
            y = _norm(x_cur + dx)
            v = _max_cos(y)
            if v < best_val:
                best_val = v
                best_x = y.copy()
            x_cur = y
            # adapt radius: if total violation decreased, keep/expand else shrink
            # violation = sum max(0, c - t)
            # use v as proxy
            if v < t_targ + 0.005:
                r = min(r * 1.1, 0.03)
            # use dual-ish info: pairs with largest residual remain stressed;
            # they will be preferentially excluded next iter via topK sampling (already)
        outer += 1
        # early exit if we already beat lowest target substantially
        if best_val < 0.4605:
            break

    # final ordinary LP polish on best
    remain = HARD_END - time.time() - 2.0
    if remain > 5.0:
        budget = min(POLISH_BUDGET, remain)
        try:
            best_x, best_val = _lp_polish_minmax(best_x, budget, t_start, HARD_END - 1.0)
        except Exception:
            pass
    best_x = _norm(best_x)
    return best_x
