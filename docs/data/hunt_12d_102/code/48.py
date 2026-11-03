import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def feas_step(x, t, r, I, J, c_sel):
    n, d = x.shape
    p = len(I)
    nv = n * d + p
    # A_ub: p x nv
    # each row k: x[J[k]] at block I[k], x[I[k]] at block J[k], -1 at slack k
    rows = np.repeat(np.arange(p), 2 * d + 1)
    # cols
    ci = (I[:, None] * d + np.arange(d)).ravel()
    cj = (J[:, None] * d + np.arange(d)).ravel()
    cs = (n * d + np.arange(p))
    # interleave per row: need cols in row order
    cols = np.empty(p * (2 * d + 1), dtype=np.int64)
    vals = np.empty(p * (2 * d + 1), dtype=float)
    # build row by row via vectorized blocks
    # for speed use python loop over p (p~300, fine)
    for k in range(p):
        base = k * (2 * d + 1)
        i = int(I[k]); j = int(J[k])
        cols[base:base+d] = i * d + np.arange(d)
        vals[base:base+d] = x[j]
        cols[base+d:base+2*d] = j * d + np.arange(d)
        vals[base+d:base+2*d] = x[i]
        cols[base+2*d] = n * d + k
        vals[base+2*d] = -1.0
    A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
    b_ub = (t - c_sel).astype(float)
    # A_eq: n x nv, row i: x[i] at block i
    erows = np.repeat(np.arange(n), d)
    ecols = np.arange(n * d)
    evals = x.ravel()
    A_eq = coo_matrix((evals, (erows, ecols)), shape=(n, nv)).tocsr()
    b_eq = np.zeros(n)
    obj = np.zeros(nv)
    obj[n*d:] = 1.0
    bounds = [(-r, r)] * (n * d) + [(0.0, None)] * p
    res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    return res

def lp_polish_minmax(x, best, deadline):
    n, d = x.shape
    radius = 1e-3
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline:
        if radius < 1e-14:
            break
        G = x @ x.T
        c = G[iu0]
        mcur = float(c.max())
        sel = c > mcur - max(20 * radius, 1e-9)
        I = iu0[0][sel]; J = iu0[1][sel]
        csel = c[sel]
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
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
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
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() >= deadline:
            break
    return x, best

def solve():
    t_start = time.time()
    deadline = t_start + 165.0
    np.random.seed(0)
    try:
        xb = np.array(np.load("best.npy"), dtype=float)
    except Exception:
        try:
            xb = np.array(np.load("record.npy"), dtype=float)
        except Exception:
            r = np.random.default_rng(0).standard_normal((102, 12))
            r /= np.linalg.norm(r, axis=1, keepdims=True)
            return r
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    n, d = 102, 12
    if xb.shape != (102, 12):
        a = np.random.default_rng(0).standard_normal((102, 12))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        return a
    x = xb.copy()
    best = max_cos(x)
    incumbent0 = best
    iu = np.triu_indices(n, 1)
    # feasibility targets
    deltas = [1e-6, 3e-6, 1e-5, 3e-5]
    K = 300  # constraint set size 200-500
    for di, delta in enumerate(deltas):
        if time.time() > deadline - 25.0:
            break
        t = best - delta  # target below current best
        # if t already infeasible check: need max < best, so recompute
        # init radius
        r = 0.008
        # compute initial total slack
        G0 = x @ x.T
        c0 = G0[iu]
        Stall = 0
        # max iters per target to bound time
        max_iter = 35
        for it in range(max_iter):
            if time.time() > deadline - 25.0:
                break
            if r < 1e-8:
                break
            G = x @ x.T
            c_all = G[iu]
            cur_max = float(c_all.max())
            if cur_max < best and cur_max < incumbent0:
                best = cur_max
                break
            # total slack vs target t
            tot = float(np.sum(np.maximum(0.0, c_all - t)))
            if tot == 0.0:
                best = cur_max
                break
            # select top-K most violated / largest pairs
            # argsort descending, take K
            if len(c_all) > K:
                part = np.argpartition(-c_all, K - 1)[:K]
                # sort those K for determinism
                o = np.argsort(-c_all[part])
                sel_idx = part[o]
            else:
                sel_idx = np.argsort(-c_all)
            I = iu[0][sel_idx]; J = iu[1][sel_idx]
            c_sel = c_all[sel_idx]
            try:
                res = feas_step(x, t, r, I, J, c_sel)
            except Exception:
                r *= 0.5
                Stall += 1
                continue
            if res.status != 0:
                r *= 0.5
                Stall += 1
                if Stall >= 6:
                    break
                continue
            u = res.x[:n*d].reshape(n, d)
            y = x + u
            nr = np.linalg.norm(y, axis=1, keepdims=True)
            nr[nr == 0] = 1.0
            y /= nr
            Gy = y @ y.T
            cy = Gy[iu]
            tot_new = float(np.sum(np.maximum(0.0, cy - t)))
            cur_max_new = float(cy.max())
            # progress if total slack decreased meaningfully
            if tot_new < tot * (1.0 - 1e-4) or cur_max_new < cur_max - 1e-12:
                x = y
                # increase radius after progress
                r = min(r * 1.35, 0.02)
                Stall = 0
                if cur_max_new < best:
                    best = cur_max_new
                    # feasible below incumbent0 -> update target to push further within same delta loop?
                    # keep t as is to drive further below; if much below, break to next delta
                    if best < incumbent0 - 1e-12:
                        break
                # check feasibility: max below t? then success for this target
                if cur_max_new <= t:
                    best = cur_max_new
                    break
            else:
                # stall: decrease r, do not accept (or accept if max improved slightly?)
                # accept small max improvement even if slack stalled to allow movement
                if cur_max_new < cur_max:
                    x = y
                r *= 0.5
                Stall += 1
                if Stall >= 8:
                    break
        # after each target, if we improved below incumbent0, proceed to next (harder) target
        # if we failed the easiest target, harder ones will fail -> break
        if best >= incumbent0 - 5e-13 and di == 0:
            # no progress on easiest; try once more with smaller K / larger r? just break to polish
            pass
        if best < incumbent0 - 1e-12:
            incumbent0 = best
            # continue to next delta to push further
            continue
    # final ordinary LP polish with remaining time
    try:
        x, best = lp_polish_minmax(x, best, deadline)
    except Exception:
        pass
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x
