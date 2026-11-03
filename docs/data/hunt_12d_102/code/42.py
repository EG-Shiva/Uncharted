import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos_and_all(x, iu):
    G = x @ x.T
    c = G[iu]
    return float(c.max()), c, G


def _feas_step(x, I, J, c_sel, t, r, n, d, P):
    nv = n * d + P
    ar = np.arange(d)
    cols = np.concatenate(
        [I[:, None] * d + ar, J[:, None] * d + ar,
         (n * d + np.arange(P))[:, None]], axis=1).ravel()
    vals = np.concatenate([x[J], x[I], -np.ones((P, 1))], axis=1).ravel()
    rows = np.repeat(np.arange(P), 2 * d + 1)
    A_ub = coo_matrix((vals, (rows, cols)), shape=(P, nv)).tocsr()
    A_eq = coo_matrix(
        (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
        shape=(n, nv)).tocsr()
    obj = np.zeros(nv)
    obj[n * d:] = 1.0
    b_ub = t - c_sel
    bounds = [(-r, r)] * (n * d) + [(0.0, None)] * P
    try:
        res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq,
                      b_eq=np.zeros(n), bounds=bounds, method="highs",
                      options={"time_limit": 5.0})
    except Exception:
        return None
    if res.status != 0:
        return None
    U = res.x[:n * d].reshape(n, d)
    return U


def _ordinary_polish(x, seconds, t_end):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    iu = np.triu_indices(n, 1)
    G = x @ x.T
    c0 = G[iu]
    best = float(c0.max())
    radius = 8e-4
    start = time.time()
    while time.time() - start < seconds and time.time() < t_end:
        if radius < 1e-14:
            break
        G = x @ x.T
        c = G[iu]
        mcur = float(c.max())
        sel = c > mcur - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        # cap to most tight to keep LP cheap
        if p > 800:
            idx = np.argpartition(-c[sel], 800)[:800]
            I, J = I[idx], J[idx]
            cs = c[sel][idx]
            p = 800
        else:
            cs = c[sel]
        nv = n * d + 1
        ar = np.arange(d)
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate(
            [I[:, None] * d + ar, J[:, None] * d + ar,
             np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix(
            (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
            shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": 5.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        G2 = y @ y.T
        vv = float(G2[iu].max())
        if vv < best:
            x, best = y, vv
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    DEADLINE = t0 + 165.0
    T_END = t0 + 172.0
    rng = np.random.default_rng(0)
    try:
        xb = np.array(np.load("best.npy"), dtype=float)
    except Exception:
        xb = None
    try:
        xr = np.array(np.load("record.npy"), dtype=float)
    except Exception:
        xr = None
    cands = []
    for c in (xb, xr):
        if c is not None:
            try:
                a = np.array(c, dtype=float)
                if a.shape != (102, 12):
                    continue
                a /= np.linalg.norm(a, axis=1, keepdims=True)
                if not np.all(np.isfinite(a)):
                    continue
                g = a @ a.T
                np.fill_diagonal(g, -2.0)
                cands.append((float(g.max()), a))
            except Exception:
                pass
    if len(cands) == 0:
        a = rng.standard_normal((102, 12))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        g = a @ a.T
        np.fill_diagonal(g, -2.0)
        best0 = float(g.max())
        x_best = a
    else:
        cands.sort(key=lambda t: t[0])
        best0, x_best = cands[0][0], cands[0][1].copy()
    best = best0
    n, d = 102, 12
    iu = np.triu_indices(n, 1)

    deltas = [1e-6, 3e-6, 1e-5, 3e-5]
    targets = [best0 - dd for dd in deltas]

    K = 300  # most-violated pairs maintained (200-500)

    for ti, t in enumerate(targets):
        if time.time() > DEADLINE - 12:
            break
        # start from current best; add tiny kick for harder targets to change basin
        x_cur = x_best.copy()
        if ti > 0:
            x_cur = x_cur + 0.002 * rng.standard_normal(x_cur.shape)
            x_cur /= np.linalg.norm(x_cur, axis=1, keepdims=True)
        # total slack w.r.t t
        G = x_cur @ x_cur.T
        c_all = G[iu]
        total_prev = float(np.sum(np.maximum(0.0, c_all - t)))
        maxc = float(c_all.max())
        if maxc < best:
            best = maxc
            x_best = x_cur.copy()
        if total_prev <= 0:
            continue  # already feasible
        r = 0.008
        stall = 0
        for it in range(40):
            if time.time() > DEADLINE - 10:
                break
            G = x_cur @ x_cur.T
            c_all = G[iu]
            maxc = float(c_all.max())
            if maxc < best - 1e-15:
                best = float(maxc)
                x_best = x_cur.copy()
            if maxc < t:
                # feasible for this target
                best = float(maxc)
                x_best = x_cur.copy()
                break
            # most violated K pairs
            if len(c_all) > K:
                part = np.argpartition(-c_all, K)[:K]
            else:
                part = np.arange(len(c_all))
            I = iu[0][part].astype(int)
            J = iu[1][part].astype(int)
            c_sel = c_all[part]
            U = _feas_step(x_cur, I, J, c_sel, t, r, n, d, K)
            if U is None:
                r = max(r * 0.5, 1e-6)
                stall += 1
                if stall >= 6:
                    # topology-changing kick: tangent jitter
                    pert = 0.001 * rng.standard_normal(x_cur.shape)
                    # project to tangent
                    dot = np.sum(pert * x_cur, axis=1, keepdims=True)
                    pert -= dot * x_cur
                    x_cur = x_cur + pert
                    x_cur /= np.linalg.norm(x_cur, axis=1, keepdims=True)
                    G2 = x_cur @ x_cur.T
                    total_prev = float(np.sum(np.maximum(0.0, G2[iu] - t)))
                    stall = 0
                    r = min(max(r * 1.2, 2e-4), 0.01)
                continue
            y = x_cur + U
            nr = np.linalg.norm(y, axis=1, keepdims=True)
            nr[nr == 0] = 1.0
            y /= nr
            Gy = y @ y.T
            cy = Gy[iu]
            total_new = float(np.sum(np.maximum(0.0, cy - t)))
            if total_new < total_prev - 1e-15:
                # progress: accept, increase r
                x_cur = y
                total_prev = total_new
                r = min(r * 1.3, 0.02)
                stall = 0
                mc = float(cy.max())
                if mc < best:
                    best = mc
                    x_best = y.copy()
                    if mc < t:
                        break
            elif total_new < total_prev:
                x_cur = y
                total_prev = total_new
                stall = 0
            else:
                # stall: decrease r
                r = max(r * 0.5, 1e-6)
                stall += 1
                if stall >= 6:
                    pert = 0.0015 * rng.standard_normal(x_cur.shape)
                    dot = np.sum(pert * x_cur, axis=1, keepdims=True)
                    pert -= dot * x_cur
                    x_cur = x_cur + pert
                    x_cur /= np.linalg.norm(x_cur, axis=1, keepdims=True)
                    G2 = x_cur @ x_cur.T
                    total_prev = float(np.sum(np.maximum(0.0, G2[iu] - t)))
                    stall = 0
                    if r < 3e-4:
                        r = 3e-4
            if r < 1e-6 and stall > 3:
                break
        # end inner: go to next (harder) target from improved best
        if time.time() > T_END - 15:
            break

    # final ordinary LP polish under strict deadline
    remain = DEADLINE - time.time()
    if remain > 3:
        try:
            x_best = _ordinary_polish(x_best, min(remain - 1.0, 30.0), T_END - 1.0)
        except Exception:
            pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
