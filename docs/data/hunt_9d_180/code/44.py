import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def lp_polish_deadline(x, deadline, radius_init=1e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = radius_init
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = (x @ x.T)[iu]
        m = float(G.max())
        sel = G > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        csub = G[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csub, A_eq=A_eq, b_eq=np.zeros(n),
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
    return x, best

def solve():
    t0 = time.time()
    HARD_END = t0 + 176.0
    FEAS_END = t0 + 140.0
    POLISH_END = t0 + 174.0
    rng = np.random.RandomState(0)

    # load start
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            c = np.asarray(np.load(fn), dtype=float)
            if c.shape == (180, 9):
                c = c / np.linalg.norm(c, axis=1, keepdims=True)
                if np.all(np.isfinite(c)):
                    cands.append(c)
        except Exception:
            pass
    if len(cands) == 0:
        x_init = rng.randn(180, 9)
        x_init /= np.linalg.norm(x_init, axis=1, keepdims=True)
    else:
        vals = [max_cos(c) for c in cands]
        x_init = cands[int(np.argmin(vals))].copy()

    n, d = 180, 9
    x_best = x_init.copy()
    b_best = max_cos(x_best)

    # small deterministic tangent kick to leave exact vertex
    kick = rng.randn(n, d) * 0.004
    kick -= (np.sum(kick * x_best, axis=1, keepdims=True)) * x_best
    x_cur = x_best + kick
    x_cur /= np.linalg.norm(x_cur, axis=1, keepdims=True)
    cur_viol_ref = None

    targets = [0.4620, 0.4615, 0.4610]
    iu0 = np.triu_indices(n, 1)
    iuI = iu0[0]; iuJ = iu0[1]
    npairs = len(iuI)

    # exclusion map: key (i,j) with i<j -> expiry iteration
    excl_exp = {}
    it = 0
    radius = 0.02
    # rotate target order deterministically
    t_order = [0, 1, 2, 1, 0, 2]

    max_feas_iters = 55
    while time.time() < FEAS_END and it < max_feas_iters and time.time() < HARD_END - 30:
        for to in t_order:
            if time.time() >= FEAS_END or it >= max_feas_iters:
                break
            if time.time() >= HARD_END - 30:
                break
            t = targets[to]
            # do 2-3 iters per target visit
            for _inner in range(3):
                it += 1
                if time.time() >= FEAS_END or time.time() >= HARD_END - 30:
                    break
                # expire exclusions
                # compute grams
                Gfull = x_cur @ x_cur.T
                Gall = Gfull[iuI, iuJ]
                # candidates above t-0.01
                thr = t - 0.01
                idx = np.where(Gall > thr)[0]
                if idx.size == 0:
                    v = max_cos(x_cur)
                    if v < b_best:
                        b_best = v; x_best = x_cur.copy()
                    break
                # take 800 tightest
                if idx.size > 800:
                    part = np.argpartition(Gall[idx], -800)[-800:]
                    idx = idx[part]
                    # sort desc for determinism
                    o = np.argsort(-Gall[idx])
                    idx = idx[o]
                else:
                    o = np.argsort(-Gall[idx])
                    idx = idx[o]
                I = iuI[idx]; J = iuJ[idx]
                csub = Gall[idx]
                # apply exclusion filter
                keep_mask = np.ones(len(idx), dtype=bool)
                if excl_exp:
                    for k in range(len(idx)):
                        a = int(I[k]); b = int(J[k])
                        if a > b: a, b = b, a
                        if (a, b) in excl_exp and excl_exp[(a, b)] > it:
                            keep_mask[k] = False
                # if everything excluded, clear oldest
                if not np.any(keep_mask):
                    keep_mask[:] = True
                    excl_exp.clear()
                I = I[keep_mask]; J = J[keep_mask]; csub = csub[keep_mask]
                p = len(I)
                if p == 0:
                    continue
                nv = n * d + p
                # build LP: min sum s  s.t. <xj,dxi>+<xi,dxj> - s <= t - c
                rows = np.repeat(np.arange(p), 2 * d + 1)
                # cols: dx part + slack part
                base_cols_dx_I = (I[:, None] * d + np.arange(d)).ravel()
                base_cols_dx_J = (J[:, None] * d + np.arange(d)).ravel()
                # interleave per row: need cols in row order
                cols = np.empty(p * (2 * d + 1), dtype=int)
                cols[0::(2*d+1)] = np.repeat(0, 0)  # placeholder, will overwrite below vectorized
                # simpler: build via concatenation per row block
                cI = (I[:, None] * d + np.arange(d))
                cJ = (J[:, None] * d + np.arange(d))
                cS = (n * d + np.arange(p))[:, None]
                cols = np.concatenate([cI, cJ, cS], axis=1).ravel()
                vI = x_cur[J]
                vJ = x_cur[I]
                vS = -np.ones((p, 1))
                vals = np.concatenate([vI, vJ, vS], axis=1).ravel()
                b_ub = t - csub
                # equality: x_i . dx_i = 0
                try:
                    A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
                    A_eq = coo_matrix((x_cur.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
                except Exception:
                    continue
                obj = np.zeros(nv); obj[n*d:] = 1.0
                bounds = [(-radius, radius)] * (n * d) + [(0.0, None)] * p
                try:
                    res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.zeros(n),
                                  bounds=bounds, method="highs")
                except Exception:
                    radius = max(0.01, radius * 0.7)
                    continue
                if res.status != 0:
                    radius = max(0.01, radius * 0.7)
                    # force rotation: drop a random subset next time
                    if p > 20:
                        selr = rng.choice(p, size=min(15, p), replace=False)
                        for s_ in selr:
                            a = int(I[s_]); b = int(J[s_])
                            if a > b: a, b = b, a
                            excl_exp[(a, b)] = it + 3
                    continue
                dx = res.x[:n*d].reshape(n, d)
                y = x_cur + dx
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                v = max_cos(y)
                if v < b_best:
                    b_best = v
                    x_best = y.copy()
                # violation w.r.t current target
                Gy = y @ y.T
                Gall_y = Gy[iuI, iuJ]
                viol = float(np.sum(np.maximum(0.0, Gall_y - t)))
                # accept if violation improved or max improved
                Gc = Gfull[iuI, iuJ]
                viol_c = float(np.sum(np.maximum(0.0, Gc - t)))
                if viol < viol_c or v < max_cos(x_cur):
                    x_cur = y
                    radius = min(0.03, radius * 1.05)
                else:
                    # stuck: accept anyway with small random tangent nudge to change basin
                    # but keep x_cur moving to avoid re-solving same point
                    nse = rng.randn(n, d) * (radius * 0.15)
                    nse -= (np.sum(nse * y, axis=1, keepdims=True)) * y
                    y2 = y + nse
                    y2 /= np.linalg.norm(y2, axis=1, keepdims=True)
                    x_cur = y2
                    radius = max(0.01, radius * 0.95)
                # dual-based stressed contact identification
                try:
                    du = None
                    if hasattr(res, "ineqlin") and res.ineqlin is not None:
                        if hasattr(res.ineqlin, "marginals"):
                            du = np.asarray(res.ineqlin.marginals, dtype=float)
                        elif hasattr(res.ineqlin, "dual"):
                            du = np.asarray(res.ineqlin.dual, dtype=float)
                    if du is None or du.size != p or not np.all(np.isfinite(du)):
                        # fallback: use slack violation magnitude
                        lin_pred = csub + np.sum(vI * dx[I], axis=1) + np.sum(vJ * dx[J], axis=1)
                        du = np.maximum(0.0, lin_pred - t)
                    ad = np.abs(du)
                    # pick rotating subset among most stressed
                    K = min(30, p)
                    if K > 0 and np.max(ad) > 0:
                        top = np.argsort(-ad)[:K]
                        # rotate: exclude ~half of them, deterministic-random choice
                        ne = min(12, K)
                        chosen = rng.choice(top, size=ne, replace=False)
                        for s_ in chosen:
                            a = int(I[s_]); b = int(J[s_])
                            if a > b: a, b = b, a
                            excl_exp[(a, b)] = it + int(rng.randint(2, 5))
                        # prune old entries
                        if len(excl_exp) > 300:
                            # drop expired / oldest
                            keys = [k for k, e in excl_exp.items() if e <= it]
                            for k in keys:
                                del excl_exp[k]
                            while len(excl_exp) > 200:
                                excl_exp.pop(next(iter(excl_exp)))
                    # early success: if feasible for this target, go to lower target
                    if v <= t:
                        break
                except Exception:
                    pass
                if time.time() >= FEAS_END:
                    break

    # reserve final time for one ordinary LP polish of the best
    if time.time() < POLISH_END:
        try:
            xp, vp = lp_polish_deadline(x_best, POLISH_END, radius_init=1e-3)
            if vp < b_best and np.all(np.isfinite(xp)):
                x_best = xp; b_best = vp
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
