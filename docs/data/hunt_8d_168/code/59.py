import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsmr

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lp_polish(x, seconds, r0=1.2e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = r0
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > 3000:
            idx = np.argsort(c[sel])[-3000:]
            I, J = I[idx], J[idx]
            cc = c[sel][idx]
            p = 3000
        else:
            cc = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cc, A_eq=A_eq,
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
        vv = _max_cos(y)
        if vv < best:
            x, best = y, vv
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _stress_weights(X, I, J, gg, iters=20):
    p = len(I)
    if p == 0:
        return np.zeros(0)
    n, d = X.shape
    w = np.ones(p) / p
    Ti = X[J] - gg[:, None] * X[I]
    Tj = X[I] - gg[:, None] * X[J]
    for _ in range(iters):
        r = np.zeros_like(X)
        np.add.at(r, I, w[:, None] * Ti)
        np.add.at(r, J, w[:, None] * Tj)
        Ri = r[I]
        Rj = r[J]
        grad = (Ti * Ri).sum(axis=1) + (Tj * Rj).sum(axis=1)
        gmax = grad.max()
        eta = 8.0
        w = w * np.exp(-eta * (grad - gmax))
        s = w.sum()
        if s <= 0 or not np.isfinite(s):
            w = np.ones(p) / p
            break
        w /= s
    return w

def _homotopy_target(X0, c, t_end):
    n, d = X0.shape
    N = n * d
    X = X0.copy()
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    iu0 = np.triu_indices(n, 1)
    ar = np.arange(d)
    def energy_at(X_):
        g = (X_ @ X_.T)[iu0]
        v = g - c
        pos = v > 0
        E = float((v[pos] ** 2).sum()) if np.any(pos) else 0.0
        nv = int(pos.sum())
        mx = float(g.max())
        return E, nv, mx
    E0, nv0, mx0 = energy_at(X)
    bestE = E0
    bestX = X.copy()
    best_nv = nv0
    best_mx = mx0
    if nv0 == 0:
        return X, True, mx0
    radius = 1e-2
    damp = 1.0
    noimp = 0
    drop_counts = {}
    drop_ban = set()
    max_it = 60
    for it in range(max_it):
        if time.time() > t_end:
            break
        G = X @ X.T
        gfull = G[iu0]
        sel = gfull > (c - 3e-4)
        Iall = iu0[0][sel]
        Jall = iu0[1][sel]
        gall = gfull[sel]
        if len(gall) == 0:
            return X, True, float(gfull.max())
        # filter banned non-violating
        if len(drop_ban) > 0 and len(gall) > 0:
            keep = np.ones(len(gall), dtype=bool)
            for k in range(len(gall)):
                i = int(Iall[k]); j = int(Jall[k])
                if gall[k] <= c and (i, j) in drop_ban:
                    keep[k] = False
            if np.any(keep):
                Iall, Jall, gall = Iall[keep], Jall[keep], gall[keep]
            # if all banned, clear ban
            if len(gall) == 0:
                drop_ban.clear()
                G = X @ X.T
                gfull = G[iu0]
                sel = gfull > (c - 3e-4)
                Iall = iu0[0][sel]; Jall = iu0[1][sel]; gall = gfull[sel]
        if len(gall) > 4000:
            idx = np.argsort(gall)[-4000:]
            Iall, Jall, gall = Iall[idx], Jall[idx], gall[idx]
        viol = gall - c
        posm = viol > 0
        E = float((viol[posm] ** 2).sum()) if np.any(posm) else 0.0
        nviol = int(posm.sum())
        cur_max = float(gall.max())
        if nviol == 0:
            if cur_max < best_mx:
                bestX = X.copy(); best_mx = cur_max
            return X, True, cur_max
        # stress test every 3 steps
        if (it % 3 == 0) and len(gall) > 10:
            try:
                w = _stress_weights(X, Iall, Jall, gall, iters=20)
                thresh = 0.2 / len(w)
                for k in range(len(w)):
                    if (not posm[k]) and w[k] < thresh:
                        key = (int(Iall[k]), int(Jall[k]))
                        drop_counts[key] = drop_counts.get(key, 0) + 1
                        if drop_counts[key] >= 2:
                            drop_ban.add(key)
                    elif posm[k]:
                        key = (int(Iall[k]), int(Jall[k]))
                        if key in drop_counts:
                            drop_counts[key] = 0
                            if key in drop_ban and gall[k] > c:
                                drop_ban.discard(key)
                if len(drop_ban) > 2000:
                    # keep only most recent: clear half
                    drop_ban.clear()
                    drop_counts.clear()
            except Exception:
                pass
        # build violating system (cap to 3000 tightest viol)
        Iv = Iall[posm]; Jv = Jall[posm]; vv = viol[posm]
        order = np.argsort(vv)
        if len(order) > 3000:
            order = order[-3000:]
            Iv, Jv, vv = Iv[order], Jv[order], vv[order]
        p = len(Iv)
        if p == 0:
            return X, True, cur_max
        try:
            # sparse Jacobian
            rrows = np.repeat(np.arange(p), 2 * d)
            cI = (Iv[:, None] * d + ar).reshape(p, d)
            cJ = (Jv[:, None] * d + ar).reshape(p, d)
            cmat = np.empty((p, 2 * d), dtype=int)
            cmat[:, :d] = cI
            cmat[:, d:] = cJ
            ccols = cmat.ravel()
            vI = X[Jv]
            vJ = X[Iv]
            vmat = np.empty((p, 2 * d), dtype=float)
            vmat[:, :d] = vI
            vmat[:, d:] = vJ
            vvals = vmat.ravel()
            A = coo_matrix((vvals, (rrows, ccols)), shape=(p, N)).tocsr()
            b = -vv
            sol = lsmr(A, b, damp=float(damp), maxiter=400, show=False)
            dx = sol[0].reshape(n, d)
        except Exception:
            radius = max(radius * 0.5, 1e-4)
            damp = min(damp * 2.0, 1e4)
            continue
        # project to tangent
        dot = (dx * X).sum(axis=1, keepdims=True)
        dx = dx - dot * X
        nrm = np.sqrt((dx * dx).sum(axis=1, keepdims=True)) + 1e-30
        sc = np.minimum(1.0, radius / nrm)
        dx = dx * sc
        Xt = X + dx
        Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
        En, nvn, mxn = energy_at(Xt)
        improved = (En < E * (1.0 - 1e-4)) or (En < E and nvn < nviol)
        if En < bestE - 1e-18:
            bestE = En; bestX = Xt.copy(); best_nv = nvn; best_mx = mxn
        elif mxn < best_mx and nvn <= best_nv:
            bestX = Xt.copy(); best_mx = mxn; best_nv = nvn
            if En < bestE:
                bestE = En
        if improved:
            X = Xt
            radius = min(1e-2, radius * 1.3)
            damp = max(damp * 0.7, 1e-3)
            noimp = 0
            # track max improvement even without full feasibility
            if mxn < best_mx:
                best_mx = mxn
        else:
            # try smaller step once
            radius = max(radius * 0.5, 1e-4)
            damp = min(damp * 2.0, 1e4)
            noimp += 1
            if noimp >= 12:
                break
            # still move if energy not worse by much? no
            if En < E:
                X = Xt
        # early exit if radius collapsed
        if radius <= 1.01e-4 and noimp >= 5:
            break
    # return best by energy / max
    # choose bestX; report feasibility
    Gb = (bestX @ bestX.T)[iu0]
    feas = bool(np.all(Gb <= c))
    return bestX, feas, float(Gb.max())

def solve():
    t0 = time.time()
    BUDGET = 175.0
    n, d = 168, 8
    rng = np.random.RandomState(0)
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape != (n, d):
                continue
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            if not np.all(np.isfinite(a)):
                continue
            va = _max_cos(a)
            if va < best_val:
                best_val = va
                best = a.copy()
        except Exception:
            pass
    if best is None:
        best = rng.randn(n, d)
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    init_val = float(best_val)
    cur_best = best.copy()
    cur_val = float(best_val)
    # homotopy targets below initial record
    offsets = [1e-6, 2e-6, 5e-6]
    # reserve ~55s for final LP polish
    t_homotopy_end = t0 + max(20.0, BUDGET - 60.0)
    per_target = (t_homotopy_end - time.time()) / max(len(offsets), 1)
    for oi, off in enumerate(offsets):
        if time.time() > t_homotopy_end:
            break
        c = init_val - off
        # if already below target, update init reference and skip
        if cur_val <= c:
            continue
        t_lim = time.time() + per_target
        # don't exceed homotopy budget
        if t_lim > t_homotopy_end:
            t_lim = t_homotopy_end
        try:
            Xt, feas, mx = _homotopy_target(cur_best, c, t_lim)
        except Exception:
            continue
        try:
            mv = _max_cos(Xt)
        except Exception:
            continue
        if mv < cur_val - 1e-12:
            cur_best = Xt.copy()
            cur_val = float(mv)
            if feas:
                init_val = float(mv) if mv < init_val else init_val
                # tighten remaining targets relative to new best? keep original offsets but allow next lower
                pass
        # if feasible, continue to next lower target immediately
        # if not feasible but improved max, still keep
        if time.time() > t0 + BUDGET - 55.0:
            break
    # finish with trust-region LP squeeze on incumbent
    remain = (t0 + BUDGET - 3.0) - time.time()
    if remain > 5.0:
        try:
            cur_best = _lp_polish(cur_best, seconds=remain, r0=1.2e-3)
            cur_val = _max_cos(cur_best)
        except Exception:
            pass
    # fallback: ensure we return at least input best
    try:
        if _max_cos(cur_best) > best_val:
            cur_best = best.copy()
    except Exception:
        cur_best = best.copy()
    cur_best /= np.linalg.norm(cur_best, axis=1, keepdims=True)
    return cur_best
