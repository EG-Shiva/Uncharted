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

def _stress_weights(X, I, J, gg, iters=25):
    p = len(I)
    if p == 0:
        return np.zeros(0)
    n, d = X.shape
    w = np.ones(p) / p
    Ti = X[J] - gg[:, None] * X[I]
    Tj = X[I] - gg[:, None] * X[J]
    for _ in range(iters):
        r = np.zeros_like(X)
        # r_i = sum w * T
        np.add.at(r, I, w[:, None] * Ti)
        np.add.at(r, J, w[:, None] * Tj)
        Ri = r[I]
        Rj = r[J]
        grad = (Ti * Ri).sum(axis=1) + (Tj * Rj).sum(axis=1)
        # exponentiated gradient on simplex
        gmax = grad.max()
        # step size
        eta = 8.0
        w = w * np.exp(-eta * (grad - gmax))
        s = w.sum()
        if s <= 0 or not np.isfinite(s):
            w = np.ones(p) / p
            break
        w /= s
    return w

def _homotopy_target(X0, c, time_lim, rng):
    n, d = X0.shape
    N = n * d
    X = X0.copy()
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    iu0 = np.triu_indices(n, 1)
    # initial violation energy
    def viol_energy(X_):
        G = X_ @ X_.T
        g = G[iu0]
        v = g - c
        pos = v > 0
        return float((v[pos] ** 2).sum()), int(pos.sum()), float(v[pos].max()) if np.any(pos) else -1.0
    bestE, bestP, _ = viol_energy(X)
    bestX = X.copy()
    best_max = _max_cos(X)
    radius = 1e-2
    damp = 1.0
    noimp = 0
    # for persistent drop tracking: map pair key -> count
    drop_counts = {}
    drop_ban = set()
    t_end = time_lim
    max_it = 60
    for it in range(max_it):
        if time.time() > t_end:
            break
        G = X @ X.T
        gfull = G[iu0]
        sel = gfull > (c - 3e-4)
        # exclude banned non-violating contacts
        if len(drop_ban) > 0 and np.any(sel):
            # build keys for selected
            pass
        Iall = iu0[0][sel]
        Jall = iu0[1][sel]
        gall = gfull[sel]
        if len(gall) == 0:
            return X, True, _max_cos(X)
        # cap to tightest 4000
        if len(gall) > 4000:
            idx = np.argsort(gall)[-4000:]
            Iall, Jall, gall = Iall[idx], Jall[idx], gall[idx]
        viol = gall - c
        posm = viol > 0
        E = float((viol[posm] ** 2).sum())
        nviol = int(posm.sum())
        cur_max = float(gall.max())
        if nviol == 0:
            return X, True, cur_max
        # stress test every few steps to prune loose contacts
        if (it % 3 == 0) and len(gall) > 10:
            try:
                w = _stress_weights(X, Iall, Jall, gall, iters=20)
                # drop persistently tiny-weight non-violating contacts
                # only consider non-violating
                tiny = (w < 0.2 / len(w)) & (~posm)
                # update counts
                # use index key (i,j)
                for k in np.where(tiny)[0]:
                    key = (int(Iall[k]), int(Jall[k]))
                    drop_counts[key] = drop_counts.get(key, 0) + 1
                    if drop_counts[key] >= 2:
                        drop_ban.add(key)
                # reset counts for non-tiny
                for k in np.where(~tiny)[0]:
                    key = (int(Iall[k]), int(Jall[k]))
                    if key in drop_counts:
                        drop_counts[key] = 0
            except Exception:
                pass
        # apply ban: remove banned non-violating from consideration (they stay out of GN)
        # violating pairs are never banned
        if len(drop_ban) > 0:
            keep = np.ones(len(Iall), dtype=bool)
            for k in range(len(Iall)):
                if not posm[k] and (int(Iall[k]), int(Jall[k])) in drop_ban:
                    keep[k] = False
            # keep at least violating
            if np.any(~keep):
                Iall, Jall, gall, viol, posm = Iall[keep], Jall[keep], gall[keep], viol[keep], posm[keep]
        # build Jacobian for violating pairs only
        Iv = Iall[posm]
        Jv = Jall[posm]
        vv = viol[posm]
        pv = len(Iv)
        if pv == 0:
            return X, True, cur_max
        # cap violating to tightest 3000 if huge
        if pv > 3000:
            o = np.argsort(vv)[-3000:]
            Iv, Jv, vv = Iv[o], Jv[o], vv[o]
            pv = 3000
        # sparse Jacobian pv x N
        try:
            rows = np.repeat(np.arange(pv), d * 2)
            cI = (Iv[:, None] * d + np.arange(d)).ravel()
            cJ = (Jv[:, None] * d + np.arange(d)).ravel()
            cols = np.empty(pv * d * 2, dtype=np.int64)
            cols[0::2] = np.empty(0)  # placeholder
            # interleave properly: first d per row are I-block, next d are J-block
            cols = np.concatenate([ (Iv[:, None]*d+np.arange(d)), (Jv[:, None]*d+np.arange(d)) ], axis=1).ravel()
            vals = np.concatenate([X[Jv], X[Iv]], axis=1).ravel() if False else np.concatenate([X[Jv], X[Iv]], axis=0).ravel(order='C')
            # Above concat is wrong order; rebuild correctly row-major:
            vals2 = np.empty((pv, 2*d))
            vals2[:, :d] = X[Jv]
            vals2[:, d:] = X[Iv]
            vals = vals2.ravel()
            Jmat = coo_matrix((vals, (rows, cols)), shape=(pv, N)).tocsr()
        except Exception:
            break
        # damped Gauss-Newton via lsmr: J dx = -vv
        try:
            res = lsmr(Jmat, -vv, damp=damp, maxiter=min(400, max(50, pv//5)))
            dx = np.asarray(res[0], dtype=np.float64)
            if not np.all(np.isfinite(dx)):
                radius *= 0.5
                damp *= 2.0
                continue
        except Exception:
            radius *= 0.5
            damp *= 2.0
            continue
        DX = dx.reshape(n, d)
        # project onto tangent spaces
        dot = (DX * X).sum(axis=1, keepdims=True)
        DX = DX - dot * X
        # trust region per-point clip
        nrms = np.sqrt((DX * DX).sum(axis=1))
        mx = nrms.max()
        if mx > radius:
            DX *= (radius / (mx + 1e-30))
        # also global safety
        Xt = X + DX
        Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
        En, nvn, _ = viol_energy(Xt)
        # feasibility improvement?
        if En < E * (1.0 - 1e-12) or (En <= E and _max_cos(Xt) < best_max):
            X = Xt
            if En < bestE:
                bestE = En
                bestX = X.copy()
                noimp = 0
            else:
                noimp += 1
            mxt = _max_cos(X)
            if mxt < best_max:
                best_max = mxt
            # expand trust
            radius = min(radius * 1.25, 1e-2)
            damp = max(damp * 0.7, 1e-3)
        else:
            noimp += 1
            radius *= 0.5
            damp = min(damp * 2.0, 1e3)
            if radius < 1e-4 and noimp >= 3:
                # try small random tangent kick to change topology
                try:
                    kick = rng.randn(n, d) * (radius * 0.5)
                    kick -= (kick * X).sum(axis=1, keepdims=True) * X
                    Xk = X + kick
                    Xk /= np.linalg.norm(Xk, axis=1, keepdims=True)
                    Ek, _, _ = viol_energy(Xk)
                    if Ek < E:
                        X = Xk
                        radius = 2e-4
                        noimp = 0
                        continue
                except Exception:
                    pass
        if noimp >= 8 and radius <= 1.1e-4:
            break
        # early exit if trust collapsed
        if radius < 5e-5 and noimp >= 5:
            break
    # check if best reached target
    m = _max_cos(bestX)
    if m <= c:
        return bestX, True, m
    # return closest
    cur = _max_cos(X)
    if cur < m:
        return X, False, cur
    return bestX, False, m

def solve():
    t0 = time.time()
    BUDGET = 175.0
    n, d = 168, 8
    rng = np.random.RandomState(7)
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
    incumbent = best.copy()
    inc_val = best_val
    # target homotopy: c = max - deltas
    deltas = [1e-6, 2e-6, 5e-6]
    # time allocation: up to ~100s for homotopy
    hom_end = t0 + 105.0
    for di, delta in enumerate(deltas):
        if time.time() > hom_end:
            break
        c = inc_val - delta
        # small tangent jitter to leave exact basin (tiny)
        X0 = incumbent.copy()
        # per spec start from best; add 1e-4 jitter to break degeneracy
        try:
            jit = rng.randn(n, d) * 1e-4
            jit -= (jit * X0).sum(axis=1, keepdims=True) * X0
            X0 = X0 + jit
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        except Exception:
            pass
        per_target = min(32.0, max(8.0, (hom_end - time.time()) / (len(deltas) - di)))
        tlim = time.time() + per_target
        try:
            Xf, ok, mv = _homotopy_target(X0, c, tlim, rng)
        except Exception:
            continue
        # check max cos of result (exact)
        try:
            mv_exact = _max_cos(Xf)
        except Exception:
            continue
        if ok:
            # feasible: reached target
            incumbent = Xf.copy()
            inc_val = mv_exact
            # continue to next (lower) target
            continue
        else:
            # not feasible but may still improve incumbent
            if mv_exact < inc_val - 1e-12:
                incumbent = Xf.copy()
                inc_val = mv_exact
                # retry same delta? do not advance excessively; continue
            # if no improvement, try next delta anyway (larger delta unlikely) -> break to polish
            # still try next smaller? deltas increasing so break if first fails badly
            if mv_exact > inc_val + 1e-9:
                continue
    # finish with trust-region LP on incumbent
    remain = BUDGET - (time.time() - t0) - 2.0
    if remain > 5.0:
        try:
            # use up to 60s for final squeeze, leave margin
            secs = min(remain, 60.0)
            incumbent = _lp_polish(incumbent, seconds=secs, r0=1.2e-3)
            inc_val = _max_cos(incumbent)
        except Exception:
            pass
    # safety: if incumbent worse than best.npy, revert
    if inc_val > best_val and best is not None:
        incumbent = best.copy()
        inc_val = best_val
    incumbent /= np.linalg.norm(incumbent, axis=1, keepdims=True)
    # final shrink guard (strict feasibility, no zero rows)
    return incumbent
