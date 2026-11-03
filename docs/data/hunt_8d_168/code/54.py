import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsmr

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lp_polish(x, deadline, r0=1.2e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = r0
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        try:
            c = (x @ x.T)[iu]
        except Exception:
            break
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > 3000:
            idx = np.argsort(c[sel])[-3000:]
            I, J = I[idx], J[idx]
            cc = c[sel][idx]
        else:
            cc = c[sel]
        p = len(I)
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d),
                                   J[:, None] * d + np.arange(d),
                                   np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
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
        try:
            y = x + res.x[:-1].reshape(n, d)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            vv = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
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
    w = np.ones(p) / p
    Ti = X[J] - gg[:, None] * X[I]
    Tj = X[I] - gg[:, None] * X[J]
    n = X.shape[0]
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

def _homotopy_target(X0, c, deadline, rng):
    n, d = X0.shape
    N = n * d
    X = X0.copy()
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    iu0 = np.triu_indices(n, 1)
    def viol_energy(X_):
        G = X_ @ X_.T
        g = G[iu0]
        v = g - c
        pos = v > 0
        if np.any(pos):
            return float((v[pos] ** 2).sum()), int(pos.sum()), float(v[pos].max())
        return 0.0, 0, -1.0
    bestE, bestP, _ = viol_energy(X)
    bestX = X.copy()
    try:
        best_max = _max_cos(X)
    except Exception:
        best_max = 10.0
    radius = 1e-2
    damp = 1.0
    noimp = 0
    drop_counts = {}
    drop_ban = set()
    for it in range(60):
        if time.time() > deadline:
            break
        G = X @ X.T
        gfull = G[iu0]
        sel = gfull > (c - 3e-4)
        Iall = iu0[0][sel]
        Jall = iu0[1][sel]
        gall = gfull[sel]
        if len(gall) == 0:
            return X, True, _max_cos(X)
        if len(gall) > 4000:
            idx = np.argsort(gall)[-4000:]
            Iall, Jall, gall = Iall[idx], Jall[idx], gall[idx]
        viol = gall - c
        posm = viol > 0
        E = float((viol[posm] ** 2).sum()) if np.any(posm) else 0.0
        nviol = int(posm.sum())
        if nviol == 0:
            return X, True, float(gall.max())
        if (it % 3 == 0) and len(gall) > 10:
            try:
                w = _stress_weights(X, Iall, Jall, gall, iters=20)
                tiny = (w < 0.2 / len(w)) & (~posm)
                for k in np.where(tiny)[0]:
                    key = (int(Iall[k]), int(Jall[k]))
                    drop_counts[key] = drop_counts.get(key, 0) + 1
                    if drop_counts[key] >= 2:
                        drop_ban.add(key)
                for k in np.where(~tiny)[0]:
                    key = (int(Iall[k]), int(Jall[k]))
                    if key in drop_counts:
                        drop_counts[key] = 0
            except Exception:
                pass
        if len(drop_ban) > 0:
            keep = np.ones(len(Iall), dtype=bool)
            for k in range(len(Iall)):
                if (not posm[k]) and ((int(Iall[k]), int(Jall[k])) in drop_ban):
                    keep[k] = False
            if np.any(~keep):
                Iall = Iall[keep]; Jall = Jall[keep]
                gall = gall[keep]; viol = viol[keep]; posm = posm[keep]
                E = float((viol[posm] ** 2).sum()) if np.any(posm) else 0.0
                nviol = int(posm.sum())
                if nviol == 0:
                    return X, True, float(gall.max()) if len(gall) > 0 else _max_cos(X)
        Iv = Iall[posm]; Jv = Jall[posm]; vv = viol[posm]
        pv = len(Iv)
        if pv == 0:
            return X, True, _max_cos(X)
        if pv > 3000:
            o = np.argsort(vv)[-3000:]
            Iv, Jv, vv = Iv[o], Jv[o], vv[o]
            pv = 3000
        try:
            rows = np.repeat(np.arange(pv), 2 * d)
            cI = Iv[:, None] * d + np.arange(d)
            cJ = Jv[:, None] * d + np.arange(d)
            cols2 = np.empty((pv, 2 * d), dtype=np.int64)
            cols2[:, :d] = cI
            cols2[:, d:] = cJ
            cols = cols2.ravel()
            vals2 = np.empty((pv, 2 * d), dtype=np.float64)
            vals2[:, :d] = X[Jv]
            vals2[:, d:] = X[Iv]
            vals = vals2.ravel()
            Jmat = coo_matrix((vals, (rows, cols)), shape=(pv, N)).tocsr()
        except Exception:
            break
        try:
            res = lsmr(Jmat, -vv, damp=damp, maxiter=min(400, max(60, pv // 5)), show=False)
            dx = np.asarray(res[0], dtype=np.float64)
            if not np.all(np.isfinite(dx)):
                radius *= 0.5
                damp = min(damp * 2.0, 1e3)
                continue
        except Exception:
            radius *= 0.5
            damp = min(damp * 2.0, 1e3)
            continue
        DX = dx.reshape(n, d)
        dot = (DX * X).sum(axis=1, keepdims=True)
        DX = DX - dot * X
        nrms = np.sqrt((DX * DX).sum(axis=1))
        mx = float(nrms.max()) if len(nrms) > 0 else 0.0
        if mx > radius and mx > 0:
            DX *= (radius / (mx + 1e-30))
        Xt = X + DX
        nr = np.linalg.norm(Xt, axis=1, keepdims=True)
        nr[~np.isfinite(nr)] = 1.0
        nr[nr < 1e-12] = 1.0
        Xt /= nr
        try:
            En, nvn, _ = viol_energy(Xt)
            mxt = _max_cos(Xt)
        except Exception:
            radius *= 0.5
            damp = min(damp * 2.0, 1e3)
            continue
        improved = (En < E * (1.0 - 1e-12)) or (En <= E + 1e-18 and mxt < best_max - 1e-15)
        if improved:
            X = Xt
            if En < bestE:
                bestE = En
                bestX = X.copy()
                noimp = 0
            else:
                noimp += 1
            if mxt < best_max:
                best_max = mxt
            radius = min(radius * 1.25, 1e-2)
            damp = max(damp * 0.7, 1e-3)
            if nvn == 0:
                return X, True, mxt
        else:
            noimp += 1
            radius *= 0.5
            damp = min(damp * 2.0, 1e3)
            if radius < 1e-4 and noimp >= 3:
                try:
                    kick = rng.randn(n, d) * (radius * 0.5 + 1e-5)
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
        if radius < 5e-5 and noimp >= 5:
            break
    try:
        m = _max_cos(bestX)
    except Exception:
        return X, False, _max_cos(X)
    if m <= c + 1e-15:
        return bestX, True, m
    try:
        cur = _max_cos(X)
    except Exception:
        return bestX, False, m
    if cur < m:
        return X, False, cur
    return bestX, False, m

def solve():
    t0 = time.time()
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
            continue
    if best is None:
        best = rng.randn(n, d)
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    incumbent = best.copy()
    inc_val = best_val
    hom_end = t0 + 95.0
    final_end = t0 + 172.0
    deltas = [1e-6, 2e-6, 5e-6]
    for di, delta in enumerate(deltas):
        if time.time() > hom_end:
            break
        c = inc_val - delta
        X0 = incumbent.copy()
        try:
            jit = rng.randn(n, d) * 1e-4
            jit -= (jit * X0).sum(axis=1, keepdims=True) * X0
            X0 = X0 + jit
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        except Exception:
            pass
        remain = hom_end - time.time()
        left_targets = len(deltas) - di
        per_target = min(32.0, max(8.0, remain / max(left_targets, 1)))
        tlim = time.time() + per_target
        try:
            Xf, ok, mv = _homotopy_target(X0, c, tlim, rng)
        except Exception:
            continue
        try:
            mv_exact = _max_cos(Xf)
        except Exception:
            continue
        if mv_exact < inc_val - 1e-15:
            incumbent = Xf.copy()
            inc_val = mv_exact
            if ok:
                continue
            # re-target from improved value without consuming next delta fully
            continue
        elif ok:
            incumbent = Xf.copy()
            inc_val = mv_exact
    # finish with trust-region LP squeeze on remaining time
    try:
        if time.time() < final_end - 2.0:
            incumbent = _lp_polish(incumbent, final_end - 1.0, r0=1.2e-3)
            inc_val = _max_cos(incumbent)
    except Exception:
        pass
    try:
        incumbent = incumbent / np.linalg.norm(incumbent, axis=1, keepdims=True)
        bad = ~np.all(np.isfinite(incumbent), axis=1)
        if np.any(bad):
            incumbent[bad] = best[bad]
            incumbent /= np.linalg.norm(incumbent, axis=1, keepdims=True)
    except Exception:
        incumbent = best.copy()
    return incumbent
