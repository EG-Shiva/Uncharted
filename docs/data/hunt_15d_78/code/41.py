import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _smooth_refine(X0, betas_lams, maxiter_each, t_deadline=None):
    n, d = X0.shape
    iu0, iu1 = np.triu_indices(n, 1)
    Xcur_flat = _norm(X0).ravel().copy()
    for (beta, lam) in betas_lams:
        if t_deadline is not None and time.time() > t_deadline:
            break
        def fun(g):
            Y0 = g.reshape(n, d)
            rn = np.linalg.norm(Y0, axis=1, keepdims=True)
            rn = np.maximum(rn, 1e-12)
            Y = Y0 / rn
            G = Y @ Y.T
            gv = G[iu0, iu1]
            m = gv.max()
            e = np.exp(beta * (gv - m))
            S = e.sum()
            L = m + np.log(S) / beta
            w = e / S
            W = np.zeros((n, n))
            W[iu0, iu1] = w
            W = W + W.T
            Gr = W @ Y
            dot = (Gr * Y).sum(axis=1, keepdims=True)
            GY = (Gr - dot * Y) / rn
            if lam != 0.0:
                # isotropy penalty on Y: ||Y'Y/n - I/d||^2
                C = (Y.T @ Y) / n
                T = np.eye(d) / d
                D = C - T
                L = L + lam * float((D * D).sum())
                # grad w.r.t Y: (4/n) Y D, then chain to Y0
                Gr2 = (4.0 / n) * (Y @ D)
                dot2 = (Gr2 * Y).sum(axis=1, keepdims=True)
                GY = GY + (Gr2 - dot2 * Y) / rn
            return float(L), GY.ravel()
        try:
            r = minimize(fun, Xcur_flat, jac=True, method='L-BFGS-B',
                         options={'maxiter': maxiter_each, 'ftol': 1e-14, 'gtol': 1e-6, 'maxfun': 10000, 'maxcor': 12})
            Xcur_flat = r.x
        except Exception:
            break
    Xopt = Xcur_flat.reshape(n, d)
    return _norm(Xopt)

def _lp_polish(X, seconds):
    X = _norm(X)
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        Gf = X @ X.T
        c = Gf[iu0, iu1]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        # cap to tightest 2000 to keep LP fast
        if sel.sum() > 2000:
            idx = np.argpartition(-c, 2000)[:2000]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = sel & mask
        I, J = iu0[sel], iu1[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return X

def _find_voids(X, k, rng, nsamp=4000):
    n, d = X.shape
    Y = rng.standard_normal((nsamp, d))
    Y = _norm(Y)
    dots = Y @ X.T
    m = dots.max(axis=1)
    order = np.argsort(m)
    out = []
    Xaug = X.copy()
    # greedy sequential to spread voids
    for t in range(k):
        # pick best among samples w.r.t current Xaug
        if t == 0:
            best_i = order[0]
            v = Y[best_i]
        else:
            dd = Y @ Xaug.T
            mm = dd.max(axis=1)
            best_i = int(np.argmin(mm))
            v = Y[best_i]
        # small jitter
        v = v + 0.02 * rng.standard_normal(d)
        v = v / max(np.linalg.norm(v), 1e-12)
        out.append(v)
        Xaug = np.vstack([Xaug, v[None, :]])
    return np.array(out)

def solve():
    t0 = time.time()
    LIMIT_EXPLORE = 118.0
    LIMIT_TOTAL = 168.0
    rng = np.random.default_rng(0)
    n, d = 78, 15
    bestX = None
    bestV = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (78, 15):
                continue
            a = _norm(a)
            v = _max_cos(a)
            if v < bestV:
                bestV = v
                bestX = a.copy()
        except Exception:
            continue
    if bestX is None:
        bestX = _norm(rng.standard_normal((n, d)))
        bestV = _max_cos(bestX)

    # helper to get contact counts with loose tol
    def counts_loose(X, tol=0.006):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        C = (G > m - tol).sum(axis=1)
        return C, m

    it = 0
    # staged beta/lam schedules: early isotropic, late sharp
    SCHED_FAST = [(150.0, 0.8, 90), (450.0, 0.15, 90), (900.0, 0.0, 90)]
    SCHED_FINE = [(200.0, 0.5, 120), (500.0, 0.1, 120), (1000.0, 0.0, 150)]

    while time.time() - t0 < LIMIT_EXPLORE - 2:
        it += 1
        time_left = LIMIT_EXPLORE - (time.time() - t0)
        if time_left < 4:
            break
        cnts, mcur = counts_loose(bestX, tol=0.006)
        order_lo = np.argsort(cnts, kind='stable')
        order_hi = order_lo[::-1]
        mode = it % 6
        Xtry = bestX.copy()
        if mode == 0:
            # big global shake to new basin
            sigma = 0.38
            Xtry = bestX + sigma * rng.standard_normal((n, d))
            Xtry = _norm(Xtry)
            sched = SCHED_FAST
        elif mode == 1:
            # shake only over-jammed hubs strongly
            k = 12
            idx = order_hi[:k]
            Xtry[idx] = bestX[idx] + 0.65 * rng.standard_normal((k, d))
            Xtry = _norm(Xtry)
            sched = SCHED_FAST
        elif mode == 2:
            # void reinsertion of loosest points
            k = 4 + (it % 3)  # 4..6
            idx = order_lo[:k]
            V = _find_voids(bestX, k, rng, nsamp=5000)
            Xtry[idx] = V + 0.05 * rng.standard_normal((k, d))
            Xtry = _norm(Xtry)
            sched = SCHED_FINE if (it % 12 == 2) else SCHED_FAST
        elif mode == 3:
            # intermix two PC clusters: move hubs from dense side to sparse side
            try:
                U, SV, Vt = np.linalg.svd(bestX - bestX.mean(axis=0, keepdims=True), full_matrices=False)
                v0 = Vt[0]
                proj = bestX @ v0
                med = np.median(proj)
                left = np.where(proj <= med)[0]
                right = np.where(proj > med)[0]
                # identify dense side (larger)
                if len(left) >= len(right):
                    dense, sparse = left, right
                else:
                    dense, sparse = right, left
                # sparse mean direction
                mu = bestX[sparse].mean(axis=0)
                mu = mu / max(np.linalg.norm(mu), 1e-12)
                # pick high-contact dense points
                dense_c = [(i, cnts[i]) for i in dense]
                dense_c.sort(key=lambda t: -t[1])
                k = 6
                idx = np.array([t[0] for t in dense_c[:k]], dtype=int)
                for ii in idx:
                    Xtry[ii] = mu + 0.45 * rng.standard_normal(d)
                # also jitter sparse slightly to make room
                Xtry[sparse] = bestX[sparse] + 0.08 * rng.standard_normal((len(sparse), d))
                Xtry = _norm(Xtry)
            except Exception:
                Xtry = _norm(bestX + 0.3 * rng.standard_normal((n, d)))
            sched = SCHED_FAST
        elif mode == 4:
            # moderate global shake + whitening to enforce isotropy
            sigma = 0.22
            Xtry = bestX + sigma * rng.standard_normal((n, d))
            # whiten step: force second moment toward I/d
            try:
                Y = _norm(Xtry)
                S = (Y.T @ Y) / n
                # S^{-1/2} via eigh
                w, Q = np.linalg.eigh(S)
                w = np.maximum(w, 1e-6)
                Sinvhalf = (Q * (1.0 / np.sqrt(w))) @ Q.T
                target_scale = np.sqrt(float(n) / float(d))
                # apply: Y @ Sinvhalf scaled then renormalize (partial strength 0.5)
                Yw = Y @ Sinvhalf * (np.sqrt(d) * np.sqrt(w.mean()))
                # blend
                Xtry = _norm(0.5 * Y + 0.5 * _norm(Yw))
            except Exception:
                Xtry = _norm(Xtry)
            sched = SCHED_FAST
        else:
            # small shake of random subset to polish nearby basin differently
            k = 6
            idx = rng.choice(n, size=k, replace=False)
            Xtry[idx] = bestX[idx] + 0.35 * rng.standard_normal((k, d))
            Xtry = _norm(Xtry)
            sched = SCHED_FAST

        betas_lams = [(b, l) for (b, l, _) in sched]
        # adapt maxiter to time left
        # run staged with per-stage iters
        Xnew = Xtry
        ok = True
        for (b, l, mi) in sched:
            if time.time() - t0 > LIMIT_EXPLORE - 1.5:
                ok = False
                break
            # shorten if many iters left
            Xnew = _smooth_refine(Xnew, [(b, l)], mi, t_deadline=LIMIT_EXPLORE - 0.5 + t0 - 118 + 118)
        if not ok:
            break
        try:
            vnew = _max_cos(Xnew)
        except Exception:
            continue
        if not np.isfinite(vnew):
            continue
        if vnew < bestV - 1e-12:
            bestX = Xnew
            bestV = vnew
        elif vnew < bestV + 0.002 and (it % 9 == 0):
            # occasional sideways step from a different basin if it is balanced
            # check balance: std of loose counts small and isotropy good
            try:
                C = (Xnew.T @ Xnew) / n
                iso = float(((C - np.eye(d) / d) ** 2).sum())
                Cold = (bestX.T @ bestX) / n
                iso_old = float(((Cold - np.eye(d) / d) ** 2).sum())
                if iso < iso_old and vnew < bestV + 0.0005:
                    bestX = Xnew
                    bestV = vnew
            except Exception:
                pass
        # early exit if big win
        if bestV < 0.1665:
            break

    # final snap with LP
    remain = LIMIT_TOTAL - (time.time() - t0)
    lp_time = max(4.0, min(30.0, remain - 2.0))
    if lp_time > 2:
        try:
            Xp = _lp_polish(bestX, lp_time)
            vp = _max_cos(Xp)
            if np.isfinite(vp) and vp < bestV:
                bestX = Xp
                bestV = vp
        except Exception:
            pass
    bestX = _norm(bestX)
    # tiny shrink guard against boundary ties (keep strictly feasible, no zero rows)
    return np.ascontiguousarray(bestX, dtype=np.float64)
