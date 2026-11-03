import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _tangent_bases(X):
    n, d = X.shape
    k5 = d - 1
    T = np.empty((n, d, k5))
    E = np.eye(d)
    for i in range(n):
        x = X[i]
        k = int(np.argmax(np.abs(x)))
        idx = [a for a in range(d) if a != k]
        B = np.empty((d, k5))
        for j, a in enumerate(idx):
            v = E[:, a] - x[a] * x
            B[:, j] = v
        Q, R = np.linalg.qr(B)
        # Q is 6x5 orthonormal, orthogonal to x (up to sign)
        T[i] = Q
    return T

def solve():
    t0 = time.time()
    T_END = t0 + 172.0
    n, d = 249, 6
    N = n * 5
    rng = np.random.default_rng(12345)
    np.random.seed(12345)

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                nr = np.linalg.norm(a, axis=1, keepdims=True)
                nr = np.maximum(nr, 1e-12)
                a = a / nr
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
        X0 = rng.standard_normal(size=(n, d))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        cands.append(X0)
    vals = [_max_cos(a) for a in cands]
    bestX = cands[int(np.argmin(vals))].copy()
    bestVal = float(np.min(vals))
    cur = bestX.copy()
    curVal = bestVal

    iu0, ju0 = np.triu_indices(n, 1)

    def full_soft(Xinit, taus, iters, creg=0.1):
        nn = Xinit.shape[0]
        xf = Xinit.ravel().copy()
        for tau, mi in zip(taus, iters):
            if time.time() > T_END - 30:
                break
            def fg(xflat, tau=float(tau)):
                X = xflat.reshape(nn, d)
                r = np.sqrt(np.sum(X * X, axis=1))
                r = np.maximum(r, 1e-12)
                Xn = X / r[:, None]
                G = Xn @ Xn.T
                g = G[iu0, ju0]
                gm = np.max(g)
                e = np.exp((g - gm) / tau)
                S = np.sum(e)
                Lsoft = gm + tau * np.log(S)
                w = e / S
                W = np.zeros((nn, nn))
                W[iu0, ju0] = w
                W = W + W.T
                A = W @ Xn
                proj = np.sum(A * Xn, axis=1)
                G0 = (A - proj[:, None] * Xn) / r[:, None]
                diff = r - 1.0
                L = Lsoft + creg * np.sum(diff * diff)
                Gr = G0 + (2.0 * creg * diff)[:, None] * (X / r[:, None])
                return float(L), Gr.ravel()
            try:
                res = minimize(fg, xf, jac=True, method='L-BFGS-B',
                               options={'maxiter': int(mi), 'maxfun': int(mi * 4),
                                        'ftol': 1e-14, 'gtol': 1e-7, 'maxcor': 10})
                xf = res.x
            except Exception:
                break
        Xo = xf.reshape(nn, d)
        Xo /= np.maximum(np.linalg.norm(Xo, axis=1, keepdims=True), 1e-12)
        return Xo

    def forced_step(X, T, pairs_I, pairs_J, cvals, Ssel, rad, delta, m):
        P = len(cvals)
        if P == 0:
            return None
        rows = []
        cols = []
        data = []
        for k in range(P):
            i = int(pairs_I[k]); j = int(pairs_J[k])
            try:
                ti = T[i].T @ X[j]
                tj = T[j].T @ X[i]
            except Exception:
                continue
            bi = i * 5
            bj = j * 5
            for a in range(5):
                rows.append(k); cols.append(bi + a); data.append(float(ti[a]))
                rows.append(k); cols.append(bj + a); data.append(float(tj[a]))
            rows.append(k); cols.append(N); data.append(-1.0)
        if not rows:
            return None
        A = coo_matrix((np.array(data), (np.array(rows), np.array(cols))), shape=(P, N + 1)).tocsr()
        b = -cvals.astype(float).copy()
        if len(Ssel) > 0:
            b[np.asarray(Ssel, dtype=int)] -= float(delta)
        cobj = np.zeros(N + 1)
        cobj[N] = 1.0
        lo = float(m) - 0.02
        hi = float(m) + 0.02
        bounds = [(-float(rad), float(rad))] * N + [(lo, hi)]
        try:
            res = linprog(cobj, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        except Exception:
            return None
        if res.status != 0 or res.x is None:
            return None
        u = np.asarray(res.x[:N], dtype=float)
        if not np.all(np.finite(u)):
            return None
        dX = np.zeros_like(X)
        for i in range(n):
            ui = u[i * 5:(i + 1) * 5]
            # clip huge
            dX[i] = T[i] @ ui
        Xt = X + dX
        nr = np.linalg.norm(Xt, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        Xt = Xt / nr
        if not np.all(np.finite(Xt)):
            return None
        return Xt

    def lp_jam(Xinit, deadline):
        X = Xinit.copy()
        X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        best = _max_cos(X)
        bx = X.copy()
        radius = 2e-3
        iuA, iuB = np.triu_indices(n, 1)
        T = None
        while time.time() < deadline and radius > 1e-13:
            if time.time() > T_END - 1.5:
                break
            Gf = X @ X.T
            cfull = Gf[iuA, iuB]
            m = float(cfull.max())
            sel = cfull > m - max(20.0 * radius, 3e-7)
            cnt = int(np.sum(sel))
            if cnt < 10:
                idx = np.argpartition(cfull, -10)[-10:]
                sel = np.zeros_like(sel, dtype=bool)
                sel[idx] = True
                cnt = 10
            if cnt > 4000:
                idx = np.argpartition(cfull, -4000)[-4000:]
                # keep only those that are also in sel? just take top 4000
                sel = np.zeros_like(sel, dtype=bool)
                sel[idx] = True
                cnt = 4000
            sI = iuA[sel]; sJ = juB[sel]
            cvals = cfull[sel]
            try:
                T = _tangent_bases(X)
            except Exception:
                break
            Xt = forced_step(X, T, sI, sJ, cvals, [], radius, 0.0, m)
            if Xt is None:
                radius *= 0.5
                continue
            v = _max_cos(Xt)
            if v < best - 1e-12:
                best = v
                bx = Xt.copy()
                X = Xt
                radius = min(radius * 1.3, 3e-3)
            else:
                # try smaller step from same X
                radius *= 0.5
                # keep X unchanged
                continue
        return bx, best

    rad_cycle = [0.01, 0.02, 0.035, 0.05, 0.015, 0.03]
    delta_cycle = [1e-4, 5e-4, 1e-3, 5e-4, 1e-4, 1e-3]

    it = 0
    # exploration with forced slack
    while time.time() < T_END - 55 and it < 60:
        it += 1
        try:
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            mcur = float(np.max(G))
            thr = 2e-4
            # gather contacts
            Iall, Jall = np.where(np.triu(G, 1) > mcur - thr)
            if len(Iall) < 300:
                thr = 5e-4
                Iall, Jall = np.where(np.triu(G, 1) > mcur - thr)
            if len(Iall) > 2500:
                c_all = G[Iall, Jall]
                ord_ = np.argsort(c_all)[-2500:]
                Iall = Iall[ord_]; Jall = Jall[ord_]
            P = len(Iall)
            if P < 20:
                # random shake
                trial = cur + 0.005 * rng.standard_normal(size=(n, d))
                trial /= np.linalg.norm(trial, axis=1, keepdims=True)
                trial = full_soft(trial, taus=[0.004, 0.0015], iters=[80, 100])
                v = _max_cos(trial)
                if v < bestVal:
                    bestVal = v; bestX = trial.copy()
                cur = trial; curVal = v
                continue
            cvals = G[Iall, Jall]
            counts = np.zeros(n, dtype=int)
            for k in range(P):
                counts[int(Iall[k])] += 1
                counts[int(Jall[k])] += 1
            hubs = np.argsort(counts)[::-1]
            # pick hub cycling among top 12
            hidx = (it - 1) % min(12, n)
            hub = int(hubs[hidx])
            # adjacency
            adj = np.zeros((n, n), dtype=bool)
            adj[Iall, Jall] = True
            adj[Jall, Iall] = True
            # BFS cluster target 25-35
            target = int([25, 30, 35][(it - 1) % 3])
            visited = set([hub])
            frontier = [hub]
            while len(visited) < target and len(frontier) > 0:
                nxt = []
                for u in frontier:
                    nbrs = np.where(adj[u])[0]
                    # order by tightness: larger G first
                    if len(nbrs) > 1:
                        gvs = G[u, nbrs]
                        o = np.argsort(gvs)[::-1]
                        nbrs = nbrs[o]
                    for v in nbrs:
                        vv = int(v)
                        if vv not in visited:
                            visited.add(vv)
                            nxt.append(vv)
                            if len(visited) >= target:
                                break
                    if len(visited) >= target:
                        break
                frontier = nxt
                if len(frontier) == 0:
                    break
            cset = visited
            # S = contacts incident to cluster
            Smask = np.zeros(P, dtype=bool)
            for k in range(P):
                if int(Iall[k]) in cset or int(Jall[k]) in cset:
                    Smask[k] = True
            Sidx = np.where(Smask)[0]
            if len(Sidx) < 100:
                # expand target: add tightest global contacts incident? just take more neighbours
                # add second hop by enlarging cset randomly
                extra = np.argsort(counts)[::-1][:40]
                for vv in extra:
                    cset.add(int(vv))
                Smask = np.zeros(P, dtype=bool)
                for k in range(P):
                    if int(Iall[k]) in cset or int(Jall[k]) in cset:
                        Smask[k] = True
                Sidx = np.where(Smask)[0]
            if len(Sidx) > 300:
                # keep 300 tightest within S
                cs = cvals[Sidx]
                o = np.argsort(cs)[-300:]
                Sidx = Sidx[o]
            if len(Sidx) == 0:
                continue
            rad = float(rad_cycle[(it - 1) % len(rad_cycle)])
            delta = float(delta_cycle[(it - 1) % len(delta_cycle)])
            T = _tangent_bases(cur)
            Xt = forced_step(cur, T, Iall, Jall, cvals, Sidx, rad, delta, mcur)
            if Xt is None:
                continue
            # step even if worsens; light re-polish
            Xt2 = full_soft(Xt, taus=[0.005, 0.002, 0.0008], iters=[70, 90, 110])
            v2 = _max_cos(Xt2)
            if v2 < bestVal:
                bestVal = v2
                bestX = Xt2.copy()
            # accept as cur if not too bad (allow +0.002 worsening)
            if v2 < curVal + 0.0025:
                cur = Xt2
                curVal = v2
            else:
                # reset to best with tiny shake to diversify
                sig = 0.002
                cur = bestX + sig * rng.standard_normal(size=(n, d))
                cur /= np.linalg.norm(cur, axis=1, keepdims=True)
                curVal = _max_cos(cur)
            # occasional return to best
            if it % 9 == 0:
                cur = bestX.copy()
                curVal = bestVal
        except Exception:
            continue

    # final annealing from best
    if time.time() < T_END - 30:
        try:
            t = full_soft(bestX, taus=[0.005, 0.002, 0.0009, 0.00045], iters=[150, 180, 200, 220])
            v = _max_cos(t)
            if v < bestVal:
                bestVal = v
                bestX = t
        except Exception:
            pass
    # final trust-region LP jam
    if time.time() < T_END - 3:
        try:
            deadline = T_END - 1.0
            # give at most ~25 sec
            bx, bv = lp_jam(bestX, deadline)
            if bv < bestVal:
                bestVal = bv
                bestX = bx
        except Exception:
            pass

    bestX /= np.maximum(np.linalg.norm(bestX, axis=1, keepdims=True), 1e-12)
    return bestX
