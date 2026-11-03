import time
import numpy as np
from scipy.optimize import linprog, minimize

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def solve():
    t0 = time.time()
    T_END = t0 + 172.0
    RESERVE = 26.0
    rng = np.random.default_rng(7)
    np.random.seed(7)
    n, d = 249, 6
    I0, J0 = np.triu_indices(n, 1)
    P = len(I0)

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                an = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
                if np.all(np.isfinite(an)):
                    cands.append(an)
        except Exception:
            pass
    if not cands:
        a = rng.standard_normal(size=(n, d))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    vals = [_max_cos(a) for a in cands]
    bestX = cands[int(np.argmin(vals))].copy()
    bestVal = float(np.min(vals))
    cur = bestX.copy()
    curVal = bestVal

    def tangent_bases(X):
        T = np.empty((n, 6, 5), dtype=float)
        E = np.eye(6)
        for i in range(n):
            a = X[i]
            sigma = -1.0 if a[0] >= 0 else 1.0
            v = a.copy()
            v[0] -= sigma
            nv = float(np.linalg.norm(v))
            if nv < 1e-12:
                T[i] = E[:, 1:]
            else:
                w = v / nv
                H = E - 2.0 * np.outer(w, w)
                T[i] = H[:, 1:]
        return T

    def full_soft(Xinit, taus, iters, creg=0.08):
        Xf = (Xinit / np.maximum(np.linalg.norm(Xinit, axis=1, keepdims=True), 1e-12)).ravel().copy()
        for tau, mi in zip(taus, iters):
            if time.time() > T_END - RESERVE:
                break
            tau = float(tau)
            def fg(xf, tau=tau):
                X = xf.reshape(n, d)
                r = np.sqrt(np.sum(X * X, axis=1))
                r = np.maximum(r, 1e-12)
                Xn = X / r[:, None]
                G = Xn @ Xn.T
                g = G[I0, J0]
                gm = np.max(g)
                e = np.exp((g - gm) / tau)
                S = np.sum(e)
                Lsoft = gm + tau * np.log(S)
                w = e / S
                W = np.zeros((n, n))
                W[I0, J0] = w
                W = W + W.T
                A = W @ Xn
                proj = np.sum(A * Xn, axis=1)
                G0 = (A - proj[:, None] * Xn) / r[:, None]
                diff = r - 1.0
                L = Lsoft + creg * np.sum(diff * diff)
                Gr = G0 + (2.0 * creg * diff)[:, None] * Xn
                return float(L), Gr.ravel()
            try:
                res = minimize(fg, Xf, jac=True, method='L-BFGS-B',
                               options={'maxiter': int(mi), 'maxfun': int(mi * 3),
                                        'ftol': 1e-14, 'gtol': 1e-7, 'maxcor': 8})
                Xf = res.x
            except Exception:
                break
            try:
                Xm = Xf.reshape(n, d)
                rn = np.maximum(np.linalg.norm(Xm, axis=1, keepdims=True), 1e-12)
                Xf = (Xm / rn).ravel()
            except Exception:
                break
        Xo = Xf.reshape(n, d)
        Xo /= np.maximum(np.linalg.norm(Xo, axis=1, keepdims=True), 1e-12)
        return Xo

    def forced_tangent_step(X, target_contacts=180, delta=5e-4, radius=0.02, hub=None):
        # build contact list with slack < 2e-4
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        g = G[I0, J0]
        sel = np.where(g > m - 2e-4)[0]
        if len(sel) < 20:
            sel = np.argsort(g)[-400:]
        Ic = I0[sel]; Jc = J0[sel]
        cc = g[sel]
        K = len(sel)
        # degrees = overloaded hubs
        cnt = np.zeros(n, dtype=int)
        np.add.at(cnt, Ic, 1)
        np.add.at(cnt, Jc, 1)
        rowmax = np.full(n, -2.0)
        # rowmax from full G
        G2 = G
        rowmax = np.max(G2, axis=1)
        if hub is None:
            order = np.lexsort((rowmax, cnt))[::-1]
            hub = int(order[0])
        # BFS cluster on contact graph to get ~target_contacts incident pairs
        adj = [[] for _ in range(n)]
        for a, b in zip(Ic, Jc):
            adj[a].append(b)
            adj[b].append(a)
        visited = np.zeros(n, dtype=bool)
        visited[hub] = True
        q = [hub]
        qi = 0
        while qi < len(q):
            # count incident contacts of current cluster
            # expand until target reached
            cl = np.where(visited)[0]
            # fast count: sum cnt of visited minus internal double? approx incident
            inc = 0
            for v in cl:
                inc += cnt[v]
            # each internal edge counted twice; incident = inc - internal? use inc/1 approx
            # we want number of pairs with at least one end in cluster
            # estimate quickly every few steps
            if qi % 5 == 0:
                # exact count
                in_cl = visited[Ic] | visited[Jc]
                if int(np.sum(in_cl)) >= int(target_contacts):
                    break
            if len(cl) > 45:
                break
            u = q[qi]; qi += 1
            for w_ in adj[u]:
                if not visited[w_]:
                    visited[w_] = True
                    q.append(w_)
            if len(q) > 60:
                break
        in_cl = visited[Ic] | visited[Jc]
        # ensure S size 100-300
        S_idx = np.where(in_cl)[0]
        if len(S_idx) < 80:
            # take top contacts by hub proximity: just take all
            S_idx = np.arange(K)
            if len(S_idx) > 300:
                # keep 250 with largest c among incident + random
                ord2 = np.argsort(cc[S_idx])[::-1]
                S_idx = S_idx[ord2[:250]]
        elif len(S_idx) > 320:
            ord2 = np.argsort(cc[S_idx])[::-1]
            S_idx = S_idx[ord2[:300]]
        S_mask = np.zeros(K, dtype=bool)
        S_mask[S_idx] = True
        # tangent bases
        T = tangent_bases(X)
        Nv = n * 5
        # Build A_ub dense: K x (Nv+1)
        try:
            Aub = np.zeros((K, Nv + 1), dtype=float)
            bub = np.zeros(K, dtype=float)
            for p in range(K):
                i = int(Ic[p]); j = int(Jc[p])
                ai = T[i].T @ X[j]  # 5
                aj = T[j].T @ X[i]
                Aub[p, i * 5:(i + 1) * 5] = ai
                Aub[p, j * 5:(j + 1) * 5] = aj
                Aub[p, Nv] = -1.0
                if S_mask[p]:
                    bub[p] = -float(cc[p]) - float(delta)
                else:
                    bub[p] = -float(cc[p])
            cvec = np.zeros(Nv + 1)
            cvec[Nv] = 1.0
            bnds = [(-float(radius), float(radius))] * Nv + [(m - 0.05, m + 0.06)]
            res = linprog(cvec, A_ub=Aub, b_ub=bub, bounds=bnds, method='highs',
                          options={'presolve': True})
            if not res.success or res.x is None:
                return None, None
            y = res.x[:Nv]
            V = np.zeros((n, 6))
            for i in range(n):
                V[i] = T[i] @ y[i * 5:(i + 1) * 5]
            Xt = X + V
            Xt /= np.maximum(np.linalg.norm(Xt, axis=1, keepdims=True), 1e-12)
            return Xt, float(res.x[Nv])
        except Exception:
            return None, None

    def lp_jam(Xinit, deadline, rounds=5):
        X = Xinit.copy()
        X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        bv = _max_cos(X)
        bx = X.copy()
        radius = 8e-4
        for _ in range(rounds):
            if time.time() > deadline:
                break
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            g = G[I0, J0]
            sel = np.where(g > m - max(20 * radius, 4e-7))[0]
            if len(sel) < 10:
                sel = np.argsort(g)[-300:]
            Ic = I0[sel]; Jc = J0[sel]; cc = g[sel]
            K = len(sel)
            T = tangent_bases(X)
            Nv = n * 5
            try:
                Aub = np.zeros((K, Nv + 1))
                bub = np.zeros(K)
                for p in range(K):
                    i = int(Ic[p]); j = int(Jc[p])
                    Aub[p, i * 5:(i + 1) * 5] = T[i].T @ X[j]
                    Aub[p, j * 5:(j + 1) * 5] = T[j].T @ X[i]
                    Aub[p, Nv] = -1.0
                    bub[p] = -float(cc[p])
                cvec = np.zeros(Nv + 1); cvec[Nv] = 1.0
                bnds = [(-radius, radius)] * Nv + [(m - 0.02, m + 0.02)]
                res = linprog(cvec, A_ub=Aub, b_ub=bub, bounds=bnds, method='highs')
                if not res.success:
                    radius *= 0.4
                    continue
                y = res.x[:Nv]
                V = np.zeros((n, 6))
                for i in range(n):
                    V[i] = T[i] @ y[i * 5:(i + 1) * 5]
                Xt = X + V
                Xt /= np.maximum(np.linalg.norm(Xt, axis=1, keepdims=True), 1e-12)
                v = _max_cos(Xt)
                if v < bv:
                    bv = v
                    bx = Xt.copy()
                    X = Xt
                    radius = min(radius * 1.5, 2e-3)
                else:
                    # accept if very close? else shrink
                    radius *= 0.35
                    if v < bv + 1e-12:
                        X = Xt
                if radius < 1e-12:
                    break
            except Exception:
                break
        return bx, bv

    deltas = [1e-4, 5e-4, 1e-3]
    radii = [0.01, 0.02, 0.04]
    targets = [120, 180, 250]

    attempt = 0
    # initial hub order from best
    while time.time() < T_END - RESERVE:
        attempt += 1
        try:
            Gc = cur @ cur.T
            np.fill_diagonal(Gc, -2.0)
            mc = float(np.max(Gc))
            cntc = np.sum(Gc > mc - 2e-4, axis=1)
            rowm = np.max(Gc, axis=1)
            order = np.lexsort((rowm, cntc))[::-1]
            nh = min(8, n)
            hub = int(order[(attempt - 1) % nh])
            delta = deltas[(attempt - 1) % 3]
            radius = radii[((attempt - 1) // 3) % 3]
            target = targets[((attempt - 1) // 2) % 3]
            Xt, _ = forced_tangent_step(cur, target_contacts=target, delta=delta, radius=radius, hub=hub)
            if Xt is None:
                # fallback small shake
                Xt = cur + 0.004 * rng.standard_normal(size=(n, d))
                Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            # step even if worsens: quick re-polish without forced constraints
            Xp = full_soft(Xt, taus=[0.005, 0.0015], iters=[55, 75], creg=0.08)
            vp = _max_cos(Xp)
            if vp < bestVal:
                bestVal = vp
                bestX = Xp.copy()
            # non-monotone acceptance for cur to create floppy pocket
            if vp < curVal + 0.0035 and vp < 0.73:
                cur = Xp.copy()
                curVal = vp
            else:
                # keep exploring from best with shake if cur too bad
                if curVal > bestVal + 0.006:
                    cur = bestX + 0.003 * rng.standard_normal(size=(n, d))
                    cur /= np.linalg.norm(cur, axis=1, keepdims=True)
                    curVal = _max_cos(cur)
            if attempt > 60:
                break
        except Exception:
            continue

    # final intensive polish of best
    try:
        if time.time() < T_END - 20:
            t = full_soft(bestX, taus=[0.004, 0.0015, 0.0006], iters=[140, 170, 200])
            v = _max_cos(t)
            if v < bestVal:
                bestVal = v
                bestX = t
    except Exception:
        pass
    try:
        if time.time() < T_END - 8:
            bx, bv = lp_jam(bestX, T_END - 2.0, rounds=7)
            if bv < bestVal:
                bestVal = bv
                bestX = bx
    except Exception:
        pass
    bestX /= np.maximum(np.linalg.norm(bestX, axis=1, keepdims=True), 1e-12)
    return bestX
