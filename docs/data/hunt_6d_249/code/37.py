import time
import numpy as np
from scipy.optimize import minimize, linprog

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def solve():
    t0 = time.time()
    T_END = t0 + 172.0
    RESERVE = 24.0
    rng = np.random.default_rng(7)
    np.random.seed(7)
    n, d = 249, 6
    p = d - 1
    N = n * p
    I0, J0 = np.triu_indices(n, 1)

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

    Eye = np.eye(d)

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

    def tangent_bases(X):
        T = np.empty((n, d, p))
        for i in range(n):
            x = X[i]
            k = int(np.argmax(np.abs(x)))
            cols = [j for j in range(d) if j != k]
            V = np.empty((d, p))
            for c, j in enumerate(cols):
                e = Eye[:, j]
                V[:, c] = e - x[j] * x
            Q, _ = np.linalg.qr(V)
            T[i] = Q
        return T

    def top_pairs(X, K=2500, big_win=0.025):
        G = X @ X.T
        c = G[I0, J0]
        m = float(np.max(c))
        # candidates within big window
        mask = c > (m - big_win)
        idx = np.where(mask)[0]
        if idx.size > K:
            # keep tightest K
            part = np.argpartition(c[idx], -K)[-K:]
            idx = idx[part]
        # sort by tightness descending for determinism
        order = np.argsort(c[idx])[::-1]
        idx = idx[order]
        pairs = np.stack([I0[idx], J0[idx]], axis=1)
        cvals = c[idx]
        return pairs, cvals, float(m), G

    def degrees_and_hubs(X, G, m, win=2e-4):
        thr = m - win
        # binary adjacency for tight graph
        tight = G > thr
        np.fill_diagonal(tight, False)
        deg = tight.sum(axis=1).astype(int)
        rowmax = G.max(axis=1)
        # rowmax with diag excluded
        return deg, rowmax, tight

    def bfs_cluster(tight, hub, target_nodes):
        # BFS expansion from hub
        seen = np.zeros(n, dtype=bool)
        seen[hub] = True
        queue = [hub]
        order = [hub]
        qpos = 0
        nbrs_all = [np.where(tight[i])[0] for i in range(n)]
        # if tight graph too sparse, fall back to dense nearest neighbours via G
        while len(order) < target_nodes and qpos < len(queue):
            u = queue[qpos]; qpos += 1
            for v in nbrs_all[u]:
                if not seen[v]:
                    seen[v] = True
                    queue.append(v)
                    order.append(v)
                    if len(order) >= target_nodes:
                        break
        return np.array(order, dtype=int), seen

    deltas = [1e-4, 5e-4, 1e-3, 5e-4, 1e-4, 1e-3]
    radii = [0.01, 0.02, 0.05, 0.015, 0.03, 0.01]
    target_list = [15, 22, 30, 18, 26, 12]
    Klist = [2200, 2500, 2000]

    it = 0
    # precompute once SVD diagnostic (not branching, just for hub weighting)
    while time.time() < T_END - RESERVE:
        it += 1
        try:
            Xc = cur.copy()
            pairs, cvals, mcur, Gc = top_pairs(Xc, K=Klist[it % len(Klist)], big_win=0.025)
            deg, rowmax, tight = degrees_and_hubs(Xc, Gc, mcur, win=2e-4)
            # choose hub: alternate max-degree vs max-rowmax
            if it % 2 == 1:
                hub = int(np.argmax(deg * 1000.0 + rowmax))
            else:
                # overloaded hub: max deg, tie break rowmax
                hub = int(np.argmax(rowmax + 1e-6 * deg))
                # if all deg zero (very loose at 2e-4), use wider window
                if np.max(deg) == 0:
                    deg2, _, tight2 = degrees_and_hubs(Xc, Gc, mcur, win=1e-3)[:3]
                    hub = int(np.argmax(deg2))
                    tight = tight2
                    deg = deg2
            target_nodes = int(target_list[it % len(target_list)])
            # expand cluster; if tight graph sparse, use G nearest neighbours to fill
            cluster_nodes, seen = bfs_cluster(tight, hub, target_nodes)
            if len(cluster_nodes) < target_nodes:
                # fill with nearest neighbours of hub by dot
                row = Gc[hub].copy()
                row[hub] = -2
                nn = np.argsort(row)[::-1]
                for v in nn:
                    if not seen[v]:
                        seen[v] = True
                        cluster_nodes = np.append(cluster_nodes, v)
                    if len(cluster_nodes) >= target_nodes:
                        break
                cluster_nodes = cluster_nodes[:target_nodes]
            in_cl = np.zeros(n, dtype=bool)
            in_cl[cluster_nodes] = True
            S_mask = in_cl[pairs[:, 0]] | in_cl[pairs[:, 1]]
            cntS = int(np.sum(S_mask))
            # adjust: want 100-300; if far outside, tweak by expanding/shrinking using big pairs count
            # if too small, include more nodes (up to 40); if too large, keep as is (still forces pocket)
            if cntS < 60:
                # expand to 40 nodes
                extra_need = 40 - len(cluster_nodes)
                row = Gc[hub].copy()
                row[hub] = -2
                nn = np.argsort(row)[::-1]
                for v in nn:
                    if not in_cl[v]:
                        in_cl[v] = True
                        cluster_nodes = np.append(cluster_nodes, v)
                        extra_need -= 1
                    if extra_need <= 0:
                        break
                S_mask = in_cl[pairs[:, 0]] | in_cl[pairs[:, 1]]
                cntS = int(np.sum(S_mask))
            delta = float(deltas[it % len(deltas)])
            radius = float(radii[it % len(radii)])
            # Build tangent bases
            T = tangent_bases(Xc)
            mm = pairs.shape[0]
            # Build LP: vars N+1
            # coeff vectors
            # To save time cap mm to 2600 already
            A_ub = np.zeros((mm, N + 1), dtype=float)
            b_ub = np.zeros(mm, dtype=float)
            for k in range(mm):
                i = int(pairs[k, 0]); j = int(pairs[k, 1])
                ci = Gc[i, j]  # same as cvals? cvals sorted; use Gc for safety
                # coeffs
                # x_j^T T_i , x_i^T T_j
                ai = Xc[j] @ T[i]  # (5,)
                aj = Xc[i] @ T[j]
                A_ub[k, i * p:(i + 1) * p] = ai
                A_ub[k, j * p:(j + 1) * p] = aj
                A_ub[k, N] = -1.0
                b_ub[k] = -ci - (delta if S_mask[k] else 0.0)
            c_obj = np.zeros(N + 1)
            c_obj[N] = 1.0
            bounds = [(-radius, radius)] * N + [(None, None)]
            try:
                res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            except Exception:
                # fallback: random shake
                Xt = Xc + 0.004 * rng.standard_normal(size=(n, d))
                Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
                Xp = full_soft(Xt, taus=[0.006, 0.002], iters=[80, 100])
                vp = _max_cos(Xp)
                if vp < bestVal:
                    bestVal = vp; bestX = Xp.copy()
                    cur = Xp.copy(); curVal = vp
                continue
            if res.status != 0 or res.x is None:
                continue
            y = res.x[:N]
            Y = y.reshape(n, p)
            # D_i = T_i @ y_i
            D = np.einsum('ndp,np->nd', T, Y)
            # trust clip (already boxed, but norm check)
            Xtrial = Xc + D
            nrm = np.linalg.norm(Xtrial, axis=1, keepdims=True)
            nrm = np.maximum(nrm, 1e-12)
            Xtrial /= nrm
            # step even if worsens; re-polish without forced constraints
            # light polish first on cluster+neighbours to create floppy pocket, then global light
            # Use full_soft short
            if it % 3 == 0:
                Xp = full_soft(Xtrial, taus=[0.005, 0.002, 0.0009], iters=[70, 90, 110])
            else:
                Xp = full_soft(Xtrial, taus=[0.005, 0.0015], iters=[80, 120])
            vp = _max_cos(Xp)
            if vp < bestVal - 1e-12:
                bestVal = vp
                bestX = Xp.copy()
                cur = Xp.copy()
                curVal = vp
            else:
                # non-monotone move: accept worse with prob to change basin
                # allow drift if not too bad (within 3e-3) or periodically
                tol = 0.003 if (it % 5 == 0) else 0.0008
                if vp < curVal + tol and rng.random() < 0.45:
                    cur = Xp.copy()
                    curVal = vp
                elif it % 7 == 0:
                    # reset to best with small shake to diversify hub
                    cur = bestX + 0.002 * rng.standard_normal(size=(n, d))
                    cur /= np.linalg.norm(cur, axis=1, keepdims=True)
                    curVal = _max_cos(cur)
                # else keep cur
            if it > 60:
                # avoid infinite loop; break to final jam if time low
                if time.time() > T_END - RESERVE - 30:
                    break
        except Exception:
            continue
        # safety: if best improved a lot, keep going
        if time.time() > T_END - RESERVE:
            break

    # final intensive jam from best
    try:
        if time.time() < T_END - 18:
            t = full_soft(bestX, taus=[0.004, 0.0018, 0.0008, 0.0004], iters=[180, 220, 260, 300])
            v = _max_cos(t)
            if v < bestVal:
                bestVal = v
                bestX = t
    except Exception:
        pass
    # greedy descent fallback polish if time remains
    try:
        if time.time() < T_END - 6:
            cur2 = bestX.copy()
            curV = _max_cos(cur2)
            step = 2e-3
            deadline = min(T_END - 1.0, time.time() + 8.0)
            while time.time() < deadline and step > 1e-12:
                G = cur2 @ cur2.T
                np.fill_diagonal(G, -2.0)
                m = float(G.max())
                window = max(10.0 * step, 2e-4)
                mask = G > (m - window)
                F = mask.astype(float) @ cur2
                dot = np.sum(F * cur2, axis=1)
                Ft = F - dot[:, None] * cur2
                trial = cur2 - step * Ft
                trial /= np.maximum(np.linalg.norm(trial, axis=1, keepdims=True), 1e-12)
                vv = _max_cos(trial)
                if vv < curV:
                    cur2 = trial; curV = vv
                    step = min(step * 1.3, 0.01)
                    if curV < bestVal:
                        bestVal = curV; bestX = cur2.copy()
                else:
                    step *= 0.5
    except Exception:
        pass
    bestX /= np.maximum(np.linalg.norm(bestX, axis=1, keepdims=True), 1e-12)
    return bestX
