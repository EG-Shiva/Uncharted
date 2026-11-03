import time
import numpy as np
from scipy.optimize import minimize, linprog
from collections import deque

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def solve():
    t0 = time.time()
    BUDGET = 172.0
    T_DIV = t0 + BUDGET - 25.0
    T_END = t0 + 178.0
    rng = np.random.default_rng(7)
    n, d = 249, 6
    I0, J0 = np.triu_indices(n, 1)
    P = len(I0)
    # pair index lookup: need fast mask, use dict? instead compute mask via vector ops
    # load
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                nr = np.linalg.norm(a, axis=1, keepdims=True)
                nr[nr == 0] = 1.0
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

    archive = [(bestVal, bestX.copy())]

    DELTA_EDGE = 2e-4
    DELTA_HUB = 5e-4

    def topk_set(X, K=1500):
        G = X @ X.T
        g = G[I0, J0]
        if K >= P:
            idx = np.arange(P)
        else:
            idx = np.argpartition(g, -K)[-K:]
        codes = I0[idx].astype(np.int64) * n + J0[idx].astype(np.int64)
        return set(codes.tolist())

    base_set = topk_set(bestX, 1500)

    def jacc_to_base(X):
        s = topk_set(X, 1500)
        inter = len(s.intersection(base_set))
        union = len(s.union(base_set))
        if union == 0:
            return 1.0
        return inter / union

    def hub_degrees(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        mask = G > (m - DELTA_HUB)
        deg = np.sum(mask, axis=1)
        return deg, m, mask, G

    def bfs_neigh(adj, start, m):
        seen = set([int(start)])
        q = deque([int(start)])
        order = [int(start)]
        while q and len(order) < m:
            u = q.popleft()
            nbrs = np.where(adj[u])[0]
            perm = rng.permutation(len(nbrs))
            for k in perm:
                v = int(nbrs[k])
                if v not in seen:
                    seen.add(v)
                    q.append(v)
                    order.append(v)
                    if len(order) >= m:
                        break
        return np.array(order[:m], dtype=int)

    def pick_del(X, m):
        deg, mx, adj, G = hub_degrees(X)
        hub = int(np.argmax(deg))
        if rng.random() < 0.5:
            G2 = G.copy()
            adj2 = G2 > (mx - 1e-3)
            # ensure connectivity: if isolated, fall back to cap
            try:
                free = bfs_neigh(adj2, hub, m)
            except Exception:
                free = np.array([hub])
            if len(free) < m:
                dots = X @ X[hub]
                free2 = np.argsort(-dots)[:m]
                free = np.union1d(free, free2)[:m]
                if len(free) < m:
                    free = np.argsort(-(X @ np.mean(X[free], axis=0)))[:m]
            if len(free) > m:
                free = free[:m]
            if hub not in free:
                free[0] = hub
            return np.asarray(free, dtype=int)
        else:
            if rng.random() < 0.6:
                u = X[hub] + 0.35 * rng.standard_normal(size=(d,))
            else:
                if rng.random() < 0.5:
                    u = X[int(rng.integers(0, n))] + 0.35 * rng.standard_normal(size=(d,))
                else:
                    u = rng.standard_normal(size=(d,))
                # ok
            nu = np.linalg.norm(u)
            if nu < 1e-9:
                u = rng.standard_normal(size=(d,))
                nu = np.linalg.norm(u)
            u /= nu
            dots = X @ u
            free = np.argsort(-dots)[:m]
            return np.asarray(free, dtype=int)

    def opt_hole_vec(Xref, init, tau=0.008, steps=180, lr=0.15):
        y = init / (np.linalg.norm(init) + 1e-12)
        for _ in range(steps):
            dd = Xref @ y
            gm = float(np.max(dd))
            e = np.exp((dd - gm) / tau)
            s = np.sum(e)
            w = e / s
            g = w @ Xref
            # project
            gp = g - float(g @ y) * y
            nrm = np.linalg.norm(gp)
            if nrm < 1e-12:
                break
            y = y - lr * gp
            y /= (np.linalg.norm(y) + 1e-12)
            # decay lr slightly
            # lr *= 0.999
        return y

    def full_anneal(Xinit, taus, iters, taboo_mask=None, taboo_pen=0.0, creg=0.08):
        Xf = Xinit.ravel().copy()
        N = Xinit.shape[0]
        i0, j0 = np.triu_indices(N, 1)
        Pp = len(i0)
        boost = None
        if taboo_mask is not None and taboo_pen > 0:
            boost = np.ones(Pp, dtype=float)
            boost[taboo_mask] = 1.0 + taboo_pen
        for stage, (tau, mi) in enumerate(zip(taus, iters)):
            if time.time() > T_DIV:
                break
            tau = float(tau)
            # only apply boost on first stage
            use_boost = boost if (stage == 0) else None
            def fg(xf, tau=tau, use_boost=use_boost):
                X = xf.reshape(N, d)
                r = np.sqrt(np.sum(X * X, axis=1))
                r = np.maximum(r, 1e-12)
                Xn = X / r[:, None]
                G = Xn @ Xn.T
                g = G[i0, j0]
                gm = np.max(g)
                e = np.exp((g - gm) / tau)
                if use_boost is not None:
                    e = e * use_boost
                S = np.sum(e)
                Lsoft = gm + tau * np.log(S)
                w = e / S
                W = np.zeros((N, N), dtype=float)
                W[i0, j0] = w
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
                               options={'maxiter': int(mi), 'maxfun': int(mi * 4),
                                        'ftol': 1e-14, 'gtol': 1e-7, 'maxcor': 8})
                Xf = res.x
            except Exception:
                break
            Xm = Xf.reshape(N, d)
            rn = np.linalg.norm(Xm, axis=1, keepdims=True)
            rn = np.maximum(rn, 1e-12)
            Xf = (Xm / rn).ravel()
            # drop boost after first stage (handled by use_boost)
        Xo = Xf.reshape(N, d)
        Xo /= np.maximum(np.linalg.norm(Xo, axis=1, keepdims=True), 1e-12)
        return Xo

    def greedy_refill_with_holes(Xbase_keep, k, pool_size=60000):
        # pool
        C = pool_size
        pool = rng.standard_normal(size=(C, d))
        pool /= np.maximum(np.linalg.norm(pool, axis=1, keepdims=True), 1e-12)
        # add hole-optimized candidates (append to pool)
        # find 4 largest holes from pool then optimize
        try:
            D0 = pool @ Xbase_keep.T
            cm0 = D0.max(axis=1)
            # take 4 best distinct
            idx4 = np.argsort(cm0)[:4]
            holes = []
            for bi in idx4:
                h = opt_hole_vec(Xbase_keep, pool[bi], tau=0.008, steps=150, lr=0.12)
                holes.append(h)
            if holes:
                holes = np.stack(holes, axis=0)
                pool = np.vstack([pool, holes])
                C = pool.shape[0]
        except Exception:
            pass
        D = pool @ Xbase_keep.T
        cur = D.max(axis=1)
        used = np.zeros(C, dtype=bool)
        picked = []
        for _ in range(k):
            cur[used] = np.inf
            bi = int(np.argmin(cur))
            used[bi] = True
            pt = pool[bi].copy()
            picked.append(pt)
            dd = pool @ pt
            # update
            # np.maximum with out
            np.maximum(cur, dd, out=cur)
            cur[used] = np.inf
        # local polish of 2 worst among picked w.r.t keep+others (single-vector)
        try:
            for _ in range(1):
                Xk = Xbase_keep
                # build full temp
                Pmat = np.stack(picked, axis=0)
                # find worst
                All = np.vstack([Xk, Pmat])
                worst_s = -1
                worst_v = -2
                for s in range(k):
                    idx_all = Xk.shape[0] + s
                    dots = All @ Pmat[s]
                    dots[idx_all] = -2
                    mv = float(np.max(dots))
                    if mv > worst_v:
                        worst_v = mv
                        worst_s = s
                if worst_s >= 0:
                    # optimize worst vs rest (leave-one-out)
                    mask = np.ones(All.shape[0], dtype=bool)
                    mask[Xk.shape[0] + worst_s] = False
                    Xref = All[mask]
                    # init from current
                    newpt = opt_hole_vec(Xref, Pmat[worst_s], tau=0.005, steps=200, lr=0.1)
                    picked[worst_s] = newpt
        except Exception:
            pass
        return np.stack(picked, axis=0)

    # ---------- avalanche loop ----------
    n_cycles = 0
    # allow true maxima up to incumbent+0.005..0.015: we handle by accepting taboo stage without rejection
    while time.time() < T_DIV:
        n_cycles += 1
        # choose base: mostly best, sometimes archive diverse
        if len(archive) > 1 and rng.random() < 0.25:
            # pick random archive entry (not necessarily best) to diversify
            ai = int(rng.integers(0, len(archive)))
            cur_base = archive[ai][1]
        else:
            cur_base = bestX
        m = int(rng.integers(18, 36))  # 18..35
        try:
            del_idx = pick_del(cur_base, m)
        except Exception:
            continue
        # taboo mask for old contacts touching del set
        try:
            G0 = cur_base @ cur_base.T
            np.fill_diagonal(G0, -2.0)
            mx0 = float(np.max(G0))
            gvec = G0[I0, J0]
            thr = mx0 - 1e-3
            is_del = np.zeros(n, dtype=bool)
            is_del[del_idx] = True
            taboo_mask = (gvec > thr) & (is_del[I0] | is_del[J0])
            # if too few, expand threshold
            if np.sum(taboo_mask) < 10:
                thr2 = mx0 - 3e-3
                taboo_mask = (gvec > thr2) & (is_del[I0] | is_del[J0])
            taboo_pen = float(rng.uniform(0.5, 2.0))
        except Exception:
            taboo_mask = None
            taboo_pen = 0.0
        # keep
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[del_idx] = False
        Xkeep = cur_base[keep_mask]
        # refill
        try:
            new_pts = greedy_refill_with_holes(Xkeep, m, pool_size=60000)
        except Exception:
            continue
        Xnew = np.empty_like(cur_base)
        Xnew[keep_mask] = Xkeep
        Xnew[~keep_mask] = new_pts
        # normalize
        Xnew /= np.maximum(np.linalg.norm(Xnew, axis=1, keepdims=True), 1e-12)
        # re-jam with softmax annealing, taboo on first stage
        try:
            # short re-jam: allow worse intermediate; we just anneal
            Y = full_anneal(Xnew, [0.02, 0.008, 0.003, 0.001], [35, 40, 45, 55],
                            taboo_mask=taboo_mask, taboo_pen=taboo_pen, creg=0.08)
        except Exception:
            continue
        try:
            v = _max_cos(Y)
        except Exception:
            continue
        # jaccard
        try:
            j = jacc_to_base(Y)
        except Exception:
            j = 0.0
        # retain logic
        improved = v < bestVal
        # non-monotone retain: if diverse even when worse by up to 2e-4
        diverse = j < 0.70
        if improved:
            bestVal = v
            bestX = Y.copy()
            base_set = topk_set(bestX, 1500)
            archive.append((v, Y.copy()))
            # keep archive sorted, trim to 8
            archive.sort(key=lambda t: t[0])
            archive = archive[:8]
        elif diverse and v < bestVal + 2e-4:
            # retain for LP polishing
            # avoid duplicates: check if already close
            archive.append((v, Y.copy()))
            archive.sort(key=lambda t: t[0])
            # keep at most 8, but ensure diversity: keep best + diverse
            if len(archive) > 8:
                archive = archive[:8]
            # do not update base_set (incumbent unchanged)
        else:
            # small chance to keep if very close to best even without diversity? to allow walk
            if v < bestVal + 5e-06 and len(archive) < 8:
                archive.append((v, Y.copy()))
                archive.sort(key=lambda t: t[0])
        # early exit if time
        if time.time() >= T_DIV:
            break

    # ---------- LP polish ----------
    def lp_polish(Xinit, time_deadline):
        X = Xinit.copy()
        X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        cur = _max_cos(X)
        N = X.shape[0]
        dim = N * d
        eps = 0.008
        it = 0
        while time.time() < time_deadline - 1.0 and it < 40:
            it += 1
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            gvec = G[I0, J0]
            # select near-max pairs: within 2e-3 or top 2500
            K = 2500
            if P > K:
                # threshold for top K
                part = np.argpartition(gvec, -K)[-K:]
                thr = float(np.min(gvec[part]))
                thr = max(thr, m - 0.003)
                sel = np.where(gvec >= thr)[0]
                # cap at 3000
                if len(sel) > 3000:
                    sel = part[np.argsort(-gvec[part])[:3000]]
            else:
                sel = np.arange(P)
            sI = I0[sel]
            sJ = J0[sel]
            nc = len(sel)
            # Build sparse A_ub: nc x (dim+1)
            # Use dense if small else sparse via lil? Use scipy sparse coo
            try:
                from scipy.sparse import coo_matrix
                rows = []
                cols = []
                data = []
                b = np.empty(nc, dtype=float)
                for r_, ii, jj in zip(range(nc), sI, sJ):
                    xi = X[ii]
                    xj = X[jj]
                    # dxi coeffs = xj, dxj coeffs = xi, t coeff = -1
                    for k in range(d):
                        rows.append(r_); cols.append(ii * d + k); data.append(float(xj[k]))
                        rows.append(r_); cols.append(jj * d + k); data.append(float(xi[k]))
                    rows.append(r_); cols.append(dim); data.append(-1.0)
                    b[r_] = -float(xi @ xj)
                A_ub = coo_matrix((data, (rows, cols)), shape=(nc, dim + 1)).tocsr()
                # equality: xi . dxi = 0
                rows2 = []
                cols2 = []
                data2 = []
                for i in range(N):
                    xi = X[i]
                    for k in range(d):
                        rows2.append(i); cols2.append(i * d + k); data2.append(float(xi[k]))
                A_eq = coo_matrix((data2, (rows2, cols2)), shape=(N, dim + 1)).tocsr()
                beq = np.zeros(N, dtype=float)
                c = np.zeros(dim + 1, dtype=float)
                c[dim] = 1.0
                bounds = [(-eps, eps)] * dim + [(m - 0.02, m + 0.005)]
                res = linprog(c, A_ub=A_ub, b_ub=b, A_eq=A_eq, b_eq=beq,
                              bounds=bounds, method='highs',
                              options={'time_limit': 5.0, 'presolve': True})
                if not res.success:
                    eps *= 0.5
                    if eps < 1e-5:
                        break
                    continue
                dx = res.x[:dim].reshape(N, d)
                # step: trust region already bounds, take full step
                Xn = X + dx
                Xn /= np.maximum(np.linalg.norm(Xn, axis=1, keepdims=True), 1e-12)
                vn = _max_cos(Xn)
                if vn < cur - 1e-12:
                    X = Xn
                    cur = vn
                    # keep eps or slightly increase
                    eps = min(0.01, eps * 1.2)
                else:
                    eps *= 0.5
                    if eps < 1e-6:
                        break
            except Exception:
                break
        return X, cur

    # reserve final 25s for LP polishing best archived
    try:
        archive.sort(key=lambda t: t[0])
        # polish top up to 2 candidates if time
        deadline = T_END - 2.0
        # start with best
        cand0 = archive[0][1].copy()
        # LP polish best
        Xlp, vlp = lp_polish(cand0, deadline - 6.0)
        # sharp softmax after LP
        try:
            Xsharp = full_anneal(Xlp, [0.005, 0.0015, 0.0005], [80, 80, 80], taboo_mask=None, taboo_pen=0.0, creg=0.05)
            # full_anneal checks T_DIV; bypass by manual anneal with T_END
            # if it returned early due to T_DIV, do manual small polish
            vsharp = _max_cos(Xsharp)
        except Exception:
            Xsharp, vsharp = Xlp, vlp
        if vsharp < bestVal:
            bestVal = vsharp
            bestX = Xsharp.copy()
        elif vlp < bestVal:
            bestVal = vlp
            bestX = Xlp.copy()
        # if time left, try second archive entry
        if time.time() < deadline - 8.0 and len(archive) > 1:
            cand1 = archive[1][1].copy()
            # only if diverse
            Xlp2, vlp2 = lp_polish(cand1, deadline - 2.0)
            if vlp2 < bestVal:
                bestVal = vlp2
                bestX = Xlp2.copy()
    except Exception:
        pass

    # final normalize + tiny shrink not needed (already unit)
    bestX /= np.maximum(np.linalg.norm(bestX, axis=1, keepdims=True), 1e-12)
    # ensure finite
    bestX = np.where(np.isfinite(bestX), bestX, 0.0)
    # renormalize after fix
    nr = np.linalg.norm(bestX, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    bestX = bestX / nr
    return bestX
