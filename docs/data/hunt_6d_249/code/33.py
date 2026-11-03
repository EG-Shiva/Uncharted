import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix
from collections import deque

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def solve():
    t0 = time.time()
    BUDGET = 172.0
    T_DIVERSE_END = t0 + BUDGET - 25.0
    T_END = t0 + 178.0
    rng = np.random.default_rng(7)
    np.random.seed(7)
    n, d = 249, 6

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
    curX = bestX.copy()
    curVal = bestVal

    archive = [(bestVal, bestX.copy())]

    DELTA_EDGE = 2e-4
    DELTA_HUB = 5e-4

    def edge_set(X, delta):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        iu = np.triu_indices(n, 1)
        c = G[iu]
        sel = np.where(c > m - delta)[0]
        s = set((iu[0][sel] * n + iu[1][sel]).tolist())
        return s, m

    base_edges, _ = edge_set(bestX, DELTA_EDGE)

    def jaccard_to_base(X):
        s, _ = edge_set(X, DELTA_EDGE)
        if not s and not base_edges:
            return 1.0
        inter = len(s.intersection(base_edges))
        union = len(s.union(base_edges))
        if union == 0:
            return 1.0
        return inter / union

    def hub_degrees(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        mask = G > (m - DELTA_HUB)
        deg = np.sum(mask, axis=1)
        return deg, m, mask

    def bfs_neigh(adj, start, m):
        seen = set([int(start)])
        q = deque([int(start)])
        order = [int(start)]
        while q and len(order) < m:
            u = q.popleft()
            nbrs = np.where(adj[u])[0]
            # deterministic order shuffled by rng
            idx = rng.permutation(len(nbrs))
            for k in idx:
                v = int(nbrs[k])
                if v not in seen:
                    seen.add(v)
                    q.append(v)
                    order.append(v)
                    if len(order) >= m:
                        break
        # if not enough, fill by cap proximity
        return np.array(order[:m], dtype=int)

    def pick_free(X, m):
        deg, mx, adj = hub_degrees(X)
        hub = int(np.argmax(deg))
        if rng.random() < 0.5:
            # hub BFS neighbourhood
            # use slightly larger adjacency for connectivity
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            adj2 = G > (mx - 1e-3)
            free = bfs_neigh(adj2, hub, m)
            if len(free) < m:
                # pad with closest to hub direction
                dots = X @ X[hub]
                thr = np.sort(dots)[n - m]
                extra = np.where(dots >= thr)[0]
                free = np.union1d(free, extra)[:m]
                if len(free) < m:
                    extra2 = np.argsort(-dots)[:m]
                    free = np.union1d(free, extra2)[:m]
            # expand to exactly m by nearest to centroid
            if len(free) < m:
                free = np.argsort(-(X @ np.mean(X[free], axis=0)))[:m]
            elif len(free) > m:
                free = free[:m]
            # ensure hub included
            if hub not in free:
                free[0] = hub
            return np.asarray(free, dtype=int)
        else:
            # spherical cap around random/hub direction
            if rng.random() < 0.6:
                u = X[hub] + 0.35 * rng.standard_normal(size=(d,))
            else:
                if rng.random() < 0.5:
                    u = X[int(rng.integers(0, n))] + 0.35 * rng.standard_normal(size=(d,))
                else:
                    u = rng.standard_normal(size=(d,))
            nu = np.linalg.norm(u)
            if nu < 1e-9:
                u = rng.standard_normal(size=(d,))
                nu = np.linalg.norm(u)
            u /= nu
            dots = X @ u
            thr = np.sort(dots)[n - m]
            free = np.where(dots >= thr)[0]
            if len(free) > m:
                # take top m
                free = np.argsort(-dots)[:m]
            elif len(free) < m:
                free = np.argsort(-dots)[:m]
            return np.asarray(free, dtype=int)

    def opt_hole(Xref, init, tau=0.008, steps=220, lr=0.12):
        y = init.copy()
        y /= (np.linalg.norm(y) + 1e-12)
        for s in range(steps):
            dd = Xref @ y
            gm = float(np.max(dd))
            e = np.exp((dd - gm) / tau)
            S = float(np.sum(e))
            w = e / S
            gvec = w @ Xref
            # tangential
            gt = gvec - float(gvec @ y) * y
            st = lr * (0.97 ** (s / 20.0))
            y = y - st * gt
            nr = np.linalg.norm(y)
            if nr < 1e-12:
                y = rng.standard_normal(size=(d,))
                nr = np.linalg.norm(y)
            y /= nr
            if time.time() > T_DIVERSE_END:
                break
        return y

    def greedy_refill(Xfix, m, npool=60000):
        # random pool
        C = rng.standard_normal(size=(npool, d))
        C /= np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)
        # current reference for max
        # cmax over fix
        Df = C @ Xfix.T
        cmax = np.max(Df, axis=1)
        picks = []
        # 2 locally optimized holes
        n_opt = min(2, m)
        Xref = Xfix.copy()
        for _ in range(n_opt):
            bi = int(np.argmin(cmax))
            init = C[bi].copy()
            y = opt_hole(Xref, init)
            picks.append(y)
            # update cmax with new pick
            dn = C @ y
            cmax = np.maximum(cmax, dn)
            Xref = np.vstack([Xref, y[None, :]])
        # remaining greedy from pool
        for _ in range(m - n_opt):
            bi = int(np.argmin(cmax))
            y = C[bi].copy()
            picks.append(y)
            dn = C @ y
            cmax = np.maximum(cmax, dn)
        # diversity: avoid duplicates too close (rare)
        P = np.vstack(picks)
        # renormalize
        P /= np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)
        return P

    def subset_jam(Xfull, free_idx, taboo_fix_mask, taboo_ff_shift, taus, iters, shifts):
        free_idx = np.asarray(free_idx, dtype=int)
        m = len(free_idx)
        mask = np.ones(n, dtype=bool)
        mask[free_idx] = False
        fix_idx = np.where(mask)[0]
        Xfix = Xfull[fix_idx].copy()
        nf = len(fix_idx)
        if m > 1:
            iu = np.triu_indices(m, 1)
            Im, Jm = iu
            nff = len(Im)
        else:
            Im = Jm = np.zeros(0, dtype=int)
            nff = 0
        y = Xfull[free_idx].copy()
        yflat = y.ravel().copy()
        for tau, mi, sh in zip(taus, iters, shifts):
            if time.time() > T_DIVERSE_END:
                break
            Tm = taboo_fix_mask  # m x nf
            def fg(yf, tau=float(tau), sh=float(sh)):
                Y = yf.reshape(m, d)
                r = np.sqrt(np.sum(Y * Y, axis=1))
                r = np.maximum(r, 1e-12)
                Yn = Y / r[:, None]
                D = Yn @ Xfix.T
                Ds = D + sh * Tm
                if nff > 0:
                    Gf = Yn @ Yn.T
                    gff = Gf[Im, Jm]
                    # add shift for taboo ff
                    gffs = gff + sh * taboo_ff_shift
                    gm1 = float(np.max(gffs)) if nff else -2.0
                else:
                    gffs = np.zeros(0)
                    gm1 = -2.0
                gm2 = float(np.max(Ds)) if Ds.size else -2.0
                gm = gm1 if gm1 > gm2 else gm2
                e1 = np.exp((gffs - gm) / tau) if nff else np.zeros(0)
                e2 = np.exp((Ds - gm) / tau)
                S = float(np.sum(e1) + np.sum(e2))
                Lsoft = gm + tau * np.log(S + 1e-300)
                w1 = e1 / S
                w2 = e2 / S
                A = np.zeros((m, d))
                if nff:
                    W = np.zeros((m, m))
                    W[Im, Jm] = w1
                    W = W + W.T
                    A += W @ Yn
                A += w2 @ Xfix
                proj = np.sum(A * Yn, axis=1)
                G0 = (A - proj[:, None] * Yn) / r[:, None]
                creg = 0.05
                diff = r - 1.0
                L = Lsoft + creg * float(np.sum(diff * diff))
                Gr = G0 + (2.0 * creg * diff)[:, None] * (Y / r[:, None])
                return float(L), Gr.ravel()
            try:
                res = minimize(fg, yflat, jac=True, method='L-BFGS-B',
                               options={'maxiter': int(mi), 'maxfun': int(mi * 4),
                                        'ftol': 1e-14, 'gtol': 1e-7, 'maxcor': 8})
                yflat = res.x
            except Exception:
                break
            Ym = yflat.reshape(m, d)
            Ym /= np.maximum(np.linalg.norm(Ym, axis=1, keepdims=True), 1e-12)
            yflat = Ym.ravel().copy()
        Yf = yflat.reshape(m, d)
        Yf /= np.maximum(np.linalg.norm(Yf, axis=1, keepdims=True), 1e-12)
        Xout = Xfull.copy()
        Xout[free_idx] = Yf
        return Xout

    # main diverse loop
    it = 0
    taus = [0.02, 0.008, 0.003, 0.001]
    iters = [40, 50, 60, 75]
    while time.time() < T_DIVERSE_END:
        it += 1
        m = int(rng.integers(18, 36))
        allow = float(rng.uniform(0.005, 0.015))
        try:
            free = pick_free(curX, m)
            # taboo structures from curX incumbent topology
            Gc = curX @ curX.T
            np.fill_diagonal(Gc, -2.0)
            mc = float(np.max(Gc))
            keep = np.ones(n, dtype=bool)
            keep[free] = False
            fix_idx = np.where(keep)[0]
            Xfix_c = curX[fix_idx]
            O = curX[free]  # old positions m x d
            # taboo fix mask m x nf
            Dold = O @ Xfix_c.T
            taboo_fix = (Dold > (mc - 5e-4)).astype(float)
            # taboo ff
            if m > 1:
                iu = np.triu_indices(m, 1)
                Im0, Jm0 = iu
                Gold = O @ O.T
                g_old = Gold[Im0, Jm0]
                taboo_ff = (g_old > (mc - 5e-4)).astype(float)
            else:
                taboo_ff = np.zeros(0)
            # refill greedily w.r.t. cur fixed
            P = greedy_refill(Xfix_c, m, npool=60000)
            trial0 = curX.copy()
            trial0[free] = P
            shifts = [0.004, 0.002, 0.0, 0.0]
            trial = subset_jam(trial0, free, taboo_fix, taboo_ff, taus, iters, shifts)
        except Exception:
            continue
        try:
            vt = _max_cos(trial)
        except Exception:
            continue
        # update base edges if new best (recompute reference? keep original for diversity; but also update best)
        if vt < bestVal - 1e-12:
            bestVal = vt
            bestX = trial.copy()
            curX = trial.copy()
            curVal = vt
            archive.append((vt, trial.copy()))
            base_edges, _ = edge_set(bestX, DELTA_EDGE)
            # trim archive
            archive = sorted(archive, key=lambda t: t[0])[:8]
        else:
            try:
                jc = jaccard_to_base(trial)
            except Exception:
                jc = 1.0
            if jc < 0.7 and vt < bestVal + 2e-4:
                archive.append((vt, trial.copy()))
                archive = sorted(archive, key=lambda t: t[0])[:8]
                # accept diverse even if worse to cross barrier
                if vt < curVal + allow and rng.random() < 0.7:
                    curX = trial.copy()
                    curVal = vt
            elif vt < curVal + allow:
                # non-monotone step
                # bias to accept if diverse
                if jc < 0.85 or rng.random() < 0.35:
                    curX = trial.copy()
                    curVal = vt
            # pull back if drifted too far
            if curVal > bestVal + 0.012:
                # jump to archive diverse element
                if len(archive) > 1 and rng.random() < 0.6:
                    k = int(rng.integers(0, len(archive)))
                    curX = archive[k][1].copy()
                    curVal = archive[k][0]
                else:
                    curX = bestX.copy()
                    curVal = bestVal
        if it % 25 == 0 and time.time() < T_DIVERSE_END:
            # occasional restart pressure
            if curVal > bestVal + 0.004 and rng.random() < 0.5:
                curX = bestX.copy()
                curVal = bestVal

    # final LP polish of best archived
    archive = sorted(archive, key=lambda t: t[0])
    Xfin = archive[0][1].copy() if archive else bestX.copy()
    # ensure best tracked
    for v, X in archive:
        if v < bestVal:
            bestVal = v
            Xfin = X.copy()

    def lp_polish(x, deadline):
        x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        best = _max_cos(x)
        bestXl = x.copy()
        radius = 8e-4
        iu = np.triu_indices(n, 1)
        ar = np.arange(d)
        rep = np.repeat(np.arange(n), d)
        while time.time() < deadline and radius > 1e-14:
            if time.time() > T_END:
                break
            Gf = x @ x.T
            c = Gf[iu]
            mm = float(np.max(c))
            sel = c > mm - max(20.0 * radius, 2e-7)
            p0 = int(np.sum(sel))
            if p0 == 0:
                radius *= 0.5
                continue
            if p0 > 5000:
                idx = np.argpartition(c, -5000)[-5000:]
                selm = np.zeros_like(sel, dtype=bool)
                selm[idx] = True
                sel = selm
            I = iu[0][sel]
            J = iu[1][sel]
            c_sel = c[sel]
            p = c_sel.shape[0]
            nv = n * d + 1
            try:
                rows = np.repeat(np.arange(p), 2 * d + 1)
                cols_block = np.concatenate([(I[:, None] * d + ar), (J[:, None] * d + ar), np.full((p, 1), n * d)], axis=1).ravel()
                vals_block = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
                A_ub = coo_matrix((vals_block, (rows, cols_block)), shape=(p, nv)).tocsr()
                A_eq = coo_matrix((x.ravel(), (rep, np.arange(n * d))), shape=(n, nv)).tocsr()
                obj = np.zeros(nv)
                obj[-1] = 1.0
                bounds = [(-radius, radius)] * (n * d) + [(None, None)]
                res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n), bounds=bounds, method="highs")
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            y = x + res.x[:-1].reshape(n, d)
            y /= np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
            v = _max_cos(y)
            if v < best:
                x = y
                best = v
                bestXl = y.copy()
                radius = min(radius * 1.5, 0.02)
            else:
                radius *= 0.5
        return bestXl, best

    try:
        if time.time() < T_END - 1.0:
            Xfin, vlp = lp_polish(Xfin, T_END - 0.5)
            if vlp < bestVal:
                bestVal = vlp
                bestX = Xfin.copy()
            else:
                # keep lp result only if not worse
                if vlp <= _max_cos(Xfin) + 1e-12:
                    bestX = Xfin.copy()
                    bestVal = vlp
    except Exception:
        pass
    bestX /= np.maximum(np.linalg.norm(bestX, axis=1, keepdims=True), 1e-12)
    # safety: remove NaN
    if not np.all(np.isfinite(bestX)):
        bestX = curX.copy()
        bestX /= np.maximum(np.linalg.norm(bestX, axis=1, keepdims=True), 1e-12)
    return bestX
