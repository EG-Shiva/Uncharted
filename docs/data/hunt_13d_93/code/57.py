import time
import numpy as np

def solve():
    t0 = time.time()
    DEADLINE = 172.0
    n, d = 93, 13
    seed_base = 999

    def _norm(X):
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        return X / nr

    def _maxcos(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        return float(np.max(G))

    def _rowmax(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        return G.max(axis=1)

    # load best
    gbest = None
    gval = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (n, d):
                continue
            if not np.all(np.isfinite(a)):
                continue
            a = _norm(a)
            v = _maxcos(a)
            if v < gval:
                gval = v
                gbest = a.copy()
        except Exception:
            pass
    rng0 = np.random.default_rng(seed_base)
    if gbest is None:
        gbest = _norm(rng0.standard_normal((n, d)))
        gval = _maxcos(gbest)
    cur_best = gbest.copy()
    cur_val = float(gval)

    # ---- void pool refinement (vectorized tangent Adam on LSE) ----
    def refine_pool(C, F, stages, t0, deadline):
        C = _norm(C.copy())
        m_ad = np.zeros_like(C)
        v_ad = np.zeros_like(C)
        b1, b2, eps = 0.9, 0.999, 1e-8
        tt = 0
        for (tau, steps, lr) in stages:
            for s in range(steps):
                if time.time() - t0 > deadline:
                    break
                G = C @ F.T
                mr = G.max(axis=1, keepdims=True)
                E = np.exp((G - mr) / tau)
                S = E.sum(axis=1, keepdims=True) + 1e-300
                W = E / S
                grad = W @ F
                dot = np.einsum('ij,ij->i', grad, C)[:, None]
                gt = grad - dot * C
                tt += 1
                m_ad = b1 * m_ad + (1 - b1) * gt
                v_ad = b2 * v_ad + (1 - b2) * gt * gt
                mh = m_ad / (1.0 - b1 ** tt)
                vh = v_ad / (1.0 - b2 ** tt)
                upd = mh / (np.sqrt(vh) + eps)
                dd = np.einsum('ij,ij->i', upd, C)[:, None]
                upd_t = upd - dd * C
                C = _norm(C - lr * upd_t)
            if time.time() - t0 > deadline:
                break
        return C

    def refine_single(z0, Sset, stages, t0, deadline):
        C = _norm(np.asarray(z0, dtype=np.float64).reshape(1, d))
        C = refine_pool(C, Sset, stages, t0, deadline)
        return C[0:1, :]

    def smooth_full(X, stages, t0, deadline):
        X = _norm(X.copy())
        N = X.shape[0]
        iu0_, iu1_ = np.triu_indices(N, 1)
        m_ad = np.zeros_like(X)
        v_ad = np.zeros_like(X)
        b1, b2, eps = 0.9, 0.999, 1e-8
        tt = 0
        for (tau, steps, lr) in stages:
            for s in range(steps):
                if time.time() - t0 > deadline:
                    break
                G = X @ X.T
                c = G[iu0_, iu1_]
                mx = float(c.max())
                E = np.exp((c - mx) / tau)
                S = float(E.sum()) + 1e-300
                W = np.zeros((N, N), dtype=np.float64)
                W[iu0_, iu1_] = E
                W[iu1_, iu0_] = E
                grad = (W @ X) / S
                dot = np.einsum('ij,ij->i', grad, X)[:, None]
                gt = grad - dot * X
                tt += 1
                m_ad = b1 * m_ad + (1 - b1) * gt
                v_ad = b2 * v_ad + (1 - b2) * gt * gt
                mh = m_ad / (1.0 - b1 ** tt)
                vh = v_ad / (1.0 - b2 ** tt)
                upd = mh / (np.sqrt(vh) + eps)
                dd = np.einsum('ij,ij->i', upd, X)[:, None]
                upd_t = upd - dd * X
                X = _norm(X - lr * upd_t)
            if time.time() - t0 > deadline:
                break
        return X

    def lp_polish(Xinit, budget_s, t0, deadline):
        try:
            from scipy.optimize import linprog
            from scipy.sparse import coo_matrix
        except Exception:
            return Xinit, _maxcos(Xinit)
        X = _norm(Xinit.copy())
        cur = _maxcos(X)
        bestX = X.copy(); bestV = cur
        N = X.shape[0]
        iu0_, iu1_ = np.triu_indices(N, 1)
        M = len(iu0_)
        dm = d - 1
        Nvar = N * dm + 1
        ar = np.arange(dm)
        radius = 3e-3
        c_obj = np.zeros(Nvar); c_obj[-1] = 1.0
        bounds = [(-radius, radius)] * (N * dm) + [(None, None)]
        # will update bounds each iter
        t_start = time.time()
        it = 0
        while True:
            if time.time() - t0 > deadline:
                break
            if time.time() - t_start > budget_s:
                break
            it += 1
            if radius < 1e-12 or it > 40:
                break
            G = X @ X.T
            c_all = G[iu0_, iu1_]
            cmax = float(c_all.max())
            K = min(800, M)
            order = np.argpartition(-c_all, K - 1)[:K]
            thr = cmax - 2e-3
            mask = np.zeros(M, dtype=bool)
            mask[order] = True
            mask |= (c_all >= thr)
            I = iu0_[mask]; J = iu1_[mask]; csub = c_all[mask]
            p = len(I)
            if p == 0:
                break
            T = np.empty((N, d, dm))
            for ii in range(N):
                xi = X[ii]
                k = int(np.argmin(np.abs(xi)))
                B = np.empty((d, dm))
                col = 0
                for jj in range(d):
                    if jj == k:
                        continue
                    vv = -xi * xi[jj]; vv[jj] += 1.0
                    B[:, col] = vv; col += 1
                Qq, _ = np.linalg.qr(B)
                T[ii] = Qq
            TI = T[I]; TJ = T[J]
            XI = X[I]; XJ = X[J]
            CI = np.einsum('pdm,pd->pm', TI, XJ)
            CJ = np.einsum('pdm,pd->pm', TJ, XI)
            nnz = p * (2 * dm + 1)
            rows = np.empty(nnz, dtype=np.int64)
            cols = np.empty(nnz, dtype=np.int64)
            data = np.empty(nnz, dtype=np.float64)
            r1 = np.repeat(np.arange(p), dm)
            c1 = (I[:, None] * dm + ar[None, :]).ravel()
            c2 = (J[:, None] * dm + ar[None, :]).ravel()
            rows[0:p*dm] = r1; cols[0:p*dm] = c1; data[0:p*dm] = CI.ravel()
            rows[p*dm:2*p*dm] = r1; cols[p*dm:2*p*dm] = c2; data[p*dm:2*p*dm] = CJ.ravel()
            rows[2*p*dm:2*p*dm+p] = np.arange(p); cols[2*p*dm:2*p*dm+p] = Nvar-1; data[2*p*dm:2*p*dm+p] = -1.0
            A = coo_matrix((data, (rows, cols)), shape=(p, Nvar)).tocsr()
            b = -csub
            bnds = [(-radius, radius)] * (N * dm) + [(cmax - 0.05, cmax + 0.005)]
            try:
                res = linprog(c_obj, A_ub=A, b_ub=b, bounds=bnds, method='highs',
                              options={'presolve': True})
            except Exception:
                break
            if not res.success or res.x is None:
                radius *= 0.5
                continue
            y = res.x[:N*dm].reshape(N, dm)
            t_pred = float(res.x[-1])
            # apply
            dX = np.einsum('idm,im->id', T, y)
            Xn = _norm(X + dX)
            vn = _maxcos(Xn)
            if vn < cur - 1e-12:
                cur = vn
                X = Xn
                if vn < bestV:
                    bestV = vn; bestX = Xn.copy()
                # allow growth if prediction good
                radius = min(radius * 1.5, 1e-2)
            else:
                radius *= 0.5
                if vn < bestV - 1e-12:
                    bestV = vn; bestX = Xn.copy()
        return bestX, bestV

    POOL_STAGES = [(0.01, 60, 0.05), (0.003, 60, 0.02), (0.001, 50, 0.01)]
    SINGLE_STAGES = [(0.008, 80, 0.05), (0.002, 80, 0.02), (0.0007, 60, 0.01)]
    FULL_STAGES = [(0.012, 120, 0.012), (0.005, 120, 0.008), (0.002, 120, 0.005), (0.0008, 100, 0.003)]

    nseeds = 5
    for seed in range(nseeds):
        if time.time() - t0 > DEADLINE - 25:
            break
        rng = np.random.default_rng(1000 + seed * 7919)
        Xbase = cur_best.copy()
        # identify bottleneck points
        try:
            G = Xbase @ Xbase.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            # exact maximal pairs within 1e-9
            iu0_, iu1_ = np.triu_indices(n, 1)
            c_all = G[iu0_, iu1_]
            exact_mask = c_all >= m - 1e-9
            eset = set()
            for a, b in zip(iu0_[exact_mask], iu1_[exact_mask]):
                eset.add(int(a)); eset.add(int(b))
            # any within 1e-8 by rowmax
            rm = G.max(axis=1)
            extra = np.where(rm >= m - 1e-8)[0]
            rset = set(eset) | set(int(x) for x in extra.tolist())
            if len(rset) == 0:
                # fallback: worst 4
                order = np.argsort(-rm)[:4]
                rset = set(int(x) for x in order.tolist())
            if len(rset) > 8:
                # keep 8 worst by rowmax
                lst = sorted(list(rset), key=lambda i: -rm[i])[:8]
                rset = set(lst)
            rem = sorted(list(rset))
        except Exception:
            rem = [0, 1, 2, 3]
        keep = [i for i in range(n) if i not in set(rem)]
        F = Xbase[keep, :].copy()
        nf = F.shape[0]
        kneed = n - nf
        # ---- large void sampling: >=5e5 ----
        TOTAL = 500000
        BATCH = 50000
        nb = TOTAL // BATCH
        best_scores = np.full(200, 1e9, dtype=np.float64)
        best_dirs = np.zeros((200, d), dtype=np.float64)
        # keep global top200 via iterative merge
        first = True
        for bi in range(nb):
            if time.time() - t0 > DEADLINE - 20:
                break
            Z = rng.standard_normal((BATCH, d))
            Z = _norm(Z)
            D = Z @ F.T
            sc = D.max(axis=1)
            # want 200 smallest
            if first and bi == 0:
                idx = np.argpartition(sc, 200)[:200]
                best_scores = sc[idx].copy()
                best_dirs = Z[idx].copy()
                first = False
            else:
                all_s = np.concatenate([best_scores, sc])
                # argpartition for 200 smallest
                idx = np.argpartition(all_s, 200)[:200]
                # need corresponding dirs: first 200 are best_dirs, rest Z
                # build stacked dirs
                stacked = np.vstack([best_dirs, Z])
                best_scores = all_s[idx].copy()
                best_dirs = stacked[idx].copy()
            # sort descending for stability? keep as is
        # order best_dirs by score
        o = np.argsort(best_scores)
        best_scores = best_scores[o]
        best_dirs = best_dirs[o]
        # refine pool
        try:
            if time.time() - t0 > DEADLINE - 18:
                break
            Cref = refine_pool(best_dirs.copy(), F, POOL_STAGES, t0, DEADLINE - 12)
        except Exception:
            Cref = best_dirs.copy()
        # score refined by true max
        try:
            Dref = Cref @ F.T
            sref = Dref.max(axis=1)
            o2 = np.argsort(sref)
            Cref = Cref[o2]; sref = sref[o2]
        except Exception:
            pass
        # sequential reinsertion
        inserted = []
        cur_set = F.copy()
        used = np.zeros(len(Cref), dtype=bool)
        for ri in range(kneed):
            if time.time() - t0 > DEADLINE - 14:
                break
            # score unused pool against cur_set
            cand_idx = np.where(~used)[0]
            if len(cand_idx) == 0:
                # random new
                z0 = _norm(rng.standard_normal((1, d)))
                zref = refine_single(z0, cur_set, SINGLE_STAGES, t0, DEADLINE - 12)
                inserted.append(zref[0])
                cur_set = np.vstack([cur_set, zref])
                continue
            Cc = Cref[cand_idx]
            D = Cc @ cur_set.T
            sc = D.max(axis=1)
            bi = int(np.argmin(sc))
            gi = int(cand_idx[bi])
            used[gi] = True
            z0 = Cref[gi:gi+1, :].copy()
            try:
                zref = refine_single(z0, cur_set, SINGLE_STAGES, t0, DEADLINE - 12)
            except Exception:
                zref = _norm(z0)
            inserted.append(zref[0].copy())
            cur_set = np.vstack([cur_set, zref])
        if len(inserted) != kneed:
            continue
        # assemble full config: keep frozen order + inserted (order doesn't matter)
        Xnew = np.vstack([F, np.array(inserted)])
        # shuffle to avoid bias? keep as is but full polish will mix
        # unfreeze annealed Adam polish
        try:
            Xp = smooth_full(Xnew, FULL_STAGES, t0, DEADLINE - 6)
        except Exception:
            Xp = Xnew
        vp = _maxcos(Xp)
        # tangent-LP polish short budget
        try:
            budget = 4.0
            if seed == 0:
                budget = 6.0
            Xl, vl = lp_polish(Xp, budget, t0, DEADLINE - 2)
        except Exception:
            Xl, vl = Xp, vp
        # also try LP on unpolished? keep best of Xp/Xl
        if vp < vl:
            Xl, vl = Xp, vp
        if vl < cur_val - 1e-12:
            cur_val = float(vl)
            cur_best = _norm(Xl.copy())
    cur_best = _norm(cur_best)
    return cur_best
