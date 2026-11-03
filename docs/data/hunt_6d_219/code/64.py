import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def solve():
    t0 = time.time()
    BUDGET = 168.0
    n, d = 219, 6
    iu0, iu1 = np.triu_indices(n, 1)

    rng0 = np.random.RandomState(0)
    x_best = None
    best_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.load(fn)
            v = np.asarray(v, dtype=float)
            if v.shape == (n, d):
                v = _norm(v)
                c = _max_cos(v)
                if c < best_val:
                    best_val = c
                    x_best = v.copy()
        except Exception:
            pass
    if x_best is None:
        x_best = _norm(rng0.randn(n, d))
        best_val = _max_cos(x_best)

    # ---------- helpers ----------
    def refine_single(v, Others, iters=100, lr0=0.12):
        v = v / max(1e-12, np.linalg.norm(v))
        lr = lr0
        best = float(np.max(Others @ v))
        bv = v.copy()
        for it in range(iters):
            dots = Others @ v
            m = float(np.max(dots))
            tau = 0.0035
            w = np.exp((dots - m) / tau)
            w[dots < m - 0.02] = 0.0
            sw = float(np.sum(w)) + 1e-300
            w = w / sw
            g = w @ Others
            gt = g - float(g @ v) * v
            nm = float(np.linalg.norm(gt))
            if not np.isfinite(nm) or nm < 1e-14:
                break
            step = (lr / max(nm, 1e-12)) * gt * 0.5
            vn = v - step
            vn = vn / max(1e-12, np.linalg.norm(vn))
            mv = float(np.max(Others @ vn))
            if mv < m + 1e-12:
                v = vn
                if mv < best:
                    best = mv
                    bv = vn.copy()
                if it % 30 == 29:
                    lr *= 0.9
            else:
                lr *= 0.6
                if lr < 1e-4:
                    break
        return bv

    def run_subset(X0, mask, beta, maxiter):
        X0 = _norm(X0)
        mask = np.asarray(mask, dtype=int)
        k = len(mask)
        if k == 0:
            return X0
        if k >= n:
            return run_full(X0, beta, maxiter)
        is_free = np.zeros(n, dtype=bool)
        is_free[mask] = True
        sel = is_free[iu0] | is_free[iu1]
        s0 = iu0[sel]; s1 = iu1[sel]
        comp = np.where(~is_free)[0]
        Xfrozen = X0[comp].copy()
        # map full index -> position in mask / comp for assembly speed use index arrays
        y0 = X0[mask].ravel().copy()
        # for assembly: need comp placement
        def fun(y):
            Ym = y.reshape(k, d)
            rn = np.linalg.norm(Ym, axis=1, keepdims=True)
            rn = np.maximum(rn, 1e-12)
            Xm = Ym / rn
            X = np.empty((n, d))
            X[mask] = Xm
            X[comp] = Xfrozen
            G = X @ X.T
            g = G[s0, s1]
            m = float(np.max(g))
            e = np.exp(beta * (g - m))
            S = float(np.sum(e)) + 1e-300
            loss = m + np.log(S) / beta
            w = e / S
            W = np.zeros((n, n))
            W[s0, s1] = w
            W = W + W.T
            GX = W @ X
            GXm = GX[mask]
            dot = (GXm * Xm).sum(axis=1, keepdims=True)
            GY = (GXm - dot * Xm) / rn
            return float(loss), GY.ravel()
        try:
            res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                           options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-7, "maxfun": 15000})
            Ym = res.x.reshape(k, d)
            Xn = np.empty((n, d))
            Xn[mask] = Ym / np.maximum(np.linalg.norm(Ym, axis=1, keepdims=True), 1e-12)
            Xn[comp] = Xfrozen
            return _norm(Xn)
        except Exception:
            return X0

    def run_full(X0, beta, maxiter):
        X0 = _norm(X0)
        y0 = X0.ravel().copy()
        def fun(y):
            Y = y.reshape(n, d)
            rn = np.linalg.norm(Y, axis=1, keepdims=True)
            rn = np.maximum(rn, 1e-12)
            X = Y / rn
            G = X @ X.T
            g = G[iu0, iu1]
            m = float(np.max(g))
            e = np.exp(beta * (g - m))
            S = float(np.sum(e)) + 1e-300
            loss = m + np.log(S) / beta
            w = e / S
            W = np.zeros((n, n))
            W[iu0, iu1] = w
            W = W + W.T
            GX = W @ X
            dot = (GX * X).sum(axis=1, keepdims=True)
            GY = (GX - dot * X) / rn
            return float(loss), GY.ravel()
        try:
            res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                           options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-7, "maxfun": 15000})
            Y = res.x.reshape(n, d)
            return _norm(Y)
        except Exception:
            return X0

    def lp_polish(x, deadline, r_init=1.0e-3, max_steps=10):
        x = _norm(x.copy())
        best = _max_cos(x)
        bx = x.copy()
        radius = r_init
        steps = 0
        while time.time() < deadline and steps < max_steps:
            steps += 1
            if radius < 1e-14:
                break
            Gf = x @ x.T
            c = Gf[iu0, iu1]
            m = float(c.max())
            thr = m - max(20.0 * radius, 1e-9)
            selm = c > thr
            I = iu0[selm]; J = iu1[selm]; csel = c[selm]
            p = len(I)
            if p == 0:
                break
            if p > 3500:
                kk = 3500
                part = np.argpartition(-csel, kk - 1)[:kk]
                I, J, csel = I[part], J[part], csel[part]
                p = kk
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, n * d + 1)).tocsr()
            obj = np.zeros(n * d + 1); obj[-1] = 1.0
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            y = _norm(x + res.x[:-1].reshape(n, d))
            v = _max_cos(y)
            if v < best:
                x, best = y, v
                bx = y.copy()
                radius = min(radius * 1.5, 0.02)
            else:
                radius *= 0.5
        return bx, best

    ks_cycle = [4, 3, 5, 4, 5, 3]
    trial = 0
    # main surgery loop
    while time.time() - t0 < BUDGET - 14 and trial < 60:
        # contact analysis on incumbent
        Gc = x_best @ x_best.T
        np.fill_diagonal(Gc, -2.0)
        m0 = float(np.max(Gc))
        deg = np.sum(Gc > m0 - 1e-9, axis=1)
        rattler = int(np.argmin(deg))
        order = np.argsort(-deg, kind="stable")
        # pick hub among top with degree>=14 if possible
        hub = int(order[trial % 5])
        # ensure distinct
        if hub == rattler:
            hub = int(order[5 % n])
        k = int(ks_cycle[trial % len(ks_cycle)])
        # neighbour rankings
        gr = Gc[rattler].copy()
        gh = Gc[hub].copy()
        gr[rattler] = -2; gr[hub] = -2
        gh[hub] = -2; gh[rattler] = -2
        topR = np.argsort(-gr)[:20]
        topH = np.argsort(-gh)[:20]
        setR = set(topR[:15].tolist()); setH = set(topH[:15].tolist())
        shared = [i for i in topH if i in setR]
        # order shared by max(gr,gh)
        shared = sorted(shared, key=lambda i: max(gr[i], gh[i]), reverse=True)
        cand_list = []
        for i in shared:
            if i not in cand_list:
                cand_list.append(i)
        # hub neighbours with offset to diversify
        off = (trial // 5) % 4
        seqH = np.argsort(-gh)
        # filter out r/h
        seqH = [i for i in seqH if i != rattler and i != hub]
        seqH = seqH[off:] + seqH[:off]
        for i in seqH:
            if i not in cand_list:
                cand_list.append(i)
            if len(cand_list) > 30:
                break
        need = k - 2
        D = [rattler, hub]
        for i in cand_list:
            if i not in D:
                D.append(int(i))
            if len(D) >= k:
                break
        D = np.array(D[:k], dtype=int)
        surv_idx = np.array([i for i in range(n) if i not in set(D.tolist())], dtype=int)
        S = x_best[surv_idx].copy()

        rng = np.random.RandomState(9000 + trial * 131)
        Cnum = 20000
        Cand = rng.randn(Cnum, d)
        Cand = _norm(Cand)
        # max to survivors
        # chunk to be safe
        surv_max = np.empty(Cnum)
        # single matmul is fine (20000x215)
        try:
            M = Cand @ S.T
            surv_max = np.max(M, axis=1)
        except Exception:
            for a in range(0, Cnum, 5000):
                b = min(Cnum, a + 5000)
                surv_max[a:b] = np.max(Cand[a:b] @ S.T, axis=1)
        nref = 60
        top_idx = np.argpartition(surv_max, nref)[:nref]
        Ref = np.empty((nref, d))
        for ii, ci in enumerate(top_idx):
            Ref[ii] = refine_single(Cand[ci], S, iters=90)
        Ref = _norm(Ref)
        Pool = np.vstack([Cand, Ref])
        try:
            Mref = Ref @ S.T
            ref_max = np.max(Mref, axis=1)
        except Exception:
            ref_max = np.array([float(np.max(S @ v)) for v in Ref])
        Pool_surv = np.concatenate([surv_max, ref_max])
        P = Pool.shape[0]
        used = np.zeros(P, dtype=bool)
        picked = []
        cur_to_ins = np.full(P, -2.0)
        for p_ in range(k):
            if p_ == 0:
                scores = Pool_surv.copy()
            else:
                scores = np.maximum(Pool_surv, cur_to_ins)
            scores[used] = np.inf
            bi = int(np.argmin(scores))
            used[bi] = True
            picked.append(bi)
            # update cur_to_ins with new vector
            nv = Pool[bi]
            dots = Pool @ nv
            cur_to_ins = np.maximum(cur_to_ins, dots)
        Inserted = Pool[np.array(picked)]
        # build initial X0
        X0 = np.empty((n, d))
        X0[surv_idx] = S
        for ii, di in enumerate(D):
            X0[di] = Inserted[ii]
        X0 = _norm(X0)

        # mask A: inserted + survivors with cosine>0.55 to any inserted
        try:
            G0 = X0 @ X0.T
            max_to_D = np.max(G0[:, D], axis=1)
            maskA = [int(i) for i in D] + [int(i) for i in surv_idx if max_to_D[i] > 0.55]
            maskA = np.array(sorted(set(maskA)), dtype=int)
        except Exception:
            maskA = D.copy()
        # frozen-stage optimisation beta 200 -> 2000 ladder (use 250,1200 for stability)
        if time.time() - t0 > BUDGET - 14:
            break
        X1 = run_subset(X0, maskA, 250.0, 60)
        if time.time() - t0 > BUDGET - 13:
            break
        X1 = run_subset(X1, maskA, 1200.0, 60)
        # expanding shells
        try:
            G1 = X1 @ X1.T
            maxD1 = np.max(G1[:, D], axis=1)
            maskB = sorted(set(D.tolist() + [i for i in range(n) if maxD1[i] > 0.45]))
            maskB = np.array(maskB, dtype=int)
        except Exception:
            maskB = maskA
        X2 = X1
        if len(maskB) < n and time.time() - t0 < BUDGET - 12:
            X2 = run_subset(X1, maskB, 400.0, 60)
        try:
            G2 = X2 @ X2.T
            maxD2 = np.max(G2[:, D], axis=1)
            maskC = sorted(set(D.tolist() + [i for i in range(n) if maxD2[i] > 0.30]))
            maskC = np.array(maskC, dtype=int)
        except Exception:
            maskC = maskB
        X3 = X2
        if len(maskC) < n and len(maskC) > len(maskB) and time.time() - t0 < BUDGET - 11:
            X3 = run_subset(X2, maskC, 400.0, 50)
        # full release
        X4 = X3
        if time.time() - t0 < BUDGET - 10:
            X4 = run_full(X3, 350.0, 60)
        if time.time() - t0 < BUDGET - 9:
            # second sharpen only if promising
            vtmp = _max_cos(X4)
            if vtmp < best_val + 0.005:
                X4b = run_full(X4, 650.0, 40)
                if _max_cos(X4b) < vtmp:
                    X4 = X4b
        # LP polish
        v4 = _max_cos(X4)
        if v4 < best_val + 0.008 and time.time() - t0 < BUDGET - 8:
            try:
                Xp, vp = lp_polish(X4, deadline=min(time.time() + 3.5, t0 + BUDGET - 2.0), max_steps=10)
            except Exception:
                Xp, vp = X4, v4
        else:
            Xp, vp = X4, v4
        # judge after full release by exact max and topology change (count)
        try:
            Gp = Xp @ Xp.T
            np.fill_diagonal(Gp, -2.0)
            vp_exact = float(np.max(Gp))
            cnt = int(np.sum(Gp > vp_exact - 1e-9) // 2)
        except Exception:
            vp_exact = vp
            cnt = -1
        if vp_exact < best_val - 1e-12:
            x_best = _norm(Xp.copy())
            best_val = vp_exact
        trial += 1

    # final squeeze on best if time remains
    if time.time() - t0 < BUDGET - 2:
        try:
            Xp, vp = lp_polish(x_best, deadline=t0 + BUDGET - 0.5, max_steps=20)
            if vp < best_val:
                x_best, best_val = Xp, vp
        except Exception:
            pass
    x_best = _norm(x_best)
    return x_best
