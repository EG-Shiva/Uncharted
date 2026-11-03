import time
import numpy as np

def solve():
    t0 = time.time()
    BUDGET = 168.0
    n, d = 93, 13
    m = d - 1
    iu0, iu1 = np.triu_indices(n, 1)

    def _norm(X):
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        return X / nr

    def _maxcos(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        return float(np.max(G))

    # load base
    base_best = None
    base_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (n, d):
                continue
            a = _norm(a)
            v = _maxcos(a)
            if v < base_val:
                base_val = v
                base_best = a.copy()
        except Exception:
            pass
    if base_best is None:
        rng0 = np.random.default_rng(0)
        base_best = _norm(rng0.standard_normal((n, d)))
        base_val = _maxcos(base_best)

    # identify bottleneck points: pairs within 1e-8 of max
    G0 = base_best @ base_best.T
    np.fill_diagonal(G0, -2.0)
    c_all0 = G0[iu0, iu1]
    cmax0 = float(np.max(c_all0))
    mask = (cmax0 - c_all0) < 1e-8
    R = np.unique(np.concatenate([iu0[mask], iu1[mask]]))
    if R.size < 2:
        # fallback: take the exact max pairs
        idx = np.argmax(c_all0)
        R = np.array([iu0[idx], iu1[idx]], dtype=int)
    if R.size > 12:
        # keep only tightest points if too many (order by row max)
        rowmax = np.max(G0, axis=1)
        order = np.argsort(-rowmax)
        keep = order[:12]
        # ensure max pair included
        R = np.unique(R[np.isin(R, keep)] if np.any(np.isin(R, keep)) else keep[:4])
        if R.size < 4:
            R = keep[:4]
    R = np.asarray(R, dtype=int)
    k = len(R)
    keep_mask = np.ones(n, dtype=bool)
    keep_mask[R] = False
    Kidx = np.where(keep_mask)[0]
    Xfrozen = base_best[Kidx].copy()
    Nf = Xfrozen.shape[0]

    x_best = base_best.copy()
    best_val = float(base_val)

    # ---------- void refinement helpers ----------
    def void_batch_refine(Y0, Xr, steps, tau0, tau1, lr0, lr1):
        Y = _norm(Y0.copy())
        B = Y.shape[0]
        mt = np.zeros_like(Y)
        vt = np.zeros_like(Y)
        b1, b2, eps = 0.9, 0.999, 1e-8
        tt = 0
        ltau = np.log(tau1 / tau0) if tau0 != tau1 else 0.0
        llr = np.log(lr1 / lr0) if lr0 != lr1 else 0.0
        for s in range(steps):
            f = s / max(1, steps - 1)
            tau = tau0 * np.exp(ltau * f)
            lr = lr0 * np.exp(llr * f)
            S = Y @ Xr.T  # B x Nr
            mr = S.max(axis=1)  # B
            E = np.exp((S - mr[:, None]) / tau)
            Esum = E.sum(axis=1, keepdims=True) + 1e-300
            Q = E / Esum  # softmax
            grad = Q @ Xr  # B x d
            dot = np.einsum('ij,ij->i', grad, Y)[:, None]
            gt = grad - dot * Y
            tt += 1
            mt = b1 * mt + (1.0 - b1) * gt
            vt = b2 * vt + (1.0 - b2) * (gt * gt)
            mh = mt / (1.0 - b1 ** tt)
            vh = vt / (1.0 - b2 ** tt)
            step = lr * mh / (np.sqrt(vh) + eps)
            sd = np.einsum('ij,ij->i', step, Y)[:, None]
            step_t = step - sd * Y
            Y = _norm(Y - step_t)
        return Y

    def void_single_refine(y0, Xr, steps, tau0, tau1, lr0, lr1):
        y = y0 / (np.linalg.norm(y0) + 1e-300)
        y = y.copy()
        mt = np.zeros_like(y)
        vt = np.zeros_like(y)
        b1, b2, eps = 0.9, 0.999, 1e-8
        tt = 0
        ltau = np.log(tau1 / tau0) if tau0 != tau1 else 0.0
        llr = np.log(lr1 / lr0) if lr0 != lr1 else 0.0
        for s in range(steps):
            f = s / max(1, steps - 1)
            tau = tau0 * np.exp(ltau * f)
            lr = lr0 * np.exp(llr * f)
            s_ = Xr @ y  # Nr
            mr = float(np.max(s_))
            E = np.exp((s_ - mr) / tau)
            S = float(np.sum(E)) + 1e-300
            Q = E / S
            grad = Q @ Xr  # d
            dot = float(np.dot(grad, y))
            gt = grad - dot * y
            tt += 1
            mt = b1 * mt + (1.0 - b1) * gt
            vt = b2 * vt + (1.0 - b2) * (gt * gt)
            mh = mt / (1.0 - b1 ** tt)
            vh = vt / (1.0 - b2 ** tt)
            step = lr * mh / (np.sqrt(vh) + eps)
            step = step - float(np.dot(step, y)) * y
            y = y - step
            y = y / (np.linalg.norm(y) + 1e-300)
        return y

    def full_adam_lse(X0, steps, tau0, tau1, lr0, lr1):
        X = _norm(X0.copy())
        mt = np.zeros_like(X)
        vt = np.zeros_like(X)
        b1, b2, eps = 0.9, 0.999, 1e-8
        tt = 0
        bv = _maxcos(X)
        bX = X.copy()
        ltau = np.log(tau1 / tau0) if tau0 != tau1 else 0.0
        llr = np.log(lr1 / lr0) if lr0 != lr1 else 0.0
        for s in range(steps):
            if time.time() - t0 > BUDGET - 8:
                break
            f = s / max(1, steps - 1)
            tau = tau0 * np.exp(ltau * f)
            lr = lr0 * np.exp(llr * f)
            G = X @ X.T
            np.fill_diagonal(G, -10.0)
            cur = float(G.max())
            if cur < bv:
                bv = cur
                bX = X.copy()
            W = np.exp((G - cur) / tau)
            np.fill_diagonal(W, 0.0)
            S = float(W.sum() * 0.5)
            if not np.isfinite(S) or S <= 0:
                continue
            grad = (W @ X) / S
            dot = np.einsum('ij,ij->i', grad, X)[:, None]
            gt = grad - dot * X
            tt += 1
            mt = b1 * mt + (1.0 - b1) * gt
            vt = b2 * vt + (1.0 - b2) * (gt * gt)
            mh = mt / (1.0 - b1 ** tt)
            vh = vt / (1.0 - b2 ** tt)
            step = lr * mh / (np.sqrt(vh) + eps)
            sd = np.einsum('ij,ij->i', step, X)[:, None]
            step_t = step - sd * X
            X = _norm(X - step_t)
        return bX, bv

    def lp_polish(Xin, max_iters=6, topk=900):
        try:
            from scipy.optimize import linprog
            from scipy.sparse import coo_matrix
        except Exception:
            return Xin, _maxcos(Xin)
        X = _norm(Xin.copy())
        cur = _maxcos(X)
        bX = X.copy()
        bv = cur
        nv = n * m + 1
        radius = 3e-3
        rmax = 8e-3
        c_obj = np.zeros(nv)
        c_obj[-1] = 1.0
        ar = np.arange(m)
        for it in range(max_iters):
            if time.time() - t0 > BUDGET - 4:
                break
            G = X @ X.T
            c_all = G[iu0, iu1]
            cmax = float(np.max(c_all))
            cur = cmax
            # active set: topk largest
            P = len(c_all)
            K = min(topk, P)
            if K < P:
                idx = np.argpartition(-c_all, K)[:K]
            else:
                idx = np.arange(P)
            I = iu0[idx]; J = iu1[idx]
            csub = c_all[idx]
            p = len(I)
            # tangent bases
            T = np.empty((n, d, m))
            for i in range(n):
                xi = X[i]
                kk = int(np.argmin(np.abs(xi)))
                B = np.empty((d, m))
                col = 0
                for j in range(d):
                    if j == kk:
                        continue
                    v = -xi * xi[j]
                    v[j] += 1.0
                    B[:, col] = v
                    col += 1
                Q, _ = np.linalg.qr(B)
                T[i] = Q
            TI = T[I]; TJ = T[J]
            XI = X[I]; XJ = X[J]
            CI = np.einsum('pdm,pd->pm', TI, XJ)
            CJ = np.einsum('pdm,pd->pm', TJ, XI)
            nnz = p * (2 * m + 1)
            rows = np.empty(nnz, dtype=np.int64)
            cols = np.empty(nnz, dtype=np.int64)
            data = np.empty(nnz, dtype=np.float64)
            r1 = np.repeat(np.arange(p), m)
            c1 = (I[:, None] * m + ar[None, :]).ravel()
            r2 = np.repeat(np.arange(p), m)
            c2 = (J[:, None] * m + ar[None, :]).ravel()
            rows[0:p*m] = r1; cols[0:p*m] = c1; data[0:p*m] = CI.ravel()
            rows[p*m:2*p*m] = r2; cols[p*m:2*p*m] = c2; data[0+ p*m:2*p*m] = CJ.ravel()
            rows[2*p*m:2*p*m+p] = np.arange(p)
            cols[2*p*m:2*p*m+p] = nv - 1
            data[2*p*m:2*p*m+p] = -1.0
            A_ub = coo_matrix((data, (rows, cols)), shape=(p, nv)).tocsr()
            b_ub = -csub
            bounds = [(-radius, radius)] * (n * m) + [(-1.0, cur)]
            try:
                res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                              options={'presolve': True})
            except Exception:
                break
            if not res.success:
                radius *= 0.5
                if radius < 1e-12:
                    break
                continue
            t_pred = float(res.x[-1])
            U = res.x[:n*m].reshape(n, m)
            DX = np.einsum('idm,im->id', T, U)
            Xt = _norm(X + DX)
            vt = _maxcos(Xt)
            if vt < cur - 1e-12:
                X = Xt
                cur = vt
                if vt < bv:
                    bv = vt; bX = Xt.copy()
                # expand if good prediction
                if t_pred < cur + 1e-9:
                    radius = min(radius * 1.5, rmax)
                else:
                    radius = min(radius * 1.2, rmax)
            else:
                radius *= 0.5
                if radius < 5e-7:
                    # small random tangent kick if stuck
                    break
        return bX, bv

    N_ATTEMPTS = 5
    for att in range(N_ATTEMPTS):
        if time.time() - t0 > BUDGET - 25:
            break
        rng = np.random.default_rng(777 + att * 101)
        # ---- generate >=5e5 random unit directions in batches ----
        NBATCH = 20
        BS = 25000  # 20*25k = 500k
        PBEST = 200
        best_scores = np.full(PBEST, np.inf)
        best_Y = np.zeros((PBEST, d))
        # init with random to avoid empty
        for b in range(NBATCH):
            if time.time() - t0 > BUDGET - 20:
                break
            Yb = rng.standard_normal((BS, d))
            Yb = Yb / (np.linalg.norm(Yb, axis=1, keepdims=True) + 1e-300)
            Sb = Yb @ Xfrozen.T
            sc = Sb.max(axis=1)
            # merge with current best
            # combine
            all_s = np.concatenate([best_scores, sc])
            all_Y = np.vstack([best_Y, Yb])
            idx = np.argpartition(all_s, PBEST)[:PBEST]
            best_scores = all_s[idx]
            best_Y = all_Y[idx]
        # order best by score
        o = np.argsort(best_scores)
        best_scores = best_scores[o]
        best_Y = best_Y[o]
        # ---- refine best 200 by tangent Adam on LSE void objective ----
        try:
            Yref = void_batch_refine(best_Y, Xfrozen, steps=140, tau0=0.02, tau1=0.002, lr0=0.03, lr1=0.005)
        except Exception:
            Yref = best_Y.copy()
        # score refined vs frozen, keep order
        try:
            sref = (Yref @ Xfrozen.T).max(axis=1)
            o2 = np.argsort(sref)
            Yref = Yref[o2]
        except Exception:
            pass
        pool = Yref.copy()  # 200 x d
        # ---- sequential reinsertion ----
        Xcur = Xfrozen.copy()  # grows
        new_pts = []
        ok = True
        for s in range(k):
            if time.time() - t0 > BUDGET - 18:
                ok = False
                break
            # score pool vs current set
            Sp = pool @ Xcur.T
            scp = Sp.max(axis=1)
            bi = int(np.argmin(scp))
            y0 = pool[bi].copy()
            # refine chosen
            try:
                y_star = void_single_refine(y0, Xcur, steps=350, tau0=0.015, tau1=0.0015, lr0=0.03, lr1=0.004)
            except Exception:
                y_star = y0 / (np.linalg.norm(y0) + 1e-300)
            # challengers: fresh random search vs Xcur to avoid pool staleness
            try:
                Nc = 30000
                Yc = rng.standard_normal((Nc, d))
                Yc = Yc / (np.linalg.norm(Yc, axis=1, keepdims=True) + 1e-300)
                scc = (Yc @ Xcur.T).max(axis=1)
                tki = np.argpartition(scc, 3)[:3]
                Yt = Yc[tki]
                Yt = void_batch_refine(Yt, Xcur, steps=120, tau0=0.015, tau1=0.002, lr0=0.03, lr1=0.005)
                # pick best among y_star and Yt
                cand = np.vstack([y_star[None, :], Yt])
                sc2 = (cand @ Xcur.T).max(axis=1)
                y_star = cand[int(np.argmin(sc2))].copy()
                y_star = y_star / (np.linalg.norm(y_star) + 1e-300)
                # extra polish on winner
                y_star = void_single_refine(y_star, Xcur, steps=150, tau0=0.006, tau1=0.001, lr0=0.015, lr1=0.004)
            except Exception:
                pass
            new_pts.append(y_star)
            Xcur = np.vstack([Xcur, y_star])
            # remove used from pool
            pool = np.delete(pool, bi, axis=0)
            if pool.shape[0] == 0:
                # replenish with random refined
                try:
                    Yc2 = rng.standard_normal((50, d))
                    Yc2 = Yc2 / (np.linalg.norm(Yc2, axis=1, keepdims=True) + 1e-300)
                    pool = void_batch_refine(Yc2, Xcur, steps=80, tau0=0.015, tau1=0.002, lr0=0.03, lr1=0.005)
                except Exception:
                    break
        if not ok:
            continue
        # build full config
        Xnew = np.empty((n, d))
        Xnew[Kidx] = Xfrozen
        Xnew[R] = np.array(new_pts)
        Xnew = _norm(Xnew)
        vnew = _maxcos(Xnew)
        # ---- short annealed Adam polish (unfrozen) ----
        try:
            Xp, vp = full_adam_lse(Xnew, steps=450, tau0=0.008, tau1=0.0007, lr0=0.004, lr1=0.0009)
        except Exception:
            Xp, vp = Xnew, vnew
        # ---- tangent-LP polish ----
        try:
            Xlp, vlp = lp_polish(Xp, max_iters=5, topk=900)
        except Exception:
            Xlp, vlp = Xp, vp
        if vlp < best_val and np.all(np.isfinite(Xlp)):
            best_val = float(vlp)
            x_best = _norm(Xlp.copy())
        # also consider pre-LP if better
        if vp < best_val and np.all(np.isfinite(Xp)):
            best_val = float(vp)
            x_best = _norm(Xp.copy())

    # final safety: ensure at least base
    if best_val > base_val + 1e-12:
        x_best = base_best.copy()
        best_val = float(base_val)
    x_best = _norm(np.asarray(x_best, dtype=np.float64))
    # tiny shrink not needed; just return
    return x_best
