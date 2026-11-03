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
    rng = np.random.RandomState(42)

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
        x_best = _norm(rng.randn(n, d))
        best_val = _max_cos(x_best)

    def row_max(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        return G.max(axis=1), float(G.max())

    def core_max_excl(X, r):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        mask = np.ones(n, dtype=bool); mask[r] = False
        # max over pairs with both != r
        Gc = G[np.ix_(mask, mask)]
        np.fill_diagonal(Gc, -2.0)
        return float(Gc.max())

    def contacts_of(X, m, tol=1e-7):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        return np.sum(G > m - tol, axis=1)

    def free_lbfgs(X0, beta, maxiter):
        y0 = X0.ravel().copy()
        def fun(y):
            Y = y.reshape(n, d)
            rn = np.linalg.norm(Y, axis=1, keepdims=True)
            rn = np.maximum(rn, 1e-12)
            X = Y / rn
            G = X @ X.T
            g = G[iu0, iu1]
            m = float(g.max())
            e = np.exp(beta * (g - m))
            S = float(e.sum()) + 1e-300
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
                           options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-7, "maxfun": 20000})
            Y = res.x.reshape(n, d)
            return _norm(Y)
        except Exception:
            return _norm(X0)

    def frozen_lbfgs(X0, r, nn_idx, delta, beta, maxiter):
        # freeze row r, optimise others with augmented target <r,i>+delta
        free_idx = [i for i in range(n) if i != r]
        nfree = len(free_idx)
        free_pos = np.zeros(n, dtype=int)
        for k, i in enumerate(free_idx):
            free_pos[i] = k
        # core pair indices in terms of free positions handled via full matrix
        Xr_fixed = X0[r].copy()
        Y0free = X0[free_idx].copy()
        y0 = Y0free.ravel().copy()
        # precompute iu for core-core in full indexing for weight fill
        def fun(y):
            Yf = y.reshape(nfree, d)
            rnf = np.linalg.norm(Yf, axis=1, keepdims=True)
            rnf = np.maximum(rnf, 1e-12)
            Xf_n = Yf / rnf
            Xfull = np.zeros((n, d))
            Xfull[r] = Xr_fixed
            Xfull[free_idx] = Xf_n
            G = Xfull @ Xfull.T
            # core-core pairs (both free)
            # gather via iu mask
            # faster: extract submatrix
            # use full iu but mask out pairs involving r
            g_all = G[iu0, iu1]
            involves_r = (iu0 == r) | (iu1 == r)
            g_core = g_all[~involves_r]
            m_core = float(g_core.max())
            # r-nn values with offset
            gr = G[r, nn_idx] + delta
            m_aug = m_core if m_core > float(gr.max()) else float(gr.max())
            e_core = np.exp(beta * (g_core - m_aug))
            e_r = np.exp(beta * (gr - m_aug))
            S = float(e_core.sum() + e_r.sum()) + 1e-300
            loss = m_aug + np.log(S) / beta
            w_core = e_core / S
            w_r = e_r / S
            # build GX for free rows
            W = np.zeros((n, n))
            # fill core weights
            c0 = iu0[~involves_r]; c1 = iu1[~involves_r]
            W[c0, c1] = w_core
            W = W + W.T
            # r-nn weights: W[r,i]=W[i,r]=w_r
            for k2, idx in enumerate(nn_idx):
                W[r, idx] = w_r[k2]
                W[idx, r] = w_r[k2]
            GX = W @ Xfull
            # grad w.r.t Xf_n for free rows
            GXf = GX[free_idx]
            dot = (GXf * Xf_n).sum(axis=1, keepdims=True)
            GYf = (GXf - dot * Xf_n) / rnf
            return float(loss), GYf.ravel()
        try:
            res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                           options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-7, "maxfun": 15000})
            Yf = res.x.reshape(nfree, d)
            Xout = np.zeros((n, d))
            Xout[r] = Xr_fixed
            Xout[free_idx] = Yf
            return _norm(Xout)
        except Exception:
            return _norm(X0)

    def lp_polish(x, deadline, r_init=1.1e-3, max_steps=14):
        x = _norm(x.copy())
        best = _max_cos(x)
        bx = x.copy()
        radius = r_init
        steps = 0
        while time.time() < deadline and time.time() - t0 < BUDGET - 1.5 and steps < max_steps:
            steps += 1
            if radius < 1e-14:
                break
            Gf = x @ x.T
            c = Gf[iu0, iu1]
            m = float(c.max())
            thr = m - max(20.0 * radius, 1e-9)
            sel = c > thr
            I = iu0[sel]; J = iu1[sel]; csel = c[sel]
            p = len(I)
            if p == 0:
                break
            if p > 3500:
                k2 = 3500
                part = np.argpartition(-csel, k2 - 1)[:k2]
                I, J, csel = I[part], J[part], csel[part]
                p = k2
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

    def largest_void(core, rng_local, n_rand=20000):
        # core: (m,6)
        cand = rng_local.randn(n_rand, d)
        cand = cand / np.linalg.norm(cand, axis=1, keepdims=True)
        dots = cand @ core.T
        maxd = dots.max(axis=1)
        order = np.argsort(maxd)
        top = order[:6]
        best_v = cand[top[0]].copy()
        best_m = float(maxd[top[0]])
        # refine top candidates
        for ti in top[:4]:
            v = cand[ti].copy()
            lr0 = 0.06
            for it in range(350):
                if time.time() - t0 > BUDGET - 5:
                    break
                dd = core @ v
                mm = float(dd.max())
                # softmax weights
                w = np.exp(35.0 * (dd - mm))
                w = w / (w.sum() + 1e-300)
                grad = w @ core
                gt = grad - (float(grad @ v)) * v
                gn = float(np.linalg.norm(gt))
                if gn < 1e-12:
                    break
                step = lr0 / (1.0 + it * 0.02)
                v = v - (step / (gn + 1e-12)) * gt * gn  # ~ step*gt normalized
                # simpler: v - step*gt
                # redo correctly:
                # we already scaled; use plain
                # recompute with plain step for stability:
                # (undo above scaling confusion: use step*gt)
                # To avoid double scaling, just do v = norm(v - step*gt)
                # Our v currently wrong, fix by re-deriving: we did v-step*gt*(gn/gn)? Actually (step/gn)*gt*gn = step*gt, ok same.
                v = v / (np.linalg.norm(v) + 1e-300)
                # track
                # evaluate occasionally
                if it % 25 == 24:
                    mm2 = float((core @ v).max())
                    if mm2 < best_m:
                        best_m = mm2
                        best_v = v.copy()
            mmf = float((core @ v).max())
            if mmf < best_m:
                best_m = mmf
                best_v = v.copy()
        return best_v / (np.linalg.norm(best_v) + 1e-300)

    # main rattler-first homotopy
    x_cur = x_best.copy()
    cur_val = best_val
    # identify initial rattler
    rm, _ = row_max(x_cur)
    r = int(np.argmin(rm))

    deltas = [1e-5, 2e-5, 5e-5, 1e-4]
    cycle = 0
    rng_void = np.random.RandomState(7)
    rng_pert = np.random.RandomState(11)

    while time.time() - t0 < BUDGET - 22 and cycle < 8:
        cycle += 1
        # re-identify rattler of x_cur
        rm_cur, m_cur = row_max(x_cur)
        r = int(np.argmin(rm_cur))
        core_m_before = core_max_excl(x_cur, r)
        # ---- void move: place r at largest void of core ----
        core_pts = np.delete(x_cur, r, axis=0)
        try:
            v_star = largest_void(core_pts, rng_void, n_rand=20000)
        except Exception:
            v_star = x_cur[r].copy()
        x_void = x_cur.copy()
        x_void[r] = v_star
        # settle void-inserted config briefly
        try:
            x_set = free_lbfgs(x_void, 220.0, maxiter=45)
        except Exception:
            x_set = x_void
        v_set = _max_cos(x_set)
        if v_set < best_val - 1e-12:
            # polish and take
            try:
                xp, vp = lp_polish(x_set, deadline=time.time() + 4.0)
                if vp < v_set:
                    x_set, v_set = xp, vp
            except Exception:
                pass
            if v_set < best_val:
                x_best, best_val = x_set.copy(), v_set
        # allow x_cur to follow void move if not too bad (explore new basin)
        # accept if within 3e-4 of best and core improved or rattler activated
        rm_s, m_s = row_max(x_set)
        degs = contacts_of(x_set, m_s, tol=1e-7)
        # core max after
        # find new rattler index for core measure
        r2 = int(np.argmin(rm_s))
        cm_after = core_max_excl(x_set, r2)
        if v_set < cur_val + 3e-4:
            # prefer it as new exploration point (diversify)
            x_cur, cur_val = x_set.copy(), v_set
            r = r2
        if time.time() - t0 > BUDGET - 22:
            break
        # ---- forced-activation sweeps with r frozen ----
        for di, delta in enumerate(deltas):
            if time.time() - t0 > BUDGET - 20:
                break
            # nearest neighbours of r
            Gc = x_cur @ x_cur.T
            Gc[r, r] = -2.0
            nn = np.argsort(-Gc[r])[:10]  # 10 nearest
            # optional small tangential noise to core to help topology change
            x_start = x_cur.copy()
            if di > 0:
                nz = rng_pert.randn(n, d)
                nz -= (nz * x_start).sum(1, keepdims=True) * x_start
                nz[r] = 0.0
                nnz = np.linalg.norm(nz, axis=1, keepdims=True) + 1e-12
                x_start = _norm(x_start + 0.004 * nz / nnz)
                x_start[r] = x_cur[r]  # keep frozen point exact
            try:
                x_froz = frozen_lbfgs(x_start, r, nn, delta, beta=320.0, maxiter=55)
            except Exception:
                continue
            # release r
            try:
                x_rel = free_lbfgs(x_froz, 300.0, maxiter=35)
            except Exception:
                x_rel = x_froz
            v_rel = _max_cos(x_rel)
            rm_r, _ = row_max(x_rel)
            # contacts of original r (track same index; after release index still r)
            degs_r = contacts_of(x_rel, v_rel, tol=1e-7)
            rc = int(degs_r[r])
            cm_rel = core_max_excl(x_rel, r)
            # acceptance for exploration: core dropped and r has >=5 contacts
            core_dropped = cm_rel < core_m_before - 1e-9
            topology_ok = (rc >= 5) and core_dropped
            # polish if promising or topology_ok
            if topology_ok or v_rel < best_val + 5e-4:
                try:
                    # quick polish only if close to best to save time, else polish topology_ok
                    if topology_ok or v_rel < best_val + 1e-4:
                        xp2, vp2 = lp_polish(x_rel, deadline=time.time() + 3.5, max_steps=12)
                        if vp2 < v_rel:
                            x_rel, v_rel = xp2, vp2
                            degs_r = contacts_of(x_rel, v_rel, tol=1e-7)
                            rc = int(degs_r[r])
                except Exception:
                    pass
            if v_rel < best_val - 1e-12:
                x_best, best_val = x_rel.copy(), v_rel
            # update exploration state
            if topology_ok:
                x_cur, cur_val = x_rel.copy(), v_rel
                core_m_before = core_max_excl(x_cur, r)
                # re-identify rattler after successful activation (may have changed)
                rm_new, _ = row_max(x_cur)
                r = int(np.argmin(rm_new))
            elif v_rel < cur_val - 2e-6:
                x_cur, cur_val = x_rel.copy(), v_rel
            # early exit if big win
            if time.time() - t0 > BUDGET - 20:
                break
        # if stuck, perturb exploration point more strongly but keep best fallback
        if cur_val > best_val + 2e-4:
            # reset exploration to best with small noise to try new basin near best
            nz = rng_pert.randn(n, d)
            nz -= (nz * x_best).sum(1, keepdims=True) * x_best
            nz = nz / (np.linalg.norm(nz, axis=1, keepdims=True) + 1e-12)
            x_cur = _norm(x_best + 0.01 * nz)
            cur_val = _max_cos(x_cur)

    # final squeeze on incumbent
    if time.time() - t0 < BUDGET - 8:
        try:
            Xf = free_lbfgs(x_best, 350.0, maxiter=40)
            vf = _max_cos(Xf)
            if vf < best_val:
                x_best, best_val = Xf, vf
        except Exception:
            pass
    if time.time() - t0 < BUDGET - 2:
        try:
            Xp, vp = lp_polish(x_best, deadline=t0 + BUDGET - 0.8, max_steps=22)
            if vp < best_val:
                x_best, best_val = Xp, vp
        except Exception:
            pass
    x_best = _norm(x_best)
    return x_best
