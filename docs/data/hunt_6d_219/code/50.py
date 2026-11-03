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

def _row_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1), float(G.max())

def solve():
    t0 = time.time()
    BUDGET = 168.0
    n, d = 219, 6
    iu0, iu1 = np.triu_indices(n, 1)
    rng = np.random.RandomState(7)
    rng_void = np.random.RandomState(1234)

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
    x_cur = x_best.copy()
    cur_val = best_val

    def lbfgs_full(X0, beta, maxiter):
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
                           options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-7, "maxfun": 20000})
            Y = res.x.reshape(n, d)
            return _norm(Y)
        except Exception:
            return _norm(X0)

    def lbfgs_frozen(X0, r_fix, beta, maxiter):
        # optimize 218 others, r_fix frozen
        idx = [i for i in range(n) if i != r_fix]
        idx = np.array(idx, dtype=int)
        m218 = len(idx)
        j0, j1 = np.triu_indices(m218, 1)
        Xr = X0[r_fix].copy()
        Xc0 = X0[idx].copy()
        y0 = Xc0.ravel().copy()
        def fun(y):
            Y = y.reshape(m218, d)
            rn = np.linalg.norm(Y, axis=1, keepdims=True)
            rn = np.maximum(rn, 1e-12)
            Xc = Y / rn
            G = Xc @ Xc.T
            g = G[j0, j1]
            m = float(np.max(g))
            e = np.exp(beta * (g - m))
            S = float(np.sum(e)) + 1e-300
            loss = m + np.log(S) / beta
            w = e / S
            W = np.zeros((m218, m218))
            W[j0, j1] = w
            W = W + W.T
            GX = W @ Xc
            dot = (GX * Xc).sum(axis=1, keepdims=True)
            GY = (GX - dot * Xc) / rn
            return float(loss), GY.ravel()
        try:
            res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                           options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-7, "maxfun": 15000})
            Y = res.x.reshape(m218, d)
            Xc = _norm(Y)
        except Exception:
            Xc = _norm(Xc0)
        Xfull = np.zeros((n, d))
        Xfull[idx] = Xc
        Xfull[r_fix] = Xr
        return _norm(Xfull)

    def lp_polish(X, deadline, r_init=1.0e-3, max_steps=12):
        X = _norm(X.copy())
        best = _max_cos(X)
        bx = X.copy()
        radius = r_init
        steps = 0
        while time.time() < deadline and steps < max_steps:
            steps += 1
            if radius < 1e-14:
                break
            Gf = X @ X.T
            c = Gf[iu0, iu1]
            m = float(c.max())
            thr = m - max(18.0 * radius, 1e-9)
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
            vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, n * d + 1)).tocsr()
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
            y = _norm(X + res.x[:-1].reshape(n, d))
            v = _max_cos(y)
            if v < best:
                X, best = y, v
                bx = y.copy()
                radius = min(radius * 1.5, 0.02)
            else:
                radius *= 0.5
        return bx, best

    def largest_void(core, n_rand=20000, beta_ref=400):
        # core: (218,6) normalized
        m = core.shape[0]
        # chunked sampling to bound memory
        best_v = None
        best_depth = 1e9
        cands_best = None
        # generate in chunks of 5000
        all_c = []
        all_r = []
        for _ in range(n_rand // 5000):
            C = rng_void.randn(5000, d)
            C = _norm(C)
            G = C @ core.T
            rm = G.max(axis=1)
            all_c.append(C)
            all_r.append(rm)
            k = int(np.argmin(rm))
            if float(rm[k]) < best_depth:
                best_depth = float(rm[k])
                best_v = C[k].copy()
        C_all = np.vstack(all_c)
        r_all = np.concatenate(all_r)
        order = np.argpartition(r_all, 6)[:6]
        # refine each top candidate by projected gradient with line search on true max
        refined_v = best_v.copy()
        refined_d = best_depth
        for ti in order:
            v = C_all[ti].copy()
            cur_rm = float(r_all[ti])
            for it in range(250):
                if time.time() - t0 > BUDGET - 12:
                    break
                dots = core @ v
                mm = float(np.max(dots))
                w = np.exp(beta_ref * (dots - mm))
                w = w / (np.sum(w) + 1e-300)
                grad = w @ core
                gt = grad - float(grad @ v) * v
                gn = float(np.linalg.norm(gt))
                if gn < 1e-14:
                    break
                direc = gt / gn
                improved = False
                for step in (0.03, 0.015, 0.007, 0.003):
                    vn = v - step * direc
                    vn = vn / max(np.linalg.norm(vn), 1e-12)
                    nm = float(np.max(core @ vn))
                    if nm < cur_rm - 1e-12:
                        v = vn
                        cur_rm = nm
                        improved = True
                        break
                if not improved:
                    # try smaller beta direction (mean)
                    break
            if cur_rm < refined_d:
                refined_d = cur_rm
                refined_v = v.copy()
        return refined_v, refined_d

    def rattler_index(X):
        rm, m = _row_max(X)
        return int(np.argmin(rm)), rm, m

    def degree_info(X, m, tol=1e-7):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        deg = np.sum(G > m - tol, axis=1)
        return deg, G

    # main rattler-first loop
    deltas = [1e-5, 2e-5, 5e-5, 1e-4]
    # map delta to pinch alpha (attraction strength)
    alphas = [0.015, 0.025, 0.04, 0.06]
    outer = 0
    # need at least one void search per outer; limit by time
    while time.time() - t0 < BUDGET - 30 and outer < 12:
        outer += 1
        r, rm_vec, m_cur = rattler_index(x_cur)
        # core max excluding r
        Gc = x_cur @ x_cur.T
        np.fill_diagonal(Gc, -2.0)
        mask = np.ones((n, n), dtype=bool)
        mask[r, :] = False; mask[:, r] = False
        core_max_before = float(np.max(Gc[mask].reshape(n - 1, n - 1)) if n > 1 else m_cur)
        # build core matrix
        core_idx = np.array([i for i in range(n) if i != r])
        core = x_cur[core_idx].copy()
        # largest void of core
        try:
            v_star, depth = largest_void(core, n_rand=20000)
        except Exception:
            continue
        # try delta sweep
        improved_this_outer = False
        for di, delta in enumerate(deltas):
            if time.time() - t0 > BUDGET - 28:
                break
            alpha = alphas[di % len(alphas)]
            X_try = x_cur.copy()
            X_try[r] = v_star
            # pinch K nearest neighbours toward r to force activation
            # find K=10 nearest to v_star among core
            dots = core @ v_star
            K = 10
            nn = np.argpartition(-dots, K - 1)[:K]
            # map back to full indices
            for k in nn:
                fi = core_idx[k]
                X_try[fi] = _norm(((1 - alpha) * X_try[fi] + alpha * v_star).reshape(1, -1))[0]
            X_try = _norm(X_try)
            # freeze r, optimise others (void expansion with protected void is mimicked
            # by core-only phase; then release)
            try:
                X_fro = lbfgs_frozen(X_try, r, beta=180.0, maxiter=35)
            except Exception:
                X_fro = X_try
            # release r: full optimisation anneal
            try:
                X_rel = lbfgs_full(X_fro, beta=220.0, maxiter=45)
                if time.time() - t0 < BUDGET - 25:
                    X_rel = lbfgs_full(X_rel, beta=550.0, maxiter=30)
            except Exception:
                continue
            # quick evaluate before expensive LP
            m_rel = _max_cos(X_rel)
            # LP polish only if promising (within 0.004 of best or better than cur)
            if m_rel > min(best_val, cur_val) + 0.004:
                continue
            try:
                X_pol, v_pol = lp_polish(X_rel, deadline=time.time() + 4.0, max_steps=10)
            except Exception:
                X_pol, v_pol = X_rel, m_rel
            # acceptance: r ends with >=5 contacts within 1e-7 and core max dropped
            try:
                deg, _ = degree_info(X_pol, v_pol, tol=1e-7)
                r_deg = int(deg[r])
                Gp = X_pol @ X_pol.T
                np.fill_diagonal(Gp, -2.0)
                mask2 = np.ones((n, n), dtype=bool)
                mask2[r, :] = False; mask2[:, r] = False
                core_max_after = float(np.max(Gp[mask2].reshape(n - 1, n - 1)))
            except Exception:
                continue
            integrated = (r_deg >= 5) and (core_max_after < core_max_before - 1e-9)
            if v_pol < best_val - 1e-12:
                x_best = X_pol.copy()
                best_val = v_pol
                # also move cur to new topology if integrated or better
                x_cur = X_pol.copy()
                cur_val = v_pol
                improved_this_outer = True
                break
            elif integrated and v_pol < cur_val + 0.0015:
                # accept topology change even if slightly worse to escape basin
                x_cur = X_pol.copy()
                cur_val = v_pol
                improved_this_outer = True
                # continue sweeping from new cur in next outer (break delta loop)
                break
            elif v_pol < cur_val - 2e-7:
                x_cur = X_pol.copy()
                cur_val = v_pol
        if not improved_this_outer:
            # random kick to avoid re-finding same void: perturb cur slightly tangentially
            if time.time() - t0 < BUDGET - 30:
                try:
                    pert = rng.randn(n, d) * 0.015
                    # project tangent
                    pert = pert - (pert * x_cur).sum(axis=1, keepdims=True) * x_cur
                    # don't move too much: keep best fallback, but diversify cur
                    # kick worst 20 points harder
                    rm2, _ = _row_max(x_cur)
                    # points with large row max are overloaded
                    order_w = np.argsort(-rm2)[:20]
                    pert[order_w] *= 3.0
                    x_cur = _norm(x_cur + pert)
                    x_cur = lbfgs_full(x_cur, beta=180.0, maxiter=30)
                    cur_val = _max_cos(x_cur)
                    if cur_val < best_val:
                        x_best = x_cur.copy()
                        best_val = cur_val
                    # if kick made cur much worse, revert to best
                    if cur_val > best_val + 0.003:
                        x_cur = x_best.copy()
                        cur_val = best_val
                except Exception:
                    pass

    # final squeeze on global best
    if time.time() - t0 < BUDGET - 8:
        try:
            Xf = lbfgs_full(x_best, beta=300.0, maxiter=35)
            vf = _max_cos(Xf)
            if vf < best_val:
                x_best, best_val = Xf, vf
            Xp, vp = lp_polish(x_best, deadline=t0 + BUDGET - 1.5, max_steps=20)
            if vp < best_val:
                x_best, best_val = Xp, vp
        except Exception:
            pass
    x_best = _norm(x_best)
    return x_best
