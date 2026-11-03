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
    BUDGET = 166.0
    n, d = 219, 6
    iu0, iu1 = np.triu_indices(n, 1)

    # load incumbent
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
        rng0 = np.random.RandomState(0)
        x_best = _norm(rng0.randn(n, d))
        best_val = _max_cos(x_best)

    def contact_deg(X, m, tol=1e-9):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        return np.sum(G > m - tol, axis=1), G

    def lp_polish(x, deadline, r_init=1.1e-3, max_steps=14):
        x = _norm(x.copy())
        best = _max_cos(x)
        bx = x.copy()
        radius = r_init
        steps = 0
        while time.time() < deadline and time.time() - t0 < BUDGET - 1.0 and steps < max_steps:
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

    def run_full_lbfgs(x0, beta, maxiter):
        y0 = x0.ravel().copy()
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
            return _norm(x0)

    def run_core_lbfgs(x_full, r_idx, beta, maxiter, delta=None, k_neigh=12, barrier_coeff=2e4):
        # freeze point r_idx, optimise other 218 with optional barrier <xr,xi> <= m_core - delta
        c_idx = np.array([i for i in range(n) if i != r_idx], dtype=int)
        nc = n - 1
        xr = x_full[r_idx].copy() / max(1e-12, np.linalg.norm(x_full[r_idx]))
        Xc0 = x_full[c_idx].copy()
        ci0, ci1 = np.triu_indices(nc, 1)
        y0 = Xc0.ravel().copy()
        def fun(y):
            Y = y.reshape(nc, d)
            rn = np.linalg.norm(Y, axis=1, keepdims=True)
            rn = np.maximum(rn, 1e-12)
            Xc = Y / rn
            G = Xc @ Xc.T
            g = G[ci0, ci1]
            m = float(np.max(g))
            e = np.exp(beta * (g - m))
            S = float(np.sum(e)) + 1e-300
            loss = m + np.log(S) / beta
            w = e / S
            W = np.zeros((nc, nc))
            W[ci0, ci1] = w
            W = W + W.T
            GX = W @ Xc
            if delta is not None:
                rv = Xc @ xr  # nc
                # only nearest k matter for barrier (largest rv)
                if k_neigh is not None and k_neigh < nc:
                    part = np.argpartition(-rv, k_neigh - 1)[:k_neigh]
                else:
                    part = np.arange(nc)
                viol = rv[part] - (m - delta)
                pos = np.maximum(viol, 0.0)
                pen = barrier_coeff * float(np.sum(pos ** 2))
                loss = loss + pen
                # grad of pen wrt Xc[part]: 2*coeff*pos * xr (threshold fixed)
                Gp = np.zeros((nc, d))
                Gp[part] = (2.0 * barrier_coeff * pos)[:, None] * xr[None, :]
                GX = GX + Gp
            dot = (GX * Xc).sum(axis=1, keepdims=True)
            GY = (GX - dot * Xc) / rn
            return float(loss), GY.ravel()
        try:
            res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                           options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-7, "maxfun": 20000})
            Y = res.x.reshape(nc, d)
            Xc = _norm(Y)
        except Exception:
            Xc = _norm(Xc0)
        Xnew = x_full.copy()
        Xnew[c_idx] = Xc
        Xnew[r_idx] = xr
        return _norm(Xnew)

    def find_void(core_X, rng, n_rand=20000):
        cand = _norm(rng.randn(n_rand, d))
        scores = cand @ core_X.T
        mx = scores.max(axis=1)
        ib = int(np.argmin(mx))
        return cand[ib], float(mx[ib])

    def refine_void(v0, core_X, iters=350, beta0=60.0, beta1=600.0):
        v = v0.copy() / max(1e-12, np.linalg.norm(v0))
        best_v = v.copy()
        G0 = core_X @ v
        best_m = float(np.max(G0))
        lr = 0.05
        for it in range(iters):
            frac = it / max(1, iters - 1)
            beta = beta0 * (beta1 / beta0) ** frac
            G = core_X @ v
            m = float(np.max(G))
            e = np.exp(beta * (G - m))
            S = float(np.sum(e)) + 1e-300
            w = e / S
            grad = w @ core_X  # 6-vector, direction to decrease: move away from weighted neighbours
            # we minimise max cosine -> step opposite grad
            gt = grad - float(grad @ v) * v
            gn = float(np.linalg.norm(gt))
            if gn < 1e-14 or not np.finite(gn):
                lr *= 0.7
                continue
            step = (lr / (1.0 + 8.0 * frac)) * gt / gn
            # adaptive: try step, accept if improves or with small tolerance
            vn = _norm((v - step)[None, :])[0]
            mn = float(np.max(core_X @ vn))
            if mn < m + 1e-12:
                v = vn
                if mn < best_m:
                    best_m = mn
                    best_v = vn.copy()
                if it % 60 == 59:
                    lr *= 0.9
            else:
                lr *= 0.6
                if lr < 1e-5:
                    lr = 0.01
        return best_v, best_m

    # identify initial rattler
    Gm = x_best @ x_best.T
    np.fill_diagonal(Gm, -2.0)
    mall = float(np.max(Gm))
    degs, _ = contact_deg(x_best, mall, tol=1e-9)
    r0 = int(np.argmin(degs))

    x_cur = x_best.copy()
    cur_val = best_val
    # core max baseline
    def core_max_of(X, r):
        C = np.delete(X, r, axis=0)
        return _max_cos(C)

    deltas = [1e-4, 5e-5, 2e-5, 1e-5]
    attempt = 0
    rng_void = np.random.RandomState(777)
    # store initial core max for acceptance
    while time.time() - t0 < BUDGET - 22 and attempt < 9:
        attempt += 1
        # re-identify rattler of current best/cur
        Gc = x_cur @ x_cur.T
        np.fill_diagonal(Gc, -2.0)
        mc = float(np.max(Gc))
        degs_c, _ = contact_deg(x_cur, mc, tol=1e-9)
        r = int(np.argmin(degs_c))
        # if no rattler (all >=1), pick min row-max (loosest point) as pseudo-rattler
        # also alternate: cycle through a few loosest to diversify
        if attempt % 3 == 0:
            rowmax = Gc.max(axis=1)
            order = np.argsort(rowmax)  # smallest rowmax = loosest
            r = int(order[(attempt // 3) % min(n, 5)])
        c_idx = np.array([i for i in range(n) if i != r])
        core_X = x_cur[c_idx]
        base_core_max = _max_cos(core_X)
        # largest void search: 20000 random + refinement
        v_raw, _ = find_void(core_X, rng_void, n_rand=20000)
        # try top candidates refinement: perturb raw a few times
        v_best, m_best = refine_void(v_raw, core_X, iters=300)
        # extra local samples around v_best
        for _ in range(2):
            if time.time() - t0 > BUDGET - 20:
                break
            cand2 = _norm((v_best[None, :] + 0.02 * np.random.RandomState(1000 + attempt * 17 + _).randn(4, d)).reshape(4, d))
            for k in range(4):
                vv, mm = refine_void(cand2[k], core_X, iters=120)
                if mm < m_best:
                    m_best = mm
                    v_best = vv
        x_try = x_cur.copy()
        x_try[r] = v_best / max(1e-12, np.linalg.norm(v_best))
        # small tangential noise to core to leave LP vertex
        rng_n = np.random.RandomState(2000 + attempt * 101)
        noise = rng_n.randn(n, d) * 0.0012
        # project noise tangent
        noise -= (noise * x_try).sum(axis=1, keepdims=True) * x_try
        # don't move r
        noise[r] = 0.0
        x_try = _norm(x_try + noise)
        # restore r exactly to void
        x_try[r] = v_best / max(1e-12, np.linalg.norm(v_best))

        for di, delta in enumerate(deltas):
            if time.time() - t0 > BUDGET - 18:
                break
            # frozen-core optimisation with barrier delta, k=10 nearest
            x_try = run_core_lbfgs(x_try, r, beta=380.0, maxiter=60, delta=delta, k_neigh=10)
            # release r: short full optimisation
            x_rel = run_full_lbfgs(x_try, beta=260.0, maxiter=22)
            # evaluate
            m_all = _max_cos(x_rel)
            # r contacts within 1e-7
            rv = x_rel[r] @ x_rel.T
            rv[r] = -2.0
            cnt_r = int(np.sum(rv > m_all - 1e-7))
            Cc = np.delete(x_rel, r, axis=0)
            cmax = _max_cos(Cc)
            # acceptance: r integrated and core improved (or overall improved)
            if cnt_r >= 5 and (cmax < base_core_max - 1e-9 or m_all < cur_val):
                # finish with stronger full + LP polish
                x_rel2 = run_full_lbfgs(x_rel, beta=550.0, maxiter=30)
                m2 = _max_cos(x_rel2)
                # recount
                rv2 = x_rel2[r] @ x_rel2.T
                rv2[r] = -2.0
                cnt2 = int(np.sum(rv2 > m2 - 1e-7))
                if cnt2 >= 5 or m2 < cur_val:
                    try:
                        xp, vp = lp_polish(x_rel2, deadline=time.time() + 5.0, max_steps=12)
                    except Exception:
                        xp, vp = x_rel2, m2
                    if vp < best_val and cnt2 >= 1:
                        x_best = xp.copy()
                        best_val = vp
                        x_cur = xp.copy()
                        cur_val = vp
                        break
                    elif vp < cur_val + 5e-6 and cnt2 >= 5:
                        # accept topology even if slightly worse to continue exploring
                        x_cur = xp.copy()
                        cur_val = vp
                        if vp < best_val:
                            x_best = xp.copy()
                            best_val = vp
                else:
                    x_try = x_rel.copy()
            else:
                # keep evolving frozen state (use released as next start)
                # only keep if not exploding
                if m_all < cur_val + 0.004:
                    x_try = x_rel.copy()
                # else keep frozen without release
            # early exit if we beat best nicely: continue to next void
            if best_val < cur_val - 1e-12:
                x_cur = x_best.copy()
                cur_val = best_val
                break
        # if this attempt produced a good x_try with integrated rattler but not yet polished, polish it
        if time.time() - t0 < BUDGET - 14:
            try:
                m_try = _max_cos(x_try)
                rvv = x_try[r] @ x_try.T
                rvv[r] = -2.0
                cnt_try = int(np.sum(rvv > m_try - 1e-7))
                if cnt_try >= 5 and m_try < best_val + 0.002:
                    x_p2 = run_full_lbfgs(x_try, beta=500.0, maxiter=28)
                    mp2 = _max_cos(x_p2)
                    xp2, vp2 = lp_polish(x_p2, deadline=time.time() + 4.0, max_steps=10)
                    if vp2 < best_val:
                        x_best = xp2.copy()
                        best_val = vp2
                        x_cur = xp2.copy()
                        cur_val = vp2
                    elif vp2 < cur_val and cnt_try >= 5:
                        x_cur = xp2.copy()
                        cur_val = vp2
            except Exception:
                pass

    # final squeeze on global best
    if time.time() - t0 < BUDGET - 6:
        try:
            Xf = run_full_lbfgs(x_best, beta=320.0, maxiter=35)
            vf = _max_cos(Xf)
            if vf < best_val:
                x_best, best_val = Xf, vf
            Xp, vp = lp_polish(x_best, deadline=t0 + BUDGET - 1.0, max_steps=20)
            if vp < best_val:
                x_best, best_val = Xp, vp
        except Exception:
            pass
    x_best = _norm(x_best)
    return x_best
