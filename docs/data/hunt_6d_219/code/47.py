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
    rm = G.max(axis=1)
    return rm, float(rm.max())

def _deg(X, m, tol):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return np.sum(G > m - tol, axis=1)

def solve():
    t0 = time.time()
    BUDGET = 168.0
    n, d = 219, 6
    iu0, iu1 = np.triu_indices(n, 1)
    rng = np.random.RandomState(7)

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

    # ---------- soft-max LBFGS (full) ----------
    def run_lbfgs(X0, beta, maxiter):
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

    # ---------- soft-max LBFGS on core only (r frozen / removed) ----------
    def run_core_lbfgs(C0, beta, maxiter, r_vec=None, t_cur=None, delta=0.0, neigh=None):
        nc = C0.shape[0]
        cu0, cu1 = np.triu_indices(nc, 1)
        y0 = C0.ravel().copy()
        def fun(y):
            Y = y.reshape(nc, d)
            rn = np.linalg.norm(Y, axis=1, keepdims=True)
            rn = np.maximum(rn, 1e-12)
            C = Y / rn
            G = C @ C.T
            g = G[cu0, cu1]
            m = float(np.max(g))
            e = np.exp(beta * (g - m))
            S = float(np.sum(e)) + 1e-300
            loss = m + np.log(S) / beta
            # extra target: <x_r,x_i> <= t-delta for nearest neighbours
            # penalty active only if violated; keeps void around frozen r
            if r_vec is not None and neigh is not None and t_cur is not None:
                rv = r_vec / max(np.linalg.norm(r_vec), 1e-12)
                dots = C[neigh] @ rv  # k
                thr = t_cur - delta
                viol = dots - thr
                # softplus-like penalty
                pen_beta = 800.0
                pm = float(np.max(viol)) if len(viol) else -1e9
                ep = np.exp(pen_beta * (viol - pm))
                Sp = float(np.sum(ep)) + 1e-300
                softm = pm + np.log(Sp) / pen_beta
                if softm > 0:
                    loss = loss + 2.0 * softm
                    wpen = (ep / Sp) * 2.0  # k
                    # gradient contrib to C[neigh]
                    Gpen = np.zeros_like(C)
                    for a, idx in enumerate(neigh):
                        Gpen[idx] += wpen[a] * rv
                    # chain to Y
                    dotp = (Gpen * C).sum(axis=1, keepdims=True)
                    GYpen = (Gpen - dotp * C) / rn
                else:
                    GYpen = np.zeros_like(Y)
            else:
                GYpen = np.zeros_like(Y)
            w = e / S
            W = np.zeros((nc, nc))
            W[cu0, cu1] = w
            W = W + W.T
            GX = W @ C
            dot = (GX * C).sum(axis=1, keepdims=True)
            GY = (GX - dot * C) / rn + GYpen
            return float(loss), GY.ravel()
        try:
            res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                           options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-7, "maxfun": 20000})
            Y = res.x.reshape(nc, d)
            return _norm(Y)
        except Exception:
            return _norm(C0)

    def lp_polish(X, deadline, r_init=1.0e-3, max_steps=10):
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

    # ---------- largest void search + refinement ----------
    def largest_voids(core, rng_loc, n_cand=20000, top=4):
        Cand = _norm(rng_loc.randn(n_cand, d))
        S = Cand @ core.T  # n_cand x nc
        mx = S.max(axis=1)
        idx = np.argpartition(mx, top - 1)[:top]
        # sort by mx
        order = np.argsort(mx[idx])
        idx = idx[order]
        return Cand[idx], mx[idx]

    def refine_void(v0, core, iters=400):
        v = v0 / max(np.linalg.norm(v0), 1e-12)
        m_best = float(np.max(core @ v))
        v_best = v.copy()
        lr = 0.08
        tau = 0.004
        for _ in range(iters):
            dots = core @ v
            m = float(np.max(dots))
            w = np.exp((dots - m) / tau)
            w = w / (np.sum(w) + 1e-300)
            grad = w @ core
            gt = grad - float(grad @ v) * v
            gn = float(np.linalg.norm(gt))
            if gn < 1e-13:
                tau *= 0.7
                if tau < 1e-5:
                    break
                continue
            step = (lr / gn) * gt
            vn = v - step
            vn = vn / max(np.linalg.norm(vn), 1e-12)
            mn = float(np.max(core @ vn))
            if mn < m:
                v = vn
                if mn < m_best:
                    m_best = mn
                    v_best = vn.copy()
            else:
                lr *= 0.7
                if lr < 1e-6:
                    tau *= 0.6
                    lr = 0.02
                    if tau < 2e-4:
                        break
        return v_best, m_best

    # identify rattler
    def find_rattler(X):
        rm, m = _row_max(X)
        deg = _deg(X, m, 1e-9)
        r = int(np.argmin(rm))  # lowest row-max
        # prefer exact zero-contact if exists
        z = np.where(deg == 0)[0]
        if len(z) > 0:
            # pick lowest among them
            r = int(z[np.argmin(rm[z])])
        return r, rm, m, deg

    Xcur = x_best.copy()
    cur_val = best_val

    deltas = [1e-5, 2e-5, 5e-5, 1e-4]
    rng_void = np.random.RandomState(1234)
    cyc = 0
    # main rattler-first homotopy loop
    while time.time() - t0 < BUDGET - 30:
        cyc += 1
        r, rm, mcur, deg = find_rattler(Xcur)
        # nearest 8-12 neighbours of r
        G = Xcur @ Xcur[r]
        G[r] = -2.0
        neigh = np.argsort(-G)[:10]
        core_idx = [i for i in range(n) if i != r]
        core = Xcur[core_idx]
        r_vec = Xcur[r].copy()
        improved_this_cycle = False

        for di, delta in enumerate(deltas):
            if time.time() - t0 > BUDGET - 30:
                break
            # 1) freeze r, re-optimise core with extra target <= t-delta
            Copt = run_core_lbfgs(core, beta=220.0, maxiter=45,
                                  r_vec=r_vec, t_cur=mcur, delta=delta, neigh=neigh)
            # 2) move r to refined largest void of new core
            cands, _ = largest_voids(Copt, rng_void, n_cand=20000, top=4)
            best_trial = None
            best_trial_val = 1e9
            for ci in range(cands.shape[0]):
                if time.time() - t0 > BUDGET - 28:
                    break
                vref, vm = refine_void(cands[ci], Copt, iters=250)
                Xt = np.zeros((n, d))
                Xt[core_idx] = Copt
                Xt[r] = vref
                Xt = _norm(Xt)
                # small tangent noise on core to leave basin (deterministic)
                if cyc % 2 == 0:
                    nz = rng_void.randn(n, d) * 0.004
                    nz -= (nz * Xt).sum(axis=1, keepdims=True) * Xt
                    Xt = _norm(Xt + nz)
                # 3) release r: joint optimisation
                Xj = run_lbfgs(Xt, beta=200.0, maxiter=55)
                # quick second sharpen
                Xj2 = run_lbfgs(Xj, beta=500.0, maxiter=25)
                if _max_cos(Xj2) < _max_cos(Xj):
                    Xj = Xj2
                vj = _max_cos(Xj)
                # contact check for r
                rmj, mj = _row_max(Xj)
                degj = _deg(Xj, mj, 1e-7)
                rdeg = int(degj[r])
                # core max (excluding r)
                Gc = Xj[core_idx] @ Xj[core_idx].T
                np.fill_diagonal(Gc, -2.0)
                core_max = float(np.max(Gc))
                # acceptance: r integrated AND core dropped AND global improved-ish
                # we polish only promising ones
                promising = (rdeg >= 5 and core_max < mcur - 1e-7) or (vj < cur_val)
                if promising:
                    Xp, vp = lp_polish(Xj, deadline=time.time() + 2.5, max_steps=7)
                    if vp >= vj and vj < best_trial_val:
                        best_trial, best_trial_val = Xj, vj
                    elif vp < best_trial_val:
                        best_trial, best_trial_val = Xp, vp
                    # check final degree
                    _, mp = _row_max(best_trial) if best_trial is Xp or best_trial is Xj else (None, None)
                else:
                    if vj < best_trial_val:
                        best_trial, best_trial_val = Xj, vj
            if best_trial is not None and best_trial_val < cur_val - 2e-9:
                # verify rattler integration for accepted topology, else still accept if better
                _, mbt = _row_max(best_trial)
                degbt = _deg(best_trial, mbt, 1e-7)
                # finish accepted topology with standard polish
                Xp2, vp2 = lp_polish(best_trial, deadline=time.time() + 3.0, max_steps=10)
                cand_best = Xp2 if vp2 < best_trial_val else best_trial
                cand_val = min(vp2, best_trial_val)
                if cand_val < cur_val:
                    Xcur = cand_best
                    cur_val = cand_val
                    mcur = cand_val
                    core = Xcur[[i for i in range(n) if i != r]]
                    improved_this_cycle = True
                    if cand_val < best_val:
                        x_best = cand_best.copy()
                        best_val = cand_val
                    # update neigh/r for next delta
                    r2, rm2, _, _ = find_rattler(Xcur)
                    # keep same r index semantics: recompute
                    r = r2
                    core_idx = [i for i in range(n) if i != r]
                    core = Xcur[core_idx]
                    r_vec = Xcur[r].copy()
                    G2 = Xcur @ Xcur[r]; G2[r] = -2.0
                    neigh = np.argsort(-G2)[:10]
                # if integrated (>=5 contacts) we can break to re-find rattler
                if int(degbt[r]) >= 5:
                    break
        if not improved_this_cycle:
            # fallback diversification: random kill-and-fill of worst cluster then joint opt
            if time.time() - t0 > BUDGET - 30:
                break
            rmw, _ = _row_max(Xcur)
            worst = int(np.argmax(rmw))
            # replace worst + rattler by two fresh voids of remaining core
            keep = [i for i in range(n) if i not in (worst, r)]
            kc = Xcur[keep]
            cands, _ = largest_voids(kc, rng_void, n_cand=15000, top=2)
            Xt = Xcur.copy()
            v0, _ = refine_void(cands[0], kc, iters=200)
            Xt[worst] = v0
            if len(cands) > 1:
                v1, _ = refine_void(cands[1], kc, iters=200)
                Xt[r] = v1
            else:
                Xt[r] = _norm(rng_void.randn(d))
            Xt = _norm(Xt)
            Xj = run_lbfgs(Xt, beta=220.0, maxiter=50)
            vj = _max_cos(Xj)
            if vj < cur_val:
                Xp, vp = lp_polish(Xj, deadline=time.time() + 2.5, max_steps=7)
                if vp < cur_val:
                    Xcur, cur_val = Xp, vp
                    if vp < best_val:
                        x_best, best_val = Xp.copy(), vp
                elif vj < cur_val:
                    Xcur, cur_val = Xj, vj
                    if vj < best_val:
                        x_best, best_val = Xj.copy(), vj
            else:
                # small random tangent kick to Xcur to try new basin next cycle
                nz = rng_void.randn(n, d) * 0.006
                nz -= (nz * Xcur).sum(axis=1, keepdims=True) * Xcur
                Xcur = _norm(Xcur + nz)
                Xcur = run_lbfgs(Xcur, beta=180.0, maxiter=40)
                vc = _max_cos(Xcur)
                if vc < cur_val:
                    cur_val = vc
                    if vc < best_val:
                        x_best, best_val = Xcur.copy(), vc
                else:
                    Xcur = x_best.copy()
                    cur_val = best_val
        # keep incumbent as fallback
        if cur_val > best_val + 1e-6:
            Xcur = x_best.copy()
            cur_val = best_val

    # final thorough polish of incumbent
    try:
        Xf = run_lbfgs(x_best, beta=300.0, maxiter=40)
        if _max_cos(Xf) < best_val:
            x_best, best_val = Xf, _max_cos(Xf)
        Xp, vp = lp_polish(x_best, deadline=t0 + BUDGET - 1.0, max_steps=20, r_init=8e-4)
        if vp < best_val:
            x_best, best_val = Xp, vp
    except Exception:
        pass
    x_best = _norm(x_best)
    return x_best
