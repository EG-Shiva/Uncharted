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

def _per_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    per = np.max(G, axis=1)
    return per, float(np.max(per))

def _deg(X, m, tol):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return np.sum(G > m - tol, axis=1)

def _lbfgs_full(X, beta, maxiter):
    X = _norm(X)
    n, d = X.shape
    iu0, iu1 = np.triu_indices(n, 1)
    y0 = X.ravel().copy()
    def fun(y):
        Y = y.reshape(n, d)
        rn = np.linalg.norm(Y, axis=1, keepdims=True)
        rn = np.maximum(rn, 1e-12)
        Xn = Y / rn
        G = Xn @ Xn.T
        g = G[iu0, iu1]
        m = float(g.max())
        e = np.exp(beta * (g - m))
        S = float(e.sum()) + 1e-300
        loss = m + np.log(S) / beta
        w = e / S
        W = np.zeros((n, n))
        W[iu0, iu1] = w
        W = W + W.T
        GX = W @ Xn
        dot = (GX * Xn).sum(axis=1, keepdims=True)
        GY = (GX - dot * Xn) / rn
        return float(loss), GY.ravel()
    try:
        res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-8, "maxfun": 20000})
        Y = res.x.reshape(n, d)
        return _norm(Y)
    except Exception:
        return X

def _lbfgs_forced(core0, rvec, K, Tfix, beta, maxiter, wpen=60.0):
    Nc, d = core0.shape
    rvec = rvec / max(1e-12, np.linalg.norm(rvec))
    # fixed neighbour set: K largest dots at start
    c0 = _norm(core0.copy())
    d0 = c0 @ rvec
    idx = np.argsort(-d0)[:K]
    iu0, iu1 = np.triu_indices(Nc, 1)
    y0 = c0.ravel().copy()
    def fun(y):
        Y = y.reshape(Nc, d)
        rn = np.linalg.norm(Y, axis=1, keepdims=True)
        rn = np.maximum(rn, 1e-12)
        Xc = Y / rn
        G = Xc @ Xc.T
        g = G[iu0, iu1]
        m = float(g.max())
        e = np.exp(beta * (g - m))
        S = float(e.sum()) + 1e-300
        loss_cc = m + np.log(S) / beta
        w = e / S
        W = np.zeros((Nc, Nc))
        W[iu0, iu1] = w
        W = W + W.T
        GX = W @ Xc
        dot = (GX * Xc).sum(axis=1, keepdims=True)
        GY_cc = (GX - dot * Xc) / rn
        # hinge pull neighbours up to Tfix
        dd = Xc @ rvec  # (Nc,)
        viol = np.zeros(Nc)
        v = Tfix - dd[idx]
        v = np.maximum(v, 0.0)
        viol[idx] = v
        loss_add = wpen * float(np.mean(v ** 2))
        # dL/dXc
        DL = np.zeros_like(Xc)
        coeff = (2.0 * wpen / float(K)) * v  # d/d(d) = -coeff? loss=w*mean(v^2), v=T-d
        # dL/dd_i = -2w/K * v, dL/dX_i = dL/dd_i * r
        for k, ii in enumerate(idx):
            if v[k] > 0:
                DL[ii] += (-coeff[k]) * rvec
        # project to Y
        d2 = (DL * Xc).sum(axis=1, keepdims=True)
        GY_add = (DL - d2 * Xc) / rn
        GY = GY_cc + GY_add
        return float(loss_cc + loss_add), GY.ravel()
    try:
        res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-7, "maxfun": 20000})
        Y = res.x.reshape(Nc, d)
        return _norm(Y)
    except Exception:
        return _norm(core0)

def _relax(X, iters, lr0=0.025, margin0=0.004):
    X = _norm(X)
    best = _max_cos(X)
    bestX = X.copy()
    cur = X.copy()
    lr = lr0
    margin = margin0
    for it in range(iters):
        G = cur @ cur.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        tau = margin * 0.2 + 1e-12
        W = np.exp((G - m) / tau)
        W[G < m - margin * 5.0] = 0.0
        np.fill_diagonal(W, 0.0)
        Grad = W @ cur
        proj = np.sum(Grad * cur, axis=1, keepdims=True)
        Gt = Grad - proj * cur
        gn = float(np.max(np.linalg.norm(Gt, axis=1)))
        if not np.isfinite(gn) or gn < 1e-14:
            margin *= 0.6
            if margin < 5e-7:
                break
            continue
        step = (lr / gn) * Gt
        Xn = _norm(cur - step)
        G2 = Xn @ Xn.T
        np.fill_diagonal(G2, -2.0)
        mn = float(np.max(G2))
        if mn < m + 1e-12:
            cur = Xn
            if mn < best:
                best = mn
                bestX = Xn.copy()
            if it % 200 == 199:
                margin *= 0.85
        else:
            lr *= 0.6
            if lr < 1e-6:
                margin *= 0.7
                lr = lr0 * 0.3
                if margin < 5e-7:
                    break
    return bestX, best

def _lp_polish(X, deadline, t0, BUDGET, r_init=1.0e-3, max_steps=12):
    X = _norm(X.copy())
    n, d = X.shape
    iu0, iu1 = np.triu_indices(n, 1)
    best = _max_cos(X)
    bx = X.copy()
    radius = r_init
    steps = 0
    while time.time() < deadline and time.time() - t0 < BUDGET - 1.5 and steps < max_steps:
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
            k = 3500
            part = np.argpartition(-csel, k - 1)[:k]
            I, J, csel = I[part], J[part], csel[part]
            p = k
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, n * d + 1)).tocsr()
            obj = np.zeros(n * d + 1); obj[-1] = 1.0
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
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
        if p < 10 and radius < 1e-6:
            break
    return bx, best

def _void_best(core, rng, n_cand=20000):
    d = core.shape[1]
    C = rng.randn(n_cand, d)
    C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-12
    D = C @ core.T
    worst = D.max(axis=1)
    bi = int(np.argmin(worst))
    return C[bi], float(worst[bi])

def _refine_void(v, others, iters=150, lr0=0.08):
    v = v / max(1e-12, np.linalg.norm(v))
    lr = lr0
    best = float(np.max(others @ v))
    bestv = v.copy()
    for _ in range(iters):
        dots = others @ v
        m = float(np.max(dots))
        tau = 0.002
        w = np.exp((dots - m) / tau)
        w[dots < m - 0.012] = 0.0
        g = w @ others
        gt = g - float(g @ v) * v
        nrm = float(np.linalg.norm(gt))
        if nrm < 1e-14 or not np.isfinite(nrm):
            break
        vn = v - (lr / nrm) * gt
        vn /= max(1e-12, np.linalg.norm(vn))
        mn = float(np.max(others @ vn))
        if mn < m:
            v = vn
            if mn < best:
                best = mn
                bestv = vn.copy()
        else:
            lr *= 0.7
            if lr < 1e-5:
                break
    return bestv, best

def solve():
    t0 = time.time()
    BUDGET = 168.0
    n, d = 219, 6
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
    incumbent = x_best.copy()
    inc_val = best_val

    # identify rattler(s)
    try:
        per, mcur = _per_max(x_best)
        deg0 = _deg(x_best, mcur, 1e-9)
        r_cands = np.where(deg0 == 0)[0]
        if len(r_cands) == 0:
            deg1 = _deg(x_best, mcur, 1e-7)
            r_idx = int(np.argmin(deg1))
        else:
            # pick the one with smallest per-max among zero-contact
            r_idx = int(r_cands[np.argmin(per[r_cands])])
    except Exception:
        per, mcur = _per_max(x_best)
        r_idx = int(np.argmin(per))

    r_vec = x_best[r_idx].copy()
    mask_core = np.arange(n) != r_idx
    core = x_best[mask_core].copy()
    core_max0 = _max_cos(core)

    deltas = [1e-5, 2e-5, 5e-5, 1e-4]
    Ks = [10, 8, 12, 10]

    # helper to evaluate full config and check rattler activation
    def eval_full(Xfull, rpos):
        m = _max_cos(Xfull)
        # contacts of rpos within 1e-7
        G = Xfull @ Xfull.T
        row = G[rpos].copy()
        row[rpos] = -2.0
        cnt5 = int(np.sum(row > m - 1e-7))
        cnt9 = int(np.sum(row > m - 1e-9))
        # core max excluding rpos
        mask = np.arange(Xfull.shape[0]) != rpos
        cm = _max_cos(Xfull[mask])
        return m, cnt5, cnt9, cm

    # main homotopy loop
    # we keep core and r_vec evolving; r position fixed at end (index n-1) for simplicity
    # represent state as (core(218,6), rvec(6,))
    cur_core = core.copy()
    cur_r = r_vec.copy()
    cur_core_max = core_max0

    cyc = 0
    for delta, K in zip(deltas, Ks):
        if time.time() - t0 > BUDGET - 28:
            break
        for rep in range(2):
            if time.time() - t0 > BUDGET - 28:
                break
            cyc += 1
            # Step A: move r to refined largest void of current core
            try:
                v0, _ = _void_best(cur_core, rng, n_cand=20000)
                vref, _ = _refine_void(v0, cur_core, iters=150)
            except Exception:
                vref = cur_r.copy()
            # Step B: forced-activation optimisation of core with r frozen at vref
            Tfix = cur_core_max - delta
            # also clamp Tfix not above current best
            try:
                forced = _lbfgs_forced(cur_core, vref, K=K, Tfix=Tfix, beta=280.0, maxiter=50, wpen=60.0)
            except Exception:
                forced = cur_core.copy()
            # light relax of forced core alone (core-core only? use full relax on core)
            try:
                forced, _ = _relax(forced, iters=120, lr0=0.02, margin0=0.003)
            except Exception:
                pass
            # Step C: release r (form full code) and re-optimise all points
            Xfull = np.vstack([forced, vref.reshape(1, d)])
            Xfull = _norm(Xfull)
            rpos = 218
            # annealed full reopt: two betas
            try:
                Xfull = _lbfgs_full(Xfull, 130.0, 45)
                if time.time() - t0 < BUDGET - 20:
                    Xfull = _lbfgs_full(Xfull, 380.0, 45)
            except Exception:
                pass
            m_new, cnt5, cnt9, cm_new = eval_full(Xfull, rpos)
            # acceptance: r has >=5 contacts within 1e-7 AND core max dropped AND overall improved or equal with better balance
            improved_core = cm_new < cur_core_max - 1e-12
            good_r = cnt5 >= 5
            # also accept if overall max dropped even without full 5 contacts (to allow progress), but require at least 3
            accept = False
            if good_r and cm_new < cur_core_max - 2e-9 and m_new <= best_val + 2e-6:
                accept = True
            elif good_r and m_new < best_val:
                accept = True
            elif cnt5 >= 3 and m_new < best_val - 1e-9:
                accept = True
            if accept:
                # LP polish this topology
                try:
                    dl = min(time.time() + 4.0, t0 + BUDGET - 12)
                    Xp, mp = _lp_polish(Xfull, dl, t0, BUDGET, r_init=9e-4, max_steps=10)
                except Exception:
                    Xp, mp = Xfull, m_new
                m_p, cnt5p, _, cm_p = eval_full(Xp, rpos)
                # keep if overall better than best, or core dropped with r active
                if m_p < best_val:
                    best_val = m_p
                    x_best = Xp.copy()
                # update current state to polished if it keeps r active and core not worse
                if cnt5p >= 5 and cm_p <= cur_core_max + 1e-9:
                    cur_core = Xp[mask_core if False else np.arange(n) != rpos].copy() if False else np.delete(Xp, rpos, axis=0).copy()
                    # need consistent ordering: core is first 218 rows
                    cur_core = Xp[:218].copy() if rpos == 218 else np.delete(Xp, rpos, axis=0)
                    cur_r = Xp[rpos].copy()
                    cur_core_max = cm_p
                elif m_p <= m_new + 1e-12:
                    cur_core = np.delete(Xp, rpos, axis=0)
                    cur_r = Xp[rpos].copy()
                    cur_core_max = cm_p
            else:
                # not accepted: try releasing with original r position alternative?
                # still track global best if lucky improvement without activation
                if m_new < best_val:
                    # polish and take as best (fallback) even if r not active
                    try:
                        dl = min(time.time() + 3.0, t0 + BUDGET - 12)
                        Xp2, mp2 = _lp_polish(Xfull, dl, t0, BUDGET, r_init=8e-4, max_steps=8)
                    except Exception:
                        Xp2, mp2 = Xfull, m_new
                    if mp2 < best_val:
                        best_val = mp2
                        x_best = Xp2.copy()
                # else keep cur state unchanged; try next delta with slightly perturbed core
                # small perturbation to escape: jitter worst point of forced core
                try:
                    if time.time() - t0 < BUDGET - 30:
                        perF, _ = _per_max(np.vstack([forced, vref.reshape(1, d)]))
                        # jitter
                        pass
                except Exception:
                    pass
            # second rep uses updated cur state automatically

    # final fallback: standard polish of incumbent/best to squeeze
    try:
        if time.time() - t0 < BUDGET - 12:
            dl = min(time.time() + 6.0, t0 + BUDGET - 2)
            Xp, mp = _lp_polish(x_best, dl, t0, BUDGET, r_init=1.0e-3, max_steps=14)
            if mp < best_val:
                best_val = mp
                x_best = Xp.copy()
    except Exception:
        pass
    # extra LBFGS squeeze if time remains
    try:
        if time.time() - t0 < BUDGET - 8:
            Xt = _lbfgs_full(x_best, 500.0, 60)
            mt = _max_cos(Xt)
            if mt < best_val:
                best_val = mt
                x_best = Xt.copy()
                if time.time() - t0 < BUDGET - 4:
                    dl2 = min(time.time() + 3.0, t0 + BUDGET - 1)
                    Xp3, mp3 = _lp_polish(x_best, dl2, t0, BUDGET, r_init=8e-4, max_steps=8)
                    if mp3 < best_val:
                        x_best = Xp3.copy()
                        best_val = mp3
    except Exception:
        pass

    # safety fallback
    if best_val > inc_val:
        x_best = incumbent.copy()
    x_best = _norm(x_best)
    # tiny shrink guard against exact ties? return as is
    return x_best
