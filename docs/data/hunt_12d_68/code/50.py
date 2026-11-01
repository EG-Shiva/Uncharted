import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _mean_contacts(x, thr=1e-7):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    c = np.sum(G > m - thr)
    # c counts ordered pairs (both triangles); mean per point = c / n
    return c / x.shape[0], m, int(np.sum(G > m - 1e-9) // 2)

def _tangent_perturb(X, scale, rng):
    n, d = X.shape
    R = rng.standard_normal((n, d))
    # project to tangent
    dot = np.sum(R * X, axis=1, keepdims=True)
    R = R - dot * X
    nr = np.linalg.norm(R, axis=1, keepdims=True) + 1e-300
    R = R / nr
    Y = X + scale * R
    Y /= (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-300)
    return Y

def _make_hinge(n, d, IU0, I0, J0, tau):
    # IU0 not needed except shape; use I0,J0 vectors
    def fg(yf):
        Y = yf.reshape(n, d)
        nr = np.linalg.norm(Y, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        X = Y / nr
        # pairwise cos for upper triangle
        # compute full Gram then index (fast for n=68)
        G = X @ X.T
        g = G[I0, J0]
        diff = g - tau
        pos = diff > 0
        if not np.any(pos):
            # feasible: zero objective, zero grad
            return 0.0, np.zeros_like(yf)
        dp = np.where(pos, diff, 0.0)
        F = float(dp @ dp)
        w = 2.0 * dp  # zero where not pos
        # P symmetric
        P = np.zeros((n, n), dtype=np.float64)
        P[I0, J0] = w
        # symmetrize: P+P.T (diagonal zero)
        # use in-place
        P = P + P.T
        GX = P @ X
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GY = (GX - dot * X) / nr
        return F, GY.ravel()
    return fg

def _run_hinge(x0, tau, I0, J0, maxiter):
    n, d = x0.shape
    fg = _make_hinge(n, d, None, I0, J0, tau)
    y0 = (x0 / (np.linalg.norm(x0, axis=1, keepdims=True) + 1e-300)).ravel().copy()
    try:
        r = minimize(fg, y0, jac=True, method='L-BFGS-B',
                     options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-7,
                              'maxfun': maxiter * 2, 'maxcor': 25})
        Y = r.x.reshape(n, d)
        Y /= (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-300)
        # r.fun is final F
        return Y, float(r.fun), bool(getattr(r, 'success', False))
    except Exception:
        return x0.copy(), 1e9, False

def _lp_polish(x, deadline, cap_p=2000, r0=8e-4):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        if time.time() > deadline - 0.4:
            break
        c = (x @ x.T)[iu]
        m = float(np.max(c))
        sel = c > m - max(20.0 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        csel = c[sel]
        if p > cap_p:
            od = np.argsort(-csel)[:cap_p]
            I, J = I[od], J[od]
            csel = csel[od]
            p = cap_p
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        step = res.x[:-1].reshape(n, d)
        y = x + step
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best - 1e-15:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            improved = False
            for shrink in (0.3, 0.1):
                yy = x + step * shrink
                yy /= np.linalg.norm(yy, axis=1, keepdims=True)
                vv = _max_cos(yy)
                if vv < best:
                    x, best = yy, vv
                    improved = True
                    break
            if not improved:
                radius *= 0.5
    return x

def solve():
    t0 = time.time()
    T_END = t0 + 170.0
    explore_end = T_END - 32.0
    rng = np.random.default_rng(20261031)
    n, d = 68, 12
    iu = np.triu_indices(n, 1)
    I0, J0 = iu[0], iu[1]

    bestX = None
    best_score = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=np.float64)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    s = _max_cos(a)
                    if s < best_score:
                        best_score = s
                        bestX = a.copy()
        except Exception:
            pass
    if bestX is None:
        bestX = rng.standard_normal((n, d))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        best_score = _max_cos(bestX)

    cur = bestX.copy()
    cur_score = best_score

    taus = [0.2198, 0.2195, 0.2192, 0.2188, 0.2184]
    # perturbation schedule
    scales = [0.0, 0.01, 0.02, 0.03, 0.05, 0.015, 0.04, 0.025]

    # track sparsest interesting basin
    sparseX = None
    sparse_score = 1e9
    sparse_mean = 1e9

    for ti, tau in enumerate(taus):
        if time.time() > explore_end - 3.0:
            break
        # attempts per tau: more for first taus
        nattempt = 7 if ti < 2 else 5
        best_F_this_tau = 1e9
        bestX_this_tau = None
        bestMax_this_tau = 1e9
        for att in range(nattempt):
            if time.time() > explore_end - 2.0:
                break
            scale = scales[att % len(scales)]
            # alternate base between cur and bestX for diversity
            if att % 3 == 2:
                base = bestX
            else:
                base = cur
            if scale == 0.0 and att > 0:
                scale = 0.02
            if att == 0 and ti == 0:
                x_start = base.copy()  # start exactly at incumbent first
            else:
                # for later taus, if cur already feasible for this tau, perturb cur
                x_start = _tangent_perturb(base, scale if scale > 0 else 0.01, rng)
            # hinge budget: larger for early attempts
            maxiter = 350 if att < 3 else 250
            # shrink if low on time
            if time.time() > explore_end - 12.0:
                maxiter = 150
            Xh, Fh, _ = _run_hinge(x_start, tau, I0, J0, maxiter)
            mh = _max_cos(Xh)
            mean_c, _, ncontact = _mean_contacts(Xh, thr=1e-7)
            # track best for this tau (by F then max)
            if Fh < best_F_this_tau - 1e-18 or (abs(Fh - best_F_this_tau) < 1e-18 and mh < bestMax_this_tau):
                best_F_this_tau = Fh
                bestX_this_tau = Xh.copy()
                bestMax_this_tau = mh
            # feasibility check
            feasible = mh <= tau + 2e-12
            if feasible:
                # accept: move cur to feasible point, update best
                cur = Xh.copy()
                cur_score = mh
                if mh < best_score - 1e-15:
                    best_score = mh
                    bestX = Xh.copy()
                # sparse tracking
                if mean_c < sparse_mean:
                    sparse_mean = mean_c
                    sparseX = Xh.copy()
                    sparse_score = mh
                break  # lower tau
            else:
                # not feasible: still update cur if hinge improved max meaningfully
                # to allow rearrangement, move cur toward hinge result if it lowers F a lot
                # keep cur as hinge result if its max is better than cur_score
                if mh < cur_score - 1e-7:
                    cur = Xh.copy()
                    cur_score = mh
                    if mh < best_score - 1e-12:
                        best_score = mh
                        bestX = Xh.copy()
                elif Fh < 1e-6:
                    # near-feasible: adopt as cur to continue squeezing
                    cur = Xh.copy()
                    cur_score = mh
                # interesting basin: mean contacts toward 11-13 even before squeeze
                # accept as interesting if sparser and not much worse
                if mean_c < 16.0 and mh < best_score + 0.002:
                    if mean_c < sparse_mean or mh < sparse_score:
                        sparse_mean = mean_c
                        sparseX = Xh.copy()
                        sparse_score = mh
                    # quick LP polish of sparse basin (2s) to test squeeze potential
                    if time.time() < explore_end - 6.0 and mh < best_score + 0.001:
                        Xp = _lp_polish(Xh, time.time() + 2.0, cap_p=1600, r0=6e-4)
                        mp = _max_cos(Xp)
                        if mp < best_score - 1e-12:
                            best_score = mp
                            bestX = Xp.copy()
                            cur = Xp.copy()
                            cur_score = mp
                        # also update sparse if polished is sparser+good
                        mc2, _, _ = _mean_contacts(Xp, thr=1e-7)
                        if mc2 < sparse_mean and mp < sparse_score + 0.0005:
                            sparse_mean = mc2
                            sparseX = Xp.copy()
                            sparse_score = mp
                # if stalled (F barely moved from start), force larger kick next att
                # (handled by scale schedule)
                pass
        # after attempts for this tau: if feasible we already broke with cur feasible
        # if not feasible but we found a better max, keep it; otherwise re-perturb cur from bestX
        if bestX_this_tau is not None and bestMax_this_tau < cur_score + 0.003:
            # keep momentum: set cur to best of this tau for next tau start
            # only if it is reasonably good
            if bestMax_this_tau < cur_score:
                cur = bestX_this_tau.copy()
                cur_score = bestMax_this_tau
        # if time is ample and this tau failed, try one large-kick escape before next tau
        if time.time() < explore_end - 8.0 and cur_score > tau:
            # one large topology kick 0.08 + hinge at same tau, quick
            xl = _tangent_perturb(bestX, 0.08, rng)
            Xh2, Fh2, _ = _run_hinge(xl, tau, I0, J0, 200)
            mh2 = _max_cos(Xh2)
            if mh2 < best_score - 1e-12:
                best_score = mh2
                bestX = Xh2.copy()
            if mh2 < cur_score:
                cur = Xh2.copy()
                cur_score = mh2
            mc2, _, _ = _mean_contacts(Xh2, thr=1e-7)
            if mc2 < sparse_mean and mh2 < best_score + 0.002:
                sparse_mean = mc2
                sparseX = Xh2.copy()
                sparse_score = mh2

    # polish sparsest interesting basin if it is promising and different
    if sparseX is not None and time.time() < explore_end - 4.0:
        # only invest if sparse is meaningfully sparser than incumbent (~20) or close to best
        cur_mean, _, _ = _mean_contacts(bestX, thr=1e-7)
        if sparse_mean < cur_mean - 1.0 and sparse_score < best_score + 0.0015:
            Xp = _lp_polish(sparseX, time.time() + 4.0, cap_p=2000, r0=8e-4)
            mp = _max_cos(Xp)
            if mp < best_score:
                best_score = mp
                bestX = Xp.copy()

    # final long LP squeeze on incumbent best
    remain = T_END - time.time()
    if remain > 5.0:
        lp_time = min(26.0, remain - 1.5)
        bestX = _lp_polish(bestX, time.time() + lp_time, cap_p=2200, r0=7e-4)
        bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
        best_score = _max_cos(bestX)

    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX
