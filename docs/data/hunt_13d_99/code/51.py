import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _tangent_bases(X):
    n, d = X.shape
    B = np.empty((n, d, d - 1))
    e0 = np.zeros(d); e0[0] = 1.0
    I = np.eye(d)
    for i in range(n):
        xi = X[i]
        s = 1.0 if xi[0] >= 0 else -1.0
        u = xi + s * e0
        nu = float(np.dot(u, u))
        if nu < 1e-24:
            B[i] = I[:, 1:]
        else:
            H = I - (2.0 / nu) * np.outer(u, u)
            B[i] = H[:, 1:]
    return B

def solve():
    t0 = time.time()
    HARD = 168.0
    n, d, k = 99, 13, 12
    N = n * k
    # load start
    best_val = 1e9
    cand = None
    for fn in ("best.npy", "record.npy"):
        try:
            v = np.load(fn).astype(float)
            if v.shape == (99, 13):
                v = _norm_rows(v)
                sc = _max_cos(v)
                if sc < best_val:
                    best_val = sc
                    cand = v.copy()
        except Exception:
            pass
    if cand is None:
        rng = np.random.default_rng(0)
        cand = _norm_rows(rng.standard_normal(size=(99, 13)))
        best_val = _max_cos(cand)
    cur_x = cand.copy()
    cur_m = float(best_val)
    best_x = cur_x.copy()
    best_m = cur_m

    iu0, iu1 = np.triu_indices(n, 1)
    nPairs = len(iu0)
    ar_k = np.arange(k)

    r = 3e-4
    r_min = 1e-8
    eps = 3e-6
    eps_min = 5e-10
    rho = 5.0

    B = _tangent_bases(cur_x)
    # initial alpha from eps
    def alpha_from_eps(e):
        a = 3.0 / max(e, 1e-9)
        if a < 5e4: a = 5e4
        if a > 2e6: a = 2e6
        return float(a)

    it = 0
    n_reject = 0
    while True:
        if time.time() - t0 > 150.0:
            break
        if r < r_min:
            break
        if it > 60:
            break
        it += 1
        G = cur_x @ cur_x.T
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        cur_m = m
        slack = m - c_all
        order = np.argsort(slack)
        cnt_eps = int(np.sum(slack < eps))
        if cnt_eps < 250:
            take = min(nPairs, max(400, cnt_eps))
            # widen eps a little if starved
            if cnt_eps < 100:
                eps = min(6e-6, eps * 1.3)
        else:
            take = min(1200, cnt_eps)
        sel_idx = order[:take]
        c_sel = c_all[sel_idx].astype(float)
        I = iu0[sel_idx]
        J = iu1[sel_idx]
        p = take
        if p < 20:
            r *= 0.7
            continue
        # coefficients in tangent coords
        BI = B[I]
        BJ = B[J]
        xJ = cur_x[J]
        xI = cur_x[I]
        coeff_I = np.einsum('pdk,pd->pk', BI, xJ)
        coeff_J = np.einsum('pdk,pd->pk', BJ, xI)
        # dense A (p,N)
        A = np.zeros((p, N), dtype=float)
        # fill row by row (p~1200, fast enough)
        for s in range(p):
            o1 = int(I[s]) * k
            o2 = int(J[s]) * k
            A[s, o1:o1 + k] = coeff_I[s]
            A[s, o2:o2 + k] = coeff_J[s]
        alpha = alpha_from_eps(eps)
        # smooth QP: min LSE_alpha(Ay+c) + rho||y||^2  s.t. |y|<=r
        # use L-BFGS-B
        def fun_grad(y):
            vals = A.dot(y) + c_sel  # p
            mv = float(vals.max())
            z = alpha * (vals - mv)
            # clip lower to avoid underflow (exp->0 anyway)
            z = np.maximum(z, -50.0)
            e = np.exp(z)
            S = float(e.sum()) + 1e-300
            LSE = mv + np.log(S) / alpha
            w = e / S  # softmax
            gL = A.T.dot(w)  # N
            f = LSE + rho * float(np.dot(y, y))
            g = gL + 2.0 * rho * y
            return float(f), g.astype(float)
        def fg(y):
            f, g = fun_grad(y)
            return f, g
        bounds = [(-r, r)] * N
        # warm start at zero
        y0 = np.zeros(N, dtype=float)
        # limit iterations to stay in budget
        try:
            res = minimize(fg, y0, jac=True, method='L-BFGS-B',
                           bounds=bounds,
                           options={'maxiter': 80, 'ftol': 1e-20, 'gtol': 1e-9, 'maxfun': 400})
            yvec = np.asarray(res.x, dtype=float)
        except Exception:
            yvec = None
        accepted = False
        if yvec is not None and np.all(np.isfinite(yvec)):
            # clip
            yvec = np.clip(yvec, -r, r)
            Y = yvec.reshape(n, k)
            # skip null step
            if np.abs(Y).max() > 1e-15:
                dX = np.einsum('ndk,nk->nd', B, Y)
                trial = _norm_rows(cur_x + dX)
                tm = _max_cos(trial)
                if np.isfinite(tm) and tm < m - 1e-13:
                    cur_x = trial
                    cur_m = tm
                    if tm < best_m:
                        best_m = tm
                        best_x = trial.copy()
                    B = _tangent_bases(cur_x)
                    r *= 0.82
                    n_reject = 0
                    accepted = True
                    # shrink eps toward collapse
                    try:
                        G2 = cur_x @ cur_x.T
                        c2 = G2[iu0, iu1]
                        m2 = float(c2.max())
                        s2 = m2 - c2
                        o2 = np.sort(s2)
                        idx_w = min(600, len(o2) - 1)
                        spread = float(o2[idx_w] - o2[0])
                        target = max(eps / 3.0, 10.0 * spread, eps_min)
                        if target < eps:
                            eps = max(eps_min, min(eps * 0.7, target))
                        else:
                            eps = max(eps_min, eps * 0.7)
                    except Exception:
                        eps = max(eps_min, eps * 0.7)
                    if r < 2e-7:
                        eps = min(eps, 5e-8)
                    if r < 5e-8:
                        eps = min(eps, 5e-9)
        if accepted:
            continue
        # smooth step failed -> try one HiGHS LP step with same active set
        if time.time() - t0 > 140.0:
            r *= 0.5
            n_reject += 1
            continue
        try:
            Nvar = N + 1
            rows = np.repeat(np.arange(p), 2 * k + 1)
            Ibase = I[:, None] * k + ar_k[None, :]
            Jbase = J[:, None] * k + ar_k[None, :]
            cols = np.concatenate([Ibase, Jbase, np.full((p, 1), Nvar - 1)], axis=1).ravel()
            vals = np.concatenate([coeff_I, coeff_J, -np.ones((p, 1))], axis=1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, Nvar)).tocsr()
            b_ub = -c_sel
            obj = np.zeros(Nvar); obj[-1] = 1.0
            bounds_lp = [(-r, r)] * N + [(None, None)]
            lpres = linprog(obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds_lp, method="highs")
            if lpres.status == 0 and lpres.x is not None:
                yv = np.asarray(lpres.x[:-1], dtype=float)
                yv = np.clip(yv, -r, r)
                Y = yv.reshape(n, k)
                dX = np.einsum('ndk,nk->nd', B, Y)
                trial = _norm_rows(cur_x + dX)
                tm = _max_cos(trial)
                if np.isfinite(tm) and tm < m - 1e-13:
                    cur_x = trial
                    cur_m = tm
                    if tm < best_m:
                        best_m = tm
                        best_x = trial.copy()
                    B = _tangent_bases(cur_x)
                    r *= 0.82
                    n_reject = 0
                    try:
                        G2 = cur_x @ cur_x.T
                        c2 = G2[iu0, iu1]
                        m2 = float(c2.max())
                        s2 = m2 - c2
                        o2 = np.sort(s2)
                        idx_w = min(600, len(o2) - 1)
                        spread = float(o2[idx_w] - o2[0])
                        target = max(eps / 3.0, 10.0 * spread, eps_min)
                        if target < eps:
                            eps = max(eps_min, min(eps * 0.7, target))
                        else:
                            eps = max(eps_min, eps * 0.7)
                    except Exception:
                        eps = max(eps_min, eps * 0.7)
                    continue
        except Exception:
            pass
        # both rejected
        r *= 0.5
        n_reject += 1
        if n_reject >= 4:
            # force eps shrink to refocus on tightest
            eps = max(eps_min, eps * 0.5)
            n_reject = 0

    best_x = _norm_rows(best_x)
    return best_x
