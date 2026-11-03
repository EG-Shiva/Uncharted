import time
import numpy as np
from scipy.optimize import linprog
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
    # Householder basis: for each x, H maps x to +/-e0, columns 1: span tangent
    n, d = X.shape
    B = np.empty((n, d, d - 1))
    e0 = np.zeros(d)
    e0[0] = 1.0
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
    HARD = 170.0
    LP_BUDGET = 110.0
    n, d, k = 99, 13, 12
    # load start
    xb = None
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
    eps = 3e-6
    r_min = 1e-8
    eps_min = 5e-10

    B = _tangent_bases(cur_x)

    it = 0
    # sequential LP homotopy
    while True:
        if time.time() - t0 > LP_BUDGET:
            break
        if r < r_min:
            break
        if it > 400:
            break
        it += 1
        # current dots
        G = cur_x @ cur_x.T
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        cur_m = m
        slack = m - c_all
        order = np.argsort(slack)  # ascending: tightest first
        # count within eps
        # take up to 1200 tightest, but only those with slack<eps (plus minimum)
        cnt_eps = int(np.sum(slack < eps))
        if cnt_eps < 200:
            # underdetermined: widen to at least 300 tightest to pin DOF
            take = min(nPairs, max(300, cnt_eps))
            # also grow eps a bit for next time
            eps = min(5e-6, eps * 1.5)
        else:
            take = min(1200, cnt_eps)
        # if plateau huge, cap at 1200 tightest
        sel_idx = order[:take]
        c_sel = c_all[sel_idx]
        I = iu0[sel_idx]
        J = iu1[sel_idx]
        p = take
        if p < 10:
            eps = min(5e-6, eps * 2.0)
            r *= 0.9
            continue
        # build LP: vars [y (n*k), t]
        Nvar = n * k + 1
        # coeffs
        BI = B[I]          # (p,d,k)
        BJ = B[J]
        xJ = cur_x[J]      # (p,d)
        xI = cur_x[I]
        coeff_I = np.einsum('pdk,pd->pk', BI, xJ)
        coeff_J = np.einsum('pdk,pd->pk', BJ, xI)
        rows = np.repeat(np.arange(p), 2 * k + 1)
        Ibase = I[:, None] * k + ar_k[None, :]
        Jbase = J[:, None] * k + ar_k[None, :]
        cols = np.concatenate([Ibase, Jbase, np.full((p, 1), Nvar - 1)], axis=1).ravel()
        vals = np.concatenate([coeff_I, coeff_J, -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, Nvar)).tocsr()
        except Exception:
            r *= 0.5
            continue
        b_ub = -c_sel
        obj = np.zeros(Nvar)
        obj[-1] = 1.0
        bounds = [(-r, r)] * (n * k) + [(None, None)]
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        except Exception:
            r *= 0.5
            continue
        if res.status != 0 or res.x is None:
            r *= 0.5
            # shrink eps slowly anyway
            continue
        yvec = res.x[:-1]
        # predicted t (unused except diagnostics)
        Y = yvec.reshape(n, k)
        # trust-region infinity check (clip tiny overshoot)
        if np.abs(Y).max() > r * (1 + 1e-9):
            # scale back (should not happen)
            Y = np.clip(Y, -r, r)
        dX = np.einsum('ndk,nk->nd', B, Y)
        trial = cur_x + dX
        trial = _norm_rows(trial)
        tm = _max_cos(trial)
        if tm < m - 1e-13 and tm < m + 1e-12 and np.isfinite(tm):
            # accept
            cur_x = trial
            cur_m = tm
            if tm < best_m:
                best_m = tm
                best_x = trial.copy()
            B = _tangent_bases(cur_x)
            # shrink trust region geometrically
            r *= 0.82
            # shrink active-set epsilon to force collapse of plateau
            # estimate spread of tightest 600 slacks
            try:
                G2 = cur_x @ cur_x.T
                c2 = G2[iu0, iu1]
                m2 = float(c2.max())
                s2 = m2 - c2
                o2 = np.sort(s2)  # ascending
                # width of plateau (e.g. 600th slack)
                idx_w = min(600, len(o2) - 1)
                spread = float(o2[idx_w] - o2[0])
                target = max(eps / 3.0, 10.0 * spread, eps_min)
                # move partway
                eps = max(eps_min, min(eps * 0.7, target if target < eps else eps * 0.7))
            except Exception:
                eps = max(eps_min, eps * 0.7)
            if r < 2e-7:
                eps = max(eps_min, min(eps, 5e-8))
            if r < 5e-8:
                eps = max(eps_min, min(eps, 5e-9))
        else:
            # reject: shrink radius
            r *= 0.5
            # keep eps, maybe slightly shrink if many rejects
            # do not rebuild B
            continue

    # return best iterate renormalised
    best_x = _norm_rows(best_x)
    # safety: if homotopy failed to improve, return it anyway (valid code)
    # ensure no NaN / zero rows
    nr = np.linalg.norm(best_x, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    best_x = best_x / nr
    # if time remains, do a tiny greedy coordinate polish? skip to stay deterministic
    return best_x
