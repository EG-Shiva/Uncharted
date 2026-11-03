import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _tangent_bases(X):
    n, d = X.shape
    k = d - 1
    B = np.empty((n, d, k))
    e0 = np.zeros(d); e0[0] = 1.0
    I = np.eye(d)
    for i in range(n):
        xi = X[i]
        s = 1.0 if xi[0] >= 0 else -1.0
        u = xi + s * e0
        nu = float(u @ u)
        if nu < 1e-24:
            B[i] = I[:, 1:]
        else:
            H = I - (2.0 / nu) * np.outer(u, u)
            B[i] = H[:, 1:]
    return B

def _lse_polish(cur, best_val, best_x, t0, budget):
    # lightweight tangent projected gradient on LogSumExp + per-point equalisation
    n, d = cur.shape
    t_end = t0 + budget
    alphas = [20000.0, 80000.0, 250000.0]
    steps_list = [1200, 1200, 1000]
    lr_list = [0.6, 0.35, 0.2]
    for alpha, steps, lr0 in zip(alphas, steps_list, lr_list):
        if time.time() > t_end:
            break
        mom = np.zeros_like(cur)
        beta = 0.85
        for s in range(steps):
            if time.time() > t_end:
                break
            G = cur @ cur.T
            np.fill_diagonal(G, -10.0)
            m = float(np.max(G))
            # exp in stable way
            E = np.exp(alpha * (G - m))
            np.fill_diagonal(E, 0.0)
            S = float(E.sum() / 2.0) + 1e-300
            gX = (E @ cur) / S
            row_max = np.max(G, axis=1)
            w = np.exp(0.5 * alpha * (row_max - m))
            w = 0.4 + 0.6 * w
            dot = np.sum(gX * cur, axis=1, keepdims=True)
            Gt = (gX - dot * cur) * w
            # clip extreme grad
            gm = float(np.max(np.abs(Gt)))
            if not np.isfinite(gm) or gm == 0:
                continue
            if gm > 5.0:
                Gt *= (5.0 / gm)
            lr = lr0 * (1.0 - 0.5 * s / max(steps - 1, 1))
            mom = beta * mom + (1 - beta) * Gt
            bc = 1.0 - beta ** (s + 1)
            cur = cur - (lr / max(bc, 1e-12)) * mom * 0.15
            # renormalise
            cur = _norm_rows(cur)
            if (s % 200 == 199) or s == steps - 1:
                v = _max_cos(cur)
                if v < best_val and np.all(np.isfinite(cur)):
                    best_val = v
                    best_x = cur.copy()
        # keep best as current to avoid drift upward
        # allow small uphill for equalisation: only reset if too bad
        v = _max_cos(cur)
        if v > best_val + 3e-6:
            cur = best_x.copy()
    return cur, best_val, best_x


def _lp_phase(cur, best_x, best_val, t0, time_limit, r_start, r_min, eps_start, eps_min, max_con, shrink_accept=0.85):
    n, d = cur.shape
    k = d - 1
    Nvar = n * k + 1
    tidx = n * k
    iu0, iu1 = np.triu_indices(n, 1)
    nPairs = len(iu0)
    ark = np.arange(k)
    r = r_start
    eps = eps_start
    try:
        B = _tangent_bases(cur)
    except Exception:
        return cur, best_x, best_val
    cur_m = _max_cos(cur)
    it = 0
    while True:
        if time.time() - t0 > time_limit:
            break
        if r < r_min:
            break
        if it > 300:
            break
        it += 1
        G = cur @ cur.T
        c_all = G[iu0, iu1]
        m = float(np.max(c_all))
        cur_m = m
        slack = m - c_all
        order = np.argsort(slack)
        cnt_eps = int(np.sum(slack < eps))
        take = max(cnt_eps, 800)
        if take > max_con:
            take = max_con
        if take > nPairs:
            take = nPairs
        if take < 50:
            take = min(nPairs, 300)
        sel_idx = order[:take]
        c_sel = c_all[sel_idx]
        I = iu0[sel_idx]
        J = iu1[sel_idx]
        p = take
        # build sparse LP: minimise t s.t. a^T d - t <= -c
        try:
            BI = B[I]
            BJ = B[J]
            xJ = cur[J]
            xI = cur[I]
            coeff_I = np.einsum('pdk,pd->pk', BI, xJ)
            coeff_J = np.einsum('pdk,pd->pk', BJ, xI)
        except Exception:
            r *= 0.5
            continue
        rows = np.repeat(np.arange(p), 2 * k + 1)
        try:
            Ibase = I[:, None] * k + ark[None, :]
            Jbase = J[:, None] * k + ark[None, :]
            cols = np.concatenate([Ibase, Jbase, np.full((p, 1), tidx)], axis=1).ravel()
            vals = np.concatenate([coeff_I, coeff_J, -np.ones((p, 1))], axis=1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, Nvar)).tocsr()
        except Exception:
            r *= 0.5
            continue
        b_ub = -c_sel
        obj = np.zeros(Nvar)
        obj[tidx] = 1.0
        bnds = [(-r, r)] * (n * k) + [(-1.0, m)]
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=b_ub, bounds=bnds, method="highs")
        except Exception:
            r *= 0.5
            continue
        if res.status != 0 or res.x is None or not np.all(np.isfinite(res.x)):
            r *= 0.5
            eps = max(eps * 0.95, eps_min)
            continue
        dz = res.x[:n * k]
        t_pred = float(res.x[tidx])
        D = dz.reshape(n, k)
        try:
            Delta = np.einsum('ndk,nk->nd', B, D)
        except Exception:
            r *= 0.5
            continue
        Y = _norm_rows(cur + Delta)
        if not np.all(np.isfinite(Y)):
            r *= 0.5
            continue
        new_max = _max_cos(Y)
        if new_max < m - 1e-13:
            cur = Y
            cur_m = new_max
            if new_max < best_val:
                best_val = new_max
                best_x = Y.copy()
            try:
                B = _tangent_bases(cur)
            except Exception:
                pass
            r = max(r * shrink_accept, r_min * 0.5)
            eps = max(eps / 2.5, eps_min)
        elif new_max <= m + 0.2 * 0.5 * r * r and t_pred < m - 1e-12:
            # lateral equalising move: does not worsen beyond curvature error
            cur = Y
            try:
                B = _tangent_bases(cur)
            except Exception:
                pass
            r = max(r * 0.95, r_min * 0.5)
            eps = max(eps / 2.0, eps_min)
        else:
            r *= 0.5
            eps = max(eps * 0.95, eps_min)
    return cur, best_x, best_val


def solve():
    t0 = time.time()
    HARD = 170.0
    n, d = 99, 13
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            v = np.asarray(np.load(fn), dtype=float)
            if v.shape == (n, d) and np.all(np.isfinite(v)):
                v = _norm_rows(v)
                cands.append(v)
        except Exception:
            pass
    if not cands:
        rng = np.random.default_rng(0)
        cands.append(_norm_rows(rng.standard_normal(size=(n, d))))
    vals = [_max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    cur = _norm_rows(cands[bi].copy())
    best_x = cur.copy()
    best_val = float(vals[bi])

    # Phase A: wide epsilon LP homotopy
    cur, best_x, best_val = _lp_phase(cur, best_x, best_val, t0, 65.0,
                                      r_start=3e-4, r_min=5e-6,
                                      eps_start=3e-6, eps_min=2e-9,
                                      max_con=1100, shrink_accept=0.85)
    # Phase B: smooth equalisation
    if time.time() - t0 < 100.0:
        cur, best_val, best_x = _lse_polish(cur, best_val, best_x, t0, 25.0)
        # restart from best for final jam
        cur = best_x.copy()
    # Phase C: tight jam to collapse plateau
    if time.time() - t0 < 150.0:
        cur, best_x, best_val = _lp_phase(cur, best_x, best_val, t0, 155.0,
                                          r_start=1e-4, r_min=1e-8,
                                          eps_start=1e-6, eps_min=5e-10,
                                          max_con=1200, shrink_accept=0.82)
    best_x = _norm_rows(best_x)
    # safety: ensure finite
    if not np.all(np.isfinite(best_x)):
        best_x = _norm_rows(cur)
    return best_x
