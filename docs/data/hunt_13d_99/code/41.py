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
    U = np.empty((n, d, k))
    I = np.eye(d)
    for i in range(n):
        x = X[i]
        # drop coordinate with smallest |x| for stability
        drop = int(np.argmin(np.abs(x)))
        B = np.delete(I, drop, axis=1)  # 13x12
        # project out x component
        coeff = x @ B  # (12,)
        B = B - np.outer(x, coeff)
        Q, _ = np.linalg.qr(B)
        # ensure orthogonal to x (fix sign, re-project tiny residual)
        # Q spans complement, should already satisfy Q^T x ~ 0
        U[i] = Q
    return U

def solve():
    t0 = time.time()
    LP_BUDGET = 110.0
    n, d = 99, 13
    k = d - 1
    nv = n * k + 1
    tidx = n * k

    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None

    cands = []
    if xb is not None:
        try:
            xb = _norm_rows(np.asarray(xb, dtype=float))
            if xb.shape == (n, d) and np.all(np.isfinite(xb)):
                cands.append(xb)
        except Exception:
            pass
    if xr is not None:
        try:
            xr = _norm_rows(np.asarray(xr, dtype=float))
            if xr.shape == (n, d) and np.all(np.isfinite(xr)):
                cands.append(xr)
        except Exception:
            pass
    if not cands:
        rng = np.random.default_rng(0)
        cands.append(_norm_rows(rng.standard_normal(size=(n, d))))

    vals = [_max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    cur = _norm_rows(cands[bi].copy())
    cur_max = float(vals[bi])
    best_x = cur.copy()
    best_val = cur_max

    iu0 = np.triu_indices(n, 1)

    r = 3e-4
    eps = 3e-6
    r_min = 1e-8

    # geometric shrink to reach 1e-8 in ~35 accepts: 0.78^35 ~ 1e-4
    # use 0.75 on accept, 0.5 on reject
    it = 0
    while True:
        if time.time() - t0 > LP_BUDGET:
            break
        if r < r_min:
            break
        if it > 200:
            break
        it += 1

        G = cur @ cur.T
        c_all = G[iu0]
        m = float(np.max(c_all))
        cur_max = m
        # active set: slack < eps
        thr = m - eps
        order = np.argsort(-c_all)  # descending
        # count within eps
        cnt_eps = int(np.sum(c_all > thr))
        if cnt_eps < 300:
            take = min(300, len(c_all))
            sel_idx = order[:take]
            eff_eps = float(m - c_all[sel_idx[-1]]) if take > 0 else eps
        elif cnt_eps > 1200:
            sel_idx = order[:1200]
            eff_eps = float(m - c_all[sel_idx[-1]])
        else:
            sel_idx = order[:cnt_eps]
            eff_eps = eps
        p = len(sel_idx)
        if p == 0:
            r *= 0.5
            continue
        I = iu0[0][sel_idx]
        J = iu0[1][sel_idx]
        csub = c_all[sel_idx]

        # tangent bases at cur
        try:
            U = _tangent_bases(cur)
        except Exception:
            r *= 0.5
            continue

        # build rows: a^T d - t <= -c
        # a for point i: U_i^T x_j, for point j: U_j^T x_i
        rows = np.repeat(np.arange(p), 2 * k + 1)
        cols = np.empty(p * (2 * k + 1), dtype=np.int64)
        valsA = np.empty(p * (2 * k + 1), dtype=float)
        # fill in chunks
        for kk in range(p):
            i = int(I[kk]); j = int(J[kk])
            xi = cur[i]; xj = cur[j]
            ai = U[i].T @ xj  # (12,)
            aj = U[j].T @ xi
            base = kk * (2 * k + 1)
            cols[base:base + k] = i * k + np.arange(k)
            valsA[base:base + k] = ai
            cols[base + k:base + 2 * k] = j * k + np.arange(k)
            valsA[base + k:base + 2 * k] = aj
            cols[base + 2 * k] = tidx
            valsA[base + 2 * k] = -1.0
        try:
            A_ub = coo_matrix((valsA, (rows, cols)), shape=(p, nv)).tocsr()
        except Exception:
            r *= 0.5
            continue
        b_ub = -csub
        obj = np.zeros(nv)
        obj[tidx] = 1.0
        # bounds: d in [-r,r], t in [-1, m]
        bnds = [(-r, r)] * (n * k) + [(-1.0, m)]
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=b_ub, bounds=bnds, method="highs")
        except Exception:
            r *= 0.5
            continue
        if res.status != 0 or res.x is None or not np.all(np.isfinite(res.x)):
            r *= 0.5
            # shrink eps slowly too
            eps = max(eps * 0.8, 5e-10)
            continue
        dz = res.x[:n * k]
        t_pred = float(res.x[tidx])
        D = dz.reshape(n, k)
        # actual move: delta_i = U_i d_i
        # einsum
        try:
            Delta = np.einsum('nik,nk->ni', U, D)
        except Exception:
            r *= 0.5
            continue
        if not np.all(np.isfinite(Delta)):
            r *= 0.5
            continue
        Y = cur + Delta
        Y = _norm_rows(Y)
        if not np.all(np.isfinite(Y)):
            r *= 0.5
            continue
        new_max = _max_cos(Y)
        # accept only if true max decreases
        if new_max < cur_max - 1e-13:
            cur = Y
            cur_max = new_max
            if new_max < best_val:
                best_val = new_max
                best_x = Y.copy()
            # shrink trust region geometrically
            r = max(r * 0.75, r_min * 0.5)
            # shrink active-set epsilon: geometric + tie to r
            # target eps proportional to r, floor 5e-10
            target = max(3e-6 * (r / 3e-4), 5e-10)
            eps = max(min(eps / 1.6, target * 1.5), 5e-10)
            # also enforce eps >= ~10x residual spread estimate:
            # estimate spread as m - median(active) clipped
            # keep eps from collapsing too fast if plateau still wide
            # compute slack spread of top 600
            try:
                top600 = c_all[order[:600]]
                spread = float(np.median(m - top600))
                eps = max(eps, min(spread * 0.5, 3e-6))
            except Exception:
                pass
        else:
            # reject: shrink radius aggressively
            r *= 0.5
            # mildly shrink eps to try different linearization
            eps = max(eps * 0.9, 5e-10)
            # if prediction was very optimistic but true failed,
            # trust region too large -> already shrunk
            continue

    best_x = _norm_rows(best_x)
    return best_x
