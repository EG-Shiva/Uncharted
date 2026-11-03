import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix


def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))


def _build_B(X0):
    n, d = X0.shape
    k = d - 1
    B = np.empty((n, d, k), dtype=np.float64)
    for i in range(n):
        _, _, Vh = np.linalg.svd(X0[i:i+1, :], full_matrices=True)
        B[i] = Vh[1:, :].T
    return B


def _select_active(X, slack, cap=4000):
    n = X.shape[0]
    G = X @ X.T
    iu0, iu1 = np.triu_indices(n, 1)
    gv = G[iu0, iu1]
    M = float(np.max(gv))
    thr = M - slack
    mask = gv >= thr
    idx = np.where(mask)[0]
    if idx.size == 0:
        idx = np.argsort(-gv)[:min(cap, gv.size)]
    if idx.size > cap:
        gsel = gv[idx]
        order = np.argsort(-gsel)[:cap]
        idx = idx[order]
    I = iu0[idx].astype(np.int64)
    J = iu1[idx].astype(np.int64)
    return I, J, M


def _equalize_once(X0, B, I, J, M0, radius, mu, maxiter, lam=1e-7):
    n, d = X0.shape
    k5 = d - 1
    p = I.shape[0]
    if p == 0:
        return X0, False
    Nvars = n * k5

    def fg(v):
        u = v[:-1].reshape(n, k5)
        t = float(v[-1])
        Y = X0 + np.einsum('ndk,nk->nd', B, u, optimize=False)
        nrm = np.sqrt(np.sum(Y * Y, axis=1))
        nrm = np.maximum(nrm, 1e-12)
        Xn = Y / nrm[:, None]
        XI = Xn[I]
        XJ = Xn[J]
        f = np.sum(XI * XJ, axis=1)
        diff = f - t
        pos = diff > 0.0
        relu = np.where(pos, diff, 0.0)
        S = float(np.dot(relu, relu))
        usq = float(np.dot(u.ravel(), u.ravel()))
        L = t + mu * S + lam * usq
        sum_relu = float(np.sum(relu))
        grad_t = 1.0 - 2.0 * mu * sum_relu
        grad_u = (2.0 * lam) * u
        if np.any(pos):
            R = 2.0 * mu * relu
            nrmI = nrm[I]
            nrmJ = nrm[J]
            AI = (XJ - XI * f[:, None]) / nrmI[:, None]
            AJ = (XI - XJ * f[:, None]) / nrmJ[:, None]
            BI = B[I]
            BJ = B[J]
            gI = np.einsum('pdk,pd->pk', BI, AI)
            gJ = np.einsum('pdk,pd->pk', BJ, AJ)
            gI *= R[:, None]
            gJ *= R[:, None]
            np.add.at(grad_u, I, gI)
            np.add.at(grad_u, J, gJ)
        grad = np.empty_like(v)
        grad[:-1] = grad_u.ravel()
        grad[-1] = grad_t
        return L, grad

    v0 = np.zeros(Nvars + 1, dtype=np.float64)
    v0[-1] = M0
    bounds = [(-radius, radius)] * Nvars + [(M0 - 0.002, M0)]
    try:
        res = minimize(fg, v0, jac=True, method="L-BFGS-B",
                       bounds=bounds,
                       options={"maxiter": maxiter, "ftol": 1e-20,
                                "gtol": 1e-9, "maxfun": maxiter * 4,
                                "maxcor": 12})
        v = res.x
    except Exception:
        return X0, False
    u = v[:-1].reshape(n, k5)
    Y = X0 + np.einsum('ndk,nk->nd', B, u, optimize=False)
    nrm = np.linalg.norm(Y, axis=1, keepdims=True)
    nrm = np.maximum(nrm, 1e-12)
    Xn = Y / nrm
    if not np.all(np.isfinite(Xn)):
        return X0, False
    return Xn, True


def _lp_polish(x, best, deadline, k_top=4000, r0=2e-3):
    n, d = x.shape
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    try:
        best_v = _max_cos(x)
    except Exception:
        return x, best
    if best_v > best:
        best_v = best
    radius = r0
    iu0, iu1 = np.triu_indices(n, 1)
    ar = np.arange(d)
    it = 0
    while time.time() < deadline - 1.5 and radius > 1e-13 and it < 120:
        it += 1
        Gall = x @ x.T
        c_all = Gall[iu0, iu1]
        m = float(np.max(c_all))
        window = max(20.0 * radius, 5e-7)
        sel = c_all > m - window
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > k_top:
            idx = np.argpartition(-c_all, k_top)[:k_top]
            I = iu0[idx]
            J = iu1[idx]
            c_sel = c_all[idx]
        else:
            I = iu0[sel]
            J = iu1[sel]
            c_sel = c_all[sel]
        p = I.shape[0]
        nv = n * d + 1
        try:
            cols = np.concatenate([I[:, None] * d + ar,
                                   J[:, None] * d + ar,
                                   np.full((p, 1), n * d)], axis=1).ravel()
            rows = np.repeat(np.arange(p), 2 * d + 1)
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            bounds = [(-radius, radius)] * (n * d) + [(None, float(m))]
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq,
                          b_eq=np.zeros(n), bounds=bounds, method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        dx = np.asarray(res.x[:-1], dtype=np.float64).reshape(n, d)
        y = x + dx
        nrm = np.linalg.norm(y, axis=1, keepdims=True)
        nrm = np.maximum(nrm, 1e-12)
        y = y / nrm
        try:
            v = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if v < best_v - 1e-12 and np.all(np.isfinite(y)):
            x = y
            best_v = v
            radius = min(radius * 1.5, 0.012)
        else:
            radius *= 0.5
    return x, best_v


def solve():
    t_start = time.time()
    BUDGET = 168.0

    def remaining():
        return BUDGET - (time.time() - t_start)

    n, d = 250, 6
    best = None
    best_val = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            z = np.asarray(np.load(fn), dtype=np.float64)
            if z.shape == (n, d):
                zn = np.linalg.norm(z, axis=1, keepdims=True)
                zn = np.maximum(zn, 1e-12)
                z = z / zn
                if np.all(np.isfinite(z)):
                    v = _max_cos(z)
                    if v < best_val:
                        best_val = v
                        best = z.copy()
        except Exception:
            continue
    if best is None:
        rng = np.random.default_rng(0)
        best = rng.standard_normal((n, d))
        best /= np.maximum(np.linalg.norm(best, axis=1, keepdims=True), 1e-12)
        best_val = _max_cos(best)
    cur = best.copy()
    cur_val = float(best_val)

    # Stage A: exact plateau equalisation on shrinking ladder
    ladder = [
        (5e-5, 3.0e-3, 220),
        (1e-5, 2.0e-3, 220),
        (3e-6, 1.5e-3, 260),
    ]
    mus = [5.0e6, 2.0e7]
    for slack, radius, maxiter in ladder:
        if remaining() < 25.0:
            break
        try:
            I, J, M0 = _select_active(cur, slack, cap=3500)
        except Exception:
            continue
        if I.size < 60:
            continue
        if remaining() < 35.0:
            maxiter = min(maxiter, 150)
        try:
            B = _build_B(cur)
        except Exception:
            continue
        Xw = cur.copy()
        for mi, mu in enumerate(mus):
            if remaining() < 14.0:
                break
            mit = maxiter if mi == 0 else max(150, maxiter // 2)
            try:
                Xn, ok = _equalize_once(Xw, B, I, J, float(M0), radius, mu, mit)
            except Exception:
                break
            if not ok:
                break
            try:
                v = _max_cos(Xn)
            except Exception:
                break
            if mi < len(mus) - 1:
                try:
                    Ga = np.sum(Xn[I] * Xn[J], axis=1)
                    Go = np.sum(Xw[I] * Xw[J], axis=1)
                    Ma = float(np.max(Ga))
                    Mo = float(np.max(Go))
                except Exception:
                    break
                if Ma < Mo + 2e-6 and v < cur_val + 5e-5:
                    Xw = Xn
                # else keep Xw
            else:
                if v < cur_val - 1e-12 and np.all(np.isfinite(Xn)):
                    cur = Xn.copy()
                    cur_val = v
                    if v < best_val:
                        best = Xn.copy()
                        best_val = v
                elif v < cur_val + 1e-7:
                    try:
                        Ga = np.sum(Xn[I] * Xn[J], axis=1)
                        Go = np.sum(cur[I] * cur[J], axis=1)
                        sp_n = float(np.max(Ga) - np.min(Ga))
                        sp_o = float(np.max(Go) - np.min(Go))
                    except Exception:
                        break
                    if sp_n < sp_o * 0.5 and v < cur_val + 5e-7:
                        cur = Xn.copy()
                        cur_val = v
                break
        if remaining() < 10.0:
            break

    # Stage B: trust-region LP descent to push tied level down
    if remaining() > 12.0:
        deadline = t_start + BUDGET - 2.0
        try:
            cur, cur_val = _lp_polish(cur, min(cur_val, best_val), deadline, k_top=4000, r0=2e-3)
        except Exception:
            pass
        if cur_val < best_val and np.all(np.isfinite(cur)):
            best = cur.copy()
            best_val = cur_val
    # Stage C: final fine LP with small radius if time remains
    if remaining() > 12.0:
        deadline = t_start + BUDGET - 1.0
        try:
            cur2, cur2_val = _lp_polish(best.copy(), best_val, deadline, k_top=4500, r0=5e-4)
        except Exception:
            cur2, cur2_val = None, np.inf
        try:
            if cur2 is not None and cur2_val < best_val and np.all(np.isfinite(cur2)):
                best = cur2.copy()
                best_val = cur2_val
        except Exception:
            pass

    out = best.copy()
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
    # strict feasibility: replace any non-finite row
    if not np.all(np.isfinite(out)):
        out = np.where(np.isfinite(out), out, 0.0)
        nrm = np.linalg.norm(out, axis=1, keepdims=True)
        bad = (nrm[:, 0] < 1e-12) | (~np.isfinite(nrm[:, 0]))
        out[bad, 0] = 1.0
        out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
    return out
