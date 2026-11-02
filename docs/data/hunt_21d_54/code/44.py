import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import svds

def _proj(x):
    nrm = np.linalg.norm(x, axis=1, keepdims=True)
    nrm[nrm == 0] = 1
    y = x / nrm
    y = np.nan_to_num(y)
    nrm2 = np.linalg.norm(y, axis=1, keepdims=True)
    nrm2[nrm2 == 0] = 1
    return y / nrm2

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds):
    x = _proj(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start > seconds - 0.2:
            break
        G = x @ x.T
        c = G[iu0]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu0[0][sel], iu0[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 2500:
            idx = np.argpartition(-c[sel], 2500)[:2500]
            I, J = I[idx], J[idx]
            csv = c[sel][idx]
            p = 2500
        else:
            csv = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csv, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": 2.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _proj(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x

def _tangent_bases(x):
    # Householder: H = I-2uu^T maps x to alpha*e0, B = H[:,1:] spans tangent
    n, d = x.shape
    B = np.empty((n, d, d - 1))
    for i in range(n):
        a = x[i]
        a0 = a[0]
        s = 1.0 if a0 >= 0 else -1.0
        alpha = -s  # +-1
        u = a.copy()
        u[0] -= alpha
        nu = float(u @ u)
        if nu < 1e-14:
            # a == alpha*e0, tangent is e1..e_{d-1}
            H = np.eye(d)
        else:
            # H = I - 2 uu^T / nu
            # compute columns 1: efficiently without full H
            # H[:,j] = e_j - 2 u * u_j / nu
            H = np.eye(d) - (2.0 / nu) * np.outer(u, u)
        B[i] = H[:, 1:]
    return B

def _build_jacobian(x, B, offset=1e-3, cap=2200):
    n, d = x.shape
    dm = d - 1
    G = x @ x.T
    iu = np.triu_indices(n, 1)
    c = G[iu]
    m = float(c.max())
    sel = c > m - offset
    I0, J0 = iu[0][sel], iu[1][sel]
    csv = c[sel]
    # keep tightest cap
    if len(I0) > cap:
        idx = np.argpartition(-csv, cap)[:cap]
        I0, J0 = I0[idx], J0[idx]
    p = len(I0)
    if p == 0:
        return None, I0, J0, m
    rows = []
    cols = []
    vals = []
    for r in range(p):
        i = int(I0[r]); j = int(J0[r])
        ci = B[i].T @ x[j]  # dm
        cj = B[j].T @ x[i]
        base_i = i * dm
        base_j = j * dm
        for k in range(dm):
            rows.append(r); cols.append(base_i + k); vals.append(float(ci[k]))
            rows.append(r); cols.append(base_j + k); vals.append(float(cj[k]))
    A = coo_matrix((vals, (rows, cols)), shape=(p, n * dm)).tocsr()
    return A, I0, J0, m

def _null_modes(A, k=16):
    try:
        m, n = A.shape
        kk = int(min(k, min(m, n) - 1))
        if kk < 2:
            return None
        # smallest magnitude
        u, s, vt = svds(A, k=kk, which='SM', maxiter=2000, tol=1e-4)
        # svds returns ascending? sort by s ascending (smallest first)
        order = np.argsort(s)
        vt = vt[order]
        return vt  # kk x N
    except Exception:
        return None

def _modes_fallback_eigh(A, k=16):
    try:
        # dense Gram 1080x1080 eigh, take smallest
        G = (A.T @ A).toarray()
        # symmetrize
        G = 0.5 * (G + G.T)
        w, V = np.linalg.eigh(G)
        order = np.argsort(w)
        V = V[:, order[:k]]  # N x k
        return V.T  # k x N
    except Exception:
        return None

def solve():
    T0 = time.time()
    HARD_END = T0 + 168.0
    EXPLORE_END = T0 + 148.0  # 20s reserved for final polish
    rng = np.random.default_rng(20261101)
    try:
        x0 = np.load("best.npy")
    except Exception:
        x0 = np.load("record.npy")
    x0 = _proj(np.nan_to_num(np.asarray(x0, dtype=float)))
    if x0.shape != (54, 21):
        try:
            r = _proj(np.asarray(np.load("record.npy"), dtype=float))
            if r.shape == (54, 21):
                x0 = r
            else:
                x0 = _proj(rng.normal(size=(54, 21)))
        except Exception:
            x0 = _proj(rng.normal(size=(54, 21)))
    n, d = 54, 21
    dm = d - 1
    best = x0.copy()
    best_val = _max_cos(best)
    try:
        rec = _proj(np.asarray(np.load("record.npy"), dtype=float))
        rv = _max_cos(rec)
        if rv < best_val:
            best, best_val = rec.copy(), rv
    except Exception:
        pass

    iu_all = np.triu_indices(n, 1)
    thetas = [0.05, 0.1, 0.2, 0.35, 0.5]

    def contact_mask(x, thresh_abs):
        G = x @ x.T
        return (G[iu_all] > thresh_abs)

    # incumbent absolute threshold for hamming
    def abs_thresh(x):
        return _max_cos(x) - 1e-3

    ncycle = 0
    while time.time() < EXPLORE_END:
        ncycle += 1
        if HARD_END - time.time() < 25:
            break
        m_inc = _max_cos(best)
        B = _tangent_bases(best)
        A, Ip, Jp, mA = _build_jacobian(best, B, offset=1e-3, cap=2200)
        if A is None:
            break
        vt = _null_modes(A, k=16)
        if vt is None:
            vt = _modes_fallback_eigh(A, k=16)
        if vt is None:
            # fallback random tangent
            vt = rng.normal(size=(12, n * dm))
        K = vt.shape[0]
        mask_inc = contact_mask(best, m_inc - 1e-3)
        # generate random combos
        ncombo = 24
        trials = []  # (val, ham, y)
        for ci in range(ncombo):
            if time.time() > EXPLORE_END - 2:
                break
            sz = int(rng.integers(2, 6))  # 2-5
            sz = min(sz, K)
            idx = rng.choice(K, size=sz, replace=False)
            coeff = rng.normal(size=sz)
            coeff /= (np.linalg.norm(coeff) + 1e-12)
            u = np.zeros(n * dm)
            for q, j in enumerate(idx):
                u += coeff[q] * vt[j]
            U = u.reshape(n, dm)
            # dx per point
            # dx_i = B_i @ U_i
            dx = np.einsum('nik,nk->ni', B, U)
            norms = np.linalg.norm(dx, axis=1)
            mdx = float(norms.max())
            if mdx < 1e-12:
                continue
            # unit directions
            dirx = dx / (norms[:, None] + 1e-12)
            for th in thetas:
                scale = th / mdx  # so max scaled norm = th
                sc_norms = norms * scale
                # rotate: y = cos(sc)*x + sin(sc)*dir
                y = np.cos(sc_norms)[:, None] * best + np.sin(sc_norms)[:, None] * dirx
                y = _proj(y)
                v = _max_cos(y)
                if v > 0.12:
                    continue
                # hamming vs incumbent at absolute threshold (use incumbent thresh)
                # use same absolute cutoff for both
                mask_y = contact_mask(y, m_inc - 1e-3)
                ham = int(np.count_nonzero(mask_y != mask_inc))
                trials.append((v, -ham, ham, y))
        if not trials:
            continue
        # sort by val first, then large ham (i.e., small -ham)
        trials.sort(key=lambda t: (t[0], t[1]))
        # keep trials materially different: ham > 40, and val < 0.12
        polished = 0
        for (v, negham, ham, y) in trials:
            if time.time() > EXPLORE_END:
                break
            if polished >= 4:
                break
            if ham < 40 and v > best_val:
                continue
            # quick skip if v much worse and ham small
            if v > best_val + 0.02 and ham < 80:
                continue
            budget = min(4.0, max(1.0, (EXPLORE_END - time.time()) / 6.0))
            if budget < 1.0:
                break
            yp = _lp_polish(y, budget)
            vp = _max_cos(yp)
            polished += 1
            if vp < best_val - 1e-12:
                best, best_val = yp, vp
                # update incumbent, break to recompute modes
                break
        # if no improvement after polishing top candidates, continue with new random combos
        # but avoid infinite loop: if time low break
        if ncycle > 12:
            break
    # final LP polish of best
    if time.time() < HARD_END - 1:
        remain = HARD_END - time.time() - 0.5
        remain = min(max(remain, 2.0), 22.0)
        best = _lp_polish(best, remain)
    best = _proj(best)
    return best
