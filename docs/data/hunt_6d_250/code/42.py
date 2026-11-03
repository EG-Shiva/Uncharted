import time
import numpy as np
from scipy.optimize import minimize

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

_R0 = None
def _get_R0():
    global _R0
    if _R0 is None:
        r = np.random.default_rng(0).standard_normal((6, 5))
        q, _ = np.linalg.qr(r)
        _R0 = q  # 6x5 orthonormal
    return _R0

def _build_B(X):
    n, d = X.shape
    R0 = _get_R0()
    B = np.empty((n, d, 5), dtype=np.float64)
    for i in range(n):
        xi = X[i]
        # project R0 orthogonal to xi
        coeff = xi @ R0  # (5,)
        Q = R0 - np.outer(xi, coeff)
        q, _ = np.linalg.qr(Q)
        # q is 6x5 orthonormal, orthogonal to xi by construction
        B[i] = q
    return B

def _select_active(X, thr, cap):
    n = X.shape[0]
    G = X @ X.T
    iu0, iu1 = np.triu_indices(n, 1)
    vals = G[iu0, iu1]
    m = float(vals.max())
    slack = m - vals
    # pairs with slack < thr (thr is ladder threshold, at least 3e-5 on first step handled by caller)
    mask = slack < thr
    idx = np.where(mask)[0]
    if idx.size == 0:
        # take tightest 200
        idx = np.argsort(slack)[:200]
    if idx.size > cap:
        # keep tightest cap
        order = np.argsort(slack[idx])[:cap]
        idx = idx[order]
    I = iu0[idx].astype(np.int64)
    J = iu1[idx].astype(np.int64)
    return I, J, m

def _slsqp_step(Xb, B, I, J, m, radius, maxiter):
    n, d = Xb.shape
    N = n * 5
    P = len(I)
    if P == 0:
        return Xb, m, False
    BI = B[I]  # P,6,5
    BJ = B[J]
    # precompute for speed: nothing else
    def obj_fun(z):
        return float(z[-1])
    def obj_jac(z):
        g = np.zeros_like(z)
        g[-1] = 1.0
        return g
    # closures using numpy
    def cons_fun(z):
        u = z[:-1].reshape(n, 5)
        t = float(z[-1])
        # y = Xb + B u
        # einsum
        y = Xb + np.einsum('nij,nj->ni', B, u, optimize=False)
        nrm = np.sqrt(np.sum(y * y, axis=1))
        nrm = np.maximum(nrm, 1e-12)
        yI = y[I]
        yJ = y[J]
        dots = np.sum(yI * yJ, axis=1)
        c = dots / (nrm[I] * nrm[J])
        return t - c
    def cons_jac(z):
        u_mat = z[:-1].reshape(n, 5)
        y = Xb + np.einsum('nij,nj->ni', B, u_mat, optimize=False)
        nrm = np.sqrt(np.sum(y * y, axis=1))
        nrm = np.maximum(nrm, 1e-12)
        yI = y[I]
        yJ = y[J]
        ni = nrm[I]
        nj = nrm[J]
        dots = np.sum(yI * yJ, axis=1)
        den = ni * nj
        c = dots / np.maximum(den, 1e-300)
        # S_I = BI^T yJ, S_J = BJ^T yI
        # (P,6,5)*(P,6,1) sum axis1
        S_I = np.sum(BI * yJ[:, :, None], axis=1)  # P,5
        S_J = np.sum(BJ * yI[:, :, None], axis=1)
        uI = u_mat[I]
        uJ = u_mat[J]
        inv_den = 1.0 / np.maximum(den, 1e-300)
        inv_ni2 = 1.0 / np.maximum(ni * ni, 1e-300)
        inv_nj2 = 1.0 / np.maximum(nj * nj, 1e-300)
        termI = S_I * inv_den[:, None] - (c * inv_ni2)[:, None] * uI
        termJ = S_J * inv_den[:, None] - (c * inv_nj2)[:, None] * uJ
        Jac = np.zeros((P, N + 1), dtype=np.float64)
        # scatter: 5 passes
        ar = np.arange(P)
        for p in range(5):
            Jac[ar, I * 5 + p] = -termI[:, p]
            Jac[ar, J * 5 + p] = -termJ[:, p]
        Jac[:, -1] = 1.0
        return Jac
    z0 = np.zeros(N + 1, dtype=np.float64)
    z0[-1] = m
    bounds = [(-radius, radius)] * N + [(m - 0.01, m)]
    cons = {'type': 'ineq', 'fun': cons_fun, 'jac': cons_jac}
    try:
        res = minimize(obj_fun, z0, jac=obj_jac, method='SLSQP',
                       constraints=[cons], bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'disp': False, 'eps': 1e-9})
    except Exception:
        return Xb, m, False
    try:
        z = np.asarray(res.x, dtype=np.float64)
        u_mat = z[:-1].reshape(n, 5)
        y = Xb + np.einsum('nij,nj->ni', B, u_mat, optimize=False)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn = np.maximum(yn, 1e-12)
        Xn = y / yn
        return Xn, float(z[-1]), True
    except Exception:
        return Xb, m, False

def _penalty_step(Xb, B, I, J, m, radius, maxiter=80, K=3e6):
    n, d = Xb.shape
    N = n * 5
    P = len(I)
    if P == 0:
        return Xb, m, False
    BI = B[I]
    BJ = B[J]
    def fg(z):
        u_mat = z[:-1].reshape(n, 5)
        t = float(z[-1])
        y = Xb + np.einsum('nij,nj->ni', B, u_mat, optimize=False)
        nrm = np.sqrt(np.sum(y * y, axis=1))
        nrm = np.maximum(nrm, 1e-12)
        yI = y[I]
        yJ = y[J]
        ni = nrm[I]; nj = nrm[J]
        dots = np.sum(yI * yJ, axis=1)
        den = np.maximum(ni * nj, 1e-300)
        c = dots / den
        viol = c - t
        pos = np.maximum(viol, 0.0)
        loss = t + K * float(np.sum(pos * pos)) + 1e-6 * float(np.sum(u_mat * u_mat))
        # grad
        # dc/du as before
        S_I = np.sum(BI * yJ[:, :, None], axis=1)
        S_J = np.sum(BJ * yI[:, :, None], axis=1)
        uI = u_mat[I]; uJ = u_mat[J]
        inv_den = 1.0 / den
        inv_ni2 = 1.0 / np.maximum(ni * ni, 1e-300)
        inv_nj2 = 1.0 / np.maximum(nj * nj, 1e-300)
        termI = S_I * inv_den[:, None] - (c * inv_ni2)[:, None] * uI
        termJ = S_J * inv_den[:, None] - (c * inv_nj2)[:, None] * uJ
        w = 2.0 * K * pos  # P,
        g_u = np.zeros((n, 5), dtype=np.float64)
        # accumulate: g_u[I_k] += w_k * termI_k ; use np.add.at
        np.add.at(g_u, I, w[:, None] * termI)
        np.add.at(g_u, J, w[:, None] * termJ)
        g_u += 2e-6 * u_mat
        g_t = 1.0 - float(np.sum(w))
        g = np.empty_like(z)
        g[:-1] = g_u.ravel()
        g[-1] = g_t
        return loss, g
    z0 = np.zeros(N + 1, dtype=np.float64)
    z0[-1] = m
    bounds = [(-radius, radius)] * N + [(m - 0.01, m)]
    try:
        res = minimize(fg, z0, jac=True, method='L-BFGS-B',
                       bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-20, 'gtol': 1e-10, 'maxcor': 8})
    except Exception:
        return Xb, m, False
    try:
        z = np.asarray(res.x, dtype=np.float64)
        u_mat = z[:-1].reshape(n, 5)
        y = Xb + np.einsum('nij,nj->ni', B, u_mat, optimize=False)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        Xn = y / np.maximum(yn, 1e-12)
        return Xn, float(z[-1]), True
    except Exception:
        return Xb, m, False

def solve():
    t_start = time.time()
    BUDGET = 168.0
    def remaining():
        return BUDGET - (time.time() - t_start)
    rng = np.random.default_rng(7)
    n, d = 250, 6
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            z = np.asarray(np.load(fn), dtype=np.float64)
            if z.shape == (n, d):
                zn = np.linalg.norm(z, axis=1, keepdims=True)
                zn = np.maximum(zn, 1e-12)
                z = z / zn
                if np.all(np.isfinite(z)):
                    cands.append(z)
        except Exception:
            pass
    if not cands:
        z = rng.standard_normal((n, d))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        cands.append(z)
    elite = min(cands, key=_max_cos).copy()
    elite_val = _max_cos(elite)
    cur = elite.copy()
    cur_val = elite_val

    cap = 1800
    # threshold ladder 1e-4 -> 1e-5 -> 1e-6 with radii
    ladder = [(1e-4, 4e-3, 30), (3e-5, 2e-3, 30), (1e-5, 1.2e-3, 35), (3e-6, 8e-4, 35), (1e-6, 5e-4, 35)]
    # also enforce slack<3e-5 cap at first: 1e-4 covers it; fine
    for lad_idx, (thr, radius, maxiter) in enumerate(ladder):
        if remaining() < 28:
            break
        # re-extract active set from cur (or elite if better?)
        base = cur if cur_val <= elite_val + 1e-9 else elite
        # try two SLSQP rounds per ladder level
        for rep in range(2):
            if remaining() < 25:
                break
            try:
                B = _build_B(base)
                I, J, m = _select_active(base, max(thr, 3e-5) if lad_idx == 0 else thr, cap)
                # ensure we have ~1200-1500 when possible: if too few, widen
                if len(I) < 800:
                    I, J, m = _select_active(base, 3e-5, cap)
            except Exception:
                break
            m_before = _max_cos(base)
            Xn = None
            try:
                Xn, t_pred, ok = _slsqp_step(base, B, I, J, m_before, radius, maxiter)
            except Exception:
                ok = False
                Xn = None
            improved = False
            if ok and Xn is not None:
                try:
                    v = _max_cos(Xn)
                except Exception:
                    v = 1.0
                # keep only if beats (or ties with more equalisation?) - require strict decrease for cur
                if v < m_before - 1e-12:
                    base = Xn
                    m_before = v
                    improved = True
                    if v < elite_val - 1e-12:
                        elite = Xn.copy(); elite_val = v
                    if v < cur_val - 1e-12:
                        cur = Xn.copy(); cur_val = v
                else:
                    # try penalty least-squares as second chance
                    try:
                        if remaining() > 20:
                            Xp, _, ok2 = _penalty_step(base, B, I, J, m_before, radius, maxiter=60)
                            if ok2:
                                vp = _max_cos(Xp)
                                if vp < m_before - 1e-12:
                                    base = Xp; m_before = vp; improved = True
                                    if vp < elite_val - 1e-12:
                                        elite = Xp.copy(); elite_val = vp
                                    if vp < cur_val - 1e-12:
                                        cur = Xp.copy(); cur_val = vp
                    except Exception:
                        pass
            if not improved:
                # no progress: try penalty once if SLSQP failed
                try:
                    if Xn is not None:
                        v = _max_cos(Xn)
                        # even if not better than m_before, it may tie plateau tighter; accept as new base if not worse
                        if v < m_before + 5e-7 and lad_idx < 2:
                            # check tying: count within 1e-9
                            G = Xn @ Xn.T
                            np.fill_diagonal(G, -2.0)
                            mm = float(G.max())
                            tight = int(np.sum(G[np.triu_indices(n, 1)] > mm - 1e-9))
                            # accept for further ladder if it increases tying without hurting much
                            if tight > 10 and v < cur_val + 2e-7:
                                base = Xn.copy()
                                if v < cur_val:
                                    cur = Xn.copy(); cur_val = v
                except Exception:
                    pass
                break
        # update cur/base for next ladder level
        try:
            bv = _max_cos(base)
            if bv < cur_val:
                cur = base.copy(); cur_val = bv
            if bv < elite_val:
                elite = base.copy(); elite_val = bv
        except Exception:
            pass

    # Small topology kicks + re-equalise to test different basin, time permitting
    trial = 0
    iu0, iu1 = np.triu_indices(n, 1)
    while remaining() > 22 and trial < 12:
        trial += 1
        if elite_val < cur_val - 1e-9:
            base = elite.copy()
        else:
            base = cur.copy()
        try:
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            flat = int(np.argmax(G))
            wi, wj = flat // n, flat % n
            rowmax = G.max(axis=1)
            order = np.argsort(-rowmax)
        except Exception:
            wi, wj = 0, 1
            order = np.arange(n)
        Y = base.copy()
        # kick: perturb worst-pair neighbourhood in tangent space
        sigma = 0.002 if (trial % 2 == 0) else 0.004
        K = 8 + (trial % 5)
        victims = list(order[:K])
        # push wi,wj apart + random tangent noise on victims
        try:
            for v in victims:
                noise = rng.standard_normal(d)
                # make tangent
                noise = noise - float(noise @ Y[v]) * Y[v]
                nn = np.linalg.norm(noise)
                if nn > 1e-12:
                    noise /= nn
                    Y[v] = Y[v] + noise * sigma * (0.5 + rng.random())
            # extra split for worst pair
            a = Y[wi].copy(); b = Y[wj].copy()
            da = a - float(a @ b) * b
            dn = np.linalg.norm(da)
            if dn > 1e-12:
                da /= dn
                Y[wi] = Y[wi] + da * sigma
                Y[wj] = Y[wj] - da * sigma * 0.7
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        except Exception:
            continue
        # quick re-equalise with single threshold
        try:
            B = _build_B(Y)
            I, J, m0 = _select_active(Y, 3e-5, cap)
            Xn, _, ok = _slsqp_step(Y, B, I, J, _max_cos(Y), 1.5e-3, 25)
            if not ok:
                continue
            v = _max_cos(Xn)
            # second ladder refine if promising
            if v < cur_val + 5e-6 and remaining() > 20:
                try:
                    B2 = _build_B(Xn)
                    I2, J2, m2 = _select_active(Xn, 1e-5, cap)
                    Xn2, _, ok2 = _slsqp_step(Xn, B2, I2, J2, v, 8e-4, 25)
                    if ok2:
                        v2 = _max_cos(Xn2)
                        if v2 < v:
                            Xn, v = Xn2, v2
                except Exception:
                    pass
            if v < cur_val:
                cur = Xn.copy(); cur_val = v
            if v < elite_val - 1e-12:
                elite = Xn.copy(); elite_val = v
        except Exception:
            continue

    elite = elite / np.linalg.norm(elite, axis=1, keepdims=True)
    return elite
