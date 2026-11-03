import time
import numpy as np
from scipy.optimize import linprog, lsq_linear
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _tangent_bases(X):
    n, d = X.shape
    k = d - 1
    T = np.empty((n, d, k), dtype=np.float64)
    for i in range(n):
        xi = X[i]
        kc = int(np.argmax(np.abs(xi)))
        M = np.zeros((d, k), dtype=np.float64)
        col = 0
        denom = xi[kc]
        if abs(denom) < 1e-12:
            denom = 1e-12 if denom >= 0 else -1e-12
        for j in range(d):
            if j == kc:
                continue
            v = np.zeros(d, dtype=np.float64)
            v[j] = 1.0
            v[kc] = -xi[j] / denom
            M[:, col] = v
            col += 1
        Q, _ = np.linalg.qr(M)
        T[i] = Q
    return T

def _dense_J(J_I, J_J, Ai, Aj, Nv, k):
    # J_I,J_J : (q,)  Ai,Aj : (q,k)
    q = len(J_I)
    J = np.zeros((q, Nv), dtype=np.float64)
    # fill row by row (q<=120 typically)
    for r in range(q):
        J[r, int(J_I[r])*k:int(J_I[r])*k+k] = Ai[r]
        J[r, int(J_J[r])*k:int(J_J[r])*k+k] = Aj[r]
    return J

def _null_equalize(Xc, T, iu0, iu1, cfull, m_cur, radius, rng, deadline):
    n, d = Xc.shape
    k = d - 1
    Nv = n * k
    # tightest q for null projection
    slack = m_cur - cfull
    order = np.argsort(slack)
    q = int(min(80, len(order)))
    if q < 1:
        return Xc, m_cur, False
    sel_q = order[:q]
    Iq = iu0[sel_q]; Jq = iu1[sel_q]
    Xi = Xc[Iq]; Xj = Xc[Jq]
    Ti = T[Iq]; Tj = T[Jq]
    Aq_i = np.einsum('pdk,pd->pk', Ti, Xj)
    Aq_j = np.einsum('pdk,pd->pk', Tj, Xi)
    Jtop = _dense_J(Iq, Jq, Aq_i, Aq_j, Nv, k)
    # rank test
    try:
        sv = np.linalg.svd(Jtop, compute_uv=False)
        smax = float(sv[0]) if len(sv) else 0.0
        smin = float(sv[-1]) if len(sv) else 0.0
        rankdef = (smin < 1e-8 * max(1.0, smax)) or (q > Nv)
    except Exception:
        rankdef = True
    # precompute JJt for projection
    try:
        JJt = Jtop @ Jtop.T
        # regularize slightly
        JJt += np.eye(q) * 1e-12
    except Exception:
        return Xc, m_cur, rankdef
    bestX = Xc
    bestv = m_cur
    improved = False
    # try several null-projected random moves
    ntry = 6
    for t in range(ntry):
        if time.time() >= deadline:
            break
        r = rng.standard_normal(Nv) * radius
        try:
            Jr = Jtop @ r
            # solve JJt w = Jr
            w = np.linalg.solve(JJt, Jr)
            y = r - Jtop.T @ w
        except Exception:
            # fallback: plain random
            y = r
        # scale to trust radius
        mx = float(np.max(np.abs(y))) if y.size else 0.0
        if mx > radius and mx > 0:
            y = y * (radius / mx)
        Y = y.reshape(n, k)
        dX = np.einsum('ndk,nk->nd', T, Y)
        for alpha in (1.0, 0.5, 0.25):
            Yt = Xc + alpha * dX
            nr = np.linalg.norm(Yt, axis=1, keepdims=True)
            if not np.all(np.isfinite(nr)) or np.any(nr < 1e-12):
                continue
            Yt /= nr
            v = _max_cos(Yt)
            if v < bestv - 1e-15:
                bestv = v
                bestX = Yt
                improved = True
                break
        # also try opposite direction
        if not improved:
            for alpha in (1.0, 0.5):
                Yt = Xc - alpha * dX
                nr = np.linalg.norm(Yt, axis=1, keepdims=True)
                if not np.all(np.isfinite(nr)) or np.any(nr < 1e-12):
                    continue
                Yt /= nr
                v = _max_cos(Yt)
                if v < bestv - 1e-15:
                    bestv = v
                    bestX = Yt
                    improved = True
                    break
        if improved:
            break
    # if no strict improvement but rank-deficient, allow sideways equalizing move
    # that reduces LSE of plateau without worsening max
    if not improved and rankdef:
        # LSE gradient step with large k to feel 3e-8 plateau
        try:
            kk = 3e6
            # use top 800 pairs to avoid overflow
            o2 = np.argsort(slack)[:800]
            c8 = cfull[o2]
            m8 = float(c8.max())
            w8 = np.exp(kk * (c8 - m8))
            # build gradient in ambient: accumulate
            Grad = np.zeros_like(Xc)
            I8 = iu0[o2]; J8 = iu1[o2]
            # weighted sum: grad_i += w*x_j
            # vectorized via bincount-like loop with numpy add
            np.add.at(Grad, I8, (w8[:, None] * Xc[J8]))
            np.add.at(Grad, J8, (w8[:, None] * Xc[I8]))
            S = float(w8.sum())
            if S > 0 and np.all(np.isfinite(Grad)):
                Grad /= S
                dot = (Grad * Xc).sum(axis=1, keepdims=True)
                Gt = Grad - dot * Xc
                # map to tangent coords, take small step
                # Gt = T Y  => Y = T^T Gt
                Yg = np.einsum('ndk,nd->nk', T, Gt)
                # normalize to radius
                mg = float(np.max(np.abs(Yg))) if Yg.size else 0.0
                if mg > 0:
                    Yg = Yg * (radius * 0.5 / mg)
                    Y = Yg.reshape(n, k)
                    dX = np.einsum('ndk,nk->nd', T, Y)
                    Yt = Xc - dX
                    Yt /= np.linalg.norm(Yt, axis=1, keepdims=True)
                    v = _max_cos(Yt)
                    if v <= m_cur + 2e-12 and np.all(np.isfinite(Yt)):
                        # check plateau variance reduction: compare mean slack of top 800
                        G2 = Yt @ Yt.T
                        iu = np.triu_indices(n, 1)
                        cf2 = G2[iu]
                        m2 = float(cf2.max())
                        # accept if max not worse and mean of top improves
                        # use sum of top slacks
                        s_old = float(np.sort(slack)[:200].mean())
                        slack2 = m2 - cf2
                        s_new = float(np.sort(slack2)[:200].mean())
                        if s_new > s_old + 1e-15 or v < m_cur:
                            return Yt, v, rankdef
        except Exception:
            pass
    if improved:
        return bestX, bestv, rankdef
    return Xc, m_cur, rankdef

def _cc_sqp_polish(X0, best0, deadline, rng):
    Xcur = X0.copy()
    n, d = Xcur.shape
    k = d - 1
    Nv = n * k
    iu0, iu1 = np.triu_indices(n, 1)
    G0 = Xcur @ Xcur.T
    np.fill_diagonal(G0, -2.0)
    m_cur = float(G0.max())
    Xbest = Xcur.copy()
    best = best0 if best0 <= m_cur else m_cur
    if m_cur < best:
        best = m_cur
        Xbest = Xcur.copy()
    radius = 1e-3
    it = 0
    stall = 0
    while time.time() < deadline and radius > 1e-13:
        it += 1
        if time.time() >= deadline:
            break
        G = Xcur @ Xcur.T
        cfull = G[iu0, iu1]
        m_cur = float(cfull.max())
        slack = m_cur - cfull
        thresh = max(1e-6, 10.0 * radius)
        sel = slack < thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 2500:
            idx = np.argpartition(slack, 2500)[:2500]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = mask
            cnt = 2500
        I = iu0[sel]; J = iu1[sel]
        csel = cfull[sel]
        T = _tangent_bases(Xcur)
        Xi = Xcur[I]; Xj = Xcur[J]
        Ti = T[I]; Tj = T[J]
        Ai = np.einsum('pdk,pd->pk', Ti, Xj)
        Aj = np.einsum('pdk,pd->pk', Tj, Xi)
        p = cnt
        nv1 = Nv + 1
        rows = np.repeat(np.arange(p), 2 * k + 1)
        offI = I * k
        offJ = J * k
        kk = np.arange(k)
        cI = (offI[:, None] + kk[None, :]).ravel()
        cJ = (offJ[:, None] + kk[None, :]).ravel()
        ct = np.full(p, Nv)
        cols = np.concatenate([cI.reshape(p, k), cJ.reshape(p, k), ct[:, None]], axis=1).ravel()
        vals = np.concatenate([Ai, Aj, -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv1)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        b_ub = -csel
        obj = np.zeros(nv1); obj[-1] = 1.0
        bounds = [(-radius, radius)] * Nv + [(None, None)]
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            stall += 1
            radius *= 0.5
            if stall % 2 == 0 and time.time() < deadline - 0.5:
                Xn, vn, _ = _null_equalize(Xcur, T, iu0, iu1, cfull, m_cur, max(radius, 1e-7), rng, deadline)
                if vn < m_cur - 1e-15:
                    Xcur = Xn
                    m_cur = vn
                    if vn < best - 1e-15:
                        best = vn
                        Xbest = Xn.copy()
                    stall = 0
                    radius = min(radius * 1.5 + 1e-7, 1e-3)
            continue
        y = res.x[:-1]
        Y = y.reshape(n, k)
        dX = np.einsum('ndk,nk->nd', T, Y)
        accepted = False
        # curvature-corrected backtracking with exact renormalization
        for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625):
            if time.time() >= deadline:
                break
            Yt = Xcur + alpha * dX
            nr = np.linalg.norm(Yt, axis=1, keepdims=True)
            if not np.all(np.isfinite(nr)) or np.any(nr < 1e-12):
                continue
            Yt /= nr
            v = _max_cos(Yt)
            if v < m_cur - 1e-15:
                Xcur = Yt
                m_cur = v
                accepted = True
                if v < best - 1e-15:
                    best = v
                    Xbest = Yt.copy()
                break
        if accepted:
            stall = 0
            radius = min(radius * 1.4, 1e-3)
            # periodic rank test + null equalization to spread contacts
            if it % 5 == 0 and time.time() < deadline - 1.0:
                # recompute for rank test
                G2 = Xcur @ Xcur.T
                cf2 = G2[iu0, iu1]
                m2 = float(cf2.max())
                T2 = _tangent_bases(Xcur)
                Xn, vn, rd = _null_equalize(Xcur, T2, iu0, iu1, cf2, m2, radius, rng, deadline)
                if vn < m_cur - 1e-15:
                    Xcur = Xn
                    m_cur = vn
                    if vn < best:
                        best = vn
                        Xbest = Xn.copy()
        else:
            stall += 1
            radius *= 0.5
            if stall % 2 == 0 and time.time() < deadline - 0.5:
                Xn, vn, _ = _null_equalize(Xcur, T, iu0, iu1, cfull, m_cur, max(radius * 2.0, 1e-7), rng, deadline)
                if vn < m_cur - 1e-15:
                    Xcur = Xn
                    m_cur = vn
                    if vn < best - 1e-15:
                        best = vn
                        Xbest = Xn.copy()
                    stall = 0
                    radius = min(radius * 2.0, 1e-3)
        # keep trust radii on schedule 1e-3,3e-4,1e-4,...: snap if close
        # (adaptive already covers it)
    return Xbest, best, Xcur, m_cur

def solve():
    T0 = time.time()
    BUDGET = 168.0
    END = T0 + BUDGET
    POLISH_BUDGET = 60.0
    rng = np.random.default_rng(12345)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.standard_normal(size=(94, 13))
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    n, d = xb.shape
    if n != 94 or d != 13:
        # fix shape by resampling/padding deterministically
        tmp = rng.standard_normal(size=(94, 13))
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        m = min(n, 94)
        tmp[:m, :min(d, 13)] = xb[:m, :min(d, 13)]
        xb = tmp / np.linalg.norm(tmp, axis=1, keepdims=True)
    init_best = _max_cos(xb)
    deadline = min(T0 + POLISH_BUDGET, END - 5.0)
    try:
        Xbest, best, _, _ = _cc_sqp_polish(xb, init_best, deadline, rng)
    except Exception:
        Xbest, best = xb, init_best
    Xbest /= np.linalg.norm(Xbest, axis=1, keepdims=True)
    # keep only if beats best.npy
    try:
        x0 = np.load("best.npy").astype(float)
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        v0 = _max_cos(x0)
    except Exception:
        v0 = init_best
        x0 = xb
    vb = _max_cos(Xbest)
    if vb < v0 - 1e-15 and np.all(np.isfinite(Xbest)):
        out = Xbest
    else:
        # return incumbent (never worse)
        if _max_cos(x0) <= _max_cos(xb):
            out = x0
        else:
            out = xb
    out /= np.linalg.norm(out, axis=1, keepdims=True)
    return out
