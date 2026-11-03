import time
import numpy as np
from scipy.optimize import linprog
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
            denom = 1e-12
        for j in range(d):
            if j == kc:
                continue
            M[j, col] = 1.0
            M[kc, col] = -xi[j] / denom
            col += 1
        Q, _ = np.linalg.qr(M)
        T[i] = Q
    return T

def _rank_check_and_null_step(X):
    try:
        n = X.shape[0]
        G = X @ X.T
        iu0, iu1 = np.triu_indices(n, 1)
        c = G[iu0, iu1]
        m = c.max()
        slack = m - c
        sel = slack < 1e-6
        if int(np.count_nonzero(sel)) < 2:
            return False
        T = _tangent_bases(X)
        k = X.shape[1] - 1
        Nv = n * k
        idx = np.where(sel)[0][:250]
        I = iu0[idx]; J = iu1[idx]
        Ai = np.einsum('pdk,pd->pk', T[I], X[J])
        Aj = np.einsum('pdk,pd->pk', T[J], X[I])
        A = np.zeros((len(idx), Nv))
        for r in range(len(idx)):
            A[r, int(I[r])*k:int(I[r])*k+k] = Ai[r]
            A[r, int(J[r])*k:int(J[r])*k+k] = Aj[r]
        s = np.linalg.svd(A, compute_uv=False)
        _ = (s > max(A.shape)*np.finfo(float).eps*s[0]).sum()
        return True
    except Exception:
        return False

def _null_equalize(X, T, cfull, slack, iu0, iu1, m_cur, best, radius, rng, deadline):
    n, d = X.shape
    k = d - 1
    Nv = n * k
    exact = slack < 1e-9
    p_exact = int(np.count_nonzero(exact))
    q0 = min(600, len(slack))
    order = np.argpartition(slack, q0-1)[:q0] if q0 < len(slack) else np.arange(len(slack))
    I = iu0[order]; J = iu1[order]
    Xi = X[I]; Xj = X[J]
    Ti = T[I]; Tj = T[J]
    Ai = np.einsum('pdk,pd->pk', Ti, Xj)
    Aj = np.einsum('pdk,pd->pk', Tj, Xi)
    q = len(order)
    Aact = np.zeros((q, Nv), dtype=np.float64)
    for r in range(q):
        Aact[r, int(I[r])*k:int(I[r])*k+k] = Ai[r]
        Aact[r, int(J[r])*k:int(J[r])*k+k] = Aj[r]
    cact = cfull[order]
    mean = cact.mean()
    resid = (cact - mean)
    g = (Aact.T @ resid) / max(1, q)
    gn = np.linalg.norm(g)
    if not np.isfinite(gn) or gn < 1e-16:
        g = rng.randn(Nv)
        gn = np.linalg.norm(g)
        if gn < 1e-16:
            return X, m_cur, best, False
    Ae = None
    pe = 0
    if p_exact > 0 and p_exact < 50:
        ex_idx = np.where(exact)[0]
        Ie = iu0[ex_idx]; Je = iu1[ex_idx]
        Tie = T[Ie]; Tje = T[Je]
        Aie = np.einsum('pdk,pd->pk', Tie, X[Je])
        Aje = np.einsum('pdk,pd->pk', Tje, X[Ie])
        pe = len(ex_idx)
        Ae = np.zeros((pe, Nv))
        for r in range(pe):
            Ae[r, int(Ie[r])*k:int(Ie[r])*k+k] = Aie[r]
            Ae[r, int(Je[r])*k:int(Je[r])*k+k] = Aje[r]
        try:
            M = Ae @ Ae.T
            b = Ae @ g
            w, *_ = np.linalg.lstsq(M + 1e-12*np.eye(pe), b, rcond=None)
            g = g - Ae.T @ w
        except Exception:
            pass
    gn = np.linalg.norm(g)
    if not np.isfinite(gn) or gn < 1e-16:
        return X, m_cur, best, False
    g = g / gn
    try:
        qq = min(150, q)
        s = np.linalg.svd(Aact[:qq], compute_uv=False)
        smin_over_max = s[-1] / max(1e-30, s[0])
    except Exception:
        smin_over_max = 1.0
    scales = [radius, radius*0.5, radius*2.0]
    best_local_X = X
    best_local_v = m_cur
    improved_global = False
    for sgn in (-1.0, 1.0):
        for sc in scales:
            if time.time() >= deadline:
                break
            step = sgn * sc * 2.0 * g.reshape(n, k)
            if smin_over_max < 1e-8:
                step *= 1.5
            dX = np.einsum('ndk,nk->nd', T, step)
            Yt = X + dX
            nr = np.linalg.norm(Yt, axis=1, keepdims=True)
            if not np.all(np.isfinite(nr)) or np.any(nr < 1e-12):
                continue
            Yt /= nr
            v = _max_cos(Yt)
            if v < best_local_v - 1e-15:
                best_local_v = v
                best_local_X = Yt
            if v < best - 1e-15:
                improved_global = True
    if best_local_v < m_cur - 1e-15:
        X = best_local_X
        m_cur = best_local_v
        if best_local_v < best:
            best = best_local_v
            improved_global = True
        return X, m_cur, best, True
    if smin_over_max < 1e-6 and time.time() < deadline - 0.5 and Ae is not None:
        for _ in range(2):
            r = rng.randn(Nv)
            try:
                b2 = Ae @ r
                M = Ae @ Ae.T
                w2, *_ = np.linalg.lstsq(M + 1e-12*np.eye(pe), b2, rcond=None)
                r = r - Ae.T @ w2
            except Exception:
                pass
            nr0 = np.linalg.norm(r)
            if nr0 < 1e-16:
                continue
            r = r / nr0 * (radius * 2.0)
            dX = np.einsum('ndk,nk->nd', T, r.reshape(n, k))
            Yt = X + dX
            Yt /= np.linalg.norm(Yt, axis=1, keepdims=True)
            v = _max_cos(Yt)
            if v < best - 1e-15:
                return Yt, v, v, True
            if v < m_cur - 1e-15:
                return Yt, v, best, True
    return X, m_cur, best, improved_global

def _cc_sqp_polish(X0, best0, deadline, rng):
    X = X0.copy()
    best = best0
    n, d = X.shape
    k = d - 1
    Nv = n * k
    iu0, iu1 = np.triu_indices(n, 1)
    radius = 1e-3
    it = 0
    stall = 0
    m_cur = best
    while time.time() < deadline and radius > 1e-13:
        it += 1
        G = X @ X.T
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
        T = _tangent_bases(X)
        p = cnt
        nv1 = Nv + 1
        Xi = X[I]
        Xj = X[J]
        Ti = T[I]
        Tj = T[J]
        Ai = np.einsum('pdk,pd->pk', Ti, Xj)
        Aj = np.einsum('pdk,pd->pk', Tj, Xi)
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
            radius *= 0.5
            stall += 1
            if stall % 3 == 0 and time.time() < deadline - 0.5:
                X, m_cur, best, improved = _null_equalize(X, T, cfull, slack, iu0, iu1, m_cur, best, radius, rng, deadline)
                if improved:
                    radius = min(radius * 1.5, 1e-3)
                    stall = 0
                else:
                    radius *= 0.5
            continue
        y = res.x[:-1]
        Y = y.reshape(n, k)
        dX = np.einsum('ndk,nk->nd', T, Y)
        alphas = [1.0, 0.5, 0.25, 0.125, 0.0625]
        accepted = False
        for alpha in alphas:
            if time.time() >= deadline:
                break
            Yt = X + alpha * dX
            nr = np.linalg.norm(Yt, axis=1, keepdims=True)
            if not np.all(np.isfinite(nr)) or np.any(nr < 1e-12):
                continue
            Yt /= nr
            v = _max_cos(Yt)
            if v < best - 1e-15:
                X = Yt
                best = v
                m_cur = v
                accepted = True
                break
        if accepted:
            radius = min(radius * 1.4, 1e-3)
            stall = 0
            if it % 5 == 0:
                _rank_check_and_null_step(X)
        else:
            if cnt > 150 and time.time() < deadline - 1.0:
                Xn, mn, bn, imp = _null_equalize(X, T, cfull, slack, iu0, iu1, m_cur, best, radius, rng, deadline)
                if imp:
                    X, m_cur, best = Xn, mn, bn
                    radius = min(radius * 1.2, 1e-3)
                    stall = 0
                    continue
            radius *= 0.5
            stall += 1
        if stall >= 8:
            radius = 3e-4 if radius < 1e-6 else radius
            stall = 0
    return X, best

def solve():
    t0 = time.time()
    HARD_END = t0 + 178.0
    POLISH_BUDGET = 60.0
    polish_dead = min(t0 + POLISH_BUDGET, HARD_END - 1.0)
    rng = np.random.RandomState(0)
    def load_norm(p):
        try:
            a = np.load(p)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (94, 13):
                return None
            nrm = np.linalg.norm(a, axis=1, keepdims=True)
            if not np.all(np.isfinite(nrm)) or np.any(nrm < 1e-12):
                return None
            a = a / nrm
            if not np.all(np.isfinite(a)):
                return None
            return a
        except Exception:
            return None
    Xb = load_norm("best.npy")
    if Xb is None:
        Xb = load_norm("record.npy")
    if Xb is None:
        rng2 = np.random.RandomState(0)
        Xb = rng2.randn(94, 13)
        Xb /= np.linalg.norm(Xb, axis=1, keepdims=True)
    Xb = Xb / np.linalg.norm(Xb, axis=1, keepdims=True)
    best0 = _max_cos(Xb)
    Xcand = Xb.copy()
    best_cand = best0
    try:
        Xp, bp = _cc_sqp_polish(Xb, best0, polish_dead, rng)
        # keep only if beats incumbent
        if np.all(np.isfinite(Xp)) and bp < best0 - 1e-15:
            Xcand, best_cand = Xp, bp
        else:
            # also accept tiny improvement
            if bp < best_cand:
                Xcand, best_cand = Xp, bp
    except Exception:
        pass
    # safety: if not better, return incumbent
    try:
        v_cand = _max_cos(Xcand)
        v_base = _max_cos(Xb)
        if not np.isfinite(v_cand) or v_cand > v_base:
            Xcand = Xb
    except Exception:
        Xcand = Xb
    Xcand = Xcand / np.linalg.norm(Xcand, axis=1, keepdims=True)
    return Xcand
