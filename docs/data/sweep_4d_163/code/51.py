import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _tangent_bases_det(x):
    n, d = x.shape
    t = d - 1
    B = np.empty((n, d, t))
    E = np.eye(d)
    for i in range(n):
        a = x[i]
        k = int(np.argmax(np.abs(a)))
        cols = [j for j in range(d) if j != k]
        M = np.empty((d, t))
        for c, j in enumerate(cols):
            e = E[:, j]
            v = e - float(a @ e) * a
            M[:, c] = v
        Q, _ = np.linalg.qr(M)
        # re-project for safety
        Q = Q - a[:, None] * (a @ Q)[None, :]
        Q, _ = np.linalg.qr(Q)
        B[i] = Q
    return B

def _build_R(x, B, I, J):
    n = x.shape[0]
    t = B.shape[2]
    Nv = n * t
    p = len(I)
    R = np.empty((p, Nv), dtype=float)
    for k in range(p):
        i = int(I[k]); j = int(J[k])
        row = np.zeros(Nv, dtype=float)
        vi = B[i].T @ x[j]
        vj = B[j].T @ x[i]
        row[i*t:(i+1)*t] = vi
        row[j*t:(j+1)*t] = vj
        R[k] = row
    return R

def _rotation_modes(x, B):
    n, d = x.shape
    t = B.shape[2]
    Nv = n * t
    pairs = [(a, b) for a in range(d) for b in range(a+1, d)]
    Q = np.zeros((Nv, len(pairs)))
    for k, (a, b) in enumerate(pairs):
        Ax = np.zeros_like(x)
        Ax[:, a] = x[:, b]
        Ax[:, b] = -x[:, a]
        for i in range(n):
            Q[i*t:(i+1)*t, k] = B[i].T @ Ax[i]
    Qn, _ = np.linalg.qr(Q)
    return Qn

def _lp_polish(x, seconds, r0=2e-3, cap=1200):
    x = _norm(x)
    n, d = x.shape
    iu0, iu1 = np.triu_indices(n, 1)
    cd = np.arange(d)
    best = _max_cos(x)
    radius = r0
    start = time.time()
    while time.time() - start < seconds and radius > 1e-13:
        G = x @ x.T
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = np.where(c_all > thr)[0]
        if sel.size == 0:
            break
        if sel.size > cap:
            v = c_all[sel]
            kth = np.sort(v)[-cap]
            sel = np.where(c_all >= kth)[0]
            if sel.size > cap + 300:
                sel = sel[::2][:cap]
        I = iu0[sel]; J = iu1[sel]
        c = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2*d+1)
        cols = np.concatenate([I[:, None]*d + cd, J[:, None]*d + cd,
                               np.full((p, 1), n*d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n*d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)]*(n*d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = _norm(x + res.x[:-1].reshape(n, d))
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.6, 0.03)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    HARD = 168.0
    RESERVE = 22.0
    rng = np.random.default_rng(20261031)
    n, d = 163, 4
    tdim = 3
    Nv = n * tdim
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            c = np.asarray(np.load(fn), dtype=float)
            if c.shape == (n, d):
                c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
                nr = np.linalg.norm(c, axis=1, keepdims=True)
                nr = np.maximum(nr, 1e-12)
                c = c / nr
                if np.all(np.isfinite(c)):
                    cands.append(c)
        except Exception:
            pass
    if not cands:
        cands.append(_norm(rng.normal(size=(n, d))))
    best = None
    best_val = 1e9
    for c in cands:
        v = _max_cos(c)
        if v < best_val:
            best_val = v
            best = c.copy()
    incumbent = best.copy()
    inc_val = best_val
    iu0, iu1 = np.triu_indices(n, 1)
    B = _tangent_bases_det(best)
    G0 = best @ best.T
    c_all0 = G0[iu0, iu1]
    m0 = float(c_all0.max())
    slack0 = m0 - c_all0
    sel_c = np.where(slack0 < 1e-9)[0]
    if sel_c.size < 200:
        sel_c = np.where(slack0 < 1e-7)[0]
    if sel_c.size < 200:
        sel_c = np.argsort(slack0)[:450]
    if sel_c.size > 900:
        sel_c = np.argsort(slack0)[:900]
    Ic = iu0[sel_c]; Jc = iu1[sel_c]
    smask = (slack0 >= 1e-7) & (slack0 < 5e-4)
    sidx = np.where(smask)[0]
    S2I = iu0[sidx]; S2J = iu1[sidx]
    try:
        R = _build_R(best, B, Ic, Jc)
    except Exception:
        return _lp_polish(best, max(5.0, HARD - (time.time()-t0) - 2.0))
    try:
        Qrot = _rotation_modes(best, B)
    except Exception:
        Qrot, _ = np.linalg.qr(rng.normal(size=(Nv, 6)))
    # near-null modes
    try:
        Gg = R.T @ R
        w, V = np.linalg.eigh(Gg)
        K = min(40, Nv)
        Vk = V[:, :K]
        # remove rotation components
        proj = Qrot @ (Qrot.T @ Vk)
        Vp = Vk - proj
        norms = np.linalg.norm(Vp, axis=0)
        # keep vectors not dominated by rotations
        keep = np.where(norms > 0.3)[0]
        if keep.size < 3:
            keep = np.argsort(-norms)[:max(6, min(K, 8))]
        Vn = Vp[:, keep] / np.maximum(norms[keep], 1e-12)
        # stiffness
        RV = R @ Vn
        stiff = np.linalg.norm(RV, axis=0)
        order = np.argsort(stiff)
        Vn = Vn[:, order]
        stiff = stiff[order]
    except Exception:
        Vn = None
        stiff = None
    modes = []
    if Vn is not None:
        nsoft = min(8, Vn.shape[1])
        # softest indices
        soft = Vn[:, :nsoft]
        ssoft = stiff[:nsoft]
        # pure soft modes both signs
        for j in range(min(4, nsoft)):
            v = soft[:, j] / max(1e-12, float(np.linalg.norm(soft[:, j])))
            modes.append(v.copy())
            modes.append((-v).copy())
        # random combos weighted to softer
        wgt = 1.0 / (ssoft * 4.0 + 0.15)
        wgt = wgt / wgt.sum() * len(wgt)
        for _ in range(26):
            coeff = rng.normal(size=nsoft) * wgt
            u = soft @ coeff
            # occasionally mix a stiffer component
            if rng.random() < 0.2 and Vn.shape[1] > nsoft:
                j0 = rng.integers(nsoft, Vn.shape[1])
                u = u + rng.normal() * 0.25 * Vn[:, j0]
            nu = float(np.linalg.norm(u))
            if nu < 1e-12:
                continue
            u = u / nu
            # orthogonalize vs rotations (safety)
            u = u - Qrot @ (Qrot.T @ u)
            nu2 = float(np.linalg.norm(u))
            if nu2 < 1e-12:
                continue
            u = u / nu2
            modes.append(u)
    else:
        for _ in range(12):
            u = rng.normal(size=Nv)
            u = u - Qrot @ (Qrot.T @ u)
            u /= max(1e-12, np.linalg.norm(u))
            modes.append(u)
    if len(modes) > 34:
        modes = modes[:34]
    alphas = np.array([0.02, 0.05, 0.09, 0.14, 0.22, 0.35, 0.55, 0.85, 1.3, 1.9])
    tol = 2e-4
    cur_best = best.copy()
    cur_val = best_val
    m_thr = m0
    for mi, u in enumerate(modes):
        if time.time() - t0 > HARD - RESERVE - 4.0:
            break
        U3 = u.reshape(n, tdim)
        DX = np.einsum('nqk,nk->nq', B, U3)
        # try both signs via cheap line search, pick better to polish
        best_cand = None
        best_cand_val = 1e9
        for sgn in (1.0, -1.0):
            if time.time() - t0 > HARD - RESERVE - 2.0:
                break
            DXs = sgn * DX
            Yfar = None
            Yfar_val = 1e9
            for a in alphas:
                Y = best + a * DXs
                Y = _norm(Y)
                Gy = Y @ Y.T
                np.fill_diagonal(Gy, -2.0)
                ma = float(np.max(Gy))
                if ma > m_thr + tol:
                    break
                act = False
                if S2I.size > 0:
                    s2c = Gy[S2I, S2J]
                    if float(np.max(s2c)) >= ma - 1e-9:
                        act = True
                Yfar = Y
                Yfar_val = ma
                if act:
                    break
            if Yfar is None:
                continue
            if Yfar_val < best_cand_val:
                best_cand_val = Yfar_val
                best_cand = Yfar
        if best_cand is None:
            continue
        remain = HARD - RESERVE - (time.time() - t0)
        if remain < 1.5:
            break
        nleft = len(modes) - mi
        pol_sec = min(2.5, max(1.0, remain / max(1, nleft) * 1.1))
        try:
            Yp = _lp_polish(best_cand, seconds=pol_sec, r0=2e-3, cap=1200)
        except Exception:
            continue
        try:
            vp = _max_cos(Yp)
        except Exception:
            continue
        if vp < cur_val - 1e-12:
            cur_val = vp
            cur_best = Yp.copy()
            m_thr = vp
            Gn = cur_best @ cur_best.T
            cn = Gn[iu0, iu1]
            mn = float(cn.max())
            sl = mn - cn
            sm = (sl >= 1e-7) & (sl < 5e-4)
            si = np.where(sm)[0]
            S2I = iu0[si]; S2J = iu1[si]
    rem = HARD - (time.time() - t0) - 1.0
    if rem > 0.5:
        fsec = min(RESERVE - 1.0, rem)
        if fsec > 0.5:
            try:
                Yf = _lp_polish(cur_best, seconds=fsec, r0=2e-3, cap=1500)
                vf = _max_cos(Yf)
                if vf < cur_val:
                    cur_best = Yf
                    cur_val = vf
            except Exception:
                pass
    if cur_val > inc_val:
        return incumbent
    return _norm(cur_best)
