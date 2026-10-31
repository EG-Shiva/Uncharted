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

def _tangent_bases(x, rng):
    n, d = x.shape
    t = d - 1
    B = np.empty((n, d, t))
    for i in range(n):
        A = rng.normal(size=(d, t))
        A = A - x[i][:, None] * (x[i] @ A)[None, :]
        Q, _ = np.linalg.qr(A)
        Q = Q - x[i][:, None] * (x[i] @ Q)[None, :]
        Q, _ = np.linalg.qr(Q)
        B[i] = Q
    return B

def _build_R(x, B, I, J):
    n = x.shape[0]
    t = B.shape[2]
    Nv = n * t
    p = len(I)
    R = np.empty((p, Nv))
    for k in range(p):
        i = int(I[k]); j = int(J[k])
        row = np.zeros(Nv)
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

def _near_null_modes(R, Qrot, K=30):
    Nv = R.shape[1]
    K = min(K, Nv)
    G = R.T @ R
    w, V = np.linalg.eigh(G)
    Vk = V[:, :K]
    proj = Qrot @ (Qrot.T @ Vk)
    Vp = Vk - proj
    U, s, _ = np.linalg.svd(Vp, full_matrices=False)
    keep_mask = s > 0.3
    if int(np.count_nonzero(keep_mask)) < 2:
        idx = np.argsort(-s)[:max(2, min(6, K))]
        keep_mask = np.zeros(K, dtype=bool)
        keep_mask[idx] = True
    Ukeep = U[:, keep_mask]
    RU = R @ Ukeep
    stiff = np.linalg.norm(RU, axis=0)
    return Ukeep, stiff, w[:K]

def _lp_polish(x, seconds, r0=2e-3, cap=1500):
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
            if sel.size > cap + 400:
                sel = sel[::2][:cap]
        I = iu0[sel]; J = iu1[sel]
        c = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
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
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
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
    RESERVE = 20.0
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
        z = rng.normal(size=(n, d))
        cands.append(_norm(z))
    best = None
    best_val = 1e9
    for c in cands:
        v = _max_cos(c)
        if v < best_val:
            best_val = v
            best = c.copy()
    iu0, iu1 = np.triu_indices(n, 1)
    B = _tangent_bases(best, np.random.default_rng(12345))
    G0 = best @ best.T
    c_all0 = G0[iu0, iu1]
    m0 = float(c_all0.max())
    slack0 = m0 - c_all0
    sel_c = np.where(slack0 < 1e-9)[0]
    if sel_c.size < 200:
        sel_c = np.where(slack0 < 1e-7)[0]
    if sel_c.size < 200:
        o = np.argsort(slack0)[:450]
        sel_c = o
    if sel_c.size > 900:
        o = np.argsort(slack0)[:900]
        sel_c = o
    Ic = iu0[sel_c]; Jc = iu1[sel_c]
    second_mask = (slack0 >= 1e-7) & (slack0 < 5e-4)
    second_idx = np.where(second_mask)[0]
    S2I = iu0[second_idx]
    S2J = iu1[second_idx]
    try:
        R = _build_R(best, B, Ic, Jc)
    except Exception:
        return _lp_polish(best, max(5.0, HARD - (time.time()-t0) - 2.0))
    try:
        Qrot = _rotation_modes(best, B)
    except Exception:
        Qrot, _ = np.linalg.qr(rng.normal(size=(Nv, 6)))
    try:
        Ukeep, stiff, _ = _near_null_modes(R, Qrot, K=30)
    except Exception:
        Ukeep = None
    modes = []
    if Ukeep is not None and Ukeep.shape[1] >= 1:
        order = np.argsort(stiff)
        thr = 0.7
        soft_idx = np.where(stiff < thr)[0]
        if soft_idx.size < 2:
            soft_idx = order[:min(3, len(order))]
        if soft_idx.size > 8:
            soft_idx = order[:8]
        Usoff = Ukeep[:, soft_idx]
        ssoft = stiff[soft_idx]
        w = 1.0 / (ssoft * 4.0 + 0.15)
        w = w / w.sum() * len(w)
        n_modes = 28
        for mi in range(n_modes):
            coeff = rng.normal(size=len(soft_idx)) * w
            if mi % 5 == 4 and Ukeep.shape[1] > len(soft_idx):
                rest = [j for j in range(Ukeep.shape[1]) if j not in set(soft_idx.tolist())]
                if rest:
                    j0 = rest[rng.integers(len(rest))]
                    coeff_full = np.zeros(Ukeep.shape[1])
                    coeff_full[soft_idx] = coeff
                    coeff_full[j0] = rng.normal() * 0.25
                    u = Ukeep @ coeff_full
                else:
                    u = Usoff @ coeff
            else:
                u = Usoff @ coeff
            nu = float(np.linalg.norm(u))
            if nu < 1e-12:
                continue
            u = u / nu
            modes.append(u)
        for j in range(min(4, Usoff.shape[1])):
            v = Usoff[:, j] / max(1e-12, float(np.linalg.norm(Usoff[:, j])))
            modes.append(v.copy())
            modes.append((-v).copy())
    else:
        for _ in range(10):
            u = rng.normal(size=Nv); u /= np.linalg.norm(u)
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
        best_y = None
        for a in alphas:
            if time.time() - t0 > HARD - RESERVE - 2.0:
                break
            Y = best + a * DX
            Y = _norm(Y)
            Gy = Y @ Y.T
            np.fill_diagonal(Gy, -2.0)
            ma = float(np.max(Gy))
            if ma > m_thr + tol:
                break
            active_new = False
            if S2I.size > 0:
                s2c = Gy[S2I, S2J]
                if float(np.max(s2c)) >= ma - 1e-9:
                    active_new = True
            best_y = Y
            if active_new:
                break
        if best_y is None:
            continue
        remain = HARD - RESERVE - (time.time() - t0)
        if remain < 2.0:
            break
        pol_sec = min(3.0, max(1.2, remain / (len(modes) - mi + 1) * 1.2))
        try:
            Yp = _lp_polish(best_y, seconds=pol_sec, r0=2e-3, cap=1500)
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
            second_mask = (sl >= 1e-7) & (sl < 5e-4)
            second_idx = np.where(second_mask)[0]
            S2I = iu0[second_idx]
            S2J = iu1[second_idx]
    # final reserve polish
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
    # safety: fallback to incumbent if worse
    if cur_val > best_val and best_val < 1e8:
        # return polished incumbent if our search failed
        return best
    return _norm(cur_best)
