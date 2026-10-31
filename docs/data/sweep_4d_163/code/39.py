import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _smooth_full(x, iters, T0, T1, lr0, lr1):
    n, d = x.shape
    for k in range(iters):
        frac = k / max(iters - 1, 1)
        T = T0 * (T1 / T0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp((G - m) / T)
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        rs = W.sum(axis=1, keepdims=True)
        step = gt / np.maximum(rs, 1e-12)
        sn = np.linalg.norm(step, axis=1, keepdims=True)
        step = step / np.maximum(sn, 1.0)
        x = x - lr * step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

_IU = None
def _lp_polish(x, seconds, r0=2e-3, cap=1500):
    global _IU
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    if _IU is None or _IU[0].shape[0] != n * (n - 1) // 2:
        _IU = np.triu_indices(n, 1)
    iu = _IU
    best = _max_cos(x)
    radius = r0
    start = time.time()
    while time.time() - start < seconds and radius > 1e-13:
        G = x @ x.T
        c_all = G[iu]
        m = float(c_all.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = c_all > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > cap:
            v = c_all[sel]
            kth = np.sort(v)[-cap]
            sel = c_all >= kth
            cnt2 = int(np.count_nonzero(sel))
            if cnt2 > cap + 400:
                ids = np.where(sel)[0]
                keep = ids[::2][:cap]
                ns = np.zeros_like(sel)
                ns[keep] = True
                sel = ns
        I = iu[0][sel]
        J = iu[1][sel]
        c = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cd = np.arange(d)
        cols = np.concatenate([I[:, None] * d + cd, J[:, None] * d + cd,
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.6, 0.03)
        else:
            radius *= 0.5
    return x

def _expand_SP(Y, F, m):
    # Y (q,4), F (nf,4) or None -> X (nf+q*m,4)
    if Y is None or len(Y) == 0:
        return F.copy()
    q = Y.shape[0]
    c = np.cos(2.0 * np.pi * np.arange(m) / m)
    s = np.sin(2.0 * np.pi * np.arange(m) / m)
    Y1 = Y[:, 0:1]
    Y2 = Y[:, 1:2]
    Y3 = Y[:, 2:3]
    Y4 = Y[:, 3:4]
    Xo0 = Y1 * c[None, :] - Y2 * s[None, :]
    Xo1 = Y1 * s[None, :] + Y2 * c[None, :]
    Xo2 = np.broadcast_to(Y3, (q, m))
    Xo3 = np.broadcast_to(Y4, (q, m))
    Xo = np.stack([Xo0, Xo1, Xo2, Xo3], axis=2).reshape(q * m, 4)
    if F is not None and len(F) > 0:
        return np.concatenate([F, Xo], axis=0)
    return Xo

def _sym_smooth(Y0, F0, m, iters, T0, T1, lr0, lr1, pin_first=True):
    q = Y0.shape[0]
    Y = Y0 / np.linalg.norm(Y0, axis=1, keepdims=True)
    if F0 is not None and len(F0) > 0:
        F = F0 / np.linalg.norm(F0, axis=1, keepdims=True)
        F[:, 0:2] = 0.0
        F /= np.linalg.norm(F, axis=1, keepdims=True)
        nf = F.shape[0]
    else:
        F = None
        nf = 0
    c = np.cos(2.0 * np.pi * np.arange(m) / m)
    s = np.sin(2.0 * np.pi * np.arange(m) / m)
    for k in range(iters):
        frac = k / max(iters - 1, 1)
        T = T0 * (T1 / T0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        X = _expand_SP(Y, F, m)
        N = X.shape[0]
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        mv = float(G.max())
        W = np.exp((G - mv) / T)
        np.fill_diagonal(W, 0.0)
        grad = W @ X
        dot = np.sum(grad * X, axis=1, keepdims=True)
        gt = grad - dot * X
        rs = W.sum(axis=1, keepdims=True)
        stepX = gt / np.maximum(rs, 1e-12)
        sn = np.linalg.norm(stepX, axis=1, keepdims=True)
        stepX = stepX / np.maximum(sn, 1.0)
        # fold orbit part
        S = stepX[nf:].reshape(q, m, 4)
        g0 = S[:, :, 0]
        g1 = S[:, :, 1]
        gr0 = g0 * c[None, :] + g1 * s[None, :]
        gr1 = -g0 * s[None, :] + g1 * c[None, :]
        gr2 = S[:, :, 2]
        gr3 = S[:, :, 3]
        mean = np.stack([gr0.mean(axis=1), gr1.mean(axis=1),
                         gr2.mean(axis=1), gr3.mean(axis=1)], axis=1)
        dY = np.sum(mean * Y, axis=1, keepdims=True)
        mt = mean - dY * Y
        nm = np.linalg.norm(mt, axis=1, keepdims=True) + 1e-12
        Y = Y - lr * (mt / nm)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        # fixed points (keep first pinned to fix gauge)
        if F is not None and nf > 0:
            sF = stepX[:nf].copy()
            sF[:, 0:2] = 0.0
            dF = np.sum(sF * F, axis=1, keepdims=True)
            mtF = sF - dF * F
            mtF[:, 0:2] = 0.0
            nmF = np.linalg.norm(mtF, axis=1, keepdims=True) + 1e-12
            upd = lr * (mtF / nmF)
            if pin_first and nf >= 1:
                upd[0, :] = 0.0
            F = F - upd
            F[:, 0:2] = 0.0
            F /= np.linalg.norm(F, axis=1, keepdims=True)
    return Y, F

def solve():
    rng = np.random.default_rng(12345)
    t0 = time.time()
    HARD = 168.0
    xb = None
    xr = None
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy")
    except Exception:
        xr = None
    cands = []
    for cc in (xb, xr):
        if cc is not None:
            try:
                cc = np.asarray(cc, dtype=float)
                if cc.shape == (163, 4):
                    cc = cc / np.linalg.norm(cc, axis=1, keepdims=True)
                    cc = np.nan_to_num(cc)
                    cc = cc / np.linalg.norm(cc, axis=1, keepdims=True)
                    cands.append(cc)
            except Exception:
                pass
    if not cands:
        z = rng.normal(size=(163, 4))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        cands.append(z)
    best = None
    best_val = 1e9
    for cc in cands:
        v = _max_cos(cc)
        if v < best_val:
            best_val = v
            best = cc.copy()
    # symmetry-constrained search: (m, nfix) with nfix+q*m=163
    combos = [(9, 1), (6, 1), (10, 3), (8, 3), (9, 10), (6, 7)]
    for (m, nfix) in combos:
        if time.time() - t0 > HARD - 50:
            break
        q = (163 - nfix) // m
        if nfix + q * m != 163 or q <= 0:
            continue
        # init representatives with small 12-radius for feasibility
        Y0 = rng.normal(size=(q, 4))
        Y0[:, 0:2] *= 0.45
        Y0 /= np.linalg.norm(Y0, axis=1, keepdims=True)
        if nfix == 1:
            F0 = np.array([[0.0, 0.0, 0.0, 1.0]])
        else:
            ang = np.linspace(0, 2 * np.pi, nfix, endpoint=False) + float(rng.random())
            F0 = np.zeros((nfix, 4))
            F0[:, 2] = np.cos(ang)
            F0[:, 3] = np.sin(ang)
        try:
            Yopt, Fopt = _sym_smooth(Y0, F0, m, iters=380, T0=0.020, T1=0.002,
                                     lr0=0.025, lr1=0.003)
            Xsym = _expand_SP(Yopt, Fopt, m)
            Xsym /= np.linalg.norm(Xsym, axis=1, keepdims=True)
        except Exception:
            continue
        # release symmetry: one or two noise levels
        for scale in (0.012, 0.030):
            if time.time() - t0 > HARD - 40:
                break
            try:
                Xb = Xsym + rng.normal(scale=scale, size=Xsym.shape)
                Xb /= np.linalg.norm(Xb, axis=1, keepdims=True)
                Xb = _smooth_full(Xb, iters=200, T0=0.015, T1=0.0015, lr0=0.008, lr1=0.001)
                remain = HARD - (time.time() - t0)
                if remain > 22:
                    Xb = _lp_polish(Xb, seconds=min(1.1, remain - 20.0))
                v = _max_cos(Xb)
                if v < best_val - 1e-12:
                    best = Xb.copy()
                    best_val = v
            except Exception:
                continue
    # small basin-hopping from incumbent to squeeze further
    cur = best.copy()
    cur_val = best_val
    for trial in range(8):
        if time.time() - t0 > HARD - 28:
            break
        try:
            sig = 0.03 if (trial % 2 == 0) else 0.08
            y = cur + rng.normal(scale=sig, size=cur.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth_full(y, iters=180, T0=0.012, T1=0.0012, lr0=0.006, lr1=0.0009)
            remain = HARD - (time.time() - t0)
            if remain > 24:
                y = _lp_polish(y, seconds=min(0.9, remain - 22.0))
            v = _max_cos(y)
            if v < best_val - 1e-12:
                best = y.copy()
                best_val = v
                cur = y.copy()
                cur_val = v
            elif v < cur_val:
                cur = y.copy()
                cur_val = v
        except Exception:
            continue
    try:
        left = HARD - (time.time() - t0) - 1.0
        if left > 2.0:
            best = _lp_polish(best, seconds=min(left, 25.0), r0=1.5e-3)
    except Exception:
        pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    best = np.nan_to_num(best)
    best /= np.linalg.norm(best, axis=1, keepdims=True)
    return best
