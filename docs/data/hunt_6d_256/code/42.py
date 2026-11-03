import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _signs64():
    S = np.empty((64, 6))
    for m in range(64):
        for i in range(6):
            S[m, i] = 1.0 if ((m >> i) & 1) == 0 else -1.0
    return S

_SIGNS = _signs64()

def _expand(S):
    # S (ns,6) -> (ns*64,6), X = S[a]*signs
    ns = S.shape[0]
    X = np.empty((ns * 64, 6))
    for a in range(ns):
        X[a*64:(a+1)*64, :] = S[a][None, :] * _SIGNS
    # normalize rows (S normalized => X normalized)
    return _norm(X)

def _enforce_min(S, m=0.12):
    S = S.copy()
    for _ in range(2):
        nr = np.linalg.norm(S, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        S = S / nr
        small = np.abs(S) < m
        if not np.any(small):
            break
        sgn = np.sign(S)
        sgn[sgn == 0] = 1.0
        S[small] = m * sgn[small]
    nr = np.linalg.norm(S, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    return S / nr

def _cluster_protos(Xinc, rng):
    A = np.abs(Xinc)  # (256,6), positive
    # k-means k=4 on abs vectors
    idx = rng.choice(A.shape[0], 4, replace=False)
    C = A[idx].copy()
    for _ in range(50):
        d2 = ((A[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        a = d2.argmin(1)
        Cn = C.copy()
        for k in range(4):
            pts = A[a == k]
            if len(pts) > 0:
                Cn[k] = pts.mean(0)
        if np.allclose(Cn, C):
            C = Cn
            break
        C = Cn
    # ensure spread: if two centers too close, perturb
    C = np.maximum(C, 0.12)
    C = _norm(C)
    C = _enforce_min(C, 0.12)
    return C

def _sym_adam(S0, steps, K0, K1, lr):
    S = _enforce_min(S0.copy(), 0.12)
    ns = S.shape[0]
    m = np.zeros_like(S)
    v = np.zeros_like(S)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    la0, la1 = np.log(K0), np.log(K1)
    # initial best
    Xb0 = _expand(S)
    best_val = _max_cos(Xb0)
    best_S = S.copy()
    for t in range(1, steps + 1):
        K = float(np.exp(la0 + (la1 - la0) * t / steps))
        X = _expand(S)
        n = X.shape[0]
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        # avoid overflow: clip
        E = K * (G - gmax)
        np.clip(E, -60.0, 0.0, out=E)
        W = np.exp(E)
        np.fill_diagonal(W, 0.0)
        Ssum = float(W.sum() * 0.5)
        if not np.isfinite(Ssum) or Ssum <= 0:
            continue
        gradX = (W @ X) / Ssum  # (256,6)
        GX = gradX.reshape(ns, 64, 6)
        # gradS[a,l] = sum_k GX[a,k,l]*signs[k,l]
        gradS = (GX * _SIGNS[None, :, :]).sum(axis=1)
        dot = np.sum(gradS * S, axis=1, keepdims=True)
        gt = gradS - dot * S
        b1p *= b1
        b2p *= b2
        m *= b1
        m += (1 - b1) * gt
        v *= b2
        v += (1 - b2) * (gt * gt)
        mh = m / (1 - b1p)
        vh = v / (1 - b2p)
        step = lr * mh / (np.sqrt(vh) + eps)
        sdot = np.sum(step * S, axis=1, keepdims=True)
        step = step - sdot * S
        S = S - step
        S = _enforce_min(S, 0.12)
        if t % 150 == 0 or t == steps:
            Xc = _expand(S)
            val = _max_cos(Xc)
            if val < best_val:
                best_val = val
                best_S = S.copy()
    return best_S, _expand(best_S), best_val

def _proto_lp(S0, deadline, r0=2e-3, cap=4000):
    S = _enforce_min(S0.copy(), 0.12)
    ns, d = S.shape
    X = _expand(S)
    best = _max_cos(X)
    best_S = S.copy()
    radius = r0
    iuI, iuJ = np.triu_indices(X.shape[0], 1)
    while time.time() < deadline and radius > 1e-12:
        X = _expand(S)
        G = X @ X.T
        c = G[iuI, iuJ]
        mv = float(c.max())
        margin = max(20.0 * radius, 8e-5)
        sel = c > (mv - margin)
        I = iuI[sel]; J = iuJ[sel]
        csel = c[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > cap:
            part = np.argpartition(-csel, cap)[:cap]
            I = I[part]; J = J[part]; csel = csel[part]
            p = cap
        # build A_ub: p x (ns*d+1)
        nd = ns * d
        rows = np.repeat(np.arange(p), 2 * d + 1)
        # coeffs
        # for each pair, coeff vector length nd
        Coef = np.zeros((p, nd))
        for r in range(p):
            i = int(I[r]); j = int(J[r])
            a = i // 64; ka = i % 64
            b = j // 64; kb = j % 64
            sa = _SIGNS[ka]; sb = _SIGNS[kb]
            xi = X[i]; xj = X[j]
            # ds_a contribution: sa*xj, ds_b contribution: sb*xi
            ca = sa * xj  # (6,)
            cb = sb * xi
            if a == b:
                Coef[r, a*d:(a+1)*d] = ca + cb
            else:
                Coef[r, a*d:(a+1)*d] = ca
                Coef[r, b*d:(b+1)*d] = cb
        cols_ds = np.tile(np.arange(nd), (p, 1))
        rr = np.repeat(np.arange(p), nd)
        cc = cols_ds.ravel()
        vv = Coef.ravel()
        # add t column
        rr2 = np.concatenate([rr, np.arange(p)])
        cc2 = np.concatenate([cc, np.full(p, nd)])
        vv2 = np.concatenate([vv, -np.ones(p)])
        try:
            A_ub = coo_matrix((vv2, (rr2, cc2)), shape=(p, nd + 1)).tocsr()
            # equality: S_a . ds_a = 0
            er = np.repeat(np.arange(ns), d)
            ec = np.arange(nd)
            ev = S.ravel().copy()
            A_eq = coo_matrix((ev, (er, ec)), shape=(ns, nd + 1)).tocsr()
        except Exception:
            break
        obj = np.zeros(nd + 1); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(ns),
                          bounds=[(-radius, radius)] * nd + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        ds = res.x[:-1].reshape(ns, d)
        Sn = _enforce_min(S + ds, 0.12)
        val = _max_cos(_expand(Sn))
        if val < best:
            S = Sn
            best = val
            best_S = S.copy()
            radius = min(radius * 1.4, 0.02)
        else:
            radius *= 0.5
    return best_S, _expand(best_S), best

def _adam_sm(x0, steps, K0, K1, lr):
    x = x0.copy()
    n, d = x.shape
    m = np.zeros_like(x); v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    la0, la1 = np.log(K0), np.log(K1)
    best_val = _max_cos(x); best_x = x.copy()
    for t in range(1, steps + 1):
        K = float(np.exp(la0 + (la1 - la0) * t / steps))
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        E = K * (G - gmax)
        np.clip(E, -60.0, 0.0, out=E)
        W = np.exp(E)
        np.fill_diagonal(W, 0.0)
        Ssum = float(W.sum() * 0.5)
        if not np.isfinite(Ssum) or Ssum <= 0:
            continue
        grad = (W @ x) / Ssum
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        b1p *= b1; b2p *= b2
        m *= b1; m += (1 - b1) * gt
        v *= b2; v += (1 - b2) * (gt * gt)
        mh = m / (1 - b1p); vh = v / (1 - b2p)
        step = lr * mh / (np.sqrt(vh) + eps)
        sdot = np.sum(step * x, axis=1, keepdims=True)
        step = step - sdot * x
        x = x - step
        x = _norm(x)
        if t % 150 == 0 or t == steps:
            val = _max_cos(x)
            if val < best_val:
                best_val = val; best_x = x.copy()
    return best_x, best_val

def _lp_polish(x, deadline, r0=2e-3, cap=3500):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    iuI, iuJ = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iuI, iuJ]
        mv = float(c.max())
        margin = max(20.0 * radius, 8e-5)
        sel = c > (mv - margin)
        I = iuI[sel]; J = iuJ[sel]
        csel = c[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > cap:
            part = np.argpartition(-csel, cap)[:cap]
            I = I[part]; J = J[part]; csel = csel[part]
            p = cap
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None]*d + np.arange(d),
                               J[:, None]*d + np.arange(d),
                               np.full((p, 1), n*d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n*d+1)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n*d))),
                              shape=(n, n*d+1)).tocsr()
        except Exception:
            break
        obj = np.zeros(n*d+1); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)]*(n*d)+[(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        val = _max_cos(y)
        if val < best:
            x, best = y, val
            radius = min(radius*1.4, 0.02)
        else:
            radius *= 0.5
    return x

def solve():
    import time as _time
    t0 = _time.time()
    T_END = t0 + 172.0
    rng = np.random.default_rng(12345)
    # load incumbent for clustering + fallback
    Xinc = None
    x_best = None; best = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=float)
            if a.shape == (256, 6) and np.all(np.isfinite(a)):
                a = _norm(a)
                vv = _max_cos(a)
                if vv < best:
                    best = vv; x_best = a.copy()
                if Xinc is None:
                    Xinc = a.copy()
        except Exception:
            continue
    if Xinc is None:
        Xinc = _norm(rng.normal(size=(256, 6)))
    if x_best is None:
        x_best = Xinc.copy(); best = _max_cos(x_best)
    fallback = x_best.copy(); fallback_val = best
    # --- symmetry construction from clustered |x| ---
    try:
        S0 = _cluster_protos(Xinc, rng)
    except Exception:
        S0 = _norm(np.abs(rng.normal(size=(4, 6))) + 0.2)
        S0 = _enforce_min(S0, 0.12)
    # sym softmax high-K
    rem = T_END - _time.time()
    S_sym, X_sym, v_sym = S0, _expand(S0), _max_cos(_expand(S0))
    try:
        if rem > 40:
            S_sym, X_sym, v_sym = _sym_adam(S0, steps=1100, K0=200.0, K1=20000.0, lr=0.006)
    except Exception:
        pass
    try:
        if _time.time() < T_END - 45:
            S_sym, X_sym, v_sym = _proto_lp(S_sym, min(_time.time()+6.0, T_END-40), r0=2e-3, cap=4000)
    except Exception:
        pass
    # staged symmetry breaking
    amps = [0.0, 1e-4, 1e-3, 1e-2]
    best_stage_X = None
    # keep track: start from X_sym
    for si, amp in enumerate(amps):
        if _time.time() > T_END - 18:
            break
        try:
            x0 = X_sym.copy()
            if amp > 0:
                Z = rng.normal(size=x0.shape)
                dproj = np.sum(Z * x0, axis=1, keepdims=True)
                Zt = Z - dproj * x0
                zn = np.linalg.norm(Zt, axis=1, keepdims=True) + 1e-16
                Zt = Zt / zn * amp
                x0 = _norm(x0 + Zt)
            # full release Adam
            rem2 = T_END - _time.time()
            if rem2 < 12:
                break
            steps = 500 if rem2 > 50 else 350
            xp, vp = _adam_sm(x0, steps, 80.0, 2200.0, 0.008)
            xp = _lp_polish(xp, min(_time.time()+3.5, T_END-8), r0=1.5e-3, cap=3500)
            vp = _max_cos(xp)
            if best_stage_X is None or vp < _max_cos(best_stage_X):
                best_stage_X = xp.copy()
                # update global best if beats fallback
                if vp < best:
                    best = vp; x_best = xp.copy()
        except Exception:
            continue
    # if symmetry path failed to beat fallback, try quick polish of fallback with remaining time
    if _time.time() < T_END - 10:
        try:
            # light polish fallback to ensure best possible
            xf = _lp_polish(fallback, min(_time.time()+3.0, T_END-5), r0=1.2e-3, cap=3500)
            vf = _max_cos(xf)
            if vf < best:
                best = vf; x_best = xf.copy()
        except Exception:
            pass
    if x_best is None:
        x_best = fallback
    x_best = _norm(np.asarray(x_best, dtype=float))
    # strict finite check
    x_best[~np.isfinite(x_best)] = 0.0
    # renormalize zero rows if any (should not happen)
    nr = np.linalg.norm(x_best, axis=1, keepdims=True)
    bad = (nr.squeeze() < 1e-12)
    if np.any(bad):
        x_best[bad] = _norm(rng.normal(size=(int(np.sum(bad)), 6)))
    x_best = _norm(x_best)
    return x_best
