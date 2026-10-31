import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _lp_polish(x, deadline):
    x = _norm(x.copy())
    n, d = x.shape
    best = _maxcos(x)
    bx = x.copy()
    radius = 2.5e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(np.max(c))
        thr = m - max(20.0 * radius, 1e-9)
        idx = np.where(c > thr)[0]
        if idx.size == 0:
            break
        if idx.size > 1400:
            o = np.argsort(c[idx])[-1400:]
            idx = idx[o]
        sel = np.zeros_like(c, dtype=bool)
        sel[idx] = True
        I = iu[0][sel]; J = iu[1][sel]
        cs = c[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _maxcos(y)
        if v < best:
            best = v; bx = y.copy(); x = y
            radius = min(radius * 1.5, 0.05)
        else:
            if v <= best + 1e-15:
                x = y
            radius *= 0.5
    return bx, best

# group ops (row vectors)
def _g_single(v):
    y = np.empty_like(v)
    y[0] = -v[5]; y[1] = v[0]; y[2] = v[1]; y[3] = v[2]; y[4] = v[3]; y[5] = v[4]
    return y
def _ginv_single(v):
    y = np.empty_like(v)
    y[5] = -v[0]; y[0] = v[1]; y[1] = v[2]; y[2] = v[3]; y[3] = v[4]; y[4] = v[5]
    return y
def _apply_g_batch(X):
    Y = np.empty_like(X)
    Y[:, 0] = -X[:, 5]; Y[:, 1] = X[:, 0]; Y[:, 2] = X[:, 1]
    Y[:, 3] = X[:, 2]; Y[:, 4] = X[:, 3]; Y[:, 5] = X[:, 4]
    return Y
def _apply_ginv_batch(X):
    Y = np.empty_like(X)
    Y[:, 5] = -X[:, 0]; Y[:, 0] = X[:, 1]; Y[:, 1] = X[:, 2]
    Y[:, 2] = X[:, 3]; Y[:, 3] = X[:, 4]; Y[:, 4] = X[:, 5]
    return Y

def _fix_basis():
    M = np.zeros((6, 6))
    M[5, 0] = -1.0; M[0, 1] = 1.0; M[1, 2] = 1.0; M[2, 3] = 1.0; M[3, 4] = 1.0; M[4, 5] = 1.0
    M4 = np.linalg.matrix_power(M, 4)
    A = (M4 - np.eye(6)).T
    _, _, Vt = np.linalg.svd(A)
    u0 = Vt[-2, :].copy(); u1 = Vt[-1, :].copy()
    # orthonormalize
    u0 = u0 / np.linalg.norm(u0)
    u1 = u1 - float(u0 @ u1) * u0
    u1 = u1 / (np.linalg.norm(u1) + 1e-15)
    return u0, u1, M

def _expand_orbit(Rgen, thetas, u0, u1):
    # Rgen (9,6) unit, thetas (2,)
    X = np.empty((116, 6))
    p = 0
    for j in range(Rgen.shape[0]):
        cur = Rgen[j].copy()
        for _ in range(12):
            X[p] = cur; p += 1
            cur = _g_single(cur)
    for th in thetas:
        s = float(np.cos(th)) * u0 + float(np.sin(th)) * u1
        n = float(np.linalg.norm(s)); s = s / (n + 1e-15)
        cur = s.copy()
        for _ in range(4):
            X[p] = cur; p += 1
            cur = _g_single(cur)
    return X

def _antipodal_adam(Y0, steps, b0, b1, lr0, lr1):
    Y = _norm(Y0.copy())
    m = np.zeros_like(Y); v = np.zeros_like(Y)
    b1p = 1.0; b2p = 1.0
    be1, be2, eps = 0.9, 0.999, 1e-8
    for it in range(1, steps + 1):
        f = it / float(max(steps, 1))
        beta = b0 * (b1 / b0) ** f
        T = 1.0 / beta
        lr = lr0 + (lr1 - lr0) * f
        G = Y @ Y.T
        np.fill_diagonal(G, 0.0)
        # max abs off-diag
        am = float(np.max(np.abs(G)))
        Wp = np.exp((G - am) / T)
        Wn = np.exp((-G - am) / T)
        np.fill_diagonal(Wp, 0.0); np.fill_diagonal(Wn, 0.0)
        S = float(np.sum(Wp) + np.sum(Wn))
        if not np.isfinite(S) or S <= 0:
            break
        Grad = ((Wp - Wn) / S) @ Y
        proj = np.sum(Grad * Y, axis=1, keepdims=True)
        Gt = Grad - proj * Y
        b1p *= be1; b2p *= be2
        m *= be1; m += (1.0 - be1) * Gt
        v *= be2; v += (1.0 - be2) * Gt * Gt
        mh = m / (1.0 - b1p); vh = v / (1.0 - b2p)
        step = lr * mh / (np.sqrt(vh) + eps)
        # clip per-point
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-15
        cap = 0.15
        sc = np.minimum(1.0, cap / (sn * 1.0 + 1e-15))
        # simpler: global clip
        Y = _norm(Y - step)
    return Y

def _unconstrained_adam(X0, steps, b0, b1, lr0, lr1):
    X = _norm(X0.copy())
    m = np.zeros_like(X); v = np.zeros_like(X)
    b1p = 1.0; b2p = 1.0
    be1, be2, eps = 0.9, 0.999, 1e-8
    for it in range(1, steps + 1):
        f = it / float(max(steps, 1))
        beta = b0 * (b1 / b0) ** f
        T = 1.0 / beta
        lr = lr0 + (lr1 - lr0) * f
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        cmax = float(np.max(G))
        W = np.exp((G - cmax) / T)
        np.fill_diagonal(W, 0.0)
        S = float(np.sum(W))
        if not np.isfinite(S) or S <= 0:
            break
        Grad = (W / S) @ X
        proj = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - proj * X
        b1p *= be1; b2p *= be2
        m *= be1; m += (1.0 - be1) * Gt
        v *= be2; v += (1.0 - be2) * Gt * Gt
        mh = m / (1.0 - b1p); vh = v / (1.0 - b2p)
        step = lr * mh / (np.sqrt(vh) + eps)
        X = _norm(X - step)
    return X

def _orbit_adam(R0, th0, u0, u1, steps, b0, b1, lr0, lr1, lrth):
    R = _norm(R0.copy())
    th = th0.copy().astype(float)
    mR = np.zeros_like(R); vR = np.zeros_like(R)
    mt = np.zeros_like(th); vt = np.zeros_like(th)
    b1p = 1.0; b2p = 1.0
    be1, be2, eps = 0.9, 0.999, 1e-8
    ng = R.shape[0]
    for it in range(1, steps + 1):
        f = it / float(max(steps, 1))
        beta = b0 * (b1 / b0) ** f
        T = 1.0 / beta
        lr = lr0 + (lr1 - lr0) * f
        X = _expand_orbit(R, th, u0, u1)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        cmax = float(np.max(G))
        W = np.exp((G - cmax) / T)
        np.fill_diagonal(W, 0.0)
        S = float(np.sum(W))
        if not np.isfinite(S) or S <= 0:
            break
        GradF = (W / S) @ X  # (116,6)
        # pullback to generic
        GradR = np.zeros_like(R)
        p = 0
        for j in range(ng):
            acc = np.zeros(6)
            for k in range(12):
                gk = GradF[p + k].copy()
                # apply inv^k
                for _ in range(k):
                    gk = _ginv_single(gk)
                acc += gk
            GradR[j] = acc
            p += 12
        # pullback to thetas
        GradT = np.zeros_like(th)
        for s_i in range(len(th)):
            acc = np.zeros(6)
            for k in range(4):
                gk = GradF[p + k].copy()
                for _ in range(k):
                    gk = _ginv_single(gk)
                acc += gk
            p += 4
            tvec = -float(np.sin(th[s_i])) * u0 + float(np.cos(th[s_i])) * u1
            GradT[s_i] = float(acc @ tvec)
        # Adam for R (tangent)
        proj = np.sum(GradR * R, axis=1, keepdims=True)
        GtR = GradR - proj * R
        b1p *= be1; b2p *= be2
        mR *= be1; mR += (1.0 - be1) * GtR
        vR *= be2; vR += (1.0 - be2) * GtR * GtR
        mhR = mR / (1.0 - b1p); vhR = vR / (1.0 - b2p)
        stepR = lr * mhR / (np.sqrt(vhR) + eps)
        R = _norm(R - stepR)
        # Adam for theta (euclidean)
        mt *= be1; mt += (1.0 - be1) * GradT
        vt *= be2; vt += (1.0 - be2) * GradT * GradT
        mhT = mt / (1.0 - b1p); vhT = vt / (1.0 - b2p)
        stepT = lrth * mhT / (np.sqrt(vhT) + eps)
        th = th - stepT
    return R, th

def solve():
    t0 = time.time()
    HARD_END = t0 + 171.0
    rng = np.random.default_rng(7)
    u0, u1, _M = _fix_basis()
    best_X = None
    best_v = 1e9
    # ---- Phase A: antipodal search ----
    # time slice ~ up to 55s
    A_END = t0 + 55.0
    anti_cands = []  # list of (v, Xfull)
    nrest = 0
    while time.time() < A_END and time.time() < HARD_END - 60.0:
        nrest += 1
        Y0 = _norm(rng.standard_normal((58, 6)))
        # vary steps/beta slightly per restart
        steps = int(rng.integers(500, 900))
        # beta 50 -> 1500 anneal
        Y = _antipodal_adam(Y0, steps, 50.0, 1500.0, 0.02, 0.004)
        Xfull = np.vstack([Y, -Y])
        Xfull = _norm(Xfull)
        v = _maxcos(Xfull)
        anti_cands.append((v, Xfull.copy()))
        if v < best_v:
            best_v = v; best_X = Xfull.copy()
        # quick extra polish of promising with more steps
        if v < 0.62 and time.time() < A_END - 1.0:
            Y2 = _antipodal_adam(Y, 400, 200.0, 2000.0, 0.005, 0.001)
            X2 = _norm(np.vstack([Y2, -Y2]))
            v2 = _maxcos(X2)
            anti_cands.append((v2, X2.copy()))
            if v2 < best_v:
                best_v = v2; best_X = X2.copy()
        if nrest > 120:
            break
    anti_cands.sort(key=lambda z: z[0])
    # ---- Phase B: orbit search ----
    B_END = t0 + 100.0
    orb_cands = []
    nrest = 0
    while time.time() < B_END and time.time() < HARD_END - 45.0:
        nrest += 1
        R0 = _norm(rng.standard_normal((9, 6)))
        th0 = rng.uniform(0, np.pi / 2, size=(2,))
        steps = int(rng.integers(400, 700))
        R, th = _orbit_adam(R0, th0, u0, u1, steps, 50.0, 1200.0, 0.02, 0.004, 0.01)
        Xfull = _expand_orbit(R, th, u0, u1)
        Xfull = _norm(Xfull)
        v = _maxcos(Xfull)
        orb_cands.append((v, Xfull.copy(), R.copy(), th.copy()))
        if v < best_v:
            best_v = v; best_X = Xfull.copy()
        if nrest > 60:
            break
    orb_cands.sort(key=lambda z: z[0])
    # ---- Phase C: release + unconstrained re-opt ----
    # pick top constrained candidates
    released = []
    if anti_cands:
        for k in range(min(2, len(anti_cands))):
            released.append(anti_cands[k][1])
    if orb_cands:
        for k in range(min(2, len(orb_cands))):
            released.append(orb_cands[k][1])
    # also random fresh for diversity if time
    # perturb and re-optimise
    C_END = t0 + 145.0
    reopt_best = []
    sigmas = [0.002, 0.006, 0.012]
    ri = 0
    for base in released:
        for sig in sigmas:
            if time.time() > C_END or time.time() > HARD_END - 28.0:
                break
            ri += 1
            Xn = _norm(base + rng.normal(scale=sig, size=base.shape))
            # annealed softmax unconstrained
            Xs = _unconstrained_adam(Xn, 700, 40.0, 1500.0, 0.02, 0.003)
            v = _maxcos(Xs)
            reopt_best.append((v, Xs))
            if v < best_v:
                best_v = v; best_X = Xs.copy()
    reopt_best.sort(key=lambda z: z[0])
    # LP polish best 2-3 released
    for k in range(min(3, len(reopt_best))):
        if time.time() > HARD_END - 26.0:
            break
        Xp, vp = _lp_polish(reopt_best[k][1], min(time.time() + 6.0, HARD_END - 20.0))
        if vp < best_v:
            best_v = vp; best_X = Xp.copy()
    # also polish current best_X once more if not yet
    if time.time() < HARD_END - 22.0 and best_X is not None:
        Xp, vp = _lp_polish(best_X, min(time.time() + 8.0, HARD_END - 16.0))
        if vp <= best_v:
            best_v = vp; best_X = Xp.copy()
    # ---- Phase D: final comparison with incumbent files ----
    def _clean(a):
        try:
            a = np.asarray(a, dtype=float)
        except Exception:
            return None
        if a.shape != (116, 6) or not np.all(np.isfinite(a)):
            return None
        n = np.linalg.norm(a, axis=1, keepdims=True)
        if np.any(n < 1e-12):
            return None
        return a / n
    incumb = None; incumb_v = 1e9
    for f in ("best.npy", "record.npy"):
        try:
            c = _clean(np.load(f))
            if c is not None:
                vv = _maxcos(c)
                if vv < incumb_v:
                    incumb_v = vv; incumb = c
        except Exception:
            pass
    if incumb is not None and incumb_v < best_v:
        # give incumbent a short LP touch if time allows
        if time.time() < HARD_END - 8.0:
            Xp, vp = _lp_polish(incumb, min(time.time() + 6.0, HARD_END - 2.0))
            if vp < incumb_v:
                incumb = Xp; incumb_v = vp
        best_X = incumb.copy(); best_v = incumb_v
    if best_X is None:
        best_X = _norm(rng.standard_normal((116, 6)))
    best_X = _norm(best_X)
    return best_X
