import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _mc(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _adam_relax(x, steps, lr, T0, T1, deadline=1e18):
    n, d = x.shape
    m_ad = np.zeros_like(x)
    v_ad = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p = 1.0
    b2p = 1.0
    if steps > 1:
        logr = np.log(max(T1, 1e-12) / max(T0, 1e-12)) / (steps - 1)
    else:
        logr = 0.0
    for t in range(1, steps + 1):
        if t % 200 == 0 and time.time() > deadline:
            break
        T = T0 * np.exp(logr * (t - 1))
        if T < 1e-12:
            T = 1e-12
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        mv = G.max()
        E = np.exp((G - mv) / T)
        np.fill_diagonal(E, 0.0)
        S = E.sum() * 0.5
        if S <= 0 or not np.isfinite(S):
            break
        Grad = (E @ x) / S
        dot = (Grad * x).sum(axis=1, keepdims=True)
        Gt = Grad - dot * x
        b1p *= b1
        b2p *= b2
        m_ad = b1 * m_ad + (1.0 - b1) * Gt
        v_ad = b2 * v_ad + (1.0 - b2) * (Gt * Gt)
        mhat = m_ad / (1.0 - b1p)
        vhat = v_ad / (1.0 - b2p)
        step = mhat / (np.sqrt(vhat) + eps)
        x = x - lr * step
        x = _norm(x)
    return x

def _tangent_bases(X, rng):
    n, d = X.shape
    dm = d - 1
    U = np.empty((n, d, dm))
    for i in range(n):
        xi = X[i]
        R = rng.standard_normal((d, dm))
        R = R - np.outer(xi, (xi @ R))
        Q, _ = np.linalg.qr(R)
        # ensure tangent (fix sign drift due to rank)
        # Q is orthonormal; force orthogonality to xi
        Q = Q - np.outer(xi, (xi @ Q))
        # re-orthogonalize once if needed
        try:
            Q, _ = np.linalg.qr(Q)
        except Exception:
            pass
        Q = Q - np.outer(xi, (xi @ Q))
        nrm = np.linalg.norm(Q, axis=0, keepdims=True)
        nrm[nrm == 0] = 1.0
        Q = Q / nrm
        U[i] = Q
    return U

def _soft_modes(X, U, rng, thresh=2e-5, k=32):
    n, d = X.shape
    dm = d - 1
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    slack = m - G
    iu = np.triu_indices(n, 1)
    sl = slack[iu]
    # diagnostics thresholds (not strictly needed)
    # c1 = int((sl < 1e-6).sum()); c2 = int((sl < 1e-5).sum()); c3 = int((sl < 1e-4).sum())
    sel = sl < thresh
    I = iu[0][sel]
    J = iu[1][sel]
    p = len(I)
    # if too few, widen; if too many, keep tightest
    if p < 30:
        sel2 = sl < 1e-4
        I = iu[0][sel2]
        J = iu[1][sel2]
        p = len(I)
    if p > 3500:
        order = np.argpartition(sl[sel], 3500)[:3500]
        I = I[order]
        J = J[order]
        p = len(I)
    if p < 10:
        return None, None, p
    # assemble sparse Jacobian p x (n*dm)
    N = n * dm
    # values: U_I^T x_J and U_J^T x_I
    # U shape (n,d,dm)
    row = np.repeat(np.arange(p), 2 * dm)
    col = np.empty(p * 2 * dm, dtype=np.int64)
    val = np.empty(p * 2 * dm, dtype=float)
    # vectorized gather
    XI = X[I]  # p x d
    XJ = X[J]
    UI = U[I]  # p x d x dm
    UJ = U[J]
    # A = UI^T XJ : (p x dm)
    A = np.einsum('pdm,pd->pm', UI, XJ)
    B = np.einsum('pdm,pd->pm', UJ, XI)
    col[0::2] = None  # placeholder
    # fill
    # first block
    base_col_I = I[:, None] * dm + np.arange(dm)[None, :]
    base_col_J = J[:, None] * dm + np.arange(dm)[None, :]
    # rows repeated
    # layout: for each pair k, 2*dm entries contiguous
    cols_mat = np.concatenate([base_col_I, base_col_J], axis=1)  # p x 2dm
    vals_mat = np.concatenate([A, B], axis=1)
    col = cols_mat.ravel()
    # rows
    rows_mat = np.repeat(np.arange(p)[:, None], 2 * dm, axis=1).ravel()
    vals = vals_mat.ravel()
    try:
        from scipy.sparse import csr_matrix
        Jmat = coo_matrix((vals, (rows_mat, col)), shape=(p, N)).tocsr()
    except Exception:
        return None, None, p
    kk = int(k)
    if kk >= min(p, N):
        kk = min(p, N) - 1
    if kk < 4:
        return None, None, p
    if kk > 40:
        kk = 40
    try:
        from scipy.sparse.linalg import svds
        v0 = rng.standard_normal(N)
        # smallest magnitude
        u, s, vt = svds(Jmat, k=kk, which='SM', maxiter=2000, tol=1e-4, v0=v0)
        # sort by s ascending (svds returns ascending? ensure)
        o = np.argsort(s)
        s = s[o]
        vt = vt[o, :]
        return vt, s, p
    except Exception:
        return None, None, p

def _geodesic_move(X, U, vflat, step_rad):
    n, d = X.shape
    dm = d - 1
    V = vflat.reshape(n, dm)
    # T_i = U_i @ v_i
    T = np.einsum('ndm,nm->nd', U, V)
    norms = np.linalg.norm(T, axis=1)
    mx = float(norms.max())
    if mx <= 1e-15 or not np.finite(mx):
        return _norm(X.copy())
    scale = step_rad / mx
    T = T * scale
    norms = norms * scale
    Y = X.copy()
    for i in range(n):
        ni = norms[i]
        if ni < 1e-15:
            continue
        ti = T[i] / (np.linalg.norm(T[i]) + 1e-18)
        # geodesic: cos(ni)*x + sin(ni)*t_hat (t_hat already unit approx)
        # renormalize t_hat orthogonal part
        Y[i] = np.cos(ni) * X[i] + np.sin(ni) * ti
    return _norm(Y)

def _lp_polish(x, seconds, deadline=1e18):
    x = _norm(x)
    n, d = x.shape
    best = _mc(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < deadline:
        if radius < 1e-14:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _mc(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    DEAD = t0 + 172.0
    HOP_END = t0 + 132.0
    LP_BUDGET = 25.0
    rng = np.random.default_rng(20261103)
    n, d = 154, 10
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            if a.shape == (n, d):
                a = _norm(a.astype(float))
                v = _mc(a)
                if v < best_val:
                    best_val = float(v)
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.standard_normal((n, d)))
        best_val = _mc(best)
    centre = best.copy()
    centre_val = float(best_val)
    step_sizes = [0.02, 0.06, 0.12, 0.20]
    outer = 0
    # anneal params
    T0 = 3e-3
    T1 = 2e-5
    while time.time() < HOP_END and time.time() < DEAD - LP_BUDGET - 5:
        outer += 1
        # build bases + soft modes at centre (use best as centre mostly)
        # occasionally use perturbed centre to diversify
        if outer % 6 == 0 and outer > 1:
            # small random kick as centre for mode computation
            tmp = centre + rng.standard_normal(centre.shape) * 0.01
            tmp = _norm(tmp)
        else:
            tmp = centre
        try:
            U = _tangent_bases(tmp, rng)
        except Exception:
            U = None
        vt = None
        if U is not None:
            try:
                vt, svals, npairs = _soft_modes(tmp, U, rng, thresh=2e-5, k=32)
            except Exception:
                vt = None
        # if svds failed, make random tangent directions
        ntrials = 8
        trials = []
        if vt is not None:
            kk = vt.shape[0]
            # 8 trials: 2 per step size
            for ti in range(ntrials):
                s = step_sizes[ti % len(step_sizes)]
                coeff = rng.standard_normal(kk)
                # randomize combination sparsity a bit
                if rng.random() < 0.5:
                    mask = rng.random(kk) < 0.5
                    coeff = coeff * mask
                    if np.linalg.norm(coeff) < 1e-9:
                        coeff = rng.standard_normal(kk)
                vflat = coeff @ vt  # (N,)
                # normalize to avoid huge disparity
                nv = float(np.linalg.norm(vflat))
                if nv > 0 and np.isfinite(nv):
                    vflat = vflat / nv
                else:
                    vflat = rng.standard_normal(tmp.size - tmp.shape[0]) * 0.01
                try:
                    # need U aligned with tmp; if tmp != centre, rebuild for centre?
                    # rebuild quickly: use centre bases if tmp != centre
                    if outer % 6 == 0 and outer > 1:
                        Uc = _tangent_bases(centre, rng)
                    else:
                        Uc = U
                    Xt = _geodesic_move(centre, Uc, vflat, s)
                except Exception:
                    continue
                trials.append((Xt, s))
        else:
            # fallback: isotropic tangent noise at multiple scales
            for ti in range(6):
                s = step_sizes[ti % len(step_sizes)]
                Xt = centre + rng.standard_normal(centre.shape) * (s * 0.5)
                Xt = _norm(Xt)
                trials.append((Xt, s))
        # relax each trial
        for (Xt, s) in trials:
            if time.time() > HOP_END or time.time() > DEAD - LP_BUDGET - 5:
                break
            # annealed softmax Adam: fewer steps for large s? use 500-700
            steps = 550
            if s >= 0.12:
                steps = 650
            lr = 0.016
            try:
                Xr = _adam_relax(Xt, steps=steps, lr=lr, T0=T0, T1=T1, deadline=HOP_END)
            except Exception:
                continue
            try:
                v = _mc(Xr)
            except Exception:
                continue
            if v < best_val and np.isfinite(v):
                best_val = float(v)
                best = Xr.copy()
                centre = Xr.copy()
                centre_val = float(v)
            # also allow centre drift if close (to explore plateau)
            # accept if within 2e-4 of best to move along plateau
            elif v < centre_val + 5e-5 and rng.random() < 0.15:
                centre = Xr.copy()
                centre_val = float(v)
        # periodic re-anchor centre to best
        if outer % 4 == 0:
            centre = best.copy()
            centre_val = float(best_val)
        # early exit if time low
        if time.time() > HOP_END:
            break
    # final LP polish (at most 25s), retain best
    remain = DEAD - time.time() - 1.5
    if remain < 3:
        return _norm(best)
    if remain > LP_BUDGET:
        remain = LP_BUDGET
    try:
        polished = _lp_polish(best, seconds=float(remain), deadline=DEAD - 0.5)
        vp = _mc(polished)
        if vp < best_val and np.isfinite(vp):
            best = polished
            best_val = float(vp)
    except Exception:
        pass
    best = _norm(best)
    return best
