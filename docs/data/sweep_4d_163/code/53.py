import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-12)

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _smooth_release(x, iters, T0, T1, lr0=0.07, lr1=0.015):
    n, d = x.shape
    for k in range(iters):
        frac = k / max(iters - 1, 1)
        T = T0 * (T1 / T0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp((G - m) / max(T, 1e-12))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        rs = W.sum(axis=1, keepdims=True)
        step = gt / np.maximum(rs, 1e-12)
        sn = np.linalg.norm(step, axis=1, keepdims=True)
        step = step / np.maximum(sn, 1.0)
        x = x - lr * step
        x = _norm(x)
    return x

_IU = None
def _lp_polish(x, seconds, r0=2e-3, cap=1500):
    global _IU
    x = _norm(x)
    n, d = x.shape
    if _IU is None or _IU[0].shape[0] != n * (n - 1) // 2:
        _IU = np.triu_indices(n, 1)
    iu0, iu1 = _IU
    best = _max_cos(x)
    radius = r0
    start = time.time()
    cd = np.arange(d)
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
        cols = np.concatenate([I[:, None] * d + cd, J[:, None] * d + cd,
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            y = _norm(x + res.x[:-1].reshape(n, d))
        except Exception:
            radius *= 0.5
            continue
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.6, 0.02)
        else:
            radius *= 0.5
        if time.time() - start > seconds:
            break
    return x, best

def _sym_optimize(m, ks, pole, U0, T_sched, maxiter, t_end):
    nrep = len(ks)
    N = 1 + nrep * m
    th = 2.0 * np.pi / m
    s = np.arange(m)
    c1 = np.cos(s * th); s1 = np.sin(s * th)
    c2l = np.stack([np.cos(s * th * k) for k in ks])
    s2l = np.stack([np.sin(s * th * k) for k in ks])
    iu0, iu1 = np.triu_indices(N, 1)

    def expand(R):
        X = np.empty((N, 4))
        X[0] = pole
        for j in range(nrep):
            r0, r1, r2, r3 = R[j]
            c2 = c2l[j]; s2 = s2l[j]
            seg = np.empty((m, 4))
            seg[:, 0] = r0 * c1 - r1 * s1
            seg[:, 1] = r0 * s1 + r1 * c1
            seg[:, 2] = r2 * c2 - r3 * s2
            seg[:, 3] = r2 * s2 + r3 * c2
            X[1 + j * m:1 + (j + 1) * m] = seg
        return X

    def fun_grad(uflat, T):
        U = uflat.reshape(nrep, 4)
        nr = np.linalg.norm(U, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        R = U / nr
        X = expand(R)
        G = X @ X.T
        c_all = G[iu0, iu1]
        m0 = float(c_all.max())
        ex = np.exp((c_all - m0) / max(T, 1e-12))
        S = float(ex.sum())
        sval = m0 + T * np.log(S + 1e-300)
        w = ex / (S + 1e-300)
        W = np.zeros((N, N))
        W[iu0, iu1] = w
        W = W + W.T
        GX = W @ X
        GR = np.zeros((nrep, 4))
        for j in range(nrep):
            Gs = GX[1 + j * m:1 + (j + 1) * m]
            g0 = Gs[:, 0]; g1 = Gs[:, 1]; g2 = Gs[:, 2]; g3 = Gs[:, 3]
            c2 = c2l[j]; s2 = s2l[j]
            GR[j, 0] = float(np.sum(g0 * c1 + g1 * s1))
            GR[j, 1] = float(np.sum(-g0 * s1 + g1 * c1))
            GR[j, 2] = float(np.sum(g2 * c2 + g3 * s2))
            GR[j, 3] = float(np.sum(-g2 * s2 + g3 * c2))
        dot = np.sum(GR * R, axis=1, keepdims=True)
        GU = (GR - dot * R) / nr
        for j in range(nrep):
            nn = float(nr[j, 0])
            GU[j] += 2e-4 * (nn - 1.0) * (U[j] / max(nn, 1e-12))
            sval += 1e-4 * (nn - 1.0) ** 2
        return sval, GU.ravel()

    u = U0.ravel().copy()
    for T in T_sched:
        if time.time() > t_end:
            break
        def f(uu):
            v, g = fun_grad(uu, T)
            return v, g
        try:
            res = minimize(f, u, jac=True, method="L-BFGS-B",
                           options={"maxiter": maxiter, "ftol": 1e-14, "gtol": 1e-10})
            u = res.x
        except Exception:
            break
    U = u.reshape(nrep, 4)
    nr = np.linalg.norm(U, axis=1, keepdims=True)
    R = U / np.maximum(nr, 1e-12)
    X = expand(R)
    X = _norm(X)
    return X, _max_cos(X)

def solve():
    t0 = time.time()
    T_END = t0 + 171.0
    rng0 = np.random.RandomState(0)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng0.randn(163, 4)
    xb = _norm(np.asarray(xb, dtype=float))
    if xb.shape != (163, 4):
        tmp = rng0.randn(163, 4)
        xb = _norm(tmp)
    try:
        xr = np.load("record.npy").astype(float)
        xr = _norm(xr)
        if xr.shape == (163, 4) and _max_cos(xr) < _max_cos(xb):
            xb = xr
    except Exception:
        pass
    best_x = xb.copy()
    best_v = _max_cos(best_x)
    pole = np.array([0.0, 0.0, 0.0, 1.0])

    configs = [
        (27, [1, 2, 4, 5, 7, 8]),
        (27, [1, 2, 4, 5, 7, 11]),
        (27, [1, 1, 1, 1, 1, 1]),
        (27, [2, 2, 2, 2, 2, 2]),
        (27, [1, 2, 5, 7, 10, 11]),
        (54, [1, 5, 7]),
        (81, [1, 2]),
    ]
    T_sched = [0.02, 0.007, 0.002]
    sym_cands = []
    # deterministic multi-start
    seed_base = 100
    for ci, (m, ks) in enumerate(configs):
        if time.time() > T_END - 95.0:
            break
        nrep = len(ks)
        # check size
        if 1 + nrep * m != 163:
            continue
        nstarts = 3 if m == 27 else 2
        for si in range(nstarts):
            if time.time() > T_END - 95.0:
                break
            rng = np.random.RandomState(seed_base + ci * 100 + si * 17)
            U0 = rng.randn(nrep, 4)
            U0 /= np.linalg.norm(U0, axis=1, keepdims=True)
            maxiter = 120 if m == 27 else 100
            try:
                X, v = _sym_optimize(m, ks, pole, U0, T_sched, maxiter, T_END - 90.0)
            except Exception:
                continue
            # degeneracy check: duplicate points -> max cos ~1
            if not np.all(np.isfinite(X)) or v > 0.99:
                # try nearby decomposition fallback: skip
                continue
            sym_cands.append((v, X.copy(), (m, tuple(ks), si)))
            # keep sorted, keep top 6
            sym_cands.sort(key=lambda t: t[0])
            sym_cands = sym_cands[:6]
    # rank sym candidates
    sym_cands.sort(key=lambda t: t[0])
    top_sym = sym_cands[:4]

    # break symmetry: perturb + release + LP
    # always include incumbent as fallback (already best)
    # evaluate top_sym after short release to choose
    rngB = np.random.RandomState(777)
    amps = [0.004, 0.015, 0.045]
    # quick screening: for each top sym, perturb and smooth briefly
    expanded_trials = []
    for ti, (v0, Xs, info) in enumerate(top_sym):
        if time.time() > T_END - 80.0:
            break
        for ai, amp in enumerate(amps):
            if time.time() > T_END - 75.0:
                break
            rng = np.random.RandomState(5000 + ti * 131 + ai * 37)
            Xp = Xs + amp * rng.randn(*Xs.shape)
            Xp = _norm(Xp)
            # short smooth descent
            try:
                Xp = _smooth_release(Xp, iters=220, T0=0.008, T1=0.0008, lr0=0.06, lr1=0.015)
            except Exception:
                continue
            v = _max_cos(Xp)
            expanded_trials.append((v, Xp, (ti, ai)))
    # also add unperturbed smoothed versions
    for ti, (v0, Xs, info) in enumerate(top_sym):
        if time.time() > T_END - 75.0:
            break
        try:
            Xp = _smooth_release(Xs.copy(), iters=220, T0=0.008, T1=0.0008)
            v = _max_cos(Xp)
            expanded_trials.append((v, Xp, (ti, -1)))
        except Exception:
            pass
    expanded_trials.sort(key=lambda t: t[0])
    # polish best few with LP + longer smooth
    for idx in range(min(6, len(expanded_trials))):
        if time.time() > T_END - 12.0:
            break
        v, Xp, _ = expanded_trials[idx]
        # longer smooth
        try:
            if time.time() < T_END - 20.0:
                Xp = _smooth_release(Xp, iters=300, T0=0.004, T1=0.0003, lr0=0.05, lr1=0.01)
                v = _max_cos(Xp)
        except Exception:
            pass
        # LP polish with remaining budget slice
        remain = T_END - 8.0 - time.time()
        if remain < 0.6:
            # just compare
            if v < best_v:
                best_v, best_x = v, Xp.copy()
            continue
        lp_sec = min(3.0 if v < best_v + 0.005 else 1.5, remain)
        try:
            Xq, vq = _lp_polish(Xp, seconds=lp_sec)
        except Exception:
            Xq, vq = Xp, v
        if vq < best_v:
            best_v, best_x = float(vq), Xq.copy()
        elif v < best_v:
            best_v, best_x = float(v), Xp.copy()
    # final squeeze on incumbent best with remaining time
    if time.time() < T_END - 2.0:
        try:
            sec = max(0.6, min(4.0, T_END - 1.0 - time.time()))
            best_x, bv = _lp_polish(best_x, seconds=sec)
            best_v = min(best_v, _max_cos(best_x))
        except Exception:
            pass
    best_x = _norm(best_x)
    return best_x
