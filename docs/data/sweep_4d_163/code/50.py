import time
import numpy as np
from scipy.optimize import minimize, linprog
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
def _lp_polish(x, seconds, r0=2e-3, cap=2000):
    global _IU
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    if _IU is None or _IU[0].shape[0] != n * (n - 1) // 2:
        _IU = np.triu_indices(n, 1)
    iu = _IU
    best = _max_cos(x)
    radius = r0
    start = time.time()
    cd = np.arange(d)
    while time.time() - start < seconds and radius > 1e-14:
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
            if cnt2 > cap + 500:
                ids = np.where(sel)[0]
                keep = ids[::2][:cap]
                ns = np.zeros_like(sel)
                ns[keep] = True
                sel = ns
        I = iu[0][sel]; J = iu[1][sel]
        c = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + cd, J[:, None] * d + cd,
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
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
            radius = min(radius * 1.6, 0.04)
        else:
            radius *= 0.5
    return x

def _expand_2plane(Y, p, m, k):
    # Y (q,4), p (4,) -> X (1+q*m,4)
    q = Y.shape[0]
    th = 2.0 * np.pi / m
    ang1 = th * np.arange(m)
    ang2 = th * float(k) * np.arange(m)
    c1 = np.cos(ang1); s1 = np.sin(ang1)
    c2 = np.cos(ang2); s2 = np.sin(ang2)
    Y1 = Y[:, 0:1]; Y2 = Y[:, 1:2]; Y3 = Y[:, 2:3]; Y4 = Y[:, 3:4]
    # broadcast (q,m)
    Xo0 = Y1 * c1[None, :] - Y2 * s1[None, :]
    Xo1 = Y1 * s1[None, :] + Y2 * c1[None, :]
    Xo2 = Y3 * c2[None, :] - Y4 * s2[None, :]
    Xo3 = Y3 * s2[None, :] + Y4 * c2[None, :]
    Xo = np.stack([Xo0, Xo1, Xo2, Xo3], axis=2).reshape(q * m, 4)
    # norms preserved (Y unit)
    X = np.concatenate([p[None, :], Xo], axis=0)
    return X

def _sym_obj_grad(z, q, m, k, T):
    # z: (1+q)*4 raw
    Z = z.reshape(1 + q, 4)
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    N = Z / norms
    p = N[0]
    Y = N[1:]
    X = _expand_2plane(Y, p, m, k)
    n = X.shape[0]
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m0 = float(G.max())
    # upper triangle sum
    iu = np.triu_indices(n, 1)
    gvec = G[iu]
    Evec = np.exp((gvec - m0) / T)
    Zsum = float(Evec.sum())
    S = m0 + T * np.log(Zsum + 1e-300)
    # W symmetric
    W = np.zeros((n, n))
    W[iu] = Evec / Zsum
    W = W + W.T
    gradX = W @ X
    # pole grad
    grad_p = gradX[0]
    grad_orb = gradX[1:].reshape(q, m, 4)
    th = 2.0 * np.pi / m
    ang1 = th * np.arange(m)
    ang2 = th * float(k) * np.arange(m)
    c1 = np.cos(ang1); s1 = np.sin(ang1)
    c2 = np.cos(ang2); s2 = np.sin(ang2)
    # R^T g sum over t
    g0 = grad_orb[:, :, 0]; g1 = grad_orb[:, :, 1]
    g2 = grad_orb[:, :, 2]; g3 = grad_orb[:, :, 3]
    gy0 = (g0 * c1[None, :] + g1 * s1[None, :]).sum(axis=1)
    gy1 = (-g0 * s1[None, :] + g1 * c1[None, :]).sum(axis=1)
    gy2 = (g2 * c2[None, :] + g3 * s2[None, :]).sum(axis=1)
    gy3 = (-g2 * s2[None, :] + g3 * c2[None, :]).sum(axis=1)
    gradY = np.stack([gy0, gy1, gy2, gy3], axis=1)
    gradN = np.concatenate([grad_p[None, :], gradY], axis=0)
    # chain through normalization: (I-yy^T)/norm * grad
    dot = (gradN * N).sum(axis=1, keepdims=True)
    gradZ = (gradN - dot * N) / norms
    gflat = gradZ.ravel()
    return S, gflat

def _run_sym(Y0, p0, m, k, Ts, maxiter):
    q = Y0.shape[0]
    N0 = np.concatenate([p0[None, :], Y0], axis=0)
    z = N0.ravel().copy()
    for T in Ts:
        def fun(zz):
            s, g = _sym_obj_grad(zz, q, m, k, T)
            return s, g
        try:
            res = minimize(fun, z, method='L-BFGS-B', jac=True,
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-7})
            z = res.x
        except Exception:
            break
    S, _ = _sym_obj_grad(z, q, m, k, Ts[-1])
    Z = z.reshape(1 + q, 4)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    p = Z[0] / np.linalg.norm(Z[0])
    Y = Z[1:] / np.linalg.norm(Z[1:], axis=1, keepdims=True)
    X = _expand_2plane(Y, p, m, k)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    v = _max_cos(X)
    return X, Y, p, v

def solve():
    t0 = time.time()
    DEAD = 170.0
    rng = np.random.default_rng(7)
    # incumbent
    cands = []
    try:
        b = np.load("best.npy")
        cands.append(np.asarray(b, dtype=float))
    except Exception:
        pass
    try:
        r = np.load("record.npy")
        cands.append(np.asarray(r, dtype=float))
    except Exception:
        pass
    n = 163
    if len(cands) == 0:
        x0 = rng.normal(size=(n, 4))
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        cands.append(x0)
    vals0 = []
    for c in cands:
        c = np.asarray(c, dtype=float)
        if c.shape != (163, 4):
            continue
        c = c / np.linalg.norm(c, axis=1, keepdims=True)
        vals0.append(_max_cos(c))
    # keep valid
    vc = []
    for c in cands:
        c = np.asarray(c, dtype=float)
        if c.shape == (163, 4):
            c = c / np.linalg.norm(c, axis=1, keepdims=True)
            vc.append(c)
    cands = vc
    vals0 = [_max_cos(c) for c in cands]
    bi = int(np.argmin(vals0))
    x_best = cands[bi].copy()
    best = float(vals0[bi])

    sym_pool = []  # (val, X, m,k)
    # main decomposition 1+6*27
    twists = [1, 2, 4, 5, 8, 11]
    Ts = [0.02, 0.008, 0.0028]
    # deterministic starts
    srng = np.random.default_rng(123456)
    inits = []
    for k in twists:
        for s in range(3):
            Y0 = srng.normal(size=(6, 4)); Y0 /= np.linalg.norm(Y0, axis=1, keepdims=True)
            p0 = srng.normal(size=(4,)); p0 /= np.linalg.norm(p0)
            inits.append((k, Y0, p0))
    # shuffle deterministic to spread twists
    order = [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 4, 10, 16, 5, 11, 17]
    order = [o for o in order if o < len(inits)]
    for oi in order:
        if time.time() - t0 > 72.0:
            break
        k, Y0, p0 = inits[oi]
        # quick degeneracy guard: ensure r12 not tiny
        Xsym, Y, p, v = _run_sym(Y0, p0, 27, k, Ts, maxiter=110)
        # check orbit degeneracy: min pairwise within same orbit
        # if degenerate (duplicate points) max_cos ~1 -> discard
        if v > 0.99:
            # try re-init with forced balanced radii
            continue
        sym_pool.append((v, Xsym.copy(), 27, k))
    # nearby decompositions if time permits: 1+3*54 and 1+2*81
    if time.time() - t0 < 80.0:
        for (mm, qq, kk) in [(54, 3, 1), (81, 2, 1)]:
            if time.time() - t0 > 88.0:
                break
            Y0 = srng.normal(size=(qq, 4)); Y0 /= np.linalg.norm(Y0, axis=1, keepdims=True)
            p0 = srng.normal(size=(4,)); p0 /= np.linalg.norm(p0)
            try:
                Xsym, Y, p, v = _run_sym(Y0, p0, mm, kk, Ts, maxiter=110)
            except Exception:
                continue
            if v < 0.99:
                sym_pool.append((v, Xsym.copy(), mm, kk))
    # sort symmetric by value
    sym_pool.sort(key=lambda t: t[0])
    # keep top 3
    top_sym = sym_pool[:3] if len(sym_pool) > 0 else []
    # break symmetry: perturb + full smooth
    brng = np.random.default_rng(999)
    broken = []
    sigmas = [0.004, 0.012, 0.028]
    for idx, (v, Xs, mm, kk) in enumerate(top_sym):
        for sg in sigmas:
            if time.time() - t0 > 118.0:
                break
            Xp = Xs + brng.normal(scale=sg, size=Xs.shape)
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
            # short smooth descent
            Xp = _smooth_full(Xp, 170, 0.014, 0.0014, 0.05, 0.010)
            vv = _max_cos(Xp)
            broken.append((vv, Xp.copy()))
            if vv < best:
                x_best = Xp.copy(); best = vv
    # also include incumbent in broken comparison: polish incumbent lightly vs broken
    # choose overall best among broken + incumbent for final polish
    if len(broken) > 0:
        broken.sort(key=lambda t: t[0])
        # best broken
        if broken[0][0] < best:
            x_best = broken[0][1].copy(); best = broken[0][0]
    # if symmetric/broken failed to beat, still try shaking incumbent a bit (different basin attempt already done via sym)
    # final stage: short smooth + LP polish on top 2 candidates (best overall + best symmetric-broken)
    final_cands = [(best, x_best.copy())]
    if len(broken) > 1:
        # add second best distinct
        final_cands.append((broken[1][0], broken[1][1].copy()))
    # micro smooth on final
    for i in range(len(final_cands)):
        if time.time() - t0 > 128.0:
            break
        v, Xc = final_cands[i]
        y = Xc + brng.normal(scale=0.003, size=Xc.shape)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        y = _smooth_full(y, 100, 0.006, 0.0012, 0.025, 0.007)
        vy = _max_cos(y)
        if vy < v:
            final_cands[i] = (vy, y)
    # pick best
    final_cands.sort(key=lambda t: t[0])
    x_best = final_cands[0][1].copy(); best = float(final_cands[0][0])
    # LP polish with remaining time
    remain = DEAD - (time.time() - t0)
    lp_time = float(min(38.0, max(6.0, remain - 3.0)))
    if lp_time > 2.5 and DEAD - (time.time() - t0) > 5:
        x_best = _lp_polish(x_best, lp_time, r0=2e-3)
        best = _max_cos(x_best)
        # extra micro shake + repolish if time
        if DEAD - (time.time() - t0) > 12:
            for _ in range(2):
                if DEAD - (time.time() - t0) < 9:
                    break
                y = x_best + brng.normal(scale=0.0022, size=x_best.shape)
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                y = _smooth_full(y, 90, 0.005, 0.0012, 0.02, 0.006)
                vy = _max_cos(y)
                if vy < best:
                    x_best, best = y, vy
            r2 = float(min(9.0, DEAD - (time.time() - t0) - 1.0))
            if r2 > 2:
                x_best = _lp_polish(x_best, r2, r0=6e-4)
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
