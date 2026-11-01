import time
import math
import numpy as np

def _norm_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _build_sym(R, C, S, m, norb):
    # R: (norb+1,12), C,S: (m,6) -> X (norb*m+1,12)
    Rx = R[:norb, 0::2]  # (norb,6)
    Ry = R[:norb, 1::2]
    # (norb,m,6)
    Xx = Rx[:, None, :] * C[None, :, :] - Ry[:, None, :] * S[None, :, :]
    Xy = Rx[:, None, :] * S[None, :, :] + Ry[:, None, :] * C[None, :, :]
    Xorb = np.empty((norb, m, 12), dtype=np.float64)
    Xorb[:, :, 0::2] = Xx
    Xorb[:, :, 1::2] = Xy
    X = np.empty((norb * m + 1, 12), dtype=np.float64)
    X[:norb * m] = Xorb.reshape(norb * m, 12)
    X[norb * m] = R[norb]
    return X

def _sym_quench(Y0, freqs, m, norb, steps=220, beta0=40.0, beta1=260.0, lr=0.03, rng=None):
    # precompute rotations
    j = np.arange(m)[:, None]  # (m,1)
    f = np.asarray(freqs, dtype=float)[None, :]  # (1,6)
    ang = 2.0 * np.pi * j * f / float(m)  # (m,6)
    C = np.cos(ang)
    S = np.sin(ang)
    Y = Y0.copy()
    b1, b2, eps = 0.9, 0.999, 1e-8
    mt = np.zeros_like(Y)
    vt = np.zeros_like(Y)
    best_val = 1e100
    best_X = None
    best_R = None
    nrep = norb + 1
    for t in range(1, steps + 1):
        beta = beta0 + (beta1 - beta0) * (t / steps)
        T = 1.0 / beta
        rn = np.linalg.norm(Y, axis=1, keepdims=True)
        rn = np.maximum(rn, 1e-12)
        R = Y / rn
        X = _build_sym(R, C, S, m, norb)
        G = X @ X.T
        np.fill_diagonal(G, -1e100)
        m0 = float(G.max())
        Q = (G - m0) / T
        np.clip(Q, -50, 0, out=Q)
        E = np.exp(Q)
        np.fill_diagonal(E, 0.0)
        ssum = float(E.sum())
        if not np.isfinite(ssum) or ssum == 0:
            ssum = 1.0
        W = E / ssum
        GradX = W @ X  # (71,12)
        # pullback
        gradR = np.zeros_like(R)
        # orbit part
        Go = GradX[:norb * m].reshape(norb, m, 12)
        Gx = Go[:, :, 0::2]  # (norb,m,6)
        Gy = Go[:, :, 1::2]
        # C,S (m,6)
        gradRx = (C[None, :, :] * Gx + S[None, :, :] * Gy).sum(axis=1)  # (norb,6)
        gradRy = (-S[None, :, :] * Gx + C[None, :, :] * Gy).sum(axis=1)
        gradR[:norb, 0::2] = gradRx
        gradR[:norb, 1::2] = gradRy
        gradR[norb] = GradX[norb * m]
        # tangent project
        dot = (gradR * R).sum(axis=1, keepdims=True)
        Gt = gradR - dot * R
        gradY = Gt / rn
        # adam
        mt = b1 * mt + (1 - b1) * gradY
        vt = b2 * vt + (1 - b2) * (gradY * gradY)
        mh = mt / (1 - b1 ** t)
        vh = vt / (1 - b2 ** t)
        Y = Y - lr * mh / (np.sqrt(vh) + eps)
        # track by true max
        # evaluate every 20 steps and last
        if (t % 20 == 0) or (t == steps):
            Gc = X @ X.T
            np.fill_diagonal(Gc, -2.0)
            v = float(Gc.max())
            if v < best_val:
                best_val = v
                best_X = X.copy()
                best_R = R.copy()
    if best_X is None:
        R = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
        best_X = _build_sym(R, C, S, m, norb)
        Gc = best_X @ best_X.T
        np.fill_diagonal(Gc, -2.0)
        best_val = float(Gc.max())
    return best_X, best_val, freqs, m, norb

def _full_quench(X0, steps, T0, T1, lr, mu=0.9):
    X = _norm_rows(X0.copy())
    V = np.zeros_like(X)
    G0 = X @ X.T
    np.fill_diagonal(G0, -2.0)
    bv = float(G0.max())
    bX = X.copy()
    lt0, lt1 = np.log(T0), np.log(T1)
    n = X.shape[0]
    for s in range(steps):
        frac = s / max(1, steps - 1)
        T = float(np.exp(lt0 + (lt1 - lt0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -1e100)
        m0 = float(G.max())
        Q = (G - m0) / T
        np.clip(Q, -50, 0, out=Q)
        E = np.exp(Q)
        np.fill_diagonal(E, 0.0)
        rs = E.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        Grad = (E @ X) / rs
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        gn = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        bad = gn[:, 0] > 2.0
        if np.any(bad):
            Gt[bad] *= (2.0 / gn[bad])
        V = mu * V + Gt
        X = X - lr * V
        X = _norm_rows(X)
        if (s % 100 == 0) or (s == steps - 1):
            Gc = X @ X.T
            np.fill_diagonal(Gc, -2.0)
            vc = float(Gc.max())
            if vc < bv:
                bv = vc
                bX = X.copy()
    return bX, bv

def _lbfgs_smooth(X0, betas, iters):
    from scipy.optimize import minimize
    n, d = X0.shape
    iu = np.triu_indices(n, 1)
    Y = _norm_rows(X0.copy())
    for beta, it in zip(betas, iters):
        def fun(Yflat, beta=beta):
            Ym = Yflat.reshape(n, d)
            rn = np.linalg.norm(Ym, axis=1, keepdims=True)
            rn = np.maximum(rn, 1e-12)
            X = Ym / rn
            G = X @ X.T
            gv = G[iu]
            gm = gv.max()
            e = np.exp(beta * (gv - gm))
            S = e.sum()
            F = gm + np.log(S) / beta
            w = e / S
            W = np.zeros((n, n))
            W[iu] = w
            W = W + W.T
            gradX = W @ X
            dot = (gradX * X).sum(axis=1, keepdims=True)
            gradY = (gradX - dot * X) / rn
            return float(F), gradY.ravel()
        try:
            res = minimize(fun, Y.ravel(), method='L-BFGS-B', jac=True,
                           options={'maxiter': int(it), 'ftol': 1e-12, 'gtol': 1e-7, 'maxcor': 10})
            Y = res.x.reshape(n, d)
        except Exception:
            pass
        Y = _norm_rows(Y)
    return _norm_rows(Y)

def _lp_polish(x, seconds, deadline=None):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    x = _norm_rows(x.copy())
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = 1.2e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if deadline is not None and time.time() > deadline:
            break
        if radius < 1e-14:
            break
        G = x @ x.T
        cuv = G[iu]
        m = float(cuv.max())
        sel = cuv > m - max(20 * radius, 1e-9)
        if np.count_nonzero(sel) == 0:
            break
        I = iu[0][sel]; J = iu[1][sel]
        csel = cuv[sel]
        p = len(I)
        if p > 1500:
            order = np.argsort(-csel)[:1500]
            I, J = I[order], J[order]
            csel = csel[order]
            p = 1500
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm_rows(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _gram_sig(X, thr=1e-3):
    G = X @ X.T
    iu = np.triu_indices(X.shape[0], 1)
    gv = np.sort(G[iu])[::-1]
    return gv[:60]

def solve():
    import time as _time
    t0 = _time.time()
    HARD_END = t0 + 172.0
    try:
        Xbest = np.load("best.npy")
    except Exception:
        try:
            Xbest = np.load("record.npy")
        except Exception:
            r0 = np.random.default_rng(0)
            Xbest = r0.standard_normal(size=(71, 12))
    Xbest = _norm_rows(np.asarray(Xbest, dtype=float))
    if Xbest.shape != (71, 12):
        r0 = np.random.default_rng(1)
        Xbest = _norm_rows(r0.standard_normal(size=(71, 12)))
    best = _max_cos(Xbest)
    sig_best = _gram_sig(Xbest)
    rng = np.random.default_rng(987654321)
    combos = [(14, 5), (10, 7), (35, 2), (7, 10), (70, 1)]
    archive = []  # list of (score, X)
    SYM_END = t0 + 82.0
    trial = 0
    # scan several hundred group/frequency choices
    while _time.time() < SYM_END and _time.time() < HARD_END - 80 and trial < 400:
        m, norb = combos[trial % len(combos)]
        # sample frequency vector with gcd==1
        freqs = None
        for _ in range(12):
            f = rng.integers(0, m, size=6)
            if np.all(f == 0):
                continue
            g = int(m)
            for v in f:
                g = math.gcd(g, int(v))
            if g == 1:
                freqs = f
                break
        if freqs is None:
            trial += 1
            continue
        Y0 = rng.standard_normal(size=(norb + 1, 12))
        # vary hyperparams slightly
        b0 = 30.0 if (trial % 2 == 0) else 60.0
        b1 = 260.0 if (trial % 3 != 0) else 400.0
        steps = 200 if m <= 14 else 160
        lr = 0.03 if (trial % 2 == 0) else 0.02
        try:
            Xs, vs, _, _, _ = _sym_quench(Y0, freqs, m, norb, steps=steps, beta0=b0, beta1=b1, lr=lr)
        except Exception:
            trial += 1
            continue
        trial += 1
        # retain diverse: keep if reasonable; diversity via gram spectrum distance
        if vs < 0.45:
            # spectrum distance to best
            try:
                s = _gram_sig(Xs)
                dist = float(np.linalg.norm(s - sig_best))
            except Exception:
                dist = 1.0
            archive.append((vs, Xs, dist, tuple(map(int, freqs)), m))
            # keep sorted, prune to 12, prefer score but keep diverse (dist>2e-4) even if 1e-3 worse
            archive = sorted(archive, key=lambda z: z[0])[:14]
    # release phase: take top symmetric + incumbent
    # build elite list
    elites = []
    for vs, Xs, dist, fr, m in archive[:8]:
        elites.append(Xs)
    # add incumbent perturbed copies for safety
    elites.append(Xbest.copy())
    # also add a couple of random shake of best to have fallback basin
    for k in range(2):
        Xn = Xbest + 0.05 * rng.standard_normal(size=Xbest.shape)
        elites.append(_norm_rows(Xn))
    # full optimisation of each elite (released: all 71 free)
    REL_END = HARD_END - 30.0
    for eidx, Xe in enumerate(elites):
        if _time.time() > REL_END:
            break
        X0 = _norm_rows(Xe.copy())
        # skip hopeless far codes unless diverse (retain per instruction)
        v0 = _max_cos(X0)
        # release quench: two-stage full quench
        try:
            X1, v1 = _full_quench(X0, steps=350, T0=0.02, T1=0.002, lr=0.02, mu=0.9)
        except Exception:
            continue
        # if still far and not diverse, skip expensive steps
        diverse = False
        try:
            s1 = _gram_sig(X1)
            d1 = float(np.linalg.norm(s1 - sig_best))
            if d1 > 5e-4:
                diverse = True
        except Exception:
            pass
        if v1 > best + 0.004 and not diverse and v1 > best + 0.0015:
            # still try short LBFGS only for first few
            if eidx > 3:
                continue
        # L-BFGS smooth
        try:
            if v1 < best + 0.006 or diverse:
                X2 = _lbfgs_smooth(X1, betas=[80.0, 300.0], iters=[60, 60])
                v2 = _max_cos(X2)
            else:
                X2, v2 = X1, v1
        except Exception:
            X2, v2 = X1, v1
        # LP polish if promising or diverse
        if v2 < best + 0.002 or diverse:
            try:
                X3 = _lp_polish(X2, seconds=1.5, deadline=HARD_END - 20)
                v3 = _max_cos(X3)
            except Exception:
                X3, v3 = X2, v2
        else:
            X3, v3 = X2, v2
        if v3 < best - 1e-12:
            Xbest, best = X3, v3
            sig_best = _gram_sig(Xbest)
        elif diverse and v3 < best + 0.001:
            # keep diverse elite for further rounds: try extra tightening
            try:
                X4, v4 = _full_quench(X3, steps=250, T0=0.005, T1=0.0008, lr=0.012, mu=0.9)
                X4 = _lp_polish(X4, seconds=1.0, deadline=HARD_END - 20)
                v4 = _max_cos(X4)
                if v4 < best - 1e-12:
                    Xbest, best = X4, v4
                    sig_best = _gram_sig(Xbest)
            except Exception:
                pass
        if _time.time() > REL_END:
            break
    # final squeeze on best
    remain = HARD_END - _time.time() - 2.0
    if remain > 3:
        try:
            Xbest = _lp_polish(Xbest, seconds=min(remain, 20.0), deadline=HARD_END - 1.0)
        except Exception:
            pass
    Xbest = _norm_rows(Xbest)
    return Xbest
