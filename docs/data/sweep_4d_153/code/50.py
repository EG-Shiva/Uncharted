import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    X = np.asarray(X, dtype=float)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n == 0] = 1.0
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _lp_polish(X, seconds, t_end=1e20):
    X = _norm(np.asarray(X, dtype=float))
    n, d = X.shape
    try:
        best = _maxcos(X)
    except Exception:
        return X
    bestX = X.copy()
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while True:
        if time.time() - start > seconds:
            break
        if time.time() > t_end:
            break
        if radius < 1e-14:
            break
        try:
            G = X @ X.T
            c = G[iu0]
            m = float(np.max(c))
            thr = m - max(20.0 * radius, 1e-9)
            sel = c > thr
            if np.count_nonzero(sel) == 0:
                break
            I = iu0[0][sel].astype(int)
            J = iu0[1][sel].astype(int)
            p = int(I.shape[0])
            # cap constraints for speed
            if p > 3000:
                idx = np.argsort(-c[sel])[:3000]
                I = I[idx]; J = J[idx]; p = 3000
            nv = n * d + 1
            cols = np.concatenate([(I[:, None] * d + np.arange(d)),
                                   (J[:, None] * d + np.arange(d)),
                                   np.full((p, 1), n * d)], axis=1).ravel()
            rows = np.repeat(np.arange(p), 2 * d + 1)
            vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            b_ub = -1.0 * (X[I] * X[J]).sum(axis=1)
            res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if getattr(res, "status", 1) != 0:
            radius *= 0.5
            continue
        try:
            y = X + np.asarray(res.x[:-1], dtype=float).reshape(n, d)
            y = _norm(y)
            v = _maxcos(y)
        except Exception:
            radius *= 0.5
            continue
        if v < best:
            best = v
            bestX = y.copy()
            X = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return _norm(bestX)

def _expand_reps(Y, m, a):
    # Y: (R,4) normalized reps -> X: (R*m,4)
    R = Y.shape[0]
    A = Y[:, 0] + 1j * Y[:, 1]
    B = Y[:, 2] + 1j * Y[:, 3]
    k = np.arange(m)
    W1 = np.exp(2j * np.pi * k / m)
    W2 = np.exp(2j * np.pi * a * k / m)
    Ae = A[:, None] * W1[None, :]
    Be = B[:, None] * W2[None, :]
    Xe = np.empty((R, m, 4), dtype=float)
    Xe[:, :, 0] = Ae.real; Xe[:, :, 1] = Ae.imag
    Xe[:, :, 2] = Be.real; Xe[:, :, 3] = Be.imag
    X = Xe.reshape(R * m, 4)
    return _norm(X)

def _sym_adam(m, R, a, rng, iters, lr=0.025, k0=10.0, k1=90.0):
    # optimize R reps generating R*m points; returns expanded best
    reps = rng.standard_normal((R, 4))
    reps = _norm(reps)
    # Adam state
    mt = np.zeros_like(reps)
    vt = np.zeros_like(reps)
    b1, b2, eps = 0.9, 0.999, 1e-8
    karr = np.arange(m)
    W1 = np.exp(2j * np.pi * karr / m)
    W2 = np.exp(2j * np.pi * a * karr / m)
    cW1 = np.conj(W1); cW2 = np.conj(W2)
    N = R * m
    best_val = 1e9
    best_X = None
    for t in range(iters):
        frac = t / max(1, iters - 1)
        kk = k0 * (k1 / k0) ** frac
        try:
            rn = np.linalg.norm(reps, axis=1, keepdims=True)
            rn[rn == 0] = 1.0
            Y = reps / rn
            A = Y[:, 0] + 1j * Y[:, 1]
            B = Y[:, 2] + 1j * Y[:, 3]
            Ae = A[:, None] * W1[None, :]
            Be = B[:, None] * W2[None, :]
            X = np.empty((R, m, 4), dtype=float)
            X[:, :, 0] = Ae.real; X[:, :, 1] = Ae.imag
            X[:, :, 2] = Be.real; X[:, :, 3] = Be.imag
            Xf = X.reshape(N, 4)
            # renorm (should already be unit)
            Xf = _norm(Xf)
            G = Xf @ Xf.T
            np.fill_diagonal(G, -2.0)
            gm = float(np.max(G))
            # track best by true max
            if gm < best_val:
                best_val = gm
                best_X = Xf.copy()
            # softmax weights
            E = np.exp(kk * (G - gm))
            np.fill_diagonal(E, 0.0)
            S = float(np.sum(E)) + 1e-30
            gradX = (E @ Xf) / S  # descent direction factor omitted
            # tangential projection
            dot = np.sum(gradX * Xf, axis=1, keepdims=True)
            gradX = gradX - dot * Xf
            g3 = gradX.reshape(R, m, 4)
            gA = g3[:, :, 0] + 1j * g3[:, :, 1]
            gB = g3[:, :, 2] + 1j * g3[:, :, 3]
            gAr = np.sum(gA * cW1[None, :], axis=1)
            gBr = np.sum(gB * cW2[None, :], axis=1)
            gY = np.empty_like(Y)
            gY[:, 0] = gAr.real; gY[:, 1] = gAr.imag
            gY[:, 2] = gBr.real; gY[:, 3] = gBr.imag
            # chain through Y=reps/|reps|
            d = np.sum(gY * Y, axis=1, keepdims=True)
            g = (gY - d * Y) / rn
            # Adam update (minimize)
            mt = b1 * mt + (1 - b1) * g
            vt = b2 * vt + (1 - b2) * (g * g)
            mhat = mt / (1 - b1 ** (t + 1))
            vhat = vt / (1 - b2 ** (t + 1))
            reps = reps - lr * mhat / (np.sqrt(vhat) + eps)
            # keep finite
            if not np.all(np.isfinite(reps)):
                reps = _norm(rng.standard_normal((R, 4)))
                mt[:] = 0; vt[:] = 0
        except Exception:
            continue
    if best_X is None:
        best_X = _expand_reps(_norm(rng.standard_normal((R, 4))), m, a)
    return _norm(best_X), float(best_val)

def solve():
    t0 = time.time()
    T_LIM = 170.0
    n, d = 153, 4
    # load fallback record
    fallback = None
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                fallback = _norm(a)
                break
        except Exception:
            continue
    if fallback is None:
        try:
            rng0 = np.random.default_rng(0)
            fallback = _norm(rng0.standard_normal((n, d)))
        except Exception:
            fallback = np.ones((n, d)) / 2.0
    try:
        fb_val = _maxcos(fallback)
    except Exception:
        fb_val = 1.0
    incumbent = fallback.copy()
    inc_val = fb_val
    try:
        # quick LP touch on fallback (cheap, keeps valid return)
        if time.time() - t0 < 150:
            try:
                cand0 = _lp_polish(incumbent, 2.0, t0 + T_LIM - 2)
                v0 = _maxcos(cand0)
                if v0 < inc_val and np.all(np.isfinite(cand0)):
                    incumbent = cand0; inc_val = v0
            except Exception:
                pass
        rng = np.random.default_rng(20261031)
        configs = []
        for a in range(1, 9):
            configs.append((17, 9, a))
        for a in range(1, 5):
            configs.append((9, 17, a))
        cands = []  # (val, X)
        # brief try each config
        for (m, R, a) in configs:
            if time.time() - t0 > 110:
                break
            try:
                # iters scaled to stay fast
                it = 320 if m == 17 else 260
                Xsym, v = _sym_adam(m, R, a, rng, it)
                if Xsym.shape != (153, 4) or not np.all(np.isfinite(Xsym)):
                    continue
                # validate no exact duplicates causing 1.0
                vv = _maxcos(Xsym)
                cands.append((vv, Xsym.copy()))
            except Exception:
                continue
        # rank and break symmetry + LP polish best few
        try:
            cands.sort(key=lambda t: t[0])
        except Exception:
            pass
        K = min(3, len(cands))
        for i in range(K):
            if time.time() - t0 > 150:
                break
            try:
                v, Xs = cands[i]
                # break symmetry with 1e-3 tangential noise
                pert = rng.standard_normal(Xs.shape)
                dpr = np.sum(pert * Xs, axis=1, keepdims=True)
                pert = pert - dpr * Xs
                Xn = _norm(Xs + 1e-3 * pert)
                # LP polish
                Xp = _lp_polish(Xn, 6.0, t0 + T_LIM - 2)
                vp = _maxcos(Xp)
                if vp < inc_val and np.all(np.isfinite(Xp)) and Xp.shape == (n, d):
                    incumbent = Xp.copy(); inc_val = vp
                # also try without noise (pure symmetric polish)
                if time.time() - t0 < 150:
                    try:
                        Xp2 = _lp_polish(Xs, 3.0, t0 + T_LIM - 2)
                        vp2 = _maxcos(Xp2)
                        if vp2 < inc_val and np.all(np.isfinite(Xp2)):
                            incumbent = Xp2.copy(); inc_val = vp2
                    except Exception:
                        pass
            except Exception:
                continue
        # final safety: if incumbent worse than fallback keep fallback polished
        try:
            if _maxcos(incumbent) > fb_val + 1e-12:
                incumbent = fallback.copy()
        except Exception:
            incumbent = fallback.copy()
    except Exception:
        try:
            incumbent = _norm(fallback)
        except Exception:
            pass
    # validate output contract
    try:
        incumbent = np.asarray(incumbent, dtype=float)
        if incumbent.shape != (n, d) or not np.all(np.isfinite(incumbent)):
            incumbent = _norm(fallback)
        else:
            incumbent = _norm(incumbent)
            # no zero rows (norm guards), no NaN
            bad = ~np.all(np.isfinite(incumbent), axis=1)
            if np.any(bad):
                incumbent[bad] = fallback[bad]
                incumbent = _norm(incumbent)
    except Exception:
        incumbent = _norm(fallback)
    return incumbent
