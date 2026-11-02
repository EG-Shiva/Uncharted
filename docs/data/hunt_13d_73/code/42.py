import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lp_polish(X, seconds, t_end):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < t_end and radius > 1e-14:
        Cfull = X @ X.T
        c = Cfull[iu0, iu1]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        if np.count_nonzero(sel) > 2200:
            idx = np.where(sel)[0]
            order = np.argsort(-c[idx])[:2200]
            mask = np.zeros_like(sel)
            mask[idx[order]] = True
            sel = mask
        I, J = iu0[sel], iu1[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        csel = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cI = I[:, None] * d + np.arange(d)
        cJ = J[:, None] * d + np.arange(d)
        ct = np.full((p, 1), n * d)
        cols = np.concatenate([cI, cJ, ct], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        nrm = np.linalg.norm(y, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        y /= nrm
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X, best

def solve():
    t0 = time.time()
    T_END = t0 + 165.0
    SEARCH_END = t0 + 148.0
    n, d = 73, 13
    rng = np.random.default_rng(0)
    # load incumbent
    Xrec = None
    best_file = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            v = np.load(fn)
            v = np.asarray(v, dtype=np.float64)
            if v.shape == (n, d):
                v = v / np.linalg.norm(v, axis=1, keepdims=True)
                if np.all(np.isfinite(v)):
                    c = _max_cos(v)
                    if c < best_file:
                        best_file = c
                        Xrec = v.copy()
        except Exception:
            pass
    if Xrec is None:
        Xrec = rng.standard_normal(size=(n, d))
        Xrec /= np.linalg.norm(Xrec, axis=1, keepdims=True)
        best_file = _max_cos(Xrec)
    Xbest = Xrec.copy()
    best = float(best_file)

    iu0, iu1 = np.triu_indices(n, 1)
    ar12 = np.arange(12)
    exp_perm12 = [((ar12 - k) % 12) for k in range(12)]
    acc_perm12 = [((ar12 + k) % 12) for k in range(12)]
    ar13 = np.arange(13)
    exp_perm13 = [((ar13 - k) % 13) for k in range(13)]
    acc_perm13 = [((ar13 + k) % 13) for k in range(13)]
    e13 = np.zeros(d); e13[-1] = 1.0

    def expand_c12(Tn):
        # Tn (6,13) normalized
        X = np.empty((73, 13))
        for j in range(6):
            t12 = Tn[j, :12]
            b = Tn[j, 12]
            base = j * 12
            for k in range(12):
                X[base + k, :12] = t12[exp_perm12[k]]
                X[base + k, 12] = b
        X[72] = e13
        return X

    def expand_c13(Tn, F):
        # Tn (5,13), F (8,13)
        X = np.empty((73, 13))
        for j in range(5):
            tj = Tn[j]
            base = j * 13
            for k in range(13):
                X[base + k] = tj[exp_perm13[k]]
        X[65:73] = F
        return X

    # ---- annealed Adam on C12 templates ----
    def adam_c12(Y0, steps, s0, s1, lr0, lr1, t_lim):
        Y = Y0.copy()
        m = np.zeros_like(Y); vv = np.zeros_like(Y)
        b1, b2, eps = 0.9, 0.999, 1e-8
        b1p, b2p = b1, b2
        l0, l1 = np.log(lr0), np.log(lr1)
        bestL_T = None; bestL_v = 1e9
        for k in range(steps):
            if time.time() > t_lim:
                break
            frac = k / max(steps - 1, 1)
            s = s0 + (s1 - s0) * frac
            lr = np.exp(l0 + (l1 - l0) * frac)
            nrY = np.sqrt(np.sum(Y * Y, axis=1, keepdims=True))
            nrY[nrY < 1e-12] = 1e-12
            Tn = Y / nrY
            X = expand_c12(Tn)
            G = X @ X.T
            c = G[iu0, iu1]
            mmax = float(c.max())
            if mmax < bestL_v:
                bestL_v = mmax
                bestL_T = Tn.copy()
            e = np.exp(s * (c - mmax))
            se = float(e.sum())
            if se == 0 or not np.isfinite(se):
                # uniform subgradient fallback
                mk = c >= mmax - 1e-12
                w = mk.astype(float) / max(mk.sum(), 1)
            else:
                w = e / se
            W = np.zeros((n, n))
            W[iu0, iu1] = w
            W[iu1, iu0] = w
            GradX = W @ X
            # tangent for X not strictly needed before pullback, pullback then project
            gT = np.zeros((6, 13))
            for j in range(6):
                base = j * 12
                Gj = GradX[base:base + 12]  # 12x13
                acc = np.zeros(12)
                for kk in range(12):
                    acc += Gj[kk, :12][acc_perm12[kk]]
                gT[j, :12] = acc
                gT[j, 12] = float(Gj[:, 12].sum())
            # to Y grad
            dot = np.sum(gT * Tn, axis=1, keepdims=True)
            gY = (gT - dot * Tn) / nrY
            if not np.all(np.isfinite(gY)):
                b1p *= b1; b2p *= b2
                continue
            m = b1 * m + (1 - b1) * gY
            vv = b2 * vv + (1 - b2) * (gY * gY)
            mhat = m / (1 - b1p); vhat = vv / (1 - b2p)
            b1p *= b1; b2p *= b2
            Y -= lr * mhat / (np.sqrt(vhat) + eps)
        if bestL_T is None:
            nrY = np.linalg.norm(Y, axis=1, keepdims=True); nrY[nrY == 0] = 1.0
            bestL_T = Y / nrY
            X = expand_c12(bestL_T); bestL_v = _max_cos(X)
        return bestL_T, bestL_v

    def adam_c13(Y0, F0, steps, s0, s1, lr0, lr1, t_lim):
        Y = Y0.copy(); F_ = F0.copy()
        mY = np.zeros_like(Y); vY = np.zeros_like(Y)
        mF = np.zeros_like(F_); vF = np.zeros_like(F_)
        b1, b2, eps = 0.9, 0.999, 1e-8
        b1p, b2p = b1, b2
        l0, l1 = np.log(lr0), np.log(lr1)
        bestT = None; bestF = None; bestv = 1e9
        for k in range(steps):
            if time.time() > t_lim:
                break
            frac = k / max(steps - 1, 1)
            s = s0 + (s1 - s0) * frac
            lr = np.exp(l0 + (l1 - l0) * frac)
            nrY = np.sqrt(np.sum(Y * Y, axis=1, keepdims=True)); nrY[nrY < 1e-12] = 1e-12
            nrF = np.sqrt(np.sum(F_ * F_, axis=1, keepdims=True)); nrF[nrF < 1e-12] = 1e-12
            Tn = Y / nrY; Fn = F_ / nrF
            X = expand_c13(Tn, Fn)
            G = X @ X.T
            c = G[iu0, iu1]
            mmax = float(c.max())
            if mmax < bestv:
                bestv = mmax; bestT = Tn.copy(); bestF = Fn.copy()
            e = np.exp(s * (c - mmax))
            se = float(e.sum())
            w = e / se if se > 0 and np.isfinite(se) else np.ones_like(c) / len(c)
            W = np.zeros((n, n)); W[iu0, iu1] = w; W[iu1, iu0] = w
            GradX = W @ X
            gT = np.zeros_like(Y)
            for j in range(5):
                base = j * 13
                Gj = GradX[base:base + 13]
                acc = np.zeros(13)
                for kk in range(13):
                    acc += Gj[kk][acc_perm13[kk]]
                gT[j] = acc
            gF = GradX[65:73]
            dotY = np.sum(gT * Tn, axis=1, keepdims=True)
            dotF = np.sum(gF * Fn, axis=1, keepdims=True)
            gYY = (gT - dotY * Tn) / nrY
            gFF = (gF - dotF * Fn) / nrF
            mY = b1 * mY + (1 - b1) * gYY; vY = b2 * vY + (1 - b2) * (gYY * gYY)
            mF = b1 * mF + (1 - b1) * gFF; vF = b2 * vF + (1 - b2) * (gFF * gFF)
            mhatY = mY / (1 - b1p); vhatY = vY / (1 - b2p)
            mhatF = mF / (1 - b1p); vhatF = vF / (1 - b2p)
            b1p *= b1; b2p *= b2
            Y -= lr * mhatY / (np.sqrt(vhatY) + eps)
            F_ -= lr * mhatF / (np.sqrt(vhatF) + eps)
        return bestT, bestF, bestv

    def adam_free(X0, steps, s0, s1, lr0, lr1, t_lim):
        Y = (X0 / np.linalg.norm(X0, axis=1, keepdims=True)).copy()
        # use normalized X with momentum on sphere
        M = np.zeros_like(Y); V = np.zeros_like(Y)
        b1, b2, eps = 0.9, 0.999, 1e-8
        b1p, b2p = b1, b2
        l0, l1 = np.log(lr0), np.log(lr1)
        Xn = Y.copy()
        bestX = Xn.copy(); bestv = _max_cos(Xn)
        for k in range(steps):
            if time.time() > t_lim:
                break
            frac = k / max(steps - 1, 1)
            s = s0 + (s1 - s0) * frac
            lr = np.exp(l0 + (l1 - l0) * frac)
            G = Xn @ Xn.T
            c = G[iu0, iu1]
            mmax = float(c.max())
            if mmax < bestv:
                bestv = mmax; bestX = Xn.copy()
            e = np.exp(s * (c - mmax))
            se = float(e.sum())
            w = e / se if se > 0 and np.isfinite(se) else np.ones_like(c) / len(c)
            W = np.zeros((n, n)); W[iu0, iu1] = w; W[iu1, iu0] = w
            Grad = W @ Xn
            Gt = Grad - np.sum(Grad * Xn, axis=1, keepdims=True) * Xn
            M = b1 * M + (1 - b1) * Gt
            V = b2 * V + (1 - b2) * (Gt * Gt)
            mhat = M / (1 - b1p); vhat = V / (1 - b2p)
            b1p *= b1; b2p *= b2
            Xn = Xn - lr * mhat / (np.sqrt(vhat) + eps)
            nr = np.linalg.norm(Xn, axis=1, keepdims=True); nr[nr == 0] = 1.0
            Xn /= nr
        return bestX, bestv

    # Phase 1: C12 symmetry, several inits
    best_sym_X = None; best_sym = 1e9
    n_init_c12 = 5
    for ii in range(n_init_c12):
        if time.time() > SEARCH_END - 60:
            break
        rs = np.random.default_rng(100 + ii)
        Y0 = rs.standard_normal(size=(6, 13))
        Y0 /= np.linalg.norm(Y0, axis=1, keepdims=True)
        Tn, v = adam_c12(Y0, steps=650, s0=40.0, s1=380.0, lr0=0.04, lr1=0.004, t_lim=SEARCH_END - 55)
        Xc = expand_c12(Tn / np.linalg.norm(Tn, axis=1, keepdims=True))
        vv = _max_cos(Xc)
        if vv < best_sym:
            best_sym = vv; best_sym_X = Xc.copy()
        if vv < best:
            best = vv; Xbest = Xc.copy()

    # Phase 2: C13 5-orbit + 8 free, if time
    if time.time() < SEARCH_END - 55:
        for ii in range(3):
            if time.time() > SEARCH_END - 50:
                break
            rs = np.random.default_rng(1000 + ii)
            Y0 = rs.standard_normal(size=(5, 13)); Y0 /= np.linalg.norm(Y0, axis=1, keepdims=True)
            F0 = rs.standard_normal(size=(8, 13)); F0 /= np.linalg.norm(F0, axis=1, keepdims=True)
            Tn, Fn, v = adam_c13(Y0, F0, steps=550, s0=40.0, s1=350.0, lr0=0.04, lr1=0.004, t_lim=SEARCH_END - 48)
            Xc = expand_c13(Tn / np.linalg.norm(Tn, axis=1, keepdims=True), Fn / np.linalg.norm(Fn, axis=1, keepdims=True))
            vv = _max_cos(Xc)
            if vv < best_sym:
                best_sym = vv; best_sym_X = Xc.copy()
            if vv < best:
                best = vv; Xbest = Xc.copy()

    # Phase 3: break symmetry, unconstrained polish
    seeds = []
    if best_sym_X is not None:
        seeds.append(best_sym_X.copy())
    seeds.append(Xrec.copy())
    sigmas = [1e-4, 1e-3, 4e-3, 1e-2]
    # direct polish of symmetric without noise first
    if best_sym_X is not None and time.time() < SEARCH_END - 20:
        Xp, vp = adam_free(best_sym_X, steps=500, s0=250.0, s1=450.0, lr0=0.02, lr1=0.003, t_lim=SEARCH_END - 15)
        if vp < best:
            best = vp; Xbest = Xp.copy()
    for s_idx, sig in enumerate(sigmas):
        if time.time() > SEARCH_END - 15:
            break
        for b_idx, sd in enumerate(seeds):
            if time.time() > SEARCH_END - 15:
                break
            # limit number of combos to fit time
            if s_idx * len(seeds) + b_idx > 7:
                continue
            rs = np.random.default_rng(5000 + s_idx * 10 + b_idx)
            Xn0 = sd + rs.standard_normal(size=(n, d)) * sig
            Xn0 /= np.linalg.norm(Xn0, axis=1, keepdims=True)
            Xp, vp = adam_free(Xn0, steps=450, s0=180.0, s1=420.0, lr0=0.03, lr1=0.003, t_lim=SEARCH_END - 12)
            if vp < best:
                best = vp; Xbest = Xp.copy()

    # Phase 4: final LP polish on incumbent
    if time.time() < T_END - 4:
        try:
            remain = T_END - time.time() - 1.0
            if remain > 2.0:
                Xp, vp = _lp_polish(Xbest, seconds=min(remain, 12.0), t_end=T_END - 0.5)
                if vp < best and np.all(np.isfinite(Xp)):
                    best = vp; Xbest = Xp.copy()
        except Exception:
            pass

    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    if not np.all(np.isfinite(Xbest)):
        Xbest = Xrec / np.linalg.norm(Xrec, axis=1, keepdims=True)
    return Xbest
