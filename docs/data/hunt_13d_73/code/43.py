import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lp_polish(X, seconds):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        Cfull = X @ X.T
        c = Cfull[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            X, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def _gen_C12(T):
    # T: (6,13) unit rows -> X (73,13)
    X = np.empty((73, 13))
    for k in range(6):
        a = T[k, :12]
        b = T[k, 12]
        for s in range(12):
            X[k*12+s, :12] = np.roll(a, s)
            X[k*12+s, 12] = b
    X[72, :] = 0.0
    X[72, 12] = 1.0
    return X

def _gen_C13(U, F):
    # U: (5,13) rows, F: (8,13) rows -> X (73,13)
    X = np.empty((73, 13))
    for k in range(5):
        u = U[k]
        for s in range(13):
            X[k*13+s, :] = np.roll(u, s)
    X[65:, :] = F
    return X

def solve():
    t0 = time.time()
    HARD = 168.0
    SEARCH_END = 148.0
    rng = np.random.RandomState(12345)
    n, d = 73, 13
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    for z in (xb, xr):
        if z is not None and np.shape(z) == (n, d):
            zn = z / np.linalg.norm(z, axis=1, keepdims=True)
            if np.all(np.isfinite(zn)):
                cands.append((_max_cos(zn), zn))
    if len(cands) == 0:
        zn = rng.randn(n, d); zn /= np.linalg.norm(zn, axis=1, keepdims=True)
        cands.append((_max_cos(zn), zn))
    cands.sort(key=lambda t: t[0])
    best_val, Xbest = cands[0][0], cands[0][1].copy()

    iu = np.triu_indices(n, 1)
    I0, J0 = iu[0], iu[1]
    P = len(I0)

    def adam_C12(T0, steps, s0, s1, lr0, lr1):
        T = T0.copy()
        T /= np.linalg.norm(T, axis=1, keepdims=True)
        m = np.zeros_like(T); v = np.zeros_like(T)
        b1, b2, eps = 0.9, 0.999, 1e-8
        b1p, b2p = b1, b2
        ls0, ls1 = np.log(s0), np.log(s1)
        ll0, ll1 = np.log(lr0), np.log(lr1)
        gradX = np.empty((73, 13))
        gradT = np.empty_like(T)
        for k in range(steps):
            frac = k / (steps - 1) if steps > 1 else 1.0
            s = np.exp(ls0 + (ls1 - ls0) * frac)
            lr = np.exp(ll0 + (ll1 - ll0) * frac)
            X = _gen_C12(T)
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            c = G[I0, J0]
            cmax = c.max()
            e = np.exp(s * (c - cmax))
            se = e.sum()
            if se == 0 or not np.isfinite(se):
                w = np.zeros_like(c); w[np.argmax(c)] = 1.0
            else:
                w = e / se
            gradX.fill(0.0)
            np.add.at(gradX, I0, w[:, None] * X[J0])
            np.add.at(gradX, J0, w[:, None] * X[I0])
            for kk in range(6):
                Gk = gradX[kk*12:(kk+1)*12, :]
                ga = np.zeros(12)
                for ss in range(12):
                    ga += np.roll(Gk[ss, :12], -ss)
                gb = float(Gk[:, 12].sum())
                gradT[kk, :12] = ga
                gradT[kk, 12] = gb
            dot = (gradT * T).sum(axis=1, keepdims=True)
            gt = gradT - dot * T
            m *= b1; m += (1 - b1) * gt
            v *= b2; v += (1 - b2) * (gt * gt)
            mhat = m / (1 - b1p); vhat = v / (1 - b2p)
            b1p *= b1; b2p *= b2
            step = mhat / (np.sqrt(vhat) + eps)
            T -= lr * step
            nn = np.linalg.norm(T, axis=1, keepdims=True); nn[nn == 0] = 1.0
            T /= nn
            if (k & 255) == 255 and time.time() - t0 > SEARCH_END - 40:
                break
        return T

    def adam_C13(U0, F0, steps, s0, s1, lr0, lr1):
        U = U0.copy(); F = F0.copy()
        U /= np.linalg.norm(U, axis=1, keepdims=True)
        F /= np.linalg.norm(F, axis=1, keepdims=True)
        mU = np.zeros_like(U); vU = np.zeros_like(U)
        mF = np.zeros_like(F); vF = np.zeros_like(F)
        b1, b2, eps = 0.9, 0.999, 1e-8
        b1p, b2p = b1, b2
        ls0, ls1 = np.log(s0), np.log(s1)
        ll0, ll1 = np.log(lr0), np.log(lr1)
        gradX = np.empty((73, 13))
        for k in range(steps):
            frac = k / (steps - 1) if steps > 1 else 1.0
            s = np.exp(ls0 + (ls1 - ls0) * frac)
            lr = np.exp(ll0 + (ll1 - ll0) * frac)
            X = _gen_C13(U, F)
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            c = G[I0, J0]
            cmax = c.max()
            e = np.exp(s * (c - cmax))
            se = e.sum()
            if se == 0 or not np.isfinite(se):
                w = np.zeros_like(c); w[np.argmax(c)] = 1.0
            else:
                w = e / se
            gradX.fill(0.0)
            np.add.at(gradX, I0, w[:, None] * X[J0])
            np.add.at(gradX, J0, w[:, None] * X[I0])
            gU = np.empty_like(U)
            for kk in range(5):
                Gk = gradX[kk*13:(kk+1)*13, :]
                ga = np.zeros(13)
                for ss in range(13):
                    ga += np.roll(Gk[ss, :], -ss)
                gU[kk, :] = ga
            gF = gradX[65:, :]
            for arr, g, mm, vv in ((U, gU, mU, vU), (F, gF, mF, vF)):
                pass
            # tangent project
            dotU = (gU * U).sum(axis=1, keepdims=True)
            gtU = gU - dotU * U
            dotF = (gF * F).sum(axis=1, keepdims=True)
            gtF = gF - dotF * F
            mU *= b1; mU += (1 - b1) * gtU
            vU *= b2; vU += (1 - b2) * (gtU * gtU)
            mF *= b1; mF += (1 - b1) * gtF
            vF *= b2; vF += (1 - b2) * (gtF * gtF)
            mhatU = mU / (1 - b1p); vhatU = vU / (1 - b2p)
            mhatF = mF / (1 - b1p); vhatF = vF / (1 - b2p)
            b1p *= b1; b2p *= b2
            U -= lr * mhatU / (np.sqrt(vhatU) + eps)
            F -= lr * mhatF / (np.sqrt(vhatF) + eps)
            U /= np.linalg.norm(U, axis=1, keepdims=True)
            F /= np.linalg.norm(F, axis=1, keepdims=True)
            if (k & 255) == 255 and time.time() - t0 > SEARCH_END - 40:
                break
        return U, F

    def smooth_run(X0, steps, s0, s1, lr0, lr1):
        X = X0.copy()
        m1 = np.zeros_like(X); v1 = np.zeros_like(X)
        b1, b2, eps = 0.9, 0.999, 1e-8
        b1p, b2p = b1, b2
        ls0, ls1 = np.log(s0), np.log(s1)
        ll0, ll1 = np.log(lr0), np.log(lr1)
        grad = np.empty_like(X)
        for k in range(steps):
            frac = k / (steps - 1) if steps > 1 else 1.0
            s = np.exp(ls0 + (ls1 - ls0) * frac)
            lr = np.exp(ll0 + (ll1 - ll0) * frac)
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            c = G[I0, J0]
            cmax = c.max()
            e = np.exp(s * (c - cmax))
            ss = e.sum()
            if ss == 0 or not np.isfinite(ss):
                w = np.zeros_like(c); w[np.argmax(c)] = 1.0
            else:
                w = e / ss
            grad.fill(0.0)
            np.add.at(grad, I0, w[:, None] * X[J0])
            np.add.at(grad, J0, w[:, None] * X[I0])
            dot = (grad * X).sum(axis=1, keepdims=True)
            gt = grad - dot * X
            m1 *= b1; m1 += (1 - b1) * gt
            v1 *= b2; v1 += (1 - b2) * (gt * gt)
            mhat = m1 / (1 - b1p); vhat = v1 / (1 - b2p)
            b1p *= b1; b2p *= b2
            X -= lr * mhat / (np.sqrt(vhat) + eps)
            nn = np.linalg.norm(X, axis=1, keepdims=True); nn[nn == 0] = 1.0
            X /= nn
            if (k & 255) == 255 and time.time() - t0 > SEARCH_END:
                break
        return X

    # Phase 1: C12 symmetric search, several random templates
    sym_best_val = 1e9
    sym_best_X = None
    n_sym_trials = 10
    for ti in range(n_sym_trials):
        if time.time() - t0 > 42:
            break
        T0 = rng.randn(6, 13)
        T0 /= np.linalg.norm(T0, axis=1, keepdims=True)
        steps = int(rng.choice([800, 1100, 1500]))
        s0 = float(10 ** rng.uniform(1.2, 1.7))
        s1 = float(10 ** rng.uniform(2.3, 2.9))
        if s1 < s0:
            s0, s1 = s1, s0
        lr0 = float(10 ** rng.uniform(-2.0, -1.4))
        lr1 = float(10 ** rng.uniform(-3.0, -2.3))
        Topt = adam_C12(T0, steps, s0, s1, lr0, lr1)
        Xsym = _gen_C12(Topt)
        v = _max_cos(Xsym)
        if v < sym_best_val:
            sym_best_val = v
            sym_best_X = Xsym.copy()
    # Phase 1b: C13 five-orbit + 8 free, if time permits
    c13_best_val = 1e9
    c13_best_X = None
    n_c13 = 6
    for ti in range(n_c13):
        if time.time() - t0 > 70:
            break
        U0 = rng.randn(5, 13); U0 /= np.linalg.norm(U0, axis=1, keepdims=True)
        F0 = rng.randn(8, 13); F0 /= np.linalg.norm(F0, axis=1, keepdims=True)
        steps = int(rng.choice([800, 1100]))
        s0 = float(10 ** rng.uniform(1.2, 1.7))
        s1 = float(10 ** rng.uniform(2.3, 2.8))
        if s1 < s0:
            s0, s1 = s1, s0
        lr0 = float(10 ** rng.uniform(-2.0, -1.4))
        lr1 = float(10 ** rng.uniform(-3.0, -2.3))
        Uopt, Fopt = adam_C13(U0, F0, steps, s0, s1, lr0, lr1)
        Xc = _gen_C13(Uopt, Fopt)
        v = _max_cos(Xc)
        if v < c13_best_val:
            c13_best_val = v
            c13_best_X = Xc.copy()
    # choose symmetry base
    sym_base = None
    sym_base_val = 1e9
    if sym_best_X is not None and sym_best_val < sym_base_val:
        sym_base, sym_base_val = sym_best_X, sym_best_val
    if c13_best_X is not None and c13_best_val < sym_base_val:
        sym_base, sym_base_val = c13_best_X, c13_best_val

    # Phase 2: break symmetry with perturbations 1e-4..1e-2 + unconstrained smooth
    sigmas = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    it = 0
    while time.time() - t0 < SEARCH_END:
        it += 1
        if sym_base is not None and rng.rand() < 0.75:
            base = sym_base
            base_val = sym_base_val
        else:
            base = Xbest
            base_val = best_val
        sigma = float(rng.choice(sigmas))
        # occasionally mix sigma
        if rng.rand() < 0.3:
            sigma = float(10 ** rng.uniform(-4, -2))
        X0 = base + sigma * rng.randn(n, d)
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        steps = int(rng.choice([700, 1000, 1400, 1800]))
        s0 = float(10 ** rng.uniform(1.3, 1.9))
        s1 = float(10 ** rng.uniform(2.2, 2.9))
        if s1 < s0:
            s0, s1 = s1, s0
        lr0 = float(10 ** rng.uniform(-2.2, -1.6))
        lr1 = float(10 ** rng.uniform(-3.6, -2.9))
        X1 = smooth_run(X0, steps, s0, s1, lr0, lr1)
        v1 = _max_cos(X1)
        if v1 < best_val:
            best_val = v1
            Xbest = X1.copy()
        # also track improvement of symmetric basin lightly (don't overwrite sym_base with broken one)
        if time.time() - t0 >= SEARCH_END:
            break
        # quick escape if very close to timeout
        if time.time() - t0 > HARD - 25:
            break

    # Final LP polish with remaining time (reserve ~1s for return)
    remain = HARD - (time.time() - t0) - 1.0
    if remain > 2:
        budget = min(remain, 18.0)
        try:
            Xp = _lp_polish(Xbest, seconds=budget)
            vp = _max_cos(Xp)
            if vp < best_val and np.all(np.isfinite(Xp)):
                Xbest, best_val = Xp, vp
        except Exception:
            pass
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest
