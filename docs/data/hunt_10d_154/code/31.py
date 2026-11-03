import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _fingerprint(X, slack=0.005):
    n = X.shape[0]
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    A = (G > m - slack).astype(float)
    np.fill_diagonal(A, 0.0)
    deg = np.sort(np.sum(A, axis=1))
    try:
        ev = np.linalg.eigvalsh(A)
        top = np.sort(ev)[-3:]
    except Exception:
        top = np.zeros(3)
    return deg, top, m

def _fp_dist(f1, f2):
    d1, t1, _ = f1
    d2, t2, _ = f2
    dd = float(np.linalg.norm(d1 - d2))
    de = float(np.linalg.norm(t1 - t2))
    return dd + 3.0 * de

def _brief_opt(X, kind, p, steps, lr0, deadline):
    n, d = X.shape
    m_ad = np.zeros_like(X)
    v_ad = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    for t in range(1, steps + 1):
        if t % 100 == 0 and time.time() > deadline:
            break
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        if kind == 'lse':
            alpha = float(p)
            M = float(np.max(G))
            E = np.exp(alpha * (G - M))
            np.fill_diagonal(E, 0.0)
            S = float(np.sum(E)) * 0.5
            if not np.isfinite(S) or S <= 0:
                break
            Grad = (E @ X) / max(S, 1e-18)
        else:
            D = 2.0 - 2.0 * G
            np.fill_diagonal(D, 1.0)
            D = np.clip(D, 1e-6, 4.0)
            logD = np.log(D)
            coeff = -(float(p) * 0.5 + 1.0)
            L = coeff * logD
            np.fill_diagonal(L, -1e18)
            Lm = float(np.max(L))
            W = np.exp(L - Lm)
            np.fill_diagonal(W, 0.0)
            S = float(np.sum(W)) * 0.5
            if not np.isfinite(S) or S <= 0:
                break
            Grad = (W @ X) / max(S, 1e-18)
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        b1p *= b1
        b2p *= b2
        m_ad = b1 * m_ad + (1.0 - b1) * Gt
        v_ad = b2 * v_ad + (1.0 - b2) * (Gt * Gt)
        mhat = m_ad / (1.0 - b1p)
        vhat = v_ad / (1.0 - b2p)
        step = mhat / (np.sqrt(vhat) + eps)
        X = X - lr0 * step
        X = _norm(X)
    return X

def _lp_polish(X, seconds, hard_end):
    X = _norm(X)
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < hard_end:
        if radius < 1e-14:
            break
        c = (X @ X.T)[iu]
        m = float(np.max(c))
        sel = c > m - max(20.0 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            y = X + res.x[:-1].reshape(n, d)
        except Exception:
            radius *= 0.5
            continue
        y = _norm(y)
        if not np.all(np.isfinite(y)):
            radius *= 0.5
            continue
        v = _max_cos(y)
        if v < best:
            X, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def solve():
    t0 = time.time()
    HARD = t0 + 165.0
    T_GEN_END = t0 + 82.0
    T_LP_END = t0 + 136.0
    rng = np.random.default_rng(7)
    n, d = 154, 10
    Xbest = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = _norm(a)
                if np.all(np.isfinite(a)):
                    v = _max_cos(a)
                    if v < best_val:
                        best_val = v
                        Xbest = a.copy()
        except Exception:
            continue
    if Xbest is None:
        Xbest = _norm(rng.standard_normal((n, d)))
        best_val = _max_cos(Xbest)
    Xinc = Xbest.copy()
    try:
        f_inc = _fingerprint(Xinc)
    except Exception:
        return _norm(Xinc)
    tested = [f_inc]
    pool = []
    energies = [('lse', 32), ('riesz', 16), ('lse', 64), ('riesz', 32), ('lse', 16), ('riesz', 64), ('lse', 8), ('riesz', 8)]
    trial = 0
    try:
        while time.time() < T_GEN_END and time.time() < HARD - 40:
            trial += 1
            ei = energies[(trial - 1) % len(energies)]
            kind, p = ei
            r = trial % 6
            try:
                if r == 0 or r == 1:
                    Xtry = rng.standard_normal((n, d))
                    Xtry = _norm(Xtry)
                    steps, lr0 = 650, 0.03
                elif r == 2:
                    sig = 0.35 + 0.1 * ((trial // 6) % 3)
                    Xtry = Xinc + sig * rng.standard_normal((n, d))
                    Xtry = _norm(Xtry)
                    steps, lr0 = 550, 0.025
                elif r == 3:
                    k = int(rng.integers(12, 28))
                    idx = rng.choice(n, size=k, replace=False)
                    Xtry = Xinc.copy()
                    Xtry[idx] = rng.standard_normal((k, d))
                    Xtry = _norm(Xtry + 0.02 * rng.standard_normal((n, d)))
                    steps, lr0 = 600, 0.025
                elif r == 4:
                    Gc = Xinc @ Xinc.T
                    np.fill_diagonal(Gc, -2.0)
                    thr = best_val - 0.01
                    deg = np.sum(Gc > thr, axis=1)
                    prob = (deg.astype(float) + 1.0)
                    prob /= prob.sum()
                    k = int(rng.integers(8, 18))
                    idx = rng.choice(n, size=k, replace=False, p=prob)
                    Xtry = Xinc.copy()
                    Xtry[idx] = rng.standard_normal((k, d))
                    Xtry = _norm(Xtry + 0.015 * rng.standard_normal((n, d)))
                    steps, lr0 = 600, 0.025
                else:
                    Xtry = Xinc + 0.55 * rng.standard_normal((n, d))
                    Xtry = _norm(Xtry)
                    steps, lr0 = 500, 0.03
            except Exception:
                continue
            try:
                Xopt = _brief_opt(Xtry, kind, p, steps, lr0, deadline=T_GEN_END + 2.0)
                Xopt = _norm(Xopt)
                val = _max_cos(Xopt)
                if not np.isfinite(val):
                    continue
                fp = _fingerprint(Xopt)
            except Exception:
                continue
            dup = False
            try:
                for f in tested:
                    if _fp_dist(fp, f) < 12.0:
                        dup = True
                        break
            except Exception:
                dup = False
            tested.append(fp)
            if dup:
                continue
            pool.append((val, Xopt.copy(), fp, kind, p))
            if len(pool) > 20:
                pool.sort(key=lambda z: z[0])
                pool = pool[:20]
            if time.time() > T_GEN_END:
                break
    except Exception:
        return _norm(Xbest)
    try:
        pool.sort(key=lambda z: z[0])
    except Exception:
        pass
    K = min(5, len(pool))
    cand_lp = pool[:K] if K > 0 else []
    try:
        for k in range(len(cand_lp)):
            if time.time() > T_LP_END or time.time() > HARD - 30:
                break
            val0, X0, _, _, _ = cand_lp[k]
            if val0 > best_val + 0.02:
                continue
            secs = min(8.0, max(2.0, (T_LP_END - time.time()) / max(1, len(cand_lp) - k)))
            if secs < 1.0:
                break
            if time.time() + secs > HARD - 28:
                secs = max(1.0, HARD - 28 - time.time())
            try:
                Xp = _lp_polish(X0, secs, hard_end=HARD - 26)
                vp = _max_cos(Xp)
            except Exception:
                continue
            if vp < best_val:
                Xbest = _norm(Xp)
                best_val = vp
    except Exception:
        return _norm(Xbest)
    try:
        remain = min(HARD - time.time() - 1.0, 28.0)
        if remain > 3.0 and time.time() < HARD - 2.0:
            try:
                Xf = _lp_polish(Xbest, remain, hard_end=HARD - 0.5)
                vf = _max_cos(Xf)
                if np.isfinite(vf) and vf < best_val:
                    Xbest, best_val = _norm(Xf), vf
            except Exception:
                pass
    except Exception:
        pass
    Xbest = _norm(Xbest)
    return Xbest
