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

def _gradX_softmax(X, alpha):
    G = X @ X.T
    np.fill_diagonal(G, -10.0)
    cmax = float(np.max(G))
    W = np.exp(alpha * (G - cmax))
    np.fill_diagonal(W, 0.0)
    s = float(np.sum(W))
    if not np.isfinite(s) or s <= 0:
        return np.zeros_like(X), float(cmax), 0.0
    Gr = (W @ X) / s
    return Gr, cmax, s

def _adam_free_quench(X0, steps, a0, a1, lr0, lr1):
    X = _norm(X0.copy())
    m = np.zeros_like(X); v = np.zeros_like(X)
    b1, b2 = 0.9, 0.999; eps = 1e-8
    b1p, b2p = 1.0, 1.0
    for it in range(1, steps + 1):
        f = it / float(steps)
        alpha = a0 + (a1 - a0) * f
        lr = lr0 + (lr1 - lr0) * f
        Gr, _, s = _gradX_softmax(X, alpha)
        if s == 0:
            break
        proj = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - proj * X
        b1p *= b1; b2p *= b2
        m *= b1; m += (1.0 - b1) * Gt
        vv = v; vv *= b2; vv += (1.0 - b2) * Gt * Gt
        mh = m / (1.0 - b1p); vh = vv / (1.0 - b2p)
        X = X - lr * mh / (np.sqrt(vh) + eps)
        X = _norm(X)
    return X

def _adam_antipodal(Y0, steps, a0, a1, lr0, lr1):
    Y = _norm(Y0.copy())
    m = np.zeros_like(Y); v = np.zeros_like(Y)
    b1, b2 = 0.9, 0.999; eps = 1e-8
    b1p, b2p = 1.0, 1.0
    for it in range(1, steps + 1):
        f = it / float(steps)
        alpha = a0 + (a1 - a0) * f
        lr = lr0 + (lr1 - lr0) * f
        X = np.vstack([Y, -Y])
        Gr, _, s = _gradX_softmax(X, alpha)
        if s == 0:
            break
        GY = Gr[:Y.shape[0]] - Gr[Y.shape[0]:]
        proj = np.sum(GY * Y, axis=1, keepdims=True)
        Gt = GY - proj * Y
        b1p *= b1; b2p *= b2
        m *= b1; m += (1.0 - b1) * Gt
        v *= b2; v += (1.0 - b2) * Gt * Gt
        mh = m / (1.0 - b1p); vh = v / (1.0 - b2p)
        Y = Y - lr * mh / (np.sqrt(vh) + eps)
        Y = _norm(Y)
    X = np.vstack([Y, -Y])
    return Y, X

def _expand_orbit(G, S):
    X = np.empty((116, 6))
    for i in range(9):
        g = G[i]
        for k in range(6):
            X[i * 12 + k] = np.roll(g, k)
            X[i * 12 + 6 + k] = -np.roll(g, k)
    for j in range(2):
        p = S[j]
        n = float(np.linalg.norm(p))
        if n < 1e-12:
            p = np.array([1.0, 0.0])
        else:
            p = p / n
        vv = np.array([p[0], p[1], p[0], p[1], p[0], p[1]]) / np.sqrt(3.0)
        pv = np.roll(vv, 1)
        X[108 + j * 4 + 0] = vv
        X[108 + j * 4 + 1] = pv
        X[108 + j * 4 + 2] = -vv
        X[108 + j * 4 + 3] = -pv
    return X

def _adam_orbit(G0, S0, steps, a0, a1, lr0, lr1):
    G = _norm(G0.copy())
    S = S0.copy()
    sn = np.linalg.norm(S, axis=1, keepdims=True); sn[sn == 0] = 1.0
    S = S / sn
    mG = np.zeros_like(G); vG = np.zeros_like(G)
    mS = np.zeros_like(S); vS = np.zeros_like(S)
    b1, b2 = 0.9, 0.999; eps = 1e-8
    b1p, b2p = 1.0, 1.0
    for it in range(1, steps + 1):
        f = it / float(steps)
        alpha = a0 + (a1 - a0) * f
        lr = lr0 + (lr1 - lr0) * f
        X = _expand_orbit(G, S)
        Gr, _, s = _gradX_softmax(X, alpha)
        if s == 0:
            break
        # pullback generic
        GG = np.zeros_like(G)
        for i in range(9):
            acc = np.zeros(6)
            for k in range(6):
                acc += np.roll(Gr[i * 12 + k], -k)
                acc += -np.roll(Gr[i * 12 + 6 + k], -k)
            GG[i] = acc
        # pullback small
        GS = np.zeros_like(S)
        for j in range(2):
            g0 = Gr[108 + j * 4 + 0]
            g1 = Gr[108 + j * 4 + 1]
            g2 = Gr[108 + j * 4 + 2]
            g3 = Gr[108 + j * 4 + 3]
            c = g0 + np.roll(g1, -1) - g2 - np.roll(g3, -1)
            # B^T / sqrt3
            GS[j, 0] = (c[0] + c[2] + c[4]) / np.sqrt(3.0)
            GS[j, 1] = (c[1] + c[3] + c[5]) / np.sqrt(3.0)
        projG = np.sum(GG * G, axis=1, keepdims=True)
        GtG = GG - projG * G
        projS = np.sum(GS * S, axis=1, keepdims=True)
        GtS = GS - projS * S
        b1p *= b1; b2p *= b2
        mG *= b1; mG += (1.0 - b1) * GtG
        vG *= b2; vG += (1.0 - b2) * GtG * GtG
        mS *= b1; mS += (1.0 - b1) * GtS
        vS *= b2; vS += (1.0 - b2) * GtS * GtS
        mhG = mG / (1.0 - b1p); vhG = vG / (1.0 - b2p)
        mhS = mS / (1.0 - b1p); vhS = vS / (1.0 - b2p)
        G = G - lr * mhG / (np.sqrt(vhG) + eps)
        S = S - lr * mhS / (np.sqrt(vhS) + eps)
        G = _norm(G)
        sn2 = np.linalg.norm(S, axis=1, keepdims=True); sn2[sn2 == 0] = 1.0
        S = S / sn2
    X = _expand_orbit(G, S)
    return G, S, X

def _lp_polish(X, deadline):
    X = _norm(X.copy())
    n, d = X.shape
    best = _maxcos(X)
    bx = X.copy()
    radius = 2.0e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        if time.time() > deadline - 0.4:
            break
        G = X @ X.T
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
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = _norm(y)
        vv = _maxcos(y)
        if vv < best:
            best = vv; bx = y.copy(); X = y
            radius = min(radius * 1.5, 0.05)
        else:
            if vv <= best + 1e-15:
                X = y
            radius *= 0.5
    return bx, best

def solve():
    t0 = time.time()
    HARD = t0 + 171.0
    END_A = t0 + 48.0
    END_B = t0 + 86.0
    END_C = t0 + 148.0
    rng = np.random.default_rng(20261031)
    # constrained archives (store expanded X)
    ant_list = []  # (val, X, Y)
    orb_list = []  # (val, X, G, S)
    # Phase A: antipodal search, many restarts
    a_steps = 1300
    cnt = 0
    while time.time() < END_A:
        cnt += 1
        Y0 = _norm(rng.standard_normal((58, 6)))
        # vary schedule slightly per restart
        if cnt % 3 == 0:
            Y, X = _adam_antipodal(Y0, a_steps, 50, 800, 0.03, 0.003)
            Y, X = _adam_antipodal(Y, 600, 800, 1500, 0.008, 0.002)
        elif cnt % 3 == 1:
            Y, X = _adam_antipodal(Y0, a_steps, 40, 600, 0.04, 0.004)
        else:
            Y, X = _adam_antipodal(Y0, a_steps, 60, 1000, 0.025, 0.002)
        v = _maxcos(X)
        ant_list.append((v, X.copy(), Y.copy()))
        ant_list.sort(key=lambda t: t[0])
        ant_list = ant_list[:6]
        if time.time() > HARD - 30:
            break
    # Phase B: orbit search 9x12+2x4
    o_steps = 1300
    cnt = 0
    while time.time() < END_B:
        cnt += 1
        G0 = _norm(rng.standard_normal((9, 6)))
        S0 = rng.standard_normal((2, 2))
        S0 /= np.linalg.norm(S0, axis=1, keepdims=True) + 1e-15
        if cnt % 2 == 0:
            G, S, X = _adam_orbit(G0, S0, o_steps, 50, 800, 0.03, 0.003)
            G, S, X = _adam_orbit(G, S, 500, 800, 1500, 0.008, 0.002)
        else:
            G, S, X = _adam_orbit(G0, S0, o_steps, 40, 600, 0.035, 0.003)
        v = _maxcos(X)
        if v < 0.999:  # ignore collapsed
            orb_list.append((v, X.copy(), G.copy(), S.copy()))
            orb_list.sort(key=lambda t: t[0])
            orb_list = orb_list[:6]
        if time.time() > HARD - 30:
            break
    # Phase C: release + perturb + free quench
    # collect constrained candidates
    candX = []
    for _, X, _ in ant_list[:3]:
        candX.append(X)
    for _, X, _, _ in orb_list[:3]:
        candX.append(X)
    # if nothing (timeout), random
    if not candX:
        candX.append(_norm(rng.standard_normal((116, 6))))
    free_best = []
    scales = [0.002, 0.006, 0.012]
    for Xc in candX:
        if time.time() > END_C:
            break
        for sc in scales:
            if time.time() > END_C:
                break
            Xp = _norm(Xc + sc * rng.standard_normal((116, 6)))
            # annealed free quench, two lengths depending on time
            Xq = _adam_free_quench(Xp, 1600, 60, 1100, 0.025, 0.002)
            # short high-beta finish
            if time.time() < END_C - 4:
                Xq = _adam_free_quench(Xq, 500, 1100, 1600, 0.006, 0.0015)
            v = _maxcos(Xq)
            free_best.append((v, Xq))
            free_best.sort(key=lambda t: t[0])
            free_best = free_best[:4]
    # also try pure random free quenches to add diversity if time left
    while time.time() < END_C - 8 and len(free_best) < 4:
        Xp = _norm(rng.standard_normal((116, 6)))
        Xq = _adam_free_quench(Xp, 1600, 60, 1000, 0.025, 0.002)
        free_best.append((_maxcos(Xq), Xq))
        free_best.sort(key=lambda t: t[0])
        free_best = free_best[:4]
    if not free_best:
        # fallback to constrained
        free_best = [(float(_maxcos(X)), X) for X in candX[:2]]
    free_best.sort(key=lambda t: t[0])
    # Phase D: LP polish best 2-3 released candidates
    polished = []
    for i, (v, X) in enumerate(free_best[:3]):
        if time.time() > HARD - 6:
            polished.append((v, X))
            continue
        budget = min(9.0, max(2.0, (HARD - 3.0 - time.time()) / (3 - i)))
        Xp, vp = _lp_polish(X, time.time() + budget)
        polished.append((vp, Xp))
    polished.sort(key=lambda t: t[0])
    best_v, best_x = polished[0]
    # final comparison with files (ignore until now)
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
    for f in ("best.npy", "record.npy"):
        try:
            c = _clean(np.load(f))
            if c is not None:
                cv = _maxcos(c)
                if cv < best_v:
                    best_v, best_x = cv, c
        except Exception:
            pass
    best_x = _norm(best_x)
    return best_x
