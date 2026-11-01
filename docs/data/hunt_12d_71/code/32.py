import time
import math
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _quench_full(X, steps, T0, T1, lr, mu=0.9, t_deadline=None):
    n, d = X.shape
    V = np.zeros_like(X)
    lt0 = np.log(T0); lt1 = np.log(T1)
    G0 = X @ X.T
    np.fill_diagonal(G0, -2.0)
    best_v = float(G0.max())
    best_X = X.copy()
    for s in range(steps):
        if t_deadline is not None and (s % 250 == 0) and time.time() > t_deadline:
            break
        frac = s / max(1, steps - 1)
        T = float(np.exp(lt0 + (lt1 - lt0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -1e100)
        m = float(G.max())
        Q = (G - m) / T
        np.clip(Q, -50, 0, out=Q)
        E = np.exp(Q)
        np.fill_diagonal(E, 0.0)
        rs = E.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        Grad = (E @ X) / rs
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        gn = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        big = gn[:, 0] > 2.0
        if np.any(big):
            Gt[big] *= (2.0 / gn[big])
        V = mu * V + Gt
        X = X - lr * V
        X = _norm_rows(X)
        if (s % 100 == 0) or (s == steps - 1):
            Gc = X @ X.T
            np.fill_diagonal(Gc, -2.0)
            vc = float(Gc.max())
            if vc < best_v:
                best_v = vc
                best_X = X.copy()
    return best_X, best_v

def _lp_polish(x, seconds, deadline=None):
    x = _norm_rows(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if deadline is not None and time.time() > deadline:
            break
        if radius < 1e-14:
            break
        cmat = (x @ x.T)[iu]
        m = float(cmat.max())
        sel = cmat > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 1500:
            order = np.argsort(-cmat[sel])[:1500]
            I, J = I[order], J[order]
            csel = cmat[sel][order]
            p = 1500
        else:
            csel = cmat[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cc = np.concatenate([I[:, None] * d + np.arange(d),
                             J[:, None] * d + np.arange(d),
                             np.full((p, 1), n * d)], 1).ravel()
        vv = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vv, (rows, cc)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
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

def _build_full(Y, P, cosA, sinA):
    n_orb = Y.shape[0]
    m = cosA.shape[0]
    N = n_orb * m + 1
    X = np.empty((N, 12), dtype=np.float64)
    for k in range(6):
        c = cosA[:, k]
        s = sinA[:, k]
        y0 = Y[:, 2 * k]
        y1 = Y[:, 2 * k + 1]
        Xo0 = y0[:, None] * c[None, :] - y1[:, None] * s[None, :]
        Xo1 = y0[:, None] * s[None, :] + y1[:, None] * c[None, :]
        X[:n_orb * m, 2 * k] = Xo0.ravel()
        X[:n_orb * m, 2 * k + 1] = Xo1.ravel()
    X[-1] = P
    return X

def _sym_quench(Y0, P0, cosA, sinA, mask12, steps, T0, T1, lr, mu=0.9):
    Y = _norm_rows(Y0.copy())
    P = P0.copy()
    pn = np.linalg.norm(P) + 1e-12
    P = P / pn
    n_orb = Y.shape[0]
    m = cosA.shape[0]
    N = n_orb * m + 1
    VY = np.zeros_like(Y)
    VP = np.zeros_like(P)
    lt0 = np.log(T0); lt1 = np.log(T1)
    X = np.empty((N, 12))
    best_v = 1e100
    bestY = Y.copy(); bestP = P.copy()
    for s in range(steps):
        frac = s / max(1, steps - 1)
        T = float(np.exp(lt0 + (lt1 - lt0) * frac))
        for k in range(6):
            c = cosA[:, k]; ss = sinA[:, k]
            y0 = Y[:, 2 * k]; y1 = Y[:, 2 * k + 1]
            Xo0 = y0[:, None] * c[None, :] - y1[:, None] * ss[None, :]
            Xo1 = y0[:, None] * ss[None, :] + y1[:, None] * c[None, :]
            X[:n_orb * m, 2 * k] = Xo0.ravel()
            X[:n_orb * m, 2 * k + 1] = Xo1.ravel()
        X[-1] = P
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        cur_max = float(G.max())
        if cur_max < best_v:
            best_v = cur_max
            bestY = Y.copy(); bestP = P.copy()
        Q = (G - cur_max) / T
        np.clip(Q, -50, 0, out=Q)
        E = np.exp(Q)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if not np.isfinite(S) or S <= 0:
            S = 1.0
        Grad = (E @ X) / S
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        Gorb = Gt[:-1].reshape(n_orb, m, 12)
        Gpole = Gt[-1]
        Ygrad = np.empty_like(Y)
        for k in range(6):
            c = cosA[:, k]; ss = sinA[:, k]
            g0 = Gorb[:, :, 2 * k]
            g1 = Gorb[:, :, 2 * k + 1]
            Ygrad[:, 2 * k] = ((g0 * c[None, :] + g1 * ss[None, :]).sum(axis=1)) / float(m)
            Ygrad[:, 2 * k + 1] = ((-g0 * ss[None, :] + g1 * c[None, :]).sum(axis=1)) / float(m)
        dotY = np.sum(Ygrad * Y, axis=1, keepdims=True)
        GtY = Ygrad - dotY * Y
        gn = np.linalg.norm(GtY, axis=1, keepdims=True) + 1e-12
        big = gn[:, 0] > 2.0
        if np.any(big):
            GtY[big] *= (2.0 / gn[big])
        gp_sub = Gpole * mask12
        gp_dot = float(gp_sub @ P)
        GtP = gp_sub - gp_dot * P
        gpn = float(np.linalg.norm(GtP)) + 1e-12
        if gpn > 2.0:
            GtP *= (2.0 / gpn)
        VY = mu * VY + GtY
        VP = mu * VP + GtP
        Y = Y - lr * VY
        Y = _norm_rows(Y)
        P = P - lr * VP
        P = P * mask12
        pn2 = np.linalg.norm(P) + 1e-12
        if not np.isfinite(pn2) or pn2 < 1e-9:
            P = bestP.copy()
            VP = np.zeros_like(P)
        else:
            P = P / pn2
    return bestY, bestP, best_v

def solve():
    t0 = time.time()
    TIME_LIMIT = 173.0
    rng = np.random.default_rng(7)
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = _norm_rows(np.asarray(np.load(fn), dtype=np.float64))
            if a.shape == (71, 12):
                cands.append(a)
        except Exception:
            pass
    if not cands:
        cands.append(_norm_rows(rng.standard_normal(size=(71, 12))))
    bestX = min(cands, key=_max_cos).copy()
    bestVal = _max_cos(bestX)
    # ---- symmetry-constrained scan ----
    sym_deadline = t0 + 62.0
    sym_archive = []
    trials = 0
    m_choices = [7, 10, 14, 35, 70]
    while time.time() < sym_deadline and trials < 400:
        trials += 1
        m = int(m_choices[rng.integers(0, len(m_choices))])
        n_orb = 70 // m
        f = np.zeros(6, dtype=int)
        f[0] = 0
        for k in range(1, 6):
            f[k] = int(rng.integers(0, m))
        # enforce full-orbit possibility: gcd of active freqs with m == 1
        g = m
        for k in range(1, 6):
            g = math.gcd(g, int(f[k]))
        if g != 1:
            # force a coprime frequency
            kfix = int(rng.integers(1, 6))
            # pick 1 which is coprime to all m
            f[kfix] = 1
        # limit number of extra zeros to keep fixed subspace small
        nz = int(np.sum(f[1:] == 0))
        if nz > 1 and rng.random() < 0.8:
            for k in range(1, 6):
                if f[k] == 0 and rng.random() < 0.7:
                    f[k] = int(rng.integers(1, m))
        tvec = np.arange(m, dtype=np.float64)
        cosA = np.empty((m, 6)); sinA = np.empty((m, 6))
        for k in range(6):
            ang = 2.0 * np.pi * tvec * float(f[k]) / float(m)
            cosA[:, k] = np.cos(ang); sinA[:, k] = np.sin(ang)
        mask12 = np.zeros(12)
        for k in range(6):
            if f[k] == 0:
                mask12[2 * k] = 1.0; mask12[2 * k + 1] = 1.0
        Y0 = _norm_rows(rng.standard_normal(size=(n_orb, 12)))
        Praw = rng.standard_normal(size=(12,)) * mask12
        if np.linalg.norm(Praw) < 1e-9:
            Praw[0] = 1.0
        P0 = Praw / (np.linalg.norm(Praw) + 1e-12)
        T0 = float(rng.choice([0.014, 0.010, 0.008, 0.012]))
        T1 = float(rng.choice([0.0008, 0.0005, 0.0004, 0.0010]))
        lr = float(rng.choice([0.06, 0.05, 0.04, 0.07]))
        steps = 300
        try:
            bY, bP, bv = _sym_quench(Y0, P0, cosA, sinA, mask12, steps, T0, T1, lr, mu=0.9)
        except Exception:
            continue
        try:
            Xf = _build_full(bY, bP, cosA, sinA)
            Xf = _norm_rows(Xf)
            vv = _max_cos(Xf)
        except Exception:
            continue
        sym_archive.append((vv, Xf.copy(), int(m), f.copy()))
        sym_archive.sort(key=lambda t: t[0])
        del sym_archive[10:]
        if vv < bestVal:
            bestVal = vv
            bestX = Xf.copy()
        if time.time() > sym_deadline:
            break
    # ---- release symmetric tops with full quench + LP ----
    sym_archive_sorted = sorted(sym_archive, key=lambda t: t[0])[:6]
    for (sv, sx, mm, ff) in sym_archive_sorted:
        if time.time() > t0 + 110.0:
            break
        try:
            y, _ = _quench_full(sx.copy(), steps=1300, T0=0.008, T1=0.0004, lr=0.05, mu=0.9, t_deadline=t0 + 108.0)
            v = _max_cos(y)
            if v < sv + 0.01:
                y2, _ = _quench_full(y.copy(), steps=500, T0=0.0008, T1=0.00015, lr=0.02, mu=0.9, t_deadline=t0 + 109.0)
                v2 = _max_cos(y2)
                if v2 < v:
                    y, v = y2, v2
            if v < bestVal:
                bestVal = v
                bestX = y.copy()
        except Exception:
            continue
    # ---- baseline basin hopping from best (guarantee) ----
    curX = bestX.copy(); curVal = bestVal
    explore_deadline = t0 + 150.0
    it = 0
    while time.time() < explore_deadline:
        it += 1
        base = bestX if rng.random() < 0.7 else curX
        Xt = base.copy()
        k = int(rng.choice([1, 2, 2, 3, 4]))
        idx = rng.choice(71, size=k, replace=False)
        if rng.random() < 0.05:
            Xt[idx] = _norm_rows(rng.standard_normal(size=(k, 12)))
        else:
            sigma = float(np.exp(rng.uniform(np.log(0.015), np.log(0.45))))
            Xt[idx] = Xt[idx] + rng.standard_normal(size=(k, 12)) * sigma
            Xt = _norm_rows(Xt)
        T0 = float(rng.choice([0.010, 0.007, 0.015, 0.012]))
        T1 = float(rng.choice([0.0005, 0.001, 0.0003, 0.0008]))
        lr = float(rng.choice([0.05, 0.03, 0.07, 0.04]))
        steps = int(rng.choice([1400, 2000, 1100, 1800]))
        try:
            Xt, _ = _quench_full(Xt, steps=steps, T0=T0, T1=T1, lr=lr, mu=0.9, t_deadline=explore_deadline)
            v = _max_cos(Xt)
        except Exception:
            continue
        if v < bestVal + 0.003 and time.time() < explore_deadline - 0.5:
            try:
                Xt2, _ = _quench_full(Xt.copy(), steps=400, T0=0.0008, T1=0.0002, lr=0.02, mu=0.9, t_deadline=explore_deadline)
                v2 = _max_cos(Xt2)
                if v2 < v:
                    Xt, v = Xt2, v2
            except Exception:
                pass
        if v < bestVal:
            bestVal = v; bestX = Xt.copy(); curX = Xt.copy(); curVal = v
        else:
            if v < curVal or rng.random() < np.exp(-(v - curVal) / 0.0015):
                curX = Xt.copy(); curVal = v
        if it % 25 == 0:
            curX = bestX.copy(); curVal = bestVal
    # ---- final polish ----
    if time.time() < t0 + TIME_LIMIT - 3:
        try:
            y, _ = _quench_full(bestX.copy(), steps=1500, T0=0.001, T1=0.00012, lr=0.018, mu=0.9, t_deadline=t0 + TIME_LIMIT - 8)
            vy = _max_cos(y)
            if vy < bestVal:
                bestVal = vy; bestX = y.copy()
        except Exception:
            pass
        rem = TIME_LIMIT - (time.time() - t0)
        if rem > 2.5:
            try:
                y2 = _lp_polish(bestX, seconds=rem - 0.6, deadline=t0 + TIME_LIMIT - 0.3)
                v2 = _max_cos(y2)
                if v2 < bestVal:
                    bestX = y2.copy(); bestVal = v2
            except Exception:
                pass
    bestX = _norm_rows(bestX)
    return bestX
