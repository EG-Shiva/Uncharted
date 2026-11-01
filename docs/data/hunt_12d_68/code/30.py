import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _per_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return np.max(G, axis=1)

def _make_fg(n, d, IU, beta):
    def fg(yf):
        Y = yf.reshape(n, d)
        nr = np.sqrt(np.sum(Y*Y, axis=1, keepdims=True))
        nr = np.maximum(nr, 1e-12)
        X = Y / nr
        G = X @ X.T
        g = G[IU]
        m = float(np.max(g))
        e = np.exp(beta * (g - m))
        S = float(np.sum(e)) + 1e-300
        F = m + np.log(S) / beta
        w = e / S
        P = np.zeros((n, n), dtype=np.float64)
        P[IU] = w
        P = P + P.T
        GX = P @ X
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GY = (GX - dot * X) / nr
        return F, GY.ravel()
    return fg

def _run_lbfgs(x0, beta, maxiter):
    n, d = x0.shape
    IU = np.triu_indices(n, 1)
    fg = _make_fg(n, d, IU, beta)
    y0 = x0.ravel().copy()
    try:
        r = minimize(fg, y0, jac=True, method='L-BFGS-B',
                     options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-7,
                              'maxfun': maxiter*3, 'maxcor': 12})
        Y = r.x.reshape(n, d)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        return Y
    except Exception:
        return x0

def _quench(x0, betas, iters):
    X = x0
    for b, it in zip(betas, iters):
        X = _run_lbfgs(X, b, it)
    return X

def _best_hole(rest, rng, ncand=8000):
    d = rest.shape[1]
    cand = rng.standard_normal((ncand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    scores = np.max(cand @ rest.T, axis=1)
    return cand[int(np.argmin(scores))]

def _lp_polish(X, deadline, cap_p=2000):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = 2e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        c = (X @ X.T)[iu]
        m = float(np.max(c))
        sel = c > m - max(20.0 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        csel = c[sel]
        if p > cap_p:
            o = np.argsort(-csel)[:cap_p]
            I, J = I[o], J[o]
            csel = csel[o]
            p = cap_p
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None]*d + np.arange(d),
                               J[:, None]*d + np.arange(d),
                               np.full((p, 1), n*d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n*d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)]*(n*d)+[(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        Y = X + res.x[:-1].reshape(n, d)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        v = _max_cos(Y)
        if v < best:
            X, best, radius = Y, v, min(radius*1.6, 0.05)
        else:
            radius *= 0.5
    return X

def solve():
    t0 = time.time()
    T_END = t0 + 172.0
    rng = np.random.default_rng(987654321)
    n, d = 68, 12
    bestX = None
    best = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=np.float64)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                s = _max_cos(a)
                if np.all(np.isfinite(a)) and s < best:
                    best = s
                    bestX = a.copy()
        except Exception:
            pass
    if bestX is None:
        bestX = rng.standard_normal((n, d))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        best = _max_cos(bestX)
    base_best = bestX.copy()

    def time_left():
        return T_END - time.time()

    # Phase 1: diverse basin enumeration
    phase1_end = t0 + 95.0
    trial = 0
    # first squeeze current best a little with high beta to set bar
    if time_left() > 20:
        try:
            Y0 = _quench(bestX.copy(), [120.0, 400.0, 900.0], [200, 200, 250])
            s0 = _max_cos(Y0)
            if s0 < best:
                best, bestX = s0, Y0.copy()
        except Exception:
            pass
    while time.time() < phase1_end and time_left() > 70.0:
        trial += 1
        kind = trial % 6
        Xi = None
        if kind == 0:
            Xi = rng.standard_normal((n, d))
            Xi /= np.linalg.norm(Xi, axis=1, keepdims=True)
            betas, iters = [60.0, 220.0, 650.0], [220, 220, 300]
        elif kind == 1:
            sig = 0.35 + 0.15 * ((trial // 6) % 3)
            Xi = base_best + sig * rng.standard_normal((n, d))
            Xi /= np.linalg.norm(Xi, axis=1, keepdims=True)
            betas, iters = [80.0, 280.0, 700.0], [220, 220, 280]
        elif kind == 2:
            Xi = base_best.copy()
            k = 8 + int(rng.integers(0, 6))
            idx = rng.choice(n, size=k, replace=False)
            R = rng.standard_normal((k, d))
            R /= np.linalg.norm(R, axis=1, keepdims=True)
            Xi[idx] = R
            betas, iters = [70.0, 250.0, 650.0], [220, 220, 280]
        elif kind == 3:
            # rattler hole reinsertion from current best
            Xi = bestX.copy()
            G = Xi @ Xi.T; np.fill_diagonal(G, -2.0)
            pm = np.max(G, axis=1)
            order = np.argsort(pm)  # smallest first = rattlers
            k = 3 + (trial // 6) % 3
            for kk in range(k):
                victim = int(order[kk])
                rest = np.delete(Xi, victim, axis=0)
                h = _best_hole(rest, rng, ncand=6000)
                h = h + 0.01 * rng.standard_normal(d)
                h /= np.linalg.norm(h)
                Xi[victim] = h
            Xi += 0.015 * rng.standard_normal((n, d))
            Xi /= np.linalg.norm(Xi, axis=1, keepdims=True)
            betas, iters = [120.0, 350.0, 800.0], [200, 200, 280]
        elif kind == 4:
            # cap transplant: coherent move of a cluster
            Xi = bestX.copy()
            u = rng.standard_normal(d); u /= np.linalg.norm(u)
            proj = Xi @ u
            msz = 5 + int(rng.integers(0, 8))
            idx = np.argsort(proj)[-msz:]
            v = rng.standard_normal(d); v -= (v @ u) * u
            vv = np.linalg.norm(v)
            if vv < 1e-12:
                v = rng.standard_normal(d); vv = np.linalg.norm(v)
            v /= vv
            alpha = 0.2 + 0.15 * ((trial // 6) % 3)
            Xi[idx] += alpha * v + 0.03 * rng.standard_normal((msz, d))
            Xi /= np.linalg.norm(Xi, axis=1, keepdims=True)
            betas, iters = [90.0, 300.0, 750.0], [220, 220, 280]
        else:
            # antipode-mixed random: random + far-apart Coulomb pre-spread
            Xi = rng.standard_normal((n, d))
            Xi /= np.linalg.norm(Xi, axis=1, keepdims=True)
            for _ in range(120):
                G = Xi @ Xi.T
                np.fill_diagonal(G, 0.0)
                # simple repulsion step
                W = np.clip(G, -1, 0.9)
                W = W ** 2 * np.sign(W)
                np.fill_diagonal(W, 0.0)
                Gr = W @ Xi
                Gr -= np.sum(Gr * Xi, axis=1, keepdims=True) * Xi
                Xi = Xi - 0.02 * Gr
                Xi /= np.linalg.norm(Xi, axis=1, keepdims=True)
            betas, iters = [50.0, 200.0, 600.0], [220, 220, 300]
        if time_left() < 55.0:
            break
        try:
            Y = _quench(Xi, betas, iters)
        except Exception:
            continue
        s = _max_cos(Y)
        if np.all(np.isfinite(Y)) and s < best:
            # quick lock with short LP
            if time_left() > 60.0:
                try:
                    Yq = _lp_polish(Y, min(time.time()+3.0, T_END-45.0))
                    sq = _max_cos(Yq)
                    if sq < s and np.all(np.isfinite(Yq)):
                        Y, s = Yq, sq
                except Exception:
                    pass
            best, bestX = s, Y.copy()
        if trial > 120:
            break

    # Phase 2: aggressive rattler hopping around best
    phase2_end = T_END - 34.0
    hop = 0
    while time.time() < phase2_end and time_left() > 38.0:
        hop += 1
        Xi = bestX.copy()
        G = Xi @ Xi.T; np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        pm = np.max(G, axis=1)
        # victims: most rattling (lowest per-max) + one hub
        order_low = np.argsort(pm)
        k = 2 + (hop % 3)
        for kk in range(k):
            victim = int(order_low[kk])
            rest = np.delete(Xi, victim, axis=0)
            h = _best_hole(rest, rng, ncand=10000)
            # small jitter to break symmetry
            h = h + 0.004 * rng.standard_normal(d)
            h /= np.linalg.norm(h)
            Xi[victim] = h
        # also jitter a hub slightly
        hub = int(np.argmax(pm))
        Xi[hub] += 0.01 * rng.standard_normal(d)
        Xi /= np.linalg.norm(Xi, axis=1, keepdims=True)
        betas = [150.0, 450.0, 950.0]
        iters = [180, 180, 260]
        if time_left() < 42.0:
            betas, iters = [200.0, 800.0], [150, 200]
        try:
            Y = _quench(Xi, betas, iters)
        except Exception:
            continue
        s = _max_cos(Y)
        if np.all(np.isfinite(Y)) and s < best:
            best, bestX = s, Y.copy()
        if hop > 40:
            break

    # Phase 3: final squeeze: high-beta LBFGS + LP
    if time_left() > 10.0:
        try:
            Y = _quench(bestX.copy(), [300.0, 1200.0], [250, 350])
            s = _max_cos(Y)
            if s < best and np.all(np.isfinite(Y)):
                best, bestX = s, Y.copy()
        except Exception:
            pass
    if time_left() > 6.0:
        try:
            lp_t = min(28.0, time_left() - 1.5)
            Xp = _lp_polish(bestX, time.time() + lp_t)
            sp = _max_cos(Xp)
            if sp < best and np.all(np.isfinite(Xp)):
                best, bestX = sp, Xp.copy()
        except Exception:
            pass
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX
