import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _per_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return np.max(G, axis=1)

def _lse_fg(n, d, IU, beta):
    def fg(yf):
        Y = yf.reshape(n, d)
        nr = np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
        X = Y / nr
        g = (X @ X.T)[IU]
        m = float(np.max(g))
        e = np.exp(beta * (g - m))
        S = float(np.sum(e)) + 1e-300
        F = m + np.log(S) / beta
        w = e / S
        P = np.zeros((n, n))
        P[IU] = w
        P = P + P.T
        GX = P @ X
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GY = (GX - dot * X) / nr
        return F, GY.ravel()
    return fg

def _riesz_fg(n, d, IU, s):
    def fg(yf):
        Y = yf.reshape(n, d)
        nr = np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
        X = Y / nr
        G = X @ X.T
        g = G[IU]
        r2 = np.maximum(2.0 - 2.0 * g, 1e-12)
        # energy
        E = np.sum(np.exp((-s / 2.0) * np.log(r2)))
        w = s * np.exp((-s / 2.0 - 1.0) * np.log(r2))  # s * r2^{-s/2-1}
        P = np.zeros((n, n))
        P[IU] = w
        P = P + P.T
        GX = P @ X
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GY = (GX - dot * X) / nr
        # scale to avoid huge values
        sc = 1.0 / max(float(np.sum(w)), 1e-300)
        return E * sc, (GY * sc).ravel()
    return fg

def _run_opt(X0, fg_maker, betas, maxiter):
    n, d = X0.shape
    IU = np.triu_indices(n, 1)
    y = (X0 / np.linalg.norm(X0, axis=1, keepdims=True)).ravel().copy()
    for b in betas:
        fg = fg_maker(n, d, IU, b)
        try:
            r = minimize(fg, y, jac=True, method='L-BFGS-B',
                         options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-7, 'maxfun': maxiter * 2})
            y = r.x.copy()
        except Exception:
            break
    Y = y.reshape(n, d)
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return Y

def _run_lse(X0, betas=(200.0, 800.0), maxiter=150):
    return _run_opt(X0, _lse_fg, betas, maxiter)

def _run_riesz(X0, s_list=(2.0,), maxiter=200):
    return _run_opt(X0, _riesz_fg, s_list, maxiter)

def _anti_fg(H, d, IU68, beta):
    # optimize Y (H,d), X=[Yn;-Yn]
    def fg(yf):
        Y = yf.reshape(H, d)
        nr = np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
        Yn = Y / nr
        X = np.empty((2 * H, d))
        X[:H] = Yn
        X[H:] = -Yn
        g = (X @ X.T)[IU68]
        m = float(np.max(g))
        e = np.exp(beta * (g - m))
        S = float(np.sum(e)) + 1e-300
        F = m + np.log(S) / beta
        w = e / S
        P = np.zeros((2 * H, 2 * H))
        P[IU68] = w
        P = P + P.T
        GX = P @ X
        dYn = GX[:H] - GX[H:]
        dot = np.sum(dYn * Yn, axis=1, keepdims=True)
        GY = (dYn - dot * Yn) / nr
        return F, GY.ravel()
    return fg

def _run_anti(Y0, betas=(150.0, 500.0, 1500.0), maxiter=200):
    H, d = Y0.shape
    IU68 = np.triu_indices(2 * H, 1)
    y = (Y0 / np.linalg.norm(Y0, axis=1, keepdims=True)).ravel().copy()
    for b in betas:
        fg = _anti_fg(H, d, IU68, b)
        try:
            r = minimize(fg, y, jac=True, method='L-BFGS-B',
                         options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-7, 'maxfun': maxiter * 2})
            y = r.x.copy()
        except Exception:
            break
    Y = y.reshape(H, d)
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    X = np.empty((2 * H, d))
    X[:H] = Y
    X[H:] = -Y
    return X

def _lp_polish(x, deadline, r0=6e-4, cap_p=2000):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        if time.time() > deadline - 0.4:
            break
        G = x @ x.T
        c = G[iu]
        m = float(np.max(c))
        sel = c > m - max(20.0 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        csel = c[sel]
        if p > cap_p:
            od = np.argsort(-csel)[:cap_p]
            I, J = I[od], J[od]
            csel = csel[od]
            p = cap_p
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        step = res.x[:-1].reshape(n, d)
        y = x + step
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best - 1e-15:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            ok = False
            for sh in (0.3, 0.1):
                yy = x + step * sh
                yy /= np.linalg.norm(yy, axis=1, keepdims=True)
                vv = _max_cos(yy)
                if vv < best:
                    x, best = yy, vv
                    ok = True
                    break
            if not ok:
                radius *= 0.5
    return x

def _best_hole(Xr, rng, ncand=20000):
    d = Xr.shape[1]
    cand = rng.standard_normal((ncand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    s = (cand @ Xr.T).max(axis=1)
    return cand[int(np.argmin(s))]

def _rand_rot(d, rng, strength=1.0):
    A = rng.standard_normal((d, d)) * strength
    A = A - A.T
    # exp via Cayley or QR: use QR of random
    Q, R = np.linalg.qr(rng.standard_normal((d, d)))
    # make Q close to identity proportional to strength: interpolate
    # simple: Qs = (1-a)*I + a*Q projected? Instead use: mix A exp approx (I+A/2)(I-A/2)^{-1}
    try:
        Q2 = np.linalg.solve(np.eye(d) - 0.5 * A, np.eye(d) + 0.5 * A)
        return Q2
    except Exception:
        return Q

def solve():
    t0 = time.time()
    T_END = t0 + 170.0
    rng = np.random.default_rng(7)
    n, d = 68, 12
    bestX = None
    best = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=np.float64)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    s = _max_cos(a)
                    if s < best:
                        best = s
                        bestX = a.copy()
        except Exception:
            pass
    if bestX is None:
        bestX = rng.standard_normal((n, d))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        best = _max_cos(bestX)
    # quick polish incumbent
    try:
        Xp = _run_lse(bestX, betas=(300.0, 900.0), maxiter=120)
        sp = _max_cos(Xp)
        if sp < best:
            best, bestX = sp, Xp
        Xp2 = _lp_polish(bestX, min(T_END - 150, time.time() + 8.0), r0=5e-4)
        sp2 = _max_cos(Xp2)
        if sp2 < best:
            best, bestX = sp2, Xp2
    except Exception:
        pass

    # 1) antipodal projective attempts
    if time.time() < T_END - 140:
        for att in range(2):
            if time.time() > T_END - 135:
                break
            try:
                Y0 = rng.standard_normal((34, d))
                Y0 /= np.linalg.norm(Y0, axis=1, keepdims=True)
                Xa = _run_anti(Y0, betas=(150.0, 600.0, 2000.0), maxiter=150)
                Xa = _run_lse(Xa, betas=(400.0,), maxiter=60)
                dl = time.time() + 2.5
                if dl > T_END - 120:
                    dl = T_END - 120
                Xa = _lp_polish(Xa, dl, r0=4e-4)
                sa = _max_cos(Xa)
                if sa < best:
                    best, bestX = sa, Xa.copy()
            except Exception:
                continue

    # 2) Riesz diverse basins
    for att in range(4):
        if time.time() > T_END - 100:
            break
        try:
            X0 = rng.standard_normal((n, d))
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
            Xr = _run_riesz(X0, s_list=(2.0, 4.0), maxiter=150)
            Xr = _run_lse(Xr, betas=(250.0, 800.0), maxiter=120)
            dl = time.time() + 2.0
            Xr = _lp_polish(Xr, dl, r0=4e-4)
            sr = _max_cos(Xr)
            if sr < best:
                best, bestX = sr, Xr.copy()
            # also try heavy-tailed: large noise on incumbent then riesz
            if time.time() > T_END - 100:
                break
        except Exception:
            continue

    # 3) partition-rotation crossover on incumbent
    for att in range(30):
        if time.time() > T_END - 55:
            break
        try:
            perm = rng.permutation(n)
            k = 34
            Aidx = perm[:k]; Bidx = perm[k:]
            Q = _rand_rot(d, rng, strength=0.6 + 0.4 * rng.random())
            Xc = bestX.copy()
            Xc[Bidx] = (Xc[Bidx] @ Q.T)
            # add noise
            sig = 0.02 + 0.08 * rng.random()
            Xc = Xc + sig * rng.standard_normal((n, d))
            Xc /= np.linalg.norm(Xc, axis=1, keepdims=True)
            Xc = _run_lse(Xc, betas=(300.0, 900.0), maxiter=100)
            sa = _max_cos(Xc)
            # only LP if promising (within 0.02 of best to save time)
            if sa < best + 0.015:
                dl = time.time() + 1.8
                Xc = _lp_polish(Xc, dl, r0=4e-4)
                sa = _max_cos(Xc)
            if sa < best:
                best, bestX = sa, Xc.copy()
        except Exception:
            continue

    # 4) kill-and-fill worst points
    for att in range(12):
        if time.time() > T_END - 25:
            break
        try:
            pm = _per_max(bestX)
            # contacts count within 1e-9
            G = bestX @ bestX.T
            np.fill_diagonal(G, -2)
            m0 = float(np.max(G))
            cnt = np.sum(G > m0 - 1e-9, axis=1)
            score_worst = pm + 0.002 * (cnt / max(1, int(np.max(cnt))))
            kkill = int(rng.integers(5, 11))
            order = np.argsort(-score_worst)[:kkill]
            mask = np.ones(n, dtype=bool)
            mask[order] = False
            Xr = bestX[mask]
            # greedy reinsert
            newpts = []
            for _ in range(kkill):
                Xr_full = Xr if len(newpts) == 0 else np.vstack([Xr] + newpts)
                h = _best_hole(Xr_full, rng, ncand=15000)
                # small jitter + local single-point tweak via hole candidates
                newpts.append(h[None, :])
            Xc = np.vstack([Xr] + newpts)
            # shuffle order to avoid bias (keep shape)
            Xc = Xc + 0.01 * rng.standard_normal((n, d))
            Xc /= np.linalg.norm(Xc, axis=1, keepdims=True)
            Xc = _run_lse(Xc, betas=(300.0, 900.0), maxiter=100)
            sa = _max_cos(Xc)
            if sa < best + 0.01:
                dl = time.time() + 1.8
                Xc = _lp_polish(Xc, dl, r0=4e-4)
                sa = _max_cos(Xc)
            if sa < best:
                best, bestX = sa, Xc.copy()
        except Exception:
            continue

    # final polish of global best with remaining time
    try:
        remain = T_END - time.time() - 0.5
        if remain > 2.0:
            # LSE final
            bestX = _run_lse(bestX, betas=(600.0, 1500.0), maxiter=100)
            best = _max_cos(bestX)
            bestX = _lp_polish(bestX, T_END - 0.3, r0=6e-4)
            best = _max_cos(bestX)
    except Exception:
        pass
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX
