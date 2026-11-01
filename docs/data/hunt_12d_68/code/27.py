import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _per_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return np.max(G, axis=1)

def _best_hole(rest, rng, ncand=12000):
    d = rest.shape[1]
    cand = rng.standard_normal((ncand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True) + 1e-300
    scores = np.max(cand @ rest.T, axis=1)
    return cand[int(np.argmin(scores))]

def _make_lse_fg(n, d, IU, beta):
    def fg(yf):
        Y = yf.reshape(n, d)
        nr = np.linalg.norm(Y, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
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

def _run_lbfgs(x0, beta, maxiter):
    n, d = x0.shape
    IU = np.triu_indices(n, 1)
    fg = _make_lse_fg(n, d, IU, beta)
    y0 = (x0 / np.linalg.norm(x0, axis=1, keepdims=True)).ravel().copy()
    try:
        r = minimize(fg, y0, jac=True, method='L-BFGS-B',
                     options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-7, 'maxfun': maxiter*2})
        Y = r.x.reshape(n, d)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True) + 1e-300
        return Y
    except Exception:
        return x0

def _gd_mixed(x, steps, b0, b1, lr0, lr1, mu, rng, topM=600, topW=0.35, noise=0.0):
    n, d = x.shape
    IU = np.triu_indices(n, 1)
    vel = np.zeros_like(x)
    lb = np.log(b1 / b0) if b0 > 0 and b1 != b0 else 0.0
    for s in range(steps):
        t = s / max(steps - 1, 1)
        beta = b0 * np.exp(lb * t) if lb != 0.0 else b0
        lr = lr0 + (lr1 - lr0) * t
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        Glse = W @ x
        proj = np.sum(Glse * x, axis=1, keepdims=True)
        Tlse = Glse - proj * x
        rlse = np.sqrt(np.mean(Tlse * Tlse)) + 1e-12
        Tlse = Tlse / rlse
        if topM > 0:
            gf = G[IU]
            kth = np.partition(gf, -topM)[-topM]
            Mask = (G >= kth)
            np.fill_diagonal(Mask, False)
            Gtop = Mask.astype(np.float64) @ x
            proj2 = np.sum(Gtop * x, axis=1, keepdims=True)
            Ttop = Gtop - proj2 * x
            rtop = np.sqrt(np.mean(Ttop * Ttop)) + 1e-12
            Ttop = Ttop / rtop
            Tg = (1.0 - topW) * Tlse + topW * Ttop
        else:
            Tg = Tlse
        rms = np.sqrt(np.mean(Tg * Tg)) + 1e-12
        step = Tg * (lr / rms)
        if mu > 0:
            vel = mu * vel + step
            step = vel
        x = x - step
        if noise > 0:
            ns = noise * (1.0 - t)
            nz = rng.standard_normal(size=x.shape)
            nz -= np.sum(nz * x, axis=1, keepdims=True) * x
            nn = np.linalg.norm(nz, axis=1, keepdims=True) + 1e-12
            x = x + ns * (nz / nn)
        x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-300
    return x

def _gd_riesz(x, steps, s0, s1, lr0, lr1, rng):
    n, d = x.shape
    ls = np.log(s1 / s0) if s0 > 0 and s1 != s0 else 0.0
    for s in range(steps):
        t = s / max(steps - 1, 1)
        sv = s0 * np.exp(ls * t) if ls != 0.0 else s0
        lr = lr0 + (lr1 - lr0) * t
        G = x @ x.T
        np.clip(G, -1.0, 0.999999, out=G)
        np.fill_diagonal(G, 0.0)
        D2 = np.maximum(2.0 - 2.0 * G, 1e-9)
        W = D2 ** (-sv / 2.0 - 1.0)
        np.fill_diagonal(W, 0.0)
        # normalize to avoid overflow
        W /= (np.mean(W) + 1e-300)
        Gr = W @ x
        proj = np.sum(Gr * x, axis=1, keepdims=True)
        Tg = Gr - proj * x
        rms = np.sqrt(np.mean(Tg * Tg)) + 1e-12
        x = x - (lr / rms) * Tg
        x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-300
    return x

def _top_mean_descent(x, steps, M, lr0):
    n, d = x.shape
    IU = np.triu_indices(n, 1)
    for s in range(steps):
        t = s / max(steps - 1, 1)
        lr = lr0 * (1.0 - 0.8 * t)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gf = G[IU]
        kth = np.partition(gf, -M)[-M]
        Mask = (G >= kth)
        np.fill_diagonal(Mask, False)
        Gr = Mask.astype(np.float64) @ x
        proj = np.sum(Gr * x, axis=1, keepdims=True)
        Tg = Gr - proj * x
        rms = np.sqrt(np.mean(Tg * Tg)) + 1e-12
        x = x - (lr / rms) * Tg
        x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-300
    return x

def _lp_polish(x, deadline, cap_p=1600):
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-300)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(np.max(c))
        sel = c > m - max(20 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        csel = c[sel]
        if p > cap_p:
            order = np.argsort(-csel)[:cap_p]
            I, J = I[order], J[order]
            csel = csel[order]
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
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True) + 1e-300
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    TLIM = 172.0
    T_END = t0 + TLIM
    rng = np.random.default_rng(7)
    n, d = 68, 12
    bestX = None
    best_score = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=np.float64)
            if a.shape == (n, d):
                a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-300)
                if np.all(np.isfinite(a)):
                    s = _max_cos(a)
                    if s < best_score:
                        best_score = s
                        bestX = a.copy()
        except Exception:
            pass
    if bestX is None:
        bestX = rng.standard_normal((n, d))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True) + 1e-300
        best_score = _max_cos(bestX)
    x_best = bestX.copy()
    # Phase A: quick diverse inits (limited time)
    try:
        # 3 pure random short runs to sample other basins
        for si in range(3):
            if time.time() > t0 + 22:
                break
            r0 = np.random.default_rng(1000 + si).standard_normal((n, d))
            r0 /= np.linalg.norm(r0, axis=1, keepdims=True) + 1e-300
            r0 = _gd_riesz(r0, 250, 4.0, 8.0, 0.08, 0.02, rng)
            r0 = _run_lbfgs(r0, 120.0, 60)
            v = _max_cos(r0)
            if v < best_score:
                best_score = v
                x_best = r0.copy()
        # 2 large-shaken best runs
        for si, sig in enumerate([0.5, 0.9]):
            if time.time() > t0 + 30:
                break
            y = x_best + sig * np.random.default_rng(2000 + si).standard_normal((n, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True) + 1e-300
            y = _gd_mixed(y, 350, 60.0, 220.0, 0.06, 0.015, 0.5, rng, topM=650, topW=0.4)
            y = _run_lbfgs(y, 250.0, 70)
            v = _max_cos(y)
            if v < best_score:
                best_score = v
                x_best = y.copy()
    except Exception:
        pass
    explore_end = T_END - 26.0
    trial = 0
    while time.time() < explore_end and trial < 300:
        trial += 1
        x = x_best.copy()
        u = rng.random()
        try:
            if u < 0.45:
                # load-guided delete + hole reinsert (topology change)
                G = x @ x.T
                np.fill_diagonal(G, -2.0)
                m = float(np.max(G))
                # near-contact load at plateau width
                thr = m - 3e-7
                loads = np.sum(G > thr, axis=1)
                order = np.argsort(-loads)
                k = int(rng.integers(3, 9))
                # pick mostly overloaded, sometimes random rattler
                del_idx = []
                for q in order[:k + 2]:
                    if len(del_idx) >= k:
                        break
                    if rng.random() < 0.8 or len(del_idx) == 0:
                        del_idx.append(int(q))
                while len(del_idx) < k:
                    del_idx.append(int(rng.integers(0, n)))
                for q in del_idx:
                    mask = np.ones(n, dtype=bool); mask[q] = False
                    hole = _best_hole(x[mask], rng, ncand=8000)
                    alpha = float(rng.uniform(0.85, 1.0))
                    v = alpha * hole + (1 - alpha) * rng.standard_normal(d)
                    v /= (np.linalg.norm(v) + 1e-12)
                    x[q] = v
                x += 0.004 * rng.standard_normal(x.shape)
                x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-300
            elif u < 0.70:
                # coherent cap kick
                uu = rng.standard_normal(d); uu /= np.linalg.norm(uu) + 1e-12
                proj = x @ uu
                thr = float(rng.uniform(0.0, 0.6))
                mask = proj > thr
                if mask.sum() < 2 or mask.sum() > n - 2:
                    x += 0.06 * rng.standard_normal(x.shape)
                    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-300
                else:
                    vv = rng.standard_normal(d); vv -= (vv @ uu) * uu
                    vv /= (np.linalg.norm(vv) + 1e-12)
                    amp = float(rng.uniform(0.15, 0.55))
                    x[mask] += amp * vv
                    x += 0.004 * rng.standard_normal(x.shape)
                    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-300
            else:
                sig = [0.05, 0.12, 0.3][int(rng.integers(0, 3))]
                x = x + sig * rng.standard_normal(x.shape)
                x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-300
            # anneal: alternate Riesz vs LSE+Top for basin diversity
            if rng.random() < 0.35:
                x = _gd_riesz(x, 300, 5.0, 12.0, 0.07, 0.015, rng)
                x = _gd_mixed(x, 300, 80.0, 300.0, 0.05, 0.012, 0.4, rng, topM=650, topW=0.3)
            else:
                b1 = float(rng.uniform(50, 120))
                b2 = float(rng.uniform(250, 500))
                x = _gd_mixed(x, 350, b1, b2, 0.06, 0.012, 0.5, rng, topM=int(rng.choice([400, 650, 800])), topW=float(rng.uniform(0.2, 0.5)), noise=0.002)
            # refine
            x = _run_lbfgs(x, float(rng.uniform(180, 350)), 80)
            if rng.random() < 0.5:
                x = _top_mean_descent(x, 120, 600, 0.02)
                x = _run_lbfgs(x, 450.0, 60)
            v = _max_cos(x)
            if v < best_score - 1e-12:
                x_best, best_score = x.copy(), v
                # squeeze big wins immediately
                if time.time() < explore_end - 5.0:
                    x_best = _lp_polish(x_best, min(time.time() + 2.0, explore_end))
                    best_score = _max_cos(x_best)
            elif v < best_score + 0.001 and (trial % 8 == 0) and (time.time() < explore_end - 4.0):
                y = _lp_polish(x, time.time() + 1.0)
                v2 = _max_cos(y)
                if v2 < best_score:
                    x_best, best_score = y.copy(), v2
        except Exception:
            continue
    # final convergence
    try:
        if T_END - time.time() > 6.0:
            x_best = _top_mean_descent(x_best, 200, 650, 0.015)
        if T_END - time.time() > 5.0:
            x_best = _run_lbfgs(x_best, 550.0, 140)
            best_score = _max_cos(x_best)
        if T_END - time.time() > 3.0:
            x_best = _lp_polish(x_best, T_END - 0.8)
    except Exception:
        pass
    x_best = np.nan_to_num(x_best)
    x_best = x_best / (np.linalg.norm(x_best, axis=1, keepdims=True) + 1e-300)
    return x_best
