import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _norm_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _quench(X_init, steps, T0, T1, lr0, lr1, alpha=0.6, rng=None, noise=0.0):
    X = X_init.copy()
    n, d = X.shape
    m1 = np.zeros_like(X)
    v1 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lT0, lT1 = np.log(T0), np.log(T1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    G0 = X @ X.T
    np.fill_diagonal(G0, -2.0)
    best_m = float(G0.max())
    best_X = X.copy()
    for s in range(1, steps + 1):
        frac = (s - 1) / max(steps - 1, 1)
        T = float(np.exp(lT0 * (1 - frac) + lT1 * frac))
        lr = float(np.exp(ll0 * (1 - frac) + ll1 * frac))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        if m < best_m:
            best_m = m
            best_X = X.copy()
        D = (G - m) / T
        np.clip(D, -50, 0, out=D)
        E = np.exp(D)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if not np.isfinite(S) or S <= 0:
            S = 1.0
        Gg = (E @ X) / S
        rm = G.max(axis=1)
        Q = (G - rm[:, None]) / T
        np.clip(Q, -50, 0, out=Q)
        Er = np.exp(Q)
        np.fill_diagonal(Er, 0.0)
        rs = Er.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        W = Er / rs
        Ws = 0.5 * (W + W.T)
        Gr = Ws @ X
        Grad = alpha * Gg + (1.0 - alpha) * Gr
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        if rng is not None and noise > 0:
            N = rng.standard_normal((n, d)) * noise
            N -= X * np.sum(N * X, axis=1, keepdims=True)
            Gt = Gt + N
        if not np.all(np.isfinite(Gt)):
            break
        m1 = b1 * m1 + (1 - b1) * Gt
        v1 = b2 * v1 + (1 - b2) * (Gt * Gt)
        c1 = 1.0 - b1 ** s
        c2 = 1.0 - b2 ** s
        step = (m1 / c1) / (np.sqrt(v1 / c2) + eps)
        sn = float(np.linalg.norm(step))
        mxn = 10.0 * np.sqrt(n * d)
        if sn > mxn:
            step *= (mxn / sn)
        X = X - lr * step
        X = _norm_rows(X)
    Gf = X @ X.T
    np.fill_diagonal(Gf, -2.0)
    mf = float(Gf.max())
    if mf < best_m:
        best_m = mf
        best_X = X.copy()
    return best_X

def _lp_polish(X, seconds, deadline=None):
    X = _norm_rows(X.copy())
    n, d = X.shape
    try:
        best = _max_cos(X)
    except Exception:
        return X
    radius = 2e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if deadline is not None and time.time() > deadline:
            break
        if radius < 1e-14:
            break
        G = X @ X.T
        c = G[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        cs = c[sel]
        p = len(I)
        if p == 0:
            break
        if p > 1500:
            k = np.argpartition(-cs, 1500)[:1500]
            I = I[k]; J = J[k]; cs = cs[k]
            p = 1500
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
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
        y = _norm_rows(y)
        if not np.all(np.isfinite(y)):
            radius *= 0.5
            continue
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def _refine_single(s0, R, steps=250, T=0.003, lr=0.12, mu=0.85):
    s = np.asarray(s0, dtype=float).copy()
    nrm = float(np.linalg.norm(s))
    if not np.isfinite(nrm) or nrm == 0:
        return s0
    s = s / nrm
    v = np.zeros_like(s)
    for _ in range(steps):
        g = R @ s
        m = float(g.max())
        e = np.exp(np.clip((g - m) / T, -50, 0))
        den = float(e.sum()) + 1e-30
        Grad = (R.T @ e) / den
        Gt = Grad - float(Grad @ s) * s
        v = mu * v + Gt
        s = s - lr * v
        nn = float(np.linalg.norm(s))
        if not np.isfinite(nn) or nn == 0:
            break
        s = s / nn
    return s

def _find_hole(R, rng, n_sample=6000):
    d = R.shape[1]
    Z = rng.standard_normal((n_sample, d))
    Z = _norm_rows(Z)
    S = Z @ R.T
    scores = S.max(axis=1)
    idx = np.argpartition(scores, 3)[:3]
    best_s = None
    best_v = 1e100
    for k in idx:
        s1 = _refine_single(Z[k], R, steps=200, T=0.005, lr=0.12, mu=0.8)
        s1 = _refine_single(s1, R, steps=200, T=0.0015, lr=0.07, mu=0.8)
        v = float((R @ s1).max())
        if v < best_v:
            best_v = v
            best_s = s1
    return best_s / (np.linalg.norm(best_s) + 1e-12)

def _lbfgs_tighten(X0, betas, maxiters, deadline=None):
    n, d = X0.shape
    iu = np.triu_indices(n, 1)
    Y = _norm_rows(X0.copy())
    cur_best = _max_cos(Y)
    cur_X = Y.copy()
    for beta, mit in zip(betas, maxiters):
        if deadline is not None and time.time() > deadline:
            break
        Yflat0 = Y.ravel().copy()
        def fun(Yflat):
            Ym = Yflat.reshape(n, d)
            rn = np.linalg.norm(Ym, axis=1, keepdims=True)
            rn = np.maximum(rn, 1e-12)
            X = Ym / rn
            G = X @ X.T
            gv = G[iu]
            gm = float(gv.max())
            ee = np.exp(np.clip(beta * (gv - gm), -50, 0))
            S = float(ee.sum())
            F = gm + np.log(max(S, 1e-300)) / beta
            w = ee / max(S, 1e-300)
            W = np.zeros((n, n))
            W[iu] = w
            W = W + W.T
            GradX = W @ X
            dot = np.sum(GradX * X, axis=1, keepdims=True)
            GradX_t = GradX - dot * X
            GradY = GradX_t / rn
            return F, GradY.ravel()
        try:
            res = minimize(fun, Yflat0, jac=True, method="L-BFGS-B",
                           options={"maxiter": mit, "ftol": 1e-14, "gtol": 1e-10, "maxfun": mit * 4})
            Ym = res.x.reshape(n, d)
            Y = _norm_rows(Ym)
            v = _max_cos(Y)
            if v < cur_best:
                cur_best = v
                cur_X = Y.copy()
            else:
                Y = cur_X.copy()
        except Exception:
            break
        if deadline is not None and time.time() > deadline:
            break
    return cur_X

def _single_point_try(X, cur_max, rng, budget=6):
    n, d = X.shape
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    pmx = G.max(axis=1)
    order = np.argsort(-pmx)
    for ii in range(min(n, budget)):
        i = int(order[ii])
        mask = np.ones(n, dtype=bool); mask[i] = False
        R = X[mask]
        Go = R @ R.T
        np.fill_diagonal(Go, -2.0)
        others_max = float(Go.max()) if Go.size else -2.0
        if others_max >= cur_max - 1e-12:
            continue
        try:
            s = _find_hole(R, rng, n_sample=5000)
        except Exception:
            continue
        Xn = X.copy()
        Xn[i] = s
        nm = _max_cos(Xn)
        if nm < cur_max - 1e-12:
            X = Xn
            cur_max = nm
    return X, cur_max

def solve():
    t0 = time.time()
    HARD_END = t0 + 174.0
    FINAL_POLISH = 24.0
    HOP_END = HARD_END - FINAL_POLISH - 1.0
    rng = np.random.default_rng(20261107)
    best_X = None
    try:
        best_X = np.load("best.npy").astype(float)
    except Exception:
        pass
    if best_X is None or best_X.shape != (71, 12) or not np.all(np.isfinite(best_X)):
        try:
            best_X = np.load("record.npy").astype(float)
        except Exception:
            best_X = rng.standard_normal((71, 12))
    best_X = np.asarray(best_X, dtype=float)
    best_X = _norm_rows(best_X)
    best_score = _max_cos(best_X)
    n, d = 71, 12
    it = 0
    sigmas = [0.015, 0.05, 0.10, 0.22, 0.33, 0.07, 0.13]
    while time.time() < HOP_END:
        it += 1
        if time.time() > HOP_END - 2.0:
            break
        Gc = best_X @ best_X.T
        np.fill_diagonal(Gc, -2.0)
        mxc = Gc.max(axis=1)
        cntc = (Gc > best_score - 2e-3).sum(axis=1)
        order_worst = np.argsort(-mxc - 0.0003 * cntc)
        r = rng.random()
        X0 = best_X.copy()
        if r < 0.34:
            sigma = sigmas[it % len(sigmas)]
            noise = rng.standard_normal((n, d))
            w = np.ones(n)
            w[order_worst[:10]] *= 2.4
            w[order_worst[10:20]] *= 1.4
            noise *= (sigma * w[:, None])
            noise -= X0 * np.sum(noise * X0, axis=1, keepdims=True)
            X0 = _norm_rows(X0 + noise)
            steps = 900 if sigma < 0.1 else 1200
            T0 = 0.012 if sigma < 0.1 else 0.018
            Xq = _quench(X0, steps, T0, 0.0003, 0.028, 0.0025, alpha=0.6,
                         rng=rng if sigma > 0.06 else None, noise=0.0006 if sigma > 0.1 else 0.0)
        elif r < 0.62:
            k = int(rng.integers(2, 6))
            kill = order_worst[:k].copy()
            if rng.random() < 0.25:
                kill = rng.choice(n, size=k, replace=False)
            for idx in kill:
                mask = np.ones(n, dtype=bool); mask[idx] = False
                R = X0[mask]
                try:
                    X0[idx] = _find_hole(R, rng, n_sample=5000)
                except Exception:
                    X0[idx] = rng.standard_normal(d)
                    X0[idx] /= np.linalg.norm(X0[idx])
            X0 = _norm_rows(X0)
            Xq = _quench(X0, 1100, 0.016, 0.0003, 0.03, 0.002, alpha=0.55, rng=rng, noise=0.0008)
        elif r < 0.82:
            u = rng.standard_normal(d); u /= np.linalg.norm(u)
            dots = X0 @ u
            m = int(rng.integers(8, 17))
            cap = np.argpartition(-dots, m)[:m]
            t = rng.standard_normal(d)
            t -= float(t @ u) * u
            tn = float(np.linalg.norm(t))
            if tn > 0:
                t /= tn
            shift = float(rng.uniform(0.05, 0.28)) * t
            jit = rng.standard_normal((m, d)) * 0.06
            X0[cap] = X0[cap] + shift[None, :] + jit
            X0 = _norm_rows(X0)
            Xq = _quench(X0, 1200, 0.019, 0.00035, 0.033, 0.0028, alpha=0.6, rng=rng, noise=0.001)
        else:
            m = int(rng.integers(9, 21))
            idx = rng.choice(n, size=m, replace=False)
            X0[idx] += rng.standard_normal((m, d)) * 0.35
            X0 = _norm_rows(X0)
            Xq = _quench(X0, 1250, 0.020, 0.0004, 0.035, 0.003, alpha=0.65, rng=rng, noise=0.001)
        try:
            qscore = _max_cos(Xq)
        except Exception:
            continue
        if qscore < best_score + 0.005 and time.time() < HOP_END - 3.0:
            try:
                Xq, qscore = _single_point_try(Xq, qscore, rng, budget=6)
            except Exception:
                pass
        if qscore < best_score + 0.0006 and (it % 6 == 0) and time.time() < HOP_END - 4:
            try:
                Xq = _lp_polish(Xq, 1.0, deadline=HOP_END)
                qscore = _max_cos(Xq)
            except Exception:
                pass
        if qscore < best_score + 0.0004 and (it % 9 == 0) and time.time() < HOP_END - 8:
            try:
                Xt = _lbfgs_tighten(Xq, [300], [60], deadline=HOP_END)
                st = _max_cos(Xt)
                if st < qscore:
                    Xq, qscore = Xt, st
            except Exception:
                pass
        if qscore < best_score - 1e-12:
            best_X, best_score = Xq, qscore
            if time.time() < HOP_END - 6 and rng.random() < 0.6:
                try:
                    Xf = _quench(best_X, 450, 0.0025, 0.00015, 0.009, 0.001, alpha=0.6)
                    sf = _max_cos(Xf)
                    if sf < best_score:
                        best_X, best_score = Xf, sf
                except Exception:
                    pass
        elif qscore < best_score + 1e-12 and rng.random() < 0.08 and time.time() < HOP_END - 6:
            try:
                Xf = _quench(best_X, 500, 0.002, 0.00012, 0.008, 0.001, alpha=0.6)
                sf = _max_cos(Xf)
                if sf < best_score:
                    best_X, best_score = Xf, sf
            except Exception:
                pass
    try:
        if time.time() < HARD_END - 3:
            Xf = _quench(best_X, 700, 0.003, 0.00012, 0.010, 0.0012, alpha=0.6)
            sf = _max_cos(Xf)
            if sf < best_score:
                best_X, best_score = Xf, sf
    except Exception:
        pass
    try:
        if time.time() < HARD_END - 6:
            Xt = _lbfgs_tighten(best_X, [250, 600], [120, 120], deadline=HARD_END - 4)
            st = _max_cos(Xt)
            if st < best_score:
                best_X, best_score = Xt, st
    except Exception:
        pass
    try:
        remain = HARD_END - time.time() - 0.6
        if remain > 3:
            Xp = _lp_polish(best_X, remain, deadline=HARD_END - 0.3)
            sp = _max_cos(Xp)
            if sp < best_score:
                best_X = Xp
    except Exception:
        pass
    best_X = _norm_rows(best_X)
    return best_X
