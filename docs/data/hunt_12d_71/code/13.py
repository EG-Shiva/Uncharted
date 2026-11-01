import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _norm_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n == 0] = 1.0
    return x / n

def _lp_polish(x, seconds, deadline=None):
    x = _norm_rows(x.copy())
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = 8e-4
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if deadline is not None and time.time() > deadline:
            break
        if radius < 1e-14:
            break
        G = x @ x.T
        iu_c = G[iu]
        m = float(iu_c.max())
        thresh = m - max(20 * radius, 1e-9)
        sel = iu_c > thresh
        if np.count_nonzero(sel) == 0:
            break
        I = iu[0][sel]; J = iu[1][sel]
        csub = iu_c[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csub, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm_rows(y)
        if not np.all(np.isfinite(y)):
            radius *= 0.5
            continue
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _quench_hybrid(X_init, steps, T0, T1, lr0, lr1, alpha=0.6):
    X = X_init.copy()
    n, d = X.shape
    m1 = np.zeros_like(X)
    v1 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lT0 = np.log(T0); lT1 = np.log(T1)
    llr0 = np.log(lr0); llr1 = np.log(lr1)
    G0 = X @ X.T
    np.fill_diagonal(G0, -2.0)
    best_m = float(G0.max())
    best_X = X.copy()
    for s in range(1, steps + 1):
        frac = (s - 1) / max(steps - 1, 1)
        T = float(np.exp(lT0 * (1 - frac) + lT1 * frac))
        lr = float(np.exp(llr0 * (1 - frac) + llr1 * frac))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        if m < best_m:
            best_m = m
            best_X = X.copy()
        # global weights
        D = (G - m) / T
        np.clip(D, -50, 0, out=D)
        E = np.exp(D)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if not np.isfinite(S) or S <= 0:
            S = 1.0
        Grad_g = (E @ X) / S
        # per-row weights
        rm = G.max(axis=1)
        Q = (G - rm[:, None]) / T
        np.clip(Q, -50, 0, out=Q)
        Er = np.exp(Q)
        np.fill_diagonal(Er, 0.0)
        rs = Er.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        W = Er / rs
        Ws = 0.5 * (W + W.T)
        Grad_r = Ws @ X
        Grad = alpha * Grad_g + (1.0 - alpha) * Grad_r
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        if not np.all(np.isfinite(Gt)):
            break
        m1 = b1 * m1 + (1 - b1) * Gt
        v1 = b2 * v1 + (1 - b2) * (Gt * Gt)
        c1 = 1 - b1 ** s
        c2 = 1 - b2 ** s
        mhat = m1 / c1
        vhat = v1 / c2
        step = mhat / (np.sqrt(vhat) + eps)
        sn = float(np.linalg.norm(step))
        maxn = 10.0 * np.sqrt(n * d)
        if sn > maxn:
            step *= (maxn / sn)
        X = X - lr * step
        X = _norm_rows(X)
    Gf = X @ X.T
    np.fill_diagonal(Gf, -2.0)
    mf = float(Gf.max())
    if mf < best_m:
        best_m = mf
        best_X = X.copy()
    return best_X

def _refine_single(s0, R, steps=350, T=0.003, lr=0.12, mu=0.85):
    s = s0 / (np.linalg.norm(s0) + 1e-12)
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
        n = float(np.linalg.norm(s))
        if not np.isfinite(n) or n == 0:
            break
        s = s / n
    return s

def _find_hole(R, rng, n_sample=8000):
    n, d = R.shape
    Z = rng.standard_normal(size=(n_sample, d))
    Z = _norm_rows(Z)
    S = Z @ R.T
    scores = S.max(axis=1)
    k = min(4, n_sample)
    idx = np.argpartition(scores, k)[:k]
    best_s = None
    best_v = 1e100
    for kk in idx:
        s0 = Z[kk]
        s1 = _refine_single(s0, R, steps=250, T=0.005, lr=0.15)
        s1 = _refine_single(s1, R, steps=250, T=0.0012, lr=0.07)
        v = float((R @ s1).max())
        if v < best_v:
            best_v = v
            best_s = s1
    best_s = best_s / (np.linalg.norm(best_s) + 1e-12)
    return best_s

def _loads(X, band=0.005):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    row_max = G.max(axis=1)
    cnt = np.sum(G > (m - band), axis=1)
    return row_max, cnt, m

def solve():
    t0 = time.time()
    HARD_END = t0 + 174.0
    FINAL_POLISH = 30.0
    HOP_END = HARD_END - FINAL_POLISH - 1.0
    np.random.seed(0)
    rng = np.random.default_rng(7)
    n, d = 71, 12
    best_X = None
    try:
        best_X = np.load("best.npy").astype(float)
    except Exception:
        try:
            best_X = np.load("record.npy").astype(float)
        except Exception:
            pass
    if best_X is None or best_X.shape != (71, 12) or not np.all(np.isfinite(best_X)):
        r = rng.standard_normal((71, 12))
        best_X = _norm_rows(r)
    best_X = _norm_rows(best_X)
    best_score = _max_cos(best_X)
    # initial squeeze of incumbent with hybrid quench (cheap)
    try:
        Xq = _quench_hybrid(best_X, steps=2500, T0=0.0012, T1=0.00012, lr0=0.004, lr1=0.0004)
        sq = _max_cos(Xq)
        if sq < best_score:
            best_X, best_score = _norm_rows(Xq), sq
    except Exception:
        pass
    # short LP to tighten
    try:
        trem0 = HOP_END - time.time()
        if trem0 > 5:
            Xp = _lp_polish(best_X, seconds=2.0, deadline=HOP_END)
            sp = _max_cos(Xp)
            if sp < best_score:
                best_X, best_score = _norm_rows(Xp), sp
    except Exception:
        pass
    try:
        _, cnt0, _ = _loads(best_X)
    except Exception:
        cnt0 = np.ones(n)
    it = 0
    while time.time() < HOP_END:
        it += 1
        tleft = HOP_END - time.time()
        if tleft < 0.8:
            break
        # choose move
        u = rng.random()
        if it % 41 == 0:
            X0 = _norm_rows(rng.standard_normal((n, d)))
            sigma_rest = 0.0
            use_hole = False
            k_rep = 0
            do_shake = False
        elif u < 0.42:
            sigma = float(10 ** rng.uniform(-2.7, -0.7))
            X0 = best_X + rng.standard_normal((n, d)) * sigma
            X0 = _norm_rows(X0)
            do_shake = False
            use_hole = False
            k_rep = 0
        elif u < 0.68:
            k_rep = int(rng.integers(1, 4))
            try:
                row_max, cnt, _ = _loads(best_X, band=0.005)
                order = np.argsort(-row_max - 0.002 * cnt + rng.random(n) * 0.001)
                idx = order[:k_rep]
            except Exception:
                idx = rng.choice(n, size=k_rep, replace=False)
            X0 = best_X.copy()
            rnd = _norm_rows(rng.standard_normal((k_rep, d)))
            X0[idx] = rnd
            sigma = float(10 ** rng.uniform(-2.5, -1.0))
            mask = np.ones(n, dtype=bool)
            mask[idx] = False
            X0[mask] += rng.standard_normal((n - k_rep, d)) * sigma * 0.5
            X0 = _norm_rows(X0)
            do_shake = False
            use_hole = False
            k_rep = 0  # already applied
        else:
            # hole reinsertion of 1-2 worst
            k_rep = 1 if rng.random() < 0.7 else 2
            try:
                row_max, cnt, _ = _loads(best_X, band=0.005)
                order = np.argsort(-row_max - 0.002 * cnt + rng.random(n) * 0.0005)
                idx = order[:k_rep]
            except Exception:
                idx = rng.choice(n, size=k_rep, replace=False)
            keep = np.ones(n, dtype=bool)
            keep[idx] = False
            R = best_X[keep]
            X0 = best_X.copy()
            for j in idx:
                try:
                    h = _find_hole(R, rng, n_sample=6000)
                except Exception:
                    h = _norm_rows(rng.standard_normal((1, d)))[0]
                X0[j] = h
                R = np.concatenate([R, h[None, :]], axis=0)
            # small noise to rest
            sigma = float(10 ** rng.uniform(-2.8, -1.2))
            X0[keep] += rng.standard_normal((n - k_rep, d)) * sigma
            X0 = _norm_rows(X0)
            do_shake = False
            use_hole = False
            k_rep = 0
        # quench hyperparams
        T0 = float(10 ** rng.uniform(-2.5, -1.7))
        T1 = float(10 ** rng.uniform(-4.2, -3.2))
        lr0 = float(10 ** rng.uniform(-2.8, -2.1))
        lr1 = lr0 * float(10 ** rng.uniform(-1.2, -0.6))
        if tleft < 6:
            steps = 1200
        elif tleft < 25:
            steps = 1800
        else:
            steps = 2100
            if rng.random() < 0.18:
                steps = 3300
        try:
            X1 = _quench_hybrid(X0, steps, T0, T1, lr0, lr1, alpha=0.6)
            s1 = _max_cos(X1)
        except Exception:
            continue
        if not np.isfinite(s1):
            continue
        if s1 < best_score + 0.0012:
            secs = 1.4
            if tleft < secs + 0.6:
                secs = max(0.5, tleft - 0.6)
            try:
                X2 = _lp_polish(X1, seconds=secs, deadline=HOP_END)
                s2 = _max_cos(X2)
            except Exception:
                continue
            if np.isfinite(s2) and s2 < best_score - 1e-12:
                best_X = _norm_rows(X2)
                best_score = float(s2)
                try:
                    _, cnt0, _ = _loads(best_X)
                except Exception:
                    pass
        if it % 25 == 0:
            try:
                _, cnt0, _ = _loads(best_X)
            except Exception:
                pass
    # final intensive squeeze
    try:
        trem = HARD_END - time.time() - 0.6
        if trem > 2.0:
            # low-T hybrid quench
            qs = 5000 if trem > 15 else 2500
            try:
                Xf = _quench_hybrid(best_X, steps=qs, T0=0.0006, T1=0.00008, lr0=0.0025, lr1=0.00025, alpha=0.5)
                sf = _max_cos(Xf)
                if sf < best_score:
                    best_X, best_score = _norm_rows(Xf), sf
            except Exception:
                pass
            trem2 = HARD_END - time.time() - 0.5
            if trem2 > 1.0:
                best_X = _lp_polish(best_X, seconds=min(trem2, 32.0), deadline=HARD_END - 0.4)
    except Exception:
        pass
    best_X = _norm_rows(best_X)
    for i in range(n):
        if not np.all(np.isfinite(best_X[i])) or float(np.linalg.norm(best_X[i])) < 1e-12:
            v = rng.standard_normal(d)
            v = v / (np.linalg.norm(v) + 1e-12)
            best_X[i] = v
    best_X = _norm_rows(best_X)
    return best_X
