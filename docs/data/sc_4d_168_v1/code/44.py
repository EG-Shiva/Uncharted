import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n

def _contact_var(X, tol=5e-4):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    deg = np.sum(G > m - tol, axis=1).astype(float)
    return float(np.var(deg)), float(np.mean(deg)), m, deg

def _smooth_full(X, steps, b0, b1, lr0, lr1, mu=0.9, deadline=None):
    X = X.copy()
    V = np.zeros_like(X)
    lb0, lb1 = np.log(b0), np.log(b1)
    for s in range(steps):
        if deadline is not None and s % 50 == 0 and time.time() > deadline:
            break
        f = s / max(1, steps - 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = lr0 * (1 - f) + lr1 * f
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        cmax = float(np.max(G))
        E = np.exp(np.clip(beta * (G - cmax), -25.0, 0.0))
        np.fill_diagonal(E, 0.0)
        rs = np.sum(E, axis=1, keepdims=True) + 1e-12
        Grad = (E @ X) / rs
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        sc = np.mean(np.linalg.norm(Gt, axis=1)) + 1e-12
        Gt = Gt / sc
        V = mu * V + (1.0 - mu) * Gt
        X = X - lr * V
        X = _norm(X)
    return X

def _lp(x, deadline, radius0=1e-3):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = radius0
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        C = (x @ x.T)[iu]
        m = float(C.max())
        sel = C > m - max(20.0 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        cs = C[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def _sigmoid(a):
    # stable
    out = np.empty_like(a)
    pos = a >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-a[pos]))
    e = np.exp(a[~pos])
    out[~pos] = e / (1.0 + e)
    return out

def _hopf_points(uF, dF, s0F, deltaFN):
    F, N = deltaFN.shape
    step = 4.0 * np.pi / float(N)
    j = np.arange(N)[None, :]  # 1xN
    s = s0F[:, None] + j * step + deltaFN  # FxN
    d_rep = dF[:, None] + np.zeros_like(s)
    u_rep = uF[:, None] + np.zeros_like(s)
    p = (s + d_rep) * 0.5
    q = (s - d_rep) * 0.5
    cu = np.cos(u_rep); su = np.sin(u_rep)
    cp = np.cos(p); sp = np.sin(p)
    cq = np.cos(q); sq = np.sin(q)
    X = np.empty((F * N, 4))
    X[:, 0] = (cu * cp).ravel()
    X[:, 1] = (cu * sp).ravel()
    X[:, 2] = (su * cq).ravel()
    X[:, 3] = (su * sq).ravel()
    aux = (cu, su, cp, sp, cq, sq, p, q, s)
    return X, aux

def _hopf_opt(F, N, rng, steps=1200, b0=60.0, b1=500.0, lr0=0.03, lr1=0.005):
    ru = rng.normal(0, 0.6, size=F)
    d = rng.uniform(0, 2 * np.pi, size=F)
    s0 = rng.uniform(0, 4 * np.pi, size=F)
    delta = np.zeros((F, N))
    m_ru = np.zeros(F); v_ru = np.zeros(F)
    m_d = np.zeros(F); v_d = np.zeros(F)
    m_s = np.zeros(F); v_s = np.zeros(F)
    m_dl = np.zeros((F, N)); v_dl = np.zeros((F, N))
    b1a, b2a, eps = 0.9, 0.999, 1e-8
    lb0, lb1 = np.log(b0), np.log(b1)
    step_const = 4.0 * np.pi / float(N)
    JJ = np.arange(N)[None, :]
    for t in range(1, steps + 1):
        frac = (t - 1) / max(1, steps - 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * frac))
        lr = lr0 * (1 - frac) + lr1 * frac
        sig = _sigmoid(ru)
        uF = (np.pi * 0.5) * sig
        dsig = sig * (1 - sig) * (np.pi * 0.5)  # du/dru
        s = s0[:, None] + JJ * step_const + delta
        d_rep = d[:, None] + np.zeros_like(s)
        u_rep = uF[:, None] + np.zeros_like(s)
        p = (s + d_rep) * 0.5
        q = (s - d_rep) * 0.5
        cu = np.cos(u_rep); su = np.sin(u_rep)
        cp = np.cos(p); sp = np.sin(p)
        cq = np.cos(q); sq = np.sin(q)
        X = np.empty((F * N, 4))
        X[:, 0] = (cu * cp).ravel()
        X[:, 1] = (cu * sp).ravel()
        X[:, 2] = (su * cq).ravel()
        X[:, 3] = (su * sq).ravel()
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        cmax = float(np.max(G))
        E = np.exp(np.clip(beta * (G - cmax), -25.0, 0.0))
        np.fill_diagonal(E, 0.0)
        Esum = float(np.sum(E)) + 1e-30
        P = E / (Esum * 0.5)
        Grad = P @ X  # (M,4)
        Gr = Grad.reshape(F, N, 4)
        cuF = cu; suF = su
        # per-point dots
        g1 = Gr[:, :, 0]; g2 = Gr[:, :, 1]; g3 = Gr[:, :, 2]; g4 = Gr[:, :, 3]
        # du dot
        adu = g1 * (-suF * cp) + g2 * (-suF * sp) + g3 * (cuF * cq) + g4 * (cuF * sq)
        adp = g1 * (-cuF * sp) + g2 * (cuF * cp)
        adq = g3 * (-suF * sq) + g4 * (suF * cq)
        add = 0.5 * adp - 0.5 * adq
        ads = 0.5 * adp + 0.5 * adq
        gru_pt = adu  # dL/du per point
        gru = np.sum(gru_pt, axis=1) * dsig  # chain to ru
        gd = np.sum(add, axis=1)
        gs = np.sum(ads, axis=1)
        gdl = ads  # per delta
        # zero-mean projection for delta per fiber (keep fiber center in s0)
        gdl = gdl - np.mean(gdl, axis=1, keepdims=True)
        # Adam updates
        m_ru = b1a * m_ru + (1 - b1a) * gru
        v_ru = b2a * v_ru + (1 - b2a) * gru * gru
        m_d = b1a * m_d + (1 - b1a) * gd
        v_d = b2a * v_d + (1 - b2a) * gd * gd
        m_s = b1a * m_s + (1 - b1a) * gs
        v_s = b2a * v_s + (1 - b2a) * gs * gs
        m_dl = b1a * m_dl + (1 - b1a) * gdl
        v_dl = b2a * v_dl + (1 - b2a) * gdl * gdl
        c1 = 1 - b1a ** t; c2 = 1 - b2a ** t
        ru -= lr * (m_ru / c1) / (np.sqrt(v_ru / c2) + eps)
        d -= lr * (m_d / c1) / (np.sqrt(v_d / c2) + eps)
        s0 -= lr * (m_s / c1) / (np.sqrt(v_s / c2) + eps)
        delta -= (lr * 0.7) * (m_dl / c1) / (np.sqrt(v_dl / c2) + eps)
        # keep delta bounded and zero-mean
        delta -= np.mean(delta, axis=1, keepdims=True)
        np.clip(delta, -0.6, 0.6, out=delta)
    sig = _sigmoid(ru)
    uF = (np.pi * 0.5) * sig
    X, _ = _hopf_points(uF, d, s0, delta)
    X = _norm(X)
    return X, float(_max_cos(X))

def solve():
    t0 = time.time()
    LIMIT = 172.0
    SEARCH_END = t0 + 130.0
    rng = np.random.default_rng(12345)
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (168, 4):
                cands.append(_norm(a))
        except Exception:
            pass
    if not cands:
        cands = [_norm(rng.normal(size=(168, 4)))]
    X_inc = None; best_inc = 1e9
    for c in cands:
        v = _max_cos(c)
        if v < best_inc:
            best_inc = v; X_inc = c.copy()
    var_inc, mean_inc, _, _ = _contact_var(X_inc)
    # Hopf search: N<=8 only (N>=12 impossible: cos(2pi/N)>record)
    configs = [(28, 6), (21, 8), (24, 7), (42, 4)]
    sym_results = []
    # budget: ~1200 steps x 2 seeds per config, cheaper for 42x4
    seed_base = [101, 202]
    for (F, N) in configs:
        for sd in seed_base:
            if time.time() > SEARCH_END - 25.0:
                break
            r2 = np.random.default_rng(1000 + F * 37 + N * 11 + sd)
            steps = 1100 if F <= 28 else 800
            try:
                Xs, vs = _hopf_opt(F, N, r2, steps=steps)
            except Exception:
                continue
            sym_results.append((vs, F, N, Xs))
            if time.time() > SEARCH_END - 25.0:
                break
        if time.time() > SEARCH_END - 25.0:
            break
    sym_results.sort(key=lambda t: t[0])
    top = sym_results[:3]
    X_best = X_inc.copy(); best = best_inc; var_best = var_inc
    # break symmetry + anneal for top candidates
    sigmas = [0.003, 0.008, 0.005]
    for idx, (vs, F, N, Xs) in enumerate(top):
        if time.time() > SEARCH_END:
            break
        sigma = sigmas[idx % len(sigmas)]
        r3 = np.random.default_rng(777 + idx * 13 + F)
        Xb = Xs + r3.normal(0, sigma, size=Xs.shape)
        Xb = _norm(Xb)
        # anneal release
        try:
            Xa = _smooth_full(Xb, 700, 40.0, 500.0, 0.02, 0.001, mu=0.88, deadline=SEARCH_END)
        except Exception:
            continue
        va = _max_cos(Xa)
        vara, _, _, _ = _contact_var(Xa)
        # accept only if variance lower than incumbent (goal: regular graph)
        if vara < var_inc - 1e-9:
            # opportunistic short LP if promising
            if va < best + 0.004 and time.time() < SEARCH_END - 4.0:
                try:
                    Xp, vp = _lp(Xa, min(time.time() + 4.0, SEARCH_END))
                except Exception:
                    Xp, vp = Xa, va
                vara2, _, _, _ = _contact_var(Xp)
                # keep variance condition on polished too (allow tiny slack)
                if vara2 <= var_inc + 1e-9 and vp < best:
                    best = vp; X_best = Xp.copy(); var_best = vara2
                elif vp < best and vara < var_inc:
                    # still track best max even if polish raised var slightly
                    pass
            else:
                if va < best:
                    best = va; X_best = Xa.copy(); var_best = vara
        else:
            # even if variance not lower, keep track but do not accept as record basin change;
            # only allow if it strictly beats incumbent max (fallback safety)
            if va < best - 2e-6:
                # strong improvement overrides variance filter
                if time.time() < SEARCH_END - 4.0:
                    try:
                        Xp, vp = _lp(Xa, min(time.time() + 4.0, SEARCH_END))
                    except Exception:
                        Xp, vp = Xa, va
                    if vp < best:
                        best = vp; X_best = Xp.copy(); var_best = vara
        if time.time() > SEARCH_END:
            break
    # if no accepted improvement, keep incumbent
    if best >= best_inc - 1e-12:
        X_best = X_inc.copy(); best = best_inc
    # final polish with remaining time
    remaining = (t0 + LIMIT - 2.0) - time.time()
    if remaining > 5.0:
        try:
            Xp, vp = _lp(X_best, time.time() + min(remaining - 1.0, 25.0))
            if vp < best + 1e-12:
                # do not regress on variance goal unless max improves
                if vp < best or True:
                    X_best, best = Xp, vp
        except Exception:
            pass
        if time.time() < t0 + LIMIT - 6.0:
            try:
                X2 = _smooth_full(X_best, 250, 60.0, 400.0, 0.008, 0.0008, mu=0.9, deadline=t0 + LIMIT - 4.0)
                v2 = _max_cos(X2)
                if v2 < best:
                    X_best, best = X2, v2
            except Exception:
                pass
    X_best = _norm(np.where(np.isfinite(X_best), X_best, 0.0))
    # safety: if somehow worse than incumbent, return incumbent
    try:
        if _max_cos(X_best) > best_inc + 1e-9:
            X_best = X_inc.copy()
    except Exception:
        pass
    return X_best
