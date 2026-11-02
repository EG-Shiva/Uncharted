import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm(X):
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-18)
    return X

def _pump(X0, t, steps, b0, b1, lr0, lr1, rng, noise, deadline):
    X = X0.copy()
    n, d = X.shape
    if noise > 0:
        N = rng.standard_normal((n, d))
        N -= np.sum(N * X, axis=1, keepdims=True) * X
        nn = np.linalg.norm(N, axis=1, keepdims=True) + 1e-12
        N /= nn
        X = X + noise * N
        X = _norm(X)
    m1 = np.zeros_like(X)
    m2 = np.zeros_like(X)
    b_1, b_2, eps = 0.9, 0.999, 1e-8
    lb0, lb1 = np.log(b0), np.log(b1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    iu = np.triu_indices(n, 1)
    for s in range(steps):
        if s % 50 == 0 and time.time() >= deadline:
            break
        f = s / max(1, steps - 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        D = G - t
        # active mask only near/above target
        Act = D > -0.01
        np.fill_diagonal(Act, False)
        Z = beta * D
        # stable sigmoid + softplus
        # clip Z for exp
        Zc = np.clip(Z, -30, 30)
        sig = 1.0 / (1.0 + np.exp(-Zc))
        # softplus: log1p(exp(Z))/beta, handle large Z
        sp = np.empty_like(D)
        # Z>20: sp ~= D
        m_big = Z > 20
        m_small = ~m_big
        sp[m_big] = D[m_big]
        # for rest: log1p(exp(Zc))/beta, for Zc<-20 ~0
        sp[m_small] = np.log1p(np.exp(Zc[m_small])) / beta
        sp[~Act] = 0.0
        sig[~Act] = 0.0
        Fp = 2.0 * sp * sig
        np.fill_diagonal(Fp, 0.0)
        # gradient S@X
        Gr = Fp @ X
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        # Adam
        m1 = b_1 * m1 + (1 - b_1) * Gt
        m2 = b_2 * m2 + (1 - b_2) * (Gt * Gt)
        c1 = 1.0 - b_1 ** (s + 1)
        c2 = 1.0 - b_2 ** (s + 1)
        mh = m1 / c1
        vh = m2 / c2
        step = lr * mh / (np.sqrt(vh) + eps)
        sdot = np.sum(step * X, axis=1, keepdims=True)
        step = step - sdot * X
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        over = sn > 0.05
        step = np.where(over, step * (0.05 / sn), step)
        X = X - step
        X = _norm(X)
    return X

def _lp_combined(X, seconds, deadline, lam=0.02, beta_w=250.0, maxit=8, pmax=550):
    X = _norm(X.copy())
    n, d = X.shape
    cur = _max_cos(X)
    radius = 8e-4
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    it = 0
    while it < maxit and time.time() - start < seconds and time.time() < deadline:
        it += 1
        G = X @ X.T
        cfull = G[iu0]
        m = float(cfull.max())
        window = max(20.0 * radius, 5e-5)
        order = np.argsort(-cfull)
        sel = []
        for idx in order:
            if cfull[idx] > m - window and len(sel) < pmax:
                sel.append(int(idx))
            elif cfull[idx] <= m - window:
                break
            if len(sel) >= pmax:
                break
        if len(sel) == 0:
            break
        sel = np.array(sel)
        I = iu0[0][sel]; J = iu0[1][sel]
        # stress weights: softmax around max
        w = np.exp(np.clip(beta_w * (cfull[sel] - m), -30, 0))
        # drop unfavourable tiny stress to allow topology change
        w[w < 0.02] = 0.0
        p = len(sel)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        # secondary objective: lam * sum w * lincos + t
        # coeff for dx_i = lam * sum_j w_ij x_j
        Wfull = np.zeros((n, n))
        Wfull[I, J] = w; Wfull[J, I] = w
        coeff = lam * (Wfull @ X)
        obj = np.zeros(nv); obj[:-1] = coeff.ravel(); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cfull[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                          options={"time_limit": 2.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        # accept if improves max, or max increase tiny but weighted sum drops
        if v < cur - 1e-15:
            X, cur = y, v
            radius = min(radius * 1.5, 0.02)
        elif v < cur + 5e-4:
            # check weighted sum improvement
            Gw = y @ y.T
            wsum_new = float((w * Gw[I, J]).sum())
            wsum_old = float((w * G[I, J]).sum())
            if wsum_new < wsum_old:
                X, cur = y, v
                radius = min(radius * 1.2, 0.02)
            else:
                radius *= 0.5
        else:
            radius *= 0.5
        if radius < 1e-13:
            break
    return X

def _lp_polish(X, seconds, deadline):
    X = _norm(X.copy())
    n, d = X.shape
    best = _max_cos(X)
    radius = 5e-4
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < deadline and radius > 1e-13:
        G = X @ X.T
        c = G[iu0]
        m = float(c.max())
        window = max(20.0 * radius, 4e-5)
        order = np.argsort(-c)
        sel = []
        for idx in order:
            if c[idx] > m - window and len(sel) < 650:
                sel.append(int(idx))
            elif c[idx] <= m - window:
                break
            if len(sel) >= 650:
                break
        if len(sel) == 0:
            break
        sel = np.array(sel)
        I = iu0[0][sel]; J = iu0[1][sel]
        p = len(sel); nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                          options={"time_limit": 3.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best - 1e-15:
            X, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return X

def solve():
    t0 = time.time()
    DEAD = t0 + 165.0
    rng = np.random.default_rng(12345)
    best = None; best_val = 1e9
    for f in ("best.npy", "record.npy"):
        try:
            v = np.load(f).astype(float)
            if v.shape != (58, 22):
                continue
            v = _norm(v)
            c = _max_cos(v)
            if c < best_val:
                best_val = c; best = v.copy()
        except Exception:
            continue
    if best is None:
        z = rng.standard_normal((58, 22)); best = _norm(z); best_val = _max_cos(best)
    base = best.copy()
    base_val = best_val
    # contact stats of incumbent for de-jam comparison
    def degs_of(X, gap=0.004):
        G = X @ X.T; np.fill_diagonal(G, -2.0)
        m = float(G.max())
        return (G > m - gap).sum(axis=1), m
    inc_degs, _ = degs_of(best)
    inc_maxdeg = int(inc_degs.max()); inc_mindeg = int(inc_degs.min())
    targets = [best_val - 0.00030, 0.0660, 0.0655]
    # quick squeeze already done in files; don't waste time re-polishing now
    noises = [0.008, 0.016, 0.006, 0.022, 0.011]
    ti = 0
    for t in targets:
        if time.time() >= DEAD - 35:
            break
        for trial in range(3):
            if time.time() >= DEAD - 30:
                break
            ns = noises[(ti) % len(noises)]; ti += 1
            Xp = _pump(base, t, steps=1400, b0=40.0, b1=350.0, lr0=0.004, lr1=0.0004,
                       rng=rng, noise=ns, deadline=DEAD - 28)
            v = _max_cos(Xp)
            dg, _ = degs_of(Xp)
            mxd, mnd = int(dg.max()), int(dg.min())
            # feasible at t -> polish and lower
            if v < t:
                Xp2 = _lp_polish(Xp, 4.0, DEAD - 12)
                v2 = _max_cos(Xp2)
                if v2 < best_val:
                    best, best_val = Xp2.copy(), v2
                    base, base_val = Xp2.copy(), v2
                    inc_degs, _ = degs_of(best)
                    inc_maxdeg, inc_mindeg = int(inc_degs.max()), int(inc_degs.min())
                continue
            # strictly better than best -> polish, record
            if v < best_val - 1e-12:
                Xp2 = _lp_polish(Xp, 4.0, DEAD - 12)
                v2 = _max_cos(Xp2)
                if v2 < best_val:
                    best, best_val = Xp2.copy(), v2
                    base, base_val = Xp2.copy(), v2
                    inc_degs, _ = degs_of(best)
                    inc_maxdeg, inc_mindeg = int(inc_degs.max()), int(inc_degs.min())
                    continue
            # de-jammed candidate? markedly fewer contacts on some point
            # or lower max degree -> adopt as base for next round even if slightly worse
            if v < base_val + 0.004 and (mnd <= inc_mindeg - 6 or mxd <= inc_maxdeg - 3):
                # one stress-weighted LP move from here to try collective rearrangement
                if time.time() < DEAD - 30:
                    Xq = _lp_combined(Xp, 2.5, DEAD - 25, lam=0.03)
                    vq = _max_cos(Xq)
                    # keep the more balanced of Xp/Xq
                    if vq < v + 0.002:
                        Xp, v = Xq, vq
                        dg, _ = degs_of(Xp)
                        mxd, mnd = int(dg.max()), int(dg.min())
                base, base_val = Xp.copy(), v
            elif v < base_val - 2e-6:
                base, base_val = Xp.copy(), v
    # final polish of global best
    try:
        if time.time() < DEAD - 20:
            best = _lp_polish(best, max(2.0, min(18.0, DEAD - 8 - time.time())), DEAD - 2)
            best_val = _max_cos(best)
    except Exception:
        pass
    best = _norm(best)
    # tiny shrink guard (no-op, keep unit rows)
    return best
