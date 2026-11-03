import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _loads(X, a=200.0):
    G = X @ X.T
    np.fill_diagonal(G, -10.0)
    m = float(G.max())
    W = np.exp(a * (G - m))
    np.fill_diagonal(W, 0.0)
    return W.sum(axis=1), G, m

def _smooth_lse(X, steps, a0, a1, lr0, lr1, noise, mu, rng):
    n, d = X.shape
    V = np.zeros_like(X)
    for t in range(steps):
        f = t / max(1, steps - 1)
        a = a0 * ((a1 / a0) ** f) if a1 != a0 else a0
        lr = lr0 * ((lr1 / lr0) ** f) if lr1 != lr0 else lr0
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        W = np.exp(a * (G - m))
        np.fill_diagonal(W, 0.0)
        s = float(W.sum())
        if not np.isfinite(s) or s <= 0:
            break
        P = W / s
        Gr = P @ X
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        mx = float(np.linalg.norm(Gt, axis=1).max())
        if mx < 1e-14:
            if noise <= 0:
                continue
            mx = 1.0
        else:
            Gt = Gt / mx
        V = mu * V + Gt
        X = X - lr * V
        if noise > 0:
            N = rng.randn(n, d)
            nd = np.sum(N * X, axis=1, keepdims=True)
            N = N - nd * X
            nn = np.linalg.norm(N, axis=1, keepdims=True) + 1e-12
            N = N / nn
            X = X + noise * (1.0 - 0.5 * f) * N
        X = _norm(X)
    return X

def _smooth_pow(X, steps, p0, p1, lr0, lr1, noise, mu, rng):
    n, d = X.shape
    V = np.zeros_like(X)
    for t in range(steps):
        f = t / max(1, steps - 1)
        p = p0 + (p1 - p0) * f
        lr = lr0 * ((lr1 / lr0) ** f) if lr1 != lr0 else lr0
        G = X @ X.T
        np.fill_diagonal(G, 0.0)
        Gp = np.clip(G, 0, 1) ** max(1.0, p - 1.0)
        # weight positive dots
        np.fill_diagonal(Gp, 0.0)
        Gr = Gp @ X
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        mx = float(np.linalg.norm(Gt, axis=1).max())
        if mx < 1e-14:
            break
        Gt = Gt / mx
        V = mu * V + Gt
        X = X - lr * V
        if noise > 0:
            N = rng.randn(n, d)
            nd = np.sum(N * X, axis=1, keepdims=True)
            N = N - nd * X
            X = X + noise * N
        X = _norm(X)
    return X

def _largest_hole(X_keep, rng, n_cand=15000):
    d = X_keep.shape[1]
    C = _norm(rng.randn(n_cand, d))
    D = C @ X_keep.T
    m = D.max(axis=1)
    return C[int(np.argmin(m))]

def _lp_polish(X, seconds, deadline=None):
    X = _norm(np.array(X, dtype=float))
    n, d = X.shape
    best = _max_cos(X)
    radius = 5e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    I0 = iu0[0]; J0 = iu0[1]
    while time.time() - start < seconds:
        if deadline is not None and time.time() > deadline:
            break
        if radius < 1e-14:
            break
        G = X @ X.T
        cf = G[I0, J0]
        m = float(cf.max())
        if m < best:
            best = m
        sel = cf > m - max(20.0 * radius, 1e-9)
        cnt = int(sel.sum())
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 3000:
            idx = np.argpartition(-cf, 3000)[:3000]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = sel & mask
            cnt = 3000
        I = I0[sel]; J = J0[sel]
        c = cf[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = _norm(X + res.x[:-1].reshape(n, d))
        v = _max_cos(y)
        if v < best - 1e-15:
            X, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def solve():
    t0 = time.time()
    deadline = t0 + 170.0
    rng = np.random.RandomState(12345)
    Xb = None; sb = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.load(fn).astype(float)
            if v.shape == (162, 8):
                v = _norm(v)
                s = _max_cos(v)
                if s < sb:
                    sb = s; Xb = v.copy()
        except Exception:
            pass
    if Xb is None:
        Xb = _norm(rng.randn(162, 8)); sb = _max_cos(Xb)
    X_best = Xb.copy(); s_best = float(sb)
    n, d = X_best.shape
    elites = []

    trial = 0
    # exploration phase
    while time.time() < t0 + 122.0:
        if time.time() > deadline - 50:
            break
        trial += 1
        mode = trial % 7
        Xt = X_best.copy()
        if mode == 0:  # tiny global shake + fine anneal
            Xt = _norm(Xt + rng.randn(n, d) * 0.007)
            Xt = _smooth_lse(Xt, 450, 40, 600, 0.010, 0.0012, 0.0015, 0.85, rng)
        elif mode == 1:  # medium shake
            Xt = _norm(Xt + rng.randn(n, d) * 0.025)
            Xt = _smooth_lse(Xt, 550, 60, 900, 0.008, 0.0008, 0.0010, 0.88, rng)
            Xt = _smooth_lse(Xt, 200, 300, 1500, 0.003, 0.0005, 0.0, 0.9, rng)
        elif mode == 2:  # overload relocation to holes
            ld, G, m = _loads(Xt, 150.0)
            k = 2 + (trial % 2)
            order = np.argsort(-ld)[:12]
            # pick k among top with randomness
            sel = order[rng.choice(len(order), k, replace=False)]
            keep_mask = np.ones(n, dtype=bool); keep_mask[sel] = False
            Xkeep = Xt[keep_mask]
            for s_idx in sel:
                h = _largest_hole(Xkeep, rng, 12000)
                Xt[s_idx] = h
                Xkeep = np.vstack([Xkeep, h[None, :]])
            Xt = _smooth_lse(Xt, 600, 50, 800, 0.009, 0.0008, 0.0012, 0.86, rng)
        elif mode == 3:  # cluster shake around random hub
            ld, G, m = _loads(Xt, 150.0)
            hub = int(np.argmax(ld + 0.3 * rng.randn(n)))
            dist = G[hub].copy(); dist[hub] = -10
            nn = np.argsort(-dist)[:7]
            grp = np.concatenate([[hub], nn])
            Xt[grp] = _norm(Xt[grp] + rng.randn(len(grp), d) * 0.12)
            Xt = _smooth_lse(Xt, 500, 50, 700, 0.009, 0.001, 0.001, 0.85, rng)
        elif mode == 4:  # rattler / low-contact reinsertion
            Gf = Xt @ Xt.T
            np.fill_diagonal(Gf, -2.0)
            m = float(Gf.max())
            cnt = (Gf > m - 0.01).sum(axis=1)
            low = np.where(cnt <= np.sort(cnt)[8])[0]
            s_idx = low[rng.randint(len(low))]
            mask = np.ones(n, dtype=bool); mask[s_idx] = False
            h = _largest_hole(Xt[mask], rng, 15000)
            Xt[s_idx] = h
            Xt = _smooth_lse(Xt, 450, 80, 1000, 0.007, 0.0007, 0.0005, 0.88, rng)
        elif mode == 5:  # power-potential basin
            Xt = _norm(Xt + rng.randn(n, d) * 0.015)
            Xt = _smooth_pow(Xt, 350, 12, 40, 0.012, 0.002, 0.001, 0.8, rng)
            Xt = _smooth_lse(Xt, 300, 100, 1200, 0.005, 0.0006, 0.0, 0.9, rng)
        else:  # random subset refresh
            k = 8 + rng.randint(8)
            sel = rng.choice(n, k, replace=False)
            Xt[sel] = _norm(rng.randn(k, d))
            Xt = _smooth_lse(Xt, 600, 40, 700, 0.010, 0.001, 0.0015, 0.84, rng)

        s = _max_cos(Xt)
        if s < s_best - 1e-12:
            X_best = Xt.copy(); s_best = s
            elites.append((s, Xt.copy()))
            elites = sorted(elites, key=lambda e: e[0])[:6]
        elif s < s_best + 0.004:
            elites.append((s, Xt.copy()))
            elites = sorted(elites, key=lambda e: e[0])[:6]
        # occasional intensify best with short LP if big gain candidate
        if trial % 40 == 0 and len(elites) > 0 and time.time() < t0 + 100:
            # quick LP on current best to tighten
            pass

    # final polish: best + elites
    cands = [(s_best, X_best)] + elites
    # dedup sort
    cands = sorted(cands, key=lambda e: e[0])[:5]
    Xf_best = X_best; sf_best = s_best
    remain = deadline - time.time() - 3.0
    if remain < 5:
        return _norm(Xf_best)
    per = remain / max(1, len(cands))
    per = min(per, 14.0)
    for (s, Xc) in cands:
        if time.time() > deadline - 4.0:
            break
        # pre-smooth hard with high beta then LP
        Xp = _smooth_lse(Xc.copy(), 250, 300, 2000, 0.004, 0.0004, 0.0, 0.9, rng)
        Xp = _lp_polish(Xp, seconds=min(per, deadline - time.time() - 2.0), deadline=deadline - 2.0)
        sp = _max_cos(Xp)
        if sp < sf_best:
            sf_best = sp; Xf_best = Xp.copy()
            # propagate improvement
            cands.append((sp, Xp.copy()))
    return _norm(Xf_best)
