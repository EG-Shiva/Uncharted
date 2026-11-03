import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _smooth_opt(X, iters, a0, a1, lr0, lr1, noise0, noise1, rng):
    n, d = X.shape
    V = np.zeros_like(X)
    mu = 0.8
    for t in range(iters):
        f = t / max(1, iters - 1)
        a = a0 * ((a1 / a0) ** f) if a1 != a0 else a0
        lr = lr0 * ((lr1 / lr0) ** f) if lr1 != lr0 else lr0
        if noise0 <= 0:
            nm = 0.0
        else:
            if noise1 is None or noise1 <= 0:
                nm = noise0 * (1.0 - f)
            else:
                nm = noise0 * ((noise1 / noise0) ** f)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        Gm = float(G.max())
        W = np.exp(a * (G - Gm))
        np.fill_diagonal(W, 0.0)
        S = float(W.sum())
        if not np.isfinite(S) or S == 0:
            break
        H = W @ X
        WG = (W * G).sum(axis=1)
        Grad = (H - WG[:, None] * X) / S
        V = mu * V + Grad
        X = X - lr * V
        if nm > 0:
            N = rng.randn(n, d)
            nd = (N * X).sum(axis=1, keepdims=True)
            N = N - nd * X
            X = X + nm * N
        X = _norm(X)
    return X

def _lp_polish(X, seconds, t_deadline=None):
    X = _norm(np.array(X, dtype=float))
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if t_deadline is not None and time.time() > t_deadline:
            break
        if radius < 1e-14:
            break
        Gfull = X @ X.T
        cfull = Gfull[iu0]
        m = float(cfull.max())
        sel = cfull > m - max(20 * radius, 1e-9)
        I = iu0[0][sel]; J = iu0[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 4000:
            idx = np.argpartition(-cfull[sel], 4000)[:4000]
            I = I[idx]; J = J[idx]
            p = len(I)
            csub = (X @ X.T)[I, J]
        else:
            csub = cfull[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csub,
                          A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def _plane_rot(d, rng, theta):
    a = rng.randn(d); b = rng.randn(d)
    u = a / max(np.linalg.norm(a), 1e-12)
    v = b - float(b @ u) * u
    nv = np.linalg.norm(v)
    if nv < 1e-12:
        return np.eye(d)
    v = v / nv
    c = np.cos(theta); s = np.sin(theta)
    R = np.eye(d) + (c - 1.0) * (np.outer(u, u) + np.outer(v, v)) + s * (np.outer(v, u) - np.outer(u, v))
    return R

def _random_rot(d, rng, theta_max, n_planes=1):
    R = np.eye(d)
    for _ in range(n_planes):
        th = float(rng.uniform(-theta_max, theta_max))
        Rp = _plane_rot(d, rng, th)
        R = Rp @ R
    return R

def _largest_hole(X_fixed, rng, n_cand=15000):
    d = X_fixed.shape[1]
    C = _norm(rng.randn(n_cand, d))
    dots = C @ X_fixed.T
    m = dots.max(axis=1)
    return C[int(np.argmin(m))].copy()

def solve():
    t0 = time.time()
    deadline = t0 + 172.0
    rng = np.random.RandomState(11)
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
    cands = []
    trial = 0
    # precompute contact counts for biased seeding (updated when best changes)
    G0 = X_best @ X_best.T
    np.fill_diagonal(G0, -2.0)
    gm0 = float(G0.max())
    cnt0 = (G0 > gm0 - 1e-4).sum(axis=1)
    while time.time() < t0 + 118 and trial < 500:
        if time.time() > deadline - 48:
            break
        trial += 1
        mode = trial % 7
        Xt = X_best.copy()
        if mode in (0, 1, 2, 3, 5):
            # rigid-body coordinated move
            if mode == 0:
                # random cap, medium rotation
                u = _norm(rng.randn(1, d))[0]
                proj = Xt @ u
                k = int(rng.randint(12, 32))
                thr = np.sort(proj)[n - k]
                S = np.where(proj >= thr)[0]
                thmax = float(rng.uniform(0.06, 0.22))
                R = _random_rot(d, rng, thmax, n_planes=1)
                Xt[S] = (R @ Xt[S].T).T
                Xt = _norm(Xt)
                Xt = _smooth_opt(Xt, 420, 30, 320, 0.012, 0.0012, 0.0022, 0.0, rng)
                Xt = _smooth_opt(Xt, 260, 80, 900, 0.006, 0.0008, 0.0, 0.0, rng)
            elif mode == 1:
                # kNN around overloaded point, larger rotation
                # pick seed biased to high contacts
                if rng.rand() < 0.7:
                    # recompute occasionally
                    order = np.argsort(-cnt0)
                    pool = order[:20]
                    seed = int(pool[rng.randint(len(pool))])
                else:
                    seed = int(rng.randint(n))
                dots = Xt @ Xt[seed]
                k = int(rng.randint(10, 30))
                S = np.argpartition(-dots, k)[:k]
                thmax = float(rng.uniform(0.08, 0.28))
                R = _random_rot(d, rng, thmax, n_planes=2 if rng.rand() < 0.4 else 1)
                Xt[S] = (R @ Xt[S].T).T
                Xt = _norm(Xt)
                Xt = _smooth_opt(Xt, 420, 25, 320, 0.013, 0.0012, 0.0025, 0.0, rng)
                Xt = _smooth_opt(Xt, 260, 80, 900, 0.006, 0.0008, 0.0, 0.0, rng)
            elif mode == 2:
                # hemisphere twist: preserves ~half internal pairs
                u = _norm(rng.randn(1, d))[0]
                proj = Xt @ u
                med = np.median(proj)
                S = np.where(proj >= med)[0]
                if len(S) < 20 or len(S) > n - 20:
                    S = np.where(proj > 0)[0]
                thmax = float(rng.uniform(0.03, 0.10))
                R = _random_rot(d, rng, thmax, n_planes=1)
                Xt[S] = (R @ Xt[S].T).T
                Xt = _norm(Xt)
                Xt = _smooth_opt(Xt, 450, 30, 350, 0.012, 0.0012, 0.0018, 0.0, rng)
                Xt = _smooth_opt(Xt, 260, 80, 900, 0.006, 0.0008, 0.0, 0.0, rng)
            elif mode == 3:
                # two disjoint caps counter-rotated
                u1 = _norm(rng.randn(1, d))[0]
                p1 = Xt @ u1
                k1 = int(rng.randint(10, 22))
                thr1 = np.sort(p1)[n - k1]
                S1 = np.where(p1 >= thr1)[0]
                u2 = _norm(rng.randn(1, d))[0]
                p2 = Xt @ u2
                k2 = int(rng.randint(10, 22))
                thr2 = np.sort(p2)[k2 - 1] if k2 < n else -2
                S2 = np.where(p2 <= thr2)[0]
                # make disjoint
                S2 = np.setdiff1d(S2, S1)
                R1 = _random_rot(d, rng, float(rng.uniform(0.05, 0.18)), n_planes=1)
                R2 = _random_rot(d, rng, float(rng.uniform(0.05, 0.18)), n_planes=1)
                Xt[S1] = (R1 @ Xt[S1].T).T
                if len(S2) > 0:
                    Xt[S2] = (R2 @ Xt[S2].T).T
                Xt = _norm(Xt)
                Xt = _smooth_opt(Xt, 420, 25, 320, 0.013, 0.0012, 0.0022, 0.0, rng)
                Xt = _smooth_opt(Xt, 260, 80, 900, 0.006, 0.0008, 0.0, 0.0, rng)
            else:  # mode 5: rattler/hole + rigid nudge of overloaded cluster
                G = Xt @ Xt.T
                np.fill_diagonal(G, -2.0)
                nn = G.max(axis=1)
                loose = int(np.argmin(nn))
                mask = np.ones(n, dtype=bool); mask[loose] = False
                hole = _largest_hole(Xt[mask], rng, n_cand=15000)
                Xt[loose] = hole
                # rigid nudge of densest cap to balance
                gm = float(nn.max())
                cnt = (G > gm - 1e-4).sum(axis=1)
                seed = int(np.argmax(cnt))
                dots = Xt @ Xt[seed]
                k = int(rng.randint(8, 18))
                S = np.argpartition(-dots, k)[:k]
                R = _random_rot(d, rng, float(rng.uniform(0.04, 0.12)), n_planes=1)
                # don't move the just-placed hole if inside
                S = S[S != loose]
                Xt[S] = (R @ Xt[S].T).T
                Xt = _norm(Xt + rng.randn(n, d) * 0.002)
                Xt = _smooth_opt(Xt, 450, 30, 380, 0.012, 0.0012, 0.0015, 0.0, rng)
                Xt = _smooth_opt(Xt, 260, 80, 900, 0.006, 0.0008, 0.0, 0.0, rng)
        elif mode == 4:
            # break worst pairs apart then anneal (interface repair)
            G = Xt @ Xt.T
            np.fill_diagonal(G, -2.0)
            iu = np.triu_indices(n, 1)
            cf = G[iu]
            k = 6
            top = np.argpartition(-cf, k)[:k]
            for q in top:
                i = int(iu[0][q]); j = int(iu[1][q])
                di = Xt[i] - Xt[j]
                mag = 0.02 + 0.01 * rng.rand()
                Xt[i] = Xt[i] + di * mag
                Xt[j] = Xt[j] - di * mag
            Xt = _norm(Xt)
            Xt = _smooth_opt(Xt, 420, 30, 380, 0.012, 0.0012, 0.002, 0.0, rng)
            Xt = _smooth_opt(Xt, 260, 80, 900, 0.006, 0.0008, 0.0, 0.0, rng)
        else:  # mode 6: moderate global shake (rare, for diversity)
            sigma = 0.022
            Xt = _norm(Xt + rng.randn(n, d) * sigma)
            Xt = _smooth_opt(Xt, 500, 25, 300, 0.014, 0.0012, 0.003, 0.0, rng)
            Xt = _smooth_opt(Xt, 260, 80, 900, 0.006, 0.0008, 0.0, 0.0, rng)
        try:
            s_try = _max_cos(Xt)
        except Exception:
            continue
        if not np.isfinite(s_try):
            continue
        if s_try < s_best:
            budget = min(6.0, max(2.0, (deadline - time.time()) / 8.0))
            if time.time() + budget < deadline - 5:
                Xt2 = _lp_polish(Xt, budget, t_deadline=deadline - 5)
                s2 = _max_cos(Xt2)
                if s2 < s_best:
                    X_best, s_best = Xt2, s2
                    G0 = X_best @ X_best.T
                    np.fill_diagonal(G0, -2.0)
                    gm0 = float(G0.max())
                    cnt0 = (G0 > gm0 - 1e-4).sum(axis=1)
                else:
                    if s_try < s_best:
                        X_best, s_best = Xt, s_try
                        G0 = X_best @ X_best.T
                        np.fill_diagonal(G0, -2.0)
                        gm0 = float(G0.max())
                        cnt0 = (G0 > gm0 - 1e-4).sum(axis=1)
                cands.append((min(s2, s_try), Xt2 if s2 < s_try else Xt.copy()))
                if len(cands) > 60:
                    cands = sorted(cands, key=lambda z: z[0])[:30]
            else:
                X_best, s_best = Xt, s_try
                G0 = X_best @ X_best.T
                np.fill_diagonal(G0, -2.0)
                gm0 = float(G0.max())
                cnt0 = (G0 > gm0 - 1e-4).sum(axis=1)
        else:
            if s_try < s_best + 0.0025 and len(cands) < 80:
                cands.append((s_try, Xt.copy()))
    cands = sorted(cands, key=lambda z: z[0])[:6]
    for _, Xc in cands:
        if time.time() > deadline - 12:
            break
        remain = deadline - time.time() - 4
        if remain < 3:
            break
        budget = min(9.0, remain / max(1, len(cands)))
        budget = max(budget, 2.5)
        Xp = _lp_polish(np.array(Xc, copy=True), budget, t_deadline=deadline - 2)
        sp = _max_cos(Xp)
        if sp < s_best:
            X_best, s_best = Xp, sp
    if time.time() < deadline - 4:
        budget = min(13.0, deadline - time.time() - 2)
        if budget > 2:
            Xp = _lp_polish(X_best, budget, t_deadline=deadline - 1)
            sp = _max_cos(Xp)
            if sp < s_best:
                X_best, s_best = Xp, sp
    X_best = _norm(X_best)
    return X_best
