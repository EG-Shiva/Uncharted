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
            if noise1 <= 0:
                nm = noise0 * (1.0 - f)
            else:
                nm = noise0 * ((noise1 / noise0) ** f)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        Gm = float(G.max())
        # weights focused on largest cosines
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
        # cap size for speed
        if p > 4000:
            # keep tightest 4000
            idx = np.argpartition(-cfull[sel], 4000)[:4000]
            I = I[idx]; J = J[idx]
            p = len(I)
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
            res = linprog(obj, A_ub=A_ub, b_ub=-cfull[sel][:p] if p == sel.sum() else -(X @ X.T)[I, J],
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

def _largest_hole(X_fixed, rng, n_cand=20000):
    n, d = X_fixed.shape
    C = rng.randn(n_cand, d)
    C = _norm(C)
    dots = C @ X_fixed.T
    m = dots.max(axis=1)
    idx = int(np.argmin(m))
    return C[idx]

def solve():
    t0 = time.time()
    deadline = t0 + 172.0
    rng = np.random.RandomState(0)
    # load start
    Xb = None
    sb = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.load(fn).astype(float)
            if v.shape == (162, 8):
                v = _norm(v)
                s = _max_cos(v)
                if s < sb:
                    sb = s
                    Xb = v.copy()
        except Exception:
            pass
    if Xb is None:
        Xb = _norm(rng.randn(162, 8))
        sb = _max_cos(Xb)
    X_best = Xb.copy()
    s_best = float(sb)
    n, d = X_best.shape

    cands = []  # (score, X)
    trial = 0
    # phase 1: exploration with smooth annealing
    while time.time() < t0 + 110 and len(cands) < 200 and trial < 400:
        if time.time() > deadline - 45:
            break
        trial += 1
        mode = trial % 6
        Xt = X_best.copy()
        if mode == 0:  # small global shake
            sigma = 0.010
            Xt = _norm(Xt + rng.randn(n, d) * sigma)
            Xt = _smooth_opt(Xt, 500, 30, 350, 0.012, 0.0015, 0.0025, 0.0, rng)
            Xt = _smooth_opt(Xt, 350, 80, 900, 0.006, 0.0008, 0.0, 0.0, rng)
        elif mode == 1:  # medium global shake
            sigma = 0.028
            Xt = _norm(Xt + rng.randn(n, d) * sigma)
            Xt = _smooth_opt(Xt, 600, 25, 300, 0.014, 0.0012, 0.0035, 0.0, rng)
            Xt = _smooth_opt(Xt, 350, 80, 900, 0.006, 0.0008, 0.0, 0.0, rng)
        elif mode == 2:  # strong subset jump
            k = 10
            idx = rng.choice(n, k, replace=False)
            Xt[idx] = _norm(Xt[idx] + rng.randn(k, d) * 0.18)
            Xt = _smooth_opt(Xt, 600, 25, 350, 0.014, 0.0012, 0.003, 0.0, rng)
            Xt = _smooth_opt(Xt, 350, 80, 900, 0.006, 0.0008, 0.0, 0.0, rng)
        elif mode == 3:  # rattler / loose to hole + jammed shake
            G = X_best @ X_best.T
            np.fill_diagonal(G, -2.0)
            nn = G.max(axis=1)
            order_loose = np.argsort(nn)  # smallest first = most isolated
            victims = order_loose[:2].tolist()
            # also add one overloaded (largest nn ties -> most contacts)
            # count contacts within 1e-4 of max
            gm = float(nn.max())
            cnt = (G > gm - 1e-4).sum(axis=1)
            victims.append(int(np.argmax(cnt)))
            victims = list(dict.fromkeys(victims))[:3]
            for v in victims:
                mask = np.ones(n, dtype=bool); mask[v] = False
                hole = _largest_hole(Xt[mask], rng, n_cand=15000)
                Xt[v] = hole
            # tiny jitter others
            Xt = _norm(Xt + rng.randn(n, d) * 0.004)
            Xt = _smooth_opt(Xt, 550, 30, 400, 0.012, 0.0012, 0.0015, 0.0, rng)
            Xt = _smooth_opt(Xt, 350, 80, 900, 0.006, 0.0008, 0.0, 0.0, rng)
        elif mode == 4:  # break worst pairs
            G = Xt @ Xt.T
            np.fill_diagonal(G, -2.0)
            # find top few pairs
            iu = np.triu_indices(n, 1)
            cf = G[iu]
            k = 6
            top = np.argpartition(-cf, k)[:k]
            for q in top:
                i = int(iu[0][q]); j = int(iu[1][q])
                # push apart
                di = Xt[i] - Xt[j]
                dj = -di
                Xt[i] = Xt[i] + di * (0.02 + 0.01 * rng.rand())
                Xt[j] = Xt[j] + dj * (0.02 + 0.01 * rng.rand())
            Xt = _norm(Xt)
            Xt = _smooth_opt(Xt, 500, 30, 400, 0.012, 0.0012, 0.002, 0.0, rng)
            Xt = _smooth_opt(Xt, 350, 80, 900, 0.006, 0.0008, 0.0, 0.0, rng)
        else:  # cluster cap move
            u = rng.randn(d); u /= np.linalg.norm(u)
            proj = X_best @ u
            thr = np.quantile(proj, 0.85)
            sel = np.where(proj >= thr)[0]
            # random small rotation of cap
            R = rng.randn(d, d) * 0.05
            R = R - R.T
            for i in sel:
                Xt[i] = Xt[i] + (R @ Xt[i]) + rng.randn(d) * 0.008
            Xt = _norm(Xt)
            Xt = _smooth_opt(Xt, 550, 25, 350, 0.013, 0.0012, 0.0025, 0.0, rng)
            Xt = _smooth_opt(Xt, 350, 80, 900, 0.006, 0.0008, 0.0, 0.0, rng)

        try:
            s_try = _max_cos(Xt)
        except Exception:
            continue
        if not np.isfinite(s_try):
            continue
        if s_try < s_best:
            # immediate short LP refine
            budget = min(6.0, max(2.0, (deadline - time.time()) / 8.0))
            if time.time() + budget < deadline - 5:
                Xt2 = _lp_polish(Xt, budget, t_deadline=deadline - 5)
                s2 = _max_cos(Xt2)
                if s2 < s_best:
                    X_best, s_best = Xt2, s2
                else:
                    # keep smooth version if still better
                    if s_try < s_best:
                        X_best, s_best = Xt, s_try
                # store also
                cands.append((min(s2, s_try), Xt2 if s2 < s_try else Xt))
            else:
                X_best, s_best = Xt, s_try
        else:
            if s_try < s_best + 0.0025:
                cands.append((s_try, Xt.copy()))
    # phase 2: LP polish most promising candidates + best
    cands = sorted(cands, key=lambda z: z[0])[:6]
    # always include current best
    for _, Xc in cands:
        if time.time() > deadline - 12:
            break
        remain = deadline - time.time() - 4
        if remain < 3:
            break
        budget = min(10.0, remain / max(1, len(cands)))
        budget = max(budget, 2.5)
        Xp = _lp_polish(np.array(Xc, copy=True), budget, t_deadline=deadline - 2)
        sp = _max_cos(Xp)
        if sp < s_best:
            X_best, s_best = Xp, sp
    # final polish of best
    if time.time() < deadline - 4:
        budget = min(14.0, deadline - time.time() - 2)
        if budget > 2:
            Xp = _lp_polish(X_best, budget, t_deadline=deadline - 1)
            sp = _max_cos(Xp)
            if sp < s_best:
                X_best, s_best = Xp, sp
    X_best = _norm(X_best)
    return X_best
