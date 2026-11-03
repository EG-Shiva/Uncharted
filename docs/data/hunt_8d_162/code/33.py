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
        a = a0 * ((a1 / a0) ** f) if a1 != a0 and a0 != 0 else a1
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
        if m < best:
            best = m
        sel = cfull > m - max(20 * radius, 1e-9)
        I = iu0[0][sel]; J = iu0[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > 4000:
            idx = np.argpartition(-cfull[sel], 4000)[:4000]
            I = I[idx]; J = J[idx]
            c = (X @ X.T)[I, J]
            p = len(I)
        else:
            c = cfull[sel]
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
            res = linprog(obj, A_ub=A_ub, b_ub=-c,
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
        if v < best - 1e-15:
            X, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def _largest_hole(X_fixed, rng, n_cand=15000, exclude=None):
    C = _norm(rng.randn(n_cand, X_fixed.shape[1]))
    dots = C @ X_fixed.T
    m = dots.max(axis=1)
    idx = int(np.argmin(m))
    return C[idx]

def _rigid_block_rotate(X, rng):
    n, d = X.shape
    Xt = X.copy()
    u = rng.randn(d); u /= np.linalg.norm(u)
    proj = X @ u
    q = float(rng.choice([0.5, 0.6, 0.72]))
    if rng.rand() < 0.5:
        thr = np.quantile(proj, q)
        sel = np.where(proj >= thr)[0]
    else:
        thr = np.quantile(proj, 1.0 - q)
        sel = np.where(proj <= thr)[0]
    if len(sel) < 5 or len(sel) > n - 5:
        return _norm(Xt + rng.randn(n, d) * 0.01)
    a = rng.randn(d); a /= np.linalg.norm(a)
    b = rng.randn(d); b = b - (b @ a) * a
    nb = np.linalg.norm(b)
    if nb < 1e-9:
        return _norm(Xt + rng.randn(n, d) * 0.01)
    b /= nb
    theta = float(rng.uniform(0.15, 0.5)) * (1.0 if rng.rand() < 0.5 else -1.0)
    ct = np.cos(theta); st = np.sin(theta)
    xa = X[sel] @ a
    xb = X[sel] @ b
    d1 = xa * (ct - 1.0) - xb * st
    d2 = xa * st + xb * (ct - 1.0)
    Xt[sel] = X[sel] + np.outer(d1, a) + np.outer(d2, b)
    Xt[sel] = _norm(Xt[sel] + rng.randn(len(sel), d) * 0.004)
    # tiny global jitter to break symmetry
    Xt = _norm(Xt + rng.randn(n, d) * 0.002)
    return Xt

def _multi_hole_reinsert(Xb, rng, k):
    n, d = Xb.shape
    Xt = Xb.copy()
    G = Xt @ Xt.T
    np.fill_diagonal(G, -2.0)
    nn = G.max(axis=1)
    gm = float(nn.max())
    cnt = (G > gm - 1e-4).sum(axis=1)
    # victims: mostly overloaded + some random
    order = np.argsort(-cnt)
    nv = min(k, n)
    victims = list(order[:max(1, nv - 2)])
    while len(victims) < nv:
        c = int(rng.randint(n))
        if c not in victims:
            victims.append(c)
    # remove and greedily refill
    keep_mask = np.ones(n, dtype=bool)
    keep_mask[victims] = False
    Xkeep = Xt[keep_mask]
    new_pts = []
    cur = Xkeep
    for _ in range(len(victims)):
        h = _largest_hole(cur, rng, n_cand=15000)
        # enforce spread among new points
        ok = False
        for _t in range(5):
            if len(new_pts) == 0:
                ok = True
                break
            md = max(float(h @ q) for q in new_pts)
            if md < 0.6:
                ok = True
                break
            h = _largest_hole(cur, rng, n_cand=5000)
        new_pts.append(h)
        cur = np.vstack([cur, h[None, :]])
    for v, h in zip(victims, new_pts):
        Xt[v] = h
    Xt = _norm(Xt + rng.randn(n, d) * 0.003)
    return Xt

def solve():
    t0 = time.time()
    deadline = t0 + 172.0
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
        Xb = _norm(rng.randn(162, 8))
        sb = _max_cos(Xb)
    n, d = Xb.shape
    X_best = Xb.copy(); s_best = float(sb)
    X_cur = Xb.copy(); s_cur = float(sb)
    cands = []
    trial = 0
    T0, T1 = 0.0022, 0.00015
    max_trials = 500
    while time.time() < t0 + 110 and trial < max_trials:
        if time.time() > deadline - 50:
            break
        trial += 1
        frac = trial / max_trials
        Temp = T0 * ((T1 / T0) ** frac)
        mode = trial % 7
        if mode == 0:
            Xt = _norm(X_cur + rng.randn(n, d) * 0.012)
            Xt = _smooth_opt(Xt, 450, 30, 350, 0.012, 0.0015, 0.0025, 0.0, rng)
            Xt = _smooth_opt(Xt, 300, 80, 900, 0.006, 0.0008, 0.0, 0.0, rng)
        elif mode == 1:
            Xt = _rigid_block_rotate(X_cur, rng)
            Xt = _smooth_opt(Xt, 500, 25, 350, 0.014, 0.0012, 0.003, 0.0, rng)
            Xt = _smooth_opt(Xt, 300, 80, 900, 0.006, 0.0008, 0.0, 0.0, rng)
        elif mode == 2:
            k = int(rng.choice([6, 8, 10, 12]))
            Xt = _multi_hole_reinsert(X_cur, rng, k)
            Xt = _smooth_opt(Xt, 600, 22, 320, 0.015, 0.0012, 0.0032, 0.0, rng)
            Xt = _smooth_opt(Xt, 350, 80, 900, 0.006, 0.0008, 0.0, 0.0, rng)
        elif mode == 3:
            Xt = X_cur.copy()
            G = Xt @ Xt.T; np.fill_diagonal(G, -2.0)
            iu = np.triu_indices(n, 1)
            cf = G[iu]
            kk = 7
            top = np.argpartition(-cf, kk)[:kk]
            for q in top:
                i = int(iu[0][q]); j = int(iu[1][q])
                di = Xt[i] - Xt[j]
                amt = 0.025 + 0.015 * rng.rand()
                Xt[i] = Xt[i] + di * amt
                Xt[j] = Xt[j] - di * amt
            Xt = _norm(Xt)
            Xt = _smooth_opt(Xt, 500, 28, 380, 0.013, 0.0012, 0.0022, 0.0, rng)
            Xt = _smooth_opt(Xt, 300, 80, 900, 0.006, 0.0008, 0.0, 0.0, rng)
        elif mode == 4:
            k = 14
            idx = rng.choice(n, k, replace=False)
            Xt = X_cur.copy()
            Xt[idx] = _norm(Xt[idx] + rng.randn(k, d) * 0.20)
            Xt = _smooth_opt(Xt, 600, 22, 320, 0.015, 0.0012, 0.003, 0.0, rng)
            Xt = _smooth_opt(Xt, 300, 80, 900, 0.006, 0.0008, 0.0, 0.0, rng)
        elif mode == 5:
            Xt = _rigid_block_rotate(X_cur, rng)
            # second block move for symmetry change
            Xt = _rigid_block_rotate(Xt, rng)
            Xt = _smooth_opt(Xt, 550, 25, 350, 0.014, 0.0012, 0.0028, 0.0, rng)
            Xt = _smooth_opt(Xt, 300, 80, 1000, 0.006, 0.0007, 0.0, 0.0, rng)
        else:
            # rattler/hole targeted: move loosest + most crowded
            Xt = X_cur.copy()
            G = X_cur @ X_cur.T; np.fill_diagonal(G, -2.0)
            nn = G.max(axis=1)
            loose = np.argsort(nn)[:3].tolist()
            gm = float(nn.max())
            cnt = (G > gm - 1e-4).sum(axis=1)
            loose.append(int(np.argmax(cnt)))
            loose = list(dict.fromkeys(loose))[:4]
            for v in loose:
                mask = np.ones(n, dtype=bool); mask[v] = False
                hole = _largest_hole(Xt[mask], rng, n_cand=15000)
                Xt[v] = hole
            Xt = _norm(Xt + rng.randn(n, d) * 0.004)
            Xt = _smooth_opt(Xt, 550, 28, 380, 0.013, 0.0012, 0.002, 0.0, rng)
            Xt = _smooth_opt(Xt, 300, 80, 900, 0.006, 0.0008, 0.0, 0.0, rng)
        try:
            s_try = _max_cos(Xt)
        except Exception:
            continue
        if not np.isfinite(s_try):
            continue
        if s_try < s_best - 1e-12:
            # quick LP refine if promising and time allows
            budget = min(6.0, max(2.0, (deadline - time.time()) / 10.0))
            improved = False
            if time.time() + budget < deadline - 8:
                try:
                    Xt2 = _lp_polish(Xt, budget, t_deadline=deadline - 5)
                    s2 = _max_cos(Xt2)
                    if s2 < s_try:
                        Xt, s_try = Xt2, s2
                except Exception:
                    pass
            X_best, s_best = Xt.copy(), float(s_try)
            X_cur, s_cur = Xt.copy(), float(s_try)
            cands.append((float(s_try), Xt.copy()))
        else:
            if s_try < s_best + 0.003:
                cands.append((float(s_try), Xt.copy()))
            # SA acceptance for cur
            if s_try < s_cur or rng.rand() < np.exp((s_cur - s_try) / max(Temp, 1e-9)):
                X_cur, s_cur = Xt.copy(), float(s_try)
            # occasional restart to best
            if rng.rand() < 0.06:
                X_cur, s_cur = X_best.copy(), float(s_best)
    cands = sorted(cands, key=lambda z: z[0])[:6]
    for _, Xc in cands:
        if time.time() > deadline - 12:
            break
        remain = deadline - time.time() - 4
        if remain < 3:
            break
        budget = min(9.0, remain / max(1, len(cands)))
        budget = max(budget, 2.5)
        try:
            Xp = _lp_polish(np.array(Xc, copy=True), budget, t_deadline=deadline - 2)
        except Exception:
            continue
        try:
            sp = _max_cos(Xp)
        except Exception:
            continue
        if sp < s_best:
            X_best, s_best = Xp, sp
    if time.time() < deadline - 4:
        budget = min(13.0, deadline - time.time() - 2)
        if budget > 2:
            try:
                Xp = _lp_polish(X_best, budget, t_deadline=deadline - 1)
                sp = _max_cos(Xp)
                if sp < s_best:
                    X_best, s_best = Xp, sp
            except Exception:
                pass
    X_best = _norm(X_best)
    return X_best
