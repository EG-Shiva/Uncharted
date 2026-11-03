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

def _riesz_opt(X, iters, s0, s1, lr0, lr1, noise0, noise1, rng):
    n, d = X.shape
    V = np.zeros_like(X)
    mu = 0.8
    for t in range(iters):
        f = t / max(1, iters - 1)
        s = s0 * ((s1 / s0) ** f) if s1 != s0 else s0
        lr = lr0 * ((lr1 / lr0) ** f) if lr1 != lr0 else lr0
        if noise0 <= 0:
            nm = 0.0
        else:
            if noise1 is None or noise1 <= 0:
                nm = noise0 * (1.0 - f)
            else:
                nm = noise0 * ((noise1 / noise0) ** f)
        G = X @ X.T
        D2 = 2.0 - 2.0 * G
        np.fill_diagonal(D2, 1.0)
        np.maximum(D2, 1e-12, out=D2)
        LogD = 0.5 * np.log(D2)
        Lw = -(s + 2.0) * LogD
        np.fill_diagonal(Lw, -1e100)
        mL = float(Lw.max())
        W = np.exp(Lw - mL)
        np.fill_diagonal(W, 0.0)
        S = float(W.sum())
        if not np.isfinite(S) or S <= 0:
            break
        H = W @ X
        Hn = H / S
        dot = (Hn * X).sum(axis=1, keepdims=True)
        Gt = Hn - dot * X
        V = mu * V + Gt
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
    iuI = iu0[0]; iuJ = iu0[1]
    while time.time() - start < seconds:
        if t_deadline is not None and time.time() > t_deadline:
            break
        if radius < 1e-14:
            break
        Gfull = X @ X.T
        cfull = Gfull[iuI, iuJ]
        m = float(cfull.max())
        if m < best:
            best = m
        sel = cfull > m - max(20 * radius, 1e-9)
        cnt = int(sel.sum())
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 3500:
            idx = np.argpartition(-cfull, 3500)[:3500]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = sel & mask
            cnt = 3500
        I = iuI[sel]; J = iuJ[sel]
        c = cfull[sel]
        p = cnt
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

def _k_holes(X_keep, k, rng, n_samp=18000, repel=0.55):
    d = X_keep.shape[1]
    C = rng.randn(n_samp, d)
    C = _norm(C)
    S = C @ X_keep.T
    scores = S.max(axis=1)
    order = np.argsort(scores)
    picked = []
    for idx in order:
        if len(picked) >= k:
            break
        v = C[idx]
        ok = True
        for q in picked:
            if float(v @ q) > repel:
                ok = False
                break
        if ok:
            picked.append(v.copy())
        if len(picked) >= k:
            break
        # also stop scanning too far: only consider best 2000
        # (implicitly by order)
    while len(picked) < k:
        v = rng.randn(d); v /= np.linalg.norm(v)
        picked.append(v)
    return np.array(picked)

def solve():
    t0 = time.time()
    deadline = t0 + 172.0
    rng = np.random.RandomState(1)
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
    cands = []
    trial = 0
    # exploration
    while time.time() < t0 + 112 and trial < 500:
        if time.time() > deadline - 50:
            break
        trial += 1
        mode = trial % 7
        Xt = X_best.copy()
        if mode == 0:
            Xt = _norm(Xt + rng.randn(n, d) * 0.009)
            Xt = _riesz_opt(Xt, 320, 4.0, 18.0, 0.014, 0.002, 0.002, 0.0, rng)
            Xt = _smooth_opt(Xt, 320, 60, 850, 0.006, 0.0008, 0.0, 0.0, rng)
        elif mode == 1:
            Xt = _norm(Xt + rng.randn(n, d) * 0.026)
            Xt = _riesz_opt(Xt, 350, 3.0, 22.0, 0.015, 0.0015, 0.003, 0.0, rng)
            Xt = _smooth_opt(Xt, 320, 70, 900, 0.006, 0.0008, 0.0, 0.0, rng)
        elif mode == 2:
            G = Xt @ Xt.T
            np.fill_diagonal(G, -2.0)
            gm = float(G.max())
            cnt = (G > gm - 1e-7).sum(axis=1)
            k = 2
            victims = np.argsort(-cnt)[:k].tolist()
            mask = np.ones(n, dtype=bool); mask[victims] = False
            holes = _k_holes(Xt[mask], k, rng)
            for vi, h in zip(victims, holes):
                Xt[vi] = h
            Xt = _norm(Xt + rng.randn(n, d) * 0.004)
            Xt = _riesz_opt(Xt, 350, 4.0, 20.0, 0.013, 0.0015, 0.0015, 0.0, rng)
            Xt = _smooth_opt(Xt, 320, 70, 900, 0.006, 0.0008, 0.0, 0.0, rng)
        elif mode == 3:
            G = Xt @ Xt.T
            np.fill_diagonal(G, -2.0)
            gm = float(G.max())
            cnt = (G > gm - 1e-7).sum(axis=1)
            nn = G.max(axis=1)
            v1 = int(np.argmax(cnt))
            v2 = int(np.argmin(nn))
            # third: random among top-10 overloaded
            top10 = np.argsort(-cnt)[:10]
            v3 = int(top10[rng.randint(len(top10))])
            victims = list(dict.fromkeys([v1, v2, v3]))[:3]
            k = len(victims)
            mask = np.ones(n, dtype=bool); mask[victims] = False
            holes = _k_holes(Xt[mask], k, rng)
            for vi, h in zip(victims, holes):
                Xt[vi] = h
            Xt = _norm(Xt + rng.randn(n, d) * 0.003)
            Xt = _riesz_opt(Xt, 380, 3.0, 22.0, 0.014, 0.0012, 0.0015, 0.0, rng)
            Xt = _smooth_opt(Xt, 320, 70, 900, 0.006, 0.0008, 0.0, 0.0, rng)
        elif mode == 4:
            G = Xt @ Xt.T
            np.fill_diagonal(G, -2.0)
            iu = np.triu_indices(n, 1)
            cf = G[iu]
            k = 6
            top = np.argpartition(-cf, k)[:k]
            for q in top:
                i = int(iu[0][q]); j = int(iu[1][q])
                di = Xt[i] - Xt[j]
                Xt[i] = Xt[i] + di * (0.02 + 0.01 * rng.rand())
                Xt[j] = Xt[j] - di * (0.02 + 0.01 * rng.rand())
            Xt = _norm(Xt)
            Xt = _riesz_opt(Xt, 320, 4.0, 20.0, 0.013, 0.0015, 0.002, 0.0, rng)
            Xt = _smooth_opt(Xt, 320, 70, 900, 0.006, 0.0008, 0.0, 0.0, rng)
        elif mode == 5:
            u = rng.randn(d); u /= np.linalg.norm(u)
            proj = Xt @ u
            thr = np.quantile(proj, 0.86)
            sel = np.where(proj >= thr)[0]
            R = rng.randn(d, d) * 0.05
            R = R - R.T
            for i in sel:
                Xt[i] = Xt[i] + (R @ Xt[i]) + rng.randn(d) * 0.008
            Xt = _norm(Xt)
            Xt = _riesz_opt(Xt, 340, 3.0, 20.0, 0.014, 0.0015, 0.002, 0.0, rng)
            Xt = _smooth_opt(Xt, 300, 70, 850, 0.006, 0.0008, 0.0, 0.0, rng)
        else:
            k = 12
            idx = rng.choice(n, k, replace=False)
            Xt[idx] = _norm(Xt[idx] + rng.randn(k, d) * 0.16)
            Xt = _norm(Xt + rng.randn(n, d) * 0.005)
            Xt = _riesz_opt(Xt, 360, 3.0, 24.0, 0.015, 0.0012, 0.003, 0.0, rng)
            Xt = _smooth_opt(Xt, 320, 60, 900, 0.006, 0.0008, 0.0, 0.0, rng)
        try:
            s_try = _max_cos(Xt)
        except Exception:
            continue
        if not np.isfinite(s_try):
            continue
        if s_try < s_best - 1e-12:
            budget = min(5.0, max(2.0, (deadline - time.time()) / 10.0))
            if time.time() + budget < deadline - 6:
                Xt2 = _lp_polish(Xt, budget, t_deadline=deadline - 5)
                s2 = _max_cos(Xt2)
                if s2 < s_best:
                    X_best, s_best = Xt2, s2
                    cands.append((s2, Xt2.copy()))
                elif s_try < s_best:
                    X_best, s_best = Xt, s_try
                    cands.append((s_try, Xt.copy()))
                else:
                    cands.append((min(s2, s_try), Xt2 if s2 < s_try else Xt))
            else:
                X_best, s_best = Xt, s_try
                cands.append((s_try, Xt.copy()))
        else:
            if s_try < s_best + 0.003:
                if len(cands) < 40:
                    cands.append((s_try, Xt.copy()))
                else:
                    # replace worst
                    worst = max(range(len(cands)), key=lambda i: cands[i][0])
                    if s_try < cands[worst][0]:
                        cands[worst] = (s_try, Xt.copy())
    cands = sorted(cands, key=lambda z: z[0])[:6]
    for _, Xc in cands:
        if time.time() > deadline - 14:
            break
        remain = deadline - time.time() - 4
        if remain < 3:
            break
        budget = min(9.0, remain / 2.0)
        budget = max(budget, 2.5)
        Xp = _lp_polish(np.array(Xc, copy=True), budget, t_deadline=deadline - 2)
        sp = _max_cos(Xp)
        if sp < s_best:
            X_best, s_best = Xp, sp
    if time.time() < deadline - 4:
        budget = min(13.0, deadline - time.time() - 1.5)
        if budget > 2:
            Xp = _lp_polish(X_best, budget, t_deadline=deadline - 1)
            sp = _max_cos(Xp)
            if sp < s_best:
                X_best, s_best = Xp, sp
    X_best = _norm(X_best)
    return X_best
