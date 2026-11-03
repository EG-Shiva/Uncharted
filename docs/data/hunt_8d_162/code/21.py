import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _sgd(X, iters, a0, a1, lr0, lr1, noise0, noise1, rng):
    n, d = X.shape
    V = np.zeros_like(X)
    mu = 0.82
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

def _lbfgs_smooth(Xinit, beta, maxiter=60):
    Xinit = _norm(np.array(Xinit, float))
    n, d = Xinit.shape
    Y0 = Xinit.ravel().copy()
    iu = np.triu_indices(n, 1)
    def fun(Yf):
        Y = Yf.reshape(n, d)
        nr = np.linalg.norm(Y, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        X = Y / nr
        G = X @ X.T
        cf = G[iu]
        m = float(cf.max())
        E = np.exp(beta * (cf - m))
        S = float(E.sum())
        loss = m + np.log(S) / beta
        # weights
        Wv = E / S  # (P,)
        # build symmetric grad: dL/dX_i = sum_j W_ij X_j
        # use sparse accumulation
        Gx = np.zeros_like(X)
        # vectorized via dense W matrix (162x162 small)
        Wm = np.zeros((n, n))
        Wm[iu] = Wv
        Wm = Wm + Wm.T
        Gx = Wm @ X
        # chain Y: dX/dY = (I - xx^T)/|y|
        dot = (Gx * X).sum(axis=1, keepdims=True)
        Gy = (Gx - dot * X) / nr
        return loss, Gy.ravel().astype(float)
    try:
        res = minimize(fun, Y0, jac=True, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-08})
        Y = res.x.reshape(n, d)
        return _norm(Y)
    except Exception:
        return Xinit

def _lp_polish(X, seconds, t_deadline=None):
    X = _norm(np.array(X, dtype=float))
    n, d = X.shape
    best = _max_cos(X)
    radius = 2e-3
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
            break
        if p > 3000:
            idx = np.argpartition(-cfull[sel], 3000)[:3000]
            I = I[idx]; J = J[idx]
            c = cfull[sel][idx]
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

def _largest_hole(X_fixed, rng, n_cand=20000, repel=()):
    d = X_fixed.shape[1]
    C = _norm(rng.randn(n_cand, d))
    dots = C @ X_fixed.T
    m = dots.max(axis=1)
    if len(repel) > 0:
        R = np.stack(repel, axis=0)
        dr = C @ R.T
        mr = dr.max(axis=1)
        m = np.maximum(m, mr + 0.02)
    idx = int(np.argmin(m))
    return C[idx]

def solve():
    t0 = time.time()
    deadline = t0 + 172.0
    rng = np.random.RandomState(12345)
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
    # quick LBFGS warm on best to ensure jammed baseline is tight
    # exploration
    while time.time() < t0 + 118 and trial < 500:
        if time.time() > deadline - 48:
            break
        trial += 1
        mode = trial % 7
        Xt = X_best.copy()
        if mode == 0:  # small global shake + sgd
            Xt = _norm(Xt + rng.randn(n, d) * 0.012)
            Xt = _sgd(Xt, 500, 30, 350, 0.012, 0.0015, 0.0025, 0.0, rng)
            Xt = _sgd(Xt, 300, 100, 900, 0.005, 0.0008, 0.0, 0.0, rng)
        elif mode == 1:  # medium shake + sgd + lbfgs
            Xt = _norm(Xt + rng.randn(n, d) * 0.030)
            Xt = _sgd(Xt, 550, 25, 300, 0.014, 0.0012, 0.003, 0.0, rng)
            if time.time() < deadline - 60 and trial % 2 == 0:
                Xt = _lbfgs_smooth(Xt, 150, maxiter=40)
            Xt = _sgd(Xt, 250, 120, 900, 0.004, 0.0008, 0.0, 0.0, rng)
        elif mode == 2:  # overload excision -> holes
            G = Xt @ Xt.T
            np.fill_diagonal(G, -2.0)
            gm = float(G.max())
            cnt = (G > gm - 5e-4).sum(axis=1)
            # also consider 2e-3 neighborhood for robustness
            cnt2 = (G > gm - 2e-3).sum(axis=1)
            score = cnt.astype(float) + 0.3 * cnt2.astype(float)
            k = int(2 + (trial % 3))  # 2..4
            victims = np.argsort(-score)[:k]
            # keep one copy of remaining for hole search
            mask = np.ones(n, dtype=bool)
            mask[victims] = False
            Xkeep = Xt[mask]
            newpts = []
            for vi in range(k):
                h = _largest_hole(Xkeep if len(newpts) == 0 else np.vstack([Xkeep, np.stack(newpts)]),
                                  rng, n_cand=18000, repel=newpts)
                newpts.append(h)
            for vi, vv in enumerate(victims):
                Xt[vv] = newpts[vi]
            Xt = _norm(Xt + rng.randn(n, d) * 0.004)
            Xt = _sgd(Xt, 550, 25, 380, 0.013, 0.0012, 0.002, 0.0, rng)
            Xt = _sgd(Xt, 250, 120, 900, 0.004, 0.0008, 0.0, 0.0, rng)
        elif mode == 3:  # dense cluster shake
            G = Xt @ Xt.T
            np.fill_diagonal(G, -2.0)
            gm = float(G.max())
            cnt = (G > gm - 5e-4).sum(axis=1)
            v0 = int(np.argmax(cnt))
            dist = G[v0].copy()
            neigh = np.argsort(-dist)[:12]
            Xt[neigh] = _norm(Xt[neigh] + rng.randn(len(neigh), d) * 0.07)
            Xt = _norm(Xt + rng.randn(n, d) * 0.005)
            Xt = _sgd(Xt, 550, 25, 350, 0.014, 0.0012, 0.0025, 0.0, rng)
            Xt = _sgd(Xt, 250, 120, 900, 0.004, 0.0008, 0.0, 0.0, rng)
        elif mode == 4:  # worst-pair split + neighbor jitter
            G = Xt @ Xt.T
            np.fill_diagonal(G, -2.0)
            iu = np.triu_indices(n, 1)
            cf = G[iu]
            k = 8
            top = np.argpartition(-cf, k)[:k]
            for q in top:
                i = int(iu[0][q]); j = int(iu[1][q])
                di = Xt[i] - Xt[j]
                nrm = np.linalg.norm(di) + 1e-12
                di = di / nrm
                amp = 0.025 + 0.015 * rng.rand()
                Xt[i] = Xt[i] + di * amp
                Xt[j] = Xt[j] - di * amp
            Xt = _norm(Xt)
            Xt = _sgd(Xt, 500, 30, 400, 0.012, 0.0012, 0.002, 0.0, rng)
            Xt = _sgd(Xt, 250, 120, 900, 0.004, 0.0008, 0.0, 0.0, rng)
        elif mode == 5:  # random subset reinit
            k = 12
            idx = rng.choice(n, k, replace=False)
            Xt[idx] = _norm(rng.randn(k, d))
            Xt = _sgd(Xt, 600, 20, 300, 0.014, 0.0012, 0.003, 0.0, rng)
            if trial % 3 == 0 and time.time() < deadline - 60:
                Xt = _lbfgs_smooth(Xt, 200, maxiter=30)
            Xt = _sgd(Xt, 250, 120, 900, 0.004, 0.0008, 0.0, 0.0, rng)
        else:  # cap rotation + shake
            u = rng.randn(d); u /= np.linalg.norm(u)
            proj = X_best @ u
            thr = np.quantile(proj, 0.85)
            sel = np.where(proj >= thr)[0]
            R = rng.randn(d, d) * 0.06
            R = R - R.T
            for i in sel:
                Xt[i] = Xt[i] + (R @ Xt[i]) + rng.randn(d) * 0.008
            Xt = _norm(Xt)
            Xt = _sgd(Xt, 550, 25, 350, 0.013, 0.0012, 0.0025, 0.0, rng)
            Xt = _sgd(Xt, 250, 120, 900, 0.004, 0.0008, 0.0, 0.0, rng)
        try:
            s_try = _max_cos(Xt)
        except Exception:
            continue
        if not np.isfinite(s_try):
            continue
        if s_try < s_best:
            budget = min(5.0, max(2.0, (deadline - time.time()) / 10.0))
            if time.time() + budget < deadline - 8:
                Xt2 = _lp_polish(Xt, budget, t_deadline=deadline - 8)
                s2 = _max_cos(Xt2)
                if s2 < s_best:
                    X_best, s_best = Xt2, s2
                    cands.append((s2, Xt2.copy()))
                elif s_try < s_best:
                    X_best, s_best = Xt, s_try
                    cands.append((s_try, Xt.copy()))
            else:
                X_best, s_best = Xt, s_try
                cands.append((s_try, Xt.copy()))
        else:
            if s_try < s_best + 0.002:
                if len(cands) < 60:
                    cands.append((s_try, Xt.copy()))
                else:
                    # replace worst if better
                    cands.sort(key=lambda z: z[0])
                    if s_try < cands[-1][0]:
                        cands[-1] = (s_try, Xt.copy())
    cands = sorted(cands, key=lambda z: z[0])[:6]
    for _, Xc in cands:
        if time.time() > deadline - 14:
            break
        remain = deadline - time.time() - 4
        if remain < 3:
            break
        budget = min(9.0, remain / 2.0)
        budget = max(budget, 2.0)
        Xp = _lp_polish(np.array(Xc, copy=True), budget, t_deadline=deadline - 2)
        sp = _max_cos(Xp)
        if sp < s_best:
            X_best, s_best = Xp, sp
    if time.time() < deadline - 4:
        budget = min(12.0, deadline - time.time() - 1.5)
        if budget > 2:
            Xp = _lp_polish(X_best, budget, t_deadline=deadline - 1)
            sp = _max_cos(Xp)
            if sp < s_best:
                X_best, s_best = Xp, sp
    X_best = _norm(X_best)
    return X_best
