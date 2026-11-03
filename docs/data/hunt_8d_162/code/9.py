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
    return float(np.max(G))

def _adam_softmax(X, iters, a0, a1, lr0, lr1, noise0, noise1, rng):
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(iters):
        f = t / max(1, iters - 1)
        a = a0 * ((a1 / a0) ** f) if a0 != a1 else a0
        lr = lr0 * ((lr1 / lr0) ** f) if lr0 != lr1 else lr0
        if noise0 <= 0:
            nm = 0.0
        else:
            nm = noise0 * ((noise1 / noise0) ** f) if noise1 > 0 else noise0 * (1.0 - f)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        gm = float(np.max(G))
        W = np.exp(a * (G - gm))
        np.fill_diagonal(W, 0.0)
        S = float(np.sum(W))
        if not np.isfinite(S) or S <= 0:
            break
        H = W @ X
        WG = np.sum(W * G, axis=1)
        Grad = (H - WG[:, None] * X) / S
        # project to tangent (already tangent) 
        m = b1 * m + (1 - b1) * Grad
        v = b2 * v + (1 - b2) * (Grad * Grad)
        mh = m / (1 - b1 ** (t + 2))
        vh = v / (1 - b2 ** (t + 2))
        step = mh / (np.sqrt(vh) + eps)
        # keep tangent
        sd = np.sum(step * X, axis=1, keepdims=True)
        step = step - sd * X
        # normalize step scale
        X = X - lr * step
        if nm > 0:
            N = rng.randn(n, d)
            nd = np.sum(N * X, axis=1, keepdims=True)
            N = N - nd * X
            # scale noise to tangent norm
            X = X + nm * N
        X = _norm(X)
    return X

def _lp_polish(X, seconds, t_deadline=None, r0=2e-3):
    X = _norm(np.array(X, dtype=float))
    n, d = X.shape
    best = _max_cos(X)
    radius = r0
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    iuI, iuJ = iu0[0], iu0[1]
    while True:
        if time.time() - start >= seconds:
            break
        if t_deadline is not None and time.time() >= t_deadline:
            break
        if radius < 1e-14:
            break
        G = X @ X.T
        cf = G[iuI, iuJ]
        m = float(np.max(cf))
        if m < best:
            best = m
        sel = cf > m - max(20.0 * radius, 1e-9)
        cnt = int(np.sum(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 3200:
            k = 3200
            idx = np.argpartition(-cf, k)[:k]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = sel & mask
            cnt = k
        I = iuI[sel]; J = iuJ[sel]
        c = cf[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([(I[:, None] * d + np.arange(d)), (J[:, None] * d + np.arange(d)), np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
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
        y = X + res.x[:-1].reshape(n, d)
        y = _norm(y)
        vv = _max_cos(y)
        if vv < best - 1e-15:
            X, best = y, vv
            radius = min(radius * 1.6, 0.04)
        else:
            radius *= 0.5
    return X

def _largest_hole(X_keep, rng, n_cand=25000):
    C = _norm(rng.randn(n_cand, X_keep.shape[1]))
    dots = C @ X_keep.T
    m = dots.max(axis=1)
    return C[int(np.argmin(m))].copy()

def solve():
    t0 = time.time()
    deadline = t0 + 172.0
    rng = np.random.RandomState(12345)
    Xb = None; sb = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            vv = np.load(fn).astype(float)
            if vv.shape == (162, 8):
                vv = _norm(vv)
                s = _max_cos(vv)
                if s < sb:
                    sb = s; Xb = vv.copy()
        except Exception:
            pass
    if Xb is None:
        Xb = _norm(rng.randn(162, 8)); sb = _max_cos(Xb)
    n, d = 162, 8
    X_best = Xb.copy(); s_best = float(sb)
    # immediate squeeze of current best
    if deadline - time.time() > 12:
        Xp = _lp_polish(X_best, min(9.0, deadline - time.time() - 4), t_deadline=deadline - 2, r0=1.5e-3)
        sp = _max_cos(Xp)
        if sp < s_best:
            X_best, s_best = Xp, sp
    cands = []
    trial = 0
    # exploration
    while time.time() < deadline - 30 and trial < 600:
        trial += 1
        if time.time() > deadline - 32:
            break
        mode = trial % 7
        Xt = X_best.copy()
        if mode == 0:
            Xt = _norm(Xt + rng.randn(n, d) * 0.008)
            Xt = _adam_softmax(Xt, 420, 30, 300, 0.015, 0.002, 0.002, 0.0, rng)
            Xt = _adam_softmax(Xt, 280, 150, 1200, 0.006, 0.0008, 0.0, 0.0, rng)
        elif mode == 1:
            Xt = _norm(Xt + rng.randn(n, d) * 0.022)
            Xt = _adam_softmax(Xt, 450, 25, 280, 0.016, 0.0015, 0.003, 0.0, rng)
            Xt = _adam_softmax(Xt, 280, 150, 1500, 0.006, 0.0007, 0.0, 0.0, rng)
        elif mode == 2:
            # hub relocation to holes
            G = Xt @ Xt.T; np.fill_diagonal(G, -2.0)
            nn = G.max(axis=1)
            gm = float(nn.max())
            cnt = (G > gm - 2e-4).sum(axis=1)
            order = np.argsort(-cnt)
            k = 2 + (trial % 2)
            victims = order[:k].tolist()
            # add rattler (min nn)
            victims.append(int(np.argmin(nn)))
            victims = list(dict.fromkeys(victims))[:3]
            for vv in victims:
                mask = np.ones(n, dtype=bool); mask[vv] = False
                Xt[vv] = _largest_hole(Xt[mask], rng, n_cand=20000)
            Xt = _norm(Xt + rng.randn(n, d) * 0.003)
            Xt = _adam_softmax(Xt, 450, 30, 350, 0.014, 0.0012, 0.0015, 0.0, rng)
            Xt = _adam_softmax(Xt, 300, 150, 1500, 0.006, 0.0007, 0.0, 0.0, rng)
        elif mode == 3:
            # strong subset jump
            k = 8 + (trial % 5)
            idx = rng.choice(n, k, replace=False)
            Xt[idx] = _norm(Xt[idx] + rng.randn(k, d) * 0.16)
            Xt = _adam_softmax(Xt, 480, 25, 300, 0.016, 0.0012, 0.0025, 0.0, rng)
            Xt = _adam_softmax(Xt, 280, 150, 1500, 0.006, 0.0007, 0.0, 0.0, rng)
        elif mode == 4:
            # break worst pairs apart
            G = Xt @ Xt.T; np.fill_diagonal(G, -2.0)
            iu = np.triu_indices(n, 1)
            cf = G[iu]
            kk = 8
            top = np.argpartition(-cf, kk)[:kk]
            for q in top:
                i = int(iu[0][q]); j = int(iu[1][q])
                di = Xt[i] - Xt[j]
                Xt[i] = Xt[i] + di * (0.025 + 0.01 * rng.rand())
                Xt[j] = Xt[j] - di * (0.025 + 0.01 * rng.rand())
            Xt = _norm(Xt)
            Xt = _adam_softmax(Xt, 420, 30, 350, 0.014, 0.0012, 0.0018, 0.0, rng)
            Xt = _adam_softmax(Xt, 280, 150, 1500, 0.006, 0.0007, 0.0, 0.0, rng)
        elif mode == 5:
            # random cap rotation
            u = rng.randn(d); u /= np.linalg.norm(u)
            proj = Xt @ u
            thr = np.quantile(proj, 0.88)
            sel = np.where(proj >= thr)[0]
            R = rng.randn(d, d) * 0.06; R = R - R.T
            for i in sel:
                Xt[i] = Xt[i] + (R @ Xt[i]) + rng.randn(d) * 0.010
            Xt = _norm(Xt)
            Xt = _adam_softmax(Xt, 450, 25, 300, 0.015, 0.0012, 0.002, 0.0, rng)
            Xt = _adam_softmax(Xt, 280, 150, 1500, 0.006, 0.0007, 0.0, 0.0, rng)
        else:
            # global large shake + cold anneal
            Xt = _norm(Xt + rng.randn(n, d) * 0.045)
            Xt = _adam_softmax(Xt, 500, 20, 250, 0.017, 0.0015, 0.004, 0.0, rng)
            Xt = _adam_softmax(Xt, 300, 150, 1800, 0.006, 0.0006, 0.0, 0.0, rng)
        try:
            s_try = _max_cos(Xt)
        except Exception:
            continue
        if not np.isfinite(s_try):
            continue
        if s_try < s_best:
            budget = min(5.0, max(2.0, (deadline - time.time() - 28) / 6.0))
            if time.time() + budget < deadline - 26:
                Xt2 = _lp_polish(Xt, budget, t_deadline=deadline - 25, r0=1.5e-3)
                s2 = _max_cos(Xt2)
                if s2 < s_best:
                    X_best, s_best = Xt2, s2
                    cands.append((s2, Xt2.copy()))
                elif s_try < s_best:
                    X_best, s_best = Xt, s_try
                    cands.append((s_try, Xt.copy()))
            else:
                X_best, s_best = Xt, s_try
        else:
            if s_try < s_best + 0.002:
                if len(cands) < 30:
                    cands.append((s_try, Xt.copy()))
                else:
                    # replace worst
                    cands.sort(key=lambda z: z[0])
                    if s_try < cands[-1][0]:
                        cands[-1] = (s_try, Xt.copy())
    # polish promising candidates
    cands.sort(key=lambda z: z[0])
    for _, Xc in cands[:5]:
        if time.time() > deadline - 14:
            break
        remain = deadline - time.time() - 3
        if remain < 2.5:
            break
        budget = min(8.0, max(2.5, remain / 3.0))
        Xp = _lp_polish(np.array(Xc, copy=True), budget, t_deadline=deadline - 2, r0=2e-3)
        sp = _max_cos(Xp)
        if sp < s_best:
            X_best, s_best = Xp, sp
    if time.time() < deadline - 3:
        budget = min(12.0, deadline - time.time() - 1.0)
        if budget > 2:
            Xp = _lp_polish(X_best, budget, t_deadline=deadline - 0.5, r0=1.2e-3)
            sp = _max_cos(Xp)
            if sp < s_best:
                X_best, s_best = Xp, sp
    X_best = _norm(X_best)
    return X_best
