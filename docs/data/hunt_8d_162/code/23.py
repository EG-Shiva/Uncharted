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
    la0 = np.log(a0) if a0>0 else 0.0
    la1 = np.log(a1) if a1>0 else 0.0
    ll0 = np.log(lr0) if lr0>0 else 0.0
    ll1 = np.log(lr1) if lr1>0 else 0.0
    for t in range(iters):
        f = t / max(1, iters - 1)
        a = float(np.exp(la0 + (la1-la0)*f))
        lr = float(np.exp(ll0 + (ll1-ll0)*f))
        if noise0 <= 0 and noise1 <= 0:
            nm = 0.0
        elif noise0==noise1:
            nm = noise0
        else:
            nm = noise0 + (noise1-noise0)*f
            if nm < 0: nm = 0.0
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
        m = b1 * m + (1 - b1) * Grad
        v = b2 * v + (1 - b2) * (Grad * Grad)
        mh = m / (1 - b1 ** (t + 2))
        vh = v / (1 - b2 ** (t + 2))
        step = mh / (np.sqrt(vh) + eps)
        sd = np.sum(step * X, axis=1, keepdims=True)
        step = step - sd * X
        X = X - lr * step
        if nm > 0:
            N = rng.randn(n, d)
            nd = np.sum(N * X, axis=1, keepdims=True)
            N = N - nd * X
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
        if cnt > 3000:
            k = 3000
            idx = np.argpartition(-cf, k)[:k]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = sel & mask
        I = iuI[sel]; J = iuJ[sel]
        c = cf[sel]
        p = int(I.shape[0])
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

def _refined_hole(X_keep, rng, n_cand=16000):
    d = X_keep.shape[1]
    C = _norm(rng.randn(n_cand, d))
    dots = C @ X_keep.T
    m = dots.max(axis=1)
    order = np.argsort(m)
    best_v = C[order[0]].copy()
    best_s = float(m[order[0]])
    # refine top few candidates by local repulsion
    for k in range(min(4, order.shape[0])):
        v = C[order[k]].copy()
        v = v / np.linalg.norm(v)
        step = 0.05
        for _ in range(60):
            s = X_keep @ v
            gm = float(np.max(s))
            w = np.exp(80.0 * (s - gm))
            g = (w[:, None] * X_keep).sum(axis=0)
            # project
            g = g - float(g @ v) * v
            nrm = float(np.linalg.norm(g))
            if nrm < 1e-12:
                break
            v = v - step * g / nrm
            v = v / np.linalg.norm(v)
            step *= 0.985
        s2 = float(np.max(X_keep @ v))
        if s2 < best_s:
            best_s = s2
            best_v = v.copy()
    return best_v

def solve():
    t0 = time.time()
    deadline = t0 + 172.0
    rng = np.random.RandomState(20240613)
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
    if deadline - time.time() > 12:
        Xp = _lp_polish(X_best, min(7.0, deadline - time.time() - 4), t_deadline=deadline - 2, r0=1.5e-3)
        sp = _max_cos(Xp)
        if sp < s_best:
            X_best, s_best = Xp, sp
    pool = [(s_best, X_best.copy())]
    trial = 0
    stuck = 0
    while time.time() < deadline - 28 and trial < 800:
        trial += 1
        if time.time() > deadline - 30:
            break
        # pick base: mostly best, sometimes pool diversity
        if len(pool) > 3 and (trial % 9 == 8):
            pool.sort(key=lambda z: z[0])
            base = pool[int(rng.randint(1, min(len(pool), 5)))][1].copy()
        else:
            base = X_best.copy()
        mode = trial % 8
        # adapt shake if stuck
        boost = 1.0
        if stuck > 40:
            boost = 1.6
        elif stuck > 15:
            boost = 1.25
        Xt = base
        if mode == 0:
            Xt = _norm(Xt + rng.randn(n, d) * (0.007*boost))
            Xt = _adam_softmax(Xt, 380, 30, 320, 0.015, 0.002, 0.002, 0.0, rng)
            Xt = _adam_softmax(Xt, 260, 150, 1400, 0.006, 0.0007, 0.0, 0.0, rng)
        elif mode == 1:
            Xt = _norm(Xt + rng.randn(n, d) * (0.020*boost))
            Xt = _adam_softmax(Xt, 420, 25, 300, 0.016, 0.0015, 0.003, 0.0, rng)
            Xt = _adam_softmax(Xt, 260, 150, 1600, 0.006, 0.0006, 0.0, 0.0, rng)
        elif mode == 2:
            G = Xt @ Xt.T; np.fill_diagonal(G, -2.0)
            nn = G.max(axis=1)
            gm = float(nn.max())
            cnt = (G > gm - 2e-4).sum(axis=1)
            order = np.argsort(-cnt)
            k = 2 + (trial % 3)
            victims = order[:k].tolist()
            victims.append(int(np.argmin(nn)))
            victims = list(dict.fromkeys(victims))[:4]
            for vv in victims:
                mask = np.ones(n, dtype=bool); mask[vv] = False
                Xt[vv] = _refined_hole(Xt[mask], rng, n_cand=14000)
            Xt = _norm(Xt + rng.randn(n, d) * 0.003)
            Xt = _adam_softmax(Xt, 420, 30, 380, 0.014, 0.0012, 0.0015, 0.0, rng)
            Xt = _adam_softmax(Xt, 280, 150, 1600, 0.006, 0.0006, 0.0, 0.0, rng)
        elif mode == 3:
            k = 7 + (trial % 6)
            idx = rng.choice(n, k, replace=False)
            Xt[idx] = _norm(Xt[idx] + rng.randn(k, d) * (0.16*boost))
            Xt = _adam_softmax(Xt, 440, 25, 300, 0.016, 0.0012, 0.0025, 0.0, rng)
            Xt = _adam_softmax(Xt, 260, 150, 1600, 0.006, 0.0006, 0.0, 0.0, rng)
        elif mode == 4:
            G = Xt @ Xt.T; np.fill_diagonal(G, -2.0)
            iu = np.triu_indices(n, 1)
            cf = G[iu]
            kk = 10
            top = np.argpartition(-cf, kk)[:kk]
            for q in top:
                i = int(iu[0][q]); j = int(iu[1][q])
                di = Xt[i] - Xt[j]
                s1 = 0.03 + 0.015 * rng.rand()
                s2 = 0.03 + 0.015 * rng.rand()
                Xt[i] = Xt[i] + di * s1
                Xt[j] = Xt[j] - di * s2
            Xt = _norm(Xt)
            Xt = _adam_softmax(Xt, 400, 30, 380, 0.014, 0.0012, 0.0018, 0.0, rng)
            Xt = _adam_softmax(Xt, 260, 150, 1600, 0.006, 0.0006, 0.0, 0.0, rng)
        elif mode == 5:
            u = rng.randn(d); u /= np.linalg.norm(u)
            proj = Xt @ u
            thr = np.quantile(proj, 0.86)
            sel = np.where(proj >= thr)[0]
            R = rng.randn(d, d) * 0.07; R = R - R.T
            for i in sel:
                Xt[i] = Xt[i] + (R @ Xt[i]) + rng.randn(d) * 0.012
            Xt = _norm(Xt)
            Xt = _adam_softmax(Xt, 420, 25, 320, 0.015, 0.0012, 0.002, 0.0, rng)
            Xt = _adam_softmax(Xt, 260, 150, 1600, 0.006, 0.0006, 0.0, 0.0, rng)
        elif mode == 6:
            k = 4 + (trial % 4)
            idx = rng.choice(n, k, replace=False)
            keep = np.ones(n, dtype=bool); keep[idx] = False
            Xk = Xt[keep]
            for j, vv in enumerate(idx):
                Xt[vv] = _refined_hole(Xk, rng, n_cand=12000)
                Xk = np.vstack([Xk, Xt[vv][None, :]])
            Xt = _norm(Xt + rng.randn(n, d) * 0.002)
            Xt = _adam_softmax(Xt, 420, 30, 380, 0.014, 0.0012, 0.0012, 0.0, rng)
            Xt = _adam_softmax(Xt, 280, 150, 1700, 0.006, 0.0006, 0.0, 0.0, rng)
        else:
            Xt = _norm(Xt + rng.randn(n, d) * (0.042*boost))
            Xt = _adam_softmax(Xt, 460, 20, 260, 0.017, 0.0015, 0.004, 0.0, rng)
            Xt = _adam_softmax(Xt, 280, 150, 1800, 0.006, 0.0006, 0.0, 0.0, rng)
        try:
            s_try = _max_cos(Xt)
        except Exception:
            stuck += 1
            continue
        if not np.isfinite(s_try):
            stuck += 1
            continue
        if s_try < s_best:
            budget = min(4.0, max(2.0, (deadline - time.time() - 28) / 6.0))
            improved = False
            if time.time() + budget < deadline - 26:
                Xt2 = _lp_polish(Xt, budget, t_deadline=deadline - 25, r0=1.5e-3)
                s2 = _max_cos(Xt2)
                if s2 < s_best:
                    X_best, s_best = Xt2, s2
                    pool.append((s2, Xt2.copy()))
                    improved = True
                elif s_try < s_best:
                    X_best, s_best = Xt, s_try
                    pool.append((s_try, Xt.copy()))
                    improved = True
            else:
                X_best, s_best = Xt, s_try
                pool.append((s_try, Xt.copy()))
                improved = True
            if improved:
                stuck = 0
                pool.sort(key=lambda z: z[0])
                pool = pool[:20]
            else:
                stuck += 1
        else:
            stuck += 1
            if s_try < s_best + 0.0025:
                if len(pool) < 20:
                    pool.append((s_try, Xt.copy()))
                else:
                    pool.sort(key=lambda z: z[0])
                    if s_try < pool[-1][0]:
                        pool[-1] = (s_try, Xt.copy())
    pool.sort(key=lambda z: z[0])
    for _, Xc in pool[:4]:
        if time.time() > deadline - 14:
            break
        remain = deadline - time.time() - 3
        if remain < 2.5:
            break
        budget = min(7.0, max(2.5, remain / 3.0))
        Xp = _lp_polish(np.array(Xc, copy=True), budget, t_deadline=deadline - 2, r0=2e-3)
        sp = _max_cos(Xp)
        if sp < s_best:
            X_best, s_best = Xp, sp
    if time.time() < deadline - 3:
        budget = min(11.0, deadline - time.time() - 0.8)
        if budget > 2:
            Xp = _lp_polish(X_best, budget, t_deadline=deadline - 0.4, r0=1.2e-3)
            sp = _max_cos(Xp)
            if sp < s_best:
                X_best, s_best = Xp, sp
    X_best = _norm(X_best)
    return X_best
