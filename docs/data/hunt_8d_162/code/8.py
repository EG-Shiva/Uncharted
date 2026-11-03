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

def _adam_smooth(X, iters, a0, a1, lr0, lr1, noise0, noise1, rng):
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
        Grad = H - WG[:, None] * X
        # Adam
        m = b1 * m + (1.0 - b1) * Grad
        v = b2 * v + (1.0 - b2) * (Grad * Grad)
        bc1 = 1.0 - b1 ** (t + 1)
        bc2 = 1.0 - b2 ** (t + 1)
        mh = m / bc1
        vh = v / bc2
        step = lr * mh / (np.sqrt(vh) + eps)
        # keep tangent approx: remove radial part of step
        rad = (step * X).sum(axis=1, keepdims=True)
        step = step - rad * X
        # clip extreme steps
        X = X - step
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
        if p > 3000:
            idx = np.argpartition(-cfull[sel], 3000)[:3000]
            I = I[idx]; J = J[idx]
            p = len(I)
        cij = (X @ X.T)[I, J]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr() if False else coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cij, A_eq=A_eq, b_eq=np.zeros(n),
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
        vv = _max_cos(y)
        if vv < best:
            X, best = y, vv
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def _best_hole(X_rem, rng, n_cand=20000):
    d = X_rem.shape[1]
    C = _norm(rng.randn(n_cand, d))
    dots = C @ X_rem.T
    mm = dots.max(axis=1)
    top = np.argpartition(mm, 5)[:5]
    best_v = C[top[0]]
    best_s = 1e9
    for k in top:
        vv = C[k].copy()
        lr = 0.08
        for it in range(300):
            dts = X_rem @ vv
            cur = float(dts.max())
            act = dts > cur - 0.02
            if int(act.sum()) == 0:
                break
            g = X_rem[act].sum(axis=0)
            gt = g - float(g @ vv) * vv
            nn = float(np.linalg.norm(gt))
            if nn < 1e-12:
                break
            gt /= nn
            step = lr * (0.15 ** (it / 300.0))
            vv = vv - step * gt
            vv = vv / max(1e-12, float(np.linalg.norm(vv)))
        s = float((X_rem @ vv).max())
        if s < best_s:
            best_s = s
            best_v = vv
    return best_v

def solve():
    t0 = time.time()
    deadline = t0 + 172.0
    rng = np.random.RandomState(7)
    Xb = None
    sb = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            vv = np.load(fn).astype(float)
            if vv.shape == (162, 8):
                vv = _norm(vv)
                s = _max_cos(vv)
                if s < sb:
                    sb = s
                    Xb = vv.copy()
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
    while time.time() < t0 + 108 and trial < 600:
        if time.time() > deadline - 48:
            break
        trial += 1
        mode = trial % 7
        Xt = X_best.copy()
        if mode == 0:
            Xt = _norm(Xt + rng.randn(n, d) * 0.008)
            Xt = _adam_smooth(Xt, 700, 25, 500, 0.004, 0.001, 0.002, 0.0, rng)
            Xt = _adam_smooth(Xt, 350, 150, 1200, 0.0015, 0.0005, 0.0, 0.0, rng)
        elif mode == 1:
            Xt = _norm(Xt + rng.randn(n, d) * 0.022)
            Xt = _adam_smooth(Xt, 800, 20, 450, 0.0045, 0.001, 0.003, 0.0, rng)
            Xt = _adam_smooth(Xt, 350, 150, 1200, 0.0015, 0.0005, 0.0, 0.0, rng)
        elif mode == 2:
            G = Xt @ Xt.T
            np.fill_diagonal(G, -2.0)
            nn = G.max(axis=1)
            gm = float(nn.max())
            cnt = (G > gm - 2e-4).sum(axis=1)
            order = np.argsort(-cnt)
            k = 2 + (trial % 2)
            victims = order[:k].tolist()
            for vv in victims:
                mask = np.ones(n, dtype=bool); mask[vv] = False
                hole = _best_hole(Xt[mask], rng, n_cand=20000)
                Xt[vv] = hole
            Xt = _norm(Xt + rng.randn(n, d) * 0.003)
            Xt = _adam_smooth(Xt, 750, 25, 500, 0.004, 0.001, 0.0015, 0.0, rng)
            Xt = _adam_smooth(Xt, 350, 150, 1200, 0.0015, 0.0005, 0.0, 0.0, rng)
        elif mode == 3:
            G = X_best @ X_best.T
            np.fill_diagonal(G, -2.0)
            nn = G.max(axis=1)
            loose = np.argsort(nn)[:2].tolist()
            gm = float(nn.max())
            cnt = (G > gm - 1e-4).sum(axis=1)
            worst = int(np.argmax(cnt))
            victims = list(dict.fromkeys(loose + [worst]))[:3]
            for vv in victims:
                mask = np.ones(n, dtype=bool); mask[vv] = False
                hole = _best_hole(Xt[mask], rng, n_cand=16000)
                Xt[vv] = hole
            Xt = _norm(Xt + rng.randn(n, d) * 0.003)
            Xt = _adam_smooth(Xt, 700, 25, 500, 0.004, 0.001, 0.0015, 0.0, rng)
            Xt = _adam_smooth(Xt, 350, 150, 1200, 0.0015, 0.0005, 0.0, 0.0, rng)
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
                amt = 0.025 + 0.015 * rng.rand()
                Xt[i] = Xt[i] + di * amt
                Xt[j] = Xt[j] - di * amt
            Xt = _norm(Xt)
            Xt = _adam_smooth(Xt, 700, 30, 550, 0.004, 0.001, 0.002, 0.0, rng)
            Xt = _adam_smooth(Xt, 350, 150, 1200, 0.0015, 0.0005, 0.0, 0.0, rng)
        elif mode == 5:
            u = rng.randn(d); u /= np.linalg.norm(u)
            proj = X_best @ u
            thr = np.quantile(proj, 0.85)
            sel = np.where(proj >= thr)[0]
            R = rng.randn(d, d) * 0.06
            R = R - R.T
            for i in sel:
                Xt[i] = Xt[i] + (R @ Xt[i]) + rng.randn(d) * 0.009
            Xt = _norm(Xt)
            Xt = _adam_smooth(Xt, 750, 22, 480, 0.0045, 0.001, 0.0022, 0.0, rng)
            Xt = _adam_smooth(Xt, 350, 150, 1200, 0.0015, 0.0005, 0.0, 0.0, rng)
        else:
            k = 12
            idx = rng.choice(n, k, replace=False)
            Xt[idx] = _norm(Xt[idx] + rng.randn(k, d) * 0.16)
            Xt = _adam_smooth(Xt, 800, 20, 420, 0.0045, 0.001, 0.0028, 0.0, rng)
            Xt = _adam_smooth(Xt, 350, 150, 1200, 0.0015, 0.0005, 0.0, 0.0, rng)
        try:
            s_try = _max_cos(Xt)
        except Exception:
            continue
        if not np.isfinite(s_try):
            continue
        if s_try < s_best:
            budget = min(6.0, max(2.0, (deadline - time.time()) / 9.0))
            if time.time() + budget < deadline - 6:
                Xt2 = _lp_polish(Xt, budget, t_deadline=deadline - 6)
                s2 = _max_cos(Xt2)
                if s2 < s_best:
                    X_best, s_best = Xt2, s2
                    cands.append((s2, Xt2.copy()))
                elif s_try < s_best:
                    X_best, s_best = Xt, s_try
                    cands.append((s_try, Xt.copy()))
                else:
                    cands.append((s2, Xt2.copy()))
            else:
                X_best, s_best = Xt, s_try
        else:
            if s_try < s_best + 0.0025:
                cands.append((s_try, Xt.copy()))
    cands = sorted(cands, key=lambda z: z[0])[:6]
    for _, Xc in cands:
        if time.time() > deadline - 13:
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
