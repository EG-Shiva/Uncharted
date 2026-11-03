import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _smooth_gd(X, steps, a0, a1, lr0, lr1, noise0, noise1, rng):
    n, d = X.shape
    V = np.zeros_like(X)
    mu = 0.82
    for t in range(steps):
        f = t / max(1, steps - 1)
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
        Grad = (H - WG[:, None] * X) / S
        V = mu * V + Grad
        # normalize velocity scale a bit
        X = X - lr * V
        if nm > 0:
            N = rng.randn(n, d)
            nd = (N * X).sum(axis=1, keepdims=True)
            N = N - nd * X
            nn = np.linalg.norm(N, axis=1, keepdims=True) + 1e-12
            N = N / nn * np.sqrt(d - 1)
            X = X + nm * N
        X = _norm(X)
    return X

def _lbfgs_smooth(X0, a, maxiter=80):
    n, d = X0.shape
    R = np.linalg.norm(X0, axis=1)
    R = np.maximum(R, 1e-12)
    x0 = (X0 / R[:, None]).ravel() * 1.0
    # work with unnormalized but start normalized; keep scale ~1
    def fun_grad(x):
        X = x.reshape(n, d)
        rn = np.linalg.norm(X, axis=1, keepdims=True)
        rn = np.maximum(rn, 1e-12)
        Y = X / rn
        G = Y @ Y.T
        # shift for stability
        np.fill_diagonal(G, -10.0)
        Gm = float(G.max())
        E = np.exp(a * (G - Gm))
        np.fill_diagonal(E, 0.0)
        S = float(E.sum())
        if not np.isfinite(S) or S <= 0:
            return 0.0, np.zeros_like(x)
        f = Gm + np.log(S) / a  # logsumexp / a, symmetric double count constant offset
        W = E / S
        # d f / dY
        HY = W @ Y  # n x d
        # symmetric: contribution from both i,j -> 2x ; W symmetric so 2*HY
        dY = 2.0 * HY
        # chain to X: (I - y y^T)/r * dY
        dot = (dY * Y).sum(axis=1, keepdims=True)
        dX = (dY - dot * Y) / rn
        return float(f), dX.ravel()
    try:
        res = minimize(fun_grad, x0, jac=True, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': 1e-20, 'gtol': 1e-12})
        X = res.x.reshape(n, d)
        X = _norm(X)
        return X
    except Exception:
        return _norm(X0)

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
        if p > 4000:
            idx = np.argpartition(-cfull[sel], 4000)[:4000]
            I = I[idx]; J = J[idx]
            c = cfull[sel][idx]
        else:
            c = cfull[sel]
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

def _largest_hole(X_fixed, rng, n_cand=20000):
    C = rng.randn(n_cand, X_fixed.shape[1])
    C = _norm(C)
    dots = C @ X_fixed.T
    m = dots.max(axis=1)
    idx = int(np.argmin(m))
    return C[idx]

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
    elites = []
    trial = 0
    # Phase 1: aggressive basin hopping
    while time.time() < t0 + 112 and trial < 500:
        if time.time() > deadline - 50:
            break
        trial += 1
        mode = trial % 7
        Xt = X_best.copy()
        if mode == 0:
            Xt = _norm(Xt + rng.randn(n, d) * 0.009)
            Xt = _smooth_gd(Xt, 420, 30, 350, 0.012, 0.0015, 0.0022, 0.0, rng)
            Xt = _smooth_gd(Xt, 300, 90, 1000, 0.006, 0.0007, 0.0, 0.0, rng)
        elif mode == 1:
            Xt = _norm(Xt + rng.randn(n, d) * 0.025)
            Xt = _smooth_gd(Xt, 520, 25, 300, 0.014, 0.0012, 0.0032, 0.0, rng)
            Xt = _smooth_gd(Xt, 300, 90, 1000, 0.006, 0.0007, 0.0, 0.0, rng)
        elif mode == 2:
            k = 8 + int(rng.randint(0, 6))
            idx = rng.choice(n, k, replace=False)
            Xt[idx] = _norm(Xt[idx] + rng.randn(k, d) * 0.22)
            Xt = _smooth_gd(Xt, 520, 25, 320, 0.014, 0.0012, 0.0028, 0.0, rng)
            Xt = _smooth_gd(Xt, 300, 90, 1000, 0.006, 0.0007, 0.0, 0.0, rng)
        elif mode == 3:
            G = X_best @ X_best.T
            np.fill_diagonal(G, -2.0)
            nn = G.max(axis=1)
            order_loose = np.argsort(nn)
            victims = order_loose[:2].tolist()
            gm = float(nn.max())
            cnt = (G > gm - 1e-4).sum(axis=1)
            victims.append(int(np.argmax(cnt)))
            # second overloaded
            cnt2 = (G > gm - 1e-6).sum(axis=1)
            victims.append(int(np.argmax(cnt2 + rng.rand(n) * 0.5)))
            victims = list(dict.fromkeys(victims))[:4]
            for v in victims:
                mask = np.ones(n, dtype=bool); mask[v] = False
                hole = _largest_hole(Xt[mask], rng, n_cand=15000)
                Xt[v] = hole
            Xt = _norm(Xt + rng.randn(n, d) * 0.004)
            Xt = _smooth_gd(Xt, 520, 30, 380, 0.012, 0.0012, 0.0014, 0.0, rng)
            Xt = _smooth_gd(Xt, 300, 90, 1000, 0.006, 0.0007, 0.0, 0.0, rng)
        elif mode == 4:
            G = Xt @ Xt.T
            np.fill_diagonal(G, -2.0)
            iu = np.triu_indices(n, 1)
            cf = G[iu]
            k = 7
            top = np.argpartition(-cf, k)[:k]
            for q in top:
                i = int(iu[0][q]); j = int(iu[1][q])
                di = Xt[i] - Xt[j]
                s = 0.025 + 0.015 * rng.rand()
                Xt[i] = Xt[i] + di * s
                Xt[j] = Xt[j] - di * s
            Xt = _norm(Xt)
            Xt = _smooth_gd(Xt, 460, 30, 380, 0.012, 0.0012, 0.0018, 0.0, rng)
            Xt = _smooth_gd(Xt, 300, 90, 1000, 0.006, 0.0007, 0.0, 0.0, rng)
        elif mode == 5:
            u = rng.randn(d); u /= np.linalg.norm(u)
            proj = X_best @ u
            thr = np.quantile(proj, 0.82)
            sel = np.where(proj >= thr)[0]
            R = rng.randn(d, d) * 0.06
            R = R - R.T
            for i in sel:
                Xt[i] = Xt[i] + (R @ Xt[i]) + rng.randn(d) * 0.009
            Xt = _norm(Xt)
            Xt = _smooth_gd(Xt, 500, 25, 330, 0.013, 0.0012, 0.0024, 0.0, rng)
            Xt = _smooth_gd(Xt, 300, 90, 1000, 0.006, 0.0007, 0.0, 0.0, rng)
        else:
            # random great-circle shift of a random subset + single point to hole
            k = 5
            idx = rng.choice(n, k, replace=False)
            w = rng.randn(d); w /= np.linalg.norm(w)
            for i in idx:
                comp = float(Xt[i] @ w)
                Xt[i] = Xt[i] - comp * w + rng.randn(d) * 0.03
            # one worst to hole
            G = Xt @ Xt.T
            np.fill_diagonal(G, -2.0)
            wrow = int(np.argmax(G.max(axis=1)))
            mask = np.ones(n, dtype=bool); mask[wrow] = False
            Xt[wrow] = _largest_hole(Xt[mask], rng, n_cand=12000)
            Xt = _norm(Xt)
            Xt = _smooth_gd(Xt, 520, 25, 320, 0.014, 0.0012, 0.0025, 0.0, rng)
            Xt = _smooth_gd(Xt, 300, 90, 1000, 0.006, 0.0007, 0.0, 0.0, rng)
        try:
            s_try = _max_cos(Xt)
        except Exception:
            continue
        if not np.isfinite(s_try):
            continue
        # opportunistic L-BFGS if close
        if s_try < s_best + 0.002 and (trial % 3 == 0 or s_try < s_best):
            if time.time() < deadline - 30:
                try:
                    Xl = _lbfgs_smooth(Xt, 500, maxiter=60)
                    sl = _max_cos(Xl)
                    if sl < s_try:
                        Xt, s_try = Xl, sl
                except Exception:
                    pass
        if s_try < s_best:
            budget = min(5.0, max(2.0, (deadline - time.time()) / 10.0))
            if time.time() + budget < deadline - 5:
                Xt2 = _lp_polish(Xt, budget, t_deadline=deadline - 5)
                s2 = _max_cos(Xt2)
                if s2 < s_best:
                    X_best, s_best = Xt2, s2
                    elites.append((s2, Xt2.copy()))
                elif s_try < s_best:
                    X_best, s_best = Xt, s_try
                    elites.append((s_try, Xt.copy()))
            else:
                X_best, s_best = Xt, s_try
                elites.append((s_try, Xt.copy()))
        else:
            if s_try < s_best + 0.0022:
                elites.append((s_try, Xt.copy()))
                if len(elites) > 40:
                    elites = sorted(elites, key=lambda z: z[0])[:20]
    # Phase 2: polish elites
    elites = sorted(elites, key=lambda z: z[0])[:8]
    for _, Xc in elites:
        if time.time() > deadline - 14:
            break
        remain = deadline - time.time() - 4
        if remain < 3:
            break
        try:
            Xl = _lbfgs_smooth(np.array(Xc, copy=True), 700, maxiter=100)
            sl = _max_cos(Xl)
            if sl < _max_cos(Xc):
                Xc = Xl
        except Exception:
            pass
        budget = min(8.0, max(2.5, remain / 3.0))
        Xp = _lp_polish(np.array(Xc, copy=True), budget, t_deadline=deadline - 2)
        sp = _max_cos(Xp)
        if sp < s_best:
            X_best, s_best = Xp, sp
    if time.time() < deadline - 4:
        budget = min(12.0, deadline - time.time() - 1.5)
        if budget > 2:
            try:
                Xl = _lbfgs_smooth(X_best, 900, maxiter=120)
                sl = _max_cos(Xl)
                if sl < s_best:
                    X_best, s_best = Xl, sl
            except Exception:
                pass
            Xp = _lp_polish(X_best, budget, t_deadline=deadline - 0.5)
            sp = _max_cos(Xp)
            if sp < s_best:
                X_best, s_best = Xp, sp
    X_best = _norm(X_best)
    return X_best
