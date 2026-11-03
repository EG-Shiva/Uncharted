import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _lp_polish(x, seconds, deadline):
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(np.max(c))
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        I = iu[0][sel]
        J = iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        csel = c[sel]
        if p > 3500:
            keep = np.argpartition(csel, -3500)[-3500:]
            I = I[keep]; J = J[keep]; csel = csel[keep]
            p = 3500
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
            A_eq = coo_matrix(
                (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                shape=(n, n * d + 1)).tocsr()
            obj = np.zeros(n * d + 1); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel,
                          A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            y = x + res.x[:-1].reshape(n, d)
            y = y / np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
            v = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _anneal(X0, steps, T0, T1, lr0, lr1, mom, rng=None, noise=0.0):
    X = X0.copy()
    n, d = X.shape
    V = np.zeros_like(X)
    lT0 = np.log(T0); lT1 = np.log(T1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    denom = max(steps - 1, 1)
    max_step = 0.12
    for s in range(steps):
        frac = s / denom
        T = float(np.exp(lT0 + frac * (lT1 - lT0)))
        lr = float(np.exp(ll0 + frac * (ll1 - ll0)))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = np.max(G, axis=1, keepdims=True)
        E = np.exp((G - M) / T)
        np.fill_diagonal(E, 0.0)
        S = np.sum(E, axis=1, keepdims=True) + 1e-300
        P = E / S
        W = P @ X
        dot = np.sum(W * X, axis=1, keepdims=True)
        Gt = W - dot * X
        V = mom * V - lr * Gt
        if noise > 0 and rng is not None:
            N = rng.normal(size=(n, d)) * noise
            Nd = np.sum(N * X, axis=1, keepdims=True)
            N = N - Nd * X
            V = V + N
        vn = np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
        sc = np.minimum(1.0, max_step / vn)
        V = V * sc
        X = X + V
        X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    return X

def _lbfgs_smooth(Xinit, betas=(60.0, 150.0, 350.0), iters=(120, 120, 150)):
    n, d = Xinit.shape
    Y = Xinit / np.maximum(np.linalg.norm(Xinit, axis=1, keepdims=True), 1e-12)
    iu = np.triu_indices(n, 1)
    for beta, mi in zip(betas, iters):
        def fun(Yf):
            Ym = Yf.reshape(n, d)
            nr = np.sqrt(np.sum(Ym * Ym, axis=1, keepdims=True)) + 1e-30
            X = Ym / nr
            G = X @ X.T
            c = G[iu]
            mv = float(np.max(c))
            E = np.exp(beta * (c - mv))
            S = float(np.sum(E)) + 1e-300
            f = mv + np.log(S) / beta
            W = np.zeros((n, n))
            W[iu] = E
            W = W + W.T
            GX = (W @ X) / S
            dot = np.sum(GX * X, axis=1, keepdims=True)
            Gt = GX - dot * X
            GY = Gt / nr
            return float(f), GY.ravel()
        try:
            res = minimize(fun, Y.ravel(), jac=True, method='L-BFGS-B',
                           options={'maxiter': int(mi), 'ftol': 1e-14, 'gtol': 1e-6, 'maxcor': 12})
            Y = res.x.reshape(n, d)
            Y = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
        except Exception:
            break
    X = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
    return X

def _refine_void(v0, Xrem, beta=140.0, steps=120, lr=0.08):
    v = v0 / (np.linalg.norm(v0) + 1e-30)
    for t in range(steps):
        dots = Xrem @ v
        mv = float(np.max(dots))
        W = np.exp(beta * (dots - mv))
        S = float(np.sum(W)) + 1e-300
        g = (W @ Xrem) / S
        gt = g - float(np.dot(g, v)) * v
        st = lr / (1.0 + t * 0.02)
        v = v - st * gt
        v = v / (np.linalg.norm(v) + 1e-30)
    return v

def _contact_counts(X, gap=0.002):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    C = (G > m - gap).sum(axis=1)
    return C, m

def solve():
    np.random.seed(0)
    t_start = time.time()
    deadline = t_start + 172.0
    n_pts, n_dim = 138, 8
    try:
        Xbest = np.load("best.npy").astype(float)
    except Exception:
        try:
            Xbest = np.load("record.npy").astype(float)
        except Exception:
            Xbest = np.random.default_rng(0).normal(size=(n_pts, n_dim))
    if Xbest.shape != (n_pts, n_dim):
        tmp = np.random.default_rng(0).normal(size=(n_pts, n_dim))
        m0 = min(Xbest.shape[0], n_pts); d0 = min(Xbest.shape[1], n_dim)
        tmp[:m0, :d0] = Xbest[:m0, :d0]
        Xbest = tmp
    Xbest = Xbest / np.maximum(np.linalg.norm(Xbest, axis=1, keepdims=True), 1e-12)
    best_val = _max_cos(Xbest)
    rng = np.random.default_rng(7)
    cycle = 0
    # elite pool for crossover of basins
    elites = [Xbest.copy()]
    elite_vals = [best_val]
    while time.time() < deadline - 12.0:
        cycle += 1
        if time.time() > deadline - 12.0:
            break
        op = rng.random()
        # choose base: mostly best, sometimes 2nd elite
        if len(elites) > 1 and rng.random() < 0.25:
            Xbase = elites[1].copy()
        else:
            Xbase = Xbest.copy()
        try:
            cnt, _m = _contact_counts(Xbase)
        except Exception:
            cnt = np.ones(n_pts)
        cmax = float(np.max(cnt)) + 1e-12
        w = 0.5 + 1.5 * (cnt.astype(float) / cmax)
        Xinit = Xbase.copy()
        if op < 0.50:
            # contact-weighted Gaussian shake + anneal + lbfgs
            sigma = float(10.0 ** rng.uniform(-2.4, -0.55))
            n_replace = int(rng.integers(0, 8))
            if n_replace > 0:
                # replace low-contact (rattlers) randomly to probe voids
                G = Xinit @ Xinit.T
                np.fill_diagonal(G, -2.0)
                per = G.max(axis=1)
                order = np.argsort(per)  # smallest first = rattlers
                # replace mix of rattlers and random
                n_r = min(n_replace, int(rng.integers(0, n_replace + 1)))
                idx_r = order[:max(1, 30)][:n_r] if n_r > 0 else np.array([], dtype=int)
                idx_rest_n = n_replace - len(idx_r)
                cand_pool = np.setdiff1d(np.arange(n_pts), idx_r) if len(idx_r) else np.arange(n_pts)
                idx_rest = rng.choice(cand_pool, size=idx_rest_n, replace=False) if idx_rest_n > 0 else np.array([], dtype=int)
                idx = np.concatenate([idx_r, idx_rest]).astype(int) if (len(idx_r) + len(idx_rest)) > 0 else np.array([], dtype=int)
                R = rng.normal(size=(len(idx), n_dim))
                R /= np.maximum(np.linalg.norm(R, axis=1, keepdims=True), 1e-12)
                Xinit[idx] = R
            N = rng.normal(size=(n_pts, n_dim)) * (sigma * w[:, None])
            Xinit = Xinit + N
            Xinit /= np.maximum(np.linalg.norm(Xinit, axis=1, keepdims=True), 1e-12)
            T0 = float(10.0 ** rng.uniform(-1.7, -1.0))
            T1 = float(10.0 ** rng.uniform(-2.7, -2.1))
            lr0 = float(10.0 ** rng.uniform(-2.4, -1.5))
            lr1 = lr0 * 0.15
            mom = float(rng.uniform(0.78, 0.93))
            steps = int(rng.integers(1200, 3200))
            noise = float(rng.uniform(0.0, 0.002)) if rng.random() < 0.3 else 0.0
            Xa = _anneal(Xinit, steps, T0, T1, lr0, lr1, mom, rng=rng if noise > 0 else None, noise=noise)
            try:
                Xa = _lbfgs_smooth(Xa)
            except Exception:
                pass
            va = _max_cos(Xa)
        elif op < 0.72:
            # void rebirth: kill worst core points, insert optimized voids
            G = Xinit @ Xinit.T
            np.fill_diagonal(G, -2.0)
            per = G.max(axis=1)
            k = int(rng.integers(1, 5))
            order = np.argsort(-per)
            pool = order[:12]
            victims = rng.choice(pool, size=k, replace=False)
            Xa = Xinit.copy()
            for v in victims:
                Xrem = np.delete(Xa, int(v), axis=0)
                Ncand = 15000
                cand = rng.normal(size=(Ncand, n_dim))
                cand /= np.maximum(np.linalg.norm(cand, axis=1, keepdims=True), 1e-12)
                smax = (cand @ Xrem.T).max(axis=1)
                bi = int(np.argmin(smax))
                v0 = cand[bi]
                try:
                    beta_v = float(rng.uniform(100.0, 180.0))
                    v0 = _refine_void(v0, Xrem, beta=beta_v, steps=140, lr=0.08)
                except Exception:
                    pass
                Xa[int(v)] = v0
            # light anneal to settle
            T0 = float(10.0 ** rng.uniform(-1.8, -1.3))
            T1 = T0 * 0.2
            lr0 = float(10.0 ** rng.uniform(-2.5, -1.9))
            steps = int(rng.integers(600, 1600))
            mom = float(rng.uniform(0.80, 0.90))
            try:
                Xa = _anneal(Xa, steps, T0, T1, lr0, lr0 * 0.2, mom)
                Xa = _lbfgs_smooth(Xa, betas=(120.0, 300.0), iters=(100, 120))
            except Exception:
                pass
            va = _max_cos(Xa)
        elif op < 0.87:
            # collective cap move: rotate a hemisphere coherently then anneal
            u = rng.normal(size=(n_dim,)); u /= (np.linalg.norm(u) + 1e-12)
            proj = Xinit @ u
            thr = float(np.quantile(proj, rng.uniform(0.55, 0.85)))
            mask = proj > thr
            if int(mask.sum()) < 3 or int(mask.sum()) > n_pts - 3:
                continue
            # random small rotation in plane (u, w)
            w = rng.normal(size=(n_dim,)); w = w - float(np.dot(w, u)) * u
            w /= (np.linalg.norm(w) + 1e-12)
            ang = float(rng.uniform(-0.25, 0.25))
            # rotate masked points by ang in (u,w) plane approx via tangent step
            Xa = Xinit.copy()
            cu, cw = np.cos(ang), np.sin(ang)
            pu = (Xa[mask] @ u)
            pw = (Xa[mask] @ w)
            # build rotation of components along u,w
            base_u = np.outer(pu, u); base_w = np.outer(pw, w)
            rest = Xa[mask] - base_u - base_w
            nu = pu * cu - pw * cw if rng.random() < 0.5 else pu * cu + pw * cw * 0.3
            nw = pu * cw + pw * cu if rng.random() < 0.5 else pw
            # simpler: apply tangential push along w proportional to pu
            push = float(rng.uniform(-0.3, 0.3))
            Xa[mask] = Xa[mask] + push * np.outer(pu - pu.mean(), w)
            Xa /= np.maximum(np.linalg.norm(Xa, axis=1, keepdims=True), 1e-12)
            T0 = float(10.0 ** rng.uniform(-1.6, -1.1))
            T1 = float(10.0 ** rng.uniform(-2.6, -2.2))
            lr0 = float(10.0 ** rng.uniform(-2.3, -1.6))
            steps = int(rng.integers(1200, 2800))
            mom = float(rng.uniform(0.80, 0.92))
            try:
                Xa = _anneal(Xa, steps, T0, T1, lr0, lr0 * 0.15, mom)
                Xa = _lbfgs_smooth(Xa)
            except Exception:
                pass
            va = _max_cos(Xa)
        else:
            # fresh random basin, short anneal
            Xr = rng.normal(size=(n_pts, n_dim))
            Xr /= np.maximum(np.linalg.norm(Xr, axis=1, keepdims=True), 1e-12)
            T0 = float(10.0 ** rng.uniform(-1.5, -1.0))
            T1 = float(10.0 ** rng.uniform(-2.6, -2.2))
            lr0 = float(10.0 ** rng.uniform(-2.2, -1.6))
            steps = int(rng.integers(1500, 3000))
            mom = float(rng.uniform(0.82, 0.92))
            try:
                Xa = _anneal(Xr, steps, T0, T1, lr0, lr0 * 0.15, mom)
                Xa = _lbfgs_smooth(Xa)
            except Exception:
                continue
            va = _max_cos(Xa)
        # LP polish promising only
        if va > best_val + 0.012 and rng.random() > 0.03:
            # keep elite diversity anyway
            if va < max(elite_vals) and len(elites) < 4:
                pass
            else:
                continue
        try:
            Xp = _lp_polish(Xa, seconds=1.5, deadline=deadline - 10.0)
            vp = _max_cos(Xp)
        except Exception:
            continue
        if vp < best_val:
            Xbest = Xp; best_val = vp
            elites.insert(0, Xp.copy()); elite_vals.insert(0, vp)
            elites = elites[:4]; elite_vals = elite_vals[:4]
        elif vp < max(elite_vals) + 1e-9 and len(elites) < 4:
            elites.append(Xp.copy()); elite_vals.append(vp)
        elif len(elites) >= 2 and vp < max(elite_vals):
            bi = int(np.argmax(elite_vals))
            elites[bi] = Xp.copy(); elite_vals[bi] = vp
    try:
        tleft = deadline - time.time() - 1.0
        if tleft > 2.0:
            Xbest = _lp_polish(Xbest, seconds=min(tleft, 10.0), deadline=deadline)
    except Exception:
        pass
    Xbest = Xbest / np.maximum(np.linalg.norm(Xbest, axis=1, keepdims=True), 1e-12)
    Xbest = np.nan_to_num(Xbest, nan=0.0, posinf=0.0, neginf=0.0)
    nr = np.linalg.norm(Xbest, axis=1, keepdims=True)
    bad = (nr.squeeze() < 1e-9)
    if np.any(bad):
        rf = np.random.default_rng(1)
        R = rf.normal(size=(int(np.sum(bad)), n_dim))
        R /= np.maximum(np.linalg.norm(R, axis=1, keepdims=True), 1e-12)
        Xbest[bad] = R
        Xbest = Xbest / np.maximum(np.linalg.norm(Xbest, axis=1, keepdims=True), 1e-12)
    return Xbest
