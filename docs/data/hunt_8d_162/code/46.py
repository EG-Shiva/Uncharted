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

def _contacts(X, tol=1e-9):
    n = X.shape[0]
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    mask = G > m - tol
    cnt = mask.sum(axis=1)
    rowmax = G.max(axis=1)
    return m, cnt, rowmax, G

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
        Gf = X @ X.T
        cf = Gf[iu0]
        m = float(cf.max())
        sel = cf > m - max(20 * radius, 1e-9)
        I = iu0[0][sel]; J = iu0[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 4000:
            idx = np.argpartition(-cf[sel], 4000)[:4000]
            I = I[idx]; J = J[idx]
            c = cf[sel][idx]
            p = 4000
        else:
            c = cf[sel]
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
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
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

def _find_holes(Xc, k, rng, n_samp=50000, sep=0.65):
    d = Xc.shape[1]
    C = _norm(rng.randn(n_samp, d))
    S = C @ Xc.T
    scores = S.max(axis=1)
    order = np.argsort(scores)
    picked = []
    picked_idx = []
    for idx in order:
        if len(picked) >= k:
            break
        v = C[idx]
        ok = True
        for q in picked:
            if float(v @ q) > sep:
                ok = False
                break
        if ok:
            picked.append(v.copy())
            picked_idx.append(idx)
    while len(picked) < k:
        v = rng.randn(d); v /= np.linalg.norm(v)
        # ensure separation loosely
        ok = True
        for q in picked:
            if float(v @ q) > 0.85:
                ok = False; break
        if ok:
            picked.append(v)
        else:
            picked.append(v)
            # still append to avoid infinite loop
    return np.array(picked)

def _refine_hole(h, Xc, steps=250):
    h = h / max(np.linalg.norm(h), 1e-12)
    for t in range(steps):
        f = t / max(1, steps - 1)
        T = 0.02 * ((0.002 / 0.02) ** f)
        lr = 0.08 * (1.0 - 0.7 * f)
        g = h @ Xc.T
        gm = float(g.max())
        w = np.exp((g - gm) / T)
        w /= w.sum()
        grad = w @ Xc  # (d,)
        # tangent projection
        grad = grad - float(grad @ h) * h
        h = h - lr * grad
        h /= max(np.linalg.norm(h), 1e-12)
    return h

def _joint_lbfgs(Xf_init, Xc, temps=(0.01, 0.003, 0.001), iters=150):
    k, d = Xf_init.shape
    m = Xc.shape[0]
    Z = Xf_init.copy()
    # ensure nonzero
    for T in temps:
        z0 = Z.ravel().copy()
        def fun(z):
            Zv = z.reshape(k, d)
            nr = np.sqrt((Zv * Zv).sum(axis=1, keepdims=True))
            nr = np.maximum(nr, 1e-12)
            Xf = Zv / nr
            # values
            # free-frozen
            Dfc = Xf @ Xc.T  # k x m
            # free-free (i<j)
            # build flat list for LSE
            vmax = float(Dfc.max())
            if k > 1:
                Dff = Xf @ Xf.T
                # mask upper
                iu = np.triu_indices(k, 1)
                ff = Dff[iu]
                fmax = float(ff.max()) if ff.size else -1e9
                vmax = max(vmax, fmax)
            else:
                ff = None
            # sum exp
            s = float(np.exp((Dfc - vmax) / T).sum())
            if ff is not None and ff.size:
                s += float(np.exp((ff - vmax) / T).sum())
            L = vmax + T * np.log(s + 1e-300)
            # weights
            Wfc = np.exp((Dfc - vmax) / T) / (s + 1e-300)
            Gxf = Wfc @ Xc  # k x d
            if ff is not None and ff.size:
                Wff = np.exp((ff - vmax) / T) / (s + 1e-300)
                # scatter to pairs
                iu0, iu1 = np.triu_indices(k, 1)
                # accumulate
                # use bincount style loop (k<=10 so loop fine)
                for t_, a, b in zip(Wff, iu0, iu1):
                    Gxf[a] += t_ * Xf[b]
                    Gxf[b] += t_ * Xf[a]
            # chain to Z
            dot = (Gxf * Xf).sum(axis=1, keepdims=True)
            Gz = (Gxf - dot * Xf) / nr
            return L, Gz.ravel()
        try:
            res = minimize(fun, z0, jac=True, method='L-BFGS-B',
                           options={'maxiter': iters, 'ftol': 1e-20, 'gtol': 1e-12, 'maxfun': 10000})
            Z = res.x.reshape(k, d)
        except Exception:
            break
    Zn = np.sqrt((Z * Z).sum(axis=1, keepdims=True))
    Zn = np.maximum(Zn, 1e-12)
    return Z / Zn

def _smooth_all(X, steps=200, beta0=60.0, beta1=400.0, lr=0.008):
    X = _norm(X.copy())
    n, d = X.shape
    for t in range(steps):
        f = t / max(1, steps - 1)
        beta = beta0 * ((beta1 / beta0) ** f)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        gm = float(G.max())
        E = np.exp(beta * (G - gm))
        np.fill_diagonal(E, 0.0)
        s = float(E.sum())
        if not np.isfinite(s) or s <= 0:
            break
        P = E / s
        Grad = 2.0 * (P @ X)
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        mx = float(np.sqrt((Gt * Gt).sum(axis=1)).max())
        if mx < 1e-14 or not np.isfinite(mx):
            break
        X = X - (lr / (mx + 1e-12)) * Gt * min(mx, 1.0)
        X = _norm(X)
    return X

def solve():
    t0 = time.time()
    BUDGET = 172.0
    deadline = t0 + BUDGET
    stop_new = t0 + 150.0
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
    X_best = Xb.copy(); s_best = float(sb)
    n, d = 162, 8
    ks = [3, 4, 6, 8, 10]
    trial = 0
    # precompute nothing; loop
    while time.time() < stop_new:
        if time.time() > deadline - 12:
            break
        trial += 1
        k = ks[(trial - 1) % len(ks)]
        mode = (trial - 1) % 3  # 0: rattler+hubs, 1: cluster, 2: random mix
        # contact analysis on current best
        try:
            m, cnt, rowmax, Gf = _contacts(X_best, tol=1e-9)
        except Exception:
            m = s_best; cnt = np.ones(n); rowmax = np.ones(n) * m
        rattler = int(np.argmin(cnt))
        order_high = np.argsort(-cnt)
        hubs = [i for i in order_high if cnt[i] >= 20]
        if len(hubs) == 0:
            hubs = list(order_high[:8])
        tied = np.where(rowmax > m - 1e-9)[0]
        if len(tied) == 0:
            tied = np.arange(n)
        del_set = set([rattler])
        if mode == 0:
            for h in order_high:
                if len(del_set) >= k:
                    break
                del_set.add(int(h))
        elif mode == 1:
            # spatial/contact cluster around a seed (cycle hubs)
            seed = int(hubs[(trial - 1) % len(hubs)])
            dots = (X_best @ X_best[seed])
            dots[seed] = 2.0
            dots[rattler] = 2.0  # keep rattler separate, add later
            nn = np.argsort(-dots)
            for idx in nn:
                if len(del_set) >= k:
                    break
                del_set.add(int(idx))
            # if seed==rattler already included
        else:
            # random mixture: rattler + sample from tied/high + global
            pool = list(set(list(tied) + hubs))
            rng.shuffle(pool)
            for idx in pool:
                if len(del_set) >= k:
                    break
                if idx == rattler:
                    continue
                # random skip for diversity
                del_set.add(int(idx))
            while len(del_set) < k:
                c = int(rng.randint(0, n))
                del_set.add(c)
        del_idx = np.array(sorted(list(del_set)), dtype=int)
        # adjust k if duplicates caused smaller (should be k)
        if len(del_idx) < k:
            extra = [i for i in range(n) if i not in del_set]
            rng.shuffle(extra)
            need = k - len(del_idx)
            del_idx = np.array(sorted(list(del_idx) + extra[:need]), dtype=int)
            k = len(del_idx)
        keep_mask = np.ones(n, dtype=bool); keep_mask[del_idx] = False
        Xc = X_best[keep_mask]
        # holes
        try:
            holes = _find_holes(Xc, k, rng, n_samp=50000, sep=0.65)
        except Exception:
            continue
        # local minimisation of each hole vs frozen
        H = np.zeros((k, d))
        for a in range(k):
            try:
                H[a] = _refine_hole(holes[a], Xc, steps=200)
            except Exception:
                H[a] = holes[a]
        # joint optimisation of free points
        try:
            Xf_opt = _joint_lbfgs(H, Xc, temps=(0.01, 0.003, 0.001), iters=150)
        except Exception:
            continue
        X_trial = np.zeros((n, d))
        X_trial[keep_mask] = Xc
        # map del_idx order to Xf rows (keep correspondence arbitrary)
        X_trial[~keep_mask] = Xf_opt
        X_trial = _norm(X_trial)
        # short smooth refinement unfreezing all
        try:
            X_trial = _smooth_all(X_trial, steps=150, beta0=80.0, beta1=500.0, lr=0.006)
        except Exception:
            pass
        s_pre = _max_cos(X_trial)
        # LP polish 3-5 s (use 3.5, less if deadline near)
        remain = deadline - time.time()
        if remain < 6:
            break
        lp_sec = 3.5
        if remain - lp_sec < 4:
            lp_sec = max(1.0, remain - 4)
        try:
            X_pol = _lp_polish(X_trial, seconds=lp_sec, t_deadline=deadline - 2)
        except Exception:
            continue
        s_pol = _max_cos(X_pol)
        if s_pol < s_best and np.all(np.isfinite(X_pol)):
            # strict improvement (any)
            X_best = _norm(X_pol); s_best = float(s_pol)
        # also allow tiny side-step to diversify? keep best only; continue
        # vary rng deterministically already
    return _norm(X_best)
