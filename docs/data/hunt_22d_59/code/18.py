import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _normed(A):
    n = np.linalg.norm(A, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return A / n

def _smooth_opt(X, steps, beta0, beta1, lr0, lr1, mu):
    X = X.copy()
    vel = np.zeros_like(X)
    best_val = _max_cos(X)
    best_X = X.copy()
    lb0 = np.log(beta0); lb1 = np.log(beta1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    for s in range(steps):
        frac = s / float(max(1, steps - 1))
        beta = float(np.exp(lb0 + (lb1 - lb0) * frac))
        lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        S = float(W.sum() * 0.5)
        if S < 1e-30:
            continue
        Grad = (W @ X) / S
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        vel = mu * vel + (1.0 - mu) * Gt
        X = X - lr * vel
        X = _normed(X)
        if (s % 100 == 99) or (s == steps - 1):
            v = _max_cos(X)
            if v < best_val:
                best_val = v
                best_X = X.copy()
    return best_X

def _lp_polish(X, max_iters, eps_init, t0, deadline, verbose=False):
    try:
        from scipy.optimize import linprog
    except Exception:
        return X
    N, D = X.shape
    ND = N * D
    eps = float(eps_init)
    iu, ju = np.triu_indices(N, 1)
    for it in range(max_iters):
        if time.time() > deadline:
            break
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        vals_all = G[iu, ju]
        margin = max(0.004, 2.5 * eps)
        # select active
        mask = vals_all >= (m - margin)
        idx = np.where(mask)[0]
        if len(idx) == 0:
            break
        if len(idx) > 1300:
            order = np.argsort(-vals_all[idx])[:1300]
            idx = idx[order]
        P = len(idx)
        vis = iu[idx]
        vjs = ju[idx]
        b_ub = -vals_all[idx]
        # build matrices
        A_ub = np.zeros((P, ND + 1), dtype=np.float64)
        # fill rows
        # loop is fine for P~1k
        for k in range(P):
            i = int(vis[k]); j = int(vjs[k])
            A_ub[k, i*D:(i+1)*D] = X[j]
            A_ub[k, j*D:(j+1)*D] = X[i]
            A_ub[k, ND] = -1.0
        A_eq = np.zeros((N, ND + 1), dtype=np.float64)
        for i in range(N):
            A_eq[i, i*D:(i+1)*D] = X[i]
        b_eq = np.zeros(N, dtype=np.float64)
        c = np.zeros(ND + 1, dtype=np.float64)
        c[ND] = 1.0
        lo_t = m - 0.05
        bounds = [(-eps, eps)] * ND + [(lo_t, m)]
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        except Exception:
            eps *= 0.5
            if eps < 1e-7:
                break
            continue
        if not res.success:
            eps *= 0.5
            if eps < 2e-7:
                break
            continue
        dx = np.asarray(res.x[:ND], dtype=np.float64).reshape(N, D)
        Xt = _normed(X + dx)
        mt = _max_cos(Xt)
        if mt < m - 1e-12:
            X = Xt
            eps = min(eps * 1.25, 0.03)
        else:
            eps *= 0.5
            if eps < 2e-7:
                break
        # if eps got tiny but still jammed, stop this round
        if eps < 5e-7:
            break
    return X

def _find_voids(Xrest, k, rng, n_cand=6000):
    d = Xrest.shape[1]
    R = rng.standard_normal((n_cand, d))
    R = R / np.maximum(np.linalg.norm(R, axis=1, keepdims=True), 1e-12)
    S = R @ Xrest.T
    rowmax = S.max(axis=1)
    order = np.argsort(rowmax)
    picked = []
    for idx in order:
        cand = R[idx]
        ok = True
        for p in picked:
            if float(cand @ p) > 0.75:
                ok = False
                break
        if ok:
            picked.append(cand)
            if len(picked) >= k:
                break
    while len(picked) < k:
        v = rng.standard_normal(d)
        v = v / max(np.linalg.norm(v), 1e-12)
        picked.append(v)
    return np.array(picked)

def solve():
    import time
    import numpy as np
    t0 = time.time()
    rng = np.random.default_rng(0)
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (59, 22) and np.all(np.isfinite(a)):
                a = _normed(a)
                cands.append(a)
        except Exception:
            pass
    if len(cands) == 0:
        best_X = _normed(rng.standard_normal((59, 22)))
    else:
        vals = [_max_cos(c) for c in cands]
        best_X = cands[int(np.argmin(vals))].copy()
    best_val = _max_cos(best_X)

    def tl(limit):
        return (time.time() - t0) < limit

    # Stage 0: LP squeeze of starting point (exact local optimum)
    try:
        Xp = _lp_polish(best_X, max_iters=35, eps_init=0.015, t0=t0, deadline=t0+38.0)
        vp = _max_cos(Xp)
        if vp < best_val:
            best_val = vp; best_X = Xp
        else:
            # keep squeezed version if essentially equal (more jammed -> better base)
            if vp <= best_val + 1e-12:
                best_X = Xp; best_val = vp
    except Exception:
        pass
    # quick smooth polish
    try:
        if tl(50.0):
            Xs = _smooth_opt(best_X, steps=2500, beta0=40.0, beta1=400.0, lr0=0.012, lr1=0.0008, mu=0.9)
            vs = _max_cos(Xs)
            if vs < best_val:
                best_val = vs; best_X = Xs
            # re-squeeze
            Xp = _lp_polish(best_X, max_iters=15, eps_init=0.008, t0=t0, deadline=t0+55.0)
            vp = _max_cos(Xp)
            if vp < best_val:
                best_val = vp; best_X = Xp
    except Exception:
        pass

    sigmas = [0.006, 0.014, 0.028, 0.055, 0.10, 0.18, 0.01, 0.04]
    trial = 0
    while tl(145.0) and trial < 120:
        mode = trial % 10
        sigma = sigmas[trial % len(sigmas)]
        Xt = None
        if mode in (2, 5, 8):
            try:
                G = best_X @ best_X.T
                np.fill_diagonal(G, -2.0)
                rowmax = G.max(axis=1)
                worst = np.argsort(-rowmax)
                k = 2 if mode == 2 else (3 if mode == 5 else 1)
                m = float(G.max())
                deg = (G >= (m - 0.001)).sum(axis=1)
                alt = np.argsort(-(rowmax + 0.002 * deg))
                cand_idx = list(dict.fromkeys(list(worst[:6]) + list(alt[:6])))[:k]
                rep_idx = np.array(cand_idx[:k])
                if len(rep_idx) < k:
                    rep_idx = worst[:k]
                mask = np.ones(59, dtype=bool)
                mask[rep_idx] = False
                Xrest = best_X[mask]
                voids = _find_voids(Xrest, k, rng, n_cand=5000)
                Xt = best_X.copy()
                for ii, ri in enumerate(rep_idx):
                    Xt[ri] = voids[ii]
                Xt = Xt + 0.006 * rng.standard_normal((59, 22))
            except Exception:
                Xt = best_X + sigma * rng.standard_normal((59, 22))
        elif mode == 9:
            Xt = best_X + 0.22 * rng.standard_normal((59, 22))
        else:
            Xt = best_X + sigma * rng.standard_normal((59, 22))
        Xt = _normed(Xt)
        b0 = 22.0 + (trial % 5) * 8.0
        b1 = 220.0 + (trial % 4) * 70.0
        try:
            Xo = _smooth_opt(Xt, steps=2200, beta0=b0, beta1=b1, lr0=0.012, lr1=0.0008, mu=0.9)
            # short LP squeeze to true basin optimum
            Xo = _lp_polish(Xo, max_iters=8, eps_init=0.008, t0=t0, deadline=t0+146.0)
        except Exception:
            trial += 1
            continue
        v = _max_cos(Xo)
        if v < best_val:
            best_val = v
            best_X = Xo.copy()
        trial += 1

    # Final exact polishing
    if tl(160.0):
        try:
            Xf = _lp_polish(best_X, max_iters=30, eps_init=0.006, t0=t0, deadline=t0+168.0)
            vf = _max_cos(Xf)
            if vf < best_val:
                best_val = vf; best_X = Xf
        except Exception:
            pass
    if tl(172.0):
        try:
            Xf = _smooth_opt(best_X, steps=1200, beta0=250.0, beta1=700.0, lr0=0.002, lr1=0.0002, mu=0.9)
            vf = _max_cos(Xf)
            if vf < best_val:
                best_val = vf; best_X = Xf
            Xf = _lp_polish(best_X, max_iters=20, eps_init=0.003, t0=t0, deadline=t0+176.0)
            vf = _max_cos(Xf)
            if vf < best_val:
                best_val = vf; best_X = Xf
        except Exception:
            pass

    best_X = _normed(best_X)
    return best_X
