import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def normed(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[~np.isfinite(n) | (n == 0)] = 1.0
    return X / n


def smooth_descent(X_init, steps, T0, T1, lr0, lr1, mu=0.85, check_every=200, t_end=1e18):
    X = X_init.copy()
    V = np.zeros_like(X)
    bestV = max_cos(X)
    bestX = X.copy()
    for s in range(steps):
        if s % 256 == 0 and time.time() > t_end:
            break
        frac = s / max(1, steps - 1)
        T = T0 * (T1 / T0) ** frac if T0 > 0 and T1 > 0 else T0
        lr = lr0 * (lr1 / lr0) ** frac if lr0 > 0 and lr1 > 0 else lr0
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        S = E.sum() * 0.5
        if not np.isfinite(S) or S <= 0:
            break
        W = E / S
        Grad = W @ X
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        V = mu * V + Gt
        maxn = np.linalg.norm(V, axis=1).max()
        if maxn > 1e-15 and np.isfinite(maxn):
            step = V / maxn * lr
        else:
            step = V * 0.0
        X = X - step
        X = normed(X)
        if (s % check_every == 0) or (s == steps - 1):
            v = max_cos(X)
            if v < bestV:
                bestV = v
                bestX = X.copy()
    return bestX, bestV


def lp_polish(x, seconds, t_end_global=1e18):
    x = normed(x.copy())
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < t_end_global:
        if radius < 1e-13:
            break
        Gfull = x @ x.T
        c = Gfull[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        if np.sum(sel) > 3000:
            # keep only tightest 3000 to bound LP size
            idx = np.argsort(-c)
            keep = np.zeros_like(sel, dtype=bool)
            # need sel among top; take top 3000 among sel
            cnt = 0
            for k in idx:
                if sel[k]:
                    keep[k] = True
                    cnt += 1
                    if cnt >= 3000:
                        break
            sel = keep
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                          options={"time_limit": max(0.5, min(5.0, seconds))})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = normed(y)
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best


def greedy_insert_vectors(X_fixed, k, rng, n_cand=8000):
    # return k vectors greedily minimizing max dot to fixed set + each other
    d = X_fixed.shape[1]
    cands = normed(rng.normal(size=(n_cand, d)))
    # scores vs fixed
    G = cands @ X_fixed.T
    best_idx = []
    chosen = []
    cur_max = G.max(axis=1)  # (n_cand,)
    for _ in range(k):
        if chosen:
            # update cur_max with dots to already chosen
            pass
        j = int(np.argmin(cur_max))
        # tie-break randomly among near-ties: pick among top 20 best with random
        # deterministic via rng
        order = np.argsort(cur_max)
        pool = order[:min(32, n_cand)]
        j = int(pool[rng.integers(0, len(pool))]) if len(pool) > 1 else j
        best_idx.append(j)
        chosen.append(cands[j].copy())
        # update cur_max
        dots = cands @ chosen[-1]
        cur_max = np.maximum(cur_max, dots)
    return np.array(chosen)


def solve():
    t0 = time.time()
    TIME_LIMIT = 171.0
    t_end = t0 + TIME_LIMIT
    rng = np.random.default_rng(7)
    n, d = 71, 12
    try:
        xb = normed(np.asarray(np.load("best.npy"), dtype=np.float64))
    except Exception:
        xb = None
    try:
        xr = normed(np.asarray(np.load("record.npy"), dtype=np.float64))
    except Exception:
        xr = None
    if xb is None or xb.shape != (71, 12):
        xb = normed(rng.normal(size=(71, 12)))
    if xr is not None and xr.shape == (71, 12):
        if max_cos(xr) < max_cos(xb):
            xb = xr.copy()
    best = normed(xb.copy())
    bestV = max_cos(best)

    def tight_order(X, v, tol=1e-4):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        cnt = np.sum(G > v - tol, axis=1)
        order = np.argsort(-cnt)
        return order, cnt, G

    # initial exact squeeze of starting point
    if time.time() < t_end - 5:
        try:
            Xp, Vp = lp_polish(best, seconds=3.0, t_end_global=t_end - 1)
            if Vp < bestV:
                best, bestV = normed(Xp), Vp
        except Exception:
            pass
    # quick anneal to be truly at bottom
    if time.time() < t_end - 5:
        Xp, Vp = smooth_descent(best, steps=500, T0=0.006, T1=0.001, lr0=0.006, lr1=0.0006,
                                mu=0.88, check_every=100, t_end=t_end - 1)
        if Vp < bestV:
            best, bestV = Xp, Vp
            try:
                Xp2, Vp2 = lp_polish(best, seconds=2.0, t_end_global=t_end - 1)
                if Vp2 < bestV:
                    best, bestV = normed(Xp2), Vp2
            except Exception:
                pass

    trial = 0
    while time.time() < t_end - 2.5:
        trial += 1
        remaining = t_end - time.time()
        if remaining < 3:
            break
        if remaining < 25:
            steps = 900
            n_cand = 4000
        elif remaining < 70:
            steps = 1500
            n_cand = 6000
        else:
            steps = 1800
            n_cand = 8000
        order, cnt, Gmat = tight_order(best, bestV, tol=5e-3)
        hub = int(order[0])
        # choose move
        r = rng.random()
        X0 = None
        T0_ = 0.012; T1_ = 0.001; lr0_ = 0.016; lr1_ = 0.0005; mu = 0.88
        if r < 0.35:
            # kill hub + greedy optimal reinsertion (topology change)
            nk = 1 if rng.random() < 0.6 else 2
            kill = [hub]
            if nk == 2:
                # second victim: among top-10 worst or random
                if rng.random() < 0.7:
                    kill.append(int(order[int(rng.integers(1, min(10, n)))]))
                else:
                    kill.append(int(rng.integers(0, n)))
                kill = sorted(set(kill))
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[kill] = False
            X_fixed = best[keep_mask]
            ins = greedy_insert_vectors(X_fixed, len(kill), rng, n_cand=n_cand)
            X0 = best.copy()
            for ii, kidx in enumerate(kill):
                X0[kidx] = ins[ii]
            # tiny jitter to others to break symmetry
            X0 = X0 + 0.0015 * rng.normal(size=(n, d))
            X0 = normed(X0)
            T0_, T1_ = 0.014, 0.0009
            lr0_, lr1_ = 0.018, 0.0005
            mu = 0.88 if rng.random() < 0.6 else 0.92
        elif r < 0.55:
            # block kill 3-4 around hub neighbourhood
            G = Gmat
            neigh = np.where(G[hub] > bestV - 0.01)[0]
            if len(neigh) < 2:
                neigh = order[1:6]
            nk = int(rng.integers(2, 5))  # 2-4
            kill = [hub]
            # add neighbors
            perm = rng.permutation(np.asarray(neigh))
            for q in perm:
                if int(q) not in kill:
                    kill.append(int(q))
                if len(kill) >= nk:
                    break
            while len(kill) < nk:
                c = int(rng.integers(0, n))
                if c not in kill:
                    kill.append(c)
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[kill] = False
            X_fixed = best[keep_mask]
            ins = greedy_insert_vectors(X_fixed, len(kill), rng, n_cand=n_cand)
            X0 = best.copy()
            for ii, kidx in enumerate(kill):
                X0[kidx] = ins[ii]
            X0 = X0 + 0.002 * rng.normal(size=(n, d))
            X0 = normed(X0)
            T0_, T1_ = 0.016, 0.0012
            lr0_, lr1_ = 0.020, 0.0006
            mu = 0.90
        elif r < 0.75:
            # global jitter log-uniform
            sigma = 10.0 ** rng.uniform(-3.0, -1.0)
            X0 = best + sigma * rng.normal(size=(n, d))
            X0 = normed(X0)
            T0_, T1_ = 0.011, 0.0010
            lr0_, lr1_ = 0.014, 0.0005
            mu = 0.85 if rng.random() < 0.7 else 0.92
        else:
            # targeted kick of hub + its contacts
            sigma = 10.0 ** rng.uniform(-2.3, -0.8)
            X0 = best.copy()
            X0[hub] = X0[hub] + sigma * rng.normal(size=(d,))
            G = Gmat
            neigh = np.where(G[hub] > bestV - 0.012)[0]
            for j in neigh:
                if rng.random() < 0.5:
                    X0[int(j)] = X0[int(j)] + 0.006 * rng.normal(size=(d,))
            # occasionally kick second hub
            if rng.random() < 0.3:
                h2 = int(order[1])
                X0[h2] = X0[h2] + sigma * 0.7 * rng.normal(size=(d,))
            X0 = normed(X0)
            T0_, T1_ = 0.010, 0.0008
            lr0_, lr1_ = 0.012, 0.0004
            mu = 0.88
        # anneal
        Xp, Vp = smooth_descent(X0, steps=steps, T0=T0_, T1=T1_, lr0=lr0_, lr1=lr1_,
                                mu=mu, check_every=300, t_end=t_end - 0.8)
        # micro squeeze if promising
        if Vp < bestV + 0.0025 and time.time() < t_end - 3:
            Xq, Vq = smooth_descent(Xp, steps=350, T0=0.0011, T1=0.00035,
                                    lr0=0.003, lr1=0.0003, mu=0.9,
                                    check_every=120, t_end=t_end - 0.8)
            if Vq < Vp:
                Xp, Vp = Xq, Vq
        # exact LP check if close to best (topology basin verification)
        if Vp < bestV + 0.001 and time.time() < t_end - 3.5:
            # short LP polish to get true basin value
            try:
                Xr_, Vr_ = lp_polish(Xp, seconds=min(2.0, max(0.6, (t_end - time.time()) / 8)), t_end_global=t_end - 1)
                if Vr_ < Vp:
                    Xp, Vp = Xr_, Vr_
            except Exception:
                pass
        if Vp < bestV - 1e-12:
            best, bestV = normed(Xp), Vp
    # final exact polish with remaining time
    rem = t_end - time.time()
    if rem > 1.2:
        try:
            Xf, Vf = lp_polish(best, seconds=min(rem - 0.4, 12.0), t_end_global=t_end - 0.3)
            if Vf < bestV:
                best, bestV = normed(Xf), Vf
        except Exception:
            pass
        # one last tiny anneal + polish alternating if time left
        rem2 = t_end - time.time()
        if rem2 > 3:
            Xq, Vq = smooth_descent(best, steps=400, T0=0.0009, T1=0.0003, lr0=0.002, lr1=0.0002,
                                    mu=0.9, check_every=100, t_end=t_end - 0.5)
            if Vq < bestV:
                best, bestV = Xq, Vq
                try:
                    rem3 = t_end - time.time()
                    if rem3 > 1:
                        Xf2, Vf2 = lp_polish(best, seconds=rem3 - 0.4, t_end_global=t_end - 0.2)
                        if Vf2 < bestV:
                            best, bestV = normed(Xf2), Vf2
                except Exception:
                    pass
    return normed(best)
