import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def _fast_polish(X, iters=1000, beta0=30.0, beta1=4000.0,
                 lr0=0.03, lr1=0.0008, mu=0.6):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    V = np.zeros_like(X)
    lb0, lb1 = np.log(beta0), np.log(beta1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for k in range(iters):
        f = k / max(1, iters - 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        # weights focused on max
        # G-m <= 0
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-30
        F = (W @ X) / rs
        dot = (F * X).sum(axis=1, keepdims=True)
        Gt = F - dot * X
        # clip extreme grads
        # heavy ball
        V = mu * V + (1.0 - mu) * Gt
        X = X - lr * V
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X /= nr
        if not np.all(np.isfinite(X)):
            break
    return X


def _lp_polish(X, seconds, t_end, r0=3e-3):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = r0
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < t_end and radius > 1e-14:
        G = X @ X.T
        c_all = G[iu0]
        m = float(c_all.max())
        tol = max(20.0 * radius, 1e-9)
        sel = c_all > (m - tol)
        if np.count_nonzero(sel) == 0:
            break
        I = iu0[0][sel]
        J = iu0[1][sel]
        c_sel = c_all[sel]
        p = len(I)
        nv = n * d + 1
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], axis=1
        ).ravel()
        rows = np.repeat(np.arange(p), 2 * d + 1)
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix(
                (X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        step = res.x[:-1].reshape(n, d)
        if np.max(np.abs(step)) < 1e-14:
            radius *= 0.5
            if radius < 1e-12:
                break
            continue
        y = X + step
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X


def _best_void(Y, rng, ncand=2500):
    d = Y.shape[1]
    C = rng.normal(size=(ncand, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    vals = (C @ Y.T).max(axis=1)
    bi = int(np.argmin(vals))
    z = C[bi]
    cur = float(vals[bi])
    # quick hill climb on sphere
    for it in range(250):
        frac = it / 249.0
        sigma = 0.15 * (0.005 / 0.15) ** frac
        r = rng.normal(size=d)
        r -= (r @ z) * z
        rn = np.linalg.norm(r)
        if rn < 1e-12:
            continue
        r /= rn
        zt = z + sigma * abs(float(rng.normal())) * r
        zt /= np.linalg.norm(zt)
        v = float((Y @ zt).max())
        if v < cur:
            z, cur = zt, v
    return z


def solve():
    t0 = time.time()
    DEADLINE = t0 + 176.0
    SEARCH_END = t0 + 152.0
    rng = np.random.default_rng(7)
    n, d = 73, 13
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    for c in (xb, xr):
        if c is not None and c.shape == (n, d):
            c = c / np.linalg.norm(c, axis=1, keepdims=True)
            if np.all(np.isfinite(c)):
                cands.append(c)
    if not cands:
        c = rng.normal(size=(n, d))
        c /= np.linalg.norm(c, axis=1, keepdims=True)
        cands.append(c)
    bestX = min(cands, key=_max_cos)
    best = _max_cos(bestX)
    Xcur = bestX.copy()
    cur = best

    Temp0, Temp1 = 0.0012, 0.00008
    trial = 0
    sig_choices = [0.05, 0.12, 0.25, 0.40]
    while time.time() < SEARCH_END and time.time() < DEADLINE - 25:
        trial += 1
        elapsed = time.time() - t0
        frac_g = elapsed / max(1e-9, (SEARCH_END - t0))
        Temp = Temp0 + (Temp1 - Temp0) * frac_g
        r = float(rng.random())
        X = Xcur.copy()
        if r < 0.15:
            # kill worst 1-3 and reinsert into voids
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            per = G.max(axis=1)
            kkill = int(rng.integers(1, 4))
            idx = np.argsort(-per)[:kkill]
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[idx] = False
            Y = X[keep_mask]
            # reinsert one by one
            newpts = []
            for _ in range(kkill):
                Yfull = Y if not newpts else np.vstack([Y] + newpts)
                z = _best_void(Yfull, rng, ncand=2500)
                newpts.append(z[None, :])
            X = np.vstack([Y] + newpts)
            # random order not needed
        elif r < 0.30:
            # coherent cap shake: pick random direction, move cap
            u = rng.normal(size=d)
            u /= np.linalg.norm(u)
            proj = X @ u
            thr = np.quantile(proj, float(rng.uniform(0.5, 0.85)))
            mask = proj >= thr
            if np.count_nonzero(mask) == 0:
                mask[int(rng.integers(0, n))] = True
            sig = float(rng.choice(np.array([0.1, 0.2, 0.35])))
            X[mask] += rng.normal(scale=sig, size=(np.count_nonzero(mask), d))
            X += rng.normal(scale=0.004, size=X.shape)
        else:
            # subset kick biased to worst
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            per = G.max(axis=1)
            order = np.argsort(-per)
            n_sub = int(rng.integers(2, 9))
            sigma_sub = sig_choices[int(rng.integers(0, len(sig_choices)))]
            sigma_all = 0.004 if (trial % 2 == 0) else 0.009
            X += rng.normal(scale=sigma_all, size=X.shape)
            idxset = set(order[:2].tolist())
            # also always kick rattlers (0 contacts)
            cc = np.count_nonzero(G > (cur - 1e-9), axis=1)
            ratt = np.where(cc == 0)[0]
            for rr in ratt:
                idxset.add(int(rr))
            while len(idxset) < n_sub + len(ratt):
                idxset.add(int(rng.integers(0, n)))
            for i in idxset:
                X[i] += rng.normal(scale=sigma_sub, size=d)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        # fast annealed polish, random length
        iters = int(rng.integers(700, 1400))
        # randomize schedule slightly
        b1 = float(rng.choice(np.array([2500.0, 4000.0, 6000.0])))
        l0 = float(rng.choice(np.array([0.02, 0.03, 0.045])))
        Xp = _fast_polish(X, iters=iters, beta0=30.0, beta1=b1,
                          lr0=l0, lr1=0.0008, mu=0.6)
        v = _max_cos(Xp)
        if v < best - 1e-13:
            best, bestX = v, Xp.copy()
            Xcur, cur = Xp.copy(), v
        else:
            # Metropolis move of current walker
            if v < cur or float(rng.random()) < np.exp((cur - v) / max(1e-12, Temp)):
                Xcur, cur = Xp.copy(), v
            # occasional restart from best
            if float(rng.random()) < 0.06:
                Xcur, cur = bestX.copy(), best
        # small time guard
        if time.time() > SEARCH_END:
            break

    # final LP squeeze
    t_left = DEADLINE - time.time() - 1.0
    if t_left > 2.0:
        # polish best; also try polishing walker if close
        bestX = _lp_polish(bestX, seconds=min(t_left * 0.8, 20.0),
                           t_end=DEADLINE - 0.8, r0=3e-3)
        best = _max_cos(bestX)
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    if not np.all(np.isfinite(bestX)):
        bestX = cands[0]
        bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX
