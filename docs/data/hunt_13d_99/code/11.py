import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))


def lp_polish(X, seconds):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = max_cos(X)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        Gfull = X @ X.T
        c = Gfull[iu]
        m = float(np.max(c))
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        # cap constraints for speed
        if p > 3000:
            # keep worst 3000
            idx = np.argsort(c[sel])[-3000:]
            I, J = I[idx], J[idx]
            cc = c[sel][idx]
            p = 3000
        else:
            cc = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cc, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X


def refine_global(X, steps, T0, T1, s0, s1, mu, rng, noise=0.0):
    n, d = X.shape
    V = np.zeros_like(X)
    logT0, logT1 = np.log(T0), np.log(T1)
    logs0, logs1 = np.log(s0), np.log(s1)
    for it in range(steps):
        frac = it / max(steps - 1, 1)
        T = float(np.exp(logT0 + (logT1 - logT0) * frac))
        st = float(np.exp(logs0 + (logs1 - logs0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        gmax = float(np.max(G))
        # global softmax weights
        np.subtract(G, gmax, out=G)
        np.divide(G, T, out=G)
        # clip to avoid overflow
        np.clip(G, -30.0, 0.0, out=G)
        np.exp(G, out=G)
        np.fill_diagonal(G, 0.0)
        # gradient: weighted sum of neighbours
        Grad = G @ X
        # tangent projection
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Grad -= dot * X
        # normalize scale: divide by mean weight so step size comparable
        # use fixed scaling: Grad magnitude ~ O(1) for active points
        V *= mu
        V += (1.0 - mu) * Grad
        X -= st * V
        if noise > 0:
            amp = noise * (1.0 - frac)
            X += amp * rng.standard_normal(X.shape) * 0.01
            # keep tangent-ish: reproject later via normalize
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def greedy_reinsert(X, idx, rng, n_cand=8000):
    n, d = X.shape
    sidx = list(idx)
    rng.shuffle(sidx)
    # mask of points being replaced
    is_rep = np.zeros(n, dtype=bool)
    is_rep[np.array(idx)] = True
    for ii in sidx:
        # candidates
        cand = rng.standard_normal((n_cand, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        # dots to fixed points + already reinserted
        # build matrix of active points (exclude not-yet-reinserted except ii)
        # simplest: compute full dot then mask columns
        S = cand @ X.T
        # ignore columns that are still stale to-be-replaced (except ii itself)
        # set them to -2 so they don't affect max
        for jj in sidx:
            if jj != ii:
                # if jj not yet processed, its current value is stale; ignore
                # check if already processed: we track processed set
                pass
        # use processed tracking
        # Instead maintain explicit active mask
        # We'll recompute with mask below
        S[:, is_rep] = -2.0
        # but ii's own column is also masked; we want dots to all except stale
        # S max over remaining
        # also need dots among? single candidate only, no self
        # for already reinserted points, include them:
        # is_rep currently True for all to-be-replaced; we cleared them, need to add back processed ones
        # So we need processed set
        # --- redo correctly with processed set (handled by caller logic) ---
        # fallback: use S as is (ignore all rep) -> not ideal for sequential
        # To include already-done ones, caller should have updated is_rep.
        # We update is_rep incrementally: clear ii first? Let's manage outside.
        m = S.max(axis=1)
        best_c = np.argmin(m)
        X[ii] = cand[best_c]
        # mark ii as done (no longer ignored for next)
        is_rep[ii] = False
    return X


def greedy_block(Xbest, k, rng, n_cand=8000):
    n, d = Xbest.shape
    X = Xbest.copy()
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    rowmax = G.max(axis=1)
    # mix of worst and random for diversity
    worst = np.argsort(rowmax)[-k:]
    # occasionally swap one worst for random
    if rng.random() < 0.4:
        worst[rng.integers(0, k)] = rng.integers(0, n)
        worst = np.unique(worst)
    idx = worst.tolist()
    # sequential greedy with active mask
    is_rep = np.zeros(n, dtype=bool)
    is_rep[np.array(idx)] = True
    order = idx.copy()
    rng.shuffle(order)
    for ii in order:
        cand = rng.standard_normal((n_cand, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        S = cand @ X.T
        S[:, is_rep] = -2.0
        # S now includes already reinserted (is_rep False) and fixed points
        m = S.max(axis=1)
        X[ii] = cand[np.argmin(m)]
        is_rep[ii] = False
    return X


def solve():
    t0 = time.time()
    rng = np.random.default_rng(7)
    try:
        Xbest = np.load("best.npy").astype(float)
    except Exception:
        try:
            Xbest = np.load("record.npy").astype(float)
        except Exception:
            Xbest = rng.standard_normal((99, 13))
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    n, d = Xbest.shape
    if (n, d) != (99, 13):
        tmp = rng.standard_normal((99, 13))
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        Xbest = tmp
        n, d = 99, 13
    best_val = max_cos(Xbest)

    trial = 0
    # main exploration, reserve ~30s for final polish
    while time.time() - t0 < 140:
        trial += 1
        mode = trial % 12
        X = Xbest.copy()
        if mode in (0, 1):
            sigma = 0.05 if mode == 0 else 0.09
            X += sigma * rng.standard_normal(X.shape)
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            steps = 2600
            T0, T1 = 0.018, 0.0015
            s0, s1 = 0.05, 0.003
            mu = 0.86
            nz = 0.0
        elif mode == 2:
            sigma = 0.30
            X += sigma * rng.standard_normal(X.shape)
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            steps = 3400
            T0, T1 = 0.030, 0.0020
            s0, s1 = 0.06, 0.004
            mu = 0.84
            nz = 1.0
        elif mode in (3, 4):
            k = 4 if mode == 3 else 9
            X = greedy_block(Xbest, k, rng, n_cand=6000)
            X += 0.015 * rng.standard_normal(X.shape)
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            steps = 3200
            T0, T1 = 0.022, 0.0012
            s0, s1 = 0.055, 0.003
            mu = 0.87
            nz = 0.0
        elif mode == 5:
            k = 16
            X = greedy_block(Xbest, k, rng, n_cand=5000)
            X += 0.02 * rng.standard_normal(X.shape)
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            steps = 3600
            T0, T1 = 0.035, 0.0025
            s0, s1 = 0.07, 0.004
            mu = 0.82
            nz = 0.5
        elif mode == 6:
            sel = rng.choice(n, 22, replace=False)
            X[sel] += 0.25 * rng.standard_normal((22, d))
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            steps = 3000
            T0, T1 = 0.025, 0.0018
            s0, s1 = 0.06, 0.0035
            mu = 0.85
            nz = 0.0
        elif mode == 7:
            # fresh random start (escapes basin entirely)
            X = rng.standard_normal((n, d))
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            steps = 4200
            T0, T1 = 0.045, 0.0020
            s0, s1 = 0.08, 0.004
            mu = 0.82
            nz = 0.0
        elif mode == 8:
            # directed split of overloaded hub: duplicate far point
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            rowmax = G.max(axis=1)
            hub = int(np.argmax(rowmax))
            # find its worst partner
            partner = int(np.argmax(G[hub]))
            # move hub away along tangent from partner + noise
            direction = X[hub] - (X[hub] @ X[partner]) * X[partner]
            nn = np.linalg.norm(direction)
            if nn < 1e-12:
                direction = rng.standard_normal(d)
                nn = np.linalg.norm(direction)
            direction /= nn
            X[hub] += 0.25 * direction + 0.05 * rng.standard_normal(d)
            X[hub] /= np.linalg.norm(X[hub])
            X[partner] -= 0.15 * direction + 0.03 * rng.standard_normal(d)
            X[partner] /= np.linalg.norm(X[partner])
            X += 0.02 * rng.standard_normal(X.shape)
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            steps = 3000
            T0, T1 = 0.020, 0.0012
            s0, s1 = 0.05, 0.003
            mu = 0.88
            nz = 0.0
        elif mode == 9:
            # shake peripheral outliers: points with large |PC| proxy = large norm in top singular vec?
            # approximate by points with smallest mean cosine (most isolated)
            G = X @ X.T
            np.fill_diagonal(G, 0.0)
            means = G.mean(axis=1)
            iso = np.argsort(means)[:12]
            X[iso] += 0.18 * rng.standard_normal((len(iso), d))
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            steps = 3000
            T0, T1 = 0.022, 0.0015
            s0, s1 = 0.06, 0.003
            mu = 0.85
            nz = 0.3
        else:
            # modes 10,11: medium shake with varied temps
            sigma = 0.12 + 0.04 * ((trial // 12) % 3)
            X += sigma * rng.standard_normal(X.shape)
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            steps = 2800 + ((trial // 12) % 3) * 500
            T0 = 0.016 + 0.008 * rng.random()
            T1 = 0.0012 + 0.0015 * rng.random()
            s0, s1 = 0.055, 0.003
            mu = 0.86 if (trial % 2 == 0) else 0.80
            nz = 0.0

        X = refine_global(X, steps, T0, T1, s0, s1, mu, rng, noise=nz)
        v = max_cos(X)
        # decide to LP polish
        need_polish = False
        polish_time = 0.0
        if v < best_val:
            need_polish = True
            polish_time = 3.0
        elif v < best_val + 0.004 and (trial % 4 == 0):
            need_polish = True
            polish_time = 1.0
        elif v < best_val + 0.0015:
            need_polish = True
            polish_time = 1.2
        if need_polish:
            # cap if little time left
            if time.time() - t0 > 138:
                polish_time = min(polish_time, 0.8)
            try:
                Xp = lp_polish(X, polish_time)
            except Exception:
                Xp = X
            vp = max_cos(Xp)
            if vp < best_val:
                Xbest, best_val = Xp, vp
            elif v < best_val:
                Xbest, best_val = X, v
        if time.time() - t0 > 141:
            break

    remain = 172 - (time.time() - t0)
    if remain < 4:
        remain = 4
    if remain > 32:
        remain = 32
    try:
        Xbest = lp_polish(Xbest, remain)
    except Exception:
        pass
    Xbest /= np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest
