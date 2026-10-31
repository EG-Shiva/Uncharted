import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def norm_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def contacts_of(X, thr=1e-7):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    mx = float(np.max(G))
    cnt = np.sum(G > mx - thr, axis=1)
    return cnt, mx, G

def smooth_opt(X, steps, b0, b1, lr0, lr1, mu=0.9):
    X = norm_rows(X)
    n = X.shape[0]
    V = np.zeros_like(X)
    eye2 = 2.0 * np.eye(n)
    for k in range(steps):
        f = k / (steps - 1) if steps > 1 else 0.0
        beta = b0 + (b1 - b0) * f
        lr = lr0 + (lr1 - lr0) * f
        G = X @ X.T
        m = float(np.max(G - eye2))
        A = beta * (G - m)
        np.clip(A, -12.0, 0.0, out=A)
        np.fill_diagonal(A, -30.0)
        np.exp(A, out=A)
        Grad = A @ X
        d = np.sum(Grad * X, axis=1, keepdims=True)
        T = Grad - d * X
        s = float(np.mean(np.sqrt(np.sum(T * T, axis=1)))) + 1e-12
        T /= s
        V *= mu
        V += (1.0 - mu) * T
        X -= lr * V
        X = norm_rows(X)
    return X

def lp_polish(x, budget):
    x = norm_rows(x.copy())
    n, d = x.shape
    best = max_cos(x)
    best_x = x.copy()
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < budget and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(np.max(c))
        sel = c > m - max(20.0 * radius, 1e-9)
        I = iu[0][sel]
        J = iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > 3000:
            # keep tightest only to bound LP size
            idx = np.argsort(-c[sel])[:3000]
            I = I[idx]
            J = J[idx]
            p = len(I)
        csel = (x[I] * x[J]).sum(axis=1)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        ci = (I[:, None] * d + np.arange(d)).ravel()
        cj = (J[:, None] * d + np.arange(d)).ravel()
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        cols[0::(2 * d + 1)] = 0  # placeholder, rebuilt below
        # build cols vectorized
        blocks = np.concatenate([I[:, None] * d + np.arange(d),
                                 J[:, None] * d + np.arange(d),
                                 np.full((p, 1), n * d)], axis=1).ravel()
        cols = blocks
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = norm_rows(y)
        v = max_cos(y)
        if v < best:
            x, best, best_x = y, v, y.copy()
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return best_x

def hole_fill_positions(Xfixed, k, rng, nprobe=4000):
    # greedily pick k points in largest holes w.r.t fixed set + each other
    d = Xfixed.shape[1]
    chosen = []
    P = rng.standard_normal((nprobe, d))
    P = norm_rows(P)
    # score vs fixed
    if len(Xfixed) > 0:
        S = P @ Xfixed.T
        smax = S.max(axis=1)
    else:
        smax = np.zeros(nprobe)
    # add slight noise to diversify
    order = np.argsort(smax)
    # take from low end with randomization
    cand_idx = order[:max(500, k * 50)]
    # greedy farthest among candidates
    cand = P[cand_idx]
    # shuffle
    perm = rng.permutation(len(cand))
    cand = cand[perm]
    sel = []
    for v in cand:
        if len(sel) >= k:
            break
        if len(sel) == 0:
            sel.append(v)
        else:
            S2 = np.array(sel) @ v
            if np.max(S2) < 0.9 or len(sel) < 3:
                # check also against fixed: already low since cand filtered
                sel.append(v)
            else:
                # too close to already chosen, skip with prob
                if rng.random() < 0.2:
                    sel.append(v)
                continue
    while len(sel) < k:
        v = rng.standard_normal(d)
        v /= np.linalg.norm(v)
        sel.append(v)
    return np.array(sel)

def solve():
    t0 = time.time()
    HARD = 176.0
    n, d = 153, 4
    rng = np.random.default_rng(777123)
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            v = np.load(fn).astype(float)
            if v.shape == (n, d):
                v = norm_rows(v)
                c = max_cos(v)
                if c < best_val:
                    best_val = c
                    best = v.copy()
        except Exception:
            pass
    if best is None:
        best = norm_rows(rng.standard_normal((n, d)))
        best_val = max_cos(best)
    base_best = best.copy()
    # quick LP touch on base to get contact structure (cheap)
    # main basin-hopping with block surgery
    trial = 0
    # block sizes cycle: small to very large to force topology change
    block_cycle = [3, 6, 10, 16, 24, 40, 8, 12, 20, 60]
    while time.time() - t0 < 132.0 and trial < 80:
        trial += 1
        elapsed = time.time() - t0
        remain_global = HARD - elapsed
        if remain_global < 20.0:
            break
        mode = trial % 10
        if mode == 0:
            # fresh random start with two-stage smoothing (new basin)
            X = norm_rows(rng.standard_normal((n, d)))
            X = smooth_opt(X, 1200, 5.0, 60.0, 0.03, 0.005, mu=0.85)
            X = smooth_opt(X, 1200, 60.0, 320.0, 0.02, 0.002, mu=0.9)
            v = max_cos(X)
            if v < best_val + 0.025:
                rb = min(6.0, HARD - (time.time() - t0) - 16.0)
                if rb > 1.0:
                    Xp = lp_polish(X, rb)
                    vp = max_cos(Xp)
                    if vp < best_val - 1e-12:
                        best, best_val = Xp, vp
                    elif vp < best_val + 0.005 and rng.random() < 0.1:
                        base_best = Xp.copy()
                else:
                    if v < best_val - 1e-12:
                        best, best_val = X, v
            continue
        # otherwise block surgery on best (or second pool)
        src = best if (trial % 3 != 2) else base_best
        try:
            cnt, mx, G = contacts_of(src, thr=1e-7)
        except Exception:
            cnt = np.ones(n)
        k = block_cycle[(trial - 1) % len(block_cycle)]
        # choose victims: rattlers first, then hubs, then random neighbors
        ratt = np.where(cnt == 0)[0].tolist()
        order_hi = np.argsort(-cnt).tolist()
        order_lo = np.argsort(cnt).tolist()
        victims = []
        for r in ratt:
            if len(victims) < k:
                victims.append(int(r))
        # alternate hub-heavy and lo-heavy to balance load
        if trial % 2 == 0:
            for idx in order_hi:
                if len(victims) >= k:
                    break
                if idx not in victims:
                    victims.append(int(idx))
        else:
            # random sample biased to low contacts + random
            for idx in order_lo[:40]:
                if len(victims) >= k:
                    break
                if idx not in victims and rng.random() < 0.5:
                    victims.append(int(idx))
        while len(victims) < k:
            r = int(rng.integers(0, n))
            if r not in victims:
                victims.append(r)
        victims = np.array(victims[:k], dtype=int)
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[victims] = False
        Xfixed = src[keep_mask]
        # hole-guided new positions
        newpts = hole_fill_positions(Xfixed, k, rng, nprobe=6000)
        # add small noise so not exactly on probe
        newpts = newpts + 0.02 * rng.standard_normal(newpts.shape)
        newpts = norm_rows(newpts)
        X = src.copy()
        X[victims] = newpts
        # small global jitter to break symmetry
        sigma = 0.008 if k < 15 else 0.003
        X = X + sigma * rng.standard_normal(X.shape)
        X = norm_rows(X)
        # two-stage smooth: coarse to establish new basin, fine to converge
        if k >= 20:
            X = smooth_opt(X, 900, 8.0, 80.0, 0.025, 0.004, mu=0.85)
            X = smooth_opt(X, 1100, 80.0, 350.0, 0.018, 0.002, mu=0.9)
        else:
            X = smooth_opt(X, 800, 15.0, 120.0, 0.022, 0.004, mu=0.88)
            X = smooth_opt(X, 1000, 120.0, 350.0, 0.018, 0.002, mu=0.9)
        v = max_cos(X)
        if v < best_val - 1e-12:
            # immediate improvement without LP
            best, best_val = X.copy(), v
            base_best = X.copy()
            # quick snap
            rb = min(5.0, HARD - (time.time() - t0) - 14.0)
            if rb > 1.0:
                Xp = lp_polish(X, rb)
                vp = max_cos(Xp)
                if vp < best_val - 1e-12:
                    best, best_val = Xp, vp
            continue
        # decide to LP-polish: allow slightly worse to discover new balanced basin
        # check balance: count rattlers
        try:
            cnt2, mx2, _ = contacts_of(X, thr=1e-7)
            nrat = int(np.sum(cnt2 == 0))
        except Exception:
            nrat = 99
        thresh = best_val + 0.015
        if nrat == 0:
            thresh = best_val + 0.025  # favor balanced topologies
        if v < thresh:
            rb = HARD - (time.time() - t0) - 14.0
            if rb < 1.0:
                continue
            if v < best_val + 0.002 or nrat == 0:
                budget = min(7.0, rb)
            else:
                budget = min(3.0, rb)
            Xp = lp_polish(X, budget)
            vp = max_cos(Xp)
            if vp < best_val - 1e-12:
                best, best_val = Xp, vp
                base_best = Xp.copy()
            elif vp < best_val + 0.008:
                # keep diverse pool: if balanced, store as alternate source
                try:
                    cN, _, _ = contacts_of(Xp, thr=1e-7)
                    if int(np.sum(cN == 0)) == 0 and rng.random() < 0.4:
                        base_best = Xp.copy()
                except Exception:
                    pass
    remain = HARD - (time.time() - t0) - 1.0
    if remain > 2.0:
        try:
            best = lp_polish(best, min(remain, 30.0))
        except Exception:
            pass
    best = norm_rows(best)
    return best
