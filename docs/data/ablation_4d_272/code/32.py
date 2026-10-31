import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def row_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1), float(G.max())


def lp_polish_sec(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        m = float(c_all.max())
        thresh = m - max(20 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 5000:
            idx = np.where(sel)[0]
            order = np.argpartition(-c_all[idx], 5000 - 1)[:5000]
            keep = np.zeros_like(sel, dtype=bool)
            keep[idx[order]] = True
            sel = keep
        c = c_all[sel]
        I = iu0[0][sel]
        J = iu0[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def smooth_gd(x, steps, a0, a1, s0, s1, mu=0.8):
    n, d = x.shape
    vel = np.zeros_like(x)
    for it in range(steps):
        frac = it / max(steps - 1, 1)
        alpha = a0 * (a1 / a0) ** frac if a0 != a1 else a0
        smax = s0 * (s1 / s0) ** frac if s0 != s1 else s0
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        vel = mu * vel + gt
        nrm = np.linalg.norm(vel, axis=1).max()
        if nrm > 1e-14:
            x = x - (smax / nrm) * vel
            x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def smooth_gd_active(x, active, steps, a0, a1, s0, s1, mu=0.8):
    active = np.asarray(active, dtype=int)
    vel = np.zeros((len(active), x.shape[1]))
    for it in range(steps):
        frac = it / max(steps - 1, 1)
        alpha = a0 * (a1 / a0) ** frac if a0 != a1 else a0
        smax = s0 * (s1 / s0) ** frac if s0 != s1 else s0
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        Xa = x[active]
        grad = W[np.ix_(active, np.arange(x.shape[0]))] @ x
        dot = np.sum(grad * Xa, axis=1, keepdims=True)
        gt = grad - dot * Xa
        vel = mu * vel + gt
        nrm = np.linalg.norm(vel, axis=1).max() if len(vel) else 0
        if nrm > 1e-14:
            Xa2 = Xa - (smax / nrm) * vel
            Xa2 /= np.linalg.norm(Xa2, axis=1, keepdims=True)
            x[active] = Xa2
    return x


def refine_single(Y, c0, steps=120, s=0.03):
    c = c0 / np.linalg.norm(c0)
    vel = np.zeros(4)
    for _ in range(steps):
        dots = Y @ c
        m = dots.max()
        sel = dots > m - 2e-3
        if np.count_nonzero(sel):
            g = Y[sel].sum(axis=0)
        else:
            g = Y[np.argmax(dots)]
        gt = g - np.dot(g, c) * c
        nn = np.linalg.norm(gt)
        if nn < 1e-14:
            break
        gt /= nn
        vel = 0.5 * vel + gt
        vn = np.linalg.norm(vel)
        if vn > 1e-14:
            c = c - (s / vn) * vel
            c /= np.linalg.norm(c)
            s *= 0.995
    return c / np.linalg.norm(c)


def largest_hole_insert(Y, rng, n_samples=8000):
    d = Y.shape[1]
    C = rng.standard_normal((n_samples, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    best_c = None
    best_s = 2.0
    bs = 3000
    for a in range(0, n_samples, bs):
        Cb = C[a:a + bs]
        S = Cb @ Y.T
        sm = S.max(axis=1)
        k2 = int(np.argmin(sm))
        if sm[k2] < best_s:
            best_s = sm[k2]
            best_c = Cb[k2].copy()
    best_c = refine_single(Y, best_c, steps=100, s=0.03)
    return best_c


def choose_delete(x, k, rng):
    n = x.shape[0]
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    contact = np.count_nonzero(G > m - 1e-3, axis=1)
    order = np.argsort(-contact, kind="stable")
    n_high = k // 2
    n_rand = k - n_high
    high_set = list(order[:max(n_high * 2, n_high + 4)])
    rng.shuffle(high_set)
    chosen = set(high_set[:n_high])
    while len(chosen) < k:
        r = int(rng.integers(n))
        chosen.add(r)
    keep = np.array([i for i in range(n) if i not in chosen], dtype=int)
    return x[keep].copy()


def prune_one(x):
    # remove point whose removal most reduces max
    n = x.shape[0]
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    # pairs near max
    ii, jj = np.where(G > m - 1e-12)
    involved = np.unique(np.concatenate([ii, jj])) if len(ii) else np.array([np.argmax(G.max(axis=1))])
    # if many involved, restrict to worst rows
    if len(involved) > 25:
        rm = G.max(axis=1)
        involved = involved[np.argsort(-rm[involved])[:25]]
    best_i = -1
    best_v = 2.0
    for i in involved:
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        Gm = G[np.ix_(mask, mask)]
        np.fill_diagonal(Gm, -2.0)
        v = float(Gm.max())
        if v < best_v:
            best_v = v
            best_i = int(i)
    if best_i < 0:
        rm = G.max(axis=1)
        best_i = int(np.argmax(rm))
    mask = np.ones(n, dtype=bool)
    mask[best_i] = False
    return x[mask].copy()


def solve():
    t0 = time.time()
    BUDGET = 172.0
    rng = np.random.default_rng(20261031)
    n, d = 272, 4
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
        r = rng.standard_normal((n, d))
        r /= np.linalg.norm(r, axis=1, keepdims=True)
        cands.append(r)
    best_x = None
    best_v = 2.0
    for c in cands:
        v = max_cos(c)
        if v < best_v:
            best_v = v
            best_x = c.copy()
    base_x = best_x.copy()
    base_v = float(best_v)

    def remain():
        return BUDGET - (time.time() - t0)

    # ---------- Path A: shrink then grow ----------
    # try two shrink sizes if time allows
    for K in (30, 22):
        if remain() < 55:
            break
        try:
            small = choose_delete(base_x if best_x is None else best_x, K, rng)
        except Exception:
            continue
        # aggressive optimisation of smaller code
        try:
            small = smooth_gd(small, steps=350, a0=50.0, a1=600.0, s0=0.02, s1=0.002, mu=0.8)
            if remain() < 40:
                pass
            else:
                small = lp_polish_sec(small, min(3.0, max(1.0, remain() * 0.08)))
                small = smooth_gd(small, steps=200, a0=80.0, a1=800.0, s0=0.01, s1=0.001, mu=0.8)
        except Exception:
            continue
        cur = small
        # grow back one at a time
        for step in range(K):
            if remain() < 28:
                break
            try:
                c_new = largest_hole_insert(cur, rng, n_samples=7000)
                cur = np.vstack([cur, c_new[None, :]])
                # anneal only new point + 20 nearest neighbours
                new_idx = cur.shape[0] - 1
                dots = cur[:-1] @ cur[-1]
                nn = np.argsort(-dots)[:20]
                active = np.concatenate([[new_idx], nn])
                cur = smooth_gd_active(cur, active, steps=130, a0=60.0, a1=500.0,
                                       s0=0.015, s1=0.0015, mu=0.7)
                # LP polish all points briefly
                cur = lp_polish_sec(cur, 0.7)
            except Exception:
                break
            if cur.shape[0] == n:
                try:
                    v = max_cos(cur)
                except Exception:
                    v = 2.0
                if v < best_v:
                    best_v = float(v)
                    best_x = cur.copy()
        # update base for next K to best found (diversify topology)
        # keep base_x as best for next cycle start from best
    # ---------- Path B: reverse - add dummies, optimise, prune ----------
    if remain() > 30:
        M = 12
        try:
            big = best_x.copy()
            for _ in range(M):
                if remain() < 25:
                    break
                c_new = largest_hole_insert(big, rng, n_samples=6000)
                big = np.vstack([big, c_new[None, :]])
            # optimise N>272
            big = smooth_gd(big, steps=300, a0=50.0, a1=600.0, s0=0.02, s1=0.002, mu=0.8)
            if remain() > 22:
                big = lp_polish_sec(big, min(3.5, max(1.0, remain() * 0.1)))
            # prune back to 272, polishing between prunes
            while big.shape[0] > n and remain() > 18:
                big = prune_one(big)
                # brief local anneal around worst region + short LP
                try:
                    rm, mv = row_max(big)
                    w = int(np.argmax(rm))
                    dots = big @ big[w]
                    nn = np.argsort(-dots)[1:21]
                    active = np.concatenate([[w], nn])
                    big = smooth_gd_active(big, active, steps=80, a0=80.0, a1=500.0,
                                           s0=0.01, s1=0.001, mu=0.7)
                    big = lp_polish_sec(big, 0.5)
                except Exception:
                    pass
                if big.shape[0] == n:
                    v = max_cos(big)
                    if v < best_v:
                        best_v = float(v)
                        best_x = big.copy()
            # if we exited early with big>272 due to time, greedily prune without polish
            while big.shape[0] > n:
                big = prune_one(big)
            if big.shape[0] == n:
                # final touch on this branch
                if remain() > 8:
                    big = smooth_gd(big, steps=150, a0=100.0, a1=700.0, s0=0.008, s1=0.001, mu=0.8)
                    big = lp_polish_sec(big, min(2.0, max(0.6, remain() * 0.15)))
                v = max_cos(big)
                if v < best_v:
                    best_v = float(v)
                    best_x = big.copy()
        except Exception:
            pass

    # ---------- final polish of global best ----------
    try:
        if best_x is None:
            best_x = base_x.copy()
        # light anneal then LP with remaining time
        if remain() > 6:
            best_x = smooth_gd(best_x, steps=180, a0=120.0, a1=800.0, s0=0.006, s1=0.0008, mu=0.8)
        left = remain() - 1.0
        left = max(1.5, min(12.0, left))
        best_x = lp_polish_sec(best_x, left)
    except Exception:
        pass
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    # strict finite check, fallback
    if not np.all(np.isfinite(best_x)) or best_x.shape != (272, 4):
        best_x = base_x.copy()
        best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
