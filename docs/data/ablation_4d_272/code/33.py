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

def lp_polish(x, seconds):
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
        I = iu0[0][sel]; J = iu0[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
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
        W = np.exp(np.clip(alpha * (G - m), -50, 0))
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

def subset_gd(x, active, steps, alpha, smax, mu=0.5):
    active = np.asarray(active, dtype=int)
    k = len(active)
    vel = np.zeros((k, 4))
    pos_of = {int(g): p for p, g in enumerate(active)}
    for it in range(steps):
        Xa = x[active]
        G = Xa @ x.T
        for p, g in enumerate(active):
            G[p, int(g)] = -2.0
        m = G.max()
        W = np.exp(np.clip(alpha * (G - m), -50, 0))
        grad = W @ x
        dot = np.sum(grad * Xa, axis=1, keepdims=True)
        gt = grad - dot * Xa
        vel = mu * vel + gt
        nrm = np.linalg.norm(vel, axis=1).max()
        if nrm < 1e-14:
            break
        Xa2 = Xa - (smax / nrm) * vel
        Xa2 /= np.linalg.norm(Xa2, axis=1, keepdims=True)
        x[active] = Xa2
    x /= np.linalg.norm(x, axis=1, keepdims=True)
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

def largest_hole(Y, rng, n_samples=8000):
    d = Y.shape[1]
    C = rng.standard_normal((n_samples, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    best_s = 2.0
    best_c = C[0].copy()
    bs = 2000
    for a in range(0, n_samples, bs):
        Cb = C[a:a+bs]
        S = Cb @ Y.T
        sm = S.max(axis=1)
        k2 = int(np.argmin(sm))
        if sm[k2] < best_s:
            best_s = float(sm[k2])
            best_c = Cb[k2].copy()
    best_c = refine_single(Y, best_c, steps=100, s=0.03)
    return best_c

def pick_victim(X):
    n = X.shape[0]
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    rowm = G.max(axis=1)
    cand = np.argsort(-rowm)[:30]
    best_i = int(cand[0])
    best_v = 2.0
    all_idx = np.arange(n)
    for i in cand:
        ii = int(i)
        mask = all_idx != ii
        Gm = G[np.ix_(mask, mask)]
        v = float(Gm.max())
        if v < best_v:
            best_v = v
            best_i = ii
    return best_i, best_v

def solve():
    t0 = time.time()
    BUDGET = 170.0
    rng = np.random.default_rng(20260717)
    n, d = 272, 4
    best_x = None; best_v = 2.0
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    v = max_cos(a)
                    if v < best_v:
                        best_v = v; best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        best_x = rng.standard_normal((n, d))
        best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
        best_v = max_cos(best_x)
    cur_src = best_x.copy()

    dels = [32, 24, 40, 28]
    adds = [12, 16, 8, 12]
    cyc = 0
    # reserve final polish
    while time.time() - t0 < BUDGET - 18.0:
        cyc += 1
        if cyc % 2 == 1:
            # ---- shrink then grow ----
            dl = dels[(cyc // 2) % len(dels)]
            if time.time() - t0 > BUDGET - 18.0 - (dl * 0.8 + 4.0):
                # not enough time for full grow; switch to reverse path
                pass
            else:
                rowm, mcur = row_max(cur_src if max_cos(cur_src) < best_v + 0.005 else best_x)
                # mix highest-contact and random
                order = np.argsort(-rowm)
                nh = dl // 2
                kill = set(order[:nh].tolist())
                # random rest
                while len(kill) < dl:
                    kill.add(int(rng.integers(n)))
                kill = sorted(kill)
                keep = np.array([i for i in range(n) if i not in kill], dtype=int)
                Y = (cur_src if max_cos(cur_src) < best_v + 0.01 else best_x)[keep].copy()
                # aggressive optimisation of smaller code
                Y = smooth_gd(Y, steps=300, a0=60.0, a1=500.0, s0=0.02, s1=0.002, mu=0.8)
                # small extra anneal
                Y = smooth_gd(Y, steps=120, a0=400.0, a1=900.0, s0=0.008, s1=0.001, mu=0.85)
                left1 = BUDGET - 18.0 - (time.time() - t0)
                if left1 > 3.0:
                    Y = lp_polish(Y, min(2.5, left1 * 0.12))
                # grow back one at a time
                for gstep in range(dl):
                    if time.time() - t0 > BUDGET - 16.0:
                        break
                    c = largest_hole(Y, rng, n_samples=8000)
                    Y = np.vstack([Y, c.reshape(1, 4)])
                    new_idx = Y.shape[0] - 1
                    # 20 nearest neighbours of new point
                    dots = (Y[:-1] @ Y[-1])
                    nn = np.argsort(-dots)[:20]
                    active = np.concatenate([np.array([new_idx]), nn])
                    # brief anneal of new point + neighbours
                    Y = subset_gd(Y, active, steps=60, alpha=250.0, smax=0.01, mu=0.6)
                    # LP polish all (short)
                    rem = BUDGET - 16.0 - (time.time() - t0)
                    if rem <= 0.2:
                        break
                    lp_sec = 0.7 if (Y.shape[0] % 2 == 0) else 0.45
                    lp_sec = min(lp_sec, rem * 0.2 + 0.2)
                    Y = lp_polish(Y, lp_sec)
                    if Y.shape[0] == n:
                        v = max_cos(Y)
                        if v < best_v:
                            best_v = float(v); best_x = Y.copy()
                            cur_src = Y.copy()
                # if finished to 272 but not best, maybe keep as new source occasionally
                if Y.shape[0] == n:
                    v = max_cos(Y)
                    if v < max_cos(cur_src) + 0.002 and rng.random() < 0.3:
                        cur_src = Y.copy()
                continue
        # ---- reverse path: add dummies, optimise N>272, prune ----
        ad = adds[cyc % len(adds)]
        if time.time() - t0 > BUDGET - 16.0:
            break
        base = best_x.copy()
        Xb = base.copy()
        for _ in range(ad):
            c = largest_hole(Xb, rng, n_samples=6000)
            Xb = np.vstack([Xb, c.reshape(1, 4)])
        # optimise big code
        Xb = smooth_gd(Xb, steps=220, a0=60.0, a1=600.0, s0=0.015, s1=0.002, mu=0.8)
        rem = BUDGET - 16.0 - (time.time() - t0)
        if rem > 2.0:
            Xb = lp_polish(Xb, min(2.0, rem * 0.15))
        # prune down to 272 greedily
        abort = False
        for _ in range(ad):
            if time.time() - t0 > BUDGET - 15.0:
                abort = True
                break
            vi, _ = pick_victim(Xb)
            Xb = np.delete(Xb, vi, axis=0)
            # light local repair: jitter neighbours of removed point region?
            # short subset anneal of points that were neighbours of victim
            # find 15 worst points and anneal them briefly
            if Xb.shape[0] > n:
                # no polish between prunes to save time
                pass
        if abort or Xb.shape[0] != n:
            continue
        # polish pruned 272
        Xb = smooth_gd(Xb, steps=150, a0=150.0, a1=700.0, s0=0.008, s1=0.001, mu=0.8)
        rem = BUDGET - 15.0 - (time.time() - t0)
        if rem > 1.0:
            Xb = lp_polish(Xb, min(2.5, max(0.8, rem * 0.2)))
        v = max_cos(Xb)
        if v < best_v:
            best_v = float(v); best_x = Xb.copy()
            cur_src = Xb.copy()
        else:
            if v < max_cos(cur_src) and rng.random() < 0.4:
                cur_src = Xb.copy()
        # occasional reseed source to best to avoid drift
        if rng.random() < 0.25:
            cur_src = best_x.copy()

    # final polish best
    try:
        best_x = smooth_gd(best_x, steps=80, a0=300.0, a1=800.0, s0=0.004, s1=0.0008, mu=0.85)
    except Exception:
        pass
    left = BUDGET - (time.time() - t0)
    left = max(1.0, min(12.0, left))
    try:
        best_x = lp_polish(best_x, left)
    except Exception:
        pass
    best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
