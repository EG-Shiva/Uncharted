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
        m = c_all.max()
        thresh = m - max(20 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(np.count_nonzero(sel))
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
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
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
        if time.time() - start >= seconds:
            break
    return x


def gd_global(x, steps, a0, a1, s0, s1, mu=0.8):
    n, d = x.shape
    vel = np.zeros_like(x)
    for it in range(steps):
        frac = it / max(steps - 1, 1)
        alpha = a0 * (a1 / a0) ** frac if a0 != a1 else a0
        smax = s0 * (s1 / s0) ** frac if (s0 != s1 and s0 > 0 and s1 > 0) else s0
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


def gd_balanced(x, steps, a0, a1, s0, s1, mu=0.8):
    # per-row softmax: sum_i logsumexp_j, balances load across points
    n, d = x.shape
    vel = np.zeros_like(x)
    for it in range(steps):
        frac = it / max(steps - 1, 1)
        alpha = a0 * (a1 / a0) ** frac if a0 != a1 else a0
        smax = s0 * (s1 / s0) ** frac if (s0 != s1 and s0 > 0 and s1 > 0) else s0
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        rm = G.max(axis=1, keepdims=True)
        W = np.exp(alpha * (G - rm))
        np.fill_diagonal(W, 0.0)
        S = W.sum(axis=1, keepdims=True)
        S = np.maximum(S, 1e-300)
        P = W / S
        grad = (P + P.T) @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        vel = mu * vel + gt
        nrm = np.linalg.norm(vel, axis=1).max()
        if nrm > 1e-14:
            x = x - (smax / nrm) * vel
            x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def hinge_gd(x, target, steps, s0, s1, mu=0.7):
    # minimize sum relu(G-target)^2 : only violators feel force, slack free
    n, d = x.shape
    vel = np.zeros_like(x)
    for it in range(steps):
        frac = it / max(steps - 1, 1)
        smax = s0 * (s1 / s0) ** frac if (s0 != s1) else s0
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        W = G - target
        W[W < 0] = 0.0
        np.fill_diagonal(W, 0.0)
        if W.max() <= 0:
            # no violation: small random tangential diffusion to explore
            break
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        vel = mu * vel + gt
        nrm = np.linalg.norm(vel, axis=1).max()
        if nrm > 1e-14:
            x = x - (smax / nrm) * vel
            x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def random_uniform(rng, n, d):
    x = rng.standard_normal((n, d))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def largest_holes(rng, Y, k, n_samples=15000):
    # Y: (m,4) kept points, find k directions far from Y
    d = Y.shape[1]
    R = rng.standard_normal((n_samples, d))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    D = R @ Y.T
    m = D.max(axis=1)
    idx = np.argpartition(m, k)[:k]
    # order by hole size
    o = np.argsort(m[idx])
    return R[idx[o]]


def solve():
    t0 = time.time()
    BUDGET = 168.0
    rng = np.random.default_rng(7)
    n, d = 272, 4

    try:
        bx = np.load("best.npy")
    except Exception:
        try:
            bx = np.load("record.npy")
        except Exception:
            bx = random_uniform(rng, n, d)
    bx = np.asarray(bx, dtype=np.float64)
    if bx.shape != (n, d):
        bx = np.resize(bx, (n, d))
    bx = bx / np.linalg.norm(bx, axis=1, keepdims=True)
    best_x = bx.copy()
    best_v = max_cos(best_x)
    try:
        rx = np.load("record.npy")
        rx = np.asarray(rx, dtype=np.float64)
        if rx.shape == (n, d):
            rx = rx / np.linalg.norm(rx, axis=1, keepdims=True)
            rv = max_cos(rx)
            if rv < best_v:
                best_x, best_v = rx.copy(), rv
    except Exception:
        pass

    cur_x = best_x.copy()
    cur_v = best_v

    temp0, temp1 = 0.0025, 0.00015
    it = 0

    while time.time() - t0 < BUDGET - 13.0:
        it += 1
        elapsed = time.time() - t0
        frac_global = elapsed / BUDGET
        temp = temp0 * (temp1 / temp0) ** frac_global

        use_best = (rng.random() < 0.35) or (cur_v > best_v + 0.006)
        base = best_x if use_best else cur_x

        r = rng.random()
        xp = None
        if r < 0.28:
            # large global kick
            sigma = float(rng.choice([0.12, 0.20, 0.30, 0.45]))
            xp = base + rng.standard_normal(base.shape) * sigma
            xp /= np.linalg.norm(xp, axis=1, keepdims=True)
        elif r < 0.55:
            # subset reset of worst points: holes or random
            try:
                rm, _ = row_max(base)
            except Exception:
                rm = np.zeros(n)
            m = int(rng.integers(8, 33))
            worst = np.argsort(-rm)[:m]
            xp = base.copy()
            if rng.random() < 0.55:
                mask = np.ones(n, dtype=bool)
                mask[worst] = False
                Y = base[mask]
                try:
                    holes = largest_holes(rng, Y, m)
                    xp[worst] = holes
                    xp[worst] += rng.standard_normal((m, d)) * 0.01
                except Exception:
                    xp[worst] = random_uniform(rng, m, d)
            else:
                xp[worst] = random_uniform(rng, m, d)
            xp /= np.linalg.norm(xp, axis=1, keepdims=True)
        elif r < 0.70:
            # avalanche: break top contact + reshuffle its neighbourhood
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            iu = np.triu_indices(n, 1)
            cg = G[iu]
            K = min(30, len(cg) - 1)
            top = np.argpartition(-cg, K)[:K]
            choice = top[rng.integers(len(top))]
            i = int(iu[0][choice]); j = int(iu[1][choice])
            xp = base.copy() + rng.standard_normal(base.shape) * 0.02
            # neighbourhood of i,j
            ni = np.argsort(-G[i])[:14]
            nj = np.argsort(-G[j])[:14]
            neigh = np.unique(np.concatenate([ni, nj, np.array([i, j])]))
            xp[neigh] += rng.standard_normal((len(neigh), d)) * 0.22
            push = 0.10 + 0.10 * rng.random()
            xp[i] += push * (base[i] - base[j])
            xp[j] += push * (base[j] - base[i])
            xp /= np.linalg.norm(xp, axis=1, keepdims=True)
        elif r < 0.82:
            # crossover between best and current after Procrustes alignment
            if np.array_equal(best_x, cur_x):
                xp = base + rng.standard_normal(base.shape) * 0.25
                xp /= np.linalg.norm(xp, axis=1, keepdims=True)
            else:
                try:
                    M = cur_x.T @ best_x
                    U, _, Vt = np.linalg.svd(M)
                    R = U @ Vt
                    cur_al = cur_x @ R
                    mask = rng.random(n) < 0.5
                    xp = np.where(mask[:, None], best_x, cur_al)
                    xp += rng.standard_normal(xp.shape) * 0.015
                    xp /= np.linalg.norm(xp, axis=1, keepdims=True)
                except Exception:
                    xp = base + rng.standard_normal(base.shape) * 0.25
                    xp /= np.linalg.norm(xp, axis=1, keepdims=True)
        else:
            # fresh random start
            xp = random_uniform(rng, n, d)

        # choose optimizer
        r2 = rng.random()
        remain = BUDGET - 13.0 - (time.time() - t0)
        if remain < 3:
            break
        if r2 < 0.36:
            # balanced -> uniform contacts
            a0 = 8.0 + 8.0 * rng.random()
            a1 = 90.0 + 90.0 * rng.random()
            s0 = 0.010 + 0.008 * rng.random()
            s1 = 0.0008 + 0.0012 * rng.random()
            steps = 700 + int(rng.integers(0, 600))
            if remain < 30:
                steps = min(steps, 500)
            xp = gd_balanced(xp, steps, a0, a1, s0, s1, mu=0.8)
            # short global clean
            xp = gd_global(xp, 150, 60.0, 200.0, 0.004, 0.0008, mu=0.8)
        elif r2 < 0.68:
            a0 = 18.0 + 10.0 * rng.random()
            a1 = 140.0 + 100.0 * rng.random()
            s0 = 0.008 + 0.007 * rng.random()
            s1 = 0.0006 + 0.0010 * rng.random()
            steps = 800 + int(rng.integers(0, 700))
            if remain < 30:
                steps = min(steps, 500)
            xp = gd_global(xp, steps, a0, a1, s0, s1, mu=0.8)
        else:
            delta = float(rng.choice([0.001, 0.002, 0.0035, 0.005]))
            target = best_v - delta
            steps_h = 700 + int(rng.integers(0, 600))
            if remain < 30:
                steps_h = min(steps_h, 500)
            xp = hinge_gd(xp, target, steps_h, 0.015, 0.002, mu=0.7)
            # follow with balanced/global to settle
            if rng.random() < 0.5:
                xp = gd_balanced(xp, 400, 12.0, 120.0, 0.006, 0.0008, mu=0.8)
            else:
                xp = gd_global(xp, 400, 40.0, 220.0, 0.006, 0.0008, mu=0.8)

        vgd = max_cos(xp)

        do_lp = (vgd < best_v + 0.004) or (rng.random() < 0.12)
        if do_lp:
            remain2 = BUDGET - 13.0 - (time.time() - t0)
            if remain2 < 2.0:
                yp, vlp = xp, vgd
            else:
                lp_time = min(3.5, max(2.0, remain2 / 7.0))
                yp = lp_polish(xp, lp_time)
                vlp = max_cos(yp)
        else:
            yp, vlp = xp, vgd

        if vlp < best_v - 1e-12:
            best_x, best_v = yp, vlp
            cur_x, cur_v = yp, vlp
        else:
            if vlp < cur_v or rng.random() < np.exp((cur_v - vlp) / max(temp, 1e-9)):
                cur_x, cur_v = yp, vlp

        if time.time() - t0 >= BUDGET - 13.0:
            break

    remain = BUDGET - (time.time() - t0)
    if remain > 3.0:
        best_x = lp_polish(best_x, min(remain - 1.0, 11.0))
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
