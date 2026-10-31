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


def smooth_gd(x, steps, a0, a1, s0, s1, mu=0.85):
    n, d = x.shape
    vel = np.zeros_like(x)
    for it in range(steps):
        frac = it / max(steps - 1, 1)
        # log interpolation
        if a0 != a1:
            alpha = a0 * (a1 / a0) ** frac
        else:
            alpha = a0
        if s0 != s1:
            smax = s0 * (s1 / s0) ** frac
        else:
            smax = s0
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


def refine_single(Y, c0, steps=200, alpha0=20.0, alpha1=400.0, s0=0.03, s1=0.002, mu=0.6):
    c = c0 / np.linalg.norm(c0)
    vel = np.zeros(4)
    for it in range(steps):
        frac = it / max(steps - 1, 1)
        alpha = alpha0 * (alpha1 / alpha0) ** frac
        s = s0 * (s1 / s0) ** frac
        dots = Y @ c
        m = dots.max()
        w = np.exp(alpha * (dots - m))
        # weighted closest direction
        g = w @ Y
        gt = g - np.dot(g, c) * c
        nrm = np.linalg.norm(gt)
        if nrm < 1e-14:
            break
        gt = gt / nrm
        vel = mu * vel + gt
        vn = np.linalg.norm(vel)
        if vn > 1e-14:
            c = c - (s / vn) * vel
            n = np.linalg.norm(c)
            if n < 1e-14:
                break
            c = c / n
    return c / np.linalg.norm(c)


def hole_reinsert(x, rng, n_samples=8000):
    n, d = x.shape
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    rm = G.max(axis=1)
    # choose point to move: worst with some randomness
    order = np.argsort(-rm)
    # pick among top few
    if rng.random() < 0.7:
        move = int(order[0])
    elif rng.random() < 0.5:
        move = int(order[int(rng.integers(1, min(6, n)))] )
    else:
        # random rattler or random
        mcur = rm.max()
        ratt = np.where(rm < mcur - 5e-4)[0]
        if len(ratt) > 0 and rng.random() < 0.7:
            move = int(rng.choice(ratt))
        else:
            move = int(rng.integers(n))
    Y = np.delete(x, move, axis=0)
    C = rng.standard_normal((n_samples, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    best_c = None
    best_s = 2.0
    bs = 2000
    for a in range(0, n_samples, bs):
        Cb = C[a:a+bs]
        S = Cb @ Y.T
        sm = S.max(axis=1)
        k2 = int(np.argmin(sm))
        if sm[k2] < best_s:
            best_s = float(sm[k2])
            best_c = Cb[k2].copy()
    best_c = refine_single(Y, best_c)
    y = x.copy()
    y[move] = best_c
    return y


def greedy_max_descent(x, seconds, start_step=0.005):
    # projected subgradient on true max with line search
    t_end = time.time() + seconds
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    best = max_cos(x)
    step = start_step
    n, d = x.shape
    iu = np.triu_indices(n, 1)
    while time.time() < t_end and step > 1e-12:
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        # contacts
        sel = G > m - 1e-4
        # build gradient of max: for each i, sum of contacting neighbours
        # use upper triangle to avoid double but symmetric
        W = (G > m - 1e-4).astype(np.float64)
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        nrm = np.linalg.norm(gt)
        if nrm < 1e-14:
            break
        gt = gt / nrm
        # try steps
        improved = False
        for trial in (step, step * 0.5, step * 0.25):
            y = x - trial * gt
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            v = max_cos(y)
            if v < best:
                x, best = y, v
                step = min(trial * 1.3, 0.02)
                improved = True
                break
        if not improved:
            step *= 0.5
    return x


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
        if np.count_nonzero(sel) > 4000:
            idx = np.where(sel)[0]
            order = np.argpartition(-c_all[idx], 4000 - 1)[:4000]
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


def solve():
    t0 = time.time()
    BUDGET = 172.0
    rng = np.random.default_rng(20261031)
    n, d = 272, 4

    try:
        bx = np.load("best.npy")
    except Exception:
        bx = None
    try:
        rx = np.load("record.npy")
    except Exception:
        rx = None

    def normed(a):
        a = np.asarray(a, dtype=np.float64)
        if a.shape != (272, 4):
            a = np.resize(a, (272, 4))
        a = a / np.linalg.norm(a, axis=1, keepdims=True)
        return a

    best_x = None
    best_v = 2.0
    for cand in (bx, rx):
        if cand is None:
            continue
        try:
            c = normed(cand)
            v = max_cos(c)
            if v < best_v:
                best_v = v
                best_x = c.copy()
        except Exception:
            continue
    if best_x is None:
        best_x = rng.standard_normal((n, d))
        best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
        best_v = max_cos(best_x)

    sigmas = [0.003, 0.008, 0.015, 0.030, 0.060, 0.120]
    it = 0

    # main basin hopping; reserve ~22s for final polishes
    while time.time() - t0 < BUDGET - 22.0:
        it += 1
        remain = BUDGET - 22.0 - (time.time() - t0)
        if remain <= 0:
            break
        u = rng.random()
        base = best_x
        # rowmax for rattler info
        try:
            Gtmp = base @ base.T
            np.fill_diagonal(Gtmp, -2.0)
            rm = Gtmp.max(axis=1)
            mcur = Gtmp.max()
            ratt = rm < (mcur - 5e-4)
        except Exception:
            ratt = np.zeros(n, dtype=bool)

        if u < 0.45:
            # global gaussian kick
            sigma = sigmas[(it - 1) % len(sigmas)] * (0.7 + 0.6 * rng.random())
            y = base + sigma * rng.standard_normal((n, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            steps = int(rng.choice([150, 250, 400]))
            a0 = float(rng.choice([8.0, 15.0, 25.0]))
            a1 = float(rng.choice([250.0, 400.0, 600.0]))
            s0 = float(rng.choice([0.02, 0.035]))
            s1 = float(rng.choice([0.002, 0.004]))
            y = smooth_gd(y, steps, a0, a1, s0, s1, mu=0.85)
        elif u < 0.70:
            # rattler-targeted kick
            sigma_big = float(rng.choice([0.05, 0.10, 0.20]))
            sigma_small = float(rng.choice([0.002, 0.005]))
            noise = rng.standard_normal((n, d))
            sc = np.where(ratt[:, None], sigma_big, sigma_small)
            # if few rattlers, kick worst 10 as well
            if np.count_nonzero(ratt) < 5:
                order = np.argsort(-rm)
                extra = order[:8]
                sc[extra] = max(sigma_big, 0.05)
            y = base + sc * noise
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            steps = int(rng.choice([200, 350]))
            a0 = 12.0
            a1 = float(rng.choice([300.0, 500.0]))
            y = smooth_gd(y, steps, a0, a1, 0.030, 0.003, mu=0.85)
        else:
            # hole reinsertion + polish
            y = hole_reinsert(base, rng, n_samples=int(rng.choice([6000, 10000])))
            steps = int(rng.choice([150, 300]))
            a0 = float(rng.choice([10.0, 20.0]))
            a1 = float(rng.choice([300.0, 600.0]))
            y = smooth_gd(y, steps, a0, a1, 0.025, 0.003, mu=0.85)

        v = max_cos(y)
        if v < best_v:
            best_x = y.copy()
            best_v = float(v)
        # opportunistic small greedy squeeze if very close
        if time.time() - t0 > BUDGET - 22.0:
            break

    # squeeze: greedy then LP with remaining time
    rem = BUDGET - (time.time() - t0)
    if rem > 4:
        # 1/3 greedy, 2/3 LP roughly, but LP gets at least 8s
        gsec = min(6.0, rem * 0.3)
        best_x = greedy_max_descent(best_x, gsec)
        best_v = max_cos(best_x)
        rem2 = BUDGET - (time.time() - t0)
        if rem2 > 2:
            best_x = lp_polish(best_x, rem2 - 0.5)
            best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)

    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    # guard NaN
    best_x = np.where(np.isfinite(best_x), best_x, 0.0)
    nr = np.linalg.norm(best_x, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    best_x = best_x / nr
    return best_x
