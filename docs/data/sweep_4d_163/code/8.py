import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def lp_polish(x, seconds, t0=None):
    """Trust-region LP polish, time-limited."""
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    if t0 is None:
        t0 = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - t0 < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
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

def smooth_anneal(x, steps, tau0, tau1, lr0, lr1, mu, rng, noise0=0.0, noise1=0.0):
    n, d = x.shape
    v = np.zeros_like(x)
    for s in range(steps):
        frac = s / max(1, steps - 1)
        tau = tau0 * (tau1 / tau0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        # max off-diagonal
        # use triu for max
        iu = np.triu_indices(n, 1)
        m = G[iu].max()
        E = np.exp((G - m) / tau)
        np.fill_diagonal(E, 0.0)
        S = E.sum()
        if not np.isfinite(S) or S <= 0:
            break
        P = E / S
        grad = P @ x
        dot = (grad * x).sum(axis=1, keepdims=True)
        gt = grad - dot * x
        rms = np.sqrt((gt ** 2).mean()) + 1e-15
        gsn = gt / rms
        v = mu * v + (1.0 - mu) * gsn
        # project momentum to tangent (old v has normal part after x moved)
        vd = (v * x).sum(axis=1, keepdims=True)
        v = v - vd * x
        x = x - lr * v
        if noise0 > 0:
            nz = noise0 * (noise1 / max(noise0, 1e-15)) ** frac if noise1 > 0 else noise0
            if nz > 0:
                r = rng.standard_normal((n, d))
                r = r - (r * x).sum(axis=1, keepdims=True) * x
                x = x + nz * r
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def contact_counts(x, thresh):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    C = G > thresh
    return C.sum(axis=1), G

def solve():
    t_start = time.time()
    LIMIT = 172.0
    rng = np.random.default_rng(0)
    # load start
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.standard_normal((163, 4))
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (163, 4):
        # try to reshape / truncate
        tmp = rng.standard_normal((163, 4))
        m = min(xb.shape[0], 163)
        tmp[:m, :min(4, xb.shape[1])] = xb[:m, :min(4, xb.shape[1])]
        xb = tmp
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best = max_cos(xb)
    # keep copy
    x_best = xb.copy()

    # quick contact info for surgery
    def get_worst_indices(xc, k=10):
        try:
            thr = max_cos(xc) - 5e-4
            cnt, _ = contact_counts(xc, thr)
            # overloaded = largest counts, plus rattlers (0)
            order = np.argsort(-cnt)
            worst = order[:k].copy()
            ratt = np.where(cnt == 0)[0]
            return worst, ratt, cnt
        except Exception:
            return np.arange(k), np.array([], dtype=int), None

    trial = 0
    # reserve final polish ~35s
    while time.time() - t_start < LIMIT - 35:
        trial += 1
        remain = LIMIT - 35 - (time.time() - t_start)
        if remain < 3:
            break
        mode = trial % 5
        if mode == 4:
            # fresh random start
            x_init = rng.standard_normal((163, 4))
            x_init /= np.linalg.norm(x_init, axis=1, keepdims=True)
            steps = 1400
            tau0, tau1 = 0.035, 0.0025
            lr0, lr1 = 0.12, 0.008
            mu = 0.9
        elif mode == 3:
            # surgical: replace overloaded + rattlers
            worst, ratt, cnt = get_worst_indices(x_best, k=10)
            x_init = x_best.copy()
            # small global jitter
            x_init = x_init + rng.standard_normal((163, 4)) * 0.01
            sel = np.unique(np.concatenate([worst, ratt])) if len(ratt) > 0 else worst
            # add a couple random others for diversity
            extra = rng.choice(163, size=2, replace=False)
            sel = np.unique(np.concatenate([sel, extra]))
            for s in sel:
                v = rng.standard_normal(4)
                v /= np.linalg.norm(v)
                x_init[s] = v
            x_init /= np.linalg.norm(x_init, axis=1, keepdims=True)
            steps = 1200
            tau0, tau1 = 0.025, 0.0018
            lr0, lr1 = 0.10, 0.006
            mu = 0.9
        else:
            sigmas = [0.03, 0.08, 0.16, 0.30]
            sigma = sigmas[(trial - 1) % 4]
            # occasionally perturb subset only
            x_init = x_best + rng.standard_normal((163, 4)) * sigma
            x_init /= np.linalg.norm(x_init, axis=1, keepdims=True)
            steps = 1000
            tau0, tau1 = 0.022, 0.0015
            lr0, lr1 = 0.09, 0.005
            mu = 0.88
            # vary a bit with trial
            if trial % 2 == 0:
                tau0 *= 1.4

        # adapt steps to remaining time (smooth is cheap)
        x_try = smooth_anneal(x_init, steps, tau0, tau1, lr0, lr1, mu, rng)
        v_smooth = max_cos(x_try)
        # decide LP: if promising or periodically
        promising = (v_smooth < best + 0.015) or (mode == 4 and v_smooth < best + 0.03)
        # also force LP every ~7 trials if close
        if (trial % 7 == 0) and (v_smooth < best + 0.04):
            promising = True
        if promising:
            # short LP, 1.2-2s
            budget = 1.6
            if v_smooth < best + 0.002:
                budget = 2.2
            # don't exceed reserve
            avail = LIMIT - 35 - (time.time() - t_start)
            budget = min(budget, max(0.6, avail - 1.0))
            if budget > 0.5:
                x_try = lp_polish(x_try, budget)
                v2 = max_cos(x_try)
            else:
                v2 = v_smooth
            if v2 < best - 1e-13:
                best = v2
                x_best = x_try.copy()
                # update quickly: continue exploiting this basin a bit
                # extra squeeze immediately if time allows
                if time.time() - t_start < LIMIT - 40:
                    x_best = lp_polish(x_best, 1.0)
                    best = max_cos(x_best)
        # opportunistic: if smooth alone beats best (rare), take it
        else:
            if v_smooth < best - 1e-12:
                # polish it
                x_try = lp_polish(x_try, 1.5)
                v2 = max_cos(x_try)
                if v2 < best:
                    best = v2
                    x_best = x_try.copy()
        # safety: keep deterministic loop bounded
        if trial > 60:
            break

    # final thorough polish of best
    rem = LIMIT - (time.time() - t_start)
    if rem > 4:
        try:
            x_best = lp_polish(x_best, rem - 1.0)
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
