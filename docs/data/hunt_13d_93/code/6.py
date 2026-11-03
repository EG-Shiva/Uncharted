import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _adam_softmax(x0, beta, steps, lr, rng_seed=0, log_every=0, track_best=True):
    # minimize M + log(Z)/beta via Adam on sphere, return best-by-max found
    x = x0.copy()
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    best_x = x.copy()
    # initial max
    g0 = x @ x.T
    np.fill_diagonal(g0, -2.0)
    best_max = float(g0.max())
    t = 0
    iu = np.triu_indices(n, 1)
    for it in range(steps):
        t += 1
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        M = float(G.max())
        # shifted exp, only upper needed but full is fine
        # avoid overflow: G-M <= 0
        E = np.exp(beta * (G - M))
        np.fill_diagonal(E, 0.0)
        Z = E[iu].sum()
        if Z == 0 or not np.isfinite(Z):
            Z = 1.0
        W = E / Z
        grad = W @ x
        # tangent projection
        dot = (grad * x).sum(axis=1, keepdims=True)
        gt = grad - dot * x
        m = b1 * m + (1 - b1) * gt
        v = b2 * v + (1 - b2) * (gt * gt)
        mhat = m / (1 - b1 ** t)
        vhat = v / (1 - b2 ** t)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        # slight decay
        x = x - step
        x = _norm(x)
        if track_best and (it % 25 == 0 or it == steps - 1):
            c = _max_cos(x)
            if c < best_max:
                best_max = c
                best_x = x.copy()
    return best_x, best_max

def _lp_polish(x, seconds, deadline=1e18):
    x = _norm(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < deadline:
        if radius < 1e-14:
            break
        Gfull = x @ x.T
        c_all = Gfull[iu]
        m = float(c_all.max())
        thresh = m - max(20 * radius, 1e-9)
        sel = c_all > thresh
        # cap to top pairs for speed
        idx = np.where(sel)[0]
        if len(idx) > 2000:
            # keep largest 2000
            part = np.argpartition(c_all[idx], -2000)[-2000:]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[part]] = True
            sel = mask
            idx = np.where(sel)[0]
        if len(idx) == 0:
            radius *= 0.5
            continue
        I, J = iu[0][sel], iu[1][sel]
        c = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    np.random.seed(0)
    rng = np.random.RandomState(0)
    t0 = time.time()
    deadline = t0 + 175.0
    n, d = 93, 13
    # load start
    x_start = None
    best_val_load = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            if a.shape == (93, 13):
                a = _norm(a.astype(float))
                v = _max_cos(a)
                if v < best_val_load:
                    best_val_load = v
                    x_start = a.copy()
        except Exception:
            pass
    if x_start is None:
        x_start = _norm(rng.randn(n, d))
        best_val_load = _max_cos(x_start)
    x_best = x_start.copy()
    best = float(best_val_load)

    # Stage A: annealed softmax Adam
    schedule = [(300, 2500, 0.008), (800, 2500, 0.005), (2000, 2500, 0.003), (5000, 2000, 0.002), (15000, 1500, 0.001)]
    x_cur = x_best.copy()
    for beta, steps, lr in schedule:
        if time.time() > deadline - 60:
            break
        # start from current best-ish (x_cur if good else x_best)
        x_opt, v_opt = _adam_softmax(x_cur, beta, steps, lr)
        if v_opt < best:
            best = v_opt
            x_best = x_opt.copy()
        # continue annealing from best
        x_cur = x_best.copy()
        # quick check: also try from slightly perturbed to avoid stall
        if time.time() > deadline - 60:
            break

    # quick LP to snap
    if time.time() < deadline - 50:
        x_lp = _lp_polish(x_best, seconds=8, deadline=deadline - 40)
        v_lp = _max_cos(x_lp)
        if v_lp < best:
            best = v_lp
            x_best = x_lp.copy()
        x_cur = x_best.copy()

    # Stage B: basin hopping with kicks + void reinsert
    # reserve 18s for final polish
    while time.time() < deadline - 18:
        remaining = deadline - 18 - time.time()
        if remaining <= 0:
            break
        trial = x_best.copy()
        r = rng.rand()
        if r < 0.55:
            # kick worst k points
            G = trial @ trial.T
            np.fill_diagonal(G, -2.0)
            # per-point worst
            rowmax = G.max(axis=1)
            k = int(rng.choice([2, 3, 4, 5]))
            worst = np.argsort(rowmax)[-k:]
            sigma = float(rng.choice([0.02, 0.05, 0.10, 0.18]))
            trial[worst] += sigma * rng.randn(k, d)
            trial = _norm(trial)
        else:
            # void reinsert single worst point
            G = trial @ trial.T
            np.fill_diagonal(G, -2.0)
            rowmax = G.max(axis=1)
            wi = int(np.argmax(rowmax))
            mask = np.ones(n, dtype=bool); mask[wi] = False
            Xr = trial[mask]
            ncand = 3000
            cand = _norm(rng.randn(ncand, d))
            dots = cand @ Xr.T
            cmax = dots.max(axis=1)
            bi = int(np.argmin(cmax))
            c = cand[bi:bi+1].copy()
            # refine void position a bit (minimize softmax max)
            beta_v = 2000
            for _ in range(300):
                dd = (c @ Xr.T).ravel()
                Mm = dd.max()
                w = np.exp(beta_v * (dd - Mm))
                w = w / (w.sum() + 1e-30)
                g = (w[:, None] * Xr).sum(axis=0, keepdims=True)
                g = g - (g * c).sum() * c
                c = _norm(c - 0.05 * g)
                if time.time() > deadline - 18:
                    break
            trial[wi] = c.ravel()
            trial = _norm(trial)
        # short refine with softmax
        beta_r = int(rng.choice([1500, 3000, 8000]))
        lr_r = 0.003 if beta_r <= 3000 else 0.0015
        steps_r = 1200
        # adapt steps to remaining time (avoid overrun)
        x_opt, v_opt = _adam_softmax(trial, beta_r, steps_r, lr_r)
        if v_opt < best - 1e-12:
            best = v_opt
            x_best = x_opt.copy()
            x_cur = x_best.copy()
        # also opportunistic: if trial itself better (rare)
        # loop continues; time check inside adam? we use short steps so ok
        if time.time() > deadline - 18:
            break

    # Stage C: final polish
    time_left = deadline - time.time() - 1.0
    if time_left > 3:
        # one more medium-beta smooth to balance then LP
        try:
            x_sm, v_sm = _adam_softmax(x_best, 5000, 1500, 0.001)
            if v_sm < best:
                best = v_sm
                x_best = x_sm.copy()
        except Exception:
            pass
        time_left2 = deadline - time.time() - 0.5
        if time_left2 > 2:
            try:
                x_lp2 = _lp_polish(x_best, seconds=min(time_left2, 14), deadline=deadline - 0.5)
                v_lp2 = _max_cos(x_lp2)
                if v_lp2 < best:
                    best = v_lp2
                    x_best = x_lp2.copy()
            except Exception:
                pass

    x_best = _norm(x_best)
    return x_best
