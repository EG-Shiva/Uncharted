import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _normalize(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def _lp_polish(x, seconds):
    x = _normalize(x)
    n, d = x.shape
    best = _max_cos(x)
    best_x = x.copy()
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while True:
        if time.time() - start >= seconds:
            break
        if radius < 1e-14:
            break
        G = x @ x.T
        c_all = G[iu]
        m = c_all.max()
        thresh = m - max(20 * radius, 1e-9)
        sel = c_all > thresh
        # truncate to worst pairs for speed
        idx = np.where(sel)[0]
        if len(idx) > 1400:
            order = np.argsort(c_all[idx])[-1400:]
            idx = idx[order]
            sel = np.zeros_like(sel)
            sel[idx] = True
        I, J = iu[0][sel], iu[1][sel]
        csel = c_all[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            # equality: x_i . dx_i = 0
            A_eq = coo_matrix(
                (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq,
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
        y = _normalize(y)
        v = _max_cos(y)
        if v < best:
            best = v
            best_x = y.copy()
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
            # stay at best? keep x as is for next linearization
    return best_x


def _smooth_run(x, steps, beta0, beta1, lr0, lr1, mu=0.9):
    x = _normalize(x.copy())
    n, d = x.shape
    vel = np.zeros_like(x)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        beta = beta0 + (beta1 - beta0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        # stable exp
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        S = W.sum()
        if not np.isfinite(S) or S == 0:
            break
        W /= S
        grad = W @ x
        proj = (grad * x).sum(axis=1, keepdims=True)
        gt = grad - proj * x
        vel = mu * vel + gt
        x = x - lr * vel
        x = _normalize(x)
    return x


def _best_hole(others, rng, m_try=3000):
    # others: (k,6) normalized, find point minimizing max dot
    cand = rng.standard_normal((m_try, 6))
    cand = _normalize(cand)
    dots = cand @ others.T
    worst = dots.max(axis=1)
    return cand[int(np.argmin(worst))]


def solve():
    t0 = time.time()
    # load start
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            rng0 = np.random.default_rng(0)
            xb = _normalize(rng0.standard_normal((116, 6)))
    if xb.shape != (116, 6):
        rng0 = np.random.default_rng(0)
        xb = _normalize(rng0.standard_normal((116, 6)))
    xb = _normalize(np.asarray(xb, dtype=float))
    # clean non-finite
    if not np.all(np.isfinite(xb)):
        rng0 = np.random.default_rng(1)
        xb = _normalize(rng0.standard_normal((116, 6)))
    best_val = _max_cos(xb)
    x_best = xb.copy()
    try:
        xr = np.load("record.npy")
        xr = _normalize(np.asarray(xr, dtype=float))
        vr = _max_cos(xr)
        if vr < best_val:
            best_val = vr
            x_best = xr.copy()
    except Exception:
        pass

    rng = np.random.default_rng(12345)
    cur_x = x_best.copy()
    cur_val = best_val

    # quick initial squeeze if time allows (very short)
    # main basin-hopping loop, reserve ~25s for final polish
    explore_deadline = t0 + 145.0

    it = 0
    while time.time() < explore_deadline:
        it += 1
        remaining = explore_deadline - time.time()
        if remaining < 2.5:
            break
        mode = it % 5
        trial = cur_x.copy()
        n = 116
        if mode == 0:
            # global shake + smooth + lp
            sigma = float(rng.uniform(0.03, 0.14))
            trial = trial + rng.standard_normal(trial.shape) * sigma
            trial = _normalize(trial)
            steps = int(rng.integers(200, 450))
            b0 = float(rng.uniform(15, 40))
            b1 = float(rng.uniform(120, 250))
            lr0 = float(rng.uniform(1.5, 4.0))
            lr1 = float(rng.uniform(0.2, 0.6))
            trial = _smooth_run(trial, steps, b0, b1, lr0, lr1)
            trial = _lp_polish(trial, min(1.8, remaining - 0.2))
        elif mode == 1:
            # worst point -> best hole
            G = trial @ trial.T
            np.fill_diagonal(G, -2.0)
            m = G.max()
            # degree with tight threshold
            tight = G > m - 0.002
            deg = tight.sum(axis=1)
            # pick one of top-5 overloaded
            order = np.argsort(-deg)
            k = int(rng.integers(0, min(5, len(order))))
            victim = int(order[k])
            mask = np.ones(n, dtype=bool)
            mask[victim] = False
            others = trial[mask]
            hole = _best_hole(others, rng, m_try=4000)
            trial[victim] = hole
            # also teleport a rattler if exists
            per_max = G.max(axis=1)
            ratt = np.where(per_max < m - 0.002)[0]
            if len(ratt) > 0 and rng.random() < 0.7:
                ridx = int(rng.choice(ratt))
                mask2 = np.ones(n, dtype=bool)
                mask2[ridx] = False
                trial[ridx] = _best_hole(trial[mask2], rng, m_try=4000)
            trial = _lp_polish(trial, min(1.8, remaining - 0.2))
        elif mode == 2:
            # focused shake on jammed cluster
            G = trial @ trial.T
            np.fill_diagonal(G, -2.0)
            m = G.max()
            tight = G > m - 0.001
            deg = tight.sum(axis=1)
            # normalize deg to shake amplitude
            amp_base = float(rng.uniform(0.02, 0.06))
            amp_hot = float(rng.uniform(0.08, 0.22))
            med = np.median(deg)
            hot = deg > med
            noise = rng.standard_normal(trial.shape) * amp_base
            noise[hot] += rng.standard_normal((int(hot.sum()), 6)) * amp_hot
            trial = trial + noise
            trial = _normalize(trial)
            trial = _smooth_run(trial, 250, 30, 180, 2.0, 0.3)
            trial = _lp_polish(trial, min(1.6, remaining - 0.2))
        elif mode == 3:
            # rattler teleport only + greedy hole fill
            G = trial @ trial.T
            np.fill_diagonal(G, -2.0)
            m = G.max()
            per_max = G.max(axis=1)
            ratt = np.where(per_max < m - 0.0015)[0]
            if len(ratt) == 0:
                # random subset shake
                ksub = int(rng.integers(4, 16))
                idx = rng.choice(n, size=ksub, replace=False)
                trial[idx] += rng.standard_normal((ksub, 6)) * 0.12
                trial = _normalize(trial)
            else:
                nr = min(len(ratt), int(rng.integers(1, 4)))
                chosen = rng.choice(ratt, size=nr, replace=False)
                for ridx in chosen:
                    mask = np.ones(n, dtype=bool)
                    mask[int(ridx)] = False
                    trial[int(ridx)] = _best_hole(trial[mask], rng, m_try=5000)
            trial = _lp_polish(trial, min(1.8, remaining - 0.2))
        else:
            # random subset strong kick + smooth
            ksub = int(rng.integers(8, 30))
            idx = rng.choice(n, size=ksub, replace=False)
            sigma = float(rng.uniform(0.05, 0.25))
            trial[idx] += rng.standard_normal((ksub, 6)) * sigma
            # small global jitter
            trial += rng.standard_normal(trial.shape) * 0.01
            trial = _normalize(trial)
            trial = _smooth_run(trial, 300, 20, 150, 2.5, 0.4)
            trial = _lp_polish(trial, min(1.6, remaining - 0.2))

        v = _max_cos(trial)
        if v < best_val:
            best_val = v
            x_best = trial.copy()
            cur_x = trial.copy()
            cur_val = v
        else:
            # random-walk acceptance to diversify (Metropolis-like)
            if v < cur_val or rng.random() < 0.08:
                cur_x = trial.copy()
                cur_val = v
            # occasionally restart from best
            if rng.random() < 0.25:
                cur_x = x_best.copy()
                cur_val = best_val

    # final long polish of best
    time_left = 178.0 - (time.time() - t0)
    if time_left > 3:
        # give most of remaining to LP, but cap to avoid overrun
        x_best = _lp_polish(x_best, min(time_left - 1.0, 28.0))
        best_val = _max_cos(x_best)

    x_best = _normalize(x_best)
    return x_best
