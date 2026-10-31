import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _maxcos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _per_point(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    pm = g.max(axis=1)
    return pm, g

def _smooth(x, steps, beta0, beta1, s0, s1, mu=0.75):
    x = _norm(x.copy())
    n, d = x.shape
    vel = np.zeros_like(x)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        beta = beta0 + (beta1 - beta0) * frac
        step = s0 + (s1 - s0) * frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        S = W.sum()
        if not np.isfinite(S) or S == 0:
            break
        W /= S
        grad = W @ x
        proj = (grad * x).sum(axis=1, keepdims=True)
        gt = grad - proj * x
        nrm = np.sqrt((gt ** 2).sum(axis=1, keepdims=True)) + 1e-15
        direc = gt / nrm
        vel = mu * vel + (1.0 - mu) * direc
        vn = np.sqrt((vel ** 2).sum(axis=1, keepdims=True)) + 1e-15
        vdir = vel / vn
        x = x - step * vdir
        x = _norm(x)
    return x

def _deep_hole(others, rng, m_try=4000, refine=80):
    cand = _norm(rng.standard_normal((m_try, 6)))
    dots = cand @ others.T
    worst = dots.max(axis=1)
    y = cand[int(np.argmin(worst))].copy()
    # local deepest-hole refinement: minimize max dot
    step = 0.05
    for it in range(refine):
        d = others @ y  # (k,)
        j = int(np.argmax(d))
        g = others[j] - (others[j] @ y) * y
        n = np.linalg.norm(g) + 1e-15
        g /= n
        y = y - step * g
        y /= (np.linalg.norm(y) + 1e-15)
        step *= 0.97
    return y / (np.linalg.norm(y) + 1e-15)

def _lp_polish(x, seconds):
    x = _norm(x.copy())
    n, d = x.shape
    best = _maxcos(x)
    best_x = x.copy()
    radius = 2e-3
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
        idx = np.where(c_all > thresh)[0]
        if len(idx) > 1500:
            order = np.argsort(c_all[idx])[-1500:]
            idx = idx[order]
        sel = np.zeros_like(c_all, dtype=bool)
        sel[idx] = True
        I, J = iu[0][sel], iu[1][sel]
        csel = c_all[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _maxcos(y)
        if v < best:
            best = v
            best_x = y.copy()
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return best_x

def solve():
    t0 = time.time()
    BUDGET = 172.0
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy")
    except Exception:
        xr = None
    rng0 = np.random.default_rng(0)
    def clean(a):
        a = np.asarray(a, dtype=float)
        if a.shape != (116, 6) or not np.all(np.isfinite(a)):
            return None
        n = np.linalg.norm(a, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return a / n
    cands = []
    if xb is not None:
        c = clean(xb)
        if c is not None:
            cands.append(c)
    if xr is not None:
        c = clean(xr)
        if c is not None:
            cands.append(c)
    if not cands:
        cands.append(_norm(rng0.standard_normal((116, 6))))
    vals = [_maxcos(c) for c in cands]
    bi = int(np.argmin(vals))
    x_best = cands[bi].copy()
    best_val = float(vals[bi])
    cur_x = x_best.copy()
    cur_val = best_val
    rng = np.random.default_rng(20261031)
    explore_end = t0 + BUDGET - 32.0
    it = 0
    temp0, temp1 = 0.004, 0.00015
    while time.time() < explore_end:
        it += 1
        frac = min(1.0, (time.time() - t0) / max(explore_end - t0, 1e-9))
        temp = temp0 + (temp1 - temp0) * frac
        trial = cur_x.copy()
        op = rng.random()
        pm, Gfull = _per_point(trial)
        order_worst = np.argsort(-pm)
        # contact counts for bias
        mcur = pm.max()
        cnt = (Gfull > mcur - 1e-7).sum(axis=1)
        if op < 0.28:
            sigma = float(rng.uniform(0.02, 0.14))
            trial = trial + sigma * rng.standard_normal(trial.shape)
            trial = _norm(trial)
            trial = _smooth(trial, steps=int(rng.integers(250, 450)),
                            beta0=30, beta1=170, s0=0.025, s1=0.004)
        elif op < 0.55:
            k = int(rng.integers(3, 13))
            # bias to overloaded + random
            w = cnt.astype(float) + 0.5
            w /= w.sum()
            idx = rng.choice(116, size=k, replace=False, p=w)
            # include a rattler sometimes
            ratt = np.where(cnt == 0)[0]
            if len(ratt) > 0 and rng.random() < 0.5:
                idx[0] = rng.choice(ratt)
            sigma = float(rng.uniform(0.08, 0.38))
            trial[idx] += sigma * rng.standard_normal((k, 6))
            trial = _norm(trial)
            trial = _smooth(trial, steps=int(rng.integers(300, 550)),
                            beta0=25, beta1=200, s0=0.03, s1=0.003)
        elif op < 0.78:
            nr = int(rng.integers(1, 4))
            worst_idx = order_worst[:nr].copy()
            # also sometimes pick random overloaded
            for ii in range(nr):
                if rng.random() < 0.3:
                    worst_idx[ii] = int(rng.integers(0, 116))
            for wi in worst_idx:
                mask = np.ones(116, dtype=bool); mask[wi] = False
                others = trial[mask]
                y = _deep_hole(others, rng, m_try=4000, refine=60)
                # small jitter
                y = y + 0.005 * rng.standard_normal(6)
                y /= np.linalg.norm(y)
                trial[wi] = y
            trial = _smooth(trial, steps=int(rng.integers(300, 500)),
                            beta0=40, beta1=220, s0=0.02, s1=0.003)
        else:
            ratt = np.where(cnt == 0)[0]
            if len(ratt) == 0:
                ratt = order_worst[-int(rng.integers(2, 6)):]
            # teleport up to all rattlers sequentially to holes
            # limit to 6 per move to save time
            if len(ratt) > 6:
                ratt = rng.choice(ratt, size=6, replace=False)
            for wi in ratt:
                mask = np.ones(116, dtype=bool); mask[wi] = False
                others = trial[mask]
                # exclude other rattlers being moved? use current others
                y = _deep_hole(others, rng, m_try=3000, refine=50)
                trial[wi] = y
            trial = _norm(trial)
            trial = _smooth(trial, steps=int(rng.integers(250, 450)),
                            beta0=35, beta1=200, s0=0.02, s1=0.004)
        v = _maxcos(trial)
        # short LP if very promising to get true value
        if v < best_val + 0.002 and (it % 25 == 0) and (time.time() < explore_end - 5):
            # tiny polish to evaluate basin correctly (1.2s)
            tp = _lp_polish(trial, 1.2)
            v = _maxcos(tp)
            trial = tp
        if v < best_val - 1e-12:
            best_val = v
            x_best = trial.copy()
            cur_x = trial.copy(); cur_val = v
        elif v < cur_val or rng.random() < np.exp(-(v - cur_val) / max(temp, 1e-12)):
            # accept move (allow sideways/worse for exploration), but cap
            if v < cur_val + 0.03:
                cur_x = trial.copy(); cur_val = v
        # periodic restart from best with kick
        if it % 120 == 0 and time.time() < explore_end:
            cur_x = x_best.copy()
            cur_val = best_val
            # diversify slightly handled next iter
        if time.time() >= explore_end:
            break
    remain = BUDGET - (time.time() - t0) - 2.0
    if remain > 5:
        x_best = _lp_polish(x_best, remain)
    else:
        x_best = _norm(x_best)
    # final safety: ensure unit rows, finite
    x_best = _norm(x_best)
    return x_best
