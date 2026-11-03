import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def per_point_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1)

def lp_polish_deadline(x, deadline, r_init=0.004, max_pairs=2200):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = max_cos(x)
    except Exception:
        return x
    radius = r_init
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-13:
        G = x @ x.T
        c_all = G[iu0]
        m = float(c_all.max())
        thresh = m - max(20.0 * radius, 2e-9)
        sel = c_all > thresh
        cnt = int(sel.sum())
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > max_pairs:
            part = np.argpartition(c_all, -max_pairs)[-max_pairs:]
            sel = np.zeros(len(c_all), dtype=bool)
            sel[part] = True
        I = iu0[0][sel]; J = iu0[1][sel]
        c_sel = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.empty((p, 2 * d + 1), dtype=np.int64)
        ar = np.arange(d)
        cols[:, :d] = I[:, None] * d + ar
        cols[:, d:2 * d] = J[:, None] * d + ar
        cols[:, 2 * d] = n * d
        cols = cols.ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        if not np.all(np.isfinite(yn)) or np.any(yn < 1e-12):
            radius *= 0.5
            continue
        y = y / yn
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return x

def repulse_anneal(x, steps, beta0, beta1, lr0, lr1):
    X = x.copy()
    n, d = X.shape
    for s in range(steps):
        frac = s / max(1, steps - 1)
        beta = beta0 * (beta1 / beta0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        W = np.exp((G - m) * beta)
        np.fill_diagonal(W, 0.0)
        Grad = W @ X
        proj = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - proj * X
        gn = float(np.linalg.norm(Gt) / np.sqrt(n))
        if not np.isfinite(gn) or gn < 1e-16:
            continue
        X = X - (lr / gn) * Gt
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X

def greedy_hole(X, idx, rng, n_cand=3000, refine=60):
    n, d = X.shape
    mask = np.ones(n, dtype=bool); mask[idx] = False
    R = X[mask]
    C = rng.standard_normal((n_cand, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    S = C @ R.T
    mx = S.max(axis=1)
    bi = int(np.argmin(mx))
    v = C[bi].copy()
    # local refinement: move away from worst neighbours
    step = 0.06
    for it in range(refine):
        dots = R @ v
        m = float(dots.max())
        # top-k neighbours
        k = 8
        if R.shape[0] > k:
            ti = np.argpartition(dots, -k)[-k:]
        else:
            ti = np.arange(R.shape[0])
        w = dots[ti] - m
        # softmax toward worst
        bw = 60.0
        ww = np.exp(w * bw)
        ww /= ww.sum()
        g = ww @ (R[ti] - dots[ti][:, None] * v[None, :].repeat(len(ti), axis=0) if False else R[ti])
        # tangent component
        g = g - float(g @ v) * v
        gn = float(np.linalg.norm(g))
        if gn < 1e-14 or not np.isfinite(gn):
            break
        g /= gn
        v = v - step * g
        v /= float(np.linalg.norm(v))
        step *= 0.96
    return v

def solve():
    t0 = time.time()
    HARD_END = t0 + 176.0
    LOOP_END = t0 + 148.0
    rng = np.random.default_rng(0)
    n, d = 176, 9
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                nr = np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(nr > 1e-12):
                    cands.append(a / nr)
        except Exception:
            pass
    if not cands:
        a = rng.standard_normal((n, d)); a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    vals = [max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    best_X = cands[bi].copy()
    best_v = float(vals[bi])
    curr_X = best_X.copy()
    curr_v = best_v
    # quick initial LP to lock local optimum baseline (short)
    curr_X = lp_polish_deadline(curr_X, min(time.time() + 8.0, LOOP_END))
    curr_v = max_cos(curr_X)
    if curr_v < best_v:
        best_X = curr_X.copy(); best_v = curr_v
    it = 0
    while time.time() < LOOP_END:
        it += 1
        time_left = LOOP_END - time.time()
        if time_left < 3.0:
            break
        # choose perturbation size cyclically
        cyc = it % 5
        if cyc == 0:
            nkill = 2
            noise = 0.0
            anneal_steps = 250
        elif cyc == 1:
            nkill = 3
            noise = 0.0
            anneal_steps = 250
        elif cyc == 2:
            nkill = 5
            noise = 0.0
            anneal_steps = 350
        elif cyc == 3:
            nkill = 1
            noise = 0.015
            anneal_steps = 200
        else:
            nkill = 4
            noise = 0.008
            anneal_steps = 300
        trial = curr_X.copy()
        # global jitter if noise
        if noise > 0:
            trial = trial + rng.standard_normal(trial.shape) * noise
            trial /= np.linalg.norm(trial, axis=1, keepdims=True)
        # contact counts based on near-max pairs
        try:
            G = trial @ trial.T
            np.fill_diagonal(G, -2.0)
            m0 = float(G.max())
            tol = 0.004
            # per-point worst and contact degree
            w = G.max(axis=1)
            deg = (G > m0 - tol).sum(axis=1)
        except Exception:
            continue
        # score for removal: worst first, then degree
        order = np.lexsort((-deg, -w))
        kills = order[:nkill].copy()
        # occasionally random kill to diversify
        if it % 7 == 0:
            kills[0] = int(rng.integers(0, n))
        for idx in kills:
            if time.time() >= LOOP_END:
                break
            v = greedy_hole(trial, int(idx), rng, n_cand=2500, refine=50)
            trial[int(idx)] = v
        # repulsion annealing with varying hardness
        beta0 = 30.0 + (it % 3) * 10.0
        beta1 = 140.0 + (it % 4) * 30.0
        trial = repulse_anneal(trial, steps=anneal_steps, beta0=beta0, beta1=beta1, lr0=0.03, lr1=0.002)
        # short LP polish
        budget = 3.5 if nkill <= 2 else 4.5
        trial = lp_polish_deadline(trial, min(time.time() + budget, LOOP_END))
        try:
            tv = max_cos(trial)
        except Exception:
            continue
        if tv < best_v - 1e-12:
            best_X = trial.copy(); best_v = tv
            curr_X = trial.copy(); curr_v = tv
        elif tv < curr_v + 0.0015 and rng.random() < 0.6:
            # sideways move to new basin
            curr_X = trial.copy(); curr_v = tv
        elif it % 12 == 0:
            # restart from best with bigger kick
            curr_X = best_X.copy()
            curr_X = curr_X + rng.standard_normal(curr_X.shape) * 0.02
            curr_X /= np.linalg.norm(curr_X, axis=1, keepdims=True)
            curr_v = max_cos(curr_X)
        if best_v < 0.4595:
            break
    # final long polish of best
    final_deadline = HARD_END - 1.0
    if time.time() < final_deadline:
        best_X = lp_polish_deadline(best_X, final_deadline, r_init=0.005, max_pairs=3000)
    best_X = best_X / np.linalg.norm(best_X, axis=1, keepdims=True)
    return best_X
