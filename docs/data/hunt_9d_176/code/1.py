import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def lp_polish_deadline(x, deadline, r_init=0.005, max_pairs=2500):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = max_cos(x)
    except Exception:
        return x
    radius = r_init
    iu0 = np.triu_indices(n, 1)
    # precompute iu arrays
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        m = float(c_all.max())
        thresh = m - max(20.0 * radius, 1e-9)
        # select
        if radius > 0.01:
            # cap to top max_pairs to avoid huge LP
            k = max_pairs
            if len(c_all) > k:
                part = np.argpartition(c_all, -k)[-k:]
                sel_mask = np.zeros(len(c_all), dtype=bool)
                sel_mask[part] = True
                # further threshold within top-k
                sel_mask = sel_mask & (c_all > thresh)
                if sel_mask.sum() < 200:
                    sel_mask = np.zeros(len(c_all), dtype=bool)
                    sel_mask[part] = True
                sel = sel_mask
            else:
                sel = c_all > thresh
        else:
            sel = c_all > thresh
            if sel.sum() > max_pairs:
                part = np.argpartition(c_all, -max_pairs)[-max_pairs:]
                sel = np.zeros(len(c_all), dtype=bool)
                sel[part] = True
        if sel.sum() == 0:
            radius *= 0.5
            continue
        I = iu0[0][sel]; J = iu0[1][sel]
        c_sel = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        # cols
        Ii = (I[:, None] * d + np.arange(d)).ravel()
        Ji = (J[:, None] * d + np.arange(d)).ravel()
        # interleave per row: need order [I-block, J-block, t]
        # Build cols row by row
        cols_I = (I[:, None] * d + np.arange(d))
        cols_J = (J[:, None] * d + np.arange(d))
        cols = np.empty((p, 2 * d + 1), dtype=int)
        cols[:, :d] = cols_I
        cols[:, d:2*d] = cols_J
        cols[:, 2*d] = n * d
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

def repulse_anneal(x, steps=700, beta0=25.0, beta1=160.0, lr0=0.025, lr1=0.002):
    X = x.copy()
    n, d = X.shape
    for s in range(steps):
        frac = s / max(1, steps - 1)
        beta = beta0 * (beta1 / beta0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        # weights focused on top
        W = np.exp((G - m) * beta)
        np.fill_diagonal(W, 0.0)
        Grad = W @ X
        proj = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - proj * X
        gnorm = float(np.linalg.norm(Gt) / np.sqrt(n))
        if gnorm < 1e-16 or not np.isfinite(gnorm):
            continue
        X = X - (lr / gnorm) * Gt
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X

def solve():
    t0 = time.time()
    HARD_END = t0 + 176.0
    LOOP_END = t0 + 152.0
    rng = np.random.default_rng(7)
    n, d = 176, 9
    # load starts
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                nr = np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(nr > 1e-12):
                    a = a / nr
                    cands.append(a)
        except Exception:
            pass
    if len(cands) == 0:
        X0 = rng.standard_normal((n, d)); X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        cands.append(X0)
    vals = [max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    best_X = cands[bi].copy()
    best_v = float(vals[bi])
    curr_X = best_X.copy()
    curr_v = best_v
    # short initial squeeze if time allows (2s)
    if time.time() < LOOP_END:
        best_X = lp_polish_deadline(best_X, min(time.time() + 4.0, LOOP_END), r_init=0.001)
        best_v = max_cos(best_X)
        curr_X = best_X.copy(); curr_v = best_v

    sigmas = [0.004, 0.008, 0.015, 0.03, 0.06, 0.12]
    T = 4e-4
    trial = 0
    while time.time() < LOOP_END:
        trial += 1
        if trial % 25 == 0:
            # restart from best
            curr_X = best_X.copy(); curr_v = best_v
        sigma = sigmas[(trial - 1) % len(sigmas)]
        mode = trial % 6
        Xp = curr_X.copy()
        # compute per-point worst cosine for targeting
        try:
            Gc = curr_X @ curr_X.T
            np.fill_diagonal(Gc, -2.0)
            rowmax = Gc.max(axis=1)
            order = np.argsort(rowmax)  # ascending
        except Exception:
            order = np.arange(n); rowmax = np.zeros(n)
        if mode == 0:
            # global shake
            Xp = Xp + sigma * rng.standard_normal((n, d))
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
        elif mode == 1:
            # targeted: worst 12 points big kick, rest small
            k = 12
            idx_worst = order[-k:]
            noise = 0.003 * rng.standard_normal((n, d))
            noise[idx_worst] += (sigma * 1.5) * rng.standard_normal((k, d))
            Xp = Xp + noise
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
        elif mode == 2:
            # greedy re-insertion of single worst point
            idx = int(order[-1])
            # sample candidates, pick hole
            M = 1500
            R = rng.standard_normal((M, d)); R /= np.linalg.norm(R, axis=1, keepdims=True)
            dots = R @ Xp.T  # M x n
            dots[:, idx] = -2.0
            worst = dots.max(axis=1)
            pick = int(np.argmin(worst))
            Xp[idx] = R[pick]
            # small shake others
            Xp = Xp + 0.002 * rng.standard_normal((n, d))
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
        elif mode == 3:
            # re-insert 3 worst points greedily sequentially
            k = 3
            idxs = order[-k:]
            Xp = Xp + 0.004 * rng.standard_normal((n, d))
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
            for idx in idxs:
                M = 800
                R = rng.standard_normal((M, d)); R /= np.linalg.norm(R, axis=1, keepdims=True)
                dots = R @ Xp.T
                dots[:, idx] = -2.0
                worst = dots.max(axis=1)
                pick = int(np.argmin(worst))
                Xp[idx] = R[pick]
        elif mode == 4:
            # block shake of random subset
            k = 24
            idxs = rng.choice(n, size=k, replace=False)
            Xp[idxs] += sigma * rng.standard_normal((k, d))
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
        else:
            # orthogonal-plane jitter: global medium
            Xp = Xp + (sigma * 0.7) * rng.standard_normal((n, d))
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)

        # repulsion annealing (fast topology move)
        steps = 500 if sigma < 0.05 else 350
        try:
            Xp = repulse_anneal(Xp, steps=steps)
        except Exception:
            pass
        # short LP polish
        budget = 2.2 if sigma < 0.05 else 3.0
        deadline = min(time.time() + budget, LOOP_END, HARD_END - 12.0)
        if deadline <= time.time() + 0.2:
            break
        try:
            # larger initial radius after big shake
            r0 = 0.008 if sigma > 0.02 else 0.004
            Xp = lp_polish_deadline(Xp, deadline, r_init=r0)
            v = max_cos(Xp)
        except Exception:
            continue
        if not np.isfinite(v):
            continue
        if v < best_v:
            best_v = v; best_X = Xp.copy()
            curr_X = Xp.copy(); curr_v = v
        else:
            # Metropolis move of curr
            if v < curr_v or rng.random() < np.exp((curr_v - v) / T):
                curr_X = Xp.copy(); curr_v = v
        # early exit if excellent? keep going to use time
    # final long polish of best
    try:
        final_deadline = HARD_END - 1.0
        if time.time() < final_deadline:
            best_X = lp_polish_deadline(best_X, final_deadline, r_init=0.0012)
    except Exception:
        pass
    best_X = best_X / np.linalg.norm(best_X, axis=1, keepdims=True)
    return best_X
