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
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        m = float(c_all.max())
        thresh = m - max(20.0 * radius, 1e-9)
        if sel_sum_needed := True:
            pass
        if radius > 0.01:
            k = max_pairs
            if len(c_all) > k:
                part = np.argpartition(c_all, -k)[-k:]
                sel = np.zeros(len(c_all), dtype=bool)
                sel[part] = True
                # keep thresholded within top-k, but ensure enough
                mask_thr = (c_all > thresh)
                # if too few above thresh, keep all top-k
                inter = sel & mask_thr
                if inter.sum() >= 200:
                    sel = inter
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

def repulse_anneal(X, steps, beta0, beta1, lr0, lr1, noise0=0.0, rng=None):
    n, d = X.shape
    for s in range(steps):
        frac = s / max(1, steps - 1)
        beta = beta0 * (beta1 / beta0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        # avoid overflow: (G-m)<=0
        W = np.exp((G - m) * beta)
        np.fill_diagonal(W, 0.0)
        Grad = W @ X
        proj = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - proj * X
        gnorm = float(np.linalg.norm(Gt) / np.sqrt(n))
        if not np.isfinite(gnorm) or gnorm < 1e-16:
            continue
        step = (lr / gnorm) * Gt
        if noise0 > 0.0 and rng is not None:
            amp = noise0 * (1.0 - frac)
            N = rng.standard_normal((n, d))
            N -= np.sum(N * X, axis=1, keepdims=True) * X
            step = step + amp * N / np.sqrt(n)
            # renormalize step scale roughly
            # keep lr scale: rescale to lr magnitude
        X = X - step
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X

def best_hole_vector(Fixed, rng, n_samples=4000, refine_steps=180):
    # Fixed: (m,d) array of fixed points (unit)
    d = Fixed.shape[1]
    # sample
    C = rng.standard_normal((n_samples, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    D = C @ Fixed.T  # (samples, m)
    vals = D.max(axis=1)
    bi = int(np.argmin(vals))
    v = C[bi].copy()
    # refine single-vector softmax descent
    beta0, beta1 = 20.0, 220.0
    lr0, lr1 = 0.15, 0.01
    for s in range(refine_steps):
        frac = s / max(1, refine_steps - 1)
        beta = beta0 * (beta1 / beta0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        dots = Fixed @ v  # (m,)
        m = float(dots.max())
        w = np.exp((dots - m) * beta)
        # gradient = sum w_j * Fixed_j
        g = w @ Fixed
        # tangential
        gt = g - float(g @ v) * v
        gn = float(np.linalg.norm(gt))
        if not np.isfinite(gn) or gn < 1e-16:
            continue
        v = v - (lr / max(gn, 1e-12)) * gt * max(gn, 1e-12) * 0.5 if False else v - (lr * 0.2) * gt / (np.mean(w)+1e-12) * 0.1
        # simpler stable update: v -= lr_t * gt / ||gt||? use fixed scale
        # redo with normalized step for stability:
        # we already did a scaled step; correct below:
        # To avoid double update confusion, recompute cleanly:
        # undo and do normalized step
        # Actually just do normalized step from current (we already stepped); instead do:
        # revert: use direct normalized step
        # For stability, recompute v from before? simpler: do normalized descent:
        # We'll just apply normalized step on top of correction by re-projecting:
        # (the above line mixed scales) -> replace with clean normalized step:
        # We need v_before; to keep code simple, just project and continue with small tangential noise handling
        # Instead do: v = v / norm (already will) and next iter corrects.
        v = v / np.linalg.norm(v)
        # NOTE: above uses a heuristic scale; we add a normalized correction to guarantee movement:
        # small extra normalized pull
        # (kept minimal to avoid divergence)
    # second cleaner refinement with normalized steps
    v = C[bi].copy()
    for s in range(refine_steps):
        frac = s / max(1, refine_steps - 1)
        beta = beta0 * (beta1 / beta0) ** frac
        lr = 0.25 * (0.02/0.25) ** frac
        dots = Fixed @ v
        m = float(dots.max())
        w = np.exp((dots - m) * beta)
        g = w @ Fixed
        gt = g - float(g @ v) * v
        gn = float(np.linalg.norm(gt))
        if not np.isfinite(gn) or gn < 1e-16:
            continue
        v = v - (lr / gn) * gt
        v = v / np.linalg.norm(v)
    return v

def solve():
    t0 = time.time()
    HARD_END = t0 + 176.0
    LOOP_END = t0 + 138.0
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
    if len(cands) == 0:
        X0 = rng.standard_normal((n, d)); X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        cands.append(X0)
    vals = [max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    best_X = cands[bi].copy()
    best_v = float(vals[bi])
    curr_X = best_X.copy()
    curr_v = best_v
    # quick temperature for SA
    T0 = 0.0015
    it = 0
    # pre-allocate iu not needed
    while time.time() < LOOP_END:
        it += 1
        frac_global = (time.time() - t0) / max(1e-9, (LOOP_END - t0))
        Temp = T0 * (1.0 - frac_global * 0.8)
        mode = rng.integers(0, 100)
        Y = None
        # choose perturbation
        if mode < 35:
            # global small shake + anneal
            sigma = float(rng.choice(np.array([0.005, 0.012, 0.03, 0.07, 0.15])))
            base = curr_X if rng.random() < 0.7 else best_X
            Y = base + sigma * rng.standard_normal((n, d))
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            steps = int(rng.choice(np.array([300, 500, 700])))
            beta0 = float(rng.choice(np.array([15.0, 25.0, 40.0])))
            beta1 = float(rng.choice(np.array([120.0, 180.0, 260.0])))
            lr0, lr1 = 0.03, 0.002
            if rng.random() < 0.3:
                lr0 = 0.05
            noise0 = 0.0
            if rng.random() < 0.25:
                noise0 = 0.01
            Y = repulse_anneal(Y, steps=steps, beta0=beta0, beta1=beta1, lr0=lr0, lr1=lr1, noise0=noise0, rng=rng)
        elif mode < 60:
            # worst-k strong shake / replace
            base = curr_X if rng.random() < 0.5 else best_X
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            rowmax = G.max(axis=1)
            k = int(rng.integers(1, 7))
            idx = np.argpartition(rowmax, -k)[-k:]
            Y = base.copy()
            if rng.random() < 0.5:
                # random replacement
                R = rng.standard_normal((k, d)); R /= np.linalg.norm(R, axis=1, keepdims=True)
                Y[idx] = R
            else:
                sig = float(rng.choice(np.array([0.2, 0.4, 0.7])))
                Y[idx] = Y[idx] + sig * rng.standard_normal((k, d))
                Y[idx] /= np.linalg.norm(Y[idx], axis=1, keepdims=True)
            steps = int(rng.choice(np.array([300, 500, 800])))
            beta0 = float(rng.choice(np.array([20.0, 35.0])))
            beta1 = float(rng.choice(np.array([150.0, 250.0])))
            Y = repulse_anneal(Y, steps=steps, beta0=beta0, beta1=beta1, lr0=0.03, lr1=0.002, noise0=0.0, rng=rng)
        else:
            # greedy hole re-insertion for worst point(s)
            base = curr_X if rng.random() < 0.5 else best_X
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            rowmax = G.max(axis=1)
            # number of holes to reinsert sequentially
            nh = int(rng.integers(1, 4))
            order = np.argsort(-rowmax)[:nh]
            Y = base.copy()
            for ii in order:
                mask = np.ones(n, dtype=bool); mask[ii] = False
                Fixed = Y[mask]
                vstar = best_hole_vector(Fixed, rng, n_samples=3000, refine_steps=120)
                Y[ii] = vstar
            # short relax anneal
            steps = int(rng.choice(np.array([200, 400])))
            Y = repulse_anneal(Y, steps=steps, beta0=25.0, beta1=180.0, lr0=0.02, lr1=0.002, noise0=0.0, rng=rng)
        # evaluate
        try:
            v = max_cos(Y)
        except Exception:
            continue
        if not np.isfinite(v):
            continue
        if v < best_v:
            best_v = v
            best_X = Y.copy()
            curr_X = Y.copy()
            curr_v = v
        elif v < curr_v or rng.random() < np.exp(-(v - curr_v) / max(Temp, 1e-6)):
            curr_X = Y.copy()
            curr_v = v
        # occasional restart curr to best if stuck
        if it % 60 == 0 and rng.random() < 0.4:
            curr_X = best_X.copy()
            curr_v = best_v
        if best_v < 0.4595:
            break
    # final polish best with remaining time
    try:
        best_X = lp_polish_deadline(best_X, deadline=HARD_END - 1.0, r_init=0.005)
        best_v = max_cos(best_X)
    except Exception:
        pass
    best_X = best_X / np.linalg.norm(best_X, axis=1, keepdims=True)
    return best_X
