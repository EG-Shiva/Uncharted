import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    if seconds <= 0.05:
        return x
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while True:
        if time.time() - start >= seconds:
            break
        if radius < 1e-14:
            break
        G = x @ x.T
        iu_c = G[iu]
        m = float(iu_c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = iu_c > thresh
        if int(sel.sum()) > 3000:
            idx = np.where(sel)[0]
            order = np.argsort(iu_c[idx])[-3000:]
            mask = np.zeros_like(sel)
            mask[idx[order]] = True
            sel = mask
        I = iu[0][sel]
        J = iu[1][sel]
        c = iu_c[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
        cols = np.concatenate([
            (I[:, None] * d + np.arange(d)),
            (J[:, None] * d + np.arange(d)),
            np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        rows2 = np.repeat(np.arange(p), 2 * d + 1)
        A_ub = coo_matrix((vals, (rows2, cols)), shape=(p, nv)).tocsr()
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
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _smooth_relax(X, steps=600, alpha0=25.0, alpha1=320.0, lr0=0.25, lr1=0.02):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    for t in range(steps):
        frac = t / max(1, steps - 1)
        alpha = alpha0 + (alpha1 - alpha0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        # weights favour near-max pairs; low alpha equilibrates, high alpha tightens
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True)
        rs = np.maximum(rs, 1e-12)
        Gw = (W @ X) / rs
        dot = (Gw * X).sum(axis=1, keepdims=True)
        Gt = Gw - dot * X
        # clip extreme tangents for stability
        X = X - lr * Gt
        xn = np.linalg.norm(X, axis=1, keepdims=True)
        xn[xn == 0] = 1.0
        X = X / xn
    return X

def _largest_holes(X, k, n_samples, rng):
    d = X.shape[1]
    R = rng.normal(size=(n_samples, d))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    S = R @ X.T
    closest = S.max(axis=1)
    idx = np.argsort(closest)[:k]
    return R[idx]

def _crowding_scores(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    # average of top-4 cosines per point (crowding)
    part = np.partition(G, -4, axis=1)[:, -4:]
    avg4 = part.mean(axis=1)
    deg = (G >= m - 1e-7).sum(axis=1)
    return m, avg4, deg, G

def solve():
    t0 = time.time()
    BUDGET = 172.0
    deadline = t0 + BUDGET
    rng = np.random.default_rng(0)
    n, d = 173, 5
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (n, d):
                an = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(an)
                if v < best_val:
                    best_val = v
                    best = an.copy()
        except Exception:
            pass
    if best is None:
        best = rng.normal(size=(n, d))
        best = best / np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    global_best = best.copy()
    global_val = float(best_val)
    # population of (val, X) diverse basins, keep best always
    pop_vals = [global_val]
    pop_Xs = [global_best.copy()]

    def pop_insert(val, X):
        nonlocal pop_vals, pop_Xs
        # avoid exact duplicates
        for pv in pop_vals:
            if abs(pv - val) < 1e-12:
                return
        pop_vals.append(float(val))
        pop_Xs.append(X.copy())
        idx = np.argsort(np.array(pop_vals))
        # keep at most 6
        keep = idx[:6]
        pop_vals = [pop_vals[i] for i in keep]
        pop_Xs = [pop_Xs[i] for i in keep]

    # Phase A: diversification with smoothed random starts
    # use limited time ~25s
    phaseA_end = t0 + 30.0
    n_init = 0
    while time.time() < phaseA_end and len(pop_vals) < 6:
        rem = deadline - time.time()
        if rem < 30:
            break
        X = rng.normal(size=(n, d))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        X = _smooth_relax(X, steps=500, alpha0=20.0, alpha1=250.0, lr0=0.25, lr1=0.02)
        v = _max_cos(X)
        # only LP-polish if remotely competitive to save time
        if v < global_val + 0.03:
            X = _lp_polish(X, 0.6)
            v = _max_cos(X)
        pop_insert(v, X)
        if v < global_val - 1e-12:
            global_best = X.copy()
            global_val = float(v)
        n_init += 1
        if n_init >= 10:
            break
    # ensure global best in pop
    pop_insert(global_val, global_best)

    trial = 0
    while True:
        rem = deadline - time.time()
        if rem < 8.0:
            break
        # pick base with softmax bias to better but allowing worse basins
        vals_arr = np.array(pop_vals)
        m0 = vals_arr.min()
        Temp = 0.004
        w = np.exp(-(vals_arr - m0) / Temp)
        w /= w.sum()
        bi = rng.choice(len(pop_vals), p=w)
        base = pop_Xs[bi]
        mode = trial % 6
        Y = base.copy()
        if mode == 0:
            # single surgery: most crowded -> largest hole
            _, avg4, deg, G = _crowding_scores(Y)
            victim = int(np.argmax(avg4 + 0.02 * deg))
            holes = _largest_holes(Y, 1, 20000, rng)
            Y[victim] = holes[0] + 0.005 * rng.normal(size=(d,))
            # small shake of old neighbours
            neigh = np.where(G[victim] > float(G.max()) - 0.03)[0]
            if len(neigh) > 0:
                Y[neigh] += 0.02 * rng.normal(size=(len(neigh), d))
        elif mode == 1:
            # double surgery
            _, avg4, deg, _ = _crowding_scores(Y)
            score = avg4 + 0.02 * deg
            victims = np.argsort(-score)[:2]
            holes = _largest_holes(Y, 2, 20000, rng)
            for k2, vv in enumerate(victims):
                Y[int(vv)] = holes[k2] + 0.008 * rng.normal(size=(d,))
            # jitter a few random
            sub = rng.choice(n, 5, replace=False)
            Y[sub] += 0.03 * rng.normal(size=(5, d))
        elif mode == 2:
            # kick overloaded neighbourhood medium
            _, _, deg, _ = _crowding_scores(Y)
            order = np.argsort(-deg)
            subset = np.unique(np.concatenate([order[:5], rng.choice(n, 5, replace=False)]))
            Y[subset] += 0.08 * rng.normal(size=(len(subset), d))
        elif mode == 3:
            # small global shake (stay in basin but equilibrate via smooth)
            Y += 0.018 * rng.normal(size=(n, d))
        elif mode == 4:
            # broad shake of 25 points
            subset = rng.choice(n, 25, replace=False)
            Y[subset] += 0.06 * rng.normal(size=(len(subset), d))
        else:
            # large kick to jump basin + fresh hole move
            subset = rng.choice(n, 10, replace=False)
            Y[subset] += 0.15 * rng.normal(size=(len(subset), d))
            # move rattler if exists
            _, _, deg, _ = _crowding_scores(Y)
            ratt = np.where(deg <= 1)[0]
            if len(ratt) > 0:
                holes = _largest_holes(Y, 1, 12000, rng)
                Y[int(ratt[0])] = holes[0] + 0.01 * rng.normal(size=(d,))
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        # equilibrating smooth relaxation before LP
        Y = _smooth_relax(Y, steps=450, alpha0=25.0, alpha1=300.0, lr0=0.22, lr1=0.02)
        v_smooth = _max_cos(Y)
        # gate LP: skip hopeless candidates
        if v_smooth > global_val + 0.025:
            trial += 1
            continue
        pt = 1.0
        rem2 = deadline - time.time()
        if rem2 < 12:
            pt = max(0.4, rem2 - 7.0)
        pt = min(pt, rem2 - 6.5)
        if pt < 0.35:
            break
        Yp = _lp_polish(Y, pt)
        v = _max_cos(Yp)
        if v < global_val - 1e-12:
            global_best = Yp.copy()
            global_val = float(v)
            pop_insert(v, Yp)
        elif v < max(pop_vals) - 1e-9:
            # accept diverse worse basins to keep exploration (non-greedy)
            # only if not too close to existing
            if all(abs(v - pv) > 5e-7 for pv in pop_vals):
                pop_insert(v, Yp)
        else:
            # occasionally keep a worse basin to preserve diversity
            if (trial % 17 == 0) and len(pop_vals) < 6:
                pop_insert(v, Yp)
        trial += 1

    rem = deadline - time.time()
    if rem > 1.5:
        global_best = _lp_polish(global_best, rem - 0.7)
    global_best = global_best / np.linalg.norm(global_best, axis=1, keepdims=True)
    return global_best
