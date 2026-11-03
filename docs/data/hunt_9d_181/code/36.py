import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def lp_polish(x, seconds, init_radius=1e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = init_radius
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start + 0.25 > seconds and radius < 5e-4:
            break
        Gfull = x @ x.T
        c_all = Gfull[iu]
        m = c_all.max()
        thresh = max(20 * radius, 1e-9)
        sel = c_all > m - thresh
        cnt = int(sel.sum())
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 2800:
            kth = np.partition(c_all, -2800)[-2800]
            sel = c_all >= kth
            cnt = int(sel.sum())
            if cnt > 3200:
                idx = np.where(sel)[0]
                order = np.argsort(-c_all[idx])[:3200]
                mask = np.zeros_like(c_all, dtype=bool)
                mask[idx[order]] = True
                sel = mask
        I = iu[0][sel]; J = iu[1][sel]
        csel = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
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
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def smooth_anneal(x, iters=250, p_start=25, p_end=70, lr=0.04):
    n, d = x.shape
    for it in range(iters):
        frac = it / max(1, iters - 1)
        p = p_start + (p_end - p_start) * frac
        G = x @ x.T
        np.fill_diagonal(G, -1.0)
        Gp = np.where(G > 0, G, 0.0)
        m = Gp.max()
        if m <= 1e-12:
            break
        W = (Gp / m) ** (p - 1)
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad_t = grad - dot * x
        nrm_mean = np.mean(np.linalg.norm(grad_t, axis=1)) + 1e-12
        step = lr * (1.0 - 0.5 * frac) / nrm_mean
        x = x - step * grad_t
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

# ---- symmetry-constrained construction ----
PERM_FLIP = np.array([0, 8, 7, 6, 5, 4, 3, 2, 1], dtype=int)
FIXED_VEC = np.ones(9) / 3.0

def expand_dihedral(reps):
    K, d = reps.shape
    n = K * 18 + 1
    X = np.empty((n, d))
    perms = []
    pos = 0
    for k in range(K):
        r = reps[k]
        rf = r[PERM_FLIP]
        for s in range(9):
            perm = (np.arange(d) - s) % d
            perms.append((k, perm.copy()))
            X[pos] = r[perm]; pos += 1
        for s in range(9):
            # copy = roll(rf, s): copy[i] = rf[(i-s)%d] = r[PERM_FLIP[(i-s)%d]]
            perm = PERM_FLIP[(np.arange(d) - s) % d]
            perms.append((k, perm.copy()))
            X[pos] = r[perm]; pos += 1
    X[pos] = FIXED_VEC
    return X, perms

def expand_cyclic(reps):
    K, d = reps.shape
    n = K * 9 + 1
    X = np.empty((n, d))
    perms = []
    pos = 0
    for k in range(K):
        r = reps[k]
        for s in range(9):
            perm = (np.arange(d) - s) % d
            perms.append((k, perm.copy()))
            X[pos] = r[perm]; pos += 1
    X[pos] = FIXED_VEC
    return X, perms

def sym_optimize(group, K, steps, seed, p_start=12, p_end=90, lr=0.25, mu=0.6):
    rng = np.random.default_rng(seed)
    d = 9
    reps = rng.normal(size=(K, d))
    reps /= np.linalg.norm(reps, axis=1, keepdims=True)
    vel = np.zeros_like(reps)
    for it in range(steps):
        frac = it / max(1, steps - 1)
        p = p_start + (p_end - p_start) * frac
        if group == 'dihedral':
            X, perms = expand_dihedral(reps)
        else:
            X, perms = expand_cyclic(reps)
        G = X @ X.T
        np.fill_diagonal(G, -1.0)
        Gp = np.where(G > 0, G, 0.0)
        m = float(Gp.max())
        if m <= 1e-12:
            break
        W = (Gp / m) ** (p - 1)
        np.fill_diagonal(W, 0.0)
        gradX = W @ X
        dot = np.sum(gradX * X, axis=1, keepdims=True)
        gradX = gradX - dot * X
        gradR = np.zeros_like(reps)
        for idx, (k, perm) in enumerate(perms):
            g = gradX[idx]
            # copy[i]=reps[k][perm[i]] => gradR[k][perm[i]] += g[i]
            np.add.at(gradR[k], perm, g)
        # project to tangent of reps
        d2 = np.sum(gradR * reps, axis=1, keepdims=True)
        gradR = gradR - d2 * reps
        # normalize
        mn = np.mean(np.linalg.norm(gradR, axis=1)) + 1e-12
        gradR = gradR / mn
        vel = mu * vel + gradR
        step = lr * (1.0 - 0.5 * frac)
        reps = reps - step * vel
        reps /= np.linalg.norm(reps, axis=1, keepdims=True)
    if group == 'dihedral':
        X, _ = expand_dihedral(reps)
    else:
        X, _ = expand_cyclic(reps)
    return X / np.linalg.norm(X, axis=1, keepdims=True), float(max_cos(X))

def solve():
    t0 = time.time()
    BUDGET = 168.0
    rng = np.random.default_rng(0)
    n, d = 181, 9
    cand = []
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cand.append(a)
        except Exception:
            pass
    if len(cand) == 0:
        tmp = rng.normal(size=(n, d))
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        cand.append(tmp)
    scores = [max_cos(a) for a in cand]
    bi = int(np.argmin(scores))
    xb = cand[bi].copy(); bb = scores[bi]

    # Phase 1: symmetry-constrained search (time-boxed ~28s)
    sym_seeds = []
    t_sym_end = t0 + 30.0
    configs = [('dihedral', 10, 1100, 11), ('dihedral', 10, 1100, 22), ('cyclic', 20, 900, 33)]
    for (group, K, steps, sd) in configs:
        if time.time() - t0 > 29.0:
            break
        try:
            Xs, vs = sym_optimize(group, K, steps, seed=sd)
            sym_seeds.append((vs, Xs))
        except Exception:
            continue
    sym_seeds.sort(key=lambda t: t[0])
    # Phase 2: break symmetry with increasing perturbations + LP polish
    # first ensure baseline polished record
    rem = BUDGET - (time.time() - t0)
    if rem > 12:
        try:
            xb = lp_polish(xb, seconds=min(3.0, rem - 9.0), init_radius=2e-3)
            bb = max_cos(xb)
        except Exception:
            pass
    # try symmetric seeds
    sigmas = [1e-4, 1e-3, 3e-3, 1e-2]
    for vs, Xs in sym_seeds[:3]:
        for sig in sigmas:
            rem = BUDGET - (time.time() - t0)
            if rem < 25.0:
                break
            try:
                y = Xs + np.random.default_rng(int(1000 + sig * 1e6) + len(sym_seeds)).normal(size=Xs.shape) * sig
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                y = smooth_anneal(y, iters=220, p_start=25, p_end=70, lr=0.04)
                rem2 = BUDGET - (time.time() - t0)
                if rem2 < 3.0:
                    break
                y2 = lp_polish(y, seconds=min(2.2, rem2 - 2.0), init_radius=min(0.02, max(0.004, sig * 2)))
                v = max_cos(y2)
                if v < bb - 1e-14:
                    xb = y2.copy(); bb = v
            except Exception:
                continue
        rem = BUDGET - (time.time() - t0)
        if rem < 25.0:
            break
    # Phase 3: basin hopping from current best (mix of symmetric-derived and record)
    xc = xb.copy(); bc = bb
    T = 4e-05
    it = 0
    while True:
        rem = BUDGET - (time.time() - t0)
        if rem < 4.0:
            break
        it += 1
        y = xc.copy()
        mode = it % 6
        if mode == 0:
            y = y + rng.normal(size=y.shape) * 0.008
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            init_r = 0.008; sm = False
        elif mode == 1:
            y = y + rng.normal(size=y.shape) * 0.03
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            init_r = 0.015; sm = True
        elif mode == 2:
            k = 2 + (it % 2)
            idx = rng.choice(n, size=k, replace=False)
            R = rng.normal(size=(k, d)); R /= np.linalg.norm(R, axis=1, keepdims=True)
            y[idx] = R
            init_r = 0.03; sm = True
        elif mode == 3:
            G = y @ y.T; np.fill_diagonal(G, -2.0)
            m = G.max(); np.fill_diagonal(G, 1.0)
            contacts = np.sum(G > m - 1e-7, axis=1) - 1
            order = np.argsort(contacts)
            for r_ in order[:3]:
                R = rng.normal(size=(7000, d)); R /= np.linalg.norm(R, axis=1, keepdims=True)
                S = R @ y.T; S[:, r_] = -10.0
                sc = S.max(axis=1)
                y[r_] = R[int(np.argmin(sc))]
            y = y + rng.normal(size=y.shape) * 0.005
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            init_r = 0.015; sm = True
        elif mode == 4:
            G = y @ y.T; np.fill_diagonal(G, -2.0)
            m = G.max(); np.fill_diagonal(G, 1.0)
            contacts = np.sum(G > m - 1e-7, axis=1) - 1
            top = int(np.argmax(contacts))
            sims = G[top].copy(); sims[top] = -10
            nn = np.argsort(-sims)[:10]
            grp = np.concatenate([[top], nn])
            y[grp] = y[grp] + rng.normal(size=(len(grp), d)) * 0.12
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            init_r = 0.025; sm = True
        else:
            y = y + rng.normal(size=y.shape) * 0.10
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            init_r = 0.03; sm = True
        if sm:
            y = smooth_anneal(y, iters=260, p_start=25, p_end=70, lr=0.04)
        rem2 = BUDGET - (time.time() - t0)
        if rem2 < 2.5:
            break
        budg = min(2.6, rem2 - 1.0)
        if budg < 0.8:
            budg = max(0.8, rem2 - 0.5)
        y2 = lp_polish(y, seconds=float(budg), init_radius=init_r)
        v = max_cos(y2)
        if v < bb - 1e-14:
            xb = y2.copy(); bb = v; xc = y2.copy(); bc = v
        else:
            if v < bc or rng.random() < np.exp((bc - v) / T):
                xc = y2.copy(); bc = v
        T *= 0.9995
        if T < 8e-06:
            T = 8e-06
    remf = BUDGET - (time.time() - t0)
    if remf > 2.0:
        try:
            yf = lp_polish(xb, seconds=min(remf - 0.5, 4.0), init_radius=1e-3)
            vf = max_cos(yf)
            if vf < bb:
                xb = yf; bb = vf
        except Exception:
            pass
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    return xb
