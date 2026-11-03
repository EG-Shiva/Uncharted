import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _contact_info(X, thr=1e-7):
    n = X.shape[0]
    G = X @ X.T
    iu = np.triu_indices(n, 1)
    c = G[iu]
    m = float(c.max())
    sel = c > m - thr
    deg = np.zeros(n, dtype=float)
    I = iu[0][sel]; J = iu[1][sel]
    # vectorized accumulation
    if len(I):
        np.add.at(deg, I, 1)
        np.add.at(deg, J, 1)
    return m, deg, sel, iu

def _jacc_dist(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return 1.0 - float(inter) / float(union)

def _refine_hole(z, Xk, steps=120, T=0.005, lr0=0.06):
    z = z / max(np.linalg.norm(z), 1e-12)
    for it in range(steps):
        dots = Xk @ z
        m = float(dots.max())
        w = np.exp((dots - m) / T)
        s = float(w.sum())
        if s == 0:
            break
        w /= s
        g = Xk.T @ w
        gt = g - float(g @ z) * z
        nrm = float(np.linalg.norm(gt))
        if nrm < 1e-12:
            break
        step = lr0 * (0.985 ** it)
        z = z - step * gt / nrm
        nz = float(np.linalg.norm(z))
        if nz < 1e-12:
            break
        z = z / nz
        if it % 60 == 59:
            T *= 0.8
    return z / max(np.linalg.norm(z), 1e-12)

def _find_hole_vs(Xk, rng, n_sample=20000):
    d = Xk.shape[1]
    R = rng.normal(size=(n_sample, d))
    R = _norm(R)
    D = R @ Xk.T
    worst = D.max(axis=1)
    # take 5 best candidates, refine each, keep best
    if n_sample >= 5:
        cand_idx = np.argpartition(worst, 5)[:5]
    else:
        cand_idx = np.array([int(np.argmin(worst))])
    best_z = None
    best_v = 1e9
    for ci in cand_idx:
        z0 = R[int(ci)].copy()
        z = _refine_hole(z0, Xk)
        v = float((Xk @ z).max())
        if v < best_v:
            best_v = v
            best_z = z
    return best_z

def _weighted_tangential_polish(X, steps=450, T0=0.0035, T1=0.0007, lr0=0.025, lr1=0.003, mom=0.85):
    X = _norm(X)
    n = X.shape[0]
    V = np.zeros_like(X)
    for k in range(steps):
        frac = k / max(steps - 1, 1)
        T = T0 * (1 - frac) + T1 * frac
        lr = lr0 * (1 - frac) + lr1 * frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp((G - m) / T)
        np.fill_diagonal(W, 0.0)
        s = float(W.sum()) / n + 1e-12
        Gvec = (W @ X) / s
        dot = np.sum(Gvec * X, axis=1, keepdims=True)
        Gt = Gvec - dot * X
        V = mom * V + (1.0 - mom) * Gt
        X = X - lr * V
        X = _norm(X)
    return X

def _lp_polish(X, seconds, deadline=None, r0=8e-4):
    X = _norm(X)
    n, d = X.shape
    best = _max_cos(X)
    radius = r0
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if deadline is not None and time.time() > deadline:
            break
        Gfull = X @ X.T
        np.fill_diagonal(Gfull, -2.0)
        cfull = Gfull[iu]
        m = float(cfull.max())
        thresh = m - max(20 * radius, 1e-9)
        sel = cfull > thresh
        if not np.any(sel):
            radius *= 0.5
            continue
        I = iu[0][sel]; J = iu[1][sel]; csel = cfull[sel]
        p = len(I)
        if p > 3000:
            idx = np.argpartition(csel, p - 3000)[p - 3000:]
            I = I[idx]; J = J[idx]; csel = csel[idx]
            p = 3000
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, n * d + 1)).tocsr()
        obj = np.zeros(n * d + 1); obj[-1] = 1.0
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
        Y = X + res.x[:-1].reshape(n, d)
        Y = _norm(Y)
        v = _max_cos(Y)
        if v < best:
            X, best = Y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def solve():
    t0 = time.time()
    HARD = t0 + 168.0
    EXP_END = t0 + 130.0
    rng = np.random.default_rng(7)
    try:
        X0 = np.load("best.npy").astype(float)
    except Exception:
        X0 = np.load("record.npy").astype(float)
    X0 = _norm(X0)
    n, d = X0.shape
    best_val = _max_cos(X0)
    x_best = X0.copy()
    _, _, best_mask, _ = _contact_info(x_best, thr=1e-7)
    archive = [{'x': x_best.copy(), 'v': best_val, 'mask': best_mask.copy()}]
    sizes = [3, 5, 8]
    trial = 0
    while time.time() < EXP_END and time.time() < HARD - 34:
        trial += 1
        # pick base
        if len(archive) > 1:
            if trial % 4 == 0:
                vs = [a['v'] for a in archive]
                base_idx = int(np.argmin(vs))
            else:
                base_idx = int(rng.integers(len(archive)))
        else:
            base_idx = 0
        base = archive[base_idx]
        bx = base['x']
        _, deg, base_mask, _ = _contact_info(bx, thr=1e-7)
        # decide surgery type: every 6th trial do rattler absorption
        rattlers = np.where(deg < 0.5)[0]
        do_rattler = (trial % 6 == 5) and (len(rattlers) > 0)
        if do_rattler:
            # delete all rattlers simultaneously (cap at 9)
            victims = list(rattlers[:9])
            rng.shuffle(victims)
            k = len(victims)
            n_samp = 16000
            steps = 380
        else:
            k = sizes[int(rng.integers(len(sizes)))]
            k = min(k, n - 2)
            w = (deg.astype(float) + 1.0) ** 3.0
            boost = np.where(deg > 20, 3.0, 1.0)
            boost = np.where(deg >= 25, boost * 1.5, boost)
            w = w * boost
            s = w.sum()
            if s <= 0:
                w = np.ones(n) / n
            else:
                w = w / s
            victims = rng.choice(n, size=k, replace=False, p=w)
            victims = list(victims)
            rng.shuffle(victims)
            n_samp = 20000
            steps = 450 if k <= 5 else 380
        retained = np.delete(bx, victims, axis=0)
        reinserted = []
        ok = True
        for vi in victims:
            if time.time() > EXP_END or time.time() > HARD - 34:
                ok = False
                break
            Xcur = retained if len(reinserted) == 0 else np.vstack([retained] + reinserted)
            z = _find_hole_vs(Xcur, rng, n_sample=n_samp)
            reinserted.append(z.reshape(1, d))
        if not ok or len(reinserted) != len(victims):
            continue
        cand = bx.copy()
        for vi, z in zip(victims, reinserted):
            cand[vi] = z.ravel()
        cand = _norm(cand)
        cand = _weighted_tangential_polish(cand, steps=steps)
        v = _max_cos(cand)
        _, _, cand_mask, _ = _contact_info(cand, thr=1e-7)
        if v < best_val - 1e-12:
            best_val = v
            x_best = cand.copy()
            best_mask = cand_mask.copy()
        min_dist = min(_jacc_dist(cand_mask, a['mask']) for a in archive)
        is_best = v < min(a['v'] for a in archive) - 1e-12
        if len(archive) < 3:
            worst_allow = best_val + 0.002
            if v < worst_allow and (min_dist > 0.02 or is_best):
                archive.append({'x': cand.copy(), 'v': v, 'mask': cand_mask.copy()})
            elif is_best:
                archive.append({'x': cand.copy(), 'v': v, 'mask': cand_mask.copy()})
        else:
            if is_best:
                idx_worst = int(np.argmax([a['v'] for a in archive]))
                archive[idx_worst] = {'x': cand.copy(), 'v': v, 'mask': cand_mask.copy()}
            else:
                worst_v = max(a['v'] for a in archive)
                if v < worst_v + 0.0004 and min_dist > 0.03:
                    if v < worst_v:
                        idx_worst = int(np.argmax([a['v'] for a in archive]))
                        archive[idx_worst] = {'x': cand.copy(), 'v': v, 'mask': cand_mask.copy()}
        if time.time() > EXP_END:
            break
    for a in archive:
        if a['v'] < best_val:
            best_val = a['v']; x_best = a['x'].copy()
    remain = HARD - time.time() - 1.0
    if remain > 3:
        x_best = _lp_polish(x_best, seconds=min(remain, 32.0), deadline=HARD - 0.5, r0=8e-4)
    x_best = _norm(x_best)
    return x_best
