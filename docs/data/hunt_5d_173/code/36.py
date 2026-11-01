import time
import numpy as np
from scipy.optimize import linprog, linear_sum_assignment
from scipy.sparse import coo_matrix

N = 173
D = 5
_IU0, _IU1 = np.triu_indices(N, 1)

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, deadline, radius0=2e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = radius0
    iu0, iu1 = _IU0, _IU1
    if n != N:
        iu0, iu1 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt > 2600:
            idx = np.where(sel)[0]
            order = np.argsort(c_all[idx])[::-1][:2600]
            msel = np.zeros_like(sel, dtype=bool)
            msel[idx[order]] = True
            sel = msel
        I = iu0[sel]; J = iu1[sel]
        csel = c_all[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
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
    return x, best

def _smooth_anneal(x, steps, T0, T1, step0, step1):
    n, d = x.shape
    x = x.copy()
    for s in range(steps):
        frac = s / max(1, steps - 1)
        T = T0 * ((T1 / T0) ** frac) if T0 != T1 else T0
        step = step0 + (step1 - step0) * frac
        G = x @ x.T
        np.fill_diagonal(G, -1e9)
        m = float(G.max())
        W = np.exp((G - m) / max(T, 1e-12))
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        grad = (W @ x) / rs
        gdot = (grad * x).sum(axis=1, keepdims=True)
        gt = grad - gdot * x
        x = x - step * gt
        xn = np.linalg.norm(x, axis=1, keepdims=True)
        xn[xn == 0] = 1.0
        x = x / xn
    return x

def _contact_mask(x, tol=1e-7):
    G = x @ x.T
    c = G[_IU0, _IU1]
    m = float(c.max())
    return (m - c) < tol

def _void_fill(rem, k, rng, n_cand=3000):
    d = rem.shape[1]
    cand = rng.standard_normal((n_cand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    s_max = (cand @ rem.T).max(axis=1)
    chosen = []
    for _ in range(k):
        idx = int(np.argmin(s_max))
        pt = cand[idx].copy()
        chosen.append(pt)
        upd = cand @ pt
        np.maximum(s_max, upd, out=s_max)
        s_max[idx] = 2.0
    return np.stack(chosen, axis=0)

def _mutate(parent_x, rng):
    n, d = parent_x.shape
    x_try = parent_x.copy()
    r = rng.random()
    if r < 0.50:
        k = int(rng.integers(2, 9))
        G = x_try @ x_try.T
        np.fill_diagonal(G, -2.0)
        mi = G.max(axis=1)
        order = np.argsort(-mi)
        pool = order[:32]
        sel = rng.choice(pool, size=min(k, len(pool)), replace=False)
        mask = np.ones(n, dtype=bool)
        mask[sel] = False
        rem = x_try[mask]
        try:
            new_pts = _void_fill(rem, len(sel), rng, n_cand=3000)
        except Exception:
            new_pts = rng.standard_normal((len(sel), d))
            new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
        x_try[sel] = new_pts
        sigma = 0.005 + 0.015 * rng.random()
        x_try = x_try + sigma * rng.standard_normal(x_try.shape)
        x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
    elif r < 0.72:
        sigma = 0.04 + 0.20 * rng.random()
        if rng.random() < 0.15:
            sigma = 0.30 + 0.20 * rng.random()
        x_try = x_try + sigma * rng.standard_normal(x_try.shape)
        x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
    elif r < 0.87:
        # worst-point to hole
        G = x_try @ x_try.T
        np.fill_diagonal(G, -2.0)
        mi = G.max(axis=1)
        worst = int(np.argmax(mi))
        S = 12000
        samp = rng.standard_normal((S, d))
        samp /= np.linalg.norm(samp, axis=1, keepdims=True)
        others = np.delete(x_try, worst, axis=0)
        s_max = (samp @ others.T).max(axis=1)
        bi = int(np.argmin(s_max))
        x_try[worst] = samp[bi]
        nbrs = np.argsort(-G[worst])[:6]
        x_try[nbrs] += 0.06 * rng.standard_normal((len(nbrs), d))
        x_try[nbrs] /= np.linalg.norm(x_try[nbrs], axis=1, keepdims=True)
    else:
        # cap restart
        u = rng.standard_normal(d); u /= np.linalg.norm(u)
        proj = x_try @ u
        k = int(rng.integers(4, 14))
        idx = np.argsort(-proj)[:k] if rng.random() < 0.5 else np.argsort(proj)[:k]
        rnd = rng.standard_normal((k, d))
        rnd /= np.linalg.norm(rnd, axis=1, keepdims=True)
        x_try[idx] = rnd
        x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
    return x_try

def _recombine(A, B, rng):
    # orthogonal Procrustes align B to A
    M = B.T @ A
    try:
        u, s, vt = np.linalg.svd(M)
        Q = u @ vt
    except Exception:
        Q = np.eye(A.shape[1])
    Bal = B @ Q
    # assignment on Euclidean distance
    # ||a-b||^2 = 2-2<a,b> for unit vectors
    G = A @ Bal.T
    G = np.clip(G, -1.0, 1.0)
    Dist = 2.0 - 2.0 * G
    try:
        ri, ci = linear_sum_assignment(Dist)
    except Exception:
        return A.copy()
    Bmatched = np.empty_like(A)
    Bmatched[ri] = Bal[ci]
    # spatial crossover: halfspace split
    u2 = rng.standard_normal(A.shape[1]); u2 /= np.linalg.norm(u2)
    proj = A @ u2
    med = float(np.median(proj))
    # randomize side
    if rng.random() < 0.5:
        takeA = proj > med
    else:
        takeA = proj <= med
    C = np.where(takeA[:, None], A, Bmatched)
    C = C + 0.008 * rng.standard_normal(C.shape)
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    return C

def solve():
    t0 = time.time()
    BUDGET = 168.0
    rng = np.random.default_rng(7)
    base = None
    best0 = 2.0
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (N, D):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    v = _max_cos(a)
                    if v < best0:
                        best0 = v
                        base = a.copy()
        except Exception:
            pass
    if base is None:
        base = rng.standard_normal((N, D))
        base /= np.linalg.norm(base, axis=1, keepdims=True)
        best0 = _max_cos(base)
    # initial polish of incumbent
    if BUDGET - (time.time() - t0) > 12:
        base, best0 = _lp_polish(base, time.time() + 3.0, radius0=1e-3)

    POOL = 14
    pool_x = []
    pool_v = []
    pool_m = []
    pool_x.append(base.copy()); pool_v.append(best0); pool_m.append(_contact_mask(base))
    x_best = base.copy(); v_best = best0

    # build diverse initial pool with quick repel + short polish
    sigmas = [0.02, 0.05, 0.10, 0.20, 0.35]
    klist = [3, 5, 8, 11]
    pi = 0
    while len(pool_x) < POOL and (time.time() - t0) < 42.0:
        pi += 1
        cand = base.copy()
        if pi % 3 == 0:
            k = klist[(pi // 3) % len(klist)]
            G = cand @ cand.T
            np.fill_diagonal(G, -2.0)
            mi = G.max(axis=1)
            order = np.argsort(-mi)
            sel = rng.choice(order[:32], size=k, replace=False)
            msk = np.ones(N, dtype=bool); msk[sel] = False
            rem = cand[msk]
            try:
                cand[sel] = _void_fill(rem, k, rng)
            except Exception:
                pass
            cand = _smooth_anneal(cand, 250, 0.02, 0.004, 0.3, 0.12)
        else:
            sigma = sigmas[pi % len(sigmas)]
            cand = cand + sigma * rng.standard_normal(cand.shape)
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            cand = _smooth_anneal(cand, 300, 0.025, 0.003, 0.35, 0.12)
        remain = BUDGET - (time.time() - t0)
        if remain < 8:
            break
        cand_p, v_p = _lp_polish(cand, time.time() + min(1.4, max(0.6, remain - 6.0)), radius0=2e-3)
        m_p = _contact_mask(cand_p)
        # diversity: require >=20 edges from existing unless better
        dmin = min(int(np.count_nonzero(m_p != mm)) for mm in pool_m)
        if dmin < 20 and v_p >= min(pool_v) - 1e-12:
            # still allow occasionally to fill pool if very early
            if len(pool_x) < 6 or rng.random() < 0.2:
                pass
            else:
                continue
        pool_x.append(cand_p); pool_v.append(v_p); pool_m.append(m_p)
        if v_p < v_best:
            v_best = v_p; x_best = cand_p.copy()

    # sort pool by value
    order = np.argsort(np.array(pool_v))
    pool_x = [pool_x[i] for i in order]
    pool_v = [pool_v[i] for i in order]
    pool_m = [pool_m[i] for i in order]

    it = 0
    while True:
        elapsed = time.time() - t0
        remain = BUDGET - elapsed
        if remain < 6.0:
            break
        it += 1
        frac = elapsed / BUDGET
        T = 2e-4 * ((1e-6 / 2e-4) ** frac)
        # parent selection
        r = rng.random()
        if r < 0.35 and len(pool_x) >= 2:
            # recombination of two parents
            # bias one parent to best
            if rng.random() < 0.5:
                ia = 0
            else:
                ia = int(rng.integers(len(pool_x)))
            ib = int(rng.integers(len(pool_x)))
            if ib == ia:
                ib = (ib + 1) % len(pool_x)
            A = pool_x[ia]; B = pool_x[ib]
            # random order
            if rng.random() < 0.5:
                A, B = B, A
            try:
                child0 = _recombine(A, B, rng)
            except Exception:
                child0 = A.copy()
            a1 = 30.0 + rng.random() * 25.0
            child = _smooth_anneal(child0, 450, 0.02 + 0.02 * rng.random(), 0.002 + 0.002 * rng.random(), 0.35, 0.12)
        else:
            if rng.random() < 0.4:
                par = pool_x[0]
            else:
                par = pool_x[int(rng.integers(len(pool_x)))]
            child0 = _mutate(par, rng)
            # brief repel with randomized temps
            T0 = 0.02 + 0.02 * rng.random()
            T1 = 0.002 + 0.003 * rng.random()
            st = 400 + int(rng.integers(400))
            child = _smooth_anneal(child0, st, T0, T1, 0.35 + 0.15 * rng.random(), 0.10 + 0.05 * rng.random())
        # polish
        lp_time = min(1.8, remain - 4.5)
        if lp_time < 0.5:
            break
        child_p, v_p = _lp_polish(child, time.time() + lp_time, radius0=2.5e-3)
        if v_p < v_best - 1e-15:
            v_best = v_p; x_best = child_p.copy()
        try:
            m_p = _contact_mask(child_p)
        except Exception:
            continue
        # diversity check
        dists = [int(np.count_nonzero(m_p != mm)) for mm in pool_m]
        dmin = min(dists) if dists else 10**9
        imin = int(np.argmin(dists)) if dists else 0
        if dmin < 20:
            if v_p < pool_v[imin]:
                pool_x[imin] = child_p; pool_v[imin] = v_p; pool_m[imin] = m_p
                # re-sort occasionally
                if it % 5 == 0:
                    order = np.argsort(np.array(pool_v))
                    pool_x = [pool_x[i] for i in order]
                    pool_v = [pool_v[i] for i in order]
                    pool_m = [pool_m[i] for i in order]
            continue
        # far enough: consider insertion
        if len(pool_x) < POOL:
            pool_x.append(child_p); pool_v.append(v_p); pool_m.append(m_p)
            order = np.argsort(np.array(pool_v))
            pool_x = [pool_x[i] for i in order]
            pool_v = [pool_v[i] for i in order]
            pool_m = [pool_m[i] for i in order]
        else:
            worst = pool_v[-1]
            if v_p < worst:
                pool_x[-1] = child_p; pool_v[-1] = v_p; pool_m[-1] = m_p
                order = np.argsort(np.array(pool_v))
                pool_x = [pool_x[i] for i in order]
                pool_v = [pool_v[i] for i in order]
                pool_m = [pool_m[i] for i in order]
            else:
                # Metropolis accept worse for diversity
                prob = float(np.exp(-(v_p - worst) / max(T, 1e-12)))
                if rng.random() < prob:
                    # replace random non-best member to keep diversity
                    idx = int(rng.integers(1, len(pool_x))) if len(pool_x) > 1 else 0
                    pool_x[idx] = child_p; pool_v[idx] = v_p; pool_m[idx] = m_p

    remain = BUDGET - (time.time() - t0)
    if remain > 2.0:
        x_best, v_best = _lp_polish(x_best, time.time() + remain - 0.6, radius0=5e-4)
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
