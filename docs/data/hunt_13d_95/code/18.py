import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def lp_polish(x, seconds, t_end=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    if t_end is None:
        t_end = start + seconds
    else:
        t_end = min(t_end, start + seconds)
    iu = np.triu_indices(n, 1)
    while time.time() < t_end and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I0, J0 = iu[0][sel], iu[1][sel]
        p0 = len(I0)
        if p0 == 0:
            break
        if p0 > 3500:
            cc = c[sel]
            kp = np.argpartition(-cc, 3500)[:3500]
            I, J = I0[kp], J0[kp]
            csel = cc[kp]
            p = 3500
        else:
            I, J = I0, J0
            csel = c[sel]
            p = p0
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def adam_relax(x, steps=800, k0=25.0, k1=500.0, lr=0.004, active=None):
    n, d = x.shape
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    m1 = np.zeros_like(x)
    m2 = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    if active is not None:
        amask = np.zeros(n, dtype=bool)
        amask[np.asarray(active)] = True
    else:
        amask = None
    for t in range(1, steps + 1):
        kk = k0 * ((k1 / k0) ** (t / steps))
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = G.max()
        W = np.exp(kk * (G - gmax))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        if amask is not None:
            gt[~amask] = 0.0
        m1 = b1 * m1 + (1 - b1) * gt
        m2 = b2 * m2 + (1 - b2) * (gt * gt)
        m1h = m1 / (1 - b1 ** t)
        m2h = m2 / (1 - b2 ** t)
        step = lr * m1h / (np.sqrt(m2h) + eps)
        sn = np.sum(step * x, axis=1, keepdims=True)
        step = step - sn * x
        snorm = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        clip = np.minimum(1.0, 0.03 / snorm)
        step = step * clip
        if amask is not None:
            step[~amask] = 0.0
        x = x - step
        if amask is not None:
            x[amask] /= np.linalg.norm(x[amask], axis=1, keepdims=True)
        else:
            x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def greedy_reseed(X, idxs, rng, n_cand=3000):
    n, d = X.shape
    s = set(int(i) for i in idxs)
    Xn = X.copy()
    order = list(s)
    rng.shuffle(order)
    frozen_idx = [i for i in range(n) if i not in s]
    Xf = Xn[frozen_idx] if len(frozen_idx) > 0 else None
    placed = []
    for idx in order:
        cand = rng.standard_normal(size=(n_cand, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        if Xf is not None and len(Xf) > 0:
            mf = (cand @ Xf.T).max(axis=1)
        else:
            mf = np.full(n_cand, -2.0)
        if placed:
            Xp = Xn[np.array(placed)]
            mp = (cand @ Xp.T).max(axis=1)
            m = np.maximum(mf, mp)
        else:
            m = mf
        Xn[idx] = cand[int(np.argmin(m))]
        placed.append(idx)
    return Xn

def crossover_mix(A, B, rng):
    try:
        M = A.T @ B
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        Q = Vt.T @ U.T
        Bq = B @ Q
        from scipy.optimize import linear_sum_assignment
        cost = -(A @ Bq.T)
        r_ind, c_ind = linear_sum_assignment(cost)
        Bp = np.empty_like(Bq)
        Bp[r_ind] = Bq[c_ind]
    except Exception:
        Bp = B.copy()
    mask = rng.random(A.shape[0]) < 0.5
    C = A.copy()
    C[mask] = Bp[mask]
    nz = rng.standard_normal(C.shape) * 1e-3
    C = C + nz
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    return C

def solve():
    t0 = time.time()
    deadline = t0 + 172.0
    rng = np.random.default_rng(7)
    np.random.seed(7)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = rng.standard_normal(size=(95, 13))
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best = max_cos(xb)
    try:
        xr = np.load("record.npy").astype(float)
        xr = xr / np.linalg.norm(xr, axis=1, keepdims=True)
        vr = max_cos(xr)
        if vr < best:
            best, xb = vr, xr
    except Exception:
        pass
    n, d = xb.shape
    pop = [(best, xb.copy())]
    trial = 0
    patch_sizes = [5, 8, 10, 12, 6, 9, 14, 7]
    while time.time() < deadline - 20.0:
        trial += 1
        # choose base: mostly global best, sometimes other pop member
        if len(pop) > 1 and (trial % 5 == 0):
            bi = int(rng.integers(len(pop)))
            base = pop[bi][1].copy()
        else:
            base = xb.copy()
        G0 = base @ base.T
        np.fill_diagonal(G0, -2.0)
        rowmax = G0.max(axis=1)
        contacts = (G0 > rowmax.max() - 1e-7).sum(axis=1)
        mode = trial % 10
        if mode <= 5:
            k = patch_sizes[(trial // 1) % len(patch_sizes)]
            # pick center: worst rowmax half the time
            if rng.random() < 0.6:
                order_w = np.argsort(-rowmax)
                # random among top 8 worst
                top = order_w[:8]
                center = int(top[int(rng.integers(len(top)))])
            else:
                center = int(rng.integers(n))
            dots = base[center] @ base.T
            cluster = np.argsort(-dots)[:k]
            x_try = greedy_reseed(base, cluster, rng, n_cand=3000)
            # local refine of patch with frozen background
            x_try = adam_relax(x_try, steps=500, k0=30.0, k1=600.0, lr=0.005, active=cluster)
            # short global relax
            x_try = adam_relax(x_try, steps=600, k0=25.0, k1=500.0, lr=0.003)
            lp_sec = 1.6
        elif mode == 6:
            if len(pop) >= 2:
                a_i, b_i = rng.choice(len(pop), size=2, replace=False)
                A, B = pop[a_i][1], pop[b_i][1]
                x_try = crossover_mix(A, B, rng)
                x_try = adam_relax(x_try, steps=800, k0=25.0, k1=600.0, lr=0.004)
                lp_sec = 1.6
            else:
                # big shake
                noise = rng.standard_normal(size=base.shape)
                noise -= np.sum(noise * base, axis=1, keepdims=True) * base
                x_try = base + 0.15 * noise
                x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
                x_try = adam_relax(x_try, steps=900, k0=25.0, k1=500.0, lr=0.004)
                lp_sec = 1.6
        elif mode == 7:
            noise = rng.standard_normal(size=base.shape)
            noise -= np.sum(noise * base, axis=1, keepdims=True) * base
            sig = 0.04 if (trial % 2 == 0) else 0.10
            x_try = base + sig * noise
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            x_try = adam_relax(x_try, steps=800, k0=25.0, k1=500.0, lr=0.004)
            lp_sec = 1.5
        elif mode == 8:
            # rattler / low-contact kick: move lowest contact points strongly
            low = np.argsort(contacts)[:5]
            noise = rng.standard_normal(size=base.shape)
            noise -= np.sum(noise * base, axis=1, keepdims=True) * base
            sig = np.full(n, 0.01)
            sig[low] = 0.35
            x_try = base + sig[:, None] * noise
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            x_try = adam_relax(x_try, steps=900, k0=25.0, k1=600.0, lr=0.004)
            lp_sec = 1.6
        else:
            noise = rng.standard_normal(size=base.shape)
            noise -= np.sum(noise * base, axis=1, keepdims=True) * base
            x_try = base + 0.25 * noise
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            x_try = adam_relax(x_try, steps=1000, k0=20.0, k1=500.0, lr=0.005)
            lp_sec = 1.8
        remain = (deadline - 20.0) - time.time()
        if remain < 0.5:
            v = max_cos(x_try)
            if v < best - 1e-12:
                best, xb = v, x_try / np.linalg.norm(x_try, axis=1, keepdims=True)
            break
        x_try = lp_polish(x_try, seconds=min(lp_sec, remain))
        v = max_cos(x_try)
        if v < best - 1e-12:
            best = v
            xb = x_try
            pop.append((v, x_try.copy()))
            pop = sorted(pop, key=lambda z: z[0])[:6]
        elif v < best + 0.002 and len(pop) < 6:
            # keep diverse suboptimum if not duplicate
            dup = False
            for _, P in pop:
                if max_cos(0.5 * (x_try + P)) < 1e-12:
                    dup = True
            if not dup:
                pop.append((v, x_try.copy()))
                pop = sorted(pop, key=lambda z: z[0])[:6]
        if time.time() >= deadline - 20.0:
            break
    remain = deadline - time.time() - 0.8
    if remain > 2.0:
        xb = lp_polish(xb, seconds=remain)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    return xb
