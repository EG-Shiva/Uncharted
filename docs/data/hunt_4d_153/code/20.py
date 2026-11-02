import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, budget):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < budget and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        m = c_all.max()
        sel = c_all > m - max(20 * radius, 1e-9)
        I = iu0[0][sel]; J = iu0[1][sel]
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
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() - start >= budget:
            break
    return x, best

def _adam_soft(x0, steps, beta0, beta1, lr0, lr1, Iu, Ju, free=None):
    n, d = x0.shape
    x = x0.copy()
    m1 = np.zeros_like(x)
    v1 = np.zeros_like(x)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    lrb = np.log(lr1 / lr0) / max(steps - 1, 1)
    betb = np.log(beta1 / beta0) / max(steps - 1, 1)
    is_free = np.ones(n, dtype=bool) if free is None else free
    has_fix = not np.all(is_free)
    for s in range(steps):
        beta = beta0 * np.exp(betb * s)
        lr = lr0 * np.exp(lrb * s)
        G = x @ x.T
        g = G[Iu, Ju]
        gmax = g.max()
        e = np.exp(beta * (g - gmax))
        Z = e.sum()
        w = e / Z
        grad = np.zeros_like(x)
        wxJ = w[:, None] * x[Ju]
        wxI = w[:, None] * x[Iu]
        np.add.at(grad, Iu, wxJ)
        np.add.at(grad, Ju, wxI)
        dot = (grad * x).sum(axis=1, keepdims=True)
        gt = grad - dot * x
        if has_fix:
            gt[~is_free] = 0.0
        m1 = b1 * m1 + (1 - b1) * gt
        v1 = b2 * v1 + (1 - b2) * (gt * gt)
        c1 = 1 - b1 ** (s + 1); c2 = 1 - b2 ** (s + 1)
        step = (m1 / c1) / (np.sqrt(v1 / c2) + eps)
        if has_fix:
            step[~is_free] = 0.0
        x = x - lr * step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        np.maximum(nr, 1e-12, out=nr)
        x /= nr
    return x

def _adam_topk(x0, steps, K, lr0, lr1, Iu, Ju, free=None, beta=80.0):
    n, d = x0.shape
    x = x0.copy()
    m1 = np.zeros_like(x)
    v1 = np.zeros_like(x)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    lrb = np.log(lr1 / lr0) / max(steps - 1, 1)
    is_free = np.ones(n, dtype=bool) if free is None else free
    has_fix = not np.all(is_free)
    P = len(Iu)
    K = min(K, P)
    for s in range(steps):
        lr = lr0 * np.exp(lrb * s)
        G = x @ x.T
        g = G[Iu, Ju]
        # top-K indices
        part = np.argpartition(g, P - K)[P - K:]
        gs = g[part]
        # softmax within topK for stability, high beta -> near uniform/hard
        gm = gs.max()
        e = np.exp(beta * (gs - gm))
        wv = e / e.sum()
        Is = Iu[part]; Js = Ju[part]
        grad = np.zeros_like(x)
        np.add.at(grad, Is, wv[:, None] * x[Js])
        np.add.at(grad, Js, wv[:, None] * x[Is])
        dot = (grad * x).sum(axis=1, keepdims=True)
        gt = grad - dot * x
        if has_fix:
            gt[~is_free] = 0.0
        m1 = b1 * m1 + (1 - b1) * gt
        v1 = b2 * v1 + (1 - b2) * (gt * gt)
        c1 = 1 - b1 ** (s + 1); c2 = 1 - b2 ** (s + 1)
        step = (m1 / c1) / (np.sqrt(v1 / c2) + eps)
        if has_fix:
            step[~is_free] = 0.0
        x = x - lr * step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        np.maximum(nr, 1e-12, out=nr)
        x /= nr
    return x

def solve():
    t0 = time.time()
    HARD = 172.0
    rng = np.random.default_rng(987654321)
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.normal(size=(153, 4))
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (153, 4):
        xb = xb.reshape(153, 4)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = _max_cos(xb)
    x_best = xb.copy()
    n, d = 153, 4
    Iu, Ju = np.triu_indices(n, 1)
    archive = [x_best.copy()]
    arch_val = [best_val]

    def consider(y, v):
        nonlocal x_best, best_val
        improved = False
        if v < best_val - 1e-12:
            x_best = y.copy()
            best_val = v
            improved = True
            archive.append(y.copy())
            arch_val.append(v)
            if len(archive) > 6:
                # keep best 6
                od = np.argsort(np.array(arch_val))
                archive[:] = [archive[i] for i in od[:6]]
                arch_val[:] = [arch_val[i] for i in od[:6]]
        elif v < best_val + 0.004:
            # diverse archive member if far
            if len(archive) < 6 and rng.random() < 0.15:
                archive.append(y.copy())
                arch_val.append(v)
        return improved

    it = 0
    # short snap of incumbent if time allows (already near optimum, keep tiny)
    # main hopping / surgery loop
    while time.time() - t0 < 140.0:
        it += 1
        if HARD - (time.time() - t0) < 32:
            break
        mode = it % 10
        y = None
        # contact stats of incumbent for targeted moves
        Gbest = x_best @ x_best.T
        gall = Gbest[Iu, Ju]
        mx = gall.max()
        tight = gall > mx - 1e-7
        cnt = np.zeros(n, dtype=int)
        np.add.at(cnt, Iu[tight], 1)
        np.add.at(cnt, Ju[tight], 1)
        ratt = np.where(cnt <= 2)[0]
        hubs = np.where(cnt >= 7)[0]

        if mode == 0:  # global small kick + soft
            y = x_best + rng.normal(0, 0.02, size=(n, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _adam_soft(y, 700, 40.0, 300.0, 0.006, 0.0004, Iu, Ju)
        elif mode == 1:  # cavity small: reinit 10-16 nearest to random cap
            k = int(rng.integers(10, 17))
            c = rng.normal(size=4); c /= np.linalg.norm(c)
            prox = x_best @ c
            idx = np.argpartition(prox, n - k)[n - k:]
            free = np.zeros(n, dtype=bool); free[idx] = True
            y = x_best.copy()
            y[idx] = rng.normal(size=(k, 4))
            y[idx] /= np.linalg.norm(y[idx], axis=1, keepdims=True)
            y = _adam_soft(y, 650, 30.0, 350.0, 0.008, 0.0005, Iu, Ju, free=free)
            y = _adam_soft(y, 250, 80.0, 300.0, 0.003, 0.0003, Iu, Ju)
        elif mode == 2:  # global medium + topK
            y = x_best + rng.normal(0, 0.05, size=(n, d))
            if len(ratt) > 0:
                for rr in ratt:
                    if rng.random() < 0.7:
                        y[rr] = rng.normal(size=4)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _adam_topk(y, 650, 700, 0.006, 0.0004, Iu, Ju, beta=60.0)
        elif mode == 3:  # cavity medium with topK subset
            k = int(rng.integers(18, 27))
            seed = int(rng.integers(n))
            dist = Gbest[seed]  # cosine to seed
            idx = np.argpartition(dist, n - k)[n - k:]
            free = np.zeros(n, dtype=bool); free[idx] = True
            y = x_best.copy()
            y[idx] += rng.normal(0, 0.35, size=(k, 4))
            y[idx] /= np.linalg.norm(y[idx], axis=1, keepdims=True)
            y = _adam_topk(y, 550, 650, 0.008, 0.0005, Iu, Ju, free=free, beta=80.0)
            y = _adam_soft(y, 250, 50.0, 280.0, 0.003, 0.0003, Iu, Ju)
        elif mode == 4:  # large global replacement
            y = x_best.copy()
            y += rng.normal(0, 0.10, size=(n, d))
            nr = int(rng.integers(1, 4))
            rep = rng.choice(n, size=nr, replace=False)
            y[rep] = rng.normal(size=(nr, 4))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _adam_soft(y, 800, 25.0, 350.0, 0.007, 0.0004, Iu, Ju)
        elif mode == 5:  # cavity reinit large-ish + soft
            k = int(rng.integers(14, 22))
            c = rng.normal(size=4); c /= np.linalg.norm(c)
            prox = x_best @ c
            idx = np.argpartition(prox, n - k)[n - k:]
            free = np.zeros(n, dtype=bool); free[idx] = True
            y = x_best.copy()
            # half reinit, half big kick
            for ii in idx:
                if rng.random() < 0.5:
                    y[ii] = rng.normal(size=4)
                else:
                    y[ii] += rng.normal(0, 0.5, size=4)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _adam_soft(y, 600, 35.0, 320.0, 0.008, 0.0004, Iu, Ju, free=free)
            y = _adam_soft(y, 250, 60.0, 300.0, 0.003, 0.0003, Iu, Ju)
        elif mode == 6:  # crossover of archive
            if len(archive) >= 2:
                a = archive[int(rng.integers(len(archive)))]
                b = archive[int(rng.integers(len(archive)))]
                y = np.empty_like(x_best)
                mask = rng.random(n) < 0.5
                y[mask] = a[mask]
                y[~mask] = b[~mask]
                y += rng.normal(0, 0.03, size=(n, d))
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                y = _adam_soft(y, 700, 40.0, 300.0, 0.006, 0.0004, Iu, Ju)
            else:
                y = x_best + rng.normal(0, 0.03, size=(n, d))
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                y = _adam_soft(y, 700, 40.0, 300.0, 0.006, 0.0004, Iu, Ju)
        elif mode == 7:  # hub relief + rattler reinsertion
            y = x_best.copy()
            if len(hubs) > 0:
                y[hubs] += rng.normal(0, 0.06, size=(len(hubs), d))
            if len(ratt) > 0:
                for rr in ratt:
                    y[rr] = rng.normal(size=4)
            # also random 1-2 replacements
            rep = rng.choice(n, size=2, replace=False)
            y[rep] = rng.normal(size=(2, 4))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _adam_topk(y, 600, 550, 0.006, 0.0004, Iu, Ju, beta=100.0)
            y = _adam_soft(y, 200, 80.0, 300.0, 0.002, 0.0003, Iu, Ju)
        elif mode == 8:  # global small + topK narrow
            y = x_best + rng.normal(0, 0.015, size=(n, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _adam_topk(y, 700, 450, 0.005, 0.0003, Iu, Ju, beta=120.0)
        else:  # large cavity 26-34
            k = int(rng.integers(26, 35))
            c = rng.normal(size=4); c /= np.linalg.norm(c)
            prox = x_best @ c
            idx = np.argpartition(prox, n - k)[n - k:]
            free = np.zeros(n, dtype=bool); free[idx] = True
            y = x_best.copy()
            y[idx] = rng.normal(size=(k, 4))
            y[idx] /= np.linalg.norm(y[idx], axis=1, keepdims=True)
            y = _adam_soft(y, 650, 25.0, 300.0, 0.009, 0.0005, Iu, Ju, free=free)
            y = _adam_soft(y, 300, 60.0, 320.0, 0.003, 0.0003, Iu, Ju)

        v = _max_cos(y)
        # LP polish only if promising
        thresh = best_val + (0.005 if mode in (4, 9) else 0.002)
        if v < thresh:
            remain = HARD - (time.time() - t0) - 22.0
            if remain < 0.8:
                consider(y, v)
                break
            b = min(2.5, max(0.9, remain / 10.0))
            # less budget for far candidates
            if v > best_val + 0.001:
                b = min(b, 1.0)
            yp, vp = _lp_polish(y, b)
            consider(yp, vp)
        if time.time() - t0 > HARD - 25:
            break

    remain = HARD - (time.time() - t0) - 1.5
    if remain > 2.0:
        remain = min(remain, 28.0)
        x_best, best_val = _lp_polish(x_best, remain)
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
