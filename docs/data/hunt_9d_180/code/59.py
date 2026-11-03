import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def anneal(x, steps, T0, T1, lr0, rng):
    n, d = x.shape
    xc = x.copy()
    vel = np.zeros_like(xc)
    mu = 0.85
    for s in range(steps):
        frac = s / max(1, steps - 1)
        T = T0 * (T1 / T0) ** frac
        lr = lr0 * (1.0 - 0.5 * frac)
        G = xc @ xc.T
        np.fill_diagonal(G, -2.0)
        gmax = float(np.max(G))
        E = np.exp((G - gmax) / T)
        np.fill_diagonal(E, 0.0)
        row = E.sum(axis=1, keepdims=True) + 1e-12
        grad = E @ xc
        dot = np.sum(grad * xc, axis=1, keepdims=True)
        gtan = grad - dot * xc
        step = gtan / row
        nm = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        step = step * np.minimum(1.0, 0.7 / nm)
        vel = mu * vel + step
        xc = xc - lr * vel
        xc /= np.linalg.norm(xc, axis=1, keepdims=True)
    return xc, max_cos(xc)

def void_vec(others, rng, n_try=2000, n_ref=50):
    d = others.shape[1]
    cand = rng.randn(n_try, d)
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ others.T
    m = dots.max(axis=1)
    y = cand[int(np.argmin(m))].copy()
    cur = float(np.min(m))
    for k in range(n_ref):
        sig = 0.05 * (0.0005 / 0.05) ** (k / max(1, n_ref - 1))
        prop = y + sig * rng.randn(d)
        nrm = np.linalg.norm(prop)
        if nrm < 1e-12:
            continue
        prop /= nrm
        pm = float(np.max(prop @ others.T))
        if pm < cur:
            y = prop
            cur = pm
    return y

def lp_polish(x, seconds, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 8e-4
    t0 = time.time()
    iu = np.triu_indices(n, 1)
    it = 0
    while time.time() - t0 < seconds and time.time() < deadline and radius > 1e-14 and it < 25:
        it += 1
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        sel = c > m - max(30 * radius, 2e-9)
        idx = np.where(sel)[0]
        if idx.size == 0:
            break
        if idx.size > 2500:
            part = np.argpartition(c[idx], -2500)[-2500:]
            idx = idx[part]
        I, J = iu[0][idx], iu[1][idx]
        cs = c[idx]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": 4.0})
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
            x, best, radius = y, v, min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    deadline = t0 + 174.0
    rng = np.random.RandomState(12345)
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            c = np.asarray(np.load(fn), dtype=float)
            if c.shape == (180, 9):
                c = c / np.linalg.norm(c, axis=1, keepdims=True)
                cands.append((max_cos(c), c))
        except Exception:
            pass
    if not cands:
        x_best = rng.randn(180, 9)
        x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
        b_best = max_cos(x_best)
    else:
        cands.sort(key=lambda t: t[0])
        b_best, x_best = cands[0][0], cands[0][1].copy()
    n, d = 180, 9
    x_cur = x_best.copy()
    cur_val = b_best
    explore_end = t0 + 135.0
    it = 0
    noimp = 0
    while time.time() < explore_end and time.time() < deadline - 38:
        it += 1
        G = x_cur @ x_cur.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        cnt = (G >= m - 1e-7).sum(axis=1)
        k = 3 + (it % 3)
        ratt = np.where(cnt == 0)[0].tolist()
        rng.shuffle(ratt)
        victims = []
        for r in ratt:
            if len(victims) < k:
                victims.append(int(r))
        for h in np.argsort(-cnt):
            if len(victims) >= k:
                break
            if int(h) not in victims:
                victims.append(int(h))
        if (it % 7) == 0 and len(victims) > 0:
            victims[-1] = int(rng.randint(0, n))
            victims = list(dict.fromkeys(victims))
            while len(victims) < k:
                a = int(rng.randint(0, n))
                if a not in victims:
                    victims.append(a)
        victims = np.array(victims[:k], dtype=int)
        mask = np.ones(n, dtype=bool)
        mask[victims] = False
        tmp = x_cur[mask].copy()
        new_vecs = []
        for _ in victims:
            y = void_vec(tmp, rng)
            new_vecs.append(y)
            tmp = np.vstack([tmp, y[None, :]])
        x_new = x_cur.copy()
        for vi, y in zip(victims, new_vecs):
            x_new[int(vi)] = y
        x_new /= np.linalg.norm(x_new, axis=1, keepdims=True)
        if (it % 3) == 0:
            T0, T1, lr0, steps = 0.02, 0.002, 0.06, 160
        elif (it % 3) == 1:
            T0, T1, lr0, steps = 0.01, 0.001, 0.04, 160
        else:
            T0, T1, lr0, steps = 0.03, 0.003, 0.08, 130
        x_ann, v_ann = anneal(x_new, steps, T0, T1, lr0, rng)
        if v_ann < b_best - 1e-12:
            b_best = v_ann
            x_best = x_ann.copy()
            x_cur = x_ann.copy()
            cur_val = v_ann
            noimp = 0
        elif v_ann < cur_val:
            x_cur = x_ann.copy()
            cur_val = v_ann
            noimp = 0
        else:
            noimp += 1
            if v_ann < cur_val + 0.004 and (it % 4) == 0:
                x_cur = x_ann.copy()
                cur_val = v_ann
            if noimp > 28:
                x_cur = x_best.copy()
                cur_val = b_best
                noimp = 0
                idx = rng.choice(n, size=8, replace=False)
                x_cur[idx] += 0.05 * rng.randn(8, d)
                x_cur /= np.linalg.norm(x_cur, axis=1, keepdims=True)
                cur_val = max_cos(x_cur)
    remain = deadline - time.time()
    if remain > 10:
        try:
            xa, va = anneal(x_best, 200, 0.008, 0.0008, 0.03, rng)
            if va < b_best:
                b_best = va
                x_best = xa
        except Exception:
            pass
    remain = deadline - time.time()
    if remain > 7:
        try:
            x_best = lp_polish(x_best, min(remain - 1.0, 24.0), deadline - 0.5)
        except Exception:
            pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
