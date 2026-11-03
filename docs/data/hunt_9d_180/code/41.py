import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def contacts_count(x, gmax, tol=0.005):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    C = (G > gmax - tol).sum(axis=1)
    return C, G

def find_void(others, rng, n_try=6000, n_ref=200):
    d = others.shape[1]
    cand = rng.randn(n_try, d)
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ others.T
    m = np.max(dots, axis=1)
    bi = int(np.argmin(m))
    y = cand[bi].copy()
    cur = float(m[bi])
    for k in range(n_ref):
        frac = k / max(1, n_ref - 1)
        sig = 0.08 * (0.005 / 0.08) ** frac
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

def fast_relax(x, rng, steps, T0, T1, lr0, mu=0.80):
    n, d = x.shape
    x = x.copy()
    vel = np.zeros_like(x)
    # randomize slightly inside? no
    lT0 = np.log(T0); lT1 = np.log(T1)
    for s in range(steps):
        frac = s / max(1, steps - 1)
        T = float(np.exp(lT0 + (lT1 - lT0) * frac))
        lr = lr0 * (1.0 - 0.4 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = float(np.max(G))
        # weights
        E = np.exp((G - gmax) / T)
        np.fill_diagonal(E, 0.0)
        row = np.sum(E, axis=1, keepdims=True) + 1e-12
        S = (E @ x) / row
        # tangential component
        dot = np.sum(S * x, axis=1, keepdims=True)
        Tan = S - dot * x
        # limit step
        nm = np.linalg.norm(Tan, axis=1, keepdims=True) + 1e-12
        # normalize to at most 1
        Tan = Tan / np.maximum(nm, np.ones_like(nm) * 0.5) * np.minimum(nm, np.ones_like(nm))
        vel = mu * vel + Tan
        x = x - lr * vel
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def lp_polish(x, seconds, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < deadline and radius > 1e-14:
        Gfull = x @ x.T
        c = Gfull[iu]
        m = float(np.max(c))
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        csub = c[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csub, A_eq=A_eq, b_eq=np.zeros(n),
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
            x, best, radius = y, v, min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    deadline = t0 + 178.0
    rng = np.random.RandomState(7)
    try:
        xb = np.asarray(np.load("best.npy"), dtype=float)
    except Exception:
        xb = None
    try:
        xr = np.asarray(np.load("record.npy"), dtype=float)
    except Exception:
        xr = None
    cands = []
    for c in (xb, xr):
        if c is not None and c.shape == (180, 9):
            c = c / np.linalg.norm(c, axis=1, keepdims=True)
            cands.append((max_cos(c), c))
    if not cands:
        x0 = rng.randn(180, 9)
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        x_best = x0; b_best = max_cos(x0)
    else:
        cands.sort(key=lambda t: t[0])
        b_best, x_best = cands[0][0], cands[0][1].copy()
    n, d = 180, 9
    x_cur = x_best.copy()
    cur_val = b_best
    explore_end = t0 + 135.0
    # precompute once contacts to guide
    it = 0
    while time.time() < explore_end and time.time() < deadline - 40:
        it += 1
        # pick mode
        r = rng.rand()
        y = x_cur.copy()
        Gtmp = y @ y.T
        np.fill_diagonal(Gtmp, -2.0)
        gmax_tmp = float(np.max(Gtmp))
        cnt = (Gtmp > gmax_tmp - 0.005).sum(axis=1)
        order = np.argsort(-cnt)  # descending
        if r < 0.65:
            # void re-insert overloaded points
            k = int(rng.randint(2, 6))
            # choose k among top 30 overloaded + random
            pool = list(order[:30])
            chosen = []
            for _ in range(k):
                if rng.rand() < 0.8 and len(pool) > 0:
                    idx = pool[int(rng.randint(len(pool)))]
                    # avoid duplicate
                    if idx in chosen:
                        continue
                    chosen.append(int(idx))
                else:
                    chosen.append(int(rng.randint(n)))
            # sequential void replacement
            for idx in chosen:
                others = np.delete(y, idx, axis=0)
                y[idx] = find_void(others, rng, n_try=5000, n_ref=150)
            steps = 320
            T0 = 0.012 + rng.rand() * 0.010
            T1 = 0.0015 + rng.rand() * 0.002
            lr0 = 0.06 + rng.rand() * 0.05
            y = fast_relax(y, rng, steps, T0, T1, lr0, mu=0.82)
        elif r < 0.85:
            # global shake + relax (different basin attempt)
            sigma = 0.02 + rng.rand() * 0.05
            # shake subset
            m = int(rng.randint(15, 60))
            idxs = rng.choice(n, size=m, replace=False)
            y[idxs] += sigma * rng.randn(m, d)
            y[idxs] /= np.linalg.norm(y[idxs], axis=1, keepdims=True)
            steps = 350
            T0 = 0.015 + rng.rand() * 0.010
            T1 = 0.002
            lr0 = 0.07
            y = fast_relax(y, rng, steps, T0, T1, lr0, mu=0.85)
        else:
            # disperse clump: pick random direction u, move points with large projection
            u = rng.randn(d); u /= np.linalg.norm(u)
            proj = y @ u
            # most clumped side
            if rng.rand() < 0.5:
                idxs = np.argsort(-proj)[:int(rng.randint(10, 30))]
            else:
                idxs = np.argsort(proj)[:int(rng.randint(10, 30))]
            # push them tangentially away with noise
            sigma = 0.05 + rng.rand() * 0.08
            y[idxs] += sigma * rng.randn(len(idxs), d)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            steps = 300
            T0 = 0.020; T1 = 0.003; lr0 = 0.08
            y = fast_relax(y, rng, steps, T0, T1, lr0, mu=0.80)
        v = max_cos(y)
        if v < b_best - 1e-12:
            b_best = v
            x_best = y.copy()
            x_cur = y.copy()
            cur_val = v
        elif v < cur_val:
            x_cur = y.copy()
            cur_val = v
        else:
            # occasional sideways acceptance to wander
            if rng.rand() < 0.10 and v < cur_val + 0.01:
                x_cur = y.copy()
                cur_val = v
        # restart from best if stuck wandering too high
        if cur_val > b_best + 0.015:
            x_cur = x_best.copy()
            cur_val = b_best
    # final polish of best
    remain = deadline - time.time()
    if remain > 5:
        # short polish, leave 2s margin
        try:
            x_best = lp_polish(x_best, min(remain - 2.0, 35.0), deadline - 1.0)
        except Exception:
            pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
