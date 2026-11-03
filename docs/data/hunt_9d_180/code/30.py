import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def contact_counts(x, thresh=1e-9):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    m = float(np.max(g))
    cm = g > m - thresh
    return np.sum(cm, axis=1), m, g

def find_void(others, rng, n_try=8000):
    d = others.shape[1]
    cand = rng.randn(n_try, d)
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ others.T
    m = np.max(dots, axis=1)
    bi = int(np.argmin(m))
    return cand[bi]

def refine_void(y, others, rng, iters=200):
    cur = float(np.max(y @ others.T))
    for k in range(iters):
        sig = 0.08 * (0.001 / 0.08) ** (k / max(1, iters - 1))
        prop = y + sig * rng.randn(y.shape[0])
        prop /= np.linalg.norm(prop)
        pm = float(np.max(prop @ others.T))
        if pm < cur:
            y = prop
            cur = pm
    return y

def smooth_descent(x, steps, T0, T1, lr0, mu, rng):
    n, d = x.shape
    x = x.copy()
    bv = max_cos(x)
    best = x.copy()
    vel = np.zeros_like(x)
    for s in range(steps):
        frac = s / max(1, steps - 1)
        T = T0 * (T1 / T0) ** frac
        lr = lr0 * (1.0 - 0.5 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = float(np.max(G))
        E = np.exp((G - gmax) / T)
        np.fill_diagonal(E, 0.0)
        row = np.sum(E, axis=1, keepdims=True) + 1e-12
        grad = E @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gtan = grad - dot * x
        step = gtan / row
        nm = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        # clip
        step = step * np.minimum(1.0, 0.5 / nm)
        vel = mu * vel + step
        x = x - lr * vel
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        if (s & 127) == 127 or s == steps - 1:
            v = max_cos(x)
            if v < bv:
                bv = v
                best = x.copy()
            # early abort if very bad
            if v > bv + 0.02 and s > 300:
                break
    return best, bv

def lp_polish(x, seconds, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < deadline and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
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
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
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

def solve():
    t0 = time.time()
    deadline = t0 + 175.0
    explore_end = t0 + 130.0
    rng = np.random.RandomState(12345)
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy")
    except Exception:
        xr = None
    cands = []
    for c in (xb, xr):
        try:
            c = np.asarray(c, dtype=float)
            if c.shape == (180, 9):
                c = c / np.linalg.norm(c, axis=1, keepdims=True)
                cands.append((max_cos(c), c))
        except Exception:
            pass
    if not cands:
        x = rng.randn(180, 9); x /= np.linalg.norm(x, axis=1, keepdims=True)
        x_best = x; b_best = max_cos(x)
    else:
        cands.sort(key=lambda t: t[0])
        b_best, x_best = cands[0][0], cands[0][1].copy()
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    # contact order for targeted deletion
    try:
        cc, _, _ = contact_counts(x_best)
        order_desc = np.argsort(-cc)
    except Exception:
        order_desc = np.arange(180)
    n, d = 180, 9
    it = 0
    # fast exploration: void re-insert overloaded points + annealed descent
    while time.time() < explore_end and time.time() < deadline - 45:
        it += 1
        # choose move size cyclically
        if it % 3 == 1:
            k = int(rng.randint(4, 10))
        elif it % 3 == 2:
            k = int(rng.randint(2, 6))
        else:
            k = int(rng.randint(8, 14))
        # pick victims: mostly overloaded, sometimes random
        x0 = x_best.copy()
        if rng.rand() < 0.75:
            # refresh contact order every 20 iters from best
            if it % 20 == 0:
                try:
                    cc, _, _ = contact_counts(x_best)
                    order_desc = np.argsort(-cc)
                except Exception:
                    pass
            nv = min(k + 2, n)
            pool = order_desc[:max(20, 2 * k)]
            victims = rng.choice(pool, size=min(k, len(pool)), replace=False)
        else:
            victims = rng.choice(n, size=k, replace=False)
        keep_mask = np.ones(n, dtype=bool); keep_mask[victims] = False
        others = x0[keep_mask]
        # sequential void insertion
        new_pts = []
        cur_set = others.copy()
        # adapt n_try to remaining time
        n_try = 6000 if k > 6 else 9000
        for j in range(k):
            y = find_void(cur_set, rng, n_try=n_try)
            # light refine only for last point or small k to save time
            if k <= 6:
                y = refine_void(y, cur_set, rng, iters=120)
            else:
                y = refine_void(y, cur_set, rng, iters=60)
            new_pts.append(y)
            cur_set = np.vstack([cur_set, y[None, :]])
        x_new = x0.copy()
        x_new[victims] = np.array(new_pts)
        # occasional extra shake of neighbours
        if rng.rand() < 0.4:
            sig = float(rng.uniform(0.02, 0.08))
            idx = rng.choice(n, size=min(30, n), replace=False)
            x_new[idx] += sig * rng.randn(len(idx), d)
            x_new[idx] /= np.linalg.norm(x_new[idx], axis=1, keepdims=True)
        # annealed descent params randomized
        T0 = float(rng.uniform(0.003, 0.010))
        T1 = float(rng.uniform(0.00015, 0.0008))
        lr0 = float(rng.uniform(0.03, 0.08))
        mu = float(rng.uniform(0.80, 0.90))
        steps = 700 if time.time() < explore_end - 20 else 400
        xc, vc = smooth_descent(x_new, steps, T0, T1, lr0, mu, rng)
        if vc < b_best:
            # verify strictly
            b_best = vc
            x_best = xc.copy()
            try:
                cc, _, _ = contact_counts(x_best)
                order_desc = np.argsort(-cc)
            except Exception:
                pass
        # also try pure subset shake branch occasionally without void
        if time.time() >= explore_end or time.time() >= deadline - 45:
            break
        if it % 5 == 0 and time.time() < explore_end - 5:
            # quick shake-only candidate
            xs = x_best.copy()
            m = int(rng.randint(10, 40))
            idx = rng.choice(n, size=m, replace=False)
            sig = float(rng.uniform(0.03, 0.12))
            xs[idx] += sig * rng.randn(m, d)
            xs[idx] /= np.linalg.norm(xs[idx], axis=1, keepdims=True)
            T0b = float(rng.uniform(0.002, 0.008))
            xc2, vc2 = smooth_descent(xs, 500, T0b, 0.0004, 0.05, 0.85, rng)
            if vc2 < b_best:
                b_best = vc2
                x_best = xc2.copy()
    # final polish with remaining time
    remain = deadline - time.time() - 1.0
    if remain > 5:
        try:
            x_best = lp_polish(x_best, min(remain, 40.0), deadline - 0.5)
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    # tiny shrink guard against NaN/zero
    x_best = np.where(np.isnan(x_best), 0.0, x_best)
    nr = np.linalg.norm(x_best, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    x_best = x_best / nr
    return x_best
