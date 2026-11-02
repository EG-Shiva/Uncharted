import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds, init_radius=8e-3, cap=2500):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = float(init_radius)
    start = time.time()
    iuI, iuJ = np.triu_indices(n, 1)
    while True:
        if time.time() - start >= seconds or radius < 1e-13:
            break
        G = x @ x.T
        c = G[iuI, iuJ]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > cap:
            idx = np.where(sel)[0]
            cv = c[idx]
            part = np.argpartition(cv, -cap)[-cap:]
            sel2 = np.zeros_like(sel)
            sel2[idx[part]] = True
            sel = sel2
            cnt = cap
        I = iuI[sel]
        J = iuJ[sel]
        csel = c[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        vals = np.empty(p * (2 * d + 1), dtype=np.float64)
        Cb = cols.reshape(p, 2 * d + 1)
        Vb = vals.reshape(p, 2 * d + 1)
        ar = np.arange(d)
        Cb[:, :d] = I[:, None] * d + ar
        Cb[:, d:2 * d] = J[:, None] * d + ar
        Cb[:, 2 * d] = n * d
        Vb[:, :d] = x[J]
        Vb[:, d:2 * d] = x[I]
        Vb[:, 2 * d] = -1.0
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
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
            x = y
            best = v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _greedy_reinsert(y, to_replace, rng, n_cand=4000):
    n, d = y.shape
    order = list(to_replace)
    rng.shuffle(order)
    fixed_mask = np.ones(n, dtype=bool)
    fixed_mask[to_replace] = False
    # maintain list of fixed points
    Xf = y[fixed_mask]
    for k in order:
        cand = rng.standard_normal(size=(n_cand, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        dots = cand @ Xf.T
        worst = dots.max(axis=1)
        bi = int(np.argmin(worst))
        y[k] = cand[bi]
        Xf = np.vstack([Xf, cand[bi][None, :]])
    # tiny local hill climb on replaced points
    for k in order:
        others = np.delete(y, k, axis=0)
        cur_w = float((others @ y[k]).max())
        trial = rng.standard_normal(size=(500, d)) * 0.03 + y[k]
        trial /= np.linalg.norm(trial, axis=1, keepdims=True)
        dw = (trial @ others.T).max(axis=1)
        bi = int(np.argmin(dw))
        if dw[bi] < cur_w:
            y[k] = trial[bi]
    return y

def solve():
    t0 = time.time()
    BUDGET = 168.0
    rng = np.random.default_rng(12345)
    x_best = None
    b_best = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (323, 7):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(a)
                if v < b_best:
                    b_best = v
                    x_best = a.copy()
        except Exception:
            pass
    if x_best is None:
        a = rng.standard_normal(size=(323, 7))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        x_best = a
        b_best = _max_cos(a)
    n, d = x_best.shape
    # basin hopping with cluster surgery
    it = 0
    sizes = [6, 8, 5, 7, 4, 9]
    while time.time() - t0 < BUDGET - 14.0:
        it += 1
        G = x_best @ x_best.T
        np.fill_diagonal(G, -2.0)
        worst = G.max(axis=1)
        m = float(worst.max())
        cnt = np.sum(G > m - 1e-7, axis=1)
        order_w = np.argsort(-worst)
        # pick hub among top 5 deterministically varied
        topk = order_w[:5]
        hub = int(topk[it % 5])
        ns = int(sizes[it % len(sizes)])
        # nearest neighbours of hub
        nn = np.argsort(-G[hub])
        # exclude hub itself (diag -2 so not first); take first ns-1
        removal = [hub] + [int(v) for v in nn[:ns - 1] if int(v) != hub][:ns - 1]
        # every 6th iter also relocate rattlers
        if it % 6 == 0:
            ratt = np.where(cnt <= 1)[0]
            for r in ratt[:2]:
                r = int(r)
                if r not in removal:
                    removal[-1] = r
                    break
        removal = list(dict.fromkeys(removal))
        y = x_best.copy()
        # kick: greedy reinsert
        y = _greedy_reinsert(y, removal, rng, n_cand=4000)
        # small global jitter to break degeneracy
        jit = rng.standard_normal(size=(n, d)) * 0.002
        y = y + jit
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        # short polish
        y = _lp_polish(y, seconds=3.2, init_radius=8e-3)
        v = _max_cos(y)
        if v < b_best:
            b_best = v
            x_best = y
        # time guard
        if time.time() - t0 >= BUDGET - 14.0:
            break
    # final polish of best
    remain = BUDGET - (time.time() - t0)
    remain = min(max(remain - 1.0, 3.0), 12.0)
    try:
        x_best = _lp_polish(x_best, seconds=float(remain), init_radius=5e-3)
    except Exception:
        pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
