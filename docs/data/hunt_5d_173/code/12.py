import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _row_worst(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1), g

def _smooth_polish(Y, iters, a0, a1, lr, mu=0.85, seed_step=0.03):
    n, d = Y.shape
    cur = Y.copy()
    V = np.zeros_like(cur)
    # initial best
    G0 = cur @ cur.T
    np.fill_diagonal(G0, -2.0)
    best_val = float(G0.max())
    best = cur.copy()
    for it in range(iters):
        frac = it / max(1, iters - 1)
        a = a0 + (a1 - a0) * frac
        lr_it = lr * (1.0 - 0.5 * frac)
        G = cur @ cur.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        # stabilized softmax weights
        E = np.exp(a * (G - m))
        np.fill_diagonal(E, 0.0)
        Gr = E @ cur
        s = np.sum(Gr * cur, axis=1, keepdims=True)
        Gt = Gr - s * cur
        step = lr_it * Gt
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        step = step * np.minimum(1.0, seed_step / sn)
        V = mu * V + step
        vn = np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
        V = V * np.minimum(1.0, 0.05 / vn)
        cur = cur - V
        cur = cur / (np.linalg.norm(cur, axis=1, keepdims=True) + 1e-16)
        if (it + 1) % 150 == 0 or it == iters - 1:
            G2 = cur @ cur.T
            np.fill_diagonal(G2, -2.0)
            v = float(G2.max())
            if v < best_val:
                best_val = v
                best = cur.copy()
    return best, best_val

def _greedy_holes(X_fixed, k, rng, S=25000):
    d = X_fixed.shape[1]
    cand = rng.normal(size=(S, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True) + 1e-16
    D = cand @ X_fixed.T
    scores = D.max(axis=1)
    chosen = []
    # greedy sequential selection
    cur_scores = scores.copy()
    for _ in range(k):
        idx = int(np.argmin(cur_scores))
        chosen.append(cand[idx])
        # update scores to include this new point
        nd = cand @ chosen[-1]
        cur_scores = np.maximum(cur_scores, nd)
        # forbid re-picking same: set to inf
        cur_scores[idx] = np.inf
    return np.array(chosen)

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
        if int(sel.sum()) > 3500:
            idx = np.where(sel)[0]
            order = np.argsort(iu_c[idx])[-3500:]
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

def solve():
    t0 = time.time()
    BUDGET = 172.0
    deadline = t0 + BUDGET
    rng = np.random.default_rng(42)
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

    trial = 0
    # main hybrid search
    while True:
        rem = deadline - time.time()
        if rem < 9.0:
            break
        mode = trial % 7
        Y = best.copy()
        if mode == 0:
            # small global shake
            Y = Y + 0.02 * rng.normal(size=(n, d))
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            Ys, vs = _smooth_polish(Y, iters=1000, a0=30.0, a1=120.0, lr=0.012)
        elif mode == 1:
            # medium subset kick
            sub = rng.choice(n, 12, replace=False)
            Y[sub] += 0.15 * rng.normal(size=(len(sub), d))
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            Ys, vs = _smooth_polish(Y, iters=1400, a0=20.0, a1=150.0, lr=0.014)
        elif mode == 2:
            # delete worst 1 -> hole
            rw, _ = _row_worst(Y)
            victims = np.argsort(-rw)[:1]
            keep = np.ones(n, dtype=bool); keep[victims] = False
            holes = _greedy_holes(Y[keep], 1, rng, S=25000)
            Y[victims] = holes
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            Ys, vs = _smooth_polish(Y, iters=1400, a0=25.0, a1=150.0, lr=0.012)
        elif mode == 3:
            # delete worst 2 -> holes
            rw, _ = _row_worst(Y)
            victims = np.argsort(-rw)[:2]
            keep = np.ones(n, dtype=bool); keep[victims] = False
            holes = _greedy_holes(Y[keep], 2, rng, S=22000)
            Y[victims] = holes
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            Ys, vs = _smooth_polish(Y, iters=1400, a0=20.0, a1=160.0, lr=0.013)
        elif mode == 4:
            # shake 25 points
            sub = rng.choice(n, 25, replace=False)
            Y[sub] += 0.06 * rng.normal(size=(len(sub), d))
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            Ys, vs = _smooth_polish(Y, iters=1100, a0=30.0, a1=130.0, lr=0.011)
        elif mode == 5:
            # move rattlers / best points to holes
            G = Y @ Y.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            cnt = (G >= m - 1e-9).sum(axis=1)
            # lowest contact points
            order = np.argsort(cnt)
            k = 3 if len(order) >= 3 else 1
            victims = order[:k]
            keep = np.ones(n, dtype=bool); keep[victims] = False
            holes = _greedy_holes(Y[keep], k, rng, S=22000)
            Y[victims] = holes
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            Ys, vs = _smooth_polish(Y, iters=1300, a0=20.0, a1=150.0, lr=0.012)
        else:
            # large global shake, different annealing
            Y = Y + 0.08 * rng.normal(size=(n, d))
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            Ys, vs = _smooth_polish(Y, iters=1000, a0=15.0, a1=100.0, lr=0.015)

        if vs < best_val:
            # refine with short LP to get true local optimum
            rem2 = deadline - time.time()
            pt = 1.0 if rem2 > 20 else 0.5
            pt = min(pt, max(0.4, rem2 - 8.0))
            if pt > 0.35 and rem2 > 9.5:
                Ys = _lp_polish(Ys, pt)
                vs = _max_cos(Ys)
            if vs < best_val:
                best = Ys
                best_val = vs
        trial += 1
        # occasional LP on best to keep it tight (rare)
        if trial % 40 == 0:
            rem2 = deadline - time.time()
            if rem2 > 15:
                best = _lp_polish(best, 0.8)

    rem = deadline - time.time()
    if rem > 1.5:
        best = _lp_polish(best, rem - 0.7)
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
