import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def adam_quench(x_init, steps, T0, T1, lr):
    x = x_init.copy()
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    l0 = np.log(T0)
    l1 = np.log(T1)
    for s in range(1, steps + 1):
        lt = l0 + (l1 - l0) * (s / steps)
        T = float(np.exp(lt))
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        M = G.max()
        W = np.exp((G - M) / T)
        np.fill_diagonal(W, 0.0)
        Grad = (W @ x) / T
        dot = np.sum(Grad * x, axis=1, keepdims=True)
        Gt = Grad - dot * x
        m = b1 * m + (1.0 - b1) * Gt
        v = b2 * v + (1.0 - b2) * (Gt * Gt)
        c1 = 1.0 - b1 ** s
        c2 = 1.0 - b2 ** s
        mh = m / c1
        vh = v / c2
        upd = mh / (np.sqrt(vh) + eps)
        # keep update tangential
        upd = upd - np.sum(upd * x, axis=1, keepdims=True) * x
        x = x - lr * upd
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x /= nr
    return x


def lp_polish(x, seconds, t_end):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < t_end:
        if radius < 1e-14:
            break
        c = (x @ x.T)[iu]
        mcur = c.max()
        sel = c > mcur - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
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
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        y /= nr
        vv = max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def greedy_holes(X_keep, nk, rng, n_samp=8000):
    d = X_keep.shape[1]
    cur = X_keep.copy()
    outs = []
    for _ in range(nk):
        S = rng.standard_normal((n_samp, d))
        S /= np.linalg.norm(S, axis=1, keepdims=True)
        dots = S @ cur.T
        worst = dots.max(axis=1)
        q = S[int(np.argmin(worst))]
        outs.append(q)
        cur = np.vstack([cur, q[None, :]])
    return np.array(outs)


def solve():
    t0 = time.time()
    T_LIMIT = 172.0
    t_end = t0 + T_LIMIT
    rng = np.random.default_rng(7)
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.array(np.load(fn), dtype=float)
            a /= np.linalg.norm(a, axis=1, keepdims=True)
            if a.shape == (102, 12) and np.all(np.isfinite(a)):
                cands.append((max_cos(a), a))
        except Exception:
            pass
    if not cands:
        a = rng.standard_normal((102, 12))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        x_best = a
        best = max_cos(a)
    else:
        cands.sort(key=lambda t: t[0])
        best, x_best = cands[0][0], cands[0][1].copy()
    n, d = 102, 12

    it = 0
    # hopping phase, reserve ~18s for final polish
    while time.time() < t_end - 19.0 and it < 1000:
        it += 1
        G0 = x_best @ x_best.T
        np.fill_diagonal(G0, -2.0)
        rowmax = G0.max(axis=1)
        cnt = (G0 > best - 0.005).sum(axis=1)
        order_row = np.argsort(-rowmax)
        order_cnt = np.argsort(-cnt)
        r = rng.random()
        x_try = x_best.copy()
        if r < 0.30:
            # multi-hole re-insertion of worst / overloaded points
            nk = int(rng.integers(1, 5))  # 1..4
            # mix rowmax and contact-count ordering
            if rng.random() < 0.5:
                victims = order_row[:nk]
            else:
                victims = order_cnt[:nk]
            # sometimes add a random victim
            if rng.random() < 0.3:
                victims = np.append(victims[:-1], rng.integers(n))
            victims = np.unique(victims.astype(int))
            nk = len(victims)
            mask = np.ones(n, dtype=bool)
            mask[victims] = False
            holes = greedy_holes(x_best[mask], nk, rng, n_samp=8000)
            x_try[victims] = holes + 0.004 * rng.standard_normal((nk, d))
            x_try[mask] += 0.004 * rng.standard_normal((n - nk, d))
            steps = int(rng.integers(600, 1100))
            T0 = float(rng.uniform(0.012, 0.035))
            T1 = float(rng.uniform(0.002, 0.005))
            lr = float(rng.uniform(0.0015, 0.0028))
        elif r < 0.55:
            # heavy kick to overloaded cluster: point + nearest neighbours
            ci = int(order_cnt[rng.integers(0, 10)])
            Gr = (x_best @ x_best[ci])
            Gr[ci] = -2.0
            nbr = np.argsort(-Gr)[:int(rng.integers(3, 7))]
            cl = np.append(nbr, ci)
            x_try[cl] += float(rng.uniform(0.05, 0.12)) * rng.standard_normal((len(cl), d))
            x_try += 0.004 * rng.standard_normal((n, d))
            steps = int(rng.integers(700, 1100))
            T0 = float(rng.uniform(0.015, 0.03))
            T1 = float(rng.uniform(0.0025, 0.005))
            lr = float(rng.uniform(0.0018, 0.0025))
        elif r < 0.75:
            # global jitter, different basin attempt
            sig = float(rng.uniform(0.010, 0.030))
            x_try = x_best + sig * rng.standard_normal((n, d))
            steps = int(rng.integers(600, 1000))
            T0 = float(rng.uniform(0.012, 0.030))
            T1 = float(rng.uniform(0.002, 0.005))
            lr = float(rng.uniform(0.0015, 0.0025))
        else:
            # partial random restart: keep 80-95 points, refill rest by holes
            nkeep = int(rng.integers(80, 96))
            # keep points with smallest rowmax preferentially
            keep_order = np.argsort(rowmax)  # ascending
            # add randomness: shuffle bottom part
            keep_idx = keep_order[:nkeep]
            # randomize which are dropped among worst
            drop = np.setdiff1d(np.arange(n), keep_idx)
            mask = np.ones(n, dtype=bool)
            mask[drop] = False
            holes = greedy_holes(x_best[mask], n - nkeep, rng, n_samp=8000)
            x_try[drop] = holes + 0.003 * rng.standard_normal((n - nkeep, d))
            x_try[mask] += 0.003 * rng.standard_normal((nkeep, d))
            steps = int(rng.integers(800, 1300))
            T0 = float(rng.uniform(0.015, 0.035))
            T1 = float(rng.uniform(0.002, 0.0045))
            lr = float(rng.uniform(0.0018, 0.0028))
        x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        x_q = adam_quench(x_try, steps, T0, T1, lr)
        vq = max_cos(x_q)
        if vq < best - 1e-12:
            budget = min(4.0, max(1.0, (t_end - 19.0 - time.time()) / 3.0))
            if time.time() + budget < t_end - 17.0:
                x_q = lp_polish(x_q, budget, t_end - 17.0)
                vq = max_cos(x_q)
            if vq < best:
                x_best, best = x_q, vq
        elif vq < best + 0.002 and (it % 7 == 0):
            budget = min(2.0, max(0.7, (t_end - 19.0 - time.time()) / 10.0))
            if budget > 0.6 and time.time() + budget < t_end - 17.0:
                x_p = lp_polish(x_q, budget, t_end - 17.0)
                vp = max_cos(x_p)
                if vp < best:
                    x_best, best = x_p, vp
        if time.time() >= t_end - 19.0:
            break
    remain = t_end - time.time() - 1.5
    if remain > 2.0:
        x_best = lp_polish(x_best, min(remain, 16.0), t_end - 0.8)
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
