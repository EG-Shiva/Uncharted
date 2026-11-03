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
    lT0 = np.log(T0); lT1 = np.log(T1)
    for s in range(1, steps + 1):
        lt = lT0 + (lT1 - lT0) * (s / steps)
        T = float(np.exp(lt))
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        Mx = G.max()
        W = np.exp((G - Mx) / T)
        np.fill_diagonal(W, 0.0)
        Grad = (W @ x) / T
        dot = np.sum(Grad * x, axis=1, keepdims=True)
        Gt = Grad - dot * x
        m = b1 * m + (1.0 - b1) * Gt
        v = b2 * v + (1.0 - b2) * (Gt * Gt)
        c1 = 1.0 - b1 ** s
        c2 = 1.0 - b2 ** s
        upd = (m / c1) / (np.sqrt(v / c2) + eps)
        x = x - lr * upd
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x /= nr
    return x

def greedy_holes(X_cur, k, rng, n_samp=6000):
    d = X_cur.shape[1]
    cur = X_cur.copy()
    outs = []
    for _ in range(k):
        S = rng.standard_normal((n_samp, d))
        S /= np.linalg.norm(S, axis=1, keepdims=True)
        dots = S @ cur.T
        worst = dots.max(axis=1)
        q = S[int(np.argmin(worst))]
        outs.append(q)
        cur = np.vstack([cur, q[None, :]])
    return np.array(outs)

def lp_polish(x, seconds, t_end):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    it = 0
    while time.time() - start < seconds and time.time() < t_end and it < 30:
        it += 1
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
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        y /= nr
        vv = max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    T_LIMIT = 172.0
    t_end = t0 + T_LIMIT
    rng = np.random.default_rng(42)
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.array(np.load(fn), dtype=float)
            a /= np.linalg.norm(a, axis=1, keepdims=True)
            if a.shape == (102, 12):
                cands.append((max_cos(a), a))
        except Exception:
            pass
    if len(cands) == 0:
        a = rng.standard_normal((102, 12))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append((max_cos(a), a))
    cands.sort(key=lambda t: t[0])
    best, x_best = cands[0][0], cands[0][1].copy()
    n, d = 102, 12
    x_cur = x_best.copy()
    cur_val = best

    it = 0
    # explore until ~ t_end-16s
    while time.time() < t_end - 16.0 and it < 2000:
        it += 1
        mode = it % 5
        if mode in (0, 1, 3):
            # large cavity: delete connected overloaded cluster
            G0 = x_cur @ x_cur.T
            np.fill_diagonal(G0, -2.0)
            rowmax = G0.max(axis=1)
            if mode == 3 or rng.random() < 0.3:
                seed = int(rng.integers(n))
            else:
                top = np.argsort(-rowmax)[:10]
                seed = int(top[rng.integers(len(top))])
            if mode == 0:
                k = int(rng.integers(8, 13))  # 8..12 big hole
            else:
                k = int(rng.integers(4, 9))   # 4..8 medium hole
            sim = x_cur @ x_cur[seed]
            sim[seed] = -2.0
            nbr = np.argsort(-sim)[:k-1]
            cl = np.append(nbr, seed)
            mask = np.ones(n, dtype=bool)
            mask[cl] = False
            X_rem = x_cur[mask]
            # let remainder spread into cavity (topology change)
            X_rem = adam_quench(X_rem, 150, 0.02, 0.005, 0.003)
            new_pts = greedy_holes(X_rem, k, rng, n_samp=6000)
            new_pts = new_pts + 0.003 * rng.standard_normal(new_pts.shape)
            new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
            x_try = np.vstack([X_rem, new_pts])
            # shuffle order to avoid bias (deterministic perm)
            perm = rng.permutation(n)
            x_try = x_try[perm]
            x_try += 0.002 * rng.standard_normal((n, d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            x_q = adam_quench(x_try, 600, 0.02, 0.0025, 0.0032)
        elif mode == 2:
            # single hole re-insertion of worst point
            G0 = x_cur @ x_cur.T
            np.fill_diagonal(G0, -2.0)
            rowmax = G0.max(axis=1)
            wi = int(np.argmax(rowmax))
            mask = np.ones(n, dtype=bool); mask[wi] = False
            hole = greedy_holes(x_cur[mask], 1, rng, n_samp=8000)[0]
            x_try = x_cur.copy()
            x_try[wi] = hole + 0.003 * rng.standard_normal(d)
            x_try += 0.003 * rng.standard_normal((n, d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            x_q = adam_quench(x_try, 500, 0.018, 0.0025, 0.003)
        else:
            # coordinated kick of worst cluster (no deletion)
            G0 = x_cur @ x_cur.T
            np.fill_diagonal(G0, -2.0)
            rowmax = G0.max(axis=1)
            order = np.argsort(-rowmax)
            cl = order[:6]
            x_try = x_cur.copy()
            x_try[cl] += 0.07 * rng.standard_normal((6, d))
            x_try += 0.004 * rng.standard_normal((n, d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            x_q = adam_quench(x_try, 500, 0.022, 0.003, 0.0032)
        vq = max_cos(x_q)
        if vq < best - 1e-12:
            x_best, best = x_q, vq
            x_cur, cur_val = x_q.copy(), vq
        elif vq < cur_val:
            x_cur, cur_val = x_q.copy(), vq
        elif vq < cur_val + 0.003 and rng.random() < 0.35:
            # wandering acceptance to reach new basin
            x_cur, cur_val = x_q.copy(), vq
        # occasional reset of wanderer to best if drifted too far
        if it % 40 == 0 and cur_val > best + 0.004:
            x_cur, cur_val = x_best.copy(), best
        if time.time() >= t_end - 16.0:
            break
    remain = t_end - time.time() - 1.5
    if remain > 2.0:
        x_best = lp_polish(x_best, min(remain, 13.0), t_end - 1.0)
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
