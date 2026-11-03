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
    lT0 = np.log(T0)
    lT1 = np.log(T1)
    for s in range(1, steps + 1):
        lt = lT0 + (lT1 - lT0) * (s / steps)
        T = float(np.exp(lt))
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        M = G.max()
        # weights focused on near-max pairs
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


def best_hole(X_rem, rng, n_samp=5000):
    d = X_rem.shape[1]
    S = rng.standard_normal((n_samp, d))
    S /= np.linalg.norm(S, axis=1, keepdims=True)
    dots = S @ X_rem.T
    worst = dots.max(axis=1)
    return S[int(np.argmin(worst))]


def solve():
    t0 = time.time()
    T_LIMIT = 172.0
    t_end = t0 + T_LIMIT
    rng = np.random.default_rng(0)
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
        if c is not None:
            try:
                a = np.array(c, dtype=float)
                a /= np.linalg.norm(a, axis=1, keepdims=True)
                cands.append((max_cos(a), a))
            except Exception:
                pass
    if len(cands) == 0:
        a = rng.standard_normal((102, 12))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        x_best = a
        best = max_cos(a)
    else:
        cands.sort(key=lambda t: t[0])
        best, x_best = cands[0][0], cands[0][1].copy()
    n, d = x_best.shape
    # quick quench params by mode
    it = 0
    # reserve ~18s for final polish
    while time.time() < t_end - 20.0 and it < 400:
        it += 1
        mode = it % 4
        # row-wise worst
        G0 = x_best @ x_best.T
        np.fill_diagonal(G0, -2.0)
        rowmax = G0.max(axis=1)
        order = np.argsort(-rowmax)
        x_try = x_best.copy()
        if mode == 0:
            # global jitter
            x_try = x_best + 0.015 * rng.standard_normal((n, d))
            steps, T0, T1, lr = 700, 0.02, 0.004, 0.002
        elif mode == 1:
            # kick worst 3 heavily
            k = 3
            idx = order[:k]
            x_try[idx] += 0.09 * rng.standard_normal((k, d))
            x_try += 0.004 * rng.standard_normal((n, d))
            steps, T0, T1, lr = 900, 0.025, 0.0035, 0.0022
        elif mode == 2:
            # kick random cluster (point + 5 nearest)
            ci = int(rng.integers(n))
            Gr = (x_best @ x_best[ci])
            Gr[ci] = -2.0
            nbr = np.argsort(-Gr)[:5]
            cl = np.append(nbr, ci)
            x_try[cl] += 0.06 * rng.standard_normal((len(cl), d))
            x_try += 0.003 * rng.standard_normal((n, d))
            steps, T0, T1, lr = 900, 0.02, 0.003, 0.002
        else:
            # hole re-insertion: replace worst point by best hole
            wi = int(order[0])
            mask = np.ones(n, dtype=bool)
            mask[wi] = False
            hole = best_hole(x_best[mask], rng, n_samp=5000)
            x_try[wi] = hole + 0.005 * rng.standard_normal(d)
            x_try[~mask is False] = x_try[~mask is False]  # no-op keep shape
            # small jitter others
            x_try[mask] += 0.004 * rng.standard_normal((n - 1, d))
            steps, T0, T1, lr = 900, 0.02, 0.003, 0.002
        x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        x_q = adam_quench(x_try, steps, T0, T1, lr)
        vq = max_cos(x_q)
        # promising? snap with short LP
        if vq < best - 1e-12:
            # squeeze immediately
            budget = min(4.0, max(1.0, (t_end - 20.0 - time.time()) / 4.0))
            if time.time() + budget < t_end - 18.0:
                x_q = lp_polish(x_q, budget, t_end - 18.0)
                vq = max_cos(x_q)
            if vq < best:
                x_best, best = x_q, vq
        elif vq < best + 0.004 and (it % 5 == 0):
            budget = min(2.5, max(0.8, (t_end - 20.0 - time.time()) / 8.0))
            if budget > 0.7 and time.time() + budget < t_end - 18.0:
                x_p = lp_polish(x_q, budget, t_end - 18.0)
                vp = max_cos(x_p)
                if vp < best:
                    x_best, best = x_p, vp
        # time guard
        if time.time() >= t_end - 20.0:
            break
    # final squeeze of incumbent
    remain = t_end - time.time() - 2.0
    if remain > 2.0:
        x_best = lp_polish(x_best, min(remain, 16.0), t_end - 1.0)
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
