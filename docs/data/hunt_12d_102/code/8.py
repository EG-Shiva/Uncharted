import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def adam_quench(x_init, steps, T0, T1, lr, rng_extra_noise=0.0, rng=None):
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
        M = float(G.max())
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
        # normalize update scale to avoid blow up
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
        G = x @ x.T
        iu_c = G[iu]
        mcur = float(iu_c.max())
        sel = iu_c > mcur - max(20 * radius, 1e-9)
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
            res = linprog(obj, A_ub=A_ub, b_ub=-iu_c[sel], A_eq=A_eq, b_eq=np.zeros(n),
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


def best_hole(X_rem, rng, n_samp=8000):
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
    rng = np.random.default_rng(7)
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
    it = 0
    # explore until ~140s, reserve rest for LP polish
    explore_end = t0 + 140.0
    while time.time() < explore_end and time.time() < t_end - 25.0:
        it += 1
        G0 = x_best @ x_best.T
        np.fill_diagonal(G0, -2.0)
        rowmax = G0.max(axis=1)
        order = np.argsort(-rowmax)
        mcur = float(rowmax.max())
        # contact load with tolerance 1e-3 for targeting
        loads = np.sum(G0 > mcur - 1e-3, axis=1)
        load_order = np.argsort(-loads)
        x_try = x_best.copy()
        mode = it % 6
        if mode == 0:
            x_try = x_best + 0.012 * rng.standard_normal((n, d))
            steps, T0, T1, lr = 600, 0.018, 0.0035, 0.0020
        elif mode == 1:
            k = 2 + int(rng.integers(0, 3))  # 2-4
            idx = order[:k]
            x_try[idx] += 0.10 * rng.standard_normal((k, d))
            x_try += 0.003 * rng.standard_normal((n, d))
            steps, T0, T1, lr = 800, 0.025, 0.0030, 0.0022
        elif mode == 2:
            k = 4 + int(rng.integers(0, 4))
            idx = load_order[:k]
            x_try[idx] += 0.07 * rng.standard_normal((k, d))
            x_try += 0.003 * rng.standard_normal((n, d))
            steps, T0, T1, lr = 800, 0.020, 0.0028, 0.0025
        elif mode == 3:
            ci = int(rng.integers(n))
            gr = (x_best @ x_best[ci])
            gr[ci] = -2.0
            nbr = np.argsort(-gr)[:5]
            cl = np.append(nbr, ci)
            x_try[cl] += 0.06 * rng.standard_normal((len(cl), d))
            x_try += 0.002 * rng.standard_normal((n, d))
            steps, T0, T1, lr = 700, 0.022, 0.0032, 0.0020
        elif mode == 4:
            # remove worst point and re-insert in best hole
            wi = int(order[0])
            mask = np.ones(n, dtype=bool)
            mask[wi] = False
            hole = best_hole(x_best[mask], rng, n_samp=8000)
            x_try[wi] = hole
            # jitter neighbours of hole slightly
            x_try += 0.003 * rng.standard_normal((n, d))
            steps, T0, T1, lr = 900, 0.020, 0.0025, 0.0020
        else:
            # random subset medium kick
            k = 10
            idx = rng.choice(n, size=k, replace=False)
            x_try[idx] += 0.05 * rng.standard_normal((k, d))
            x_try += 0.002 * rng.standard_normal((n, d))
            steps, T0, T1, lr = 700, 0.022, 0.0035, 0.0022
        nr = np.linalg.norm(x_try, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x_try /= nr
        x_q = adam_quench(x_try, steps, T0, T1, lr)
        v = max_cos(x_q)
        if v < best:
            best = v
            x_best = x_q
            # quick opportunistic extra squeeze with low T
            if time.time() < explore_end - 5:
                x_extra = adam_quench(x_best, 300, 0.008, 0.0015, 0.0015)
                ve = max_cos(x_extra)
                if ve < best:
                    best = ve
                    x_best = x_extra
        # occasional acceptance of near-best to diversify is implicit via next kick from best
    # final squeeze + LP polish with remaining time
    remain = t_end - time.time()
    if remain > 5:
        # short low-T quench before LP
        try:
            x_sq = adam_quench(x_best, 400, 0.010, 0.0018, 0.0015)
            vs = max_cos(x_sq)
            if vs < best:
                best = vs
                x_best = x_sq
        except Exception:
            pass
        remain = t_end - time.time()
        lp_time = min(remain - 1.0, 28.0)
        if lp_time > 3:
            try:
                x_p = lp_polish(x_best, lp_time, t_end - 1.0)
                vp = max_cos(x_p)
                if vp < best:
                    x_best = x_p
                    best = vp
            except Exception:
                pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
