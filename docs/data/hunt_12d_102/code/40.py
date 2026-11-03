import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _adam_quench(x_init, steps, T0, T1, lr, rng=None):
    x = x_init.copy()
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    l0 = np.log(T0); l1 = np.log(T1)
    for s in range(1, steps + 1):
        T = float(np.exp(l0 + (l1 - l0) * (s / steps)))
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        M = float(G.max())
        # avoid overflow: (G-M)<=0
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

def _best_hole(X_rem, rng, n_samp=12000):
    d = X_rem.shape[1]
    S = rng.standard_normal((n_samp, d))
    S /= np.linalg.norm(S, axis=1, keepdims=True)
    dots = S @ X_rem.T
    worst = dots.max(axis=1)
    bi = int(np.argmin(worst))
    q = S[bi].copy()
    # few projected refinement steps against soft-max
    for _ in range(80):
        dv = X_rem @ q
        mm = dv.max()
        w = np.exp((dv - mm) / 0.004)
        g = (w @ X_rem)
        g = g - np.dot(g, q) * q
        nrm = np.linalg.norm(g)
        if nrm < 1e-12 or not np.isfinite(nrm):
            break
        q = q - 0.05 * g / (nrm + 1e-12)
        q /= np.linalg.norm(q)
    return q

def _lp_polish(x, seconds, t_end):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < t_end - 1.0:
        if radius < 1e-14:
            break
        cfull = (x @ x.T)[iu]
        mcur = float(cfull.max())
        thr = mcur - max(20.0 * radius, 1e-7)
        sel = cfull > thr
        # cap to top-K to keep LP cheap
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > 2500:
            # keep largest 2500
            idx = np.argpartition(-cfull, 2500)[:2500]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = mask
            cnt = 2500
        I, J = iu[0][sel], iu[1][sel]
        p, nv = cnt, n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cfull[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        vv = _max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
        if cnt < 50 and radius < 1e-6:
            break
    return x

def solve():
    t0 = time.time()
    t_end = t0 + 170.0
    hop_end = t0 + 138.0
    rng = np.random.default_rng(7)
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.array(np.load(fn), dtype=float)
            a /= np.linalg.norm(a, axis=1, keepdims=True)
            if a.shape == (102, 12):
                cands.append((_max_cos(a), a))
        except Exception:
            pass
    if not cands:
        a = rng.standard_normal((102, 12))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append((_max_cos(a), a))
    cands.sort(key=lambda t: t[0])
    best, x_best = cands[0][0], cands[0][1].copy()
    n, d = 102, 12

    it = 0
    while time.time() < hop_end and it < 300:
        it += 1
        if time.time() > hop_end - 2.0:
            break
        G0 = x_best @ x_best.T
        np.fill_diagonal(G0, -2.0)
        m0 = float(G0.max())
        cnt0 = (G0 > m0 - 0.005).sum(axis=1)
        rowmax = G0.max(axis=1)
        order = np.argsort(-rowmax)
        order_c = np.argsort(-cnt0)
        mode = it % 6
        x_try = x_best.copy()

        if mode == 0:
            # light global jitter
            x_try = x_best + 0.012 * rng.standard_normal((n, d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            steps, T0, T1, lr = 320, 0.015, 0.003, 0.002
        elif mode == 1:
            # heavy kick to worst 3-4 individual points
            k = 3 + int(rng.integers(0, 2))
            idx = order[:k]
            x_try[idx] += 0.10 * rng.standard_normal((k, d))
            x_try += 0.003 * rng.standard_normal((n, d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            steps, T0, T1, lr = 450, 0.022, 0.003, 0.0022
        elif mode == 2:
            # coordinated block move: overloaded hub + nearest neighbours share common shift
            hub = int(order_c[0])
            gr = (x_best @ x_best[hub]).copy()
            gr[hub] = -2.0
            nbr = np.argsort(-gr)[:6]
            cl = np.append(nbr, hub)
            common = rng.standard_normal((d,)) * 0.055
            x_try[cl] += common[None, :] + 0.018 * rng.standard_normal((len(cl), d))
            # tiny elsewhere
            x_try += 0.002 * rng.standard_normal((n, d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            steps, T0, T1, lr = 500, 0.020, 0.0028, 0.0022
        elif mode == 3:
            # single hole re-insertion of worst point
            w = int(order[0])
            mask = np.ones(n, dtype=bool); mask[w] = False
            Xrem = x_best[mask]
            q = _best_hole(Xrem, rng, n_samp=12000)
            x_try[w] = q
            # small jitter around to break symmetry
            x_try += 0.002 * rng.standard_normal((n, d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            steps, T0, T1, lr = 450, 0.018, 0.0025, 0.002
        elif mode == 4:
            # double hole re-insertion
            nk = 2
            worst = order[:nk]
            mask = np.ones(n, dtype=bool); mask[worst] = False
            Xcur = x_best[mask]
            for wi in worst:
                q = _best_hole(Xcur, rng, n_samp=9000)
                x_try[wi] = q
                Xcur = np.vstack([Xcur, q[None, :]])
            x_try += 0.002 * rng.standard_normal((n, d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            steps, T0, T1, lr = 550, 0.020, 0.0025, 0.0022
        else:
            # triple hole + cluster shake
            nk = 3
            worst = order_c[:nk].copy()
            mask = np.ones(n, dtype=bool); mask[worst] = False
            Xcur = x_best[mask]
            for wi in worst:
                q = _best_hole(Xcur, rng, n_samp=7000)
                x_try[wi] = q
                Xcur = np.vstack([Xcur, q[None, :]])
            # shake a random neighbourhood slightly
            ci = int(rng.integers(n))
            gr2 = (x_try @ x_try[ci]).copy(); gr2[ci] = -2.0
            nbr2 = np.argsort(-gr2)[:4]
            x_try[nbr2] += 0.02 * rng.standard_normal((4, d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            steps, T0, T1, lr = 600, 0.022, 0.0025, 0.0022

        try:
            x_q = _adam_quench(x_try, steps, T0, T1, lr)
        except Exception:
            continue
        v = _max_cos(x_q)
        if v < best and np.all(np.isfinite(x_q)):
            best = v
            x_best = x_q.copy()
        # opportunistic early exit if time low
        if time.time() > hop_end:
            break

    # single final snap
    try:
        remain = t_end - time.time() - 1.0
        if remain > 5.0:
            x_best = _lp_polish(x_best, min(remain, 28.0), t_end)
    except Exception:
        pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
