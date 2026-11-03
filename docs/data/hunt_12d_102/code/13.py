import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def lp_polish(x, seconds, t_end=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while True:
        now = time.time()
        if now - start >= seconds:
            break
        if t_end is not None and now >= t_end:
            break
        if radius < 1e-14:
            break
        G = x @ x.T
        iu = iu0
        c = G[iu]
        mcur = c.max()
        sel = c > mcur - max(20.0 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix(
            (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
            shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
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
            x, best = y, vv
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


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
        W = np.exp((G - M) / T)
        np.fill_diagonal(W, 0.0)
        S = W.sum()
        if S == 0 or not np.isfinite(S):
            break
        P = W / S
        Grad = (P @ x) / T
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


def refine_hole(q, X_rem, steps=160, T0=0.012, T1=0.002, lr0=0.06):
    q = q / np.linalg.norm(q)
    lT0 = np.log(T0)
    lT1 = np.log(T1)
    for s in range(steps):
        f = (s + 1) / steps
        T = float(np.exp(lT0 + (lT1 - lT0) * f))
        lr = lr0 * (1.0 - 0.7 * f)
        dots = X_rem @ q
        mm = dots.max()
        w = np.exp((dots - mm) / T)
        sw = w.sum()
        if sw == 0 or not np.isfinite(sw):
            break
        grad = (w @ X_rem) / sw
        gt = grad - float(grad @ q) * q
        nrm = float(np.linalg.norm(gt))
        if nrm > 1e-12:
            q = q - lr * gt / max(nrm, 1.0)
            q /= np.linalg.norm(q)
    return q


def best_hole_insert(X_rem, rng, n_samp=16000):
    d = X_rem.shape[1]
    S = rng.standard_normal((n_samp, d))
    S /= np.linalg.norm(S, axis=1, keepdims=True)
    dots = S @ X_rem.T
    worst = dots.max(axis=1)
    q = S[int(np.argmin(worst))].copy()
    q = refine_hole(q, X_rem)
    return q


def solve():
    t0 = time.time()
    T_END = t0 + 171.0
    rng = np.random.default_rng(7)
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            a /= np.linalg.norm(a, axis=1, keepdims=True)
            if a.shape == (102, 12):
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
    iu = np.triu_indices(n, 1)

    it = 0
    # hopping phase, reserve ~14s for final polish
    while time.time() < T_END - 14.0 and it < 600:
        it += 1
        mode = it % 5
        G0 = x_best @ x_best.T
        np.fill_diagonal(G0, -2.0)
        m0 = G0.max()
        rowmax = G0.max(axis=1)
        contacts = (G0 > m0 - 0.004).sum(axis=1)
        order_row = np.argsort(-rowmax)
        order_con = np.argsort(-contacts)
        x_try = x_best.copy()

        if mode == 0:
            sig = 0.012 + 0.008 * rng.random()
            x_try = x_best + sig * rng.standard_normal((n, d))
            steps, T0, T1, lr = 650, 0.018, 0.0035, 0.0022
        elif mode == 1:
            k = int(rng.integers(2, 5))
            idx = order_con[:k].copy()
            # mix row-worst too
            if rng.random() < 0.5:
                idx = order_row[:k].copy()
            amp = 0.07 + 0.05 * rng.random()
            x_try[idx] += amp * rng.standard_normal((k, d))
            x_try += 0.004 * rng.standard_normal((n, d))
            steps, T0, T1, lr = 850, 0.022, 0.0032, 0.0025
        elif mode == 2:
            # cluster around a bad point + neighbours
            ci = int(order_row[int(rng.integers(0, 6))])
            gr = (x_best @ x_best[ci]).copy()
            gr[ci] = -2.0
            nbr = np.argsort(-gr)[:int(rng.integers(4, 8))]
            cl = np.append(nbr, ci)
            amp_c = 0.05 + 0.06 * rng.random()
            x_try[cl] += amp_c * rng.standard_normal((len(cl), d))
            x_try += 0.003 * rng.standard_normal((n, d))
            steps, T0, T1, lr = 800, 0.020, 0.0030, 0.0024
        elif mode == 3:
            nk = int(rng.integers(1, 4))
            # pick worst by contacts / rowmax
            if rng.random() < 0.5:
                worst = order_con[:nk].copy()
            else:
                worst = order_row[:nk].copy()
            mask = np.ones(n, dtype=bool)
            mask[worst] = False
            X_rem = x_best[mask]
            X_cur = X_rem.copy()
            new_pts = []
            for _ in worst:
                q = best_hole_insert(X_cur, rng, n_samp=12000)
                new_pts.append(q)
                X_cur = np.vstack([X_cur, q[None, :]])
            x_try[worst] = np.array(new_pts)
            # tiny jitter elsewhere to break symmetry
            x_try[mask] += 0.002 * rng.standard_normal(((n - nk), d))
            steps, T0, T1, lr = 900, 0.016, 0.0028, 0.0022
        else:
            k = int(rng.integers(6, 13))
            idx = rng.choice(n, size=k, replace=False)
            amp = 0.03 + 0.04 * rng.random()
            x_try[idx] += amp * rng.standard_normal((k, d))
            x_try += 0.002 * rng.standard_normal((n, d))
            steps, T0, T1, lr = 750, 0.018, 0.0032, 0.0023

        nr = np.linalg.norm(x_try, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x_try /= nr

        # quench
        try:
            x_q = adam_quench(x_try, steps, T0, T1, lr)
        except Exception:
            continue
        qv = max_cos(x_q)
        # only snap promising candidates
        thresh = best + (0.012 if mode != 3 else 0.018)
        if qv > thresh:
            continue
        if time.time() >= T_END - 14.0:
            break
        # short LP snap (1.2s)
        try:
            x_s = lp_polish(x_q, seconds=1.2, t_end=T_END - 14.0)
        except Exception:
            continue
        sv = max_cos(x_s)
        if sv < best - 1e-12:
            best = sv
            x_best = x_s.copy()
        # opportunistic: if quench alone is much better but LP failed to improve, keep quench
        elif qv < best - 1e-12:
            # polish a bit more
            try:
                x_s2 = lp_polish(x_q, seconds=0.8, t_end=T_END - 14.0)
                sv2 = max_cos(x_s2)
                if sv2 < best - 1e-12:
                    best = sv2
                    x_best = x_s2.copy()
            except Exception:
                pass

    # final thorough polish of best
    remain = T_END - time.time() - 0.5
    if remain > 2.0:
        try:
            x_best = lp_polish(x_best, seconds=min(remain, 12.0), t_end=T_END)
        except Exception:
            pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
