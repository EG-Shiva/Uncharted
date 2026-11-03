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
        m = 0.9 * m + 0.1 * Gt
        v = 0.999 * v + 0.001 * (Gt * Gt)
        c1 = 1.0 - 0.9 ** s
        c2 = 1.0 - 0.999 ** s
        mh = m / c1
        vh = v / c2
        upd = mh / (np.sqrt(vh) + 1e-8)
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
        mcur = float(c.max())
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


def solve():
    t0 = time.time()
    T_LIMIT = 172.0
    t_end = t0 + T_LIMIT
    rng = np.random.default_rng(0)
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
        x_best = a
        best = max_cos(a)
    else:
        cands.sort(key=lambda t: t[0])
        best, x_best = cands[0][0], cands[0][1].copy()
    n, d = 102, 12
    x_cur = x_best.copy()
    v_cur = best
    Tb = 0.0022
    it = 0
    # hopping phase, reserve ~14s for final polish
    while time.time() < t_end - 14.0 and it < 2000:
        it += 1
        # choose source: mostly walk, sometimes restart from best
        if rng.random() < 0.8:
            src = x_cur
            v_src = v_cur
        else:
            src = x_best
            v_src = best
        G0 = src @ src.T
        np.fill_diagonal(G0, -2.0)
        rowmax = G0.max(axis=1)
        order = np.argsort(-rowmax)
        mval = float(G0.max())
        cnt = (G0 > mval - 0.005).sum(axis=1)
        cnt_order = np.argsort(-cnt)
        x_try = src.copy()
        r = rng.random()
        if r < 0.32:
            k = int(rng.integers(2, 6))
            idx = order[:k]
            x_try[idx] += 0.085 * rng.standard_normal((k, d))
            x_try += 0.004 * rng.standard_normal((n, d))
            steps, T0, T1, lr = 600, 0.020, 0.0030, 0.0025
        elif r < 0.55:
            ci = int(rng.integers(n))
            dots = src @ src[ci]
            dots[ci] = -2.0
            nn = int(rng.integers(4, 9))
            nbr = np.argsort(-dots)[:nn]
            cl = np.append(nbr, ci)
            x_try[cl] += 0.070 * rng.standard_normal((len(cl), d))
            x_try += 0.003 * rng.standard_normal((n, d))
            steps, T0, T1, lr = 600, 0.025, 0.0035, 0.0025
        elif r < 0.78:
            nk = int(rng.integers(1, 4))
            # mix worst-row and worst-count
            if rng.random() < 0.5:
                worst = order[:nk].copy()
            else:
                worst = cnt_order[:nk].copy()
            mask = np.ones(n, dtype=bool)
            mask[worst] = False
            Xrem = src[mask]
            S = rng.standard_normal((12000, d))
            S /= np.linalg.norm(S, axis=1, keepdims=True)
            Xcur = Xrem.copy()
            new_pts = []
            for kk in range(nk):
                sc = (S @ Xcur.T).max(axis=1)
                bi = int(np.argmin(sc))
                q = S[bi].copy()
                for _ in range(150):
                    dts = Xcur @ q
                    mm = float(dts.max())
                    w = np.exp((dts - mm) / 0.005)
                    ws = float(w.sum())
                    if ws == 0 or not np.isfinite(ws):
                        break
                    w /= ws
                    g = w @ Xcur
                    g = g - float(g @ q) * q
                    q = q - 0.05 * g
                    nq = float(np.linalg.norm(q))
                    if nq == 0 or not np.isfinite(nq):
                        break
                    q /= nq
                new_pts.append(q)
                Xcur = np.vstack([Xcur, q[None, :]])
            x_try[worst] = np.array(new_pts)
            x_try += 0.002 * rng.standard_normal((n, d))
            steps, T0, T1, lr = 800, 0.020, 0.0025, 0.0020
        else:
            sigma = 0.015 + 0.030 * float(rng.random())
            # occasional big global jump
            if rng.random() < 0.12:
                sigma = 0.08
            x_try += sigma * rng.standard_normal((n, d))
            steps, T0, T1, lr = 700, 0.022, 0.0030, 0.0022
        x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        x_q = adam_quench(x_try, steps, T0, T1, lr)
        vq = max_cos(x_q)
        # Metropolis move on quenched value
        if vq <= v_cur:
            x_cur, v_cur = x_q, vq
        else:
            if rng.random() < np.exp((v_cur - vq) / Tb):
                x_cur, v_cur = x_q, vq
        # snap promising basins
        do_polish = False
        if vq < best:
            do_polish = True
        elif vq < best + 0.0015 and rng.random() < 0.25:
            do_polish = True
        elif vq < best + 0.004 and rng.random() < 0.05:
            do_polish = True
        if do_polish and time.time() < t_end - 15.0:
            xp = lp_polish(x_q, 1.0, t_end - 12.0)
            vp = max_cos(xp)
            if vp < best:
                best, x_best = vp, xp.copy()
                # pull walker to new best region
                x_cur, v_cur = xp.copy(), vp
            elif vp < v_cur and rng.random() < 0.5:
                x_cur, v_cur = xp.copy(), vp
        # occasional reset of walker to best to intensify
        if it % 60 == 0 and rng.random() < 0.5:
            x_cur, v_cur = x_best.copy(), best
    # final snap of best with remaining time
    remain = t_end - time.time() - 1.0
    if remain > 2.0:
        x_best = lp_polish(x_best, min(remain, 12.0), t_end)
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
