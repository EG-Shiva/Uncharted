import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def adam_quench_fast(x_init, steps, T0, T1, lr):
    x = x_init.copy()
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
        Mx = float(G.max())
        W = np.exp((G - Mx) / T)
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


def greedy_holes(X_base, k, rng, n_samp=2000):
    # X_base: (n-k, d) remaining points; sequentially add k farthest points
    d = X_base.shape[1]
    cur = X_base.copy()
    out = []
    for _ in range(k):
        S = rng.standard_normal((n_samp, d))
        S /= np.linalg.norm(S, axis=1, keepdims=True)
        dots = S @ cur.T
        worst = dots.max(axis=1)
        q = S[int(np.argmin(worst))]
        out.append(q)
        cur = np.vstack([cur, q[None, :]])
    return np.array(out)


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
                if a.shape == (102, 12) and np.all(np.isfinite(a)):
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

    it = 0
    # explore until ~22s before end (reserve for final LP)
    while time.time() < t_end - 22.0 and it < 2000:
        it += 1
        mode = it % 5
        x_try = x_cur.copy()
        if mode == 0:
            # large hole re-insertion: remove worst k, greedy refill
            G0 = x_cur @ x_cur.T
            np.fill_diagonal(G0, -2.0)
            rowmax = G0.max(axis=1)
            k = int(8 + rng.integers(0, 6))  # 8..13
            k = min(k, 16)
            order = np.argsort(-rowmax)
            victims = order[:k].copy()
            # randomize half of victims to diversify
            if k >= 4:
                nr = k // 3
                rv = rng.choice(n, size=nr, replace=False)
                victims[-nr:] = rv
                victims = np.unique(victims)
                k = len(victims)
            mask = np.ones(n, dtype=bool)
            mask[victims] = False
            X_rem = x_cur[mask]
            new_pts = greedy_holes(X_rem, k, rng, n_samp=2000)
            new_pts += 0.0015 * rng.standard_normal(new_pts.shape)
            new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
            x_try[victims] = new_pts
            # small jitter others
            x_try[mask] += 0.003 * rng.standard_normal((n - k, d))
            steps, T0, T1, lr = 420, 0.024, 0.0032, 0.0025
        elif mode == 1:
            # collective soft mode via contact-graph smoothing
            G0 = x_cur @ x_cur.T
            np.fill_diagonal(G0, -10.0)
            Mx = float(G0.max())
            Wk = np.exp((G0 - Mx) / 0.08)
            np.fill_diagonal(Wk, 1.0)
            D = Wk.sum(axis=1, keepdims=True)
            z = rng.standard_normal((n, d))
            z -= np.sum(z * x_cur, axis=1, keepdims=True) * x_cur
            for _ in range(4):
                z = (Wk @ z) / D
            z -= np.sum(z * x_cur, axis=1, keepdims=True) * x_cur
            rms = float(np.sqrt((z * z).mean())) + 1e-12
            z /= rms
            amp = 0.14 + 0.10 * float(rng.random())
            x_try = x_cur + amp * z
            steps, T0, T1, lr = 420, 0.022, 0.0032, 0.0025
        elif mode == 2:
            # coherent cap translation
            u = rng.standard_normal(d)
            u /= np.linalg.norm(u)
            proj = x_cur @ u
            mcap = int(14 + rng.integers(0, 10))  # 14..23
            cap = np.argsort(-proj)[:mcap]
            r = rng.standard_normal(d)
            r -= float(r @ u) * u
            rn = np.linalg.norm(r)
            if rn < 1e-9:
                r = np.ones(d)
                r -= float(r @ u) * u
                rn = np.linalg.norm(r)
            r /= rn
            amp = 0.07 + 0.07 * float(rng.random())
            x_try[cap] += amp * r[None, :]
            x_try += 0.003 * rng.standard_normal((n, d))
            steps, T0, T1, lr = 400, 0.020, 0.0030, 0.0022
        elif mode == 3:
            # overloaded cluster kick
            G0 = x_cur @ x_cur.T
            np.fill_diagonal(G0, -2.0)
            rowmax = G0.max(axis=1)
            wi = int(np.argmax(rowmax))
            gr = G0[wi].copy()
            nbr = np.argsort(-gr)[:8]
            cl = np.append(nbr, wi)
            cl = np.unique(cl)
            x_try[cl] += 0.085 * rng.standard_normal((len(cl), d))
            x_try += 0.004 * rng.standard_normal((n, d))
            steps, T0, T1, lr = 400, 0.024, 0.0035, 0.0025
        else:
            # global jitter
            x_try = x_cur + 0.012 * rng.standard_normal((n, d))
            steps, T0, T1, lr = 350, 0.018, 0.0030, 0.0020
        x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        # quench
        try:
            x_q = adam_quench_fast(x_try, steps, T0, T1, lr)
        except Exception:
            continue
        vq = max_cos(x_q)
        if not np.isfinite(vq):
            continue
        if vq < best - 1e-12:
            x_best, best = x_q.copy(), vq
            x_cur, v_cur = x_q.copy(), vq
        elif vq < v_cur or rng.random() < np.exp((v_cur - vq) / 0.0015):
            # accept wander if within reasonable bound, else keep
            if vq < v_cur + 0.004:
                x_cur, v_cur = x_q.copy(), vq
        # if wandering drifted too far, pull back to best
        if v_cur > best + 0.007:
            x_cur = x_best.copy()
            # tiny kick to avoid immediate return
            x_cur += 0.004 * rng.standard_normal((n, d))
            x_cur /= np.linalg.norm(x_cur, axis=1, keepdims=True)
            v_cur = max_cos(x_cur)
            # if still bad, hard reset
            if v_cur > best + 0.007:
                x_cur = x_best.copy()
                v_cur = best
        if time.time() >= t_end - 22.0:
            break

    # final snap: polish best and current, pick winner
    remain = t_end - time.time() - 1.0
    if remain > 3.0:
        # split budget
        b1 = min(remain * 0.6, 12.0)
        b2 = remain - b1
        try:
            x_p1 = lp_polish(x_best, b1, t_end - 0.5)
            v1 = max_cos(x_p1)
        except Exception:
            x_p1, v1 = x_best, best
        if v1 < best:
            x_best, best = x_p1, v1
        # also polish wanderer if distinct and time left
        if b2 > 2.0:
            try:
                # only if wanderer is promising
                if v_cur < best + 0.004:
                    x_p2 = lp_polish(x_cur, b2, t_end - 0.5)
                    v2 = max_cos(x_p2)
                    if v2 < best:
                        x_best, best = x_p2, v2
                else:
                    # extra polish of best
                    x_p3 = lp_polish(x_best, b2, t_end - 0.5)
                    v3 = max_cos(x_p3)
                    if v3 < best:
                        x_best, best = x_p3, v3
            except Exception:
                pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
