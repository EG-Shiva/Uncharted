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
        Gm = x @ x.T
        iu_c = Gm[iu]
        mcur = float(iu_c.max())
        sel = iu_c > mcur - max(20 * radius, 1e-9)
        I = iu[0][sel]
        J = iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-iu_c[sel], A_eq=A_eq,
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


def refine_hole(q, X_rem, iters=150, T=0.004, lr=0.02):
    q = q / np.linalg.norm(q)
    for _ in range(iters):
        dots = X_rem @ q
        mm = float(dots.max())
        w = np.exp((dots - mm) / T)
        s = float(w.sum())
        if s == 0 or not np.isfinite(s):
            break
        g = (w @ X_rem) / s
        gt = g - float(g @ q) * q
        nrm = float(np.linalg.norm(gt))
        if nrm < 1e-12:
            break
        q = q - lr * gt
        nrm2 = float(np.linalg.norm(q))
        if nrm2 == 0:
            break
        q /= nrm2
        lr *= 0.995
    return q


def solve():
    t0 = time.time()
    T_END = t0 + 170.0
    HOP_END = t0 + 150.0
    rng = np.random.default_rng(7)
    try:
        xb = np.array(np.load("best.npy"), dtype=float)
    except Exception:
        xb = None
    try:
        xr = np.array(np.load("record.npy"), dtype=float)
    except Exception:
        xr = None
    cands = []
    for c in (xb, xr):
        if c is not None and c.shape == (102, 12):
            try:
                a = c / np.linalg.norm(c, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
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
    slack = 5e-4
    it = 0
    # pre-sample pool for holes (refresh occasionally)
    while time.time() < HOP_END and time.time() < T_END - 22.0:
        it += 1
        if time.time() > T_END - 22.0:
            break
        G0 = x_best @ x_best.T
        np.fill_diagonal(G0, -2.0)
        rowmax = G0.max(axis=1)
        mval = float(rowmax.max())
        contacts = (G0 > mval - 0.005).sum(axis=1)
        order_row = np.argsort(-rowmax)
        order_con = np.argsort(-contacts)
        mode = it % 5
        x_try = x_best.copy()
        if mode == 0:
            x_try = x_best + 0.012 * rng.standard_normal((n, d))
            steps, T0, T1, lr = 650, 0.018, 0.0035, 0.0020
        elif mode == 1:
            k = 3
            idx = order_row[:k]
            x_try[idx] += 0.085 * rng.standard_normal((k, d))
            x_try += 0.003 * rng.standard_normal((n, d))
            steps, T0, T1, lr = 850, 0.022, 0.0030, 0.0022
        elif mode == 2:
            k = int(rng.integers(2, 5))
            idx = order_con[:k]
            x_try[idx] += 0.07 * rng.standard_normal((k, d))
            x_try += 0.003 * rng.standard_normal((n, d))
            steps, T0, T1, lr = 850, 0.025, 0.0032, 0.0022
        elif mode == 3:
            nk = int(rng.integers(1, 3))  # 1-2 holes
            worst = order_con[:nk].copy()
            # also sometimes use rowmax
            if rng.random() < 0.5:
                worst = order_row[:nk].copy()
            mask = np.ones(n, dtype=bool)
            mask[worst] = False
            X_rem = x_best[mask]
            cand = rng.standard_normal((12000, d))
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            X_cur = X_rem.copy()
            for wi in worst:
                sc = (cand @ X_cur.T).max(axis=1)
                bi = int(np.argmin(sc))
                q = cand[bi].copy()
                q = refine_hole(q, X_cur, iters=120, T=0.004, lr=0.02)
                # replace: grow X_cur
                X_cur = np.vstack([X_cur, q[None, :]])
                x_try[wi] = q
            # small jitter to rest to allow adjustment
            x_try[mask] += 0.002 * rng.standard_normal((n - nk, d))
            steps, T0, T1, lr = 900, 0.020, 0.0028, 0.0020
        else:
            ci = int(rng.integers(n))
            gr = (x_best @ x_best[ci]).copy()
            gr[ci] = -2.0
            nbr = np.argsort(-gr)[:5]
            cl = np.append(nbr, ci)
            x_try[cl] += 0.055 * rng.standard_normal((len(cl), d))
            x_try += 0.0025 * rng.standard_normal((n, d))
            steps, T0, T1, lr = 800, 0.022, 0.0030, 0.0021
        # normalize perturbed
        nr = np.linalg.norm(x_try, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x_try /= nr
        # quench
        x_q = adam_quench(x_try, steps, T0, T1, lr)
        v_q = max_cos(x_q)
        if v_q < best + slack:
            # snap promising basin
            remain = min(3.0, max(0.5, (HOP_END - time.time()) / 4.0))
            if time.time() + remain > T_END - 20.0:
                remain = max(0.5, T_END - 20.0 - time.time())
            if remain > 0.4:
                x_p = lp_polish(x_q, seconds=remain, t_end=T_END - 18.0)
                v_p = max_cos(x_p)
                if v_p < best - 1e-12:
                    best = v_p
                    x_best = x_p
                    # tighten slack slightly after improvement
                    slack = max(2e-4, slack * 0.95)
        # opportunistic: if quench alone beats best without LP (rare), keep LP result already handled;
        # also track direct quench improvement to avoid missing LP failure
        if v_q < best - 1e-12:
            # quench itself is better; polish quickly next loop will handle, but accept now
            # do a very short polish to make it a true optimum
            try:
                x_p2 = lp_polish(x_q, seconds=1.0, t_end=T_END - 18.0)
                v_p2 = max_cos(x_p2)
                if v_p2 < best:
                    best = v_p2
                    x_best = x_p2
                else:
                    best = v_q
                    x_best = x_q
            except Exception:
                best = v_q
                x_best = x_q
    # final polish with remaining time
    try:
        t_rem = float(T_END - time.time()) - 2.0
        if t_rem > 3.0:
            x_best = lp_polish(x_best, seconds=t_rem, t_end=T_END)
    except Exception:
        pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
