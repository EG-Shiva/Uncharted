import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def lp_polish_deadline(x, budget, t_end):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while True:
        now = time.time()
        if now >= t_end or now - start >= budget:
            break
        if radius < 1e-14:
            break
        c = (x @ x.T)[iu]
        mcur = float(c.max())
        sel = c > mcur - max(20.0 * radius, 1e-9)
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
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
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
    l0 = np.log(T0)
    l1 = np.log(T1)
    for s in range(1, steps + 1):
        f = s / steps
        T = float(np.exp(l0 + (l1 - l0) * f))
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        M = float(G.max())
        W = np.exp((G - M) / T)
        np.fill_diagonal(W, 0.0)
        S = float(W.sum())
        if not np.isfinite(S) or S == 0:
            break
        P = W / S
        Grad = P @ x
        dot = np.sum(Grad * x, axis=1, keepdims=True)
        Gt = Grad - dot * x
        m = b1 * m + (1.0 - b1) * Gt
        v = b2 * v + (1.0 - b2) * (Gt * Gt)
        c1 = 1.0 - b1 ** s
        c2 = 1.0 - b2 ** s
        mh = m / c1
        vh = v / c2
        upd = mh / (np.sqrt(vh) + eps)
        # keep tangential
        upd = upd - np.sum(upd * x, axis=1, keepdims=True) * x
        lrs = lr * (1.0 - 0.5 * f)
        x = x - lrs * upd
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x /= nr
    return x


def best_hole(X_rem, rng, n_samp=20000):
    d = X_rem.shape[1]
    S = rng.standard_normal((n_samp, d))
    S /= np.linalg.norm(S, axis=1, keepdims=True)
    dots = S @ X_rem.T
    worst = dots.max(axis=1)
    return S[int(np.argmin(worst))]


def solve():
    t0 = time.time()
    T_END = t0 + 172.0
    HOP_END = t0 + 150.0
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
    cur = best

    it = 0
    while time.time() < HOP_END and time.time() < T_END - 22.0:
        it += 1
        mode = it % 6
        # diagnostics from current
        G0 = x_cur @ x_cur.T
        np.fill_diagonal(G0, -2.0)
        rowmax = G0.max(axis=1)
        order = np.argsort(-rowmax)
        mval = float(G0.max())
        cnt = (G0 > mval - 0.005).sum(axis=1)

        x_try = x_cur.copy()
        if mode == 0:
            x_try = x_cur + 0.012 * rng.standard_normal((n, d))
            steps, T0, T1, lr = 500, 0.02, 0.004, 0.0030
        elif mode == 1:
            k = 2 + (it % 3)  # 2..4
            idx = order[:k]
            x_try[idx] += 0.10 * rng.standard_normal((k, d))
            x_try += 0.003 * rng.standard_normal((n, d))
            steps, T0, T1, lr = 600, 0.025, 0.003, 0.0032
        elif mode == 2:
            ci = int(order[0])
            gr = (x_cur @ x_cur[ci]).copy()
            gr[ci] = -2.0
            nbr = np.argsort(-gr)[:6]
            cl = np.append(nbr, ci)
            x_try[cl] += 0.055 * rng.standard_normal((len(cl), d))
            x_try += 0.003 * rng.standard_normal((n, d))
            steps, T0, T1, lr = 550, 0.02, 0.0035, 0.0030
        elif mode == 3:
            # hole re-insertion of worst 1..3 by rowmax+counts
            score = rowmax + 0.002 * cnt
            worder = np.argsort(-score)
            nk = 1 + int(rng.integers(0, 3))  # 1..3
            if rng.random() < 0.3:
                nk = 4 + int(rng.integers(0, 3))  # occasional 4..6
                nk = min(nk, 8)
            worst = worder[:nk].copy()
            mask = np.ones(n, dtype=bool)
            mask[worst] = False
            X_rem = x_cur[mask]
            # sequential best holes
            X_aug = X_rem.copy()
            new_pts = []
            for _ in range(nk):
                q = best_hole(X_aug, rng, n_samp=16000)
                # quick local refine of q against X_aug
                for __ in range(120):
                    dots = X_aug @ q
                    mm = float(dots.max())
                    w = np.exp((dots - mm) / 0.006)
                    s = float(w.sum())
                    if s == 0 or not np.isfinite(s):
                        break
                    g = (w / s) @ X_aug
                    g = g - float(g @ q) * q
                    q = q - 0.05 * g
                    q /= np.linalg.norm(q)
                new_pts.append(q)
                X_aug = np.vstack([X_aug, q[None, :]])
            x_try[mask] = X_rem
            x_try[worst] = np.array(new_pts)
            x_try += 0.002 * rng.standard_normal((n, d))
            steps, T0, T1, lr = 650, 0.022, 0.0028, 0.0030
        elif mode == 4:
            k = 12
            idx = rng.choice(n, size=k, replace=False)
            x_try[idx] += 0.045 * rng.standard_normal((k, d))
            steps, T0, T1, lr = 500, 0.03, 0.005, 0.0035
        else:
            x_try = x_cur + 0.05 * rng.standard_normal((n, d))
            # occasional full resample of 2 random points
            if rng.random() < 0.4:
                idx = rng.choice(n, size=2, replace=False)
                rr = rng.standard_normal((2, d))
                rr /= np.linalg.norm(rr, axis=1, keepdims=True)
                x_try[idx] = rr
            steps, T0, T1, lr = 600, 0.035, 0.004, 0.0035

        x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        # quench
        try:
            x_q = adam_quench(x_try, steps, T0, T1, lr)
        except Exception:
            continue
        # short LP snap
        remain = HOP_END - time.time()
        if remain < 0.4:
            break
        sb = min(0.9, remain)
        try:
            x_q = lp_polish_deadline(x_q, sb, T_END - 20.0)
        except Exception:
            pass
        try:
            v = max_cos(x_q)
        except Exception:
            continue
        if not np.isfinite(v):
            continue
        if v < best - 1e-12:
            best = v
            x_best = x_q.copy()
            x_cur = x_q.copy()
            cur = v
        elif v < cur or (v < cur + 0.002 and rng.random() < 0.25):
            x_cur = x_q.copy()
            cur = v
        # periodically restart wandering from best
        if it % 25 == 0 and time.time() < HOP_END:
            if rng.random() < 0.7:
                x_cur = x_best.copy()
                cur = best
    # final intensification
    # a few small tweaks around best
    for _ in range(3):
        if time.time() >= T_END - 18.0:
            break
        try:
            xt = x_best + 0.004 * rng.standard_normal((n, d))
            xt /= np.linalg.norm(xt, axis=1, keepdims=True)
            xt = adam_quench(xt, 400, 0.015, 0.003, 0.0025)
            xt = lp_polish_deadline(xt, 2.0, T_END - 1.0)
            v = max_cos(xt)
            if v < best:
                best = v
                x_best = xt.copy()
        except Exception:
            pass
    try:
        x_best = lp_polish_deadline(x_best, T_END - time.time() - 0.5, T_END - 0.2)
    except Exception:
        pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
