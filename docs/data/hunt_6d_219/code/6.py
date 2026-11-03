import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = c.max()
        thr = max(20 * radius, 1e-9)
        sel = c > m - thr
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > 2500:
            # keep only top 2500 pairs for speed
            top = np.argpartition(c, -2500)[-2500:]
            sel = np.zeros_like(sel)
            sel[top] = True
            cnt = 2500
        I = iu[0][sel]
        J = iu[1][sel]
        cc = c[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix(
                (x.ravel(),
                 (np.repeat(np.arange(n), d), np.arange(n * d))),
                shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cc, A_eq=A_eq,
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
        nrm = np.linalg.norm(y, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        y = y / nrm
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def smooth_adam(x, steps, T0, T1, lr):
    n, d = x.shape
    mom = np.zeros_like(x)
    vel = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        frac = t / steps
        T = T0 * ((T1 / T0) ** frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        Gm = G.max()
        E = np.exp((G - Gm) / T)
        np.fill_diagonal(E, 0.0)
        S = E.sum()
        if S == 0 or not np.isfinite(S):
            break
        W = E / S
        grad = 2.0 * (W @ x)
        dot = (grad * x).sum(axis=1, keepdims=True)
        gt = grad - dot * x
        mom = b1 * mom + (1 - b1) * gt
        vel = b2 * vel + (1 - b2) * (gt * gt)
        mhat = mom / (1 - b1 ** t)
        vhat = vel / (1 - b2 ** t)
        step = mhat / (np.sqrt(vhat) + eps)
        dot2 = (step * x).sum(axis=1, keepdims=True)
        stept = step - dot2 * x
        # mild step clipping for stability
        sn = np.linalg.norm(stept, axis=1, keepdims=True)
        # Adam steps are O(1) per coord; clip huge moves
        stept = stept / np.maximum(1.0, sn * 2.0)
        x = x - lr * stept
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x


def solve():
    t0 = time.time()
    deadline = t0 + 172.0
    rng = np.random.default_rng(7)

    def normed(a):
        a = np.asarray(a, dtype=np.float64)
        nrm = np.linalg.norm(a, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        return a / nrm

    try:
        xb = normed(np.load("best.npy"))
    except Exception:
        xb = None
    try:
        xr = normed(np.load("record.npy"))
    except Exception:
        xr = None
    if xb is None and xr is None:
        r = rng.normal(0, 1, size=(219, 6))
        x_best = normed(r)
        best = max_cos(x_best)
    elif xb is None:
        x_best = xr
        best = max_cos(x_best)
    elif xr is None:
        x_best = xb
        best = max_cos(x_best)
    else:
        vb = max_cos(xb)
        vr = max_cos(xr)
        if vr < vb:
            x_best, best = xr, vr
        else:
            x_best, best = xb, vb

    n, d = x_best.shape
    x_cur = x_best.copy()
    cur = best
    iu_all = np.triu_indices(n, 1)
    cycle = 0

    while time.time() < deadline - 20.0 and cycle < 300:
        cycle += 1
        # --- analyse worst region from current ---
        Gc = x_cur @ x_cur.T
        np.fill_diagonal(Gc, -2.0)
        c_all = Gc[iu_all]
        K = min(800, c_all.size)
        part = np.argpartition(c_all, -K)[-K:]
        Ik = iu_all[0][part]
        Jk = iu_all[1][part]
        deg = np.bincount(np.concatenate([Ik, Jk]), minlength=n)
        worst_order = np.argsort(-deg)

        if cycle % 4 == 0:
            sigma_all = 0.008
        else:
            sigma_all = 0.003
        if cycle % 5 == 0:
            k_kick = 10
        elif cycle % 2 == 0:
            k_kick = 6
        else:
            k_kick = 4
        if cycle % 7 == 0:
            sigma_kick = 0.25
        elif cycle % 3 == 0:
            sigma_kick = 0.10
        else:
            sigma_kick = 0.05
        full_resample = (cycle % 11 == 0)

        x_try = x_cur + rng.normal(0, sigma_all, size=(n, d))
        # build kick set: mix of worst + random
        n_worst = min(3, k_kick // 2)
        kick_set = set(worst_order[:n_worst].tolist())
        need = k_kick - len(kick_set)
        if need > 0:
            rnd = rng.choice(n, size=need, replace=False).tolist()
            for v in rnd:
                kick_set.add(int(v))
        kick_idx = np.array(sorted(kick_set), dtype=int)
        x_try[kick_idx] += rng.normal(0, sigma_kick, size=(len(kick_idx), d))
        if full_resample:
            ridx = rng.choice(n, size=2, replace=False)
            x_try[ridx] = rng.normal(0, 1, size=(2, d))
        x_try = normed(x_try)

        # --- smooth repulsion ---
        if cycle % 6 == 0:
            steps = 2000
        else:
            steps = 1200
        T0, T1 = 0.025, 0.004
        lr = 0.003
        x_try = smooth_adam(x_try, steps, T0, T1, lr)
        v_try = max_cos(x_try)

        # --- LP refine if promising ---
        if v_try < best + 0.01:
            budget = min(4.0, (deadline - 20.0) - time.time() - 0.3)
            if budget > 0.6:
                x_try = lp_polish(x_try, budget)
                v_try = max_cos(x_try)

        if v_try < best - 1e-12:
            x_best = x_try.copy()
            best = v_try
            x_cur = x_try.copy()
            cur = v_try
        elif v_try < cur - 1e-12:
            x_cur = x_try.copy()
            cur = v_try
        elif v_try < cur + 0.001 and rng.random() < 0.30:
            x_cur = x_try.copy()
            cur = v_try
        else:
            if cycle % 9 == 0:
                x_cur = x_best.copy()
                cur = best

    final_budget = (deadline - 2.0) - time.time()
    if final_budget > 1.0:
        x_best = lp_polish(x_best, final_budget)
    x_best = normed(x_best)
    return x_best
