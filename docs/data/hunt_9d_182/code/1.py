import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def adam_refine(x_init, steps, T0, T1, lr0, lr1, time_limit, it_offset=0):
    t0 = time.time()
    n, d = x_init.shape
    X = x_init / np.linalg.norm(x_init, axis=1, keepdims=True)
    G0 = X @ X.T
    np.fill_diagonal(G0, -2.0)
    best_val = float(G0.max())
    best_x = X.copy()
    m1 = np.zeros_like(X)
    m2 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    # geometric schedules
    lt0, lt1 = np.log(T0), np.log(T1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for it in range(steps):
        if time.time() - t0 > time_limit:
            break
        frac = it / max(1, steps - 1)
        T = float(np.exp(lt0 + (lt1 - lt0) * frac))
        lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        if m < best_val:
            best_val = m
            best_x = X.copy()
        # softmax weights, stable (G-m<=0)
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        s = float(E.sum() * 0.5)
        if not np.isfinite(s) or s <= 0:
            continue
        Grad = (E @ X) / s
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        glob = it + 1 + it_offset
        m1 *= b1
        m1 += (1 - b1) * Gt
        m2 *= b2
        m2 += (1 - b2) * (Gt * Gt)
        m1h = m1 / (1 - b1 ** glob)
        m2h = m2 / (1 - b2 ** glob)
        step = m1h / (np.sqrt(m2h) + eps)
        # clip extreme steps
        # np.clip(step, -3.0, 3.0, out=step)
        X = X - lr * step
        nrm = np.linalg.norm(X, axis=1, keepdims=True)
        # guard NaN
        nrm[~np.isfinite(nrm)] = 1.0
        X = X / nrm
    # final eval
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    if m < best_val:
        best_val = m
        best_x = X.copy()
    return best_x, best_val


def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    best_x = x.copy()
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        # cap size for speed
        if np.count_nonzero(sel) > 4000:
            # keep tightest 4000
            idx = np.argpartition(-c, 4000)[:4000]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx] = True
            sel = sel & mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, n * d + 1)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(n * d + 1)
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
        nr[~np.isfinite(nr)] = 1.0
        nr[nr < 1e-12] = 1.0
        y /= nr
        v = max_cos(y)
        if v < best:
            best = v
            best_x = y.copy()
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return best_x


def solve():
    np.random.seed(0)
    t_start = time.time()
    TIME_BUDGET = 168.0
    # load start
    try:
        x0 = np.load("best.npy")
    except Exception:
        try:
            x0 = np.load("record.npy")
        except Exception:
            rng = np.random.RandomState(0)
            x0 = rng.randn(182, 9)
    x0 = np.asarray(x0, dtype=np.float64)
    # ensure shape (182,9)
    if x0.shape != (182, 9):
        # try to reshape / truncate / pad deterministically
        rng = np.random.RandomState(1)
        y = rng.randn(182, 9)
        mh = min(x0.shape[0], 182)
        mw = min(x0.shape[1], 9) if x0.ndim == 2 else 1
        if x0.ndim == 2:
            y[:mh, :mw] = x0[:mh, :mw]
        x0 = y
    x0 = x0 / np.linalg.norm(x0, axis=1, keepdims=True)
    best_val = max_cos(x0)
    x_best = x0.copy()

    def remaining():
        return TIME_BUDGET - (time.time() - t_start)

    # Stage 1: broad annealed refine from best
    if remaining() > 20:
        xb, vb = adam_refine(x_best, steps=14000, T0=0.004, T1=0.00035,
                             lr0=0.003, lr1=0.00025,
                             time_limit=min(55.0, remaining() - 15))
        if vb < best_val:
            best_val = vb
            x_best = xb

    # Stage 2: basin hopping - perturb worst points
    # identify worst pairs once per trial
    n = 182
    for trial in range(7):
        if remaining() < 35:
            break
        rng = np.random.RandomState(100 + trial)
        # find worst pairs to focus kick
        G = x_best @ x_best.T
        np.fill_diagonal(G, -10.0)
        # top pairs
        flat = G[np.triu_indices(n, 1)]
        # threshold for involved points
        thr = np.sort(flat)[-400] if len(flat) >= 400 else flat.min()
        cnt = np.zeros(n, dtype=np.float64)
        # count involvement in top slice
        iu0, iu1 = np.triu_indices(n, 1)
        msk = flat > thr - 1e-12
        for a, b in zip(iu0[msk], iu1[msk]):
            cnt[a] += 1
            cnt[b] += 1
        cnt = cnt / max(1.0, cnt.max())
        sigma_base = [0.006, 0.010, 0.015, 0.008, 0.020, 0.012, 0.005][trial]
        noise = rng.randn(n, 9)
        # scale per point: more noise on jammed points
        scale = sigma_base * (0.5 + cnt)[:, None]
        Xp = x_best + scale * noise
        # extra: push current max pair apart a bit
        try:
            k = int(np.argmax(flat))
            a, b = int(iu0[k]), int(iu1[k])
            v = rng.randn(9)
            v /= np.linalg.norm(v) + 1e-12
            Xp[a] += sigma_base * 2.0 * v
            Xp[b] -= sigma_base * 2.0 * v
        except Exception:
            pass
        Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
        xb, vb = adam_refine(Xp, steps=4500, T0=0.002, T1=0.00025,
                             lr0=0.002, lr1=0.0002,
                             time_limit=min(12.0, remaining() - 25),
                             it_offset=14000 + trial * 4500)
        if vb < best_val - 1e-12:
            best_val = vb
            x_best = xb

    # Stage 3: sharp fine equalization with very low T
    if remaining() > 12:
        xb, vb = adam_refine(x_best, steps=6000, T0=0.0006, T1=0.00015,
                             lr0=0.0008, lr1=0.00015,
                             time_limit=min(20.0, remaining() - 8),
                             it_offset=50000)
        if vb < best_val:
            best_val = vb
            x_best = xb

    # Stage 4: LP trust-region polish for remainder
    rem = remaining() - 1.5
    if rem > 5:
        # cap to avoid overrun
        rem = min(rem, 40.0)
        try:
            x_best = lp_polish(x_best, rem)
            best_val = max_cos(x_best)
        except Exception:
            pass

    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    # strict guards
    x_best[~np.isfinite(x_best)] = 0.1
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
