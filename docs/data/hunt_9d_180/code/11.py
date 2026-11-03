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
        c_all = G[iu]
        m = float(c_all.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = c_all > thr
        # cap constraints for speed
        p = int(np.count_nonzero(sel))
        if p == 0:
            break
        I_all = iu[0][sel]
        J_all = iu[1][sel]
        c_sel = c_all[sel]
        if p > 2500:
            # keep largest 2500
            idx = np.argpartition(-c_sel, 2500)[:2500]
            I_all = I_all[idx]
            J_all = J_all[idx]
            c_sel = c_sel[idx]
            p = 2500
        I = I_all
        J = J_all
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        # cols: I*d+arange, J*d+arange, last
        ar = np.arange(d)
        cols_I = (I[:, None] * d + ar).ravel()
        cols_J = (J[:, None] * d + ar).ravel()
        cols = np.concatenate([cols_I, cols_J, np.full(p, n * d)])
        # need interleaved per row: [xJ | xI | -1]
        # build vals in row-major order matching rows repeat
        vals = np.empty(p * (2 * d + 1))
        vals[0::] = 0  # placeholder
        # fill row by row via concatenation in correct order
        block = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        vals = block
        # rows already repeat, but cols currently grouped; need row-major cols
        cols_rowmajor = np.concatenate(
            [np.concatenate([(I[k] * d + ar), (J[k] * d + ar), [n * d]]) for k in range(p)]
        ) if p < 5000 else None
        # for speed use vectorized construction when p large: build coo directly
        if cols_rowmajor is not None:
            cols = cols_rowmajor
        else:
            # fallback grouped construction needs matching rows order -> rebuild rows grouped? use simple loop-free:
            # rows repeat is already row-major if vals row-major and cols row-major
            pass
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix(
                (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                shape=(n, nv),
            ).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(
                obj,
                A_ub=A_ub,
                b_ub=-c_sel,
                A_eq=A_eq,
                b_eq=np.zeros(n),
                bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                method="highs",
            )
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def smooth_anneal(x, iters, k0, k1, lr0, lr1, mu=0.85):
    n, d = x.shape
    vel = np.zeros_like(x)
    for it in range(iters):
        f = it / (iters - 1) if iters > 1 else 1.0
        k = k0 + (k1 - k0) * f
        lr = lr0 + (lr1 - lr0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        W = np.exp(k * (G - gmax))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad -= dot * x
        rms = np.sqrt(np.mean(grad * grad)) + 1e-12
        grad /= rms
        vel = mu * vel + (1.0 - mu) * grad
        x = x - lr * vel
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def refine_void(v0, xk, steps=120, k=30.0, lr=0.03):
    v = v0 / (np.linalg.norm(v0) + 1e-12)
    for _ in range(steps):
        s = xk @ v
        smax = float(s.max())
        w = np.exp(k * (s - smax))
        g = (w[:, None] * xk).sum(axis=0)
        g = g - float(g @ v) * v
        gn = float(np.linalg.norm(g)) + 1e-12
        g /= gn
        v = v - lr * g
        v /= (np.linalg.norm(v) + 1e-12)
    return v


def solve():
    t0 = time.time()
    HARD = 168.0  # leave margin for 180s limit
    np.random.seed(0)
    # load start
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
        if c is not None and c.shape == (180, 9):
            c = c / np.linalg.norm(c, axis=1, keepdims=True)
            cands.append(c)
    if not cands:
        rng0 = np.random.default_rng(0)
        c = rng0.standard_normal((180, 9))
        c /= np.linalg.norm(c, axis=1, keepdims=True)
        cands.append(c)
    # pick best of loads
    vals = [max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    x_best = cands[bi].copy()
    best_val = float(vals[bi])

    # quick contact analysis helper
    def get_counts(x, tol):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        cnt = np.count_nonzero(G > m - tol, axis=1)
        return m, cnt

    trial = 0
    # main basin-hopping loop, reserve ~18s for final polish
    while time.time() - t0 < HARD - 18.0:
        trial += 1
        elapsed = time.time() - t0
        remaining = HARD - elapsed
        if remaining < 8.0:
            break
        rng = np.random.default_rng(1000 + trial * 7919)
        # contact info on current best
        try:
            Gtmp = x_best @ x_best.T
            np.fill_diagonal(Gtmp, -2.0)
            m0 = float(Gtmp.max())
            cnt_strict = np.count_nonzero(Gtmp > m0 - 1e-9, axis=1)
            cnt_loose = np.count_nonzero(Gtmp > m0 - 0.008, axis=1)
        except Exception:
            m0 = best_val
            cnt_strict = np.zeros(180, dtype=int)
            cnt_loose = np.zeros(180, dtype=int)
        rattlers = np.where(cnt_strict == 0)[0].tolist()
        order_loose = np.argsort(-cnt_loose)
        # number to move
        n_move = int(rng.integers(2, 6))  # 2..5
        movers = []
        for r in rattlers:
            if len(movers) < n_move:
                movers.append(int(r))
        for idx in order_loose:
            if len(movers) >= n_move:
                break
            idx = int(idx)
            if idx not in movers:
                # prefer overloaded or random
                movers.append(idx)
        # add one purely random if still short / replace one with random for diversity
        if trial % 3 == 0 and len(movers) > 0:
            rr = int(rng.integers(0, 180))
            if rr not in movers:
                movers[-1] = rr
        movers = movers[:n_move]
        keep_mask = np.ones(180, dtype=bool)
        keep_mask[movers] = False
        x_keep = x_best[keep_mask]

        # find voids sequentially
        xk_aug = x_keep.copy()
        new_pts = []
        for _ in range(n_move):
            cand = rng.standard_normal((4000, 9))
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            sc = cand @ xk_aug.T
            mx = sc.max(axis=1)
            j = int(np.argmin(mx))
            v0 = cand[j]
            v = refine_void(v0, xk_aug, steps=100, k=28.0, lr=0.03)
            new_pts.append(v)
            xk_aug = np.vstack([xk_aug, v[None, :]])

        x_try = x_best.copy()
        for k_i, mi in enumerate(movers):
            x_try[mi] = new_pts[k_i]
        # small global shake
        sigma = float(rng.uniform(0.002, 0.008))
        x_try = x_try + sigma * rng.standard_normal(x_try.shape)
        x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)

        # smooth anneal params varied
        iters = int(rng.integers(800, 1800))
        k0 = float(rng.uniform(10.0, 25.0))
        k1 = float(rng.uniform(90.0, 230.0))
        lr0 = float(rng.uniform(0.004, 0.009))
        lr1 = lr0 * 0.06
        mu = float(rng.uniform(0.80, 0.90))
        try:
            x_sm = smooth_anneal(x_try, iters, k0, k1, lr0, lr1, mu)
        except Exception:
            continue
        try:
            v_sm = max_cos(x_sm)
        except Exception:
            continue
        # only polish if promising
        if v_sm > best_val + 0.012:
            continue
        # short LP polish, budget 3-4s or remaining
        sec = 3.5
        if remaining < 25.0:
            sec = 2.5
        # ensure we don't exceed hard limit
        sec = min(sec, max(1.0, HARD - (time.time() - t0) - 14.0))
        if sec < 0.8:
            # just check smooth result
            if v_sm < best_val:
                x_best, best_val = x_sm, v_sm
            continue
        try:
            x_pol = lp_polish(x_sm, sec)
            v_pol = max_cos(x_pol)
        except Exception:
            continue
        if v_pol < best_val:
            x_best = x_pol
            best_val = v_pol
        # early exit if time low
        if time.time() - t0 > HARD - 18.0:
            break

    # final polish with remaining time
    rem = HARD - (time.time() - t0) - 1.0
    if rem > 2.0:
        try:
            x_best = lp_polish(x_best, min(rem, 14.0))
            best_val = max_cos(x_best)
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    # guard NaN / zero
    x_best = np.nan_to_num(x_best)
    nr = np.linalg.norm(x_best, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    x_best = x_best / nr
    return x_best
