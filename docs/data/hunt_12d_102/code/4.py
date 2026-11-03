import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _lp_polish(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best_val = _max_cos(x)
    best_x = x.copy()
    radius = 1e-3
    iu0 = np.triu_indices(n, 1)
    # cache iu arrays
    iu_r, iu_c = iu0
    while time.time() < deadline and radius > 1e-13:
        G = x @ x.T
        c_full = G[iu_r, iu_c]
        m = c_full.max()
        thr = m - max(20.0 * radius, 1e-9)
        sel = c_full > thr
        # cap size for speed
        p = int(sel.sum())
        if p == 0:
            radius *= 0.5
            continue
        if p > 2500:
            # keep top 2500
            idx = np.argpartition(-c_full, 2500)[:2500]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = sel & mask
            p = 2500
        I = iu_r[sel]
        J = iu_c[sel]
        c_sel = c_full[sel]
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
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel,
                          A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        if np.any(~np.isfinite(yn)) or np.any(yn < 1e-12):
            radius *= 0.5
            continue
        y /= yn
        v = _max_cos(y)
        if v < best_val:
            best_val = v
            best_x = y.copy()
            x = y
            radius = min(radius * 1.5, 0.02)
            if time.time() >= deadline:
                break
        else:
            radius *= 0.5
    return best_x


def _soft_anneal(x, steps, a0, a1, lr0, lr1):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    for s in range(steps):
        f = s / max(steps - 1, 1)
        alpha = a0 + (a1 - a0) * f
        lr = lr0 + (lr1 - lr0) * f
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        m = G.max()
        # weights, stable
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad_t = grad - dot * x
        # clip step per row to keep trust region
        step = lr * grad_t
        nm = np.linalg.norm(step, axis=1, keepdims=True)
        # max move 0.05
        scale = np.minimum(1.0, 0.05 / (nm + 1e-16))
        step = step * scale
        x = x - step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def solve():
    import time as _time
    import numpy as _np
    t0 = _time.time()
    HARD_END = t0 + 172.0
    EXPLORE_END = t0 + 148.0
    rng = _np.random.default_rng(12345)

    try:
        best = _np.load("best.npy").astype(float)
    except Exception:
        best = _np.load("record.npy").astype(float)
    # fallback shape check
    if best.shape != (102, 12):
        # try transpose or reshape attempt, else random
        try:
            rec = _np.load("record.npy").astype(float)
            best = rec
        except Exception:
            best = rng.standard_normal((102, 12))
    n, d = 102, 12
    if best.shape != (n, d):
        # truncate/pad deterministically
        tmp = rng.standard_normal((n, d))
        m = min(best.shape[0], n)
        tmp[:m, :min(best.shape[1], d)] = best[:m, :min(best.shape[1], d)]
        best = tmp
    best = best / _np.linalg.norm(best, axis=1, keepdims=True)
    best_val = _max_cos(best)

    def per_point_max(x):
        G = x @ x.T
        _np.fill_diagonal(G, -2.0)
        return G.max(axis=1), G

    def best_hole(others, kcand=3000):
        cand = rng.standard_normal((kcand, d))
        cand /= _np.linalg.norm(cand, axis=1, keepdims=True)
        dots = cand @ others.T
        worst = dots.max(axis=1)
        return cand[_np.argmin(worst)]

    it = 0
    # exploration loop
    while _time.time() < EXPLORE_END:
        it += 1
        mode = it % 7
        trial = best.copy()
        try:
            pm, Gfull = per_point_max(best)
            order_worst = _np.argsort(-pm)  # descending
        except Exception:
            order_worst = _np.arange(n)
            pm = _np.zeros(n)

        if mode == 0:
            # small full shake
            trial = best + 0.008 * rng.standard_normal((n, d))
            trial /= _np.linalg.norm(trial, axis=1, keepdims=True)
            trial = _soft_anneal(trial, 700, 15.0, 90.0, 0.010, 0.004)
        elif mode == 1:
            trial = best + 0.022 * rng.standard_normal((n, d))
            trial /= _np.linalg.norm(trial, axis=1, keepdims=True)
            trial = _soft_anneal(trial, 800, 18.0, 110.0, 0.012, 0.004)
        elif mode == 2:
            # regional shake around worst point + neighbours
            c = int(order_worst[int(rng.integers(0, 5))])
            Gc = (best @ best[c])
            Gc[c] = -10
            nn = _np.argsort(-Gc)[:11]
            region = list(set([c] + nn.tolist()))
            trial[region] += 0.12 * rng.standard_normal((len(region), d))
            trial /= _np.linalg.norm(trial, axis=1, keepdims=True)
            trial = _soft_anneal(trial, 800, 15.0, 100.0, 0.010, 0.004)
        elif mode == 3:
            # single kill + best-hole reinsert
            c = int(order_worst[int(rng.integers(0, 5))])
            mask = _np.ones(n, dtype=bool)
            mask[c] = False
            others = best[mask]
            nh = best_hole(others, 3000)
            trial[c] = nh
            trial = _soft_anneal(trial, 900, 15.0, 120.0, 0.010, 0.003)
        elif mode == 4:
            # double kill
            k2 = [int(order_worst[int(rng.integers(0, 10))]),
                  int(order_worst[int(rng.integers(0, 10))])]
            # ensure distinct
            if k2[0] == k2[1]:
                k2[1] = (k2[1] + 7) % n
            for c in k2:
                mask = _np.ones(n, dtype=bool)
                mask[c] = False
                # use current trial others for second pick
                others = trial[mask]
                trial[c] = best_hole(others, 2000)
            trial = _soft_anneal(trial, 900, 15.0, 110.0, 0.010, 0.004)
        elif mode == 5:
            # global stretch
            A = _np.eye(d) + 0.12 * rng.standard_normal((d, d))
            trial = best @ A
            trial /= _np.linalg.norm(trial, axis=1, keepdims=True)
            trial = _soft_anneal(trial, 700, 18.0, 100.0, 0.010, 0.004)
        else:
            # random subset shake
            k = 8
            idx = rng.choice(n, size=k, replace=False)
            trial[idx] += 0.20 * rng.standard_normal((k, d))
            trial /= _np.linalg.norm(trial, axis=1, keepdims=True)
            trial = _soft_anneal(trial, 700, 15.0, 90.0, 0.012, 0.004)

        v_soft = _max_cos(trial)
        # filter bad trials to save LP time (allow occasional bad)
        if v_soft > best_val + 0.015 and (it % 4 != 0):
            continue
        if _time.time() >= EXPLORE_END:
            # no time for LP, just check soft value
            if v_soft < best_val:
                # quick accept without LP (will polish later)
                best = trial / _np.linalg.norm(trial, axis=1, keepdims=True)
                best_val = v_soft
            continue
        short_deadline = min(_time.time() + 2.2, EXPLORE_END + 1.0, HARD_END - 12.0)
        if short_deadline - _time.time() < 0.4:
            continue
        polished = _lp_polish(trial, short_deadline)
        v = _max_cos(polished)
        if v < best_val - 1e-12:
            best = polished
            best_val = v
        # safety: stop if hard end near
        if _time.time() >= HARD_END - 12.0:
            break

    # final long polish of best
    remain = HARD_END - _time.time()
    if remain > 4.0:
        best = _lp_polish(best, _time.time() + remain - 1.5)
    best = best / _np.linalg.norm(best, axis=1, keepdims=True)
    return best
