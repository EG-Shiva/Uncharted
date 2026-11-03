import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _relax(x, steps, a0, a1, lr0, lr1, move=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    for s in range(steps):
        f = s / max(steps - 1, 1)
        alpha = a0 + (a1 - a0) * f
        lr = lr0 + (lr1 - lr0) * f
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        m = G.max()
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad = grad - dot * x
        if move is not None:
            grad[~move] = 0.0
        step = lr * grad
        nm = np.linalg.norm(step, axis=1, keepdims=True)
        scale = np.minimum(1.0, 0.05 / (nm + 1e-16))
        step = step * scale
        x = x - step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _lp_polish(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best_val = _max_cos(x)
    best_x = x.copy()
    radius = 1e-3
    iu_r, iu_c = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-13:
        G = x @ x.T
        c_full = G[iu_r, iu_c]
        m = c_full.max()
        thr = m - max(20.0 * radius, 1e-9)
        sel = c_full > thr
        p = int(sel.sum())
        if p == 0:
            radius *= 0.5
            continue
        if p > 2200:
            idx = np.argpartition(-c_full, 2200)[:2200]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = sel & mask
            p = 2200
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
            if time.time() >= deadline - 0.15:
                break
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

def solve():
    import time as _time
    import numpy as _np
    t0 = _time.time()
    HARD_END = t0 + 171.0
    EXPLORE_END = t0 + 122.0
    POOL_END = t0 + 163.0
    rng = _np.random.default_rng(7)
    try:
        best = _np.load("best.npy").astype(float)
    except Exception:
        try:
            best = _np.load("record.npy").astype(float)
        except Exception:
            best = rng.standard_normal((102, 12))
    n, d = 102, 12
    if best.shape != (n, d):
        try:
            rec = _np.load("record.npy").astype(float)
            if rec.shape == (n, d):
                best = rec
            else:
                raise ValueError
        except Exception:
            tmp = rng.standard_normal((n, d))
            m0 = min(best.shape[0], n)
            m1 = min(best.shape[1], d)
            tmp[:m0, :m1] = best[:m0, :m1]
            best = tmp
    best = best / _np.linalg.norm(best, axis=1, keepdims=True)
    best_val = _max_cos(best)

    def per_point_info(x):
        G = x @ x.T
        _np.fill_diagonal(G, -2.0)
        m = G.max()
        pm = G.max(axis=1)
        cnt = (G > m - 0.005).sum(axis=1)
        return pm, cnt, G, m

    def best_hole(others, kcand):
        cand = rng.standard_normal((kcand, d))
        cand /= _np.linalg.norm(cand, axis=1, keepdims=True)
        dots = cand @ others.T
        worst = dots.max(axis=1)
        return cand[_np.argmin(worst)]

    pool = []  # list of (val, arr)

    def push_pool(val, arr):
        if val > best_val + 0.012:
            return
        pool.append((val, arr.copy()))
        if len(pool) > 10:
            pool.sort(key=lambda t: t[0])
            del pool[10:]

    it = 0
    while _time.time() < EXPLORE_END:
        it += 1
        mode = it % 8
        try:
            pm, cnt, Gf, mv = per_point_info(best)
            order_worst = _np.argsort(-pm)
            order_load = _np.argsort(-cnt)
        except Exception:
            order_worst = _np.arange(n)
            order_load = _np.arange(n)
        trial = best.copy()
        if mode == 0:
            trial = best + 0.010 * rng.standard_normal((n, d))
            trial /= _np.linalg.norm(trial, axis=1, keepdims=True)
            trial = _relax(trial, 450, 15.0, 90.0, 0.010, 0.004)
        elif mode == 1:
            trial = best + 0.030 * rng.standard_normal((n, d))
            trial /= _np.linalg.norm(trial, axis=1, keepdims=True)
            trial = _relax(trial, 550, 18.0, 110.0, 0.012, 0.004)
        elif mode == 2:
            c = int(order_worst[int(rng.integers(0, 6))])
            mask = _np.ones(n, dtype=bool); mask[c] = False
            trial[c] = best_hole(best[mask], 5000)
            trial = _relax(trial, 650, 15.0, 120.0, 0.010, 0.003)
        elif mode == 3:
            k = int(rng.integers(3, 5))
            kills = []
            # pick from overloaded top-12
            cand_idx = order_load[:12].tolist()
            rng.shuffle(cand_idx)
            kills = cand_idx[:k]
            cur = trial.copy()
            for c in kills:
                msk = _np.ones(n, dtype=bool); msk[c] = False
                # use cur others excluding c but with already replaced holes
                others = cur[msk]
                cur[c] = best_hole(others, 3000)
            trial = cur
            trial = _relax(trial, 750, 12.0, 110.0, 0.012, 0.003)
        elif mode == 4:
            c = int(order_worst[int(rng.integers(0, 4))])
            Gc = (best @ best[c]); Gc[c] = -10
            nn = _np.argsort(-Gc)[:11]
            region = list(set([c] + nn.tolist()))
            trial[region] += 0.25 * rng.standard_normal((len(region), d))
            trial /= _np.linalg.norm(trial, axis=1, keepdims=True)
            trial = _relax(trial, 650, 14.0, 110.0, 0.011, 0.003)
        elif mode == 5:
            # coordinated shift of a jammed cluster
            c = int(order_load[int(rng.integers(0, 6))])
            Gc = (best @ best[c]); Gc[c] = -10
            nn = _np.argsort(-Gc)[:7]
            S = [c] + nn.tolist()
            v = rng.standard_normal(d); v /= _np.linalg.norm(v) + 1e-16
            step_len = 0.15 + 0.10 * rng.random()
            # make tangent-ish common shift
            for idx in S:
                trial[idx] = trial[idx] + step_len * (v - (v @ trial[idx]) * trial[idx])
                trial[idx] += 0.05 * rng.standard_normal(d)
            trial /= _np.linalg.norm(trial, axis=1, keepdims=True)
            trial = _relax(trial, 650, 14.0, 120.0, 0.011, 0.003)
        elif mode == 6:
            k = 12
            # random subset with two-stage relax (frozen then all)
            idx = rng.choice(n, size=k, replace=False)
            mv_mask = _np.zeros(n, dtype=bool); mv_mask[idx] = True
            trial[idx] += 0.28 * rng.standard_normal((k, d))
            trial /= _np.linalg.norm(trial, axis=1, keepdims=True)
            trial = _relax(trial, 300, 12.0, 60.0, 0.012, 0.005, move=mv_mask)
            trial = _relax(trial, 400, 20.0, 120.0, 0.010, 0.003)
        else:
            A = _np.eye(d) + 0.10 * rng.standard_normal((d, d))
            trial = best @ A
            trial /= _np.linalg.norm(trial, axis=1, keepdims=True)
            trial = _relax(trial, 550, 16.0, 100.0, 0.010, 0.004)

        v_soft = _max_cos(trial)
        if v_soft < best_val - 1e-12:
            best = trial / _np.linalg.norm(trial, axis=1, keepdims=True)
            best_val = v_soft
            push_pool(v_soft, best)
        else:
            push_pool(v_soft, trial)
        if _time.time() >= HARD_END - 15.0:
            break

    # LP-polish pool in order of soft value
    pool.sort(key=lambda t: t[0])
    # always include current best
    cand_list = []
    seen = False
    for v, a in pool[:8]:
        cand_list.append((v, a))
    # dedup not needed
    for v, a in cand_list:
        if _time.time() >= POOL_END:
            break
        dl = min(_time.time() + 4.5, POOL_END, HARD_END - 8.0)
        if dl - _time.time() < 0.5:
            break
        try:
            p = _lp_polish(a, dl)
            vv = _max_cos(p)
            if vv < best_val - 1e-12:
                best = p
                best_val = vv
        except Exception:
            continue
        if _time.time() >= HARD_END - 8.0:
            break
    remain = HARD_END - _time.time()
    if remain > 3.0:
        try:
            best = _lp_polish(best, _time.time() + remain - 1.0)
        except Exception:
            pass
    best = best / _np.linalg.norm(best, axis=1, keepdims=True)
    return best
