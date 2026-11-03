import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, deadline, max_pairs=1800):
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
        if p > max_pairs:
            idx = np.argpartition(-c_full, max_pairs)[:max_pairs]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = sel & mask
            p = max_pairs
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

def _soft_descent(x, steps, a0, a1, lr0, lr1, mu=0.8, max_move=0.05):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    vel = np.zeros_like(x)
    la = np.log(a1 / a0) if a0 > 0 and a1 > 0 else 0.0
    for s in range(steps):
        f = s / max(steps - 1, 1)
        alpha = a0 * np.exp(la * f)
        lr = lr0 + (lr1 - lr0) * f
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        m = G.max()
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad_t = grad - dot * x
        # heavy-ball
        vel = mu * vel + (1.0 - mu) * grad_t
        step = lr * (grad_t + 0.5 * vel)
        nm = np.linalg.norm(step, axis=1, keepdims=True)
        scale = np.minimum(1.0, max_move / (nm + 1e-16))
        step = step * scale
        x = x - step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def solve():
    import time as _time
    import numpy as _np
    t0 = _time.time()
    HARD_END = t0 + 172.0
    EXPLORE_END = t0 + 146.0
    rng = _np.random.default_rng(97531)
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
                raise ValueError("bad")
        except Exception:
            tmp = rng.standard_normal((n, d))
            m0 = min(best.shape[0], n)
            m1 = min(best.shape[1], d)
            tmp[:m0, :m1] = best[:m0, :m1]
            best = tmp
    best = best / _np.linalg.norm(best, axis=1, keepdims=True)
    best_val = _max_cos(best)
    cur = best.copy()
    cur_val = best_val

    def per_point_max(x):
        G = x @ x.T
        _np.fill_diagonal(G, -2.0)
        pm = G.max(axis=1)
        return pm, G

    def best_hole(others, kcand=2000):
        cand = rng.standard_normal((kcand, d))
        cand /= _np.linalg.norm(cand, axis=1, keepdims=True)
        dots = cand @ others.T
        worst = dots.max(axis=1)
        return cand[_np.argmin(worst)]

    it = 0
    TEMP0 = 0.0025
    while _time.time() < EXPLORE_END:
        it += 1
        if _time.time() >= HARD_END - 14.0:
            break
        # occasionally restart wandering state from best
        if it % 12 == 0 and rng.random() < 0.5:
            cur = best.copy()
            cur_val = best_val
        base = cur
        try:
            pm, Gfull = per_point_max(base)
            order_worst = _np.argsort(-pm)
            # overload count within 2e-3 of global max
            mglob = pm.max()
            cnt = (Gfull > mglob - 0.002).sum(axis=1)
            order_over = _np.argsort(-cnt)
        except Exception:
            order_worst = _np.arange(n)
            order_over = _np.arange(n)

        mode = it % 8
        trial = base.copy()
        if mode == 0:
            trial = base + 0.010 * rng.standard_normal((n, d))
            trial /= _np.linalg.norm(trial, axis=1, keepdims=True)
            trial = _soft_descent(trial, 450, 12.0, 90.0, 0.010, 0.004)
        elif mode == 1:
            trial = base + 0.028 * rng.standard_normal((n, d))
            trial /= _np.linalg.norm(trial, axis=1, keepdims=True)
            trial = _soft_descent(trial, 550, 15.0, 110.0, 0.012, 0.004)
        elif mode == 2:
            # regional shake around worst point
            c = int(order_worst[int(rng.integers(0, 4))])
            Gc = (base @ base[c])
            Gc[c] = -10
            nn = _np.argsort(-Gc)[:11]
            region = list(set([c] + nn.tolist()))
            trial[region] += 0.14 * rng.standard_normal((len(region), d))
            trial /= _np.linalg.norm(trial, axis=1, keepdims=True)
            trial = _soft_descent(trial, 550, 12.0, 110.0, 0.011, 0.004)
        elif mode == 3:
            # overload hub shake
            c = int(order_over[int(rng.integers(0, 6))])
            Gc = (base @ base[c])
            Gc[c] = -10
            nn = _np.argsort(-Gc)[:9]
            region = list(set([c] + nn.tolist()))
            trial[region] += 0.22 * rng.standard_normal((len(region), d))
            trial /= _np.linalg.norm(trial, axis=1, keepdims=True)
            trial = _soft_descent(trial, 600, 14.0, 120.0, 0.011, 0.003)
        elif mode == 4:
            # single kill + best hole
            c = int(order_worst[int(rng.integers(0, 6))])
            mask = _np.ones(n, dtype=bool)
            mask[c] = False
            others = base[mask]
            trial[c] = best_hole(others, 2000)
            trial = _soft_descent(trial, 600, 12.0, 120.0, 0.010, 0.003)
        elif mode == 5:
            # double kill
            k1 = int(order_worst[int(rng.integers(0, 8))])
            k2 = int(order_over[int(rng.integers(0, 8))])
            if k2 == k1:
                k2 = (k2 + 13) % n
            for c in (k1, k2):
                mask = _np.ones(n, dtype=bool)
                mask[c] = False
                others = trial[mask]
                trial[c] = best_hole(others, 1500)
            trial = _soft_descent(trial, 600, 14.0, 120.0, 0.010, 0.003)
        elif mode == 6:
            A = _np.eye(d) + 0.10 * rng.standard_normal((d, d))
            trial = base @ A
            trial /= _np.linalg.norm(trial, axis=1, keepdims=True)
            trial = _soft_descent(trial, 450, 15.0, 100.0, 0.010, 0.004)
        else:
            k = 10
            idx = rng.choice(n, size=k, replace=False)
            trial[idx] += 0.24 * rng.standard_normal((k, d))
            trial /= _np.linalg.norm(trial, axis=1, keepdims=True)
            trial = _soft_descent(trial, 500, 12.0, 100.0, 0.012, 0.004)

        v_soft = _max_cos(trial)
        # quick filter: skip LP if far worse than both cur and best
        if v_soft > best_val + 0.018 and v_soft > cur_val + 0.012:
            # still allow wandering occasionally without LP
            if rng.random() < 0.05:
                # accept soft as cur if not too bad (topology diversity)
                frac = (_time.time() - t0) / max(EXPLORE_END - t0, 1.0)
                Temp = TEMP0 * (1.0 - 0.7 * frac)
                if v_soft < cur_val or rng.random() < _np.exp((cur_val - v_soft) / max(Temp, 1e-4)):
                    cur = trial / _np.linalg.norm(trial, axis=1, keepdims=True)
                    cur_val = v_soft
            continue
        if _time.time() >= EXPLORE_END:
            if v_soft < best_val:
                best = trial / _np.linalg.norm(trial, axis=1, keepdims=True)
                best_val = v_soft
                cur = best.copy()
                cur_val = best_val
            continue
        short_deadline = min(_time.time() + 1.4, EXPLORE_END + 1.0, HARD_END - 14.0)
        if short_deadline - _time.time() < 0.35:
            continue
        polished = _lp_polish(trial, short_deadline)
        v = _max_cos(polished)
        frac = (_time.time() - t0) / max(EXPLORE_END - t0, 1.0)
        Temp = TEMP0 * (1.0 - 0.7 * frac)
        if v < best_val - 1e-12:
            best = polished
            best_val = v
            cur = polished.copy()
            cur_val = v
        elif v < cur_val or rng.random() < _np.exp((cur_val - v) / max(Temp, 1e-4)):
            cur = polished.copy()
            cur_val = v

    remain = HARD_END - _time.time()
    if remain > 5.0:
        best = _lp_polish(best, _time.time() + remain - 1.5, max_pairs=2500)
    best = best / _np.linalg.norm(best, axis=1, keepdims=True)
    return best
