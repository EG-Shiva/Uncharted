import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

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
        if p > 2500:
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

def _adam_smooth(x_init, steps, T0, T1, lr, deadline=None):
    x = x_init / np.linalg.norm(x_init, axis=1, keepdims=True)
    x = x.copy()
    n, d = x.shape
    m1 = np.zeros_like(x)
    m2 = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lT0 = np.log(T0)
    lT1 = np.log(T1)
    for s in range(steps):
        if deadline is not None and (s % 100 == 0) and time.time() > deadline:
            break
        frac = s / max(steps - 1, 1)
        T = float(np.exp(lT0 + (lT1 - lT0) * frac))
        lrs = lr * (1.0 - 0.5 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        mv = G.max()
        E = np.exp((G - mv) / T)
        np.fill_diagonal(E, 0.0)
        S = E.sum()
        if S == 0 or not np.isfinite(S):
            break
        Gt = (E @ x) / S
        dot = np.sum(Gt * x, axis=1, keepdims=True)
        Gt = Gt - dot * x
        m1 = b1 * m1 + (1 - b1) * Gt
        m2 = b2 * m2 + (1 - b2) * (Gt * Gt)
        bc1 = 1.0 - b1 ** (s + 1)
        bc2 = 1.0 - b2 ** (s + 1)
        mh = m1 / bc1
        vh = m2 / bc2
        step = lrs * mh / (np.sqrt(vh) + eps)
        # project to tangent
        sn = np.sum(step * x, axis=1, keepdims=True)
        step = step - sn * x
        # trust cap
        nm = np.linalg.norm(step, axis=1, keepdims=True)
        sc = np.minimum(1.0, 0.08 / (nm + 1e-16))
        step = step * sc
        x = x - step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _refine_hole(q, Xf, iters=220, T=0.005, lr=0.15):
    q = q / np.linalg.norm(q)
    for _ in range(iters):
        dots = Xf @ q
        mm = dots.max()
        w = np.exp((dots - mm) / T)
        ws = w.sum()
        if ws == 0 or not np.isfinite(ws):
            break
        g = (w @ Xf) / ws
        g = g - np.dot(g, q) * q
        gn = np.linalg.norm(g)
        if gn > 1e-12:
            # normalize-ish step
            st = lr * g / (gn + 0.5)
            if np.linalg.norm(st) > 0.1:
                st = st * (0.1 / np.linalg.norm(st))
            q = q - st
            q /= np.linalg.norm(q)
    return q

def solve():
    import time as _time
    import numpy as _np
    t0 = _time.time()
    HARD_END = t0 + 172.0
    EXPLORE_END = t0 + 143.0  # strict internal deadline ~165 with 20s final polish
    FINAL_END = t0 + 170.0
    rng = _np.random.default_rng(7)
    n, d = 102, 12
    try:
        best = _np.load("best.npy").astype(float)
    except Exception:
        try:
            best = _np.load("record.npy").astype(float)
        except Exception:
            best = rng.standard_normal((n, d))
    if best.shape != (n, d):
        try:
            rec = _np.load("record.npy").astype(float)
            if rec.shape == (n, d):
                best = rec
            else:
                tmp = rng.standard_normal((n, d))
                m0 = min(best.shape[0], n)
                d0 = min(best.shape[1], d)
                tmp[:m0, :d0] = best[:m0, :d0]
                best = tmp
        except Exception:
            tmp = rng.standard_normal((n, d))
            best = tmp
    best = best / _np.linalg.norm(best, axis=1, keepdims=True)
    best_val = _max_cos(best)
    incumbent = best.copy()
    inc_val = best_val
    # incumbent adjacency for diversity check
    def _adj_set(x, tol=2e-4):
        G = x @ x.T
        _np.fill_diagonal(G, -2.0)
        m = G.max()
        iu = _np.triu_indices(n, 1)
        sel = G[iu] > m - tol
        s = set(zip(iu[0][sel].tolist(), iu[1][sel].tolist()))
        return s, m
    inc_adj, _ = _adj_set(incumbent)
    pool = []  # diverse candidates

    def _deletion_set(x, seed_id, k):
        G = x @ x.T
        _np.fill_diagonal(G, -2.0)
        pm = G.max(axis=1)
        m = G.max()
        cnt = (G > m - 0.005).sum(axis=1)
        order_cnt = _np.argsort(-cnt)
        order_pm = _np.argsort(-pm)
        if seed_id == 0:
            # global top-k by combined rank
            rank = _np.zeros(n)
            for r, idx in enumerate(order_cnt):
                rank[idx] += r
            for r, idx in enumerate(order_pm):
                rank[idx] += r
            del_idx = _np.argsort(rank)[:k]
            return del_idx
        else:
            # clustered around s-th worst point
            s = (seed_id - 1) % 5
            c = int(order_pm[s % n] if s < len(order_pm) else order_cnt[s % n])
            row = (x @ x[c])
            row[c] = -10
            nn = _np.argsort(-row)[:k - 1]
            del_idx = _np.array([c] + nn.tolist(), dtype=int)
            return del_idx

    cycle = 0
    # plan ks 10-18 cycling
    ks = [12, 14, 11, 15, 10, 16, 13, 18]
    while _time.time() < EXPLORE_END and _time.time() < HARD_END - 25:
        k = ks[cycle % len(ks)]
        seed_id = cycle % 6  # 0=global, 1-5 clustered
        # use current best for deletion derivation (alternate incumbent/best)
        base = best if (cycle % 2 == 0) else incumbent
        try:
            del_idx = _deletion_set(base, seed_id, k)
        except Exception:
            del_idx = rng.choice(n, size=k, replace=False)
        keep_mask = _np.ones(n, dtype=bool)
        keep_mask[del_idx] = False
        X_rem = base[keep_mask].copy()
        # annealed LSE descent on remainder to gain margin below 0.267
        try:
            X_rem = _adam_smooth(X_rem, steps=700, T0=0.03, T1=0.004, lr=0.015,
                                 deadline=min(EXPLORE_END, HARD_END - 25))
        except Exception:
            pass
        # check time before regrowth
        if _time.time() > EXPLORE_END - 1.0:
            cycle += 1
            continue
        X_cur = X_rem.copy()
        # regrow one at a time
        failed = False
        for ins in range(k):
            if _time.time() > EXPLORE_END + 2.0 or _time.time() > HARD_END - 25:
                failed = True
                break
            nf = X_cur.shape[0]
            # global hole search >=50k samples
            try:
                cand = rng.standard_normal((50000, d))
                cand /= _np.linalg.norm(cand, axis=1, keepdims=True)
                dots = cand @ X_cur.T
                worst = dots.max(axis=1)
                # take top 5 holes, refine each, pick best vs X_cur
                top5 = _np.argpartition(worst, 5)[:5]
                best_q = None
                best_w = 1e9
                for ti in top5:
                    q0 = cand[int(ti)]
                    q1 = _refine_hole(q0, X_cur, iters=220, T=0.005, lr=0.15)
                    w1 = float((X_cur @ q1).max())
                    if w1 < best_w:
                        best_w = w1
                        best_q = q1
                q_new = best_q
            except Exception:
                q_new = rng.standard_normal(d)
                q_new /= _np.linalg.norm(q_new)
            X_cur = _np.vstack([X_cur, q_new.reshape(1, d)])
            # re-optimize all points 300-600 Adam steps (use 350)
            try:
                X_cur = _adam_smooth(X_cur, steps=350, T0=0.015, T1=0.0035, lr=0.012,
                                     deadline=min(EXPLORE_END + 4.0, HARD_END - 24))
            except Exception:
                pass
        if failed or X_cur.shape[0] != n:
            cycle += 1
            continue
        v_soft = _max_cos(X_cur)
        # allow worsening to 0.275-0.29 : still polish
        if v_soft > 0.295:
            # too bad, still try one short polish if time allows occasionally
            if (cycle % 3 != 0):
                cycle += 1
                continue
        short_dead = min(_time.time() + 2.0, EXPLORE_END + 5.0, HARD_END - 24.0)
        if short_dead - _time.time() < 0.4:
            cycle += 1
            continue
        try:
            polished = _lp_polish(X_cur, short_dead)
        except Exception:
            polished = X_cur
        try:
            v = _max_cos(polished)
        except Exception:
            cycle += 1
            continue
        # diversity: adjacency diff
        try:
            adj, _ = _adj_set(polished)
            inter = len(adj & inc_adj)
            union = len(adj | inc_adj)
            diff_frac = 1.0 - (inter / max(union, 1))
        except Exception:
            diff_frac = 1.0
        if v < best_val - 1e-12:
            best = polished / _np.linalg.norm(polished, axis=1, keepdims=True)
            best_val = v
            # keep pool of diverse near-best
            if diff_frac > 0.15:
                pool.append(polished.copy())
                if len(pool) > 3:
                    pool = pool[-3:]
        else:
            # keep materially different configs even if worse (within 0.02)
            if diff_frac > 0.30 and v < best_val + 0.02 and v < 0.29:
                pool.append(polished.copy())
                if len(pool) > 3:
                    pool = pool[-3:]
                # use diverse config as base occasionally to escape basin
                if (cycle % 4 == 3):
                    best = polished.copy()
                    best_val = v  # wander (allowed worsening)
            # also wander if stuck? keep incumbent best separately
        cycle += 1
        if _time.time() >= HARD_END - 24:
            break
        # if we beat record nicely, keep incumbent update for reference
        if best_val < inc_val - 1e-12:
            pass

    # final 20s LP polishing under strict deadline: polish best + pool
    # ensure best is at least as good as loaded best
    try:
        if best_val > inc_val:
            best = incumbent.copy()
            best_val = inc_val
    except Exception:
        pass
    candidates = [best] + pool
    # dedup: keep unique-ish, polish best first with most time
    remain = FINAL_END - _time.time()
    if remain > 3.0 and _time.time() < HARD_END - 2.0:
        # polish best with bulk of time
        try:
            t_best = min(_time.time() + max(remain - 4.0 * len(pool), 4.0), HARD_END - 1.5)
            best_p = _lp_polish(best, t_best)
            bv = _max_cos(best_p)
            if bv < best_val:
                best = best_p
                best_val = bv
        except Exception:
            pass
        for pi, pc in enumerate(pool):
            if _time.time() > HARD_END - 2.0 or _time.time() > FINAL_END:
                break
            try:
                t_d = min(_time.time() + 3.0, HARD_END - 1.5, FINAL_END)
                if t_d - _time.time() < 0.5:
                    break
                pp = _lp_polish(pc, t_d)
                pv = _max_cos(pp)
                if pv < best_val - 1e-12:
                    best = pp
                    best_val = pv
            except Exception:
                continue
    best = best / _np.linalg.norm(best, axis=1, keepdims=True)
    return best
