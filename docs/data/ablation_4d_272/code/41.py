import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def repulse_gd(x, steps, a0, a1, s0, s1, mu=0.85):
    n, d = x.shape
    x = _norm(x.copy())
    vel = np.zeros_like(x)
    for it in range(steps):
        frac = it / max(steps - 1, 1)
        alpha = a0 * (a1 / a0) ** frac if a0 != a1 else a0
        smax = s0 * (s1 / s0) ** frac if s0 != s1 else s0
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp(np.clip(alpha * (G - m), -60, 0.0))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        vel = mu * vel + gt
        nrm = float(np.linalg.norm(vel, axis=1).max())
        if nrm > 1e-14:
            x = x - (smax / nrm) * vel
            x = _norm(x)
    return x

def lp_polish(x, seconds, r0=1e-3, cap=5000):
    x = _norm(x.copy())
    n, d = x.shape
    best = max_cos(x)
    radius = float(r0)
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        m = float(c_all.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > cap:
            idx = np.where(sel)[0]
            order = np.argpartition(-c_all[idx], cap - 1)[:cap]
            keep = np.zeros_like(sel, dtype=bool)
            keep[idx[order]] = True
            sel = keep
        c = c_all[sel]
        I = iu0[0][sel]
        J = iu0[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + ar[None, :],
                               J[:, None] * d + ar[None, :],
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = max_cos(y)
        if v < best - 1e-15:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def contact_stats(X, alpha=80.0):
    n = X.shape[0]
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    rowmax = G.max(axis=1)
    thresh = m - 0.002
    deg = np.count_nonzero(G > thresh, axis=1).astype(float)
    W = np.exp(np.clip(alpha * (G - m), -60, 0.0))
    np.fill_diagonal(W, 0.0)
    stress = W.sum(axis=1)
    return G, m, rowmax, deg, stress

def delete_indices(X, mode, k, rng):
    n = X.shape[0]
    G, m, rowmax, deg, stress = contact_stats(X)
    k = min(k, n - 10)
    if mode == 'degree':
        order = np.lexsort((-rowmax, -deg))
        return order[:k]
    elif mode == 'stress':
        order = np.lexsort((-rowmax, -stress))
        return order[:k]
    elif mode == 'rowmax':
        order = np.argsort(-rowmax, kind='stable')
        return order[:k]
    elif mode == 'random':
        return rng.choice(n, size=k, replace=False)
    elif mode == 'cluster':
        seed = int(np.argmax(stress))
        sims = G[seed].copy()
        # include seed itself: set to large
        sims[seed] = 2.0
        order = np.argsort(-sims, kind='stable')
        return order[:k]
    elif mode == 'cluster2':
        seed = int(np.argmax(deg))
        sims = G[seed].copy()
        sims[seed] = 2.0
        order = np.argsort(-sims, kind='stable')
        return order[:k]
    else:
        return rng.choice(n, size=k, replace=False)

def refine_lse(Y, c0, beta=60.0, steps=250, lr0=0.12, lr1=0.004, mu=0.55):
    c = c0 / (np.linalg.norm(c0) + 1e-15)
    vel = np.zeros(4)
    for it in range(steps):
        frac = it / max(steps - 1, 1)
        lr = lr0 * (lr1 / lr0) ** frac
        dots = Y @ c
        m = float(dots.max())
        ex = np.exp(np.clip(beta * (dots - m), -60, 0.0))
        s = float(ex.sum())
        if s <= 0:
            s = 1.0
        w = ex / s
        grad = w @ Y
        gt = grad - float(grad @ c) * c
        vel = mu * vel + gt
        vn = float(np.linalg.norm(vel))
        if vn < 1e-14:
            c = c - lr * gt
        else:
            c = c - (lr / max(1.0, vn * 8.0)) * vel * 8.0 * 0.12
            # simpler stable step: normalize direction
            # fallback to plain
        nrm = float(np.linalg.norm(c))
        if nrm == 0 or not np.isfinite(nrm):
            c = c0 / (np.linalg.norm(c0) + 1e-15)
            vel = np.zeros(4)
            continue
        c = c / nrm
    return c / np.linalg.norm(c)

def largest_hole(Y, rng, n_samples=50000, beta=60.0, n_refine=8):
    d = Y.shape[1]
    best_cands = None  # store smallest max
    # batched sampling to bound memory
    bs = 5000
    cand_pool = []
    cand_vals = []
    total = 0
    while total < n_samples:
        cur_bs = min(bs, n_samples - total)
        C = rng.standard_normal((cur_bs, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        S = C @ Y.T
        sm = S.max(axis=1)
        # keep best ~ min(20, cur_bs) per batch
        kkeep = min(20, cur_bs)
        idx = np.argpartition(sm, kkeep - 1)[:kkeep]
        for ii in idx:
            cand_pool.append(C[ii].copy())
            cand_vals.append(float(sm[ii]))
        total += cur_bs
    cand_pool = np.array(cand_pool)
    cand_vals = np.array(cand_vals)
    order = np.argsort(cand_vals, kind='stable')[:n_refine]
    best_c = None
    best_v = 2.0
    for oi in order:
        c0 = cand_pool[oi]
        cr = refine_lse(Y, c0, beta=beta, steps=220)
        dots = Y @ cr
        v = float(dots.max())
        # also LSE value for tie-break
        m = v
        lse = m + (1.0 / beta) * float(np.log(np.sum(np.exp(np.clip(beta * (dots - m), -60, 0.0))) + 1e-300))
        score = v + 1e-6 * lse
        if score < best_v:
            best_v = score
            best_c = cr.copy()
    if best_c is None:
        best_c = cand_pool[order[0]] / np.linalg.norm(cand_pool[order[0]])
    return best_c / np.linalg.norm(best_c)

def single_point_polish(cur, idx, steps=120, alpha=90.0, s=0.05):
    # freeze others, optimize cur[idx] only with softmax repulsion
    others = np.delete(cur, idx, axis=0)
    c = cur[idx].copy() / np.linalg.norm(cur[idx])
    vel = np.zeros(4)
    for it in range(steps):
        frac = it / max(steps - 1, 1)
        lr = s * (0.15) ** frac
        dots = others @ c
        m = float(dots.max())
        w = np.exp(np.clip(alpha * (dots - m), -60, 0.0))
        w /= (float(w.sum()) + 1e-15)
        grad = w @ others
        gt = grad - float(grad @ c) * c
        vel = 0.5 * vel + gt
        vn = float(np.linalg.norm(vel))
        if vn < 1e-14:
            break
        c = c - (lr / vn) * vel * vn / max(vn, 1e-12) * vn
        # stable: step normalized
        c = c - 0.0  # placeholder
        # redo simple step for stability:
        # Use plain gradient step
        # recompute quickly: we already have gt; use it
        # To avoid drift from vel scaling above, do direct:
        c = cur[idx]  # no-op guard to keep code simple
        break
    # Actually do clean direct GD without momentum confusion:
    c = cur[idx].copy() / np.linalg.norm(cur[idx])
    for it in range(steps):
        dots = others @ c
        m = float(dots.max())
        w = np.exp(np.clip(alpha * (dots - m), -60, 0.0))
        w /= (float(w.sum()) + 1e-15)
        grad = w @ others
        gt = grad - float(grad @ c) * c
        nn = float(np.linalg.norm(gt))
        if nn < 1e-14:
            break
        frac = it / max(steps - 1, 1)
        lr = s * (0.1) ** frac
        c = c - (lr / nn) * gt * min(1.0, nn * 5.0 + 0.2)
        c /= np.linalg.norm(c)
    cur[idx] = c / np.linalg.norm(c)
    return cur

def solve():
    t0 = time.time()
    BUDGET = 170.0
    T_END = t0 + BUDGET
    rng = np.random.default_rng(20261031)
    n, d = 272, 4
    cands = []
    try:
        bx = np.load("best.npy")
        bx = np.asarray(bx, dtype=np.float64)
        if bx.shape == (n, d):
            cands.append(_norm(bx))
    except Exception:
        pass
    try:
        rx = np.load("record.npy")
        rx = np.asarray(rx, dtype=np.float64)
        if rx.shape == (n, d):
            cands.append(_norm(rx))
    except Exception:
        pass
    if not cands:
        cands.append(_norm(rng.standard_normal((n, d))))
    vals = [(max_cos(c), c) for c in cands]
    vals.sort(key=lambda t: t[0])
    best_v, best_x = vals[0][0], vals[0][1].copy()
    pool = [(best_v, best_x.copy())]

    # short LP lock of incumbent if time allows
    if time.time() < T_END - 120.0:
        try:
            bx2 = lp_polish(best_x, seconds=3.0)
            bv2 = max_cos(bx2)
            pool.append((bv2, bx2.copy()))
            pool.sort(key=lambda t: t[0])
            pool = pool[:4]
            if bv2 < best_v - 1e-15:
                best_v, best_x = bv2, bx2.copy()
        except Exception:
            pass

    configs = [('degree', 32), ('cluster', 32), ('random', 32),
               ('stress', 28), ('degree', 40), ('cluster2', 24)]
    for cfg_i, (mode, k) in enumerate(configs):
        remain = T_END - time.time()
        if remain < 30.0:
            break
        base = best_x.copy()
        try:
            del_idx = delete_indices(base, mode, k, rng)
        except Exception:
            continue
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[np.asarray(del_idx, dtype=int)] = False
        cur = base[keep_mask].copy()
        # anneal remaining code hard until well below record
        try:
            cur = repulse_gd(cur, steps=650, a0=14.0, a1=160.0, s0=0.010, s1=0.0009, mu=0.85)
        except Exception:
            continue
        if time.time() > T_END - 25.0:
            break
        try:
            cur = lp_polish(cur, seconds=1.5, r0=2e-3)
        except Exception:
            pass
        # regrow one by one
        broke = False
        for t in range(k):
            if time.time() > T_END - 22.0:
                broke = True
                break
            try:
                hole = largest_hole(cur, rng, n_samples=50000, beta=60.0, n_refine=8)
            except Exception:
                # fallback random hole
                hole = rng.standard_normal(4)
                hole /= np.linalg.norm(hole)
            cur = np.vstack([cur, hole.reshape(1, 4)])
            cur = _norm(cur)
            # freeze older points briefly: polish only newest point
            try:
                cur = single_point_polish(cur, cur.shape[0] - 1, steps=100, alpha=90.0, s=0.06)
                cur = _norm(cur)
            except Exception:
                pass
            if ((t + 1) % 6 == 0) or (t == k - 1):
                steps_mid = 450
                if remain < 60:
                    steps_mid = 300
                try:
                    cur = repulse_gd(cur, steps=steps_mid, a0=20.0, a1=150.0, s0=0.006, s1=0.0008, mu=0.85)
                except Exception:
                    pass
        if broke:
            # incomplete regrow: skip evaluation unless full size
            if cur.shape[0] != n:
                continue
        if cur.shape[0] != n:
            continue
        try:
            cur = repulse_gd(cur, steps=500, a0=25.0, a1=180.0, s0=0.005, s1=0.0006, mu=0.86)
        except Exception:
            pass
        if time.time() < T_END - 12.0:
            try:
                cur = lp_polish(cur, seconds=2.0, r0=1.5e-3)
            except Exception:
                pass
        try:
            v = max_cos(cur)
        except Exception:
            continue
        pool.append((v, cur.copy()))
        pool.sort(key=lambda t: t[0])
        pool = pool[:5]
        if v < best_v - 1e-15:
            best_v, best_x = v, cur.copy()

    # final intensive polish of best with remaining time
    remain = T_END - time.time()
    if remain > 4.0:
        # alternate anneal + LP
        try:
            cand = best_x.copy()
            # light perturbation ensemble + polish to squeeze last bits
            for _ in range(3):
                if T_END - time.time() < 5.0:
                    break
                cand = repulse_gd(cand, steps=350, a0=30.0, a1=220.0, s0=0.003, s1=0.0004, mu=0.88)
                cand = lp_polish(cand, seconds=min(3.0, max(1.0, (T_END - time.time()) * 0.3)))
                v = max_cos(cand)
                if v < best_v - 1e-15:
                    best_v, best_x = v, cand.copy()
                else:
                    # keep pool diversity: small noise restart from best
                    if T_END - time.time() > 8.0:
                        nn = best_x + rng.standard_normal(best_x.shape) * 0.002
                        cand = _norm(nn)
                    else:
                        break
        except Exception:
            pass
    best_x = _norm(best_x)
    return best_x
