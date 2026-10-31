import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    y = x / n
    # replace non-finite just in case
    bad = ~np.isfinite(y).all(axis=1)
    if np.any(bad):
        y[bad] = np.array([1.0, 0.0, 0.0, 0.0])
        y = y / np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
    return y

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _count_above(x, thr):
    G = x @ x.T
    iu = np.triu_indices(G.shape[0], 1)
    c = G[iu]
    return int(np.sum(c > thr))

def _per_point_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    pm = np.max(G, axis=1)
    return pm, float(np.max(pm))

def _smooth(x_init, steps, T0, T1, lr0, lr1, ms0, ms1, deadline=None):
    x = _norm(x_init.copy())
    n = x.shape[0]
    v = np.zeros_like(x)
    beta = 0.8
    logT0, logT1 = np.log(T0), np.log(T1)
    loglr0, loglr1 = np.log(lr0), np.log(lr1)
    logms0, logms1 = np.log(ms0), np.log(ms1)
    eye = np.eye(n)
    for s in range(steps):
        if deadline is not None and (s % 100 == 0) and time.time() > deadline:
            break
        frac = s / max(1, steps - 1)
        T = float(np.exp(logT0 + frac * (logT1 - logT0)))
        lr = float(np.exp(loglr0 + frac * (loglr1 - loglr0)))
        ms = float(np.exp(logms0 + frac * (logms1 - logms1)))
        G = x @ x.T
        m = float(np.max(G - 2.0 * eye))
        W = np.exp((G - m) / T)
        np.fill_diagonal(W, 0.0)
        S = float(np.sum(W) / 2.0)
        if not np.finite(S) or S <= 0:
            break
        Grad = (W @ x) / S
        dot = np.sum(Grad * x, axis=1, keepdims=True)
        Gt = Grad - dot * x
        v = beta * v + Gt
        step = lr * v
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-16
        scale = np.minimum(1.0, ms / sn)
        x = x - step * scale
        x = _norm(x)
    return x

def _lp(x, seconds, deadline=None):
    x = _norm(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while True:
        now = time.time()
        if now - start >= seconds:
            break
        if deadline is not None and now > deadline:
            break
        if radius < 1e-14:
            break
        Gfull = x @ x.T
        c = Gfull[iu0]
        m = float(np.max(c))
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu0[0][sel], iu0[1][sel]
        p = len(I)
        if p == 0:
            break
        try:
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            col_block = np.concatenate([I[:, None] * d + np.arange(d),
                                        J[:, None] * d + np.arange(d),
                                        np.full((p, 1), n * d)], axis=1).ravel()
            vals_block = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
            A_ub = coo_matrix((vals_block, (rows, col_block)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        vv = _max_cos(y)
        if vv < best:
            x, best = y, vv
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _refine_insert(frozen, Y, steps, T0, T1, lr0, lr1, rng=None):
    # local tangential optimisation of Y only, frozen fixed
    Y = _norm(Y.copy())
    for s in range(steps):
        frac = s / max(1, steps - 1)
        T = T0 * (T1 / T0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        Gyf = Y @ frozen.T
        Gyy = Y @ Y.T
        np.fill_diagonal(Gyy, -2.0)
        m = float(max(np.max(Gyf), np.max(Gyy)))
        Wf = np.exp((Gyf - m) / T)
        Wyy = np.exp((Gyy - m) / T)
        np.fill_diagonal(Wyy, 0.0)
        S = float(np.sum(Wf) + np.sum(Wyy) / 2.0)
        if not np.finite(S) or S <= 0:
            break
        Grad = (Wf @ frozen + Wyy @ Y) / max(S, 1e-16)
        dot = np.sum(Grad * Y, axis=1, keepdims=True)
        Gt = Grad - dot * Y
        # clip
        sn = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-16
        # normalize step by mean norm for stability
        mn = float(np.mean(sn)) + 1e-16
        step = lr * Gt / mn
        # per-row cap
        s2 = np.linalg.norm(step, axis=1, keepdims=True) + 1e-16
        cap = 0.05
        step = step * np.minimum(1.0, cap / s2)
        Y = _norm(Y - step)
    return Y

def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    SURG_END = t0 + 132.0
    rng = np.random.default_rng(20261031)
    n, d = 272, 4
    try:
        x0 = np.load("best.npy")
    except Exception:
        try:
            x0 = np.load("record.npy")
        except Exception:
            x0 = rng.standard_normal((n, d))
    x0 = np.asarray(x0, dtype=np.float64)
    if x0.shape != (n, d):
        if x0.size == n * d:
            x0 = x0.reshape(n, d)
        else:
            x0 = rng.standard_normal((n, d))
    x0 = _norm(x0)
    best = x0.copy()
    best_val = _max_cos(best)
    best_cnt = _count_above(best, 0.8975)
    cur = best.copy()
    cur_val = best_val
    cur_cnt = best_cnt

    k_cycle = [24, 36, 48]
    cap_cycle = [0.8975, 0.8965, 0.8980, 0.8970]
    # SA temperature schedule
    Tsa0, Tsa1 = 4e-4, 2e-6

    it = 0
    POOL = 40000
    while time.time() < SURG_END - 4.0:
        it += 1
        if time.time() > HARD_END - 14:
            break
        k = k_cycle[(it - 1) % len(k_cycle)]
        cap_thr = cap_cycle[(it - 1) % len(cap_cycle)]
        # progress fraction for SA temp
        frac_all = (time.time() - t0) / max(1e-9, (SURG_END - t0))
        Tsa = Tsa0 * (Tsa1 / Tsa0) ** min(1.0, max(0.0, frac_all))

        ybase = cur.copy()
        # random direction
        u = rng.standard_normal((d,))
        u = u / (np.linalg.norm(u) + 1e-16)
        dots = ybase @ u
        order = np.argsort(-dots)
        del_set = set(order[:k].tolist())
        # add 3 rattlers (lowest per-point max)
        pm, mglob = _per_point_max(ybase)
        ratt = np.argsort(pm)[:3]
        for r in ratt.tolist():
            del_set.add(int(r))
        del_idx = np.array(sorted(del_set), dtype=int)
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[del_idx] = False
        frozen = ybase[keep_mask]  # (n-kd, 4)
        kd = len(del_idx)

        # greedy rebuild from pool
        try:
            pool = rng.standard_normal((POOL, d))
            pool = _norm(pool)
        except Exception:
            pool = _norm(rng.standard_normal((POOL, d)))
        # max dot to frozen
        Df = pool @ frozen.T  # 40000 x ~224
        cur_max = np.max(Df, axis=1)
        sel_list = []
        # iterative farthest selection
        for _ in range(kd):
            bi = int(np.argmin(cur_max))
            b = pool[bi]
            sel_list.append(b)
            # update
            dprod = pool @ b
            # in-place maximum
            np.maximum(cur_max, dprod, out=cur_max)
            # prevent reselection (already 1.0)
        Y = np.stack(sel_list, axis=0)
        # local tangential optimisation of inserted only
        Y = _refine_insert(frozen, Y, steps=350, T0=0.012, T1=0.0025, lr0=0.25, lr1=0.03)
        Xnew = np.vstack([frozen, Y])
        Xnew = _norm(Xnew)
        # unfreeze: short global log-sum-exp anneal
        g_dead = min(HARD_END - 11, time.time() + 4.0)
        # vary anneal slightly by cap cycle index
        if (it % 2) == 0:
            Xnew = _smooth(Xnew, 500, 0.02, 0.003, 0.30, 0.025, 0.06, 0.005, deadline=g_dead)
        else:
            Xnew = _smooth(Xnew, 650, 0.028, 0.0025, 0.35, 0.02, 0.07, 0.004, deadline=g_dead)
        if time.time() > HARD_END - 11:
            v = _max_cos(_norm(Xnew))
            c = _count_above(_norm(Xnew), 0.8975)
            if v < best_val - 1e-12 or (abs(v - best_val) < 5e-12 and c < best_cnt):
                best, best_val, best_cnt = _norm(Xnew), v, c
            break
        lp_dead = min(HARD_END - 10, time.time() + 2.2)
        Xp = _lp(Xnew, seconds=1.8, deadline=lp_dead)
        new_val = _max_cos(Xp)
        new_cnt_cap = _count_above(Xp, cap_thr)
        new_cnt_ref = _count_above(Xp, 0.8975)
        # best fallback (strict max, then ref count)
        if new_val < best_val - 1e-12 or (abs(new_val - best_val) < 1e-11 and new_cnt_ref < best_cnt):
            best, best_val, best_cnt = Xp.copy(), new_val, new_cnt_ref
        # SA acceptance on energy = max + w*count(cap)
        w = 2e-6
        Ecur = cur_val + w * _count_above(cur, cap_thr)
        Enew = new_val + w * new_cnt_cap
        if Enew < Ecur:
            cur, cur_val, cur_cnt = Xp.copy(), new_val, new_cnt_ref
        else:
            # Metropolis
            prob = float(np.exp(-(Enew - Ecur) / max(Tsa, 1e-12)))
            if rng.random() < prob:
                cur, cur_val, cur_cnt = Xp.copy(), new_val, new_cnt_ref
            # else keep cur; occasionally restart from best if stuck
            if it % 15 == 0 and rng.random() < 0.3:
                # pull back toward best with small shake to diversify
                cur = _norm(best + 0.004 * rng.standard_normal(best.shape))
                cur_val = _max_cos(cur)
        # early exit if big win
        if best_val < 0.894:
            break
    # final polish of best with remaining time
    rem = HARD_END - time.time() - 0.8
    if rem > 2.0:
        best = _lp(best, seconds=min(rem - 0.3, 14.0), deadline=HARD_END - 0.4)
        best = _norm(best)
    else:
        best = _norm(best)
    return best
