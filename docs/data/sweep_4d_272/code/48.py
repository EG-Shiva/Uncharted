import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _count_above(x, thr=0.8975):
    g = x @ x.T
    iu = np.triu_indices(x.shape[0], 1)
    c = g[iu]
    return int((c > thr).sum()), float(c.max())

def _norm(x):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    x = np.where(np.isfinite(x), x, 0.0)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    bad = (n[:, 0] < 1e-12)
    if np.any(bad):
        x[bad] = np.array([1.0, 0.0, 0.0, 0.0])
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _lp_polish(x, seconds, deadline=1e18):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < deadline and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
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
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _refine_insert(F, Y, tau0=0.015, tau1=0.003, lr0=0.02, lr1=0.008, steps=260):
    Y = _norm(Y.copy())
    k2 = Y.shape[0]
    if k2 == 0:
        return Y
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        tau = tau0 * (tau1 / tau0) ** frac
        lr = lr0 + (lr1 - lr0) * frac
        D1 = Y @ F.T
        D2 = Y @ Y.T
        np.fill_diagonal(D2, -2.0)
        m = float(max(D1.max(), D2.max()))
        W1 = np.exp((D1 - m) / tau)
        W2 = np.exp((D2 - m) / tau)
        np.fill_diagonal(W2, 0.0)
        G1 = W1 @ F
        G2 = W2 @ Y
        grad = G1 + G2
        dot = (grad * Y).sum(axis=1, keepdims=True)
        gt = grad - dot * Y
        nr = np.linalg.norm(gt, axis=1, keepdims=True) + 1e-12
        Y = Y - lr * gt / nr
        Y = _norm(Y)
    return Y

def _global_anneal(x, steps=260, tau0=0.012, tau1=0.0012, lr0=0.006, lr1=0.0006):
    x = _norm(x.copy())
    for k in range(steps):
        frac = k / max(steps - 1, 1)
        tau = tau0 * (tau1 / tau0) ** frac
        lr = lr0 + (lr1 - lr0) * frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp((G - m) / tau)
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = (grad * x).sum(axis=1, keepdims=True)
        gt = grad - dot * x
        mn = float(np.mean(np.linalg.norm(gt, axis=1))) + 1e-12
        if not np.isfinite(mn):
            break
        x = x - (lr / mn) * gt
        x = _norm(x)
    return x

def solve():
    t0 = time.time()
    deadline = t0 + 174.0
    rng = np.random.default_rng(20261031)
    try:
        x0 = np.load("best.npy")
    except Exception:
        try:
            x0 = np.load("record.npy")
        except Exception:
            x0 = rng.standard_normal((272, 4))
    x0 = _norm(np.asarray(x0, dtype=float))
    if x0.shape != (272, 4):
        flat = x0.ravel()
        need = 272 * 4
        if flat.size < need:
            extra = rng.standard_normal((need - flat.size,))
            flat = np.concatenate([flat, extra])
        x0 = _norm(flat[:need].reshape(272, 4))
    n, d = 272, 4
    best = x0.copy()
    best_val = _max_cos(best)
    best_cnt, _ = _count_above(best, 0.8975)
    cur = best.copy()
    cur_val = best_val
    cur_cnt = best_cnt
    W_CNT = 2e-9
    cur_E = cur_val + W_CNT * cur_cnt
    best_E = cur_E

    k_cycle = [24, 36, 48, 36, 24, 48]
    tau_pairs = [(0.012, 0.0012), (0.018, 0.0018), (0.008, 0.0008)]
    cnt_thr_cycle = [0.8975, 0.8965, 0.8960]

    it = 0
    surg_end = t0 + 118.0
    while time.time() < surg_end and time.time() < deadline - 40:
        it += 1
        elapsed = time.time() - t0
        # SA temperature schedule
        T = 4e-6 * (0.05) ** (elapsed / max(surg_end - t0, 1.0))
        T = max(T, 1.5e-7)
        k = k_cycle[it % len(k_cycle)]
        tau0, tau1 = tau_pairs[it % len(tau_pairs)]
        cnt_thr = cnt_thr_cycle[it % len(cnt_thr_cycle)]

        # find rattlers: 3 points with smallest personal max
        G = cur @ cur.T
        np.fill_diagonal(G, -2.0)
        pmax = G.max(axis=1)
        ratt = np.argsort(pmax)[:3]

        u = rng.standard_normal((4,))
        u /= (np.linalg.norm(u) + 1e-12)
        dots = cur @ u
        order = np.argsort(-dots)
        topk = order[:k]
        del_set = np.union1d(topk, ratt)
        k2 = len(del_set)
        mask = np.ones(n, dtype=bool)
        mask[del_set] = False
        F = cur[mask]

        # pool of candidates
        M = 40000 if k2 >= 40 else 30000
        # keep time safe: reduce M if little time left
        if deadline - time.time() < 60:
            M = 20000
        pool = rng.standard_normal((M, 4))
        pool /= (np.linalg.norm(pool, axis=1, keepdims=True) + 1e-12)

        # greedy farthest-point: maximise min distance == minimise max cosine
        try:
            s_best = (pool @ F.T).max(axis=1)
        except Exception:
            continue
        chosen = np.zeros((k2, 4))
        for t in range(k2):
            idx = int(np.argmin(s_best))
            chosen[t] = pool[idx]
            s_best[idx] = 2.0
            # update with new point
            upd = pool @ chosen[t]
            # s_best = max(s_best, upd) but keep chosen slot at 2.0
            np.maximum(s_best, upd, out=s_best)
            s_best[idx] = 2.0
        # local tangential optimisation of inserted only
        try:
            Y = _refine_insert(F, chosen, steps=220)
        except Exception:
            Y = chosen

        x_new = cur.copy()
        x_new[del_set] = _norm(Y)

        # short global anneal (unfrozen)
        try:
            # vary steps slightly with k
            gs = 220 if k2 < 40 else 180
            x_new = _global_anneal(x_new, steps=gs, tau0=tau0, tau1=tau1, lr0=0.006, lr1=0.0006)
        except Exception:
            pass

        # short LP polish
        try:
            x_new = _lp_polish(x_new, min(2.5, max(0.5, (deadline - 35 - time.time()) / 8.0)), deadline - 32)
        except Exception:
            pass
        x_new = _norm(x_new)
        new_val = _max_cos(x_new)
        new_cnt, _ = _count_above(x_new, 0.8975)
        # also count at cycling cap for acceptance diversity
        new_cnt_c, _ = _count_above(x_new, cnt_thr)
        cur_cnt_c, _ = _count_above(cur, cnt_thr)
        new_E = new_val + W_CNT * new_cnt + 0.5e-9 * new_cnt_c
        cur_E_full = cur_val + W_CNT * cur_cnt + 0.5e-9 * cur_cnt_c
        # update best (primary max, secondary count)
        if (new_val < best_val - 1e-12) or (abs(new_val - best_val) <= 1e-12 and new_cnt < best_cnt):
            best = x_new.copy()
            best_val = new_val
            best_cnt = new_cnt
            best_E = new_E
        # SA accept
        if new_E < cur_E_full:
            cur, cur_val, cur_cnt, cur_E = x_new, new_val, new_cnt, new_E
        else:
            try:
                p = float(np.exp((cur_E_full - new_E) / T)) if T > 0 else 0.0
            except Exception:
                p = 0.0
            if rng.random() < p:
                cur, cur_val, cur_cnt, cur_E = x_new, new_val, new_cnt, new_E
            else:
                # occasionally reset to best if cur drifts bad
                if it % 12 == 0 and cur_val > best_val + 3e-4:
                    cur, cur_val, cur_cnt = best.copy(), best_val, best_cnt

    # final polish of best
    try:
        remain = deadline - time.time() - 1.5
        if remain > 2:
            best = _lp_polish(best, min(remain, 28.0), deadline - 1.0)
            best = _norm(best)
    except Exception:
        pass
    return _norm(best)
