import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _quench_fast(X, steps, d0, d1, lr0, lr1):
    n, d = X.shape
    for s in range(steps):
        f = s / max(1, steps - 1)
        delta = d0 * (d1 / d0) ** f if d0 != d1 else d0
        lr = lr0 * (lr1 / lr0) ** f if lr0 != lr1 else lr0
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        thr = m - delta
        W = G - thr
        W[W < 0] = 0.0
        W = W * W
        np.fill_diagonal(W, 0.0)
        # skip if degenerate
        if W.max() <= 0:
            continue
        Grad = W @ X
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        gn = float(np.linalg.norm(Gt, axis=1).mean()) + 1e-12
        X = X - (lr / gn) * Gt
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X = X / nr
    return X

def _greedy_push(X, steps, e0, e1):
    n = X.shape[0]
    for s in range(steps):
        f = s / max(1, steps - 1)
        eps = e0 * (e1 / e0) ** f if e0 != e1 else e0
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        k = int(np.argmax(G))
        i = k // n
        j = k % n
        g = float(G[i, j])
        xi = X[i].copy()
        xj = X[j].copy()
        ti = xj - g * xi
        tj = xi - g * xj
        ni = float(np.linalg.norm(ti)) + 1e-12
        nj = float(np.linalg.norm(tj)) + 1e-12
        X[i] = xi - eps * ti / ni
        X[j] = xj - eps * tj / nj
        X[i] /= np.linalg.norm(X[i]) + 1e-16
        X[j] /= np.linalg.norm(X[j]) + 1e-16
    return X

def _lp_polish(x, seconds, t_deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 8e-4
    start = time.time()
    iu = np.triu_indices(n, 1)
    it = 0
    while time.time() - start < seconds and time.time() < t_deadline and radius > 1e-13:
        it += 1
        if it > 35:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        window = max(20.0 * radius, 5e-6)
        sel = c > m - window
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 700:
            idx = np.argpartition(-c, 700)[:700]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx] = True
            sel = sel & mask
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv)
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    T_END = t0 + 172.0
    HOP_END = t0 + 118.0
    rng = np.random.default_rng(12345)
    n, d = 58, 22
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            c = np.load(fn).astype(float)
            if c.shape != (58, 22):
                continue
            c = c / np.linalg.norm(c, axis=1, keepdims=True)
            v = _max_cos(c)
            if v < best_val:
                best_val = v
                best = c.copy()
        except Exception:
            continue
    if best is None:
        best = rng.standard_normal((n, d))
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    base = best.copy()
    base_val = float(best_val)

    elite = []  # list of (val, X)
    # include base as elite fallback
    elite.append((base_val, base.copy()))

    trial = 0
    # quench presets (steps, d0, d1, lr0, lr1)
    while time.time() < HOP_END and trial < 400:
        trial += 1
        remain = HOP_END - time.time()
        if remain < 0.5:
            break
        y = base.copy()
        mode = trial % 6
        # contact degrees of base for targeting
        try:
            G0 = base @ base.T
            np.fill_diagonal(G0, -2.0)
            m0 = float(G0.max())
            deg = (G0 > m0 - 0.004).sum(axis=1)
            order = np.argsort(-deg, kind="stable")
        except Exception:
            order = np.arange(n)
        if mode == 0:
            k = 4 + (trial % 3)  # 4..6
            # top hubs + random fill
            nh = min(k - 1, 3)
            victims = list(order[:nh])
            pool = [i for i in range(n) if i not in victims]
            extra = rng.choice(pool, size=k - len(victims), replace=False)
            victims = np.array(victims + list(extra))
            y[victims] = rng.standard_normal((k, d))
            y[victims] /= np.linalg.norm(y[victims], axis=1, keepdims=True)
            y += 0.006 * rng.standard_normal(y.shape)
        elif mode == 1:
            k = 3 + (trial % 3)  # 3..5 random
            victims = rng.choice(n, size=k, replace=False)
            y[victims] = rng.standard_normal((k, d))
            y[victims] /= np.linalg.norm(y[victims], axis=1, keepdims=True)
            y += 0.008 * rng.standard_normal(y.shape)
        elif mode == 2:
            # worst pair + hubs
            try:
                kk = int(np.argmax(G0))
                i = kk // n; j = kk % n
            except Exception:
                i, j = 0, 1
            victims = [int(i), int(j)] + [int(order[0])]
            victims = np.array(list(dict.fromkeys(victims)))
            y[victims] = rng.standard_normal((len(victims), d))
            y[victims] /= np.linalg.norm(y[victims], axis=1, keepdims=True)
            y += 0.005 * rng.standard_normal(y.shape)
        elif mode == 3:
            k = 6 + (trial % 3)  # 6..8 subset shake (no full reinit)
            idx = rng.choice(n, size=k, replace=False)
            y[idx] += 0.09 * rng.standard_normal((k, d))
            y += 0.004 * rng.standard_normal(y.shape)
        elif mode == 4:
            # global small shake + one hole reinit
            victim = int(order[trial % 6])
            y[victim] = rng.standard_normal(d)
            y[victim] /= np.linalg.norm(y[victim])
            y += 0.012 * rng.standard_normal(y.shape)
        else:
            # random subspace twist of a few points
            k = 5
            idx = rng.choice(n, size=k, replace=False)
            y[idx] = rng.standard_normal((k, d))
            y[idx] /= np.linalg.norm(y[idx], axis=1, keepdims=True)
            y += 0.010 * rng.standard_normal(y.shape)
        y /= np.linalg.norm(y, axis=1, keepdims=True)

        # choose quench length adaptively
        if mode in (3,):
            steps = 220
            d0, d1, lr0, lr1 = 0.05, 0.004, 0.030, 0.004
        else:
            steps = 260
            d0, d1, lr0, lr1 = 0.04, 0.003, 0.028, 0.003
            # vary slightly
            if trial % 2 == 0:
                steps = 200
        try:
            y = _quench_fast(y, steps, d0, d1, lr0, lr1)
            y = _greedy_push(y, 60, 0.003, 0.0003)
            v = _max_cos(y)
        except Exception:
            continue
        # keep elite: top 5 by value, require diversity not needed
        if len(elite) < 6:
            elite.append((v, y.copy()))
            elite.sort(key=lambda t: t[0])
        elif v < elite[-1][0]:
            elite[-1] = (v, y.copy())
            elite.sort(key=lambda t: t[0])
        # opportunistic base switch to a different basin even if slightly worse,
        # to create slack: accept if within 0.004 and sparser contacts
        if v < base_val + 0.001 and rng.random() < 0.10:
            # check sparsity: count contacts near max
            try:
                G = y @ y.T
                np.fill_diagonal(G, -2.0)
                mm = float(G.max())
                cc = int((G > mm - 1e-4).sum() // 2)
                Gb = base @ base.T
                np.fill_diagonal(Gb, -2.0)
                mb = float(Gb.max())
                cb = int((Gb > mb - 1e-4).sum() // 2)
                if cc < cb and v < base_val + 0.004:
                    base = y.copy()
                    base_val = v
            except Exception:
                pass

    # final: LP-polish elite (best few) with remaining budget
    elite.sort(key=lambda t: t[0])
    # keep unique top 4
    cands = elite[:4]
    # always include current overall best
    final_best = base.copy() if base_val < elite[0][0] else elite[0][1].copy()
    final_val = min(base_val, elite[0][0])
    # ensure the published best is also polished if it is better
    # (elite[0] should already be it)
    budget_total = max(4.0, min(40.0, T_END - time.time() - 2.0))
    per = budget_total / max(1, len(cands))
    for (v, Xc) in cands:
        if time.time() > T_END - 3.0:
            break
        try:
            Xp = _lp_polish(Xc.copy(), per, T_END - 1.0)
            vp = _max_cos(Xp)
            if vp < final_val:
                final_val = vp
                final_best = Xp.copy()
        except Exception:
            continue
    # last rescue: if nothing beat record-holder, polish it briefly
    final_best = final_best / np.linalg.norm(final_best, axis=1, keepdims=True)
    # sanitize
    final_best = np.where(np.isfinite(final_best), final_best, 0.0)
    for i in range(n):
        nr = float(np.linalg.norm(final_best[i]))
        if not np.isfinite(nr) or nr < 1e-12:
            vv = rng.standard_normal(d)
            vv /= np.linalg.norm(vv)
            final_best[i] = vv
    final_best = final_best / np.linalg.norm(final_best, axis=1, keepdims=True)
    return final_best
