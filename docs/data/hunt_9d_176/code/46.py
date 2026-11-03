import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _lse_polish(X, steps, a0, a1, lr, deadline=None):
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(steps):
        if deadline is not None and time.time() > deadline:
            break
        frac = t / max(1, steps - 1)
        alpha = a0 * ((a1 / a0) ** frac)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = float(np.max(G))
        # avoid overflow: E in [0,1]
        E = np.exp(alpha * (G - M))
        np.fill_diagonal(E, 0.0)
        S = float(np.sum(E) * 0.5)
        if not np.isfinite(S) or S <= 0:
            break
        Grad = (E @ X) / S
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        m = b1 * m + (1 - b1) * Gt
        v = b2 * v + (1 - b2) * (Gt * Gt)
        mhat = m / (1 - b1 ** (t + 1))
        vhat = v / (1 - b2 ** (t + 1))
        step = mhat / (np.sqrt(vhat) + eps)
        np.clip(step, -3.0, 3.0, out=step)
        lrt = lr * (1.0 - 0.5 * frac)
        X = X - lrt * step
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[~np.isfinite(nr)] = 1.0
        nr[nr < 1e-12] = 1.0
        X = X / nr
    return X

def _refine_single(p, F, steps=200, alpha=200.0, lr=0.01, deadline=None):
    m = np.zeros(9, dtype=float)
    v = np.zeros(9, dtype=float)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(steps):
        if deadline is not None and time.time() > deadline:
            break
        dots = F @ p
        M = float(np.max(dots))
        E = np.exp(alpha * (dots - M))
        S = float(np.sum(E))
        if not np.isfinite(S) or S <= 0:
            break
        W = E / S
        Grad = W @ F
        Gt = Grad - float(np.dot(Grad, p)) * p
        m = b1 * m + (1 - b1) * Gt
        v = b2 * v + (1 - b2) * (Gt * Gt)
        mhat = m / (1 - b1 ** (t + 1))
        vhat = v / (1 - b2 ** (t + 1))
        step = mhat / (np.sqrt(vhat) + eps)
        step = np.clip(step, -3.0, 3.0)
        p = p - lr * step
        n = float(np.linalg.norm(p))
        if not np.isfinite(n) or n < 1e-12:
            break
        p = p / n
    return p

def _lp_polish_deadline(x, deadline, r_init=0.004, cap=2500):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = r_init
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        mval = float(c_all.max())
        thresh = mval - max(20.0 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > cap:
            idx = np.where(sel)[0]
            order = np.argpartition(-c_all[idx], cap)[:cap]
            keep = np.zeros_like(sel)
            keep[idx[order]] = True
            sel = keep
        I = iu0[0][sel]; J = iu0[1][sel]
        c_sel = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.empty((p, 2 * d + 1), dtype=int)
        cols[:, :d] = I[:, None] * d + np.arange(d)
        cols[:, d:2 * d] = J[:, None] * d + np.arange(d)
        cols[:, 2 * d] = n * d
        cols = cols.ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        if not np.all(np.isfinite(yn)) or np.any(yn < 1e-12):
            radius *= 0.5
            continue
        y = y / yn
        vv = _max_cos(y)
        if vv < best:
            x, best = y, vv
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    LOOP_END = t0 + 140.0
    HARD_END = t0 + 176.0
    THRESH = 0.4611825753
    n, d = 176, 9
    # load best
    base = None
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=float)
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                nr = np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(nr > 1e-12):
                    base = a / nr
                    break
        except Exception:
            continue
    if base is None:
        rng0 = np.random.default_rng(0)
        base = rng0.standard_normal((n, d))
        base /= np.linalg.norm(base, axis=1, keepdims=True)
    best_X = base.copy()
    best_v = _max_cos(best_X)
    # contacts for hub
    try:
        G0 = base @ base.T
        np.fill_diagonal(G0, -2.0)
        m0 = float(G0.max())
        C0 = (G0 > m0 - 1e-9).sum(axis=1)
        hub = int(np.argmax(C0))
        hub_order = np.argsort(-G0[hub])
        hub_order = hub_order[hub_order != hub]
    except Exception:
        hub = 0
        hub_order = np.arange(n)
    configs = [
        (16, 'hub', 101), (12, 'hub', 102), (8, 'hub', 103),
        (16, 'rand', 104), (12, 'rand', 105), (8, 'rand', 106),
        (16, 'hub', 107), (12, 'hub', 108), (8, 'rand', 109),
    ]
    for (k, mode, seed) in configs:
        if time.time() > LOOP_END:
            break
        rng = np.random.default_rng(seed)
        try:
            if mode == 'hub':
                # hub + part of neighbourhood
                take = hub_order[:max(0, k - 1)].tolist()
                dels = [hub] + take
                # if k larger than neighbourhood, fill randomly
                if len(dels) < k:
                    extra = rng.choice(n, size=k - len(dels), replace=False)
                    for e in extra:
                        if int(e) not in dels:
                            dels.append(int(e))
                    while len(dels) < k:
                        e = int(rng.integers(0, n))
                        if e not in dels:
                            dels.append(e)
                dels = np.array(dels[:k], dtype=int)
            else:
                dels = rng.choice(n, size=k, replace=False)
            mask = np.ones(n, dtype=bool)
            mask[dels] = False
            cur = base[mask].copy()
            # Adam-polish remainder 800 steps alpha 50->300 lr 0.003
            cur = _lse_polish(cur, 800, 50.0, 300.0, 0.003, deadline=LOOP_END)
            if time.time() > LOOP_END:
                continue
            # quick check: remaining should be decent; still proceed
            ok = True
            for _ii in range(k):
                if time.time() > LOOP_END:
                    ok = False
                    break
                # best of >=50000 candidates
                NC = 50000
                # batches to bound memory
                best_mm = np.inf
                best_cand = None
                # 5 batches of 10000
                for b in range(5):
                    if time.time() > LOOP_END:
                        ok = False
                        break
                    trials = rng.standard_normal((10000, d))
                    trials /= np.linalg.norm(trials, axis=1, keepdims=True)
                    dots = trials @ cur.T
                    mm = dots.max(axis=1)
                    bi = int(np.argmin(mm))
                    if float(mm[bi]) < best_mm:
                        best_mm = float(mm[bi])
                        best_cand = trials[bi].copy()
                if best_cand is None or not np.all(np.isfinite(best_cand)):
                    ok = False
                    break
                p = best_cand / np.linalg.norm(best_cand)
                # local refine alone 200 tangent-Adam steps
                p = _refine_single(p, cur, steps=200, alpha=200.0, lr=0.01, deadline=LOOP_END)
                cur = np.vstack([cur, p[None, :]])
                # only 300 global LogSumExp steps per insertion
                cur = _lse_polish(cur, 300, 50.0, 300.0, 0.003, deadline=LOOP_END)
            if not ok:
                continue
            if cur.shape != (n, d):
                continue
            cur = cur / np.linalg.norm(cur, axis=1, keepdims=True)
            v = _max_cos(cur)
            if v < best_v and np.all(np.isfinite(cur)):
                best_v = v
                best_X = cur.copy()
                # keep base for further deletions? use improved as base for next trials
                # but spec says from best.npy; keep original base to stay cheap.
                # update base to best to intensify
                base = best_X.copy()
                try:
                    G0 = base @ base.T
                    np.fill_diagonal(G0, -2.0)
                    m0 = float(G0.max())
                    C0 = (G0 > m0 - 1e-9).sum(axis=1)
                    hub = int(np.argmax(C0))
                    hub_order = np.argsort(-G0[hub])
                    hub_order = hub_order[hub_order != hub]
                except Exception:
                    pass
                if best_v < 0.459:
                    break
        except Exception:
            continue
    # return loaded best immediately if no completed sequence beats threshold
    if best_v >= THRESH:
        try:
            orig = np.asarray(np.load("best.npy"), dtype=float)
            orig /= np.linalg.norm(orig, axis=1, keepdims=True)
            return orig
        except Exception:
            return best_X / np.linalg.norm(best_X, axis=1, keepdims=True)
    # at most one 20s deadline-checked LP polish
    if time.time() < HARD_END - 2.0:
        dl = min(time.time() + 20.0, HARD_END)
        try:
            best_X = _lp_polish_deadline(best_X, dl)
        except Exception:
            pass
    best_X = best_X / np.linalg.norm(best_X, axis=1, keepdims=True)
    return best_X
