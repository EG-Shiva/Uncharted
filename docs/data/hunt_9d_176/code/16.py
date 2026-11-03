import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def smooth_quench(X, steps=400, a0=25.0, a1=280.0, lr0=0.025, lr1=0.002):
    n, d = X.shape
    for s in range(steps):
        frac = s / max(1, steps - 1)
        alpha = a0 * ((a1 / a0) ** frac)
        lr = lr0 * ((lr1 / lr0) ** frac)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        try:
            M = float(G.max())
        except Exception:
            break
        E = np.exp(alpha * (G - M))
        np.fill_diagonal(E, 0.0)
        Grad = E @ X
        proj = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - proj * X
        gnorm = float(np.linalg.norm(Gt) / np.sqrt(n))
        if not np.isfinite(gnorm) or gnorm < 1e-16:
            continue
        X = X - (lr / gnorm) * Gt
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[~np.isfinite(nr)] = 1.0
        nr[nr < 1e-12] = 1.0
        X = X / nr
    return X

def greedy_knockout_fill(base, idx, rng, ntrials=2048):
    n, d = base.shape
    cand = base.copy()
    mask = np.ones(n, dtype=bool)
    mask[idx] = False
    cur = base[mask].copy()
    # sequential greedy largest-hole insertion
    for j in idx:
        trials = rng.standard_normal((ntrials, d))
        tn = np.linalg.norm(trials, axis=1, keepdims=True)
        tn[tn == 0] = 1.0
        trials = trials / tn
        dots = trials @ cur.T
        mm = dots.max(axis=1)
        pick = int(np.argmin(mm))
        cand[j] = trials[pick]
        cur = np.vstack([cur, cand[j][None, :]])
    return cand

def lp_polish_deadline(x, deadline, r_init=0.004, cap=2500):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = max_cos(x)
    except Exception:
        return x
    radius = r_init
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
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
            idx_all = np.where(sel)[0]
            # keep largest cap
            part = np.argpartition(-c_all[idx_all], cap)[:cap]
            keep = np.zeros_like(sel)
            keep[idx_all[part]] = True
            sel = keep
            cnt = cap
        I = iu0[0][sel]
        J = iu0[1][sel]
        c_sel = c_all[sel]
        p = cnt
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
        obj = np.zeros(nv)
        obj[-1] = 1.0
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
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
        if time.time() >= deadline:
            break
    return x

def solve():
    t0 = time.time()
    HARD_END = t0 + 176.0
    LOOP_END = t0 + 150.0
    LP_END = t0 + 174.0
    rng = np.random.default_rng(20261103)
    n, d = 176, 9
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                nr = np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(nr > 1e-12):
                    cands.append(a / nr)
        except Exception:
            pass
    if not cands:
        X0 = rng.standard_normal((n, d))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        cands.append(X0)
    vals = [max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    best_X = cands[bi].copy()
    best_v = float(vals[bi])
    # quick initial quench of best to ensure local tightness
    it = 0
    k_cycle = [3, 6, 2, 8, 4, 5, 10, 3]
    sig_cycle = [0.04, 0.09, 0.06, 0.12, 0.03]
    while time.time() < LOOP_END:
        it += 1
        if time.time() > LOOP_END - 2:
            break
        # alternate move type
        if it % 2 == 0:
            # knockout worst points
            k = k_cycle[(it // 2) % len(k_cycle)]
            k = min(k, 12)
            try:
                G = best_X @ best_X.T
                np.fill_diagonal(G, -2.0)
                rowmax = G.max(axis=1)
                # contact overload count near top
                thr = best_v - 2e-3
                cnt = (G > thr).sum(axis=1)
                score = rowmax + 0.005 * cnt / max(1, cnt.max())
                idx = np.argpartition(-score, k)[:k]
                # add one random to diversify every 5th
                if it % 10 == 0:
                    r = int(rng.integers(0, n))
                    if r not in idx:
                        idx[-1] = r
                idx = idx[np.argsort(-score[idx])]
                cand = greedy_knockout_fill(best_X, idx, rng, ntrials=2048)
                cand = smooth_quench(cand, steps=420, a0=25.0, a1=300.0, lr0=0.025, lr1=0.002)
            except Exception:
                continue
        else:
            # global jitter + quench
            sig = sig_cycle[(it // 2) % len(sig_cycle)]
            try:
                cand = best_X + sig * rng.standard_normal((n, d))
                cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                cand = smooth_quench(cand, steps=320, a0=20.0, a1=250.0, lr0=0.02, lr1=0.002)
            except Exception:
                continue
        try:
            v = max_cos(cand)
        except Exception:
            continue
        if v < best_v and np.all(np.isfinite(cand)):
            best_X = cand
            best_v = v
        if time.time() >= LOOP_END:
            break
    # final LP squeeze on best
    try:
        if time.time() < LP_END - 1:
            best_X = lp_polish_deadline(best_X, LP_END, r_init=0.004, cap=2500)
    except Exception:
        pass
    best_X = best_X / np.linalg.norm(best_X, axis=1, keepdims=True)
    return best_X
