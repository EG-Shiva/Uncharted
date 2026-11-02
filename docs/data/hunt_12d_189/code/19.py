import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _contacts(x, thr=1e-9):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    C = (G > m - thr)
    cnt = np.sum(C, axis=1)
    ptmax = np.max(G, axis=1)
    return m, cnt, ptmax

def _relocate_holes(x, loose, rng, M=8000):
    # sequential deepest-hole reinsertion w.r.t. fixed points
    n, d = x.shape
    loose = list(loose)
    if len(loose) == 0:
        return x
    mask = np.zeros(n, dtype=bool)
    mask[loose] = True
    for idx in loose:
        R = rng.standard_normal((M, d))
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        D = R @ x.T
        # ignore all moving points (use fixed set only)
        # set masked cols to -2, but keep at least fixed cols
        D[:, mask] = -2.0
        vals = np.max(D, axis=1)
        j = int(np.argmin(vals))
        x[idx] = R[j]
        # after placing, this point becomes fixed for next ones
        mask[idx] = False
    return x

def _anneal(x, steps, a0, a1, l0, l1):
    n, d = x.shape
    for s in range(steps):
        f = s / (steps - 1) if steps > 1 else 0.0
        alpha = a0 * (a1 / a0) ** f if a0 > 0 else a1
        lr = l0 + (l1 - l0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        row = np.sum(W, axis=1, keepdims=True) + 1e-30
        S = (W @ x) / row
        dot = np.sum(S * x, axis=1, keepdims=True)
        Gt = S - dot * x
        # limit huge steps
        nr = np.linalg.norm(Gt, axis=1, keepdims=True)
        Gt = Gt * np.minimum(1.0, 1.0 / np.maximum(1.0, nr))
        x = x - lr * Gt
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _lp_polish(x, deadline, radius_init=8e-4):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = radius_init
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(np.max(c))
        sel = c > m - max(15.0 * radius, 5e-10)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > 3000:
            # subsample to keep LP cheap: keep tightest 3000
            order = np.argsort(c[sel])[-3000:]
            I, J = I[order], J[order]
            p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel][-p:] if p != np.sum(sel) else -c[sel],
                          A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.4, 0.03)
        else:
            radius *= 0.5
        if time.time() >= deadline:
            break
    return x, best

def solve():
    t0 = time.time()
    deadline = t0 + 165.0
    rng = np.random.default_rng(7)
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy")
    except Exception:
        xr = None
    cands = []
    for c in (xb, xr):
        if c is not None:
            try:
                a = np.array(c, dtype=np.float64)
                if a.shape == (189, 12):
                    a = a / np.linalg.norm(a, axis=1, keepdims=True)
                    cands.append((_max_cos(a), a))
            except Exception:
                pass
    if not cands:
        a = rng.standard_normal((189, 12))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append((_max_cos(a), a))
    cands.sort(key=lambda t: t[0])
    best_val, best_x = cands[0][0], cands[0][1].copy()
    n, d = 189, 12
    trial = 0
    while time.time() < deadline - 12.0 and trial < 40:
        time_left = deadline - 12.0 - time.time()
        if time_left < 2.0:
            break
        x = best_x.copy()
        m, cnt, ptmax = _contacts(x, thr=1e-9)
        # loose first: fewest contacts, then smallest ptmax
        order = np.lexsort((ptmax, cnt))
        mode = trial % 4
        if mode == 0:
            k = 3 + (trial % 3)  # 3..5
            loose = order[:k]
            x = _relocate_holes(x, loose, rng, M=6000)
            # light shake of overloaded hubs
            hubs = np.argsort(-cnt)[:3]
            x[hubs] += 0.006 * rng.standard_normal((len(hubs), d))
            x[hubs] /= np.linalg.norm(x[hubs], axis=1, keepdims=True)
            steps, a0, a1, l0, l1 = 350, 30.0, 220.0, 0.25, 0.02
            lp_budget = 1.8
        elif mode == 1:
            k = 2 + (trial % 2)
            loose = order[:k]
            x = _relocate_holes(x, loose, rng, M=8000)
            sel = rng.choice(n, size=10, replace=False)
            x[sel] += 0.012 * rng.standard_normal((len(sel), d))
            x[sel] /= np.linalg.norm(x[sel], axis=1, keepdims=True)
            steps, a0, a1, l0, l1 = 350, 25.0, 180.0, 0.28, 0.02
            lp_budget = 1.8
        elif mode == 2:
            sel = np.concatenate([np.argsort(-cnt)[:6], rng.choice(n, size=6, replace=False)])
            sel = np.unique(sel)
            x[sel] += 0.015 * rng.standard_normal((len(sel), d))
            x[sel] /= np.linalg.norm(x[sel], axis=1, keepdims=True)
            steps, a0, a1, l0, l1 = 300, 35.0, 250.0, 0.22, 0.015
            lp_budget = 1.5
        else:
            x += 0.004 * rng.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps, a0, a1, l0, l1 = 250, 40.0, 250.0, 0.18, 0.015
            lp_budget = 1.2
        # shrink work if little time
        if time_left < 20:
            steps = min(steps, 200)
            lp_budget = min(lp_budget, 1.0)
        x = _anneal(x, steps, a0, a1, l0, l1)
        v_quick = _max_cos(x)
        # skip LP if far worse (save time), unless mode 3 (global shake may need it)
        if v_quick > best_val + 0.015:
            trial += 1
            continue
        lp_dead = min(time.time() + lp_budget, deadline - 9.0)
        if lp_dead > time.time() + 0.4:
            x, v = _lp_polish(x, lp_dead)
        else:
            v = v_quick
        if v < best_val - 1e-12:
            best_val, best_x = v, x.copy()
        trial += 1
    # final squeeze
    if time.time() < deadline - 2.0:
        fd = min(deadline - 0.5, time.time() + 7.0)
        best_x, v2 = _lp_polish(best_x, fd, radius_init=4e-4)
        best_val = min(best_val, v2)
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
