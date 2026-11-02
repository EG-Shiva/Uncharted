import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds, init_radius=4e-3, max_pairs=2800):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = float(init_radius)
    start = time.time()
    iuI, iuJ = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start >= seconds:
            break
        G = x @ x.T
        c = G[iuI, iuJ]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > max_pairs:
            idx = np.where(sel)[0]
            cidx = c[idx]
            part = np.argpartition(cidx, -max_pairs)[-max_pairs:]
            sel = np.zeros_like(sel, dtype=bool)
            sel[idx[part]] = True
            cnt = max_pairs
        I = iuI[sel]; J = iuJ[sel]; csel = c[sel]
        p = cnt; nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        vals = np.empty(p * (2 * d + 1), dtype=np.float64)
        Rb = cols.reshape(p, 2 * d + 1); Vb = vals.reshape(p, 2 * d + 1)
        ar = np.arange(d)
        Rb[:, :d] = I[:, None] * d + ar
        Rb[:, d:2 * d] = J[:, None] * d + ar
        Rb[:, 2 * d] = n * d
        Vb[:, :d] = x[J]
        Vb[:, d:2 * d] = x[I]
        Vb[:, 2 * d] = -1.0
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        v = _max_cos(y)
        if v < best:
            x = y; best = v
            radius = min(radius * 1.5, 0.04)
        else:
            radius *= 0.5
    return x

def _smooth_relax(x, steps, a0, a1, lr, subset=None):
    n, d = x.shape
    for s in range(steps):
        alpha = a0 + (a1 - a0) * (s / max(1, steps - 1))
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        A = alpha * (G - m)
        W = np.zeros_like(G)
        mk = A > -9.0
        W[mk] = np.exp(A[mk])
        np.fill_diagonal(W, 0.0)
        F = W @ x
        proj = (F * x).sum(axis=1, keepdims=True)
        Ft = F - proj * x
        nrm = np.linalg.norm(Ft, axis=1)
        mx = nrm.max()
        if mx < 1e-18:
            continue
        step = Ft * (lr / mx)
        if subset is not None:
            S = np.zeros_like(step)
            S[subset] = step[subset]
            step = S
        x = x - step
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _per_point_worst(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def _contacts(x, m, tol=2e-7):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return np.sum(G > m - tol, axis=1)

def _greedy_reinsert(y, to_replace, rng, n_cand=15000):
    n, d = y.shape
    fixed_mask = np.ones(n, dtype=bool)
    fixed_mask[to_replace] = False
    Xf = y[fixed_mask]
    new_pts = []
    for k in to_replace:
        cand = rng.standard_normal(size=(n_cand, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        if new_pts:
            Xa = np.vstack([Xf] + new_pts)
        else:
            Xa = Xf
        dots = cand @ Xa.T
        worst = dots.max(axis=1)
        bi = int(np.argmin(worst))
        bp = cand[bi]
        new_pts.append(bp[None, :])
        y[k] = bp
    # tiny hill climb on replaced points
    for _ in range(2):
        for k in to_replace:
            cur = y[k].copy()
            others = np.delete(y, k, axis=0)
            cur_w = float((others @ cur).max())
            trial = rng.standard_normal(size=(500, d)) * 0.04 + cur[None, :]
            trial /= np.linalg.norm(trial, axis=1, keepdims=True)
            dw = (trial @ others.T).max(axis=1)
            bi = int(np.argmin(dw))
            if dw[bi] < cur_w:
                y[k] = trial[bi]
    return y

def solve():
    t0 = time.time()
    BUDGET = 168.0
    rng = np.random.default_rng(12345)
    x_best = None; best = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (323, 7):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(a)
                if v < best:
                    best = v; x_best = a.copy()
        except Exception:
            pass
    if x_best is None:
        x_best = rng.standard_normal(size=(323, 7))
        x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
        best = _max_cos(x_best)
    x_cur = x_best.copy()
    cur = best
    it = 0
    # quick smooth to diversify without LP first
    while time.time() - t0 < BUDGET - 12:
        it += 1
        remain = BUDGET - 12 - (time.time() - t0)
        if remain < 4:
            break
        n, d = x_cur.shape
        G = x_cur @ x_cur.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        worst = G.max(axis=1)
        cnt = np.sum(G > m - 2e-7, axis=1)
        rattlers = np.where(cnt == 0)[0].tolist()
        order_worst = np.argsort(-worst)
        order_cnt = np.argsort(-cnt)
        mode = it % 3
        if mode == 0:
            k = 6 + (it % 5)  # 6..10 cluster removal
            seed_idx = int(rng.integers(n))
            dist = G[seed_idx].copy()
            neigh = np.argsort(-dist)[:k]
            to_replace = sorted([int(v) for v in neigh])
        elif mode == 1:
            k = 7 + (it % 4)
            to_replace = []
            for r in rattlers:
                if len(to_replace) < k:
                    to_replace.append(int(r))
            for idx in order_worst:
                if len(to_replace) >= k: break
                if int(idx) not in to_replace:
                    to_replace.append(int(idx))
            # fill random
            while len(to_replace) < k:
                r = int(rng.integers(n))
                if r not in to_replace:
                    to_replace.append(r)
            to_replace = sorted(to_replace)
        else:
            k = 8 + (it % 5)
            to_replace = []
            for idx in order_cnt[:4]:
                if int(idx) not in to_replace:
                    to_replace.append(int(idx))
            while len(to_replace) < k:
                r = int(rng.integers(n))
                if r not in to_replace:
                    to_replace.append(r)
            to_replace = sorted(to_replace)
        y = x_cur.copy()
        # occasional extra shake of neighbours
        if it % 4 == 0:
            shake_idx = rng.choice(n, size=25, replace=False)
            y[shake_idx] += rng.standard_normal(size=(25, d)) * 0.03
            y[shake_idx] /= np.linalg.norm(y[shake_idx], axis=1, keepdims=True)
        y = _greedy_reinsert(y, to_replace, rng, n_cand=12000 if it % 2 == 0 else 18000)
        # relax: first only new points coarsely, then global anneal
        y = _smooth_relax(y, steps=120, a0=20.0, a1=120.0, lr=0.02, subset=np.array(to_replace))
        y = _smooth_relax(y, steps=220, a0=30.0, a1=250.0, lr=0.008, subset=None)
        v = _max_cos(y)
        # LP polish only if promising
        if v < best + 0.004 and remain > 8:
            # short polish
            y2 = _lp_polish(y, seconds=min(4.5, remain - 2), init_radius=3e-3)
            v2 = _max_cos(y2)
            y, v = y2, v2
        if v < best - 1e-12:
            x_best = y.copy(); best = v
            x_cur = y.copy(); cur = v
        else:
            # basin-hop acceptance: move to new basin if not too bad
            if v < cur + 0.008 and v < best + 0.006:
                x_cur = y.copy(); cur = v
            else:
                # restart from best with shake
                if rng.random() < 0.6:
                    x_cur = x_best.copy()
                    cur = best
                    sj = rng.choice(n, size=12, replace=False)
                    x_cur[sj] += rng.standard_normal(size=(12, d)) * 0.015
                    x_cur[sj] /= np.linalg.norm(x_cur[sj], axis=1, keepdims=True)
                    cur = _max_cos(x_cur)
    # final polish of best with remaining time
    rem = BUDGET - (time.time() - t0)
    if rem > 3:
        x_best = _lp_polish(x_best, seconds=max(2.0, rem - 1.0), init_radius=2e-3)
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
