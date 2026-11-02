import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _worst(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1)

def _contacts(x, m, tol=2e-4):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return np.sum(g > m - tol, axis=1)

def _smooth_relax(x, steps, lr0, lr1, tau0, tau1, mu=0.85, subset=None):
    n, d = x.shape
    v = np.zeros_like(x)
    for s in range(steps):
        frac = s / max(1, steps - 1)
        tau = tau0 * (tau1 / tau0) ** frac if tau1 > 0 else tau0
        lr = lr0 * (lr1 / lr0) ** frac if steps > 1 else lr0
        G = x @ x.T
        np.fill_diagonal(G, -1.0)
        M = float(G.max())
        # exp, avoid overflow
        E = np.exp((G - M) / tau)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if not np.isfinite(S) or S <= 0:
            break
        W = E / S
        grad = W @ x
        # tangent projection
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        if subset is not None:
            mask = np.zeros(n, dtype=bool)
            mask[np.asarray(subset)] = True
            gt[~mask] = 0.0
        # clip
        gn = np.linalg.norm(gt, axis=1, keepdims=True) + 1e-12
        # avoid huge steps
        gt = gt / np.maximum(1.0, gn * 5.0)
        v = mu * v + gt
        x = x - lr * v
        # also direct small gradient component
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _hole_greedy(y, to_replace, rng, n_cand=20000):
    n, d = y.shape
    fixed_mask = np.ones(n, dtype=bool)
    fixed_mask[np.asarray(to_replace)] = False
    Xf = y[fixed_mask]
    for k in to_replace:
        cand = rng.standard_normal(size=(n_cand, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        dots = cand @ Xf.T
        worst = dots.max(axis=1)
        bi = int(np.argmin(worst))
        y[k] = cand[bi]
        # extend fixed set for next k (use new point)
        Xf = np.vstack([Xf, y[k][None, :]])
    return y

def _lp_polish(x, deadline, init_radius=3e-3, cap=2500):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = float(init_radius)
    iuI, iuJ = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iuI, iuJ]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > cap:
            idx = np.where(sel)[0]
            cidx = c[idx]
            part = np.argpartition(cidx, -cap)[-cap:]
            sel = np.zeros_like(sel, dtype=bool)
            sel[idx[part]] = True
            cnt = cap
        I = iuI[sel]; J = iuJ[sel]
        csel = c[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        vals = np.empty(p * (2 * d + 1), dtype=np.float64)
        Cb = cols.reshape(p, 2 * d + 1)
        Vb = vals.reshape(p, 2 * d + 1)
        ar = np.arange(d)
        Cb[:, :d] = I[:, None] * d + ar
        Cb[:, d:2 * d] = J[:, None] * d + ar
        Cb[:, 2 * d] = n * d
        Vb[:, :d] = x[J]
        Vb[:, d:2 * d] = x[I]
        Vb[:, 2 * d] = -1.0
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": 2.0, "presolve": True})
        except Exception:
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                              method="highs")
            except Exception:
                radius *= 0.5
                continue
        if getattr(res, "status", 1) != 0:
            radius *= 0.5
            continue
        y = x + np.asarray(res.x[:-1]).reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        vv = _max_cos(y)
        if vv < best:
            x = y; best = vv
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
        if time.time() >= deadline:
            break
    return x

def solve():
    t0 = time.time()
    DEAD = t0 + 168.0
    rng = np.random.default_rng(12345)
    x0 = None; b0 = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (323, 7):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    v = _max_cos(a)
                    if v < b0:
                        b0 = v; x0 = a.copy()
        except Exception:
            pass
    if x0 is None:
        a = rng.standard_normal(size=(323, 7))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        x0 = a; b0 = _max_cos(a)
    x_best = x0.copy(); best = float(b0)
    n, d = x_best.shape
    it = 0
    no_imp = 0
    # main basin loop with fast smooth optimizer only
    while time.time() < DEAD - 22.0:
        x = x_best.copy()
        s = it % 8
        if s == 0:
            sig = 0.015 + 0.002 * min(no_imp, 10)
            x = x + rng.standard_normal(size=x.shape) * sig
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps, lr0, lr1, ta0, ta1 = 500, 0.03, 0.003, 0.012, 0.0025
            x = _smooth_relax(x, steps, lr0, lr1, ta0, ta1)
        elif s == 1:
            w = _worst(x)
            k = 35
            idx = np.argpartition(w, -k)[-k:]
            x[idx] += rng.standard_normal(size=(k, d)) * 0.07
            x[idx] /= np.linalg.norm(x[idx], axis=1, keepdims=True)
            steps, lr0, lr1, ta0, ta1 = 500, 0.03, 0.003, 0.012, 0.002
            x = _smooth_relax(x, steps, lr0, lr1, ta0, ta1)
        elif s == 2:
            m = best
            cc = _contacts(x, m, tol=2e-4)
            order = np.argsort(-cc)
            # rattlers (low contacts) + hubs
            low = np.where(cc <= 2)[0]
            torep = [int(v) for v in low[:4]]
            for o in order:
                if len(torep) >= 6:
                    break
                if int(o) not in torep:
                    torep.append(int(o))
            torep = torep[:6]
            if len(torep) > 0:
                x = _hole_greedy(x, torep, rng, n_cand=20000)
            x += rng.standard_normal(size=x.shape) * 0.004
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps, lr0, lr1, ta0, ta1 = 600, 0.025, 0.002, 0.010, 0.002
            x = _smooth_relax(x, steps, lr0, lr1, ta0, ta1)
        elif s == 3:
            x = x + rng.standard_normal(size=x.shape) * 0.03
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps, lr0, lr1, ta0, ta1 = 450, 0.03, 0.003, 0.015, 0.003
            x = _smooth_relax(x, steps, lr0, lr1, ta0, ta1)
        elif s == 4:
            k = 25
            idx = rng.choice(n, size=k, replace=False)
            x[idx] += rng.standard_normal(size=(k, d)) * 0.06
            x[idx] /= np.linalg.norm(x[idx], axis=1, keepdims=True)
            steps, lr0, lr1, ta0, ta1 = 450, 0.03, 0.003, 0.012, 0.0025
            x = _smooth_relax(x, steps, lr0, lr1, ta0, ta1)
        elif s == 5:
            # focused subset relax on worst neighbourhood
            w = _worst(x)
            k = 60
            idx = np.argpartition(w, -k)[-k:]
            x = x + rng.standard_normal(size=x.shape) * 0.008
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps, lr0, lr1, ta0, ta1 = 400, 0.02, 0.002, 0.008, 0.0015
            x = _smooth_relax(x, steps, lr0, lr1, ta0, ta1, subset=idx)
            # then full
            x = _smooth_relax(x, 200, 0.01, 0.002, 0.006, 0.0015)
        elif s == 6:
            w = _worst(x)
            k = 12
            idx = np.argpartition(w, -k)[-k:].tolist()
            x = _hole_greedy(x, [int(v) for v in idx[:6]], rng, n_cand=15000)
            x += rng.standard_normal(size=x.shape) * 0.005
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps, lr0, lr1, ta0, ta1 = 550, 0.025, 0.002, 0.010, 0.002
            x = _smooth_relax(x, steps, lr0, lr1, ta0, ta1)
        else:
            sig = 0.05
            x = x + rng.standard_normal(size=x.shape) * sig
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps, lr0, lr1, ta0, ta1 = 600, 0.035, 0.003, 0.015, 0.0025
            x = _smooth_relax(x, steps, lr0, lr1, ta0, ta1)
        # local hill tweak on worst point
        if time.time() < DEAD - 22.0:
            for _ in range(2):
                w = _worst(x)
                k = int(np.argmax(w))
                cur = x[k].copy()
                oth = np.delete(x, k, axis=0)
                cur_w = float((oth @ cur).max())
                trial = cur[None, :] + rng.standard_normal(size=(600, d)) * 0.02
                trial /= np.linalg.norm(trial, axis=1, keepdims=True)
                dw = (trial @ oth.T).max(axis=1)
                bi = int(np.argmin(dw))
                if dw[bi] < cur_w:
                    x[k] = trial[bi]
        v = _max_cos(x)
        if v < best - 1e-12 and np.all(np.isfinite(x)):
            best = v; x_best = x.copy()
            no_imp = 0
        else:
            no_imp += 1
        it += 1
        if time.time() >= DEAD - 22.0:
            break
    # final snap with short LP trust region if time permits
    if time.time() < DEAD - 3.0:
        try:
            y = _lp_polish(x_best, min(DEAD - 1.0, time.time() + 20.0), init_radius=2e-3)
            vy = _max_cos(y)
            if vy < best and np.all(np.isfinite(y)):
                x_best = y; best = vy
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
