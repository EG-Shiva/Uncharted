import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _lp_polish_deadline(x, deadline):
    x = _norm(x)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = 1e-3
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline:
        if radius < 1e-14:
            break
        try:
            G = x @ x.T
            c_all = G[iu0]
        except Exception:
            break
        m = float(c_all.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 3500:
            idx = np.where(sel)[0]
            order = np.argsort(-c_all[idx])[:3500]
            mask = np.zeros_like(sel)
            mask[idx[order]] = True
            sel = mask
        I = iu0[0][sel]
        J = iu0[1][sel]
        c = c_all[sel]
        p = len(I)
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate(
                [I[:, None] * d + np.arange(d),
                 J[:, None] * d + np.arange(d),
                 np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix(
                (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            y = x + res.x[:-1].reshape(n, d)
            y = _norm(y)
            if not np.all(np.isfinite(y)):
                radius *= 0.5
                continue
            v = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if deadline - time.time() < 0.25:
            break
    return x

def _smooth(x, steps, T0, T1, lr0, lr1, mom=0.85):
    x = _norm(x)
    v = np.zeros_like(x)
    n = x.shape[0]
    for k in range(steps):
        frac = k / max(1, steps - 1)
        T = T0 + (T1 - T0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        # exp weights
        W = np.exp((G - m) / max(T, 1e-6))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad = grad - dot * x
        s = float(W.sum()) / n + 1e-12
        grad = grad / s
        if not np.all(np.isfinite(grad)):
            break
        v = mom * v + (1.0 - mom) * grad
        x = x - lr * v
        x = _norm(x)
    return x

def _hole_for(Xk, rng, n_samp=4000):
    d = Xk.shape[1]
    Z = rng.standard_normal((n_samp, d))
    Z = _norm(Z)
    D = Z @ Xk.T
    worst = D.max(axis=1)
    i = int(np.argmin(worst))
    z = Z[i].copy()
    # refine hole: minimize max dot (repulsion from nearest)
    for _ in range(80):
        dots = Xk @ z
        m = float(dots.max())
        W = np.exp((dots - m) / 0.01)
        W = W / (W.sum() + 1e-12)
        g = W @ Xk
        g = g - float(g @ z) * z
        ng = np.linalg.norm(g)
        if ng < 1e-12:
            break
        # move opposite to weighted neighbors
        z = z - 0.05 * g
        z = z / (np.linalg.norm(z) + 1e-12)
    return z

def solve():
    t0 = time.time()
    HOP_END = t0 + 150.0
    HARD_END = t0 + 174.0
    rng = np.random.default_rng(12345)
    n, d = 131, 9
    xb = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            v = np.load(fn)
            v = np.asarray(v, dtype=float)
            if v.shape == (n, d) and np.all(np.isfinite(v)):
                v = _norm(v)
                s = _max_cos(v)
                if s < best_val:
                    best_val = s
                    xb = v.copy()
        except Exception:
            continue
    if xb is None:
        xb = _norm(rng.standard_normal((n, d)))
        best_val = _max_cos(xb)

    def analyze(x):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        rowmax = G.max(axis=1)
        deg = np.sum(G > m - 1e-9, axis=1)
        iu = np.triu_indices(n, 1)
        c = G[iu]
        sel = c > m - 1e-7
        I = iu[0][sel]; J = iu[1][sel]
        ratt = np.where(rowmax < m - 1e-7)[0]
        return m, deg, I, J, ratt

    try:
        _, deg, cI, cJ, rattlers = analyze(xb)
    except Exception:
        deg = np.zeros(n); cI = np.array([], dtype=int); cJ = np.array([], dtype=int); rattlers = np.array([], dtype=int)

    trial = 0
    while time.time() < HOP_END:
        if time.time() > HOP_END - 1.0:
            break
        mod = trial % 10
        y = xb.copy()
        try:
            if mod == 0:
                y = y + 0.006 * rng.standard_normal((n, d))
            elif mod == 1:
                y = y + 0.016 * rng.standard_normal((n, d))
            elif mod == 2:
                y = y + 0.003 * rng.standard_normal((n, d))
                k = 10
                idx = rng.choice(n, size=k, replace=False)
                y[idx] += 0.07 * rng.standard_normal((k, d))
            elif mod == 3:
                y = y + 0.004 * rng.standard_normal((n, d))
                if len(cI) > 0:
                    e = rng.integers(0, len(cI))
                    i, j = int(cI[e]), int(cJ[e])
                    diff = y[i] - y[j]
                    diff = diff / (np.linalg.norm(diff) + 1e-12)
                    push = 0.05 + 0.03 * rng.random()
                    y[i] += push * diff
                    y[j] -= push * diff
                else:
                    y += 0.012 * rng.standard_normal((n, d))
            elif mod == 4:
                # rattler -> hole
                y = y + 0.004 * rng.standard_normal((n, d))
                nr = len(rattlers)
                if nr > 0:
                    k = min(2, nr)
                    idx = rng.choice(rattlers, size=k, replace=False)
                    keep = np.ones(n, dtype=bool); keep[idx] = False
                    Xk = y[keep]
                    for ii in idx:
                        z = _hole_for(Xk, rng, n_samp=3000)
                        # random blend half the time for diversity
                        if rng.random() < 0.3:
                            z = z + 0.1 * rng.standard_normal(d)
                            z = z / (np.linalg.norm(z) + 1e-12)
                        y[ii] = z
                        Xk = np.vstack([Xk, z[None, :]])
                else:
                    idx = np.argsort(-deg)[:2]
                    keep = np.ones(n, dtype=bool); keep[idx] = False
                    Xk = y[keep]
                    for ii in idx:
                        z = _hole_for(Xk, rng, n_samp=3000)
                        y[ii] = z
                        Xk = np.vstack([Xk, z[None, :]])
            elif mod == 5:
                # hub kill -> hole
                y = y + 0.004 * rng.standard_normal((n, d))
                order = np.argsort(-deg)
                k = 1 if (trial % 20 != 5) else 3
                idx = order[:k]
                # for 3-kill, pick diverse: top1 + 2 random among top15
                if k == 3:
                    cand = order[:15]
                    idx = np.concatenate([[order[0]], rng.choice(cand[1:], size=2, replace=False)])
                keep = np.ones(n, dtype=bool); keep[idx] = False
                Xk = y[keep]
                for ii in idx:
                    z = _hole_for(Xk, rng, n_samp=4000)
                    y[ii] = z
                    Xk = np.vstack([Xk, z[None, :]])
            elif mod == 6:
                # multi hole swap: 3 worst replaced
                y = y + 0.003 * rng.standard_normal((n, d))
                order = np.argsort(-deg)
                idx = order[:3]
                keep = np.ones(n, dtype=bool); keep[idx] = False
                Xk = y[keep]
                for ii in idx:
                    z = _hole_for(Xk, rng, n_samp=2500)
                    y[ii] = z
                    Xk = np.vstack([Xk, z[None, :]])
            elif mod == 7:
                y = y + 0.06 * rng.standard_normal((n, d))
            elif mod == 8:
                y = y + 0.025 * rng.standard_normal((n, d))
                # plus randomize one point fully
                ii = rng.integers(0, n)
                r = rng.standard_normal(d); r /= (np.linalg.norm(r) + 1e-12)
                y[ii] = r if rng.random() < 0.5 else y[ii] + 0.3 * r
            else:
                # rare big jump
                sig = 0.18 if (trial % 30 == 9) else 0.09
                y = y + sig * rng.standard_normal((n, d))
            y = _norm(y)
        except Exception:
            trial += 1
            continue
        # smooth anneal (topology-changing)
        try:
            if mod in (4, 5, 6):
                y = _smooth(y, 260, 0.018, 0.004, 0.014, 0.0025, mom=0.85)
                y = _smooth(y, 140, 0.004, 0.0015, 0.004, 0.0012, mom=0.80)
            elif mod in (3, 9):
                y = _smooth(y, 280, 0.015, 0.003, 0.016, 0.002, mom=0.86)
            else:
                y = _smooth(y, 220, 0.016, 0.004, 0.012, 0.002, mom=0.85)
        except Exception:
            pass
        try:
            vs = _max_cos(y)
        except Exception:
            trial += 1
            continue
        # filter: only LP-polish if promising (saves time for many trials)
        # threshold adapts: allow worse early, tight later
        allow = best_val + (0.012 if mod in (7, 9) else 0.008)
        if vs > allow and rng.random() > 0.05:
            trial += 1
            # opportunistic accept of smooth-only if it beats best (rare)
            if vs < best_val - 1e-12:
                xb = _norm(y); best_val = vs
                try:
                    _, deg, cI, cJ, rattlers = analyze(xb)
                except Exception:
                    pass
            continue
        budget = min(1.4, HOP_END - time.time() - 0.05)
        if budget < 0.35:
            trial += 1
            continue
        try:
            y = _lp_polish_deadline(y, time.time() + budget)
            v = _max_cos(y)
        except Exception:
            trial += 1
            continue
        if v < best_val - 1e-12 and np.all(np.isfinite(y)):
            xb = _norm(y)
            best_val = v
            try:
                _, deg, cI, cJ, rattlers = analyze(xb)
            except Exception:
                pass
            # exploit breakthrough immediately
            if time.time() < HOP_END - 4.0:
                try:
                    y2 = _lp_polish_deadline(xb, time.time() + min(2.5, HOP_END - time.time()))
                    v2 = _max_cos(y2)
                    if v2 < best_val:
                        xb = _norm(y2); best_val = v2
                        _, deg, cI, cJ, rattlers = analyze(xb)
                except Exception:
                    pass
        trial += 1

    try:
        remain = HARD_END - time.time() - 0.5
        if remain > 1.0:
            xb = _lp_polish_deadline(xb, time.time() + remain)
    except Exception:
        pass
    xb = _norm(xb)
    return xb
