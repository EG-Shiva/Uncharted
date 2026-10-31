import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def lp_polish(x, seconds, t_deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    it = 0
    while True:
        if time.time() - start >= seconds:
            break
        if t_deadline is not None and time.time() >= t_deadline:
            break
        if radius < 1e-14:
            break
        if it >= 120:
            break
        it += 1
        G = x @ x.T
        c = G[iu0]
        m = float(c.max())
        thr = m - max(20.0 * radius, 2e-9)
        sel = c > thr
        if np.count_nonzero(sel) < 1:
            sel = c >= m
        I = iu0[0][sel]
        J = iu0[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], axis=1
        ).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix(
            (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
            shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
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
        v = max_cos(y)
        if v < best:
            x = y
            best = v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def contact_degree(x, thresh=1e-9):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    B = G >= m - thresh
    deg = B.sum(axis=1)
    return deg, m

def best_void_against(Y, rng, n_sample=8000, refine=100):
    cand = rng.standard_normal(size=(n_sample, 4))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    s = cand @ Y.T
    ms = s.max(axis=1)
    idx = int(np.argmin(ms))
    y = cand[idx].copy()
    cur = float(ms[idx])
    step = 0.02
    for _ in range(refine):
        prop = y + step * rng.standard_normal(4)
        nrm = np.linalg.norm(prop)
        if nrm == 0:
            continue
        prop /= nrm
        v = float(np.max(prop @ Y.T))
        if v < cur:
            y = prop
            cur = v
        else:
            step *= 0.97
            if step < 1e-4:
                step = 1e-4
    return y

def smooth_descent(x, rng, steps=1200, t0=0.004, t1=0.00035, lr0=0.006, lr1=0.0006, mu=0.85, jitter=0.0):
    n, d = x.shape
    y = x.copy()
    vel = np.zeros_like(y)
    b1, b2, eps = 0.9, 0.999, 1e-8
    m1 = np.zeros_like(y)
    m2 = np.zeros_like(y)
    for s in range(steps):
        frac = s / max(1, steps - 1)
        temp = t0 * (t1 / t0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = y @ y.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        # log-sum-exp weights
        E = np.exp((G - m) / temp)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum())
        if not np.isfinite(S) or S <= 0:
            break
        P = E / S
        grad = 2.0 * (P @ y)
        dot = (grad * y).sum(axis=1, keepdims=True)
        gt = grad - dot * y
        m1 = b1 * m1 + (1 - b1) * gt
        m2 = b2 * m2 + (1 - b2) * (gt * gt)
        bc1 = 1 - b1 ** (s + 1)
        bc2 = 1 - b2 ** (s + 1)
        m1h = m1 / bc1
        m2h = m2 / bc2
        step_v = lr * m1h / (np.sqrt(m2h) + eps)
        # trust: limit step
        vel = mu * vel + step_v
        y = y - vel
        # optional tiny noise
        if jitter > 0:
            y = y + jitter * rng.standard_normal(size=y.shape)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
    return y

def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    rng = np.random.default_rng(7)
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (163, 4):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = max_cos(a)
                if v < best_val:
                    best_val = v
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        a = rng.standard_normal(size=(163, 4))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        best = a
        best_val = max_cos(a)
    n, d = best.shape
    # short lock-in
    if HARD_END - time.time() > 25:
        best = lp_polish(best, seconds=2.0, t_deadline=HARD_END - 5)
        best_val = max_cos(best)
    cur = best.copy()
    cur_val = best_val
    deg, _ = contact_degree(best, thresh=1e-9)
    trial = 0
    sigmas = [0.02, 0.01, 0.005, 0.0025]
    while time.time() < HARD_END - 14:
        trial += 1
        rem = HARD_END - 14 - time.time()
        if rem <= 0:
            break
        mode = trial % 6
        # anneal acceptance tolerance
        tol = 0.0015 * (0.5 ** (trial / 60.0)) + 1e-5
        x_try = None
        if mode == 0:
            sigma = sigmas[(trial // 6) % len(sigmas)]
            x_try = cur + sigma * rng.standard_normal(size=(n, d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        elif mode == 1:
            # cluster shake around random center
            c = int(rng.integers(0, n))
            dots = cur[c] @ cur.T
            k = int(rng.integers(8, 20))
            idx = np.argsort(-dots)[:k]
            noise = np.zeros((n, d))
            noise[idx] = rng.standard_normal(size=(len(idx), d)) * 0.03
            noise += 0.001 * rng.standard_normal(size=(n, d))
            x_try = cur + noise
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        elif mode == 2:
            if trial % 8 == 2:
                deg, _ = contact_degree(cur, thresh=1e-9)
            order = np.argsort(-deg)
            k = int(rng.integers(4, 12))
            idx = order[:k]
            ratt = np.where(deg == 0)[0]
            if len(ratt):
                idx = np.unique(np.concatenate([idx, ratt]))
            noise = np.zeros((n, d))
            noise[idx] = rng.standard_normal(size=(len(idx), d)) * 0.025
            noise += 0.0015 * rng.standard_normal(size=(n, d))
            x_try = cur + noise
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        elif mode == 3:
            # rattler/void relocation
            dcur, _ = contact_degree(cur, thresh=1e-7)
            order = np.argsort(dcur)  # low first
            k = int(rng.integers(1, 4))
            move = order[:k]
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[move] = False
            Y = cur[keep_mask]
            curY = Y.copy()
            new_pts = []
            for _ in range(k):
                yv = best_void_against(curY, rng, n_sample=6000, refine=80)
                new_pts.append(yv)
                curY = np.vstack([curY, yv[None, :]])
            x_try = np.empty_like(cur)
            x_try[keep_mask] = Y
            x_try[~keep_mask] = np.array(new_pts)
            x_try = x_try + 0.002 * rng.standard_normal(size=(n, d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        elif mode == 4:
            # dropout jammed
            dcur, _ = contact_degree(cur, thresh=1e-7)
            order = np.argsort(-dcur)
            k = int(rng.integers(2, 6))
            drop = order[:k]
            if rng.random() < 0.4:
                drop = np.unique(np.append(drop, rng.integers(0, n)))
                k = len(drop)
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[drop] = False
            Y = cur[keep_mask]
            curY = Y.copy()
            new_pts = []
            for _ in range(k):
                yv = best_void_against(curY, rng, n_sample=6000, refine=80)
                new_pts.append(yv)
                curY = np.vstack([curY, yv[None, :]])
            x_try = np.empty_like(cur)
            x_try[keep_mask] = Y
            x_try[~keep_mask] = np.array(new_pts)
            x_try = x_try + 0.003 * rng.standard_normal(size=(n, d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        else:
            sigma = 0.04
            x_try = cur + sigma * rng.standard_normal(size=(n, d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        # smooth repulsion stage (fast, moves basin)
        steps = 900
        if mode in (3, 4):
            steps = 1300
        # shrink steps if little time
        if rem < 8:
            steps = 400
        x_sm = smooth_descent(x_try, rng, steps=steps,
                              t0=0.004, t1=0.0003,
                              lr0=0.005, lr1=0.0005, mu=0.8)
        v_sm = max_cos(x_sm)
        # short LP polish
        budget = 0.7
        if trial % 12 == 0:
            budget = 1.4
        budget = min(budget, max(0.3, HARD_END - 14 - time.time()))
        if budget < 0.1:
            # no time for LP, just evaluate
            y = x_sm
            v = v_sm
        else:
            y = lp_polish(x_sm, seconds=budget, t_deadline=HARD_END - 14)
            v = max_cos(y)
        # acceptance: walk on cur, track best
        if v < best_val - 1e-12:
            best = y.copy()
            best_val = v
            cur = y.copy()
            cur_val = v
            deg, _ = contact_degree(best, thresh=1e-9)
        elif v < cur_val + tol:
            cur = y.copy()
            cur_val = v
        # occasional restart from best if cur drifts too high
        if cur_val > best_val + 0.004 and rng.random() < 0.5:
            cur = best.copy()
            cur_val = best_val
    fbudget = max(0.5, HARD_END - time.time() - 1.0)
    fbudget = min(fbudget, 25.0)
    if fbudget > 0.3:
        # final smooth + LP on champion
        tmp = smooth_descent(best, rng, steps=1500, t0=0.002, t1=0.0002,
                             lr0=0.003, lr1=0.0003, mu=0.8)
        if max_cos(tmp) < best_val:
            best = tmp
            best_val = max_cos(tmp)
        best = lp_polish(best, seconds=fbudget)
        best_val = max_cos(best)
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
