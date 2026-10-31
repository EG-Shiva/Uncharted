import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds, t_deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if t_deadline is not None and time.time() > t_deadline:
            break
        if radius < 1e-14:
            break
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _anneal(X, betas, steps_list, s_list, mu=0.7):
    n, d = X.shape
    vel = np.zeros_like(X)
    for beta, steps, s0 in zip(betas, steps_list, s_list):
        for it in range(steps):
            G = X @ X.T
            np.fill_diagonal(G, -10.0)
            m = G.max()
            arg = beta * (G - m)
            # exp, clip very negative to 0
            W = np.exp(np.clip(arg, -30, 0.0))
            np.fill_diagonal(W, 0.0)
            grad = W @ X
            dot = np.sum(grad * X, axis=1, keepdims=True)
            gt = grad - dot * X
            rms = np.sqrt(np.mean(gt * gt)) + 1e-12
            gt = gt / rms
            vel = mu * vel + (1.0 - mu) * gt
            step = s0 * (1.0 - 0.5 * it / max(1, steps))
            Xn = X - step * vel
            Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
            X = Xn
    return X

def _max_descent(X, iters=400, s0=0.004):
    best = X.copy()
    best_m = _max_cos(X)
    step = s0
    n = X.shape[0]
    for it in range(iters):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        delta = 0.006 if it < 150 else 0.003
        mask = G > (m - delta)
        np.fill_diagonal(mask, False)
        W = mask.astype(float)
        grad = W @ X
        dot = np.sum(grad * X, axis=1, keepdims=True)
        gt = grad - dot * X
        rms = np.sqrt(np.mean(gt * gt)) + 1e-12
        if rms < 1e-14:
            break
        gt = gt / rms
        Xn = X - step * gt
        Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
        mn = _max_cos(Xn)
        if mn < m:
            X = Xn
            if mn < best_m:
                best = Xn.copy()
                best_m = mn
            step = min(step * 1.08, 0.02)
        else:
            step *= 0.55
            if step < 1e-6:
                break
    return best

def _find_hole(cur, rng, M=12000):
    C = rng.standard_normal(size=(M, 4))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    D = C @ cur.T
    vals = D.max(axis=1)
    bi = int(np.argmin(vals))
    h = C[bi].copy()
    step = 0.06
    for _ in range(150):
        dots = cur @ h
        md = float(dots.max())
        act = cur[dots > md - 0.06]
        if len(act) == 0:
            break
        grad = act.sum(axis=0)
        g = grad - float(grad @ h) * h
        nrm = float(np.linalg.norm(g))
        if nrm < 1e-12:
            break
        g /= nrm
        hn = h - step * g
        hn /= np.linalg.norm(hn)
        mdn = float((cur @ hn).max())
        if mdn < md:
            h = hn
            step = min(step * 1.06, 0.1)
        else:
            step *= 0.5
            if step < 1e-5:
                break
    return h / np.linalg.norm(h)

def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    rng = np.random.default_rng(12345)
    # load base
    try:
        base = np.load("best.npy").astype(float)
    except Exception:
        try:
            base = np.load("record.npy").astype(float)
        except Exception:
            base = rng.standard_normal(size=(153, 4))
    if base.shape != (153, 4):
        # fallback: tile/truncate
        tmp = rng.standard_normal(size=(153, 4))
        m = min(base.shape[0], 153)
        tmp[:m] = base[:m, :4] if base.shape[1] >= 4 else base
        base = tmp
    base = base / np.linalg.norm(base, axis=1, keepdims=True)
    base_val = _max_cos(base)
    # exploration deadline: reserve ~45s for final LP
    EXPLORE_END = t0 + 125.0
    # candidate pool: keep top few raw trials
    top = []  # list of (val, X)
    def push_top(v, X):
        top.append((v, X.copy()))
        top.sort(key=lambda z: z[0])
        del top[6:]

    push_top(base_val, base)
    n = 153
    t = 0
    betas_all = [15.0, 45.0, 110.0]
    while time.time() < EXPLORE_END and time.time() < HARD_END - 50:
        t += 1
        mode = t % 7
        X = top[0][1].copy() if (t % 3 != 0) else base.copy()
        # for diversity sometimes start from 2nd best
        if len(top) > 1 and (t % 11 == 0):
            X = top[1][1].copy()
        if mode in (0, 1, 2):
            sigmas = [0.03, 0.08, 0.16]
            sg = sigmas[mode]
            X = X + sg * rng.standard_normal(size=X.shape)
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            steps_l = [250, 250, 250]
            s_l = [0.010, 0.005, 0.0025]
        elif mode == 3:
            k = int(rng.integers(2, 5))
            idx = rng.choice(n, size=k, replace=False)
            X[idx] = rng.standard_normal(size=(k, 4))
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            steps_l = [300, 300, 300]
            s_l = [0.010, 0.005, 0.002]
        elif mode == 4:
            k = int(rng.integers(5, 10))
            idx = rng.choice(n, size=k, replace=False)
            X[idx] = X[idx] + 0.45 * rng.standard_normal(size=(k, 4))
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            steps_l = [300, 300, 300]
            s_l = [0.012, 0.006, 0.0025]
        else:
            # hole reinsertion of overloaded points
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            cnt = np.sum(G > m - 1e-9, axis=1)
            k = 2 + (t % 3)  # 2..4
            idx = np.argsort(cnt)[-k:]
            keep = np.ones(n, dtype=bool)
            keep[idx] = False
            cur_keep = X[keep]
            placed = []
            cur = cur_keep
            for _ in range(k):
                h = _find_hole(cur, rng, M=10000)
                placed.append(h)
                cur = np.vstack([cur, h[None, :]])
            for ii, h in zip(idx, placed):
                X[ii] = h
            # shake rest slightly
            X = X + 0.015 * rng.standard_normal(size=X.shape)
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            steps_l = [350, 350, 350]
            s_l = [0.010, 0.005, 0.002]
        # anneal + hard descent
        try:
            X = _anneal(X, betas_all, steps_l, s_l, mu=0.7)
            X = _max_descent(X, iters=350, s0=0.004)
        except Exception:
            continue
        # guard
        if not np.all(np.isfinite(X)):
            continue
        nr = np.linalg.norm(X, axis=1)
        if np.any(nr < 1e-8):
            continue
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        v = _max_cos(X)
        push_top(v, X)
        # if raw beats base, update base immediately (elitist)
        if v < base_val - 1e-12:
            base = X.copy()
            base_val = v
    # final selection: polish best few with LP
    top.sort(key=lambda z: z[0])
    # ensure base included
    cands = top[:3]
    best_X = cands[0][1].copy()
    best_v = cands[0][0]
    # allocate remaining time to LP polish in order
    for i, (v, Xc) in enumerate(cands):
        if time.time() > HARD_END - 5:
            break
        remain = HARD_END - time.time() - 2.0
        if remain < 3:
            break
        # give more to the best
        alloc = min(remain / (len(cands) - i), 25.0) if i == 0 else min(remain / (len(cands) - i), 10.0)
        alloc = max(alloc, 3.0)
        try:
            Y = _lp_polish(Xc, seconds=alloc, t_deadline=HARD_END - 1.0)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            vy = _max_cos(Y)
            if vy < best_v:
                best_v = vy
                best_X = Y.copy()
        except Exception:
            continue
    # also try polishing base if not already best
    best_X /= np.linalg.norm(best_X, axis=1, keepdims=True)
    # tiny shrink guard (strictly enforce unit rows, no NaN)
    best_X = np.where(np.isfinite(best_X), best_X, 0.0)
    nr = np.linalg.norm(best_X, axis=1, keepdims=True)
    nr = np.where(nr < 1e-12, 1.0, nr)
    best_X = best_X / nr
    return best_X
