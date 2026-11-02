import time
import numpy as np

def _norm_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-16)
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _adam_quench(X0, steps, a0, a1, lr0, lr1, rng=None, cap=0.05):
    X = X0.copy()
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        frac = t / float(steps)
        alpha = a0 * (a1 / a0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = np.max(G)
        # exp(alpha*(G-M)), suppress underflow
        E = np.exp(alpha * (G - M))
        np.fill_diagonal(E, 0.0)
        S = np.sum(E)
        if not np.isfinite(S) or S <= 0:
            continue
        grad = 2.0 * (E @ X) / S
        proj = np.sum(grad * X, axis=1, keepdims=True)
        gt = grad - proj * X
        m = b1 * m + (1.0 - b1) * gt
        v = b2 * v + (1.0 - b2) * (gt * gt)
        mhat = m / (1.0 - b1 ** t)
        vhat = v / (1.0 - b2 ** t)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        scale = np.minimum(1.0, cap / (sn + 1e-16))
        X = X - step * scale
        X = _norm_rows(X)
        if not np.all(np.isfinite(X)):
            break
    return X

def _find_hole(Y, rng, n_cand=2500):
    d = Y.shape[1]
    cand = rng.randn(n_cand, d)
    cand = _norm_rows(cand)
    dots = cand @ Y.T
    md = np.max(dots, axis=1)
    return cand[np.argmin(md)]

def _lp_polish_capped(x, seconds, r0=1e-3):
    # single sparing LP polish, heavily capped; skip if scipy missing
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except Exception:
        return x
    x = _norm_rows(x)
    n, d = x.shape
    best = _max_cos(x)
    xb = x.copy()
    radius = float(r0)
    start = time.time()
    iu = np.triu_indices(n, 1)
    it = 0
    while time.time() - start < seconds and radius > 1e-14 and it < 30:
        it += 1
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        cs = c[sel]
        p = len(I)
        if p == 0:
            break
        if p > 350:
            k = 350
            idx = np.argpartition(-cs, k - 1)[:k]
            I, J, cs = I[idx], J[idx], cs[idx]
            p = k
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(),
                               (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq,
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
        y = _norm_rows(y)
        vv = _max_cos(y)
        if vv < best:
            x, best, xb = y, vv, y.copy()
            radius = min(radius * 1.5, 0.01)
        else:
            radius *= 0.5
    return xb

def solve():
    t0 = time.time()
    DEADLINE = 168.0
    SEARCH_END = 143.0
    rng = np.random.RandomState(12345)
    n, d = 55, 21
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=float)
            if a.shape == (55, 21):
                a = _norm_rows(a)
                s = _max_cos(a)
                if s < best_val:
                    best_val = s
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _norm_rows(rng.randn(n, d))
        best_val = _max_cos(best)
    cur = best.copy()
    cur_val = best_val
    # SA temperatures for max-cosine differences
    T0, T1 = 3e-04, 2e-06
    trial = 0
    max_trials = 1000
    while time.time() - t0 < SEARCH_END and trial < max_trials:
        trial += 1
        remain = SEARCH_END - (time.time() - t0)
        if remain < 1.0:
            break
        frac = trial / 200.0
        T = T0 * (T1 / T0) ** min(1.0, frac)
        prop = cur.copy()
        mode = trial % 7
        # diagnostics on cur
        G = prop @ prop.T
        np.fill_diagonal(G, -2.0)
        mx = float(np.max(G))
        cnt = np.sum(G > mx - 1e-3, axis=1)
        rowmx = np.max(G, axis=1)
        if mode == 0:
            # hole reinsert 1-2 overloaded hubs
            k = 1 if rng.rand() < 0.6 else 2
            victims = np.argsort(-cnt)[:k]
            if rng.rand() < 0.3:
                victims = rng.choice(n, size=k, replace=False)
            keep = np.ones(n, dtype=bool); keep[victims] = False
            Y = prop[keep]
            for vi in victims:
                h = _find_hole(Y, rng, n_cand=2500)
                prop[vi] = h
                Y = np.vstack([Y, h[None, :]])
            prop = _norm_rows(prop + 0.008 * rng.randn(n, d))
            steps, a0, a1, lr0, lr1 = 500, 40.0, 600.0, 0.02, 0.003
        elif mode == 1:
            # subset kick of worst 8-14
            k = 8 + int(rng.rand() * 7)
            if rng.rand() < 0.5:
                idx = np.argsort(-cnt)[:k]
            else:
                idx = np.argsort(-rowmx)[:k]
            sigma = 0.05 + rng.rand() * 0.12
            prop[idx] = _norm_rows(prop[idx] + sigma * rng.randn(k, d))
            steps, a0, a1, lr0, lr1 = 450, 30.0, 500.0, 0.02, 0.003
        elif mode == 2:
            # medium global kick
            sigma = 0.04 + rng.rand() * 0.10
            prop = _norm_rows(prop + sigma * rng.randn(n, d))
            steps, a0, a1, lr0, lr1 = 450, 30.0, 550.0, 0.018, 0.003
        elif mode == 3:
            # large global kick to change basin
            sigma = 0.18 + rng.rand() * 0.25
            prop = _norm_rows(prop + sigma * rng.randn(n, d))
            steps, a0, a1, lr0, lr1 = 600, 25.0, 450.0, 0.02, 0.002
        elif mode == 4:
            # random 2D plane rotation of a contact cluster
            w = int(np.argmax(cnt))
            gw = (prop @ prop[w])
            gw[w] = -2
            cl = np.argsort(-gw)[:10]
            cl = np.append(cl, w)
            A = rng.randn(d, 2)
            Q, _ = np.linalg.qr(A)
            ang = 0.25 + rng.rand() * 0.6
            if rng.rand() < 0.5:
                ang = -ang
            R = np.eye(d) + (np.cos(ang) - 1.0) * (Q @ Q.T) + np.sin(ang) * (Q[:, [0]] @ Q[:, [1]].T - Q[:, [1]] @ Q[:, [0]].T)
            # rotate cluster + jitter
            prop[cl] = _norm_rows(prop[cl] @ R.T + 0.02 * rng.randn(len(cl), d))
            steps, a0, a1, lr0, lr1 = 500, 35.0, 550.0, 0.018, 0.003
        elif mode == 5:
            # single worst hole move + neighbour jitter
            w = int(np.argmax(rowmx))
            keep = np.ones(n, dtype=bool); keep[w] = False
            h = _find_hole(prop[keep], rng, n_cand=3500)
            prop[w] = _norm_rows(0.5 * prop[w] + 0.5 * h + 0.01 * rng.randn(d))
            gw = (prop @ prop[w]); gw[w] = -2
            nb = np.argsort(-gw)[:8]
            prop[nb] = _norm_rows(prop[nb] + 0.05 * rng.randn(len(nb), d))
            steps, a0, a1, lr0, lr1 = 500, 40.0, 600.0, 0.02, 0.003
        else:
            # occasional fresh random start quench (topology reset)
            if rng.rand() < 0.25:
                prop = _norm_rows(rng.randn(n, d))
                steps, a0, a1, lr0, lr1 = 800, 20.0, 400.0, 0.025, 0.003
            else:
                sigma = 0.10 + rng.rand() * 0.15
                # restart from global best with large kick
                prop = best.copy()
                prop = _norm_rows(prop + sigma * rng.randn(n, d))
                steps, a0, a1, lr0, lr1 = 550, 30.0, 500.0, 0.02, 0.003
        # quench with cheap annealed Adam (no LP)
        prop = _adam_quench(prop, steps=steps, a0=a0, a1=a1, lr0=lr0, lr1=lr1)
        v = _max_cos(prop)
        if v < best_val - 1e-12:
            best_val = v
            best = prop.copy()
            cur = prop.copy()
            cur_val = v
        else:
            # SA acceptance for cur to drift to new basins
            if rng.rand() < np.exp(-max(0.0, v - cur_val) / max(T, 1e-12)):
                cur = prop.copy()
                cur_val = v
            # periodic restart of walker to best to avoid wandering too far
            if trial % 40 == 0:
                cur = best.copy()
                cur_val = best_val
        # early exit if time low
        if time.time() - t0 > SEARCH_END - 0.5:
            break
    # final high-alpha squeeze of best with Adam (cheap, no timeout risk)
    rem = DEADLINE - (time.time() - t0)
    if rem > 3.0 and best is not None:
        try:
            best = _adam_quench(best, steps=400, a0=300.0, a1=2500.0, lr0=0.008, lr1=0.001, cap=0.02)
            best_val = _max_cos(best)
        except Exception:
            pass
    # one final capped LP squeeze if time remains
    rem = DEADLINE - (time.time() - t0)
    if rem > 4.0 and best is not None:
        try:
            secs = min(rem - 1.0, 6.0)
            best = _lp_polish_capped(best, seconds=secs, r0=8e-04)
        except Exception:
            pass
    best = _norm_rows(np.asarray(best, dtype=float))
    best[~np.isfinite(best)] = 0.01
    best = _norm_rows(best)
    return best
