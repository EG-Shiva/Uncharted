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
    n[~np.isfinite(n)] = 1.0
    n[n < 1e-15] = 1.0
    return x / n

def _adam_softmax(x_init, steps, T0, T1, lr0, lr1):
    n, d = x_init.shape
    X = _norm(x_init.copy())
    m1 = np.zeros_like(X)
    m2 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lt0, lt1 = np.log(T0), np.log(T1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    best_val = _max_cos(X)
    best_x = X.copy()
    for it in range(steps):
        frac = it / max(1, steps - 1)
        T = float(np.exp(lt0 + (lt1 - lt0) * frac))
        lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        if m < best_val:
            best_val = m
            best_x = X.copy()
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        s = float(np.sum(E) * 0.5)
        if not np.isfinite(s) or s <= 0:
            continue
        Grad = (E @ X) / s
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        m1 = b1 * m1 + (1.0 - b1) * Gt
        m2 = b2 * m2 + (1.0 - b2) * (Gt * Gt)
        bc1 = 1.0 - b1 ** (it + 1)
        bc2 = 1.0 - b2 ** (it + 1)
        step = (m1 / bc1) / (np.sqrt(m2 / bc2) + eps)
        X = X - lr * (step - np.sum(step * X, axis=1, keepdims=True) * X)
        X = _norm(X)
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    if m < best_val:
        best_val = m
        best_x = X.copy()
    return best_x, best_val

def _greedy(x, iters, step0=4e-4):
    x = _norm(x.copy())
    best = _max_cos(x)
    step = step0
    n = x.shape[0]
    for _ in range(iters):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        k = int(np.argmax(G))
        i = k // n
        j = k % n
        dot = float(G[i, j])
        vi = dot * x[i] - x[j]
        vj = dot * x[j] - x[i]
        ni = float(np.linalg.norm(vi)); nj = float(np.linalg.norm(vj))
        if ni < 1e-14 or nj < 1e-14:
            step *= 0.5
            if step < 1e-13:
                step = step0 * 0.5
            continue
        vi /= ni; vj /= nj
        improved = False
        y = x.copy()
        y[i] = _norm((x[i] + step * vi)[None, :])[0]
        y[j] = _norm((x[j] + step * vj)[None, :])[0]
        v = _max_cos(y)
        if v < best:
            x, best, step = y, v, min(step * 1.2, 5e-3)
            continue
        y2 = x.copy(); y2[i] = _norm((x[i] + step * vi)[None, :])[0]
        v2 = _max_cos(y2)
        y3 = x.copy(); y3[j] = _norm((x[j] + step * vj)[None, :])[0]
        v3 = _max_cos(y3)
        if v2 < best and v2 <= v3:
            x, best, step = y2, v2, min(step * 1.1, 5e-3)
        elif v3 < best:
            x, best, step = y3, v3, min(step * 1.1, 5e-3)
        else:
            step *= 0.5
            if step < 1e-13:
                step = step0 * 0.3
    return x, best

def _lp_polish(x, seconds, t_end):
    x = _norm(x.copy())
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x, 1e9
    best_x = x.copy()
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    start = time.time()
    while time.time() < t_end and (time.time() - start) < seconds and radius > 1e-14:
        if time.time() >= t_end - 0.5:
            break
        G = x @ x.T
        c = G[iu]
        m = float(np.max(c))
        sel = c > m - max(20.0 * radius, 1e-9)
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 3500:
            idx = np.argpartition(-c, 3500)[:3500]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx] = True
            sel = sel & mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, n * d + 1)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(n * d + 1); obj[-1] = 1.0
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
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            best = v; best_x = y.copy(); x = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return best_x, best

def _void_reinsert(x, rng, tries=1500, n_worst=4):
    x = _norm(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    G0 = x @ x.T
    np.fill_diagonal(G0, -2.0)
    rowmax = np.max(G0, axis=1)
    order = np.argsort(-rowmax)[:n_worst]
    for w in order:
        others = np.delete(x, w, axis=0)
        cand = rng.randn(tries, d)
        cand = cand / np.linalg.norm(cand, axis=1, keepdims=True)
        dots = cand @ others.T
        cmax = np.max(dots, axis=1)
        k = int(np.argmin(cmax))
        if cmax[k] < rowmax[w]:
            y = x.copy(); y[w] = cand[k]
            v = _max_cos(y)
            if v < best:
                x = y; best = v
                G0 = x @ x.T; np.fill_diagonal(G0, -2.0)
                rowmax = np.max(G0, axis=1)
    return x, best

def solve():
    t0 = time.time()
    T_BUDGET = 172.0
    t_end = t0 + T_BUDGET
    rng = np.random.RandomState(20261103)
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
        if c is None:
            continue
        c = np.asarray(c, dtype=np.float64)
        if c.ndim == 2 and c.shape == (182, 9):
            cands.append(_norm(c))
    if not cands:
        cands.append(_norm(rng.randn(182, 9)))
    vals = [_max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    x_best = cands[bi].copy()
    best_val = float(vals[bi])
    # quick initial LP to jam if far from optimum
    # exploration loop
    sigmas = [0.04, 0.015, 0.09, 0.006, 0.20, 0.025, 0.05, 0.010]
    cycle = 0
    # reserve final polish time
    while time.time() < t_end - 22.0:
        cycle += 1
        sig = sigmas[(cycle - 1) % len(sigmas)]
        r = np.random.RandomState(1000 + cycle * 7919)
        # pointwise worst for targeted kick
        G = x_best @ x_best.T
        np.fill_diagonal(G, -2.0)
        rowmax = np.max(G, axis=1)
        worst = np.argsort(-rowmax)
        noise = r.randn(182, 9)
        # heavier kick on worst 25 points
        wscale = np.ones(182) * 0.5
        wscale[worst[:25]] = 2.0
        wscale[worst[:6]] = 3.0
        x_try = x_best + sig * noise * wscale[:, None]
        x_try = _norm(x_try)
        # annealed softmax adam: two stages
        if sig > 0.08:
            x_try, _ = _adam_softmax(x_try, steps=500, T0=0.02, T1=0.003, lr0=0.02, lr1=0.005)
        elif sig > 0.02:
            x_try, _ = _adam_softmax(x_try, steps=400, T0=0.01, T1=0.0015, lr0=0.015, lr1=0.004)
        else:
            x_try, _ = _adam_softmax(x_try, steps=300, T0=0.006, T1=0.001, lr0=0.010, lr1=0.003)
        x_try, v_try = _greedy(x_try, iters=120, step0=4e-4)
        # void reinsertion on promising candidates
        if v_try < best_val + 0.004:
            x_try, v_try = _void_reinsert(x_try, r, tries=1200, n_worst=4)
            x_try, v_try = _greedy(x_try, iters=80, step0=3e-4)
        if v_try < best_val + 0.002 and time.time() < t_end - 30.0:
            # short LP jam to evaluate basin
            xj, vj = _lp_polish(x_try, seconds=4.0, t_end=t_end - 22.0)
            if vj < v_try:
                x_try, v_try = xj, vj
        if v_try < best_val:
            x_best = x_try.copy()
            best_val = float(v_try)
        if time.time() >= t_end - 22.0:
            break
        # occasional diversification from second-best basin: if stuck many cycles, add big kick
        if cycle % 12 == 0 and time.time() < t_end - 40.0:
            # big random restart blended with best
            r2 = np.random.RandomState(5000 + cycle)
            x_big = x_best + 0.35 * r2.randn(182, 9)
            x_big = _norm(x_big)
            x_big, _ = _adam_softmax(x_big, steps=600, T0=0.03, T1=0.002, lr0=0.025, lr1=0.004)
            x_big, vb = _greedy(x_big, iters=150, step0=5e-4)
            if vb < best_val + 0.003:
                xj, vj = _lp_polish(x_big, seconds=3.0, t_end=t_end - 22.0)
                if vj < vb:
                    x_big, vb = xj, vj
            if vb < best_val:
                x_best, best_val = x_big, float(vb)
    # final intensive polish of best
    try:
        x_best, best_val = _greedy(x_best, iters=200, step0=3e-4)
    except Exception:
        pass
    try:
        remaining = max(1.0, t_end - time.time() - 1.0)
        remaining = min(remaining, 20.0)
        x_best, best_val = _lp_polish(x_best, seconds=remaining, t_end=t_end - 0.5)
    except Exception:
        pass
    x_best = _norm(x_best)
    return x_best
