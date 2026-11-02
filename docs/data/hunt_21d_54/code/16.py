import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def proj_unit(x):
    nrm = np.linalg.norm(x, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    x = x / nrm
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    nrm2 = np.linalg.norm(x, axis=1, keepdims=True)
    nrm2[nrm2 == 0] = 1.0
    return x / nrm2

def lp_polish(x, seconds, r0=5e-3):
    x = proj_unit(x)
    n, d = x.shape
    best = max_cos(x)
    radius = r0
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        cfull = G[iu0]
        m = float(np.max(cfull))
        thr = max(20.0 * radius, 1e-9)
        sel = cfull > m - thr
        I = iu0[0][sel]
        J = iu0[1][sel]
        csv = cfull[sel]
        p = len(I)
        if p == 0:
            break
        if p > 2500:
            k = 2500
            idx = np.argpartition(-csv, k)[:k]
            I = I[idx]; J = J[idx]; csv = csv[idx]
            p = k
        nv = n * d + 1
        ar = np.arange(d)
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + ar, J[:, None] * d + ar, np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csv, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": 3.0, "presolve": "on"})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = proj_unit(y)
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def contact_info(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    C = (G > m - 1e-9)
    cnts = C.sum(axis=1)
    permax = G.max(axis=1)
    return cnts, permax, m, G

def shake_points(x, rng, strength, n_kick):
    n, d = x.shape
    cnts, _, _, _ = contact_info(x)
    order_low = np.argsort(cnts)
    order_high = np.argsort(-cnts)
    targets = set()
    targets.update(order_low[:n_kick // 3 + 1].tolist())
    targets.update(order_high[:n_kick // 3 + 1].tolist())
    while len(targets) < n_kick:
        targets.add(int(rng.integers(0, n)))
    y = x.copy()
    for i in list(targets):
        v = rng.normal(size=d)
        v -= (v @ y[i]) * y[i]
        nv = float(np.linalg.norm(v))
        if nv < 1e-12:
            continue
        v /= nv
        ang = strength * (0.5 + float(rng.random()))
        y[i] = np.cos(ang) * y[i] + np.sin(ang) * v
    noise = rng.normal(size=(n, d)) * (strength * 0.2)
    noise -= (np.sum(noise * y, axis=1, keepdims=True)) * y
    y = y + noise
    return proj_unit(y)

def repulsion_smooth(x, steps, lr, T0=0.02, T1=0.005):
    y = x.copy()
    n, d = y.shape
    for s in range(steps):
        G = y @ y.T
        np.fill_diagonal(G, -10.0)
        m = float(np.max(G))
        T = T0 + (T1 - T0) * (s / max(1, steps - 1))
        W = np.exp((G - m) / max(T, 1e-6))
        np.fill_diagonal(W, 0.0)
        grad = W @ y
        grad -= (np.sum(grad * y, axis=1, keepdims=True)) * y
        gn = np.linalg.norm(grad, axis=1, keepdims=True) + 1e-12
        grad = grad / gn
        step = lr * (1.0 - 0.5 * s / steps)
        y = y - step * grad
        y = proj_unit(y)
    return y

def refine_void(v, X, steps=80, lr=0.05):
    v = v / (np.linalg.norm(v) + 1e-12)
    for s in range(steps):
        dots = X @ v
        m = float(np.max(dots))
        T = max(0.004, 0.015 * (1.0 - s / steps))
        w = np.exp((dots - m) / T)
        g = w @ X
        g = g - (g @ v) * v
        ng = float(np.linalg.norm(g))
        if ng < 1e-12:
            break
        g /= ng
        st = lr * (1.0 - 0.5 * s / steps)
        v = v - st * g
        v = v / (np.linalg.norm(v) + 1e-12)
    return v

def find_voids(X, rng, n_needed, n_samples=12000):
    d = X.shape[1]
    C = rng.normal(size=(n_samples, d))
    C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    S = C @ X.T
    scores = S.max(axis=1)
    idx = np.argsort(scores)
    cands = C[idx[:max(n_needed * 4, 8)]]
    out = []
    for c in cands:
        r = refine_void(c, X, steps=60, lr=0.05)
        sc = float(np.max(X @ r))
        out.append((sc, r))
    out.sort(key=lambda t: t[0])
    return [r for _, r in out[:n_needed]]

def kill_reinsert(x, rng, n_kill):
    n, d = x.shape
    cnts, permax, _, _ = contact_info(x)
    order = np.argsort(-cnts - 0.1 * permax * 100.0)
    pool = order[:max(10, n_kill * 4)].tolist()
    victims = []
    while len(victims) < n_kill and pool:
        k = int(rng.integers(0, len(pool)))
        victims.append(pool.pop(k))
    while len(victims) < n_kill:
        victims.append(int(rng.integers(0, n)))
    keep = [i for i in range(n) if i not in set(victims)]
    S = x[keep].copy()
    for _ in victims:
        voids = find_voids(S, rng, 1)
        S = np.vstack([S, voids[0][None, :]])
    S = proj_unit(S)
    # put back in original order: survivors keep order, victims replaced
    y = x.copy()
    # simplest: return S with survivors first is fine (permutation invariant)
    return proj_unit(S)

def solve():
    T0 = time.time()
    LIMIT = 170.0
    elapsed = lambda: time.time() - T0
    rng = np.random.default_rng(20260707)
    try:
        x0 = np.load("best.npy")
    except Exception:
        try:
            x0 = np.load("record.npy")
        except Exception:
            x0 = rng.normal(size=(54, 21))
    x0 = proj_unit(np.asarray(x0, dtype=float).reshape(54, 21))
    best = x0.copy()
    best_val = max_cos(best)
    if elapsed() < LIMIT - 60:
        best = lp_polish(best, min(6.0, LIMIT - 60 - elapsed()), r0=1e-3)
        best_val = max_cos(best)
    strengths = [0.02, 0.05, 0.10, 0.20, 0.35, 0.03, 0.07, 0.15]
    it = 0
    while elapsed() < LIMIT - 22:
        it += 1
        mode = it % 5
        try:
            if mode in (0, 1, 3):
                s = strengths[(it - 1) % len(strengths)]
                nk = int(rng.integers(3, 12))
                cand = shake_points(best, rng, strength=s, n_kick=nk)
                cand = repulsion_smooth(cand, steps=70, lr=0.02)
            elif mode == 2:
                nk = int(rng.integers(1, 3))
                cand = kill_reinsert(best, rng, n_kill=nk)
                cand = repulsion_smooth(cand, steps=50, lr=0.015)
            else:
                cand = best + rng.normal(size=best.shape) * 0.012
                cand = proj_unit(cand)
                cand = repulsion_smooth(cand, steps=120, lr=0.03, T0=0.03, T1=0.004)
            budget = min(4.5, (LIMIT - 22 - elapsed()) * 0.5)
            if budget < 0.8:
                break
            cand = lp_polish(cand, budget, r0=5e-3)
            v = max_cos(cand)
            if v < best_val - 1e-12:
                best, best_val = cand, v
        except Exception:
            continue
    if elapsed() < LIMIT - 1.0:
        try:
            best = lp_polish(best, LIMIT - elapsed() - 0.4, r0=2e-3)
        except Exception:
            pass
    best = proj_unit(best)
    return best
