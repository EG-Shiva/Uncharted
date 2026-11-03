import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def row_maxs(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return np.max(G, axis=1), G

def smooth_polish(X, steps, lr0, lr1, tau0, tau1, rng=None):
    n, d = X.shape
    for s in range(steps):
        f = s / max(1, steps - 1)
        lr = lr0 * ((lr1 / lr0) ** f) if lr0 > 0 and lr1 > 0 else lr0
        tau = tau0 * (1 - f) + tau1 * f
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = np.max(G)
        E = np.exp((G - m) / max(tau, 1e-9))
        np.fill_diagonal(E, 0.0)
        S = np.sum(E)
        if S <= 0 or not np.isfinite(S):
            break
        P = E / S
        Gr = P @ X  # gradient of soft-max
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Ft = Gr - dot * X
        norms = np.sqrt(np.sum(Ft * Ft, axis=1))
        mx = np.max(norms)
        if mx < 1e-12 or not np.isfinite(mx):
            break
        X = X - (lr / mx) * Ft
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def refine_single(x, Y, steps, lr0, lr1, tau0, tau1):
    # minimize max(Y @ x) over sphere, Y: (m,d)
    for s in range(steps):
        f = s / max(1, steps - 1)
        lr = lr0 * ((lr1 / lr0) ** f) if lr0 > 0 and lr1 > 0 else lr0
        tau = tau0 * (1 - f) + tau1 * f
        dvec = Y @ x
        m = np.max(dvec)
        w = np.exp((dvec - m) / max(tau, 1e-9))
        w /= np.sum(w)
        g = w @ Y
        g = g - np.dot(g, x) * x
        nrm = np.linalg.norm(g)
        if nrm < 1e-12 or not np.isfinite(nrm):
            break
        x = x - (lr / nrm) * g
        x /= np.linalg.norm(x)
    return x

def best_reinsert(Y_others, rng, n_sample=4000):
    d = Y_others.shape[1]
    cand = rng.standard_normal((n_sample, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ Y_others.T
    vals = np.max(dots, axis=1)
    order = np.argsort(vals)[:4]
    best_x = None
    best_v = 1e9
    for oi in order:
        x0 = cand[oi].copy()
        x1 = refine_single(x0, Y_others, steps=250, lr0=0.05, lr1=0.002, tau0=0.02, tau1=0.003)
        v = float(np.max(Y_others @ x1))
        if v < best_v:
            best_v = v
            best_x = x1
    return best_x, best_v

def lp_polish(x, seconds, t_end=1e18):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < t_end and radius > 1e-14:
        Gfull = x @ x.T
        c_all = Gfull[iu]
        m = c_all.max()
        sel = c_all > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        c = c_all[sel]
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
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    TIME_LIMIT = 178.0
    LP_BUDGET = 28.0
    explore_end = t0 + TIME_LIMIT - LP_BUDGET - 2.0
    rng = np.random.default_rng(0)
    n, d = 131, 9
    best = None
    best_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.load(fn).astype(float)
            if v.shape == (n, d):
                v /= np.linalg.norm(v, axis=1, keepdims=True)
                c = max_cos(v)
                if c < best_val:
                    best_val = c
                    best = v.copy()
        except Exception:
            pass
    if best is None:
        best = rng.standard_normal((n, d))
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = max_cos(best)
    # elite archive
    elites = [(best_val, best.copy())]

    def push_elite(val, X):
        elites.append((val, X.copy()))
        elites.sort(key=lambda z: z[0])
        del elites[5:]

    def get_degrees(X, mv):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        C = G >= (mv - 0.005)
        deg = C.sum(axis=1)
        return deg, G

    it = 0
    while time.time() < explore_end:
        it += 1
        # choose base: mostly best, sometimes 2nd/3rd elite for diversity
        if len(elites) > 1 and rng.random() < 0.20:
            bidx = int(rng.integers(0, len(elites)))
            base, base_val = elites[bidx][1], elites[bidx][0]
        else:
            base, base_val = elites[0][1], elites[0][0]
        deg, G = get_degrees(base, base_val)
        r = rng.random()
        Y = base.copy()
        if r < 0.35:
            # small global shake
            Y += 0.015 * rng.standard_normal(Y.shape)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            Y = smooth_polish(Y, steps=220, lr0=0.02, lr1=0.002, tau0=0.03, tau1=0.004)
        elif r < 0.55:
            Y += 0.04 * rng.standard_normal(Y.shape)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            Y = smooth_polish(Y, steps=260, lr0=0.025, lr1=0.002, tau0=0.03, tau1=0.003)
        elif r < 0.72:
            # hub kick: kick 1-3 highest degree points hard
            k = 1 if rng.random() < 0.5 else (2 if rng.random() < 0.7 else 3)
            idx = np.argsort(-deg)[:max(k+2, 6)]
            # sample among top to keep determinism with randomness
            pick = rng.choice(idx, size=k, replace=False)
            Y[pick] += 0.25 * rng.standard_normal((k, d))
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            Y = smooth_polish(Y, steps=280, lr0=0.025, lr1=0.002, tau0=0.035, tau1=0.004)
        elif r < 0.90:
            # optimal reinsertion of worst point
            rm, _ = row_maxs(Y)
            worst = int(np.argmax(rm))
            # occasionally pick a hub instead
            if rng.random() < 0.4:
                worst = int(np.argmax(deg))
            mask = np.ones(n, dtype=bool); mask[worst] = False
            Y_others = Y[mask]
            nx, _ = best_reinsert(Y_others, rng, n_sample=4000)
            Y[worst] = nx
            Y = smooth_polish(Y, steps=200, lr0=0.015, lr1=0.0015, tau0=0.025, tau1=0.003)
        else:
            # large shake
            Y += 0.09 * rng.standard_normal(Y.shape)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            Y = smooth_polish(Y, steps=300, lr0=0.03, lr1=0.002, tau0=0.04, tau1=0.004)
        v = max_cos(Y)
        if v < elites[0][0] - 1e-12:
            push_elite(v, Y)
        elif v < elites[-1][0] - 5e-06 and len(elites) < 5:
            push_elite(v, Y)
        # also keep near-best diverse: if within 2e-4 and far, keep
        if time.time() >= explore_end:
            break

    # final LP squeeze on top elites
    elites.sort(key=lambda z: z[0])
    t_lp_start = time.time()
    t_lp_end = t0 + TIME_LIMIT - 1.0
    # quick smooth re-polish with sharp tau before LP
    for i in range(min(3, len(elites))):
        if time.time() >= t_lp_end:
            break
        val, X = elites[i]
        X = smooth_polish(X.copy(), steps=120, lr0=0.008, lr1=0.001, tau0=0.012, tau1=0.002)
        # allocate remaining budget
        remain = t_lp_end - time.time()
        per = remain / (min(3, len(elites)) - i)
        per = min(max(per, 3.0), 12.0)
        X = lp_polish(X, seconds=per, t_end=t_lp_end)
        v = max_cos(X)
        elites.append((v, X))
    elites.sort(key=lambda z: z[0])
    out = elites[0][1].copy()
    out /= np.linalg.norm(out, axis=1, keepdims=True)
    return out
