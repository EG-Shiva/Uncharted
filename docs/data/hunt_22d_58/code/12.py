import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def rownorm(x):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x

def quench(x, steps, T0, T1, lr0, lr1):
    # annealed soft-max (log-sum-exp) tangential descent, per-point normalized steps
    n, d = x.shape
    lT0, lT1 = np.log(T0), np.log(T1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        frac = s / max(1, steps - 1)
        T = float(np.exp(lT0 + (lT1 - lT0) * frac))
        lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
        G = x @ x.T
        np.fill_diagonal(G, -1.0)
        m = G.max()
        # weights focused on near-max pairs
        W = np.exp((G - m) / T)
        np.fill_diagonal(W, 0.0)
        Grad = W @ x
        d = (Grad * x).sum(axis=1, keepdims=True)
        Gt = Grad - d * x
        nr = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        x = x - lr * (Gt / nr)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def contacts_counts(x, tol=2e-4):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = G.max()
    C = (G > m - tol)
    return C.sum(axis=1), m

def hole_sample(x, fix_idx, rng, ns=400):
    # find vector minimizing max dot to others (excluding fix_idx)
    n, d = x.shape
    mask = np.ones(n, dtype=bool)
    mask[fix_idx] = False
    others = x[mask]
    best_v = None
    best_dot = 1e9
    # random candidates
    cand = rng.normal(size=(ns, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ others.T
    md = dots.max(axis=1)
    i = int(np.argmin(md))
    return cand[i], float(md[i])

def lp_polish(x, seconds, t_end):
    x = rownorm(x)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < t_end and radius > 1e-14:
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
            break
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = rownorm(y)
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() + 0.5 > t_end:
            break
    return x

def solve():
    t0 = time.time()
    T_LIM = 170.0
    rng = np.random.default_rng(0)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = np.load("record.npy").astype(float)
    xb = rownorm(xb)
    best_v = max_cos(xb)
    best_x = xb.copy()
    try:
        xr = np.load("record.npy").astype(float)
        xr = rownorm(xr)
        vr = max_cos(xr)
        if vr < best_v:
            best_x, best_v = xr.copy(), vr
    except Exception:
        pass
    n, d = best_x.shape

    sigmas = [0.015, 0.04, 0.08, 0.14, 0.22, 0.35]
    trial = 0
    # reserve ~25s for final LP + greedy
    while time.time() - t0 < T_LIM - 25:
        trial += 1
        cc, mcur = contacts_counts(best_x)
        order = np.argsort(-cc)
        sig = sigmas[trial % len(sigmas)]
        cand = best_x.copy()
        r = trial % 4
        if r == 0:
            # global kick
            cand = cand + sig * rng.normal(size=cand.shape)
        elif r == 1:
            # kick overloaded hubs hard, rest small
            k = int(rng.integers(4, 13))
            idx = order[:k] if trial % 2 == 0 else rng.choice(n, size=k, replace=False)
            cand = cand + 0.01 * rng.normal(size=cand.shape)
            cand[idx] += sig * 1.5 * rng.normal(size=(k, d))
        elif r == 2:
            # kick random subset + hole reinsert of worst point before quench
            k = int(rng.integers(3, 10))
            idx = rng.choice(n, size=k, replace=False)
            cand[idx] += sig * rng.normal(size=(k, d))
            # reinsert single worst point to best random hole
            G = cand @ cand.T
            np.fill_diagonal(G, -2)
            w = int(np.argmax(G.max(axis=1)))
            hv, _ = hole_sample(rownorm(cand), w, rng, ns=300)
            cand[w] = hv + 0.02 * rng.normal(size=d)
        else:
            # start partly from random to diversify basin
            if trial % 8 == 7:
                cand = rng.normal(size=cand.shape)
            else:
                cand = cand + sig * rng.normal(size=cand.shape)
        cand = rownorm(cand)
        # two-stage quench: coarse then fine
        cand = quench(cand, steps=350, T0=0.025, T1=0.004, lr0=0.04, lr1=0.006)
        if time.time() - t0 > T_LIM - 25:
            v = max_cos(cand)
            if v < best_v:
                best_x, best_v = cand, v
            break
        cand = quench(cand, steps=350, T0=0.006, T1=0.0015, lr0=0.012, lr1=0.003)
        v = max_cos(cand)
        # one round of worst-point hole retries
        if v < best_v + 0.004:
            G = cand @ cand.T
            np.fill_diagonal(G, -2)
            per_max = G.max(axis=1)
            for _ in range(2):
                w = int(np.argmax(per_max))
                if per_max[w] < v - 1e-4:
                    break
                hv, hd = hole_sample(cand, w, rng, ns=500)
                if hd < per_max[w] - 1e-5:
                    cand[w] = rownorm(hv[None, :])[0]
                    cand = quench(cand, steps=150, T0=0.004, T1=0.0015, lr0=0.008, lr1=0.002)
                    v = max_cos(cand)
                    G = cand @ cand.T
                    np.fill_diagonal(G, -2)
                    per_max = G.max(axis=1)
                else:
                    break
        if v < best_v:
            best_x, best_v = cand.copy(), float(v)
        # early exit if time low
        if time.time() - t0 > T_LIM - 26:
            break

    # greedy hill climb on tangent space (cheap, no linprog)
    t_greedy_end = t0 + T_LIM - 20
    while time.time() < t_greedy_end:
        improved = False
        G = best_x @ best_x.T
        np.fill_diagonal(G, -2)
        m = G.max()
        # pairs achieving max
        pairs = np.argwhere(G > m - 1e-9)
        inv = set(np.unique(pairs[:, 0].tolist()))
        for w in list(inv)[:8]:
            if time.time() > t_greedy_end:
                break
            for _ in range(60):
                step = 0.002 * rng.normal(size=d)
                # tangent
                step = step - (step @ best_x[w]) * best_x[w]
                y = best_x.copy()
                y[w] = rownorm((best_x[w] + step)[None, :])[0]
                v = max_cos(y)
                if v < best_v:
                    best_x, best_v = y, v
                    improved = True
                    break
        if not improved:
            break

    # final LP polish with remaining time
    t_end = t0 + 178.0
    try:
        best_x = lp_polish(best_x, 30, min(t_end, time.time() + 22))
    except Exception:
        pass
    best_x = rownorm(best_x)
    return best_x
