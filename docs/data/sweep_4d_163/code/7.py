import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def lp_polish_deadline(x, deadline, r0=1e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = r0
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
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
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def smooth_relax(x, steps, a0, a1, lr):
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        alpha = a0 * (a1 / a0) ** (t / steps)
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        gm = float(G.max())
        # avoid overflow
        E = np.exp(alpha * (G - gm))
        np.fill_diagonal(E, 0.0)
        s = E.sum()
        if not np.isfinite(s) or s <= 0:
            break
        W = E / s
        Gr = W @ x
        dot = np.sum(Gr * x, axis=1, keepdims=True)
        Pg = Gr - dot * x
        m = b1 * m + (1 - b1) * Pg
        v = b2 * v + (1 - b2) * (Pg * Pg)
        mhat = m / (1 - b1 ** t)
        vhat = v / (1 - b2 ** t)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        # clip extreme steps
        sn = np.linalg.norm(step, axis=1, keepdims=True)
        cap = 0.05
        too = sn > cap
        step[too[:, 0]] *= (cap / sn[too])
        x = x - step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def find_hole(Xfixed, n_sample, rng):
    d = Xfixed.shape[1]
    best_s = None
    best_w = np.inf
    chunk = 20000
    done = 0
    while done < n_sample:
        cs = min(chunk, n_sample - done)
        S = rng.standard_normal(size=(cs, d))
        S /= np.linalg.norm(S, axis=1, keepdims=True)
        D = S @ Xfixed.T
        w = D.max(axis=1)
        i = int(np.argmin(w))
        if float(w[i]) < best_w:
            best_w = float(w[i])
            best_s = S[i].copy()
        done += cs
    return best_s

def solve():
    t0 = time.time()
    LOOP_END = t0 + 150.0
    HARD_END = t0 + 172.0
    rng = np.random.default_rng(0)
    n, d = 163, 4
    best = None
    best_val = np.inf
    for fname in ("best.npy", "record.npy"):
        try:
            a = np.load(fname)
            if a.shape == (163, 4) and np.all(np.isfinite(a)):
                nr = np.linalg.norm(a, axis=1, keepdims=True)
                nr[nr == 0] = 1.0
                a = a / nr
                va = max_cos(a)
                if va < best_val:
                    best_val = va
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        best = rng.standard_normal(size=(n, d))
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = max_cos(best)
    # ensure we start from tight local optimum if time allows
    best = lp_polish_deadline(best, min(t0 + 8.0, LOOP_END))
    best_val = max_cos(best)

    trial = 0
    while time.time() < LOOP_END:
        trial += 1
        if time.time() + 4.0 > LOOP_END:
            break
        cand = best.copy()
        G = best @ best.T
        np.fill_diagonal(G, -2.0)
        M = float(G.max())
        deg = (G > M - 1e-9).sum(axis=1)
        rattlers = np.where(deg == 0)[0]
        order = np.argsort(-deg)
        mode = trial % 4
        try:
            if mode == 0:
                # relocate rattlers to holes
                if len(rattlers) == 0:
                    rattlers = order[-2:]
                for r in rattlers:
                    others = cand[np.arange(n) != r]
                    h = find_hole(others, 40000, rng)
                    cand[r] = h
                cand = cand / np.linalg.norm(cand, axis=1, keepdims=True)
                cand = smooth_relax(cand, steps=400, a0=60.0, a1=300.0, lr=0.002)
            elif mode == 1:
                target = int(order[0])
                # contacts + near contacts
                row = G[target]
                contacts = np.where(row > M - 3e-3)[0]
                contacts = contacts[contacts != target][:8]
                idx = np.unique(np.concatenate([[target], contacts]))
                sigma = 0.15
                cand[idx] += rng.standard_normal(size=(len(idx), d)) * sigma
                cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                cand = smooth_relax(cand, steps=700, a0=40.0, a1=250.0, lr=0.003)
            elif mode == 2:
                k = 4 + (trial % 4)
                idx = rng.choice(n, size=k, replace=False)
                if len(rattlers) > 0 and (trial % 2 == 0):
                    idx[0] = int(rattlers[0])
                sigma = 0.25 if (trial % 8) < 4 else 0.08
                cand[idx] += rng.standard_normal(size=(len(idx), d)) * sigma
                cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                cand = smooth_relax(cand, steps=600, a0=50.0, a1=300.0, lr=0.003)
            else:
                worst = int(order[0])
                others = cand[np.arange(n) != worst]
                h = find_hole(others, 60000, rng)
                cand[worst] = h
                cand += rng.standard_normal(size=(n, d)) * 0.01
                cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                cand = smooth_relax(cand, steps=600, a0=50.0, a1=250.0, lr=0.003)
        except Exception:
            continue
        # snap to local optimum with short budget
        try:
            cand = lp_polish_deadline(cand, deadline=time.time() + 3.0, r0=2e-3)
            v = max_cos(cand)
        except Exception:
            continue
        if v < best_val - 1e-12 and np.all(np.isfinite(cand)):
            best = cand
            best_val = v
    # final long polish
    try:
        best = lp_polish_deadline(best, deadline=HARD_END, r0=1e-3)
    except Exception:
        pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
