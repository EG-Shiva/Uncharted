import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _pt_max_and_contacts(x, m=None, tol=1e-9):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    if m is None:
        m = float(np.max(G))
    pt_max = np.max(G, axis=1)
    cnt = np.sum(G > m - tol, axis=1)
    return m, pt_max, cnt, G

def _anneal(x, steps, a0, a1, l0, l1):
    n, d = x.shape
    for s in range(steps):
        f = s / (steps - 1) if steps > 1 else 0.0
        alpha = a0 * (a1 / a0) ** f
        lr = l0 + (l1 - l0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        # weights, peak 1 at max
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        rs = np.sum(W, axis=1, keepdims=True) + 1e-30
        S = (W @ x) / rs
        dot = np.sum(S * x, axis=1, keepdims=True)
        Gt = S - dot * x
        nr = np.linalg.norm(Gt, axis=1, keepdims=True)
        Gt = Gt / np.maximum(1.0, nr)
        x = x - lr * Gt
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _lp_polish(x, deadline, max_iter=100, r0=1e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    iu = np.triu_indices(n, 1)
    it = 0
    while time.time() < deadline and radius > 1e-14 and it < max_iter:
        it += 1
        G = x @ x.T
        c = G[iu]
        m = float(np.max(c))
        sel = c > m - max(20.0 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
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
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def solve():
    t0 = time.time()
    t_end = t0 + 170.0
    rng = np.random.default_rng(7)
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.array(np.load(fn), dtype=np.float64)
            if a.shape == (189, 12):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                cands.append((_max_cos(a), a))
        except Exception:
            pass
    if not cands:
        a = rng.standard_normal((189, 12))
        a = a / np.linalg.norm(a, axis=1, keepdims=True)
        cands.append((_max_cos(a), a))
    cands.sort(key=lambda t: t[0])
    best_val, best_x = cands[0][0], cands[0][1].copy()
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    n, d = 189, 12
    trial = 0
    # basin hopping
    while time.time() < t_end - 14.0 and trial < 40:
        if time.time() > t_end - 14.0:
            break
        x = best_x.copy()
        m, pt_max, cnt, G = _pt_max_and_contacts(x, best_val)
        order = np.argsort(pt_max)  # loose first
        mode = trial % 4
        if mode == 0:
            k = 6 + (trial % 4)  # 6..9
            loose = list(order[:k])
            # include exact rattlers first
            ratt = np.where(cnt == 0)[0].tolist()
            for r in ratt:
                if r not in loose:
                    loose.append(r)
            loose = loose[:10]
            M = 8000
            for idx in loose:
                R = rng.standard_normal((M, d))
                R = R / np.linalg.norm(R, axis=1, keepdims=True)
                D = R @ x.T
                D[:, idx] = -2.0
                vals = np.max(D, axis=1)
                j = int(np.argmin(vals))
                x[idx] = R[j]
            # kick hubs
            tight = order[-4:]
            x[tight] += 0.008 * rng.standard_normal((len(tight), d))
            x[tight] /= np.linalg.norm(x[tight], axis=1, keepdims=True)
            steps, a0, a1, l0, l1 = 600, 30.0, 250.0, 0.22, 0.015
        elif mode == 1:
            k = 4 + (trial % 3)
            loose = order[:k]
            M = 8000
            for idx in loose:
                R = rng.standard_normal((M, d))
                R = R / np.linalg.norm(R, axis=1, keepdims=True)
                D = R @ x.T
                D[:, idx] = -2.0
                vals = np.max(D, axis=1)
                j = int(np.argmin(vals))
                x[idx] = R[j]
            # global small jitter
            x += 0.004 * rng.standard_normal((n, d))
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps, a0, a1, l0, l1 = 500, 15.0, 150.0, 0.15, 0.01
        elif mode == 2:
            # medium global shake, no hole fill
            sig = 0.01 + 0.005 * (trial % 3)
            x += sig * rng.standard_normal((n, d))
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps, a0, a1, l0, l1 = 700, 40.0, 300.0, 0.25, 0.02
        else:
            # tight-hub shake + one hole
            tight = order[-6:]
            x[tight] += 0.015 * rng.standard_normal((len(tight), d))
            x[tight] /= np.linalg.norm(x[tight], axis=1, keepdims=True)
            idx = int(order[0])
            M = 12000
            R = rng.standard_normal((M, d))
            R = R / np.linalg.norm(R, axis=1, keepdims=True)
            D = R @ x.T
            D[:, idx] = -2.0
            vals = np.max(D, axis=1)
            j = int(np.argmin(vals))
            x[idx] = R[j]
            steps, a0, a1, l0, l1 = 550, 25.0, 200.0, 0.20, 0.012
        # anneal
        x = _anneal(x, steps, a0, a1, l0, l1)
        v = _max_cos(x)
        # decide to LP polish this trial
        do_lp = False
        if v < best_val - 2e-7:
            do_lp = True
        elif v < best_val + 0.002 and (trial % 3 == 0):
            do_lp = True
        if do_lp:
            if time.time() > t_end - 14.0:
                # no time for LP, just compare raw
                if v < best_val:
                    best_val, best_x = v, x.copy()
            else:
                dl = min(time.time() + 2.5, t_end - 13.0)
                y, vy = _lp_polish(x, dl, max_iter=20, r0=8e-4)
                if vy < best_val:
                    best_val, best_x = vy, y.copy()
        else:
            if v < best_val:
                best_val, best_x = v, x.copy()
        trial += 1
    # final polish of champion with remaining time
    rem = t_end - time.time()
    if rem > 4.0:
        dl = time.time() + min(rem - 0.5, 22.0)
        best_x, best_val = _lp_polish(best_x, dl, max_iter=100, r0=1e-3)
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
