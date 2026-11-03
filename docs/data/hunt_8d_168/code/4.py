import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def _lp_polish(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        try:
            c = (x @ x.T)[iu]
        except Exception:
            break
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        if np.count_nonzero(sel) == 0:
            break
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d),
                                   J[:, None] * d + np.arange(d),
                                   np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(),
                               (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
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
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    PHASE1_END = t0 + 120.0
    rng = np.random.RandomState(0)

    cands = []
    try:
        a = np.load("best.npy")
        cands.append(np.asarray(a, dtype=np.float64))
    except Exception:
        pass
    try:
        b = np.load("record.npy")
        cands.append(np.asarray(b, dtype=np.float64))
    except Exception:
        pass

    best = None
    best_val = np.inf
    for c in cands:
        if c.shape != (168, 8):
            try:
                c = np.reshape(c, (168, 8))
            except Exception:
                continue
        nrm = np.linalg.norm(c, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        c = c / nrm
        v = _max_cos(c)
        if v < best_val:
            best_val = v
            best = c.copy()
    n, d = 168, 8
    if best is None:
        best = rng.randn(n, d)
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)

    # annealed smooth-max descent, updates best in place via list
    def run_anneal(X, steps, tau0, tau1, lr0, lr1):
        nonlocal best, best_val
        l0 = np.log(tau0)
        l1 = np.log(tau1)
        for s in range(steps):
            if time.time() > HARD_END - 35.0:
                break
            frac = s / max(steps - 1, 1)
            tau = float(np.exp(l0 * (1 - frac) + l1 * frac))
            lr = float(lr0 * (1 - frac) + lr1 * frac)
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            if m < best_val:
                best_val = m
                best = (X / np.linalg.norm(X, axis=1, keepdims=True)).copy()
            E = np.exp((G - m) / tau)
            np.fill_diagonal(E, 0.0)
            rs = E.sum(axis=1, keepdims=True)
            EX = E @ X
            Gr = np.zeros_like(X)
            mask = (rs.flatten() > 1e-14)
            Gr[mask] = EX[mask] / rs[mask]
            dot = (Gr * X).sum(axis=1, keepdims=True)
            Gt = Gr - dot * X
            X -= lr * Gt
            nn = np.linalg.norm(X, axis=1, keepdims=True)
            nn[nn == 0] = 1.0
            X /= nn
        # final eval
        try:
            m2 = _max_cos(X)
            if m2 < best_val:
                best_val = m2
                best = X.copy()
        except Exception:
            pass
        return X

    trial = 0
    while time.time() < PHASE1_END and time.time() < HARD_END - 40.0:
        trial += 1
        mode = trial % 6
        if mode == 0:
            X = rng.randn(n, d)
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            X = run_anneal(X, 3000, 0.04, 0.002, 0.06, 0.004)
        elif mode == 1:
            sigma = 0.25
            X = best + sigma * rng.randn(n, d)
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            X = run_anneal(X, 2500, 0.03, 0.002, 0.05, 0.004)
        elif mode == 2:
            sigma = 0.07
            X = best + sigma * rng.randn(n, d)
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            X = run_anneal(X, 2000, 0.025, 0.0018, 0.045, 0.0035)
        elif mode == 3:
            sigma = 0.02
            X = best + sigma * rng.randn(n, d)
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            X = run_anneal(X, 1500, 0.02, 0.0015, 0.04, 0.003)
        elif mode == 4:
            # hole-reinsertion surgery on worst points
            try:
                k = 3 + (trial % 5)  # 3..7
                G = best @ best.T
                np.fill_diagonal(G, -2.0)
                rowmax = G.max(axis=1)
                worst = np.argsort(rowmax)[-k:]
                keep_mask = np.ones(n, dtype=bool)
                keep_mask[worst] = False
                Xkeep = best[keep_mask]
                pool = rng.randn(20000, d)
                pool /= np.linalg.norm(pool, axis=1, keepdims=True)
                Mx = (pool @ Xkeep.T).max(axis=1)
                order = np.argsort(Mx)
                cand = pool[order[:2000]]
                Mx_c = Mx[order[:2000]]
                sel_idx = [0]
                for _ in range(1, k):
                    sel = cand[sel_idx]
                    mc = (cand @ sel.T).max(axis=1)
                    overall = np.maximum(Mx_c, mc)
                    overall[np.array(sel_idx, dtype=int)] = np.inf
                    nxt = int(np.argmin(overall))
                    sel_idx.append(nxt)
                chosen = cand[sel_idx]
                X = best.copy()
                for wi, ci in zip(worst, chosen):
                    X[wi] = ci
                X[worst] += 0.01 * rng.randn(k, d)
                X /= np.linalg.norm(X, axis=1, keepdims=True)
                X = run_anneal(X, 2500, 0.03, 0.0015, 0.05, 0.003)
            except Exception:
                sigma = 0.1
                X = best + sigma * rng.randn(n, d)
                X /= np.linalg.norm(X, axis=1, keepdims=True)
                X = run_anneal(X, 2000, 0.03, 0.002, 0.05, 0.004)
        else:
            sigma = 0.008
            X = best + sigma * rng.randn(n, d)
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            X = run_anneal(X, 1200, 0.015, 0.0012, 0.03, 0.002)

    # squeeze best with narrow anneal
    if time.time() < HARD_END - 32.0:
        try:
            Xs = best.copy()
            run_anneal(Xs, 2500, 0.006, 0.0008, 0.012, 0.0012)
        except Exception:
            pass

    # final trust-region LP polish
    if time.time() < HARD_END - 5.0:
        try:
            best = _lp_polish(best, min(HARD_END - 2.0, time.time() + 30.0))
        except Exception:
            pass

    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
