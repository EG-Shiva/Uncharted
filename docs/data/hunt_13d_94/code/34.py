import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _adam_full(X, steps, k0, k1, lr0, lr1, b1=0.9, b2=0.999, eps=1e-8):
    n, d = X.shape
    m1 = np.zeros_like(X)
    v1 = np.zeros_like(X)
    for s in range(steps):
        frac = s / max(1, steps - 1)
        k = k0 * (k1 / k0) ** frac if k1 != k0 else k0
        lr = lr0 * (lr1 / lr0) ** frac if lr1 != lr0 else lr0
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        mval = float(G.max())
        W = np.exp(k * (G - mval))
        np.fill_diagonal(W, 0.0)
        S = W.sum()
        if S == 0 or not np.isfinite(S):
            break
        Gr = (W @ X) / S
        dot = (Gr * X).sum(axis=1, keepdims=True)
        Gt = Gr - dot * X
        m1 = b1 * m1 + (1 - b1) * Gt
        v1 = b2 * v1 + (1 - b2) * (Gt * Gt)
        bc1 = 1.0 - b1 ** (s + 1)
        bc2 = 1.0 - b2 ** (s + 1)
        upd = lr * (m1 / bc1) / (np.sqrt(v1 / bc2) + eps)
        X = X - upd
        X = _norm(X)
        if not np.all(np.isfinite(X)):
            break
    return X

def _adam_patch(P, O, steps, k0, k1, lr0, lr1, b1=0.9, b2=0.999, eps=1e-8):
    m = P.shape[0]
    m1 = np.zeros_like(P)
    v1 = np.zeros_like(P)
    for s in range(steps):
        frac = s / max(1, steps - 1)
        k = k0 * (k1 / k0) ** frac if k1 != k0 else k0
        lr = lr0 * (lr1 / lr0) ** frac if lr1 != lr0 else lr0
        Dop = P @ O.T
        Dpp = P @ P.T
        mval = float(max(Dop.max(), (Dpp - np.eye(m) * 2.0).max()))
        Wop = np.exp(k * (Dop - mval))
        Wpp = np.exp(k * (Dpp - mval))
        np.fill_diagonal(Wpp, 0.0)
        S = Wop.sum() + Wpp.sum()
        if S == 0 or not np.isfinite(S):
            break
        Gr = (Wop @ O + Wpp @ P) / S
        dot = (Gr * P).sum(axis=1, keepdims=True)
        Gt = Gr - dot * P
        m1 = b1 * m1 + (1 - b1) * Gt
        v1 = b2 * v1 + (1 - b2) * (Gt * Gt)
        bc1 = 1.0 - b1 ** (s + 1)
        bc2 = 1.0 - b2 ** (s + 1)
        upd = lr * (m1 / bc1) / (np.sqrt(v1 / bc2) + eps)
        P = P - upd
        P = _norm(P)
        if not np.all(np.isfinite(P)):
            break
    return P

def _patch_val(P, O):
    a = float((P @ O.T).max()) if O.shape[0] > 0 else -2.0
    G = P @ P.T
    np.fill_diagonal(G, -2.0)
    b = float(G.max()) if P.shape[0] > 1 else -2.0
    return a if a > b else b

def _lp_polish(x, deadline):
    x = _norm(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
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
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    T0 = time.time()
    BUDGET = 170.0
    HOP_END = T0 + 138.0
    LP_END = T0 + 178.0
    rng = np.random.default_rng(0)

    best = None
    best_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn).astype(float)
            a = _norm(a)
            if a.shape == (94, 13) and np.all(np.isfinite(a)):
                v = _max_cos(a)
                if v < best_val:
                    best_val = v
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.standard_normal(size=(94, 13)))
        best_val = _max_cos(best)
    n, d = best.shape
    cur = best.copy()
    cur_val = best_val

    def worst_per_point(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        return G.max(axis=1), G

    trial = 0
    while time.time() < HOP_END:
        trial += 1
        elapsed = time.time() - T0
        # annealing temperature
        T = 0.0025 * max(0.05, 1.0 - elapsed / 140.0) + 0.00015
        wpp, Gall = worst_per_point(cur)
        order_worst = np.argsort(-wpp)
        cnt = (Gall > Gall.max() - 1e-3).sum(axis=1)
        order_loose = np.argsort(cnt)

        r = rng.random()
        # choose patch size and centre
        if r < 0.7:
            m = int(rng.integers(5, 13))  # 5..12
        elif r < 0.9:
            m = int(rng.integers(3, 6))
        else:
            m = int(rng.integers(12, 20))
        u = rng.random()
        if u < 0.45:
            centre = int(order_worst[int(rng.integers(0, 10))])
        elif u < 0.7:
            centre = int(order_loose[int(rng.integers(0, 10))])
        else:
            centre = int(rng.integers(0, n))
        # neighbours by dot (closest points)
        dots_c = Gall[centre].copy()
        dots_c[centre] = -10.0
        # Gall has -2 on diag, fix centre excluded; nearest = largest dots
        neigh = np.argsort(-dots_c)[: max(0, m - 1)]
        rem = np.zeros(n, dtype=bool)
        rem[centre] = True
        rem[neigh] = True
        # if m larger than unique, add random
        while rem.sum() < m:
            rem[int(rng.integers(0, n))] = True
        idx_rem = np.where(rem)[0]
        idx_keep = np.where(~rem)[0]
        O = cur[idx_keep].copy()

        # multi-start patch optimisation (keep O fixed)
        bestP = None
        bestPv = 1e9
        n_rest = 2
        # vary sharpness per restart
        for rs in range(n_rest):
            if time.time() > HOP_END:
                break
            P0 = _norm(rng.standard_normal(size=(m, d)))
            # seed one restart from far random best-of-many if first restart
            if rs == 0 and m <= 12:
                # quick screening: 400 random P sets, keep best coarse
                cand_best = 1e9
                cand_P = P0
                for _ in range(60):
                    Pc = _norm(rng.standard_normal(size=(m, d)))
                    v = _patch_val(Pc, O)
                    if v < cand_best:
                        cand_best = v
                        cand_P = Pc
                P0 = cand_P
            k0 = float(rng.choice([40.0, 60.0, 80.0]))
            k1 = float(k0 * rng.choice([1.5, 2.0, 2.5]))
            P1 = _adam_patch(P0, O, steps=700, k0=k0, k1=k1, lr0=0.06, lr1=0.01)
            v = _patch_val(P1, O)
            if v < bestPv:
                bestPv = v
                bestP = P1.copy()
        if bestP is None:
            continue
        Xtry = np.empty_like(cur)
        Xtry[idx_keep] = O
        Xtry[idx_rem] = bestP
        # joint relaxation of all points with moderate sharpness
        kk0 = float(rng.choice([30.0, 50.0, 70.0]))
        kk1 = float(kk0 * rng.choice([1.5, 2.0, 3.0]))
        lr0 = float(rng.choice([0.02, 0.03, 0.05]))
        Xtry = _adam_full(Xtry, steps=700, k0=kk0, k1=kk1, lr0=lr0, lr1=lr0 * 0.3)
        # occasional extra sharper short burst
        if rng.random() < 0.4 and time.time() < HOP_END:
            Xtry = _adam_full(Xtry, steps=250, k0=150.0, k1=250.0, lr0=0.01, lr1=0.003)
        new_val = _max_cos(Xtry)
        if new_val < cur_val or rng.random() < np.exp((cur_val - new_val) / max(T, 1e-6)):
            cur = Xtry
            cur_val = new_val
            if new_val < best_val:
                best_val = new_val
                best = Xtry.copy()
        # also occasional scattered global shake trial (no patch contiguity)
        if trial % 9 == 0 and time.time() < HOP_END:
            Xs = cur.copy()
            k = int(rng.integers(8, 25))
            victims = rng.choice(n, size=k, replace=False)
            Xs[victims] += 0.25 * rng.standard_normal(size=(k, d))
            Xs = _norm(Xs)
            Xs = _adam_full(Xs, steps=800, k0=40.0, k1=120.0, lr0=0.03, lr1=0.008)
            v = _max_cos(Xs)
            if v < cur_val or rng.random() < np.exp((cur_val - v) / max(T, 1e-6)):
                cur, cur_val = Xs, v
                if v < best_val:
                    best_val, best = v, Xs.copy()

    # final polish of the best found
    try:
        best = _lp_polish(best, LP_END)
        best_val = _max_cos(best)
    except Exception:
        pass
    # tiny shrink guard + renormalize
    best = _norm(best)
    return best
