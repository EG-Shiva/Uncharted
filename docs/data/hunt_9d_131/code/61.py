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
    n = np.maximum(n, 1e-12)
    return x / n

def _fast_polish(X, steps, lr0, lr1, m0, m1):
    for s in range(steps):
        f = s / max(1, steps - 1)
        lr = lr0 * ((lr1 / lr0) ** f) if lr0 > 0 and lr1 > 0 else lr0
        margin = m0 * (1 - f) + m1 * f
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        th = m - margin
        Mask = G >= th
        W = np.zeros_like(G)
        if margin > 1e-12:
            W[Mask] = 0.2 + 0.8 * (G[Mask] - th) / margin
        else:
            W[Mask] = 1.0
        np.fill_diagonal(W, 0.0)
        S = W @ X
        deg = np.sum(W, axis=1, keepdims=True)
        dot = np.sum(S * X, axis=1, keepdims=True)
        F = S - dot * X
        F /= np.maximum(1.0, deg)
        X = X - lr * F
        X = _norm(X)
    return X

def _adam_var(V, F, steps=300, lr=0.01):
    Nv = V.shape[0]
    if Nv == 0:
        return V
    m1 = np.zeros_like(V)
    m2 = np.zeros_like(V)
    b1, b2, eps = 0.9, 0.999, 1e-8
    iu0 = iu1 = None
    if Nv > 1:
        iu0, iu1 = np.triu_indices(Nv, 1)
    for t in range(1, steps + 1):
        alpha = 60.0 + (300.0 - 60.0) * (t - 1) / max(1, steps - 1)
        Dvf = V @ F.T
        mvf = float(Dvf.max()) if Dvf.size else -2.0
        if Nv > 1:
            Dvv = V @ V.T
            vals = Dvv[iu0, iu1]
            mvv = float(vals.max()) if vals.size else -2.0
            m = mvf if mvf > mvv else mvv
            Evf = np.exp(alpha * (Dvf - m))
            Evv = np.exp(alpha * (vals - m))
            Ssum = float(Evf.sum() + Evv.sum()) + 1e-300
            Wvf = Evf / Ssum
            Wsym = np.zeros((Nv, Nv))
            w = Evv / Ssum
            Wsym[iu0, iu1] = w
            Wsym[iu1, iu0] = w
            Grad = Wvf @ F + Wsym @ V
        else:
            m = mvf
            Evf = np.exp(alpha * (Dvf - m))
            Ssum = float(Evf.sum()) + 1e-300
            Wvf = Evf / Ssum
            Grad = Wvf @ F
        dot = np.sum(Grad * V, axis=1, keepdims=True)
        Gt = Grad - dot * V
        m1 = b1 * m1 + (1 - b1) * Gt
        m2 = b2 * m2 + (1 - b2) * (Gt * Gt)
        m1h = m1 / (1 - b1 ** t)
        m2h = m2 / (1 - b2 ** t)
        V = V - lr * m1h / (np.sqrt(m2h) + eps)
        V = _norm(V)
    return V

def _lp_polish_deadline(x, deadline):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        c_all = (x @ x.T)[iu]
        m = float(c_all.max())
        sel = c_all > m - max(20 * radius, 1e-9)
        cnt = int(np.sum(sel))
        if cnt == 0:
            break
        if cnt > 5000:
            idx = np.where(sel)[0]
            order = np.argsort(-c_all[idx])[:5000]
            mask = np.zeros_like(sel)
            mask[idx[order]] = True
            sel = mask
        I = iu[0][sel]; J = iu[1][sel]; c = c_all[sel]
        p = len(I); nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
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
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    np.random.seed(0)
    rng = np.random.default_rng(0)
    n, d = 131, 9
    best = None
    best_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.load(fn).astype(float)
            if v.shape == (n, d):
                v = _norm(v)
                c = _max_cos(v)
                if c < best_val:
                    best_val = c
                    best = v.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.standard_normal((n, d)))
        best_val = _max_cos(best)
    incumbent = best.copy()
    inc_val = float(best_val)

    trial = 0
    # trials until t=130s
    while time.time() - t0 < 130:
        trial += 1
        # contact graph at 1e-7
        G = incumbent @ incumbent.T
        np.fill_diagonal(G, -2.0)
        mcur = float(G.max())
        C = G >= (mcur - 1e-7)
        deg = C.sum(axis=1)
        order = np.argsort(deg, kind="stable")
        rattlers = order[:9].copy()
        hubs = np.where(deg >= 22)[0]
        if len(hubs) < 12:
            hubs = order[-max(12, 4):].copy()
        k = int(rng.integers(4, 13))
        k = min(k, len(hubs))
        hub_del = rng.choice(hubs, size=k, replace=False) if k > 0 else np.array([], dtype=int)
        del_set = np.unique(np.concatenate([rattlers, hub_del.astype(int)]))
        # frozen set
        mask = np.ones(n, dtype=bool)
        mask[del_set] = False
        frozen_idx = np.where(mask)[0]
        F = incumbent[frozen_idx].copy()
        Nv = len(del_set)
        S = F.copy()
        new_pts = []
        broken = False
        for _ in range(Nv):
            if time.time() - t0 > 129.0:
                broken = True
                break
            Z = rng.standard_normal((30000, d))
            Z = _norm(Z)
            D = Z @ S.T
            worst = D.max(axis=1)
            part = np.argpartition(worst, 20)[:20]
            Zb = Z[part].copy()
            T = 0.01
            step = 0.02
            for __ in range(100):
                Dots = Zb @ S.T
                mr = Dots.max(axis=1, keepdims=True)
                W = np.exp((Dots - mr) / T)
                W /= (W.sum(axis=1, keepdims=True) + 1e-300)
                Gr = W @ S
                dt = np.sum(Gr * Zb, axis=1, keepdims=True)
                Gr = Gr - dt * Zb
                Zb = Zb - step * Gr
                Zb = _norm(Zb)
            D2 = Zb @ S.T
            w2 = D2.max(axis=1)
            best_one = Zb[int(np.argmin(w2))].copy()
            S = np.vstack([S, best_one[None, :]])
            new_pts.append(best_one)
        if broken or len(new_pts) != Nv:
            continue
        V = np.stack(new_pts, axis=0)
        # optimise only reinserted, frozen fixed, 300 Adam steps alpha 60->300
        try:
            V = _adam_var(V, F, steps=300, lr=0.01)
        except Exception:
            continue
        X = np.vstack([F, V])
        X = _norm(X)
        steps_un = int(rng.integers(600, 1201))
        lr0 = float(rng.uniform(0.012, 0.028))
        lr1 = float(rng.uniform(0.0002, 0.0008))
        m0 = float(rng.uniform(0.02, 0.05))
        m1 = float(rng.uniform(0.0015, 0.004))
        try:
            X = _fast_polish(X, steps_un, lr0, lr1, m0, m1)
        except Exception:
            continue
        v = _max_cos(X)
        if v < inc_val:
            incumbent = X.copy()
            inc_val = float(v)
        if time.time() - t0 > 130:
            break

    # final trust-region LP at most 25s
    remain = 178 - (time.time() - t0)
    if remain > 3:
        lp_time = min(remain - 1, 25)
        deadline = time.time() + lp_time
        try:
            polished = _lp_polish_deadline(incumbent, deadline)
            pv = _max_cos(polished)
            if pv < inc_val:
                incumbent = polished
                inc_val = pv
        except Exception:
            pass
    incumbent = _norm(incumbent)
    return incumbent
