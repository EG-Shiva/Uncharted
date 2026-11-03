import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def anneal_smooth(X, steps, T0, T1, lr0, lr1, mu=0.80):
    n, d = X.shape
    V = np.zeros_like(X)
    lT0 = np.log(T0)
    lT1 = np.log(T1)
    ll0 = np.log(lr0)
    ll1 = np.log(lr1)
    for s in range(steps):
        f = s / max(1, steps - 1)
        T = float(np.exp(lT0 + (lT1 - lT0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        mx = float(G.max())
        # per-row softmax weights, focus on near-max
        # shift by mx for stability; values < mx-12T underflow to 0 anyway
        W = np.exp((G - mx) / T)
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-300
        Grad = (W @ X) / rs
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        # clip extreme tangential norms for stability
        nm = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-300
        clip = np.minimum(1.0, 0.5 / (nm + 1e-300))
        # actually allow up to 1.0; scale only if huge
        over = nm > 2.0
        if np.any(over):
            Gt = np.where(over, Gt / nm * 2.0, Gt)
        V = mu * V + Gt
        X = X - lr * V
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def per_point_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)


def hole_candidate(X, idx, rng, n_cand=8000):
    others = np.delete(X, idx, axis=0)
    C = rng.standard_normal((n_cand, X.shape[1]))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    dots = C @ others.T
    m = dots.max(axis=1)
    b = int(np.argmin(m))
    return C[b], float(m[b])


def lp_polish(x, seconds, t_deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    best_x = x.copy()
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if t_deadline is not None and time.time() > t_deadline:
            break
        if radius < 1e-14:
            break
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        # cap constraints for speed
        if p > 2500:
            # keep largest
            order = np.argsort(-c[sel])[:2500]
            I, J = I[order], J[order]
            p = 2500
            csel = c[sel][order]
        else:
            csel = c[sel]
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
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
            best = v
            best_x = y.copy()
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
            # keep x at best to avoid drift
            if max_cos(x) > best + 1e-12:
                x = best_x.copy()
    return best_x


def solve():
    t0 = time.time()
    deadline = t0 + 172.0
    rng = np.random.default_rng(0)
    try:
        Xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            Xb = np.load("record.npy").astype(float)
        except Exception:
            Xb = rng.standard_normal((95, 13))
    if Xb.shape != (95, 13):
        Xb = rng.standard_normal((95, 13))
    Xb = Xb / np.linalg.norm(Xb, axis=1, keepdims=True)
    best_val = max_cos(Xb)
    best = Xb.copy()
    n, d = best.shape

    search_end = t0 + 126.0
    trial = 0
    # quick evaluation helper
    while time.time() < search_end and time.time() < deadline - 40:
        trial += 1
        mode = trial % 7
        Xt = best.copy()
        if mode == 0:
            sig = 0.03
            Xt = Xt + sig * rng.standard_normal(Xt.shape)
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = anneal_smooth(Xt, 700, 0.010, 0.0010, 0.06, 0.006)
        elif mode == 1:
            sig = 0.10
            Xt = Xt + sig * rng.standard_normal(Xt.shape)
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = anneal_smooth(Xt, 800, 0.014, 0.0009, 0.08, 0.006)
        elif mode == 2:
            sig = 0.25
            Xt = Xt + sig * rng.standard_normal(Xt.shape)
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = anneal_smooth(Xt, 900, 0.016, 0.0008, 0.09, 0.006)
        elif mode == 3:
            k = int(rng.integers(2, 6))
            idx = rng.choice(n, size=k, replace=False)
            Xt[idx] = rng.standard_normal((k, d))
            Xt[idx] /= np.linalg.norm(Xt[idx], axis=1, keepdims=True)
            Xt = anneal_smooth(Xt, 800, 0.011, 0.0007, 0.08, 0.005)
        elif mode == 4:
            k = int(rng.integers(6, 14))
            idx = rng.choice(n, size=k, replace=False)
            Xt[idx] = rng.standard_normal((k, d))
            Xt[idx] /= np.linalg.norm(Xt[idx], axis=1, keepdims=True)
            Xt = anneal_smooth(Xt, 900, 0.013, 0.0007, 0.09, 0.005)
        elif mode == 5:
            # worst-point hole reinsertion
            ppm = per_point_max(Xt)
            order = np.argsort(-ppm)
            k = int(rng.integers(1, 4))
            for j in range(k):
                idx = int(order[j])
                cand, _ = hole_candidate(Xt, idx, rng, n_cand=8000)
                others = np.delete(Xt, idx, axis=0)
                cur_best = cand.copy()
                cur_val = float((cur_best @ others.T).max())
                ss = 0.04
                for _ in range(120):
                    prop = cur_best + ss * rng.standard_normal(d)
                    prop /= np.linalg.norm(prop)
                    pv = float((prop @ others.T).max())
                    if pv < cur_val:
                        cur_val = pv
                        cur_best = prop
                    else:
                        ss *= 0.97
                Xt[idx] = cur_best
            Xt = anneal_smooth(Xt, 800, 0.009, 0.0006, 0.07, 0.005)
        else:
            # rattler / random-point teleport to largest hole
            ppm = per_point_max(Xt)
            # pick a random mid point to relocate
            idx = int(rng.integers(0, n))
            cand, _ = hole_candidate(Xt, idx, rng, n_cand=12000)
            Xt[idx] = cand
            Xt = anneal_smooth(Xt, 700, 0.008, 0.0006, 0.06, 0.005)

        v = max_cos(Xt)
        if v < best_val:
            # cold tie-down to fairly evaluate basin
            Xt2 = anneal_smooth(Xt.copy(), 500, 0.003, 0.00025, 0.03, 0.003, mu=0.7)
            v2 = max_cos(Xt2)
            if v2 < v:
                Xt, v = Xt2, v2
            if v < best_val:
                best_val = v
                best = Xt.copy()
        if trial % 10 == 0:
            if time.time() >= search_end:
                break

    remain = deadline - time.time() - 1.5
    if remain > 6:
        # staged cold anneal then LP
        cold_budget = min(14.0, remain * 0.35)
        c0 = time.time()
        best = anneal_smooth(best, 1200, 0.005, 0.0004, 0.04, 0.004, mu=0.7)
        if time.time() - c0 < cold_budget - 3:
            best = anneal_smooth(best, 900, 0.0015, 0.00015, 0.015, 0.002, mu=0.6)
        best_val = max_cos(best)
        remain2 = deadline - time.time() - 1.0
        if remain2 > 3:
            best = lp_polish(best, seconds=min(30.0, remain2), t_deadline=deadline - 0.8)
            best = best / np.linalg.norm(best, axis=1, keepdims=True)

    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
