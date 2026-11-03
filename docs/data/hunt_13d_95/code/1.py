import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def gd_refine(X, steps, T0, T1, lr0, lr1, mu=0.9):
    n, d = X.shape
    V = np.zeros_like(X)
    lT0 = np.log(T0)
    lT1 = np.log(T1)
    llr0 = np.log(lr0)
    llr1 = np.log(lr1)
    for s in range(steps):
        f = s / max(1, steps - 1)
        T = float(np.exp(lT0 + (lT1 - lT0) * f))
        lr = float(np.exp(llr0 + (llr1 - llr0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        mx = G.max()
        # softmax weights
        W = np.exp((G - mx) / T)
        np.fill_diagonal(W, 0.0)
        sw = W.sum()
        if not np.isfinite(sw) or sw == 0:
            break
        W /= sw
        Grad = W @ X
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        V = mu * V + Gt
        X = X - lr * V
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def per_point_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)


def hole_candidate(X, idx, rng, n_cand=5000):
    n, d = X.shape
    others = np.delete(X, idx, axis=0)
    C = rng.standard_normal((n_cand, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    dots = C @ others.T
    m = dots.max(axis=1)
    b = int(np.argmin(m))
    return C[b], float(m[b])


def lp_polish(x, seconds, t_deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    deadline = t0 + 172.0
    rng = np.random.default_rng(0)
    np.random.seed(0)
    try:
        Xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            Xb = np.load("record.npy").astype(float)
        except Exception:
            Xb = rng.standard_normal((95, 13))
    if Xb.shape != (95, 13):
        # fallback: tile/reshape
        tmp = rng.standard_normal((95, 13))
        Xb = tmp
    Xb = Xb / np.linalg.norm(Xb, axis=1, keepdims=True)
    best_val = max_cos(Xb)
    best = Xb.copy()
    n, d = best.shape

    # quick hole fix for initial rattlers
    # identify low-contact points
    def contacts_of(X, thr=1e-7):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        C = (G > m - thr).sum(axis=1)
        return C, m

    # main basin-hopping
    search_end = t0 + 128.0
    trial = 0
    # precompute sigmas cycle
    while time.time() < search_end and time.time() < deadline - 35:
        trial += 1
        mode = trial % 6
        Xt = best.copy()
        if mode == 0:
            sig = 0.04
            Xt = Xt + sig * rng.standard_normal(Xt.shape)
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            steps = 700
            Xt = gd_refine(Xt, steps, 0.012, 0.0012, 0.08, 0.008)
        elif mode == 1:
            sig = 0.12
            Xt = Xt + sig * rng.standard_normal(Xt.shape)
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            steps = 700
            Xt = gd_refine(Xt, steps, 0.015, 0.001, 0.10, 0.008)
        elif mode == 2:
            sig = 0.30
            Xt = Xt + sig * rng.standard_normal(Xt.shape)
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            steps = 800
            Xt = gd_refine(Xt, steps, 0.018, 0.001, 0.12, 0.008)
        elif mode == 3:
            k = int(rng.integers(2, 6))
            idx = rng.choice(n, size=k, replace=False)
            Xt[idx] = rng.standard_normal((k, d))
            Xt[idx] /= np.linalg.norm(Xt[idx], axis=1, keepdims=True)
            steps = 700
            Xt = gd_refine(Xt, steps, 0.012, 0.0008, 0.10, 0.006)
        elif mode == 4:
            k = int(rng.integers(6, 12))
            idx = rng.choice(n, size=k, replace=False)
            Xt[idx] = rng.standard_normal((k, d))
            Xt[idx] /= np.linalg.norm(Xt[idx], axis=1, keepdims=True)
            steps = 800
            Xt = gd_refine(Xt, steps, 0.015, 0.001, 0.12, 0.008)
        else:
            # targeted hole reinsertion: pick worst points
            ppm = per_point_max(Xt)
            order = np.argsort(-ppm)
            k = int(rng.integers(1, 4))
            for j in range(k):
                idx = int(order[j if j < len(order) else 0])
                cand, _ = hole_candidate(Xt, idx, rng, n_cand=8000)
                # local hill climb on cand
                cur_best = cand.copy()
                others = np.delete(Xt, idx, axis=0)
                cur_val = float((cur_best @ others.T).max())
                step_s = 0.05
                for _ in range(150):
                    prop = cur_best + step_s * rng.standard_normal(d)
                    prop /= np.linalg.norm(prop)
                    pv = float((prop @ others.T).max())
                    if pv < cur_val:
                        cur_val = pv
                        cur_best = prop
                    else:
                        step_s *= 0.97
                Xt[idx] = cur_best
            steps = 700
            Xt = gd_refine(Xt, steps, 0.010, 0.0008, 0.08, 0.006)

        v = max_cos(Xt)
        if v < best_val:
            best_val = v
            best = Xt.copy()
            # intensify immediately with colder polish
            if time.time() < search_end - 5:
                Xt2 = gd_refine(best.copy(), 600, 0.005, 0.0004, 0.04, 0.004)
                v2 = max_cos(Xt2)
                if v2 < best_val:
                    best_val = v2
                    best = Xt2.copy()
        # occasional time check
        if trial % 20 == 0 and time.time() >= search_end:
            break

    # final squeeze: longer cold refine then LP
    remain = deadline - time.time() - 2.0
    if remain > 5:
        # cold smooth polish
        cold_time = min(12.0, remain * 0.35)
        c_start = time.time()
        # two-stage cold
        best = gd_refine(best, 1200, 0.006, 0.0005, 0.05, 0.005)
        if time.time() - c_start < cold_time - 2:
            best = gd_refine(best, 800, 0.002, 0.0002, 0.02, 0.003)
        best_val = max_cos(best)
        remain2 = deadline - time.time() - 1.0
        if remain2 > 3:
            best = lp_polish(best, seconds=min(28.0, remain2), t_deadline=deadline - 1.0)
            best = best / np.linalg.norm(best, axis=1, keepdims=True)

    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
