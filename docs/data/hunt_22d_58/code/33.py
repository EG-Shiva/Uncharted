import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def _quench(X, steps, b0, b1, lr0, lr1, mu=0.85, clip=0.04):
    n, d = X.shape
    V = np.zeros_like(X)
    lb0, lb1 = np.log(b0), np.log(b1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        f = s / max(1, steps - 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        sw = float(W.sum())
        if not np.isfinite(sw) or sw <= 0:
            break
        W /= sw
        Gr = W @ X
        wd = np.sum(W * G, axis=1, keepdims=True)
        Gt = Gr - wd * X
        V = mu * V + Gt
        step = lr * V
        sdot = np.sum(step * X, axis=1, keepdims=True)
        step = step - sdot * X
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        step = np.where(sn > clip, step * (clip / sn), step)
        X = X - step
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X = X / nr
    return X


def _hole_candidates(X, victims, rng, n_cand=3000):
    n, d = X.shape
    Y = X.copy()
    for v in victims:
        mask = np.ones(n, dtype=bool)
        mask[v] = False
        # also mask other victims not yet placed? keep them to avoid clustering
        R = Y[mask]
        C = rng.standard_normal((n_cand, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-12
        dots = C @ R.T
        worst = dots.max(axis=1)
        bi = int(np.argmin(worst))
        # local refine around best candidate
        best_c = C[bi]
        best_w = float(worst[bi])
        for _ in range(2):
            Cc = best_c[None, :] + 0.05 * rng.standard_normal((512, d))
            Cc /= np.linalg.norm(Cc, axis=1, keepdims=True) + 1e-12
            w2 = (Cc @ R.T).max(axis=1)
            j = int(np.argmin(w2))
            if float(w2[j]) < best_w:
                best_w = float(w2[j])
                best_c = Cc[j]
        Y[v] = best_c
    return Y


def _lp_finish(X, seconds, deadline):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = 5e-4
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    niter = 0
    while True:
        if time.time() - start >= seconds:
            break
        if time.time() >= deadline:
            break
        if radius < 1e-13:
            break
        niter += 1
        if niter > 15:
            break
        G = X @ X.T
        c = G[iu0]
        m = float(c.max())
        window = max(20.0 * radius, 4e-5)
        order = np.argsort(-c)
        sel_list = []
        for idx in order:
            if c[idx] > m - window and len(sel_list) < 500:
                sel_list.append(int(idx))
            elif c[idx] <= m - window:
                break
            if len(sel_list) >= 500:
                break
        if len(sel_list) == 0:
            break
        sel = np.array(sel_list)
        I = iu0[0][sel]
        J = iu0[1][sel]
        p = len(sel)
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                          options={"time_limit": 2.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best - 1e-15:
            X = y
            best = v
            radius = min(radius * 1.5, 0.01)
        else:
            radius *= 0.5
    return X


def solve():
    t0 = time.time()
    DEADLINE = t0 + 168.0
    SEARCH_END = t0 + 130.0
    rng = np.random.default_rng(12345)
    n, d = 58, 22

    best = None
    best_val = 1e9
    for f in ("best.npy", "record.npy"):
        try:
            v = np.load(f).astype(float)
            if v.shape != (58, 22):
                continue
            v = v / np.linalg.norm(v, axis=1, keepdims=True)
            c = _max_cos(v)
            if c < best_val:
                best_val = c
                best = v.copy()
        except Exception:
            continue
    if best is None:
        best = rng.standard_normal((n, d))
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)

    # fast numpy-only basin hopping
    n_hops = 900
    for it in range(n_hops):
        if time.time() >= SEARCH_END:
            break
        if time.time() >= DEADLINE - 25.0:
            break
        G = best @ best.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        # degrees for biased victim choice
        deg = (G > m - 0.004).sum(axis=1)
        k = int(np.argmax(G)) if rng.random() < 0.7 else -1
        if k >= 0:
            i0 = k // n
            j0 = k % n
        else:
            i0 = int(rng.integers(n))
            j0 = int(rng.integers(n))
        order_deg = np.argsort(-deg)

        mode = it % 4
        X = best.copy()
        try:
            if mode == 0:
                victim = [int(i0) if rng.random() < 0.5 else int(rng.integers(n))]
                # bias to overloaded hub occasionally
                if rng.random() < 0.5:
                    victim = [int(order_deg[0])]
                X = _hole_candidates(X, victim, rng, n_cand=2500)
                X = _quench(X, 320, 15.0, 350.0, 0.6, 0.02, mu=0.85, clip=0.04)
            elif mode == 1:
                v1 = int(i0)
                v2 = int(j0)
                if rng.random() < 0.5:
                    v1 = int(order_deg[int(rng.integers(min(n, 6)))])
                    v2 = int(order_deg[int(rng.integers(min(n, 6)))])
                    if v2 == v1:
                        v2 = (v2 + 1) % n
                X = _hole_candidates(X, [v1, v2], rng, n_cand=2200)
                X = _quench(X, 350, 20.0, 500.0, 0.5, 0.015, mu=0.87, clip=0.04)
            elif mode == 2:
                kk = int(rng.integers(3, 8))
                # prefer high-degree points
                probs = deg.astype(float) + 1.0
                probs /= probs.sum()
                victims = rng.choice(n, size=kk, replace=False, p=probs)
                sig = 0.22 + 0.12 * rng.random()
                noise = rng.standard_normal((kk, d)) * sig
                X[victims] = X[victims] + noise
                X[victims] /= np.linalg.norm(X[victims], axis=1, keepdims=True) + 1e-12
                X = _quench(X, 300, 12.0, 300.0, 0.7, 0.02, mu=0.82, clip=0.05)
            else:
                sig = 0.02 + 0.03 * rng.random()
                X = X + sig * rng.standard_normal(X.shape)
                X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
                X = _quench(X, 280, 15.0, 400.0, 0.4, 0.015, mu=0.85, clip=0.03)
        except Exception:
            continue
        try:
            v = _max_cos(X)
        except Exception:
            continue
        if v < best_val - 1e-12:
            best = X
            best_val = v

    # final squeeze: high-beta gradient polish
    try:
        if time.time() < DEADLINE - 20.0:
            best = _quench(best, 600, 150.0, 1500.0, 0.12, 0.004, mu=0.9, clip=0.012)
            best_val = _max_cos(best)
    except Exception:
        pass
    # one capped LP finish
    try:
        if time.time() < DEADLINE - 12.0:
            remain = min(10.0, DEADLINE - 3.0 - time.time())
            if remain > 1.0:
                best = _lp_finish(best, remain, DEADLINE - 1.0)
    except Exception:
        pass

    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
