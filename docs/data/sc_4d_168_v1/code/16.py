import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))


def _norm(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def _refine(X, steps, mu, beta0, beta1, lr0, lr1, deadline=1e30):
    x = X
    vel = np.zeros_like(x)
    for s in range(steps):
        if s % 200 == 0 and time.time() > deadline:
            break
        frac = s / max(1, steps - 1)
        beta = beta0 * (beta1 / beta0) ** frac
        lr = lr0 * (1.0 - frac) + lr1 * frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        cmax = float(np.max(G))
        E = np.exp(np.clip(beta * (G - cmax), -20.0, 0.0))
        np.fill_diagonal(E, 0.0)
        grad = E @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        vel = mu * vel + (1.0 - mu) * gt
        x = x - lr * vel
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def _squeeze(X, steps, step0, deadline=1e30):
    X = X.copy()
    best = _max_cos(X)
    Xb = X.copy()
    step = step0
    n = X.shape[0]
    for s in range(steps):
        if time.time() > deadline:
            break
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        window = max(12.0 * step, 3e-4)
        Mask = G > (m - window)
        cnt = np.sum(Mask, axis=1, keepdims=True) + 1e-12
        Grad = (Mask.astype(float) @ X) / cnt
        d = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - d * X
        # normalise step scale
        gn = np.linalg.norm(Gt) + 1e-30
        Y = X - step * Gt
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        v = _max_cos(Y)
        if v < best:
            best = v
            X = Y
            Xb = Y.copy()
            step = min(step * 1.25, 5e-3)
        else:
            step *= 0.55
            if step < 1e-12:
                step = step0 * 0.5
                # kick worst point
                deg = np.sum(Mask, axis=1)
                idx = int(np.argmax(deg))
                r = np.random.randn(4)
                r -= np.dot(r, X[idx]) * X[idx]
                r /= np.linalg.norm(r) + 1e-30
                X2 = X.copy()
                X2[idx] = _norm((X[idx] + 0.002 * r)[None, :])[0]
                v2 = _max_cos(X2)
                if v2 < best:
                    best = v2
                    X = X2
                    Xb = X2.copy()
    return Xb, best


def _lp_polish(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(np.max(c))
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            break
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    HARD_END = t0 + 168.0
    SEARCH_END = t0 + 136.0
    np.random.seed(0)
    rng = np.random.RandomState(0)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.randn(168, 4)
    xb = _norm(xb)
    try:
        xr = np.load("record.npy").astype(float)
        xr = _norm(xr)
        if _max_cos(xr) < _max_cos(xb):
            xb = xr
    except Exception:
        pass
    n, d = 168, 4
    if xb.shape != (168, 4):
        xb = _norm(rng.randn(168, 4))
    best_val = _max_cos(xb)
    x_best = xb.copy()
    elites = [x_best.copy()]
    elite_vals = [best_val]

    def best_hole(X_keep, n_sample):
        S = rng.randn(n_sample, 4)
        S /= np.linalg.norm(S, axis=1, keepdims=True)
        D = S @ X_keep.T
        m = D.max(axis=1)
        return S[np.argmin(m)]

    def thin_pool(A, B):
        pool = np.vstack([A, B])  # 336x4
        # greedy farthest insertion
        order = [rng.randint(0, len(pool))]
        # precompute pool gram
        Gp = pool @ pool.T
        # min dot to selected set
        best_dot = Gp[:, order[0]].copy()
        # mask selected
        used = np.zeros(len(pool), dtype=bool)
        used[order[0]] = True
        for _ in range(168 - 1):
            best_dot[used] = 2.0
            nxt = int(np.argmin(best_dot))
            order.append(nxt)
            used[nxt] = True
            best_dot = np.minimum(best_dot, Gp[:, nxt])
        return pool[np.array(order)]

    def plane_rotation(theta):
        u = rng.randn(4); u /= np.linalg.norm(u)
        v = rng.randn(4); v -= np.dot(v, u) * u; v /= np.linalg.norm(v)
        Q = np.eye(4) + (np.cos(theta) - 1.0) * (np.outer(u, u) + np.outer(v, v)) + np.sin(theta) * (np.outer(v, u) - np.outer(u, v))
        return Q

    trial = 0
    while time.time() < SEARCH_END:
        trial += 1
        if time.time() > SEARCH_END - 1.5:
            break
        # pick base: mostly best, sometimes elite
        if len(elites) > 1 and (trial % 10 == 9):
            base = elites[rng.randint(0, len(elites))].copy()
        else:
            base = x_best.copy()
        x = base
        typ = trial % 8
        if typ == 0:
            x = _norm(x + 0.020 * rng.randn(n, d))
            steps, mu, b0, b1, l0, l1 = 650, 0.9, 30.0, 400.0, 0.012, 0.0008
        elif typ == 1:
            x = _norm(x + 0.060 * rng.randn(n, d))
            steps, mu, b0, b1, l0, l1 = 950, 0.9, 25.0, 500.0, 0.014, 0.0007
        elif typ == 2:
            idx = rng.choice(n, 25, replace=False)
            x[idx] = x[idx] + 0.25 * rng.randn(len(idx), d)
            x = _norm(x)
            steps, mu, b0, b1, l0, l1 = 950, 0.88, 30.0, 450.0, 0.013, 0.0007
        elif typ == 3:
            G = x @ x.T; np.fill_diagonal(G, -2.0)
            cmax = float(np.max(G))
            deg = np.sum(G > cmax - 0.010, axis=1)
            order = np.argsort(-deg, kind="stable")
            i0 = order[rng.randint(0, 10) % n]
            nbrs = np.argsort(-G[i0], kind="stable")[:4]
            ksz = 4 + (trial % 3)  # 4..6
            cluster = np.concatenate([[i0], nbrs])[:ksz]
            keep = np.ones(n, dtype=bool); keep[cluster] = False
            cur = x[keep]
            new_pts = []
            for _ in range(len(cluster)):
                h = best_hole(cur, n_sample=14000)
                new_pts.append(h)
                cur = np.vstack([cur, h[None, :]])
            x[cluster] = np.array(new_pts)
            # tiny jitter all
            x = _norm(x + 0.008 * rng.randn(n, d))
            steps, mu, b0, b1, l0, l1 = 1100, 0.9, 30.0, 500.0, 0.012, 0.0006
        elif typ == 4:
            G = x @ x.T; np.fill_diagonal(G, -2.0)
            cmax = float(np.max(G))
            deg = np.sum(G > cmax - 1e-4, axis=1)
            loose = np.where(deg <= 4)[0]
            if len(loose) == 0:
                loose = np.argsort(deg)[:4]
            li = loose[rng.randint(0, len(loose))]
            X_keep = np.delete(x, li, axis=0)
            h = best_hole(X_keep, n_sample=18000)
            x[li] = h
            G2 = x @ x.T; np.fill_diagonal(G2, -2.0)
            nbr = np.argsort(-G2[li], kind="stable")[:6]
            x[nbr] = x[nbr] + 0.06 * rng.randn(len(nbr), d)
            x = _norm(x)
            steps, mu, b0, b1, l0, l1 = 900, 0.9, 35.0, 550.0, 0.011, 0.0006
        elif typ == 5:
            # worst-pair split
            G = x @ x.T; np.fill_diagonal(G, -2.0)
            # top 4 worst pairs
            iu = np.triu_indices(n, 1)
            c = G[iu]
            top = np.argsort(-c)[:4]
            for t in top:
                i, j = int(iu[0][t]), int(iu[1][t])
                # push apart tangentially
                s = 0.08 + 0.04 * rng.rand()
                vi = x[j] - np.dot(x[j], x[i]) * x[i]
                vj = x[i] - np.dot(x[i], x[j]) * x[j]
                x[i] = x[i] - s * vi + 0.02 * rng.randn(d)
                x[j] = x[j] - s * vj + 0.02 * rng.randn(d)
            x = _norm(x)
            # jitter neighbours of worst point
            G2 = x @ x.T; np.fill_diagonal(G2, -2.0)
            cm = float(np.max(G2))
            deg = np.sum(G2 > cm - 0.008, axis=1)
            hi = int(np.argmax(deg))
            nbr = np.argsort(-G2[hi], kind="stable")[:6]
            x[nbr] = x[nbr] + 0.05 * rng.randn(len(nbr), d)
            x = _norm(x)
            steps, mu, b0, b1, l0, l1 = 1000, 0.89, 28.0, 480.0, 0.013, 0.0007
        elif typ == 6:
            # coherent subset rotation
            k = rng.randint(12, 30)
            idx = rng.choice(n, k, replace=False)
            th = rng.uniform(0.10, 0.35) * (1 if rng.rand() < 0.5 else -1)
            Q = plane_rotation(th)
            x[idx] = (Q @ x[idx].T).T
            x = _norm(x + 0.015 * rng.randn(n, d))
            steps, mu, b0, b1, l0, l1 = 1000, 0.9, 30.0, 450.0, 0.012, 0.0007
        else:
            # pool recombination or fresh
            if len(elites) >= 2 and (trial % 16 != 7):
                A = x_best
                B = elites[rng.randint(0, len(elites))]
                # avoid identical
                if np.allclose(A, B):
                    try:
                        B = _norm(np.load("record.npy").astype(float))
                    except Exception:
                        B = _norm(rng.randn(n, d))
                x = thin_pool(A, B)
                x = _norm(x + 0.015 * rng.randn(n, d))
                steps, mu, b0, b1, l0, l1 = 1200, 0.9, 25.0, 500.0, 0.013, 0.0006
            else:
                x = _norm(rng.randn(n, d))
                steps, mu, b0, b1, l0, l1 = 1700, 0.9, 20.0, 400.0, 0.015, 0.0008

        x = _refine(x, steps, mu, b0, b1, l0, l1, deadline=SEARCH_END)
        v = _max_cos(x)
        # fast squeeze to reveal true basin depth if promising
        if v < best_val + 0.002 and time.time() < SEARCH_END - 1.0:
            xs, vs = _squeeze(x, steps=250, step0=2.5e-3, deadline=SEARCH_END)
            if vs < v:
                x, v = xs, vs
        if v < best_val - 5e-13:
            best_val = v
            x_best = x.copy()
            elites.append(x.copy()); elite_vals.append(v)
            # keep top 6
            if len(elites) > 6:
                o = np.argsort(elite_vals)
                elites = [elites[i] for i in o[:6]]
                elite_vals = [elite_vals[i] for i in o[:6]]
        elif v < (max(elite_vals) if elite_vals else 1e9) - 2e-4 and len(elites) < 6:
            # diverse decent basin
            elites.append(x.copy()); elite_vals.append(v)

    # final tighten + LP polish
    deadline = min(HARD_END - 2.0, time.time() + 26.0)
    if deadline > time.time() + 1.0:
        try:
            x_best = _refine(x_best, 350, 0.9, 150.0, 800.0, 0.004, 0.0003, deadline=deadline)
        except Exception:
            pass
        v = _max_cos(x_best)
        if v < best_val:
            best_val = v
        # short squeeze
        if deadline - time.time() > 3.0:
            try:
                xs, vs = _squeeze(x_best, steps=300, step0=2e-3, deadline=deadline - 8.0)
                if vs < best_val:
                    best_val = vs
                    x_best = xs
            except Exception:
                pass
        try:
            x_best = _lp_polish(x_best, deadline)
        except Exception:
            pass
    x_best = _norm(x_best)
    return x_best
