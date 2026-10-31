import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def lp_polish(x, budget):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    best_x = x.copy()
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < budget and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
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
            x, best, best_x = y, v, y.copy()
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() - start >= budget:
            break
    return best_x

def smooth_opt(X, steps, b0, b1, lr0, lr1, mu=0.9, top_frac=1.0):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n = X.shape[0]
    V = np.zeros_like(X)
    eye2 = 2.0 * np.eye(n)
    for k in range(steps):
        f = k / (steps - 1) if steps > 1 else 0.0
        beta = b0 + (b1 - b0) * f
        lr = lr0 + (lr1 - lr0) * f
        G = X @ X.T
        m = float(np.max(G - eye2))
        # optional top-fraction mask to focus on near-max pairs (allows topology moves early)
        if top_frac < 1.0:
            # threshold is quantile of upper triangle
            iu = np.triu_indices(n, 1)
            c = G[iu]
            thr = np.quantile(c, 1.0 - top_frac)
            thr = min(thr, m - 0.005)
        else:
            thr = -2.0
        A = beta * (G - m)
        # kill far pairs
        if top_frac < 1.0:
            A[G < thr] = -30.0
        np.clip(A, -12, 0, out=A)
        np.fill_diagonal(A, -30.0)
        np.exp(A, out=A)
        # symmetrize (already symmetric) and gradient
        Grad = A @ X
        d = np.sum(Grad * X, axis=1, keepdims=True)
        T = Grad - d * X
        s = float(np.mean(np.sqrt(np.sum(T * T, axis=1)))) + 1e-12
        T /= s
        V *= mu
        V += (1.0 - mu) * T
        X -= lr * V
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def find_holes(Xfix, k, rng, n_cand=6000):
    # find k directions minimizing max dot to Xfix, well separated
    C = rng.standard_normal((n_cand, 4))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    S = C @ Xfix.T
    mx = S.max(axis=1)  # smaller is better (larger hole)
    order = np.argsort(mx)
    chosen = []
    for idx in order:
        c = C[idx]
        ok = True
        for j in chosen:
            if float(c @ C[j]) > 0.85:
                ok = False
                break
        if ok:
            chosen.append(int(idx))
            if len(chosen) >= k:
                break
    if len(chosen) < k:
        for idx in order:
            if int(idx) not in chosen:
                chosen.append(int(idx))
                if len(chosen) >= k:
                    break
    H = C[np.array(chosen[:k])]
    # quick hill climb: move away from nearest neighbour
    for _ in range(60):
        S2 = H @ Xfix.T
        nn = np.argmax(S2, axis=1)
        Gdir = Xfix[nn]
        # tangent step away
        dot = np.sum(H * Gdir, axis=1, keepdims=True)
        Tg = Gdir - dot * H
        H = H - 0.05 * Tg
        H /= np.linalg.norm(H, axis=1, keepdims=True)
    return H

def solve():
    t0 = time.time()
    HARD = 176.0
    n, d = 153, 4
    rng = np.random.default_rng(777123)
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            v = np.load(fn).astype(float)
            if v.shape == (n, d):
                v /= np.linalg.norm(v, axis=1, keepdims=True)
                c = max_cos(v)
                if c < best_val:
                    best_val = c
                    best = v.copy()
        except Exception:
            pass
    if best is None:
        best = rng.standard_normal((n, d))
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = max_cos(best)

    def contacts_of(X, tol=1e-9):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        mx = float(G.max())
        cnt = np.sum(G > mx - tol, axis=1)
        return cnt, mx, G

    trial = 0
    sigmas = [0.02, 0.05, 0.09, 0.015, 0.04, 0.07, 0.11]
    while time.time() - t0 < 130.0 and trial < 80:
        trial += 1
        elapsed = time.time() - t0
        if HARD - elapsed < 12.0:
            break
        cnt, mx, G = contacts_of(best, tol=1e-9)
        ratt = np.where(cnt == 0)[0]
        # broad hubs
        cntB, _, _ = contacts_of(best, tol=1e-4)
        orderB = np.argsort(-cntB)
        hubs = orderB[:14]
        # contact neighbours of top hub for patch removal
        use_hole = (trial % 2 == 1)
        if use_hole:
            # victims: rattlers + 2-5 hubs/patch
            krep = 3 + (trial % 4)  # 3..6
            victims = list(ratt[:2])
            # add top hub and its contact neighbours
            try:
                h0 = int(hubs[(trial // 2) % len(hubs)])
                # neighbours tied within 1e-7
                Gb = best @ best.T
                np.fill_diagonal(Gb, -2.0)
                m0 = float(Gb.max())
                nbrs = np.where(Gb[h0] > m0 - 1e-7)[0]
                victims.append(h0)
                for nb in nbrs:
                    if len(victims) >= krep:
                        break
                    if int(nb) not in victims:
                        victims.append(int(nb))
            except Exception:
                pass
            # fill up with random hubs / random indices
            while len(victims) < krep:
                if rng.random() < 0.7 and len(hubs) > 0:
                    cand = int(rng.choice(hubs))
                else:
                    cand = int(rng.integers(0, n))
                if cand not in victims:
                    victims.append(cand)
            victims = victims[:krep]
            keep = np.ones(n, dtype=bool)
            keep[np.array(victims)] = False
            Xfix = best[keep]
            H = find_holes(Xfix, len(victims), rng, n_cand=6000)
            X = best.copy()
            for vi, h in zip(victims, H):
                # small jitter so symmetric holes break
                X[vi] = h + 0.01 * rng.standard_normal(d)
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            # anneal: early global, late focused
            if trial % 4 == 1:
                X = smooth_opt(X, 1700, 15.0, 300.0, 0.022, 0.002, mu=0.9, top_frac=0.35)
            else:
                X = smooth_opt(X, 2000, 25.0, 350.0, 0.020, 0.0015, mu=0.92, top_frac=0.5)
        else:
            sigma = sigmas[(trial - 1) % len(sigmas)]
            X = best + sigma * rng.standard_normal((n, d))
            for idx in ratt:
                X[idx] = rng.standard_normal(d)
            # replace 1-2 hubs fully at random to change topology
            k2 = 1 + (trial % 2)
            for _ in range(k2):
                idx = int(rng.choice(hubs)) if len(hubs) > 0 else int(rng.integers(0, n))
                if rng.random() < 0.6:
                    X[idx] = rng.standard_normal(d)
                else:
                    X[idx] += 0.4 * rng.standard_normal(d)
            if rng.random() < 0.4:
                X[int(rng.integers(0, n))] = rng.standard_normal(d)
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            steps = 1800 if sigma < 0.08 else 2200
            X = smooth_opt(X, steps, 30.0, 320.0, 0.020, 0.002, mu=0.9, top_frac=1.0)
        v = max_cos(X)
        remain = HARD - (time.time() - t0) - 14.0
        if remain < 1.2:
            if v < best_val - 1e-12:
                best, best_val = X, v
            break
        if v < best_val + 0.02:
            if v < best_val + 0.003:
                budget = min(7.0, remain)
            else:
                budget = min(3.5, remain)
            Xp = lp_polish(X, budget)
            vp = max_cos(Xp)
            if vp < best_val - 1e-12:
                best, best_val = Xp, vp
        else:
            if v < best_val - 1e-12:
                best, best_val = X, v
    remain = HARD - (time.time() - t0) - 1.0
    if remain > 2.0:
        best = lp_polish(best, min(remain, 26.0))
    best /= np.linalg.norm(best, axis=1, keepdims=True)
    return best
