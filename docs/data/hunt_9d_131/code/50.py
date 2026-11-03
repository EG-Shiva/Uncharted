import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm(X):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _rand_orth(rng, d):
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    return Q

def _smooth_opt(X, steps, a0, a1, lr0):
    n, d = X.shape
    M = np.zeros_like(X)
    V = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    bestX = X.copy()
    G0 = X @ X.T
    np.fill_diagonal(G0, -2.0)
    best = float(G0.max())
    for t in range(1, steps + 1):
        frac = t / steps
        alpha = a0 * (a1 / a0) ** frac
        lr = lr0 * 0.5 * (1.0 + np.cos(np.pi * frac))
        lr = max(lr, lr0 * 0.08)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        if (t % 100 == 0) or (t == steps):
            if gmax < best:
                best = gmax
                bestX = X.copy()
        E = np.clip(alpha * (G - gmax), -30.0, 0.0)
        W = np.exp(E)
        np.fill_diagonal(W, 0.0)
        Grad = W @ X
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        # normalize scale
        s = float(W.sum()) / n + 1e-12
        Gt = Gt / s
        M = b1 * M + (1 - b1) * Gt
        V = b2 * V + (1 - b2) * (Gt * Gt)
        Mh = M / (1 - b1 ** t)
        Vh = V / (1 - b2 ** t)
        step = Mh / (np.sqrt(Vh) + eps)
        X = X - lr * step
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    v = _max_cos(X)
    if v < best:
        best = v
        bestX = X.copy()
    return bestX, best

def _lp_polish(X, budget, t_deadline):
    X = _norm(X.copy())
    n, d = X.shape
    best = _max_cos(X)
    bestX = X.copy()
    radius = 1e-3
    t0 = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - t0 < budget and time.time() < t_deadline:
        if radius < 1e-14:
            break
        G = X @ X.T
        c_full = G[iu0]
        m = c_full.max()
        thresh = m - max(20 * radius, 1e-9)
        sel = c_full > thresh
        I = iu0[0][sel]; J = iu0[1][sel]
        cc = c_full[sel]
        p = len(I)
        if p == 0:
            break
        if p > 2500:
            idx = np.argpartition(-cc, 2500)[:2500]
            I, J, cc = I[idx], J[idx], cc[idx]
            p = 2500
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cc, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            best = v
            bestX = y.copy()
            X = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return bestX, best

def _greedy_holes(X_keep, m_need, rng, n_cand=4000):
    # X_keep: (n-m, d), return m new points far from keep and each other
    d = X_keep.shape[1]
    cur = X_keep.copy()
    outs = []
    for _ in range(m_need):
        C = rng.standard_normal((n_cand, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        dots = C @ cur.T
        md = dots.max(axis=1)
        bi = int(np.argmin(md))
        z = C[bi].copy()
        # quick local refine on worst-case
        for __ in range(120):
            dv = cur @ z
            mx = dv.max()
            w = np.exp(np.clip(40.0 * (dv - mx), -30, 0))
            g = (w[:, None] * cur).sum(axis=0)
            g = g - (g @ z) * z
            gn = np.linalg.norm(g)
            if gn < 1e-12:
                break
            z = z - 0.03 * g / (gn + 1e-12)
            z /= np.linalg.norm(z)
        outs.append(z)
        cur = np.vstack([cur, z[None, :]])
    return np.array(outs)

def solve():
    t_start = time.time()
    T_END = t_start + 172.0
    T_EXPL = t_start + 148.0
    rng = np.random.default_rng(7)
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (131, 9):
                a = _norm(a)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
        X0 = rng.standard_normal((131, 9))
        cands.append(_norm(X0))
    bestX = None
    best = 1e9
    for a in cands:
        v = _max_cos(a)
        if v < best:
            best = v
            bestX = a.copy()
    n, d = bestX.shape
    pool = [(best, bestX.copy())]

    def consider(Xs, vs):
        global_best = [best]
        # use closure via list
        return vs

    trial = 0
    # precompute a couple of fresh smoothed seeds to diversify pool early
    while time.time() < T_EXPL:
        mode = trial % 7
        X = None
        steps, a0, a1, lr0 = 2600, 25.0, 280.0, 0.003
        if mode == 0:
            # fresh random basin
            X = _norm(rng.standard_normal((n, d)))
            steps, a0, a1, lr0 = 3200, 12.0, 300.0, 0.004
        elif mode == 1:
            # large global kick
            sig = 0.35 if (trial // 7) % 2 == 0 else 0.55
            X = bestX + sig * rng.standard_normal((n, d))
            X = _norm(X)
            steps, a0, a1, lr0 = 3000, 15.0, 280.0, 0.004
        elif mode == 2:
            # hub extinction: delete most-connected, greedy reinsert
            G = bestX @ bestX.T
            np.fill_diagonal(G, -2.0)
            gmax = float(G.max())
            cnts = np.sum(G > gmax - 0.01, axis=1)
            m = 18 + int(rng.integers(0, 12))  # 18-29
            order = np.argsort(-cnts)
            kill = order[:m]
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[kill] = False
            Xkeep = bestX[keep_mask]
            newpts = _greedy_holes(Xkeep, m, rng, n_cand=4000)
            X = np.vstack([Xkeep, newpts])
            # shuffle to avoid ordering bias
            rng.shuffle(X)
            steps, a0, a1, lr0 = 3200, 18.0, 300.0, 0.0035
        elif mode == 3:
            # cap transplant from rotated best onto itself
            Q = _rand_orth(rng, d)
            B = bestX @ Q.T
            c = rng.standard_normal(d); c /= np.linalg.norm(c)
            thr = float(rng.uniform(0.1, 0.7))
            inside = (bestX @ c) > thr
            outside = (B @ c) <= thr
            A_in = bestX[inside]
            B_out = B[outside]
            built = np.vstack([A_in, B_out]) if (len(A_in) + len(B_out)) > 0 else bestX.copy()
            if len(built) > n:
                idx = rng.choice(len(built), size=n, replace=False)
                X = built[idx]
            elif len(built) < n:
                need = n - len(built)
                extra = _greedy_holes(built if len(built) > 0 else bestX[:10], need, rng, n_cand=3000)
                X = np.vstack([built, extra])
            else:
                X = built
            X = _norm(X + 0.01 * rng.standard_normal(X.shape))
            steps, a0, a1, lr0 = 3000, 18.0, 280.0, 0.0035
        elif mode == 4:
            # crossover between pool members via cap
            if len(pool) >= 2 and rng.random() < 0.7:
                # pick two distinct pool entries
                i, j = rng.choice(len(pool), size=2, replace=False)
                A = pool[i][1]
                B0 = pool[j][1]
                Q = _rand_orth(rng, d)
                B = B0 @ Q.T
            else:
                A = bestX
                Q = _rand_orth(rng, d)
                B = bestX @ Q.T
            c = rng.standard_normal(d); c /= np.linalg.norm(c)
            thr = float(rng.uniform(-0.2, 0.5))
            A_in = A[(A @ c) > thr]
            B_out = B[(B @ c) <= thr]
            built = np.vstack([A_in, B_out])
            if len(built) > n:
                idx = rng.choice(len(built), size=n, replace=False)
                X = built[idx]
            elif len(built) < n:
                need = n - len(built)
                extra = _greedy_holes(built if len(built) > 0 else A[:10], need, rng, n_cand=3000)
                X = np.vstack([built, extra])
            else:
                X = built
            X = _norm(X + 0.008 * rng.standard_normal(X.shape))
            steps, a0, a1, lr0 = 3000, 18.0, 300.0, 0.0035
        elif mode == 5:
            # subset hard kick + hole reinsert worst points
            X = bestX + 0.01 * rng.standard_normal((n, d))
            X = _norm(X)
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            ppm = G.max(axis=1)
            k = 10
            worst = np.argsort(-ppm)[:k]
            X[worst] += 0.5 * rng.standard_normal((k, d))
            X = _norm(X)
            steps, a0, a1, lr0 = 2800, 20.0, 300.0, 0.003
        else:
            # shake rattlers + jammed together
            X = bestX.copy()
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            gmax = float(G.max())
            ppm = G.max(axis=1)
            ratt = np.where(ppm < gmax - 1e-3)[0]
            cnts = np.sum(G > gmax - 0.01, axis=1)
            order = np.argsort(-cnts)
            jam = order[:10]
            X[jam] += 0.08 * rng.standard_normal((10, d))
            X = _norm(X)
            if len(ratt) > 0:
                # relocate up to 9 rattlers to holes
                sel = ratt[:9]
                keep_mask = np.ones(n, dtype=bool)
                keep_mask[sel] = False
                Xkeep = X[keep_mask]
                newpts = _greedy_holes(Xkeep, len(sel), rng, n_cand=4000)
                X = np.vstack([Xkeep, newpts])
                rng.shuffle(X)
            steps, a0, a1, lr0 = 2800, 22.0, 300.0, 0.003

        Xs, vs = _smooth_opt(X, steps, a0, a1, lr0)

        # maintain diverse pool (top 4)
        if len(pool) < 4 or vs < max(p[0] for p in pool) - 1e-9:
            # check diversity: min dist? just add if not duplicate
            pool.append((vs, Xs.copy()))
            pool = sorted(pool, key=lambda z: z[0])[:4]

        # LP polish only if promising
        thresh_lp = best + 0.004
        if mode == 0:
            thresh_lp = best + 0.001  # fresh basins rarely good; be strict
        if vs < thresh_lp:
            if vs < best:
                b = 3.0
            else:
                b = 1.2
            rem = T_EXPL - time.time()
            b = min(b, max(0.3, rem - 0.2))
            if b > 0.3:
                Xp, vp = _lp_polish(Xs, b, T_END)
                if vp < best - 1e-12:
                    best = vp
                    bestX = Xp.copy()
                    pool = sorted([(best, bestX.copy())] + pool, key=lambda z: z[0])[:4]
                elif vs < best - 1e-12:
                    best = vs
                    bestX = Xs.copy()
                    pool = sorted([(best, bestX.copy())] + pool, key=lambda z: z[0])[:4]
            else:
                if vs < best - 1e-12:
                    best = vs
                    bestX = Xs.copy()
        else:
            if vs < best - 1e-12:
                best = vs
                bestX = Xs.copy()
        trial += 1
        if time.time() >= T_EXPL:
            break

    rem = T_END - time.time()
    if rem > 1.2:
        b = min(rem - 0.8, 25.0)
        if b > 0.8:
            Xp, vp = _lp_polish(bestX, b, T_END)
            if vp < best:
                best = vp
                bestX = Xp.copy()
    bestX = _norm(bestX)
    return bestX
