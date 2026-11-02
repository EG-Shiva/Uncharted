import time
import numpy as np

def solve():
    t0 = time.time()
    BUDGET = 165.0
    rng = np.random.RandomState(1)

    def normalize(X):
        n = np.linalg.norm(X, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return X / n

    def max_cos(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        return float(np.max(G))

    best = None
    best_val = 2.0
    for fn in ("best.npy", "record.npy"):
        try:
            v = np.load(fn).astype(np.float64)
            if v.shape == (163, 4):
                v = normalize(v)
                c = max_cos(v)
                if c < best_val:
                    best_val = c
                    best = v.copy()
        except Exception:
            pass
    if best is None:
        best = normalize(rng.randn(163, 4))
        best_val = max_cos(best)

    def polish(X_init, steps, tau0, tau1, lr0, lr1):
        X = X_init.copy()
        m1 = np.zeros_like(X)
        v1 = np.zeros_like(X)
        b1, b2, eps = 0.9, 0.999, 1e-8
        lbv = max_cos(X)
        lb = X.copy()
        for t in range(1, steps + 1):
            if (t & 15) == 0 and time.time() - t0 > BUDGET:
                break
            frac = t / max(1, steps)
            tau = tau0 * (tau1 / tau0) ** frac
            lr = lr0 * (lr1 / lr0) ** frac
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            cur_max = float(np.max(G))
            S = (G - cur_max) / tau
            W = np.exp(np.maximum(S, -12.0))
            np.fill_diagonal(W, 0.0)
            Grad = W @ X
            proj = np.sum(Grad * X, axis=1, keepdims=True)
            Gt = Grad - proj * X
            m1 = b1 * m1 + (1.0 - b1) * Gt
            v1 = b2 * v1 + (1.0 - b2) * (Gt * Gt)
            mhat = m1 / (1.0 - b1 ** t)
            vhat = v1 / (1.0 - b2 ** t)
            step = lr * mhat / (np.sqrt(vhat) + eps)
            X = X - step
            X = normalize(X)
            if (t & 31) == 0 or t == steps:
                c = max_cos(X)
                if c < lbv:
                    lbv = c
                    lb = X.copy()
        return lb, lbv

    def contact_info(X, m, thr=2e-7):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        deg = np.sum(G > (m - thr), axis=1)
        rowmax = np.max(G, axis=1)
        return deg, rowmax, G

    def find_hole(X, exclude=None, n_cand=15000):
        R = normalize(rng.randn(n_cand, 4))
        if exclude is not None and len(exclude) > 0:
            mask = np.ones(X.shape[0], dtype=bool)
            mask[np.array(exclude, dtype=int)] = False
            Y = X[mask]
        else:
            Y = X
        D = R @ Y.T
        vals = np.max(D, axis=1)
        i = int(np.argmin(vals))
        return R[i].copy(), float(vals[i])

    def refine_hole(X, h, steps=150, lr=0.06):
        h = h.copy() / np.linalg.norm(h)
        for _ in range(steps):
            d = X @ h
            m = float(np.max(d))
            w = np.exp((d - m) / 0.008)
            g = w @ X
            g = g - np.dot(g, h) * h
            ng = np.linalg.norm(g)
            if ng > 0:
                g = g / ng
            h = h - lr * g
            # also add small repulsion balancing: re-project
            nh = np.linalg.norm(h)
            if nh == 0:
                break
            h = h / nh
            lr *= 0.995
        return h

    def lp_polish(X, seconds):
        try:
            from scipy.optimize import linprog
            from scipy.sparse import coo_matrix
        except Exception:
            return X, max_cos(X)
        X = normalize(X.copy())
        bv = max_cos(X)
        bx = X.copy()
        n, d = X.shape
        radius = 1.5e-3
        start = time.time()
        iu = np.triu_indices(n, 1)
        while time.time() - start < seconds and time.time() - t0 < BUDGET:
            if radius < 1e-13:
                break
            Gfull = X @ X.T
            c = Gfull[iu]
            m = float(np.max(c))
            thr = max(20 * radius, 1e-9)
            sel = c > m - thr
            # cap size for speed
            if np.sum(sel) > 2500:
                k = 2500
                idx = np.argsort(-c)
                sel = np.zeros_like(sel, dtype=bool)
                sel[idx[:k]] = True
            I, J = iu[0][sel], iu[1][sel]
            p = len(I)
            if p == 0:
                break
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
            try:
                A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
                A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
                obj = np.zeros(nv); obj[-1] = 1.0
                res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            y = X + res.x[:-1].reshape(n, d)
            y = normalize(y)
            v = max_cos(y)
            if v < bv:
                bv = v
                bx = y.copy()
                X = y
                radius = min(radius * 1.6, 0.05)
            else:
                radius *= 0.5
        return bx, bv

    # initial squeeze
    try:
        q, qv = polish(best, steps=1800, tau0=0.007, tau1=0.0018, lr0=0.010, lr1=0.002)
        if qv < best_val:
            best, best_val = q, qv
    except Exception:
        pass

    it = 0
    while time.time() - t0 < BUDGET - 14.0:
        it += 1
        Xtry = best.copy()
        mode = it % 7
        try:
            m0 = max_cos(Xtry)
            deg, rowmax, G = contact_info(Xtry, m0)
        except Exception:
            deg, rowmax = np.zeros(163), np.zeros(163)
        if mode in (0, 1):
            # worst hub -> deep hole
            order = np.lexsort((-rowmax, -deg))
            victim = int(order[0])
            # occasionally move two hubs
            victims = [victim]
            if mode == 1 and it % 2 == 0:
                victims.append(int(order[1]))
            for v in victims:
                others = [u for u in victims if u != v]
                excl = [v] + others
                # need hole w.r.t. remaining points
                keep = np.ones(163, dtype=bool); keep[v] = False
                # for second victim use updated Xtry
                Y = Xtry[keep]
                hole, _ = find_hole(Y, exclude=None, n_cand=12000)
                hole = refine_hole(Y, hole, steps=130, lr=0.06)
                Xtry[v] = hole
            # nudge neighbours of victim to unlock
            if rng.rand() < 0.7:
                Gv = Xtry @ Xtry[victim]
                nbrs = np.argsort(-Gv)[1:6]
                Xtry[nbrs] += 0.02 * rng.randn(len(nbrs), 4)
                Xtry = normalize(Xtry)
        elif mode == 2:
            # rattler / low-degree -> hole
            lows = np.where(deg <= 1)[0]
            if len(lows) == 0:
                lows = np.argsort(deg)[:3]
            victim = int(lows[rng.randint(len(lows))])
            keep = np.ones(163, dtype=bool); keep[victim] = False
            Y = Xtry[keep]
            hole, _ = find_hole(Y, exclude=None, n_cand=12000)
            hole = refine_hole(Y, hole, steps=160, lr=0.06)
            Xtry[victim] = hole
        elif mode == 3:
            # jammed cluster shake around worst pair
            G2 = Xtry @ Xtry.T
            np.fill_diagonal(G2, -2.0)
            ij = int(np.argmax(G2))
            a, b = divmod(ij, 163)
            da = G2[a]
            idx = np.argsort(-da)[:8]
            Xtry[idx] += 0.07 * rng.randn(len(idx), 4)
            Xtry = normalize(Xtry)
        elif mode == 4:
            k = int(rng.randint(4, 18))
            idx = rng.choice(163, size=k, replace=False)
            sigma = [0.05, 0.12, 0.25][rng.randint(3)]
            Xtry[idx] = Xtry[idx] + sigma * rng.randn(k, 4)
            Xtry = normalize(Xtry)
        elif mode == 5:
            sigma = [0.01, 0.03, 0.09][rng.randint(3)]
            Xtry = Xtry + sigma * rng.randn(*Xtry.shape)
            Xtry = normalize(Xtry)
        else:
            # swap: duplicate hole, delete second-worst hub region: move random mid-degree point to second-best hole
            order = np.lexsort((-rowmax, -deg))
            victim = int(order[rng.randint(0, 6)])
            keep = np.ones(163, dtype=bool); keep[victim] = False
            Y = Xtry[keep]
            R = normalize(rng.randn(15000, 4))
            D = R @ Y.T
            vals = np.max(D, axis=1)
            # pick random among 5 deepest to diversify
            bi = np.argsort(vals)[:5]
            hole = R[bi[rng.randint(len(bi))]].copy()
            hole = refine_hole(Y, hole, steps=100, lr=0.05)
            Xtry[victim] = hole
            Xtry = normalize(Xtry)

        remain = BUDGET - 14.0 - (time.time() - t0)
        if remain < 15:
            s1 = 350
        else:
            s1 = 550 if mode in (0, 2, 6) else 450
        try:
            Xf, vf = polish(Xtry, steps=s1, tau0=0.013, tau1=0.003, lr0=0.022, lr1=0.0035)
        except Exception:
            continue
        # promising? extend
        if vf < best_val + 0.008 and time.time() - t0 < BUDGET - 14.0:
            try:
                Xf2, vf2 = polish(Xf, steps=700, tau0=0.005, tau1=0.0012, lr0=0.009, lr1=0.0015)
                Xf, vf = (Xf2, vf2) if vf2 < vf else (Xf, vf)
            except Exception:
                pass
        if vf < best_val - 1e-15:
            # exact LP squeeze on improvement
            try:
                Xlp, vlp = lp_polish(Xf, seconds=2.5)
                if vlp < vf:
                    Xf, vf = Xlp, vlp
            except Exception:
                pass
            best, best_val = Xf, vf

    # final intensification
    try:
        Xf, vf = polish(best, steps=2200, tau0=0.004, tau1=0.0007, lr0=0.006, lr1=0.0008)
        if vf < best_val:
            best, best_val = Xf, vf
    except Exception:
        pass
    if time.time() - t0 < BUDGET - 2.0:
        try:
            Xlp, vlp = lp_polish(best, seconds=min(9.0, max(1.0, BUDGET - (time.time() - t0) - 1.0)))
            if vlp < best_val:
                best, best_val = Xlp, vlp
        except Exception:
            pass

    best = normalize(best)
    return best
