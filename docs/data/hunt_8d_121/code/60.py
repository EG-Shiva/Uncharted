import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def solve():
    t0 = time.time()
    DEADLINE = t0 + 171.0
    KNOCK_DEADLINE = t0 + 148.0
    n, d = 121, 8
    iu = np.triu_indices(n, 1)

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
        r0 = np.random.default_rng(0)
        X0 = r0.standard_normal((n, d))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        cands = [X0]
    vals = [_max_cos(a) for a in cands]
    Xbest = cands[int(np.argmin(vals))].copy()
    Vbest = float(min(vals))

    # record edge set at 1e-4 slack
    try:
        Rrec = np.load("record.npy").astype(float)
        Rrec /= np.linalg.norm(Rrec, axis=1, keepdims=True)
    except Exception:
        Rrec = Xbest.copy()
    Gr = Rrec @ Rrec.T
    np.fill_diagonal(Gr, -2.0)
    mr = float(Gr.max())
    Er = set()
    ii_r, jj_r = np.where(np.triu(Gr >= mr - 1e-4, 1))
    for a, b in zip(ii_r.tolist(), jj_r.tolist()):
        Er.add((a, b))

    def edge_set(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        ii, jj = np.where(np.triu(G >= m - 1e-4, 1))
        s = set()
        for a, b in zip(ii.tolist(), jj.tolist()):
            s.add((a, b))
        return s, m

    def degrees(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        deg = (G >= m - 1e-4).sum(axis=1)
        return deg, G, m

    def hinge_subset_adam(Yrest, Ymov0, target=0.455, steps=300, lr=0.015):
        k = Ymov0.shape[0]
        X = Ymov0.copy()
        mt = np.zeros_like(X)
        vt = np.zeros_like(X)
        b1, b2, eps = 0.9, 0.999, 1e-8
        tt = float(target)
        for t in range(1, steps + 1):
            if time.time() > DEADLINE - 4:
                break
            nr = np.linalg.norm(X, axis=1, keepdims=True)
            nr = np.maximum(nr, 1e-12)
            Y = X / nr
            Sc = Y @ Yrest.T
            Dc = Sc - tt
            Wc = np.where(Dc > 0, 2.0 * Dc, 0.0)
            Gs = Y @ Y.T
            Ds = Gs - tt
            Ws = np.where(Ds > 0, 2.0 * Ds, 0.0)
            np.fill_diagonal(Ws, 0.0)
            GY = Wc @ Yrest + Ws @ Y
            dot = (GY * Y).sum(axis=1, keepdims=True)
            GYp = GY - dot * Y
            GX = GYp / nr
            mt = b1 * mt + (1 - b1) * GX
            vt = b2 * vt + (1 - b2) * (GX * GX)
            mh = mt / (1 - b1 ** t)
            vh = vt / (1 - b2 ** t)
            X -= lr * mh / (np.sqrt(vh) + eps)
        Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        return Xn

    def lse_global_adam(Xinit, steps_total=800):
        # schedule beta 100 -> 2000
        stages = [(100.0, 200), (300.0, 200), (800.0, 200), (2000.0, 200)]
        # trim if steps_total smaller
        X = Xinit.copy()
        mt = np.zeros_like(X)
        vt = np.zeros_like(X)
        b1, b2, eps = 0.9, 0.999, 1e-8
        t = 0
        Iall, Jall = iu[0], iu[1]
        for beta, st in stages:
            lr = 0.012 if beta < 500 else 0.006
            for _ in range(st):
                if time.time() > DEADLINE - 4:
                    return X / np.linalg.norm(X, axis=1, keepdims=True)
                t += 1
                nr = np.linalg.norm(X, axis=1, keepdims=True)
                nr = np.maximum(nr, 1e-12)
                Y = X / nr
                G = Y @ Y.T
                c = G[iu]
                m = float(c.max())
                e = np.exp(beta * (c - m))
                s = float(e.sum())
                if not np.finite(s) or s <= 0:
                    continue
                w = e / s
                W = np.zeros((n, n))
                W[Iall, Jall] = w
                W[Jall, Iall] = w
                GY = W @ Y
                dot = (GY * Y).sum(axis=1, keepdims=True)
                GYp = GY - dot * Y
                GX = GYp / nr
                mt = b1 * mt + (1 - b1) * GX
                vt = b2 * vt + (1 - b2) * (GX * GX)
                mh = mt / (1 - b1 ** t)
                vh = vt / (1 - b2 ** t)
                X -= lr * mh / (np.sqrt(vh) + eps)
        Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        return Xn

    def lp_polish(x, seconds, radius_init=3e-4):
        start = time.time()
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        best = _max_cos(x)
        radius = radius_init
        it = 0
        while time.time() - start < seconds and time.time() < DEADLINE - 1.0:
            it += 1
            if it > 60 or radius < 1e-14:
                break
            G = x @ x.T
            cf = G[iu]
            m = float(cf.max())
            thr = max(20.0 * radius, 2e-9)
            sel = cf > m - thr
            cnt = int(sel.sum())
            if cnt == 0:
                radius *= 0.5
                continue
            if cnt > 2500:
                kth = np.sort(cf)[-2500]
                sel = cf >= kth
                cnt = int(sel.sum())
                if cnt > 3000:
                    idxp = np.argpartition(-cf, 3000)[:3000]
                    mk = np.zeros_like(sel)
                    mk[idxp] = True
                    sel = mk
                    cnt = 3000
            c = cf[sel]
            I = iu[0][sel]; J = iu[1][sel]
            p = cnt
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d),
                                   J[:, None] * d + np.arange(d),
                                   np.full((p, 1), n * d)], 1).ravel()
            valsA = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            try:
                A_ub = coo_matrix((valsA, (rows, cols)), shape=(p, nv)).tocsr()
                A_eq = coo_matrix((x.ravel(),
                                   (np.repeat(np.arange(n), d), np.arange(n * d))),
                                  shape=(n, nv)).tocsr()
            except Exception:
                radius *= 0.5
                continue
            obj = np.zeros(nv); obj[-1] = 1.0
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq,
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
                x, best = y, v
                radius = min(radius * 1.6, 0.02)
            else:
                radius *= 0.5
        return x, best

    sizes = [12, 16, 20, 12, 16, 20, 12, 16, 20, 12]
    survivors = []
    trial = 0
    for k in sizes:
        if time.time() > KNOCK_DEADLINE:
            break
        if trial >= 10:
            break
        trial += 1
        rng = np.random.default_rng(5000 + trial * 977)
        deg, G0, m0 = degrees(Xbest)
        order = np.argsort(-deg)
        # seed among highest-degree points
        top_pool = order[:10]
        seed = int(top_pool[(trial - 1) % len(top_pool)])
        # 11-19 nearest neighbours by angle (largest dot to seed)
        dots = (Xbest @ Xbest[seed])
        dots[seed] = -2.0
        nn = np.argsort(-dots)[:k - 1]
        knock = np.array([seed] + nn.tolist(), dtype=int)
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[knock] = False
        Yf = Xbest[keep_mask].copy()
        m_f = Yf.shape[0]
        # greedy from 50000 samples minimising max cosine to frozen set
        pool = rng.standard_normal((50000, d))
        pool /= np.linalg.norm(pool, axis=1, keepdims=True)
        try:
            S0 = pool @ Yf.T
            bv = S0.max(axis=1)
        except Exception:
            continue
        new_pts = []
        for _ in range(k):
            j = int(np.argmin(bv))
            ch = pool[j].copy()
            new_pts.append(ch)
            dd = pool @ ch
            np.maximum(bv, dd, out=bv)
        Ymov0 = np.array(new_pts)
        # optimise only inserted with hinge at t=0.455 for 300 Adam steps
        Ymov1 = hinge_subset_adam(Yf, Ymov0, target=0.455, steps=300, lr=0.015)
        Xtry = Xbest.copy()
        Xtry[knock] = Ymov1
        Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
        # unfreeze all for LogSumExp annealing beta 100->2000 <=800 steps
        try:
            Xtry2 = lse_global_adam(Xtry, steps_total=800)
        except Exception:
            Xtry2 = Xtry
        v = _max_cos(Xtry2)
        # reject if contact graph differs from record by fewer than 50 edges
        try:
            Es, _ = edge_set(Xtry2)
            diff = len(Es.symmetric_difference(Er))
        except Exception:
            diff = 100
        if diff < 50:
            continue
        survivors.append((v, Xtry2.copy()))
        # opportunistic update of Xbest without polish if clearly better? keep for seeding
        if v < Vbest - 1e-12:
            # keep best for subsequent seeding but still polish later
            pass

    # pick best two outcomes for LP polish (at most 15s each)
    survivors.sort(key=lambda z: z[0])
    polished = []
    for idx in range(min(2, len(survivors))):
        if time.time() > DEADLINE - 4:
            break
        v0, X0 = survivors[idx]
        # only polish if promising (within reasonable range)
        Xp, Vp = lp_polish(X0, seconds=15.0, radius_init=3e-4)
        polished.append((Vp, Xp))
        if Vp < Vbest:
            Xbest, Vbest = Xp, Vp
    # also consider unpolished survivors that beat best
    for v, X in survivors:
        if v < Vbest:
            Xbest, Vbest = X, v
    for v, X in polished:
        if v < Vbest:
            Xbest, Vbest = X, v

    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest
