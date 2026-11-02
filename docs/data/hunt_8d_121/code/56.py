import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def solve():
    t0 = time.time()
    DEADLINE = t0 + 168.0
    KNOCK_DEADLINE = t0 + 128.0
    rng = np.random.RandomState(0)
    n, d = 121, 8
    iu_all = np.triu_indices(n, 1)
    Iall, Jall = iu_all[0], iu_all[1]

    def _normed(a):
        a = np.asarray(a, dtype=float)
        nr = np.linalg.norm(a, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        return a / nr

    # load best / record
    def _load(fn):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (n, d):
                a = _normed(a)
                if np.all(np.isfinite(a)):
                    return a
        except Exception:
            pass
        return None
    Xb = _load("best.npy")
    Xr = _load("record.npy")
    if Xb is None and Xr is None:
        Xb = _normed(rng.randn(n, d))
    if Xb is None:
        Xb = Xr.copy()
    if Xr is None:
        Xr = Xb.copy()
    Vb = _max_cos(Xb)
    Vr = _max_cos(Xr)
    # reference best value
    Xbest0 = Xb.copy()
    Vbest0 = float(Vb)

    # contact helpers at 1e-4
    def contact_edges(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        E = G >= (m - 1e-4)
        np.fill_diagonal(E, False)
        return E, m

    Erec, mrec = contact_edges(Xr)
    # degrees of best for seed choice
    Ebest, mbest = contact_edges(Xb)
    deg = Ebest.sum(axis=1)
    Gbest = Xb @ Xb.T
    np.fill_diagonal(Gbest, -2.0)

    seed_order = np.argsort(-deg, kind="stable")
    # keep only top ~30 to stay in dense region
    if len(seed_order) > 30:
        seed_order = seed_order[:30]

    sizes = [12, 16, 20, 12, 16, 20, 12, 16, 20]
    ntrials = len(sizes)

    # ---------- Adam helpers ----------
    def adam_hinge_subset(Y, var_idx, t=0.455, steps=300, lr=0.006):
        var_idx = np.asarray(var_idx, dtype=int)
        k = len(var_idx)
        m1 = np.zeros((k, d))
        m2 = np.zeros((k, d))
        b1, b2, eps = 0.9, 0.999, 1e-8
        Y = Y.copy()
        for s in range(1, steps + 1):
            if time.time() > DEADLINE - 2:
                break
            Yv = Y[var_idx]  # k x d
            # G var vs all
            Gva = Yv @ Y.T  # k x n
            D = Gva - t
            D[D < 0] = 0.0
            # zero self entries (var vs itself at same position)
            # map: for each vi, position in all = var_idx[i]
            # set that column to 0 to avoid self penalty
            # vectorized:
            # D[i, var_idx[i]] = 0
            D[np.arange(k), var_idx] = 0.0
            W = 2.0 * D  # k x n
            GY = W @ Y  # k x d
            dot = (GY * Yv).sum(axis=1, keepdims=True)
            Gr = GY - dot * Yv
            # adam update
            m1 = b1 * m1 + (1 - b1) * Gr
            m2 = b2 * m2 + (1 - b2) * (Gr * Gr)
            c1 = 1 - b1 ** s
            c2 = 1 - b2 ** s
            mh = m1 / c1
            vh = m2 / c2
            step = lr * mh / (np.sqrt(vh) + eps)
            # clip step norm
            Yv = Yv - step
            Yv /= np.maximum(np.linalg.norm(Yv, axis=1, keepdims=True), 1e-12)
            Y[var_idx] = Yv
        return _normed(Y)

    def adam_lse_all(Y, betas_steps, lr=0.004):
        Y = Y.copy()
        m1 = np.zeros_like(Y)
        m2 = np.zeros_like(Y)
        b1, b2, eps = 0.9, 0.999, 1e-8
        s = 0
        for beta, steps in betas_steps:
            beta = float(beta)
            for _ in range(int(steps)):
                s += 1
                if time.time() > DEADLINE - 2:
                    return _normed(Y)
                G = Y @ Y.T
                np.fill_diagonal(G, -10.0)
                m = float(np.max(G))
                A = beta * (G - m)
                np.clip(A, -700, 0, out=A)
                E = np.exp(A)
                np.fill_diagonal(E, 0.0)
                S = float(np.sum(np.triu(E, 1)))
                if not np.isfinite(S) or S <= 0:
                    S = 1e-300
                EY = E @ Y
                GY = EY / S
                dot = (GY * Y).sum(axis=1, keepdims=True)
                Gr = GY - dot * Y
                # scale grad to avoid explosion: normalize by max norm
                gn = float(np.max(np.linalg.norm(Gr, axis=1)))
                if gn > 10.0:
                    Gr *= (10.0 / gn)
                m1 = b1 * m1 + (1 - b1) * Gr
                m2 = b2 * m2 + (1 - b2) * (Gr * Gr)
                c1 = 1 - b1 ** s
                c2 = 1 - b2 ** s
                mh = m1 / c1
                vh = m2 / c2
                Y = Y - lr * mh / (np.sqrt(vh) + eps)
                Y /= np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
        return _normed(Y)

    def lp_polish_timed(x_init, seconds):
        x = _normed(x_init.copy())
        try:
            cur = _max_cos(x)
        except Exception:
            return x
        radius = 1e-3
        start = time.time()
        nv = n * d + 1
        while (time.time() - start < seconds and time.time() < DEADLINE - 1.0
               and radius > 1e-13):
            G = x @ x.T
            c_all = G[Iall, Jall]
            m = float(c_all.max())
            win = max(20.0 * radius, 2e-9)
            sel = c_all > m - win
            I = Iall[sel]; J = Jall[sel]; c = c_all[sel]
            p = len(I)
            if p == 0:
                break
            if p > 2500:
                idx = np.argpartition(-c, 2500)[:2500]
                I = I[idx]; J = J[idx]; c = c[idx]
                p = 2500
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d),
                                   J[:, None] * d + np.arange(d),
                                   np.full((p, 1), n * d)], axis=1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
            try:
                A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
                A_eq = coo_matrix((x.ravel(),
                                   (np.repeat(np.arange(n), d), np.arange(n * d))),
                                  shape=(n, nv)).tocsr()
                obj = np.zeros(nv); obj[-1] = 1.0
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
            y = _normed(x + res.x[:-1].reshape(n, d))
            v = _max_cos(y)
            if v < cur:
                x, cur = y, v
                radius = min(radius * 1.5, 0.05)
            else:
                radius *= 0.5
        return x

    outcomes = []  # (val, X)

    for ti in range(ntrials):
        if time.time() > KNOCK_DEADLINE:
            break
        k = int(sizes[ti])
        seed = int(seed_order[ti % len(seed_order)])
        # nearest neighbours by angle (largest cosine to seed)
        sims = Gbest[seed].copy()
        sims[seed] = -2.0
        nn = np.argsort(-sims, kind="stable")[:k - 1]
        R = np.concatenate([[seed], nn]).astype(int)
        Rset = set(R.tolist())
        frozen_mask = np.ones(n, dtype=bool)
        frozen_mask[R] = False
        # frozen points
        Yfrozen = Xb[frozen_mask]  # Nf x d (but use current Xb? use original best)
        # 50k random samples
        try:
            C = rng.randn(50000, d)
            C /= np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)
        except Exception:
            continue
        # greedy insertions
        chosen = []
        # maintain matrix of frozen+chosen
        cur_mat = Yfrozen.copy()
        for s_idx in range(k):
            if time.time() > KNOCK_DEADLINE:
                break
            # D = C @ cur_mat.T -> 50k x (Nf+s)
            D = C @ cur_mat.T
            cm = D.max(axis=1)
            pick = int(np.argmin(cm))
            pt = C[pick].copy()
            chosen.append(pt)
            cur_mat = np.vstack([cur_mat, pt[None, :]])
        if len(chosen) != k:
            continue
        X0 = Xb.copy()
        X0[R] = np.vstack(chosen)
        X0 = _normed(X0)
        # hinge on inserted only at t=0.455, 300 Adam steps
        try:
            X1 = adam_hinge_subset(X0, R, t=0.455, steps=300, lr=0.006)
        except Exception:
            continue
        # unfreeze all: LSE annealing beta 100->2000, <=800 steps
        # use 500 Adam steps split
        try:
            sched = [(100, 125), (300, 125), (800, 125), (2000, 125)]
            X2 = adam_lse_all(X1, sched, lr=0.004)
        except Exception:
            continue
        try:
            v2 = _max_cos(X2)
        except Exception:
            continue
        if not np.isfinite(v2):
            continue
        # topology filter: contact graph at 1e-4 differs by >=50 edges
        try:
            E2, _ = contact_edges(X2)
            # symmetric difference on upper triangle
            diff = np.sum((E2.astype(np.int8) != Erec.astype(np.int8))) // 2
            # E includes both triangles + diag false, so //2 gives upper count*? diag equal so fine
            # diff counts both (i,j) and (j,i); divide by 2
            if diff < 50:
                # reject per spec (do not promote), but still could keep if very good?
                # spec says reject
                continue
        except Exception:
            pass
        outcomes.append((float(v2), X2.copy()))
        # update Gbest slightly? keep original seeds fixed; no

    # select best two outcomes for LP polish
    if not outcomes:
        return _normed(Xbest0.copy())
    outcomes.sort(key=lambda z: z[0])
    # take best two (already filtered)
    top2 = outcomes[:2]
    polished = []
    for v, X in top2:
        if time.time() > DEADLINE - 20:
            break
        try:
            Xp = lp_polish_timed(X, seconds=15.0)
            vp = _max_cos(Xp)
            polished.append((float(vp), Xp))
        except Exception:
            polished.append((float(v), X))
    # choose best among polished + original best
    bestX = Xbest0.copy()
    bestV = float(Vbest0)
    for v, X in polished:
        if np.isfinite(v) and v < bestV:
            bestV = float(v)
            bestX = _normed(X.copy())
    # also consider unpolished outcomes that might beat polished (if polish timed out)
    # but spec says return best LP-polished survivor, fallback to best.npy
    # if no polished beats best, fallback
    if bestV < Vbest0:
        return _normed(bestX)
    else:
        # check if any unpolished beats best but was filtered? they were filtered already.
        # fallback
        return _normed(Xbest0.copy())
