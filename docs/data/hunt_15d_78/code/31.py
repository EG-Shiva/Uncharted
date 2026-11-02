import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _contacts(X, tol=1e-9):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    C = (G > m - tol).sum(axis=1)
    return C, m, G

def _lp_polish(X, seconds):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        Gfull = X @ X.T
        c = Gfull[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
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
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X


def solve():
    t0 = time.time()
    LIMIT = 165.0
    rng = np.random.default_rng(42)
    n, d = 78, 15

    bestX = None
    bestV = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (n, d):
                continue
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            if np.isnan(a).any():
                continue
            v = _max_cos(a)
            if v < bestV:
                bestV = v
                bestX = a.copy()
        except Exception:
            continue
    if bestX is None:
        bestX = rng.standard_normal((n, d))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        bestV = _max_cos(bestX)

    target_scale = n / d

    def adam_quench(X0, betas, steps_list, lam_list, lr0=0.008):
        X = X0 / np.linalg.norm(X0, axis=1, keepdims=True)
        m1 = np.zeros_like(X)
        m2 = np.zeros_like(X)
        t = 0
        b1, b2 = 0.9, 0.999
        eps = 1e-8
        cur_best = _max_cos(X)
        best_snap = X.copy()
        I15 = np.eye(d)
        for bi, beta in enumerate(betas):
            steps = steps_list[bi]
            lam = lam_list[bi]
            lr = lr0 * (1.0 if bi < len(betas)-1 else 0.6)
            for s in range(steps):
                if time.time() - t0 > LIMIT - 26:
                    break
                t += 1
                G = X @ X.T
                np.fill_diagonal(G, 0.0)
                # max off-diag
                mx = G.max()
                E = np.exp(beta * (G - mx))
                np.fill_diagonal(E, 0.0)
                S = E.sum()
                if S <= 0 or not np.isfinite(S):
                    S = 1.0
                W = E / S
                Grad = W @ X
                if lam != 0.0:
                    C = X.T @ X
                    Dm = C - target_scale * I15
                    Grad = Grad + lam * (X @ Dm)
                dot = (Grad * X).sum(axis=1, keepdims=True)
                Gt = Grad - dot * X
                m1 = b1 * m1 + (1 - b1) * Gt
                m2 = b2 * m2 + (1 - b2) * (Gt * Gt)
                c1 = 1 - b1 ** t
                c2 = 1 - b2 ** t
                mh = m1 / c1
                vh = m2 / c2
                delta = mh / (np.sqrt(vh) + eps)
                # slight lr decay inside stage
                lrc = lr * (1.0 - 0.4 * s / max(1, steps))
                X = X - lrc * delta
                X = X / np.linalg.norm(X, axis=1, keepdims=True)
                if (s & 31) == 31 or s == steps - 1:
                    v = _max_cos(X)
                    if v < cur_best:
                        cur_best = v
                        best_snap = X.copy()
            if time.time() - t0 > LIMIT - 26:
                break
        return best_snap, cur_best

    def void_centers(X, k, nsamp=6000):
        R = rng.standard_normal((nsamp, d))
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        Gc = R @ X.T
        sc = Gc.max(axis=1)
        idx = np.argpartition(sc, k)[:k]
        return R[idx]

    def op_shake(X, sigma):
        Y = X + sigma * rng.standard_normal((n, d))
        return Y / np.linalg.norm(Y, axis=1, keepdims=True)

    def op_void_reinsert(X, k=8):
        cnts, mcur, G = _contacts(X, tol=2e-3)
        order = np.argsort(cnts, kind='stable')
        victims = order[:k]
        Vc = void_centers(X, k)
        Y = X.copy()
        for ii, vi in enumerate(victims):
            Y[int(vi)] = Vc[ii]
        Y = Y + 0.005 * rng.standard_normal((n, d))
        return Y / np.linalg.norm(Y, axis=1, keepdims=True)

    def op_transplant(X, nmove=12):
        # split by top PC to find two clusters, move points across gap
        try:
            _, _, Vt = np.linalg.svd(X - X.mean(axis=0, keepdims=True), full_matrices=False)
            v0 = Vt[0]
        except Exception:
            v0 = rng.standard_normal(d); v0 /= np.linalg.norm(v0)
        s = X @ v0
        order = np.argsort(s, kind='stable')
        lo = order[:n//2]; hi = order[n//2:]
        cL = X[lo].mean(axis=0); cL /= np.linalg.norm(cL)
        cR = X[hi].mean(axis=0); cR /= np.linalg.norm(cR)
        Y = X.copy()
        pick_lo = rng.choice(lo, size=min(nmove, len(lo)), replace=False)
        pick_hi = rng.choice(hi, size=min(nmove//2, len(hi)), replace=False)
        for ii in pick_lo:
            Y[int(ii)] = cR + 0.45 * rng.standard_normal(d)
        for ii in pick_hi:
            Y[int(ii)] = cL + 0.45 * rng.standard_normal(d)
        return Y / np.linalg.norm(Y, axis=1, keepdims=True)

    def op_subset_rotate(X, k=20):
        idx = rng.choice(n, size=k, replace=False)
        Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
        if np.linalg.det(Q) < 0:
            Q[:, 0] = -Q[:, 0]
        Y = X.copy()
        Y[idx] = (X[idx] @ Q.T)
        # small jitter to break exact rigidity
        Y[idx] += 0.01 * rng.standard_normal((k, d))
        return Y / np.linalg.norm(Y, axis=1, keepdims=True)

    # ---- initial diverse starts ----
    starts = []
    starts.append(bestX.copy())
    starts.append(op_shake(bestX, 0.55))
    starts.append(op_shake(bestX, 0.95))
    starts.append(op_void_reinsert(bestX, k=10))
    starts.append(op_transplant(bestX, nmove=12))
    R = rng.standard_normal((n, d)); R /= np.linalg.norm(R, axis=1, keepdims=True)
    starts.append(R)

    champ = bestX.copy()
    champV = bestV
    # quench starts with short budget
    for si, S0 in enumerate(starts):
        if time.time() - t0 > LIMIT - 40:
            break
        # shorter for initial screen
        Q, v = adam_quench(S0, betas=[35.0, 130.0, 400.0], steps_list=[350, 350, 500],
                           lam_list=[3e-4, 2e-4, 0.0], lr0=0.008)
        if v < champV - 1e-12:
            champ = Q; champV = v

    # ---- main basin hopping with big topology moves ----
    it = 0
    while time.time() - t0 < LIMIT - 28:
        it += 1
        r = it % 5
        if r == 0:
            T0 = op_shake(champ, float(rng.uniform(0.35, 0.75)))
        elif r == 1:
            T0 = op_void_reinsert(champ, k=int(rng.integers(6, 13)))
        elif r == 2:
            T0 = op_transplant(champ, nmove=int(rng.integers(8, 16)))
        elif r == 3:
            T0 = op_subset_rotate(champ, k=int(rng.integers(14, 28)))
        else:
            # combine void + shake
            T0 = op_void_reinsert(champ, k=int(rng.integers(4, 9)))
            T0 = op_shake(T0, 0.25)
        Q, v = adam_quench(T0, betas=[40.0, 150.0, 450.0], steps_list=[300, 300, 450],
                           lam_list=[3e-4, 1e-4, 0.0], lr0=0.008)
        if v < champV - 1e-12:
            champ = Q; champV = v
        # also occasionally keep a lateral move that improves isotropy a lot
        # (checked cheaply every few iters)
        if time.time() - t0 > LIMIT - 28:
            break

    # ---- final polish: short high-beta snap + LP ----
    try:
        remain = LIMIT - (time.time() - t0) - 1.0
        if remain > 5:
            Q2, v2 = adam_quench(champ, betas=[500.0], steps_list=[400],
                                 lam_list=[0.0], lr0=0.004)
            if v2 < champV:
                champ, champV = Q2, v2
        remain = LIMIT - (time.time() - t0) - 0.8
        if remain > 3:
            Xp = _lp_polish(champ, min(remain, 20.0))
            vp = _max_cos(Xp)
            if vp < champV:
                champ, champV = Xp, vp
    except Exception:
        pass

    champ = champ / np.linalg.norm(champ, axis=1, keepdims=True)
    champ = np.where(np.isnan(champ), 0.0, champ)
    nr = np.linalg.norm(champ, axis=1, keepdims=True)
    nr = np.maximum(nr, 1e-12)
    champ = champ / nr
    return champ
