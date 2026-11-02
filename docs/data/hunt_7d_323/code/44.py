import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _refine(X, steps, tau0, tau1, lr0, lr1):
    # vectorized annealed soft-max tangential repulsion
    # X assumed normalized
    lt0 = np.log(tau0)
    lt1 = np.log(tau1)
    ll0 = np.log(lr0)
    ll1 = np.log(lr1)
    for s in range(steps):
        f = s / (steps - 1) if steps > 1 else 1.0
        tau = float(np.exp(lt0 + (lt1 - lt0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = (G - m) / tau
        np.clip(E, -20.0, 0.0, out=E)
        np.exp(E, out=E)
        np.fill_diagonal(E, 0.0)
        W = E
        ssum = W.sum(axis=1, keepdims=True)
        # weighted neighbour mean
        Ave = (W @ X) / np.maximum(ssum, 1e-12)
        dot = (Ave * X).sum(axis=1, keepdims=True)
        At = Ave - dot * X
        X = X - lr * At
        nrm = np.linalg.norm(X, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        X = X / nrm
    return X

def _hole_replace(X, idx_list, rng, ncand=15000):
    n, d = X.shape
    for k in idx_list:
        cand = rng.standard_normal(size=(ncand, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        mask = np.ones(n, dtype=bool)
        mask[k] = False
        Xf = X[mask]
        dots = cand @ Xf.T
        worst = dots.max(axis=1)
        bi = int(np.argmin(worst))
        X[k] = cand[bi]
    return X

def _lp_polish(X, deadline):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = 3e-3
    iuI, iuJ = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = X @ X.T
        c = G[iuI, iuJ]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > 3000:
            idx = np.where(sel)[0]
            ci = c[idx]
            part = np.argpartition(ci, -3000)[-3000:]
            sel = np.zeros_like(sel, dtype=bool)
            sel[idx[part]] = True
            cnt = 3000
        I = iuI[sel]
        J = iuJ[sel]
        csel = c[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        vals = np.empty(p * (2 * d + 1), dtype=np.float64)
        Cb = cols.reshape(p, 2 * d + 1)
        Vb = vals.reshape(p, 2 * d + 1)
        ar = np.arange(d)
        Cb[:, :d] = I[:, None] * d + ar
        Cb[:, d:2 * d] = J[:, None] * d + ar
        Cb[:, 2 * d] = n * d
        Vb[:, :d] = X[J]
        Vb[:, d:2 * d] = X[I]
        Vb[:, 2 * d] = -1.0
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
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
        Y = X + res.x[:-1].reshape(n, d)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        v = _max_cos(Y)
        if v < best:
            X = Y
            best = v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() >= deadline:
            break
    return X

def solve():
    t0 = time.time()
    TIME_LIMIT = 165.0
    EXPL_END = t0 + 128.0
    LP_END = t0 + TIME_LIMIT
    rng = np.random.default_rng(12345)
    Xbest = None
    bval = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (323, 7):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(a)
                if v < bval:
                    bval = v
                    Xbest = a.copy()
        except Exception:
            pass
    if Xbest is None:
        a = rng.standard_normal(size=(323, 7))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        Xbest = a
        bval = _max_cos(a)
    Gbest = Xbest.copy()
    gval = bval
    trial = 0
    # precomputed sigma / param cycles for diversity
    sigmas = [0.015, 0.04, 0.09, 0.18, 0.06, 0.025]
    while time.time() < EXPL_END:
        trial += 1
        if time.time() >= EXPL_END:
            break
        Xtry = Gbest.copy()
        mode = trial % 4
        if mode == 0 or mode == 1:
            # hole surgery + anneal : change topology in coordinated way
            G = Xtry @ Xtry.T
            np.fill_diagonal(G, -2.0)
            worst = G.max(axis=1)
            k = 2 + (trial % 4)  # 2..5
            # pick worst k, plus one random to diversify
            order = np.argsort(-worst)
            idx = list(order[:k])
            # occasionally include a random index
            if trial % 3 == 0:
                idx[-1] = int(rng.integers(0, 323))
            nc = 12000 if (trial % 2 == 0) else 20000
            _hole_replace(Xtry, idx, rng, ncand=nc)
            # small global jitter after insertion
            sig = 0.008 + 0.004 * (trial % 3)
            Xtry = Xtry + sig * rng.standard_normal(size=Xtry.shape)
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            steps = 500 + (trial % 3) * 200
            tau0 = 0.025 if mode == 0 else 0.04
            tau1 = 0.0025 if (trial % 2 == 0) else 0.0012
            lr0 = 0.35
            lr1 = 0.04
            Xtry = _refine(Xtry, steps, tau0, tau1, lr0, lr1)
        elif mode == 2:
            # global correlated shake + anneal
            sig = sigmas[(trial // 4) % len(sigmas)]
            Xtry = Xtry + sig * rng.standard_normal(size=Xtry.shape)
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            steps = 700 + (trial % 2) * 300
            tau0 = 0.03
            tau1 = 0.002 if (trial % 2 == 0) else 0.004
            lr0 = 0.3
            lr1 = 0.03
            Xtry = _refine(Xtry, steps, tau0, tau1, lr0, lr1)
        else:
            # cluster shake: pick random centre, shake neighbours together
            ci = int(rng.integers(0, 323))
            d2 = ((Xtry - Xtry[ci]) ** 2).sum(axis=1)
            # nearest ~30 points move coherently + individual noise
            nn = np.argsort(d2)[:30]
            shift = rng.standard_normal(size=(7,)) * 0.12
            Xtry[nn] = Xtry[nn] + shift[None, :] * 0.5 + rng.standard_normal(size=(len(nn), 7)) * 0.03
            # rest small noise
            Xtry = Xtry + rng.standard_normal(size=Xtry.shape) * 0.008
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            steps = 600
            Xtry = _refine(Xtry, steps, 0.035, 0.002, 0.3, 0.03)
        v = _max_cos(Xtry)
        if v < gval:
            gval = v
            Gbest = Xtry.copy()
            # quick opportunistic micro-polish is deferred to final LP
        # safety: if very close but different topology, keep exploring from best only
        if time.time() >= EXPL_END:
            break
    # final trust-region LP polish with remaining budget
    if time.time() < LP_END - 2:
        Gbest = _lp_polish(Gbest, LP_END - 1.0)
    Gbest = Gbest / np.linalg.norm(Gbest, axis=1, keepdims=True)
    return Gbest
