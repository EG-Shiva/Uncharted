import time
import numpy as np
try:
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    _HAS = True
except Exception:
    _HAS = False

def _mc(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _permax(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1), G

def smooth_polish(X, steps, beta0, beta1, lr0, lr1, mu=0.85):
    n, d = X.shape
    V = np.zeros_like(X)
    for s in range(steps):
        f = s / max(steps - 1, 1)
        beta = beta0 * (beta1 / beta0) ** f
        lr = lr0 * (lr1 / lr0) ** f
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        E = np.exp(beta * (G - m))
        np.fill_diagonal(E, 0.0)
        rs = E.sum(axis=1, keepdims=True) + 1e-12
        Gr = E @ X
        dot = (Gr * X).sum(axis=1, keepdims=True)
        Gt = (Gr - dot * X) / rs
        V = mu * V + (1.0 - mu) * Gt
        bc = 1.0 - mu ** (s + 1)
        Vhat = V / bc
        X = X - lr * Vhat
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X = X / nr
    return X

def lp_polish(x, seconds):
    if not _HAS:
        return x
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _mc(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        cfull = (x @ x.T)[iu]
        m = cfull.max()
        thr = m - max(20 * radius, 1e-9)
        sel = cfull > thr
        I = iu[0][sel]; J = iu[1][sel]
        cs = cfull[sel]
        p = len(I)
        if p == 0:
            break
        if p > 2500:
            k = np.argsort(cs)[-2500:]
            I, J, cs = I[k], J[k], cs[k]
            p = 2500
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _mc(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def refine_point(y0, Xr, iters=350, lr0=0.05, rng=None):
    y = y0 / np.linalg.norm(y0)
    cur = float((Xr @ y).max())
    lr = lr0
    for _ in range(iters):
        dots = Xr @ y
        j = int(np.argmax(dots))
        g = Xr[j] - (Xr[j] @ y) * y
        gn = np.linalg.norm(g)
        if gn < 1e-12:
            break
        g /= gn
        yn = y - lr * g
        nyn = np.linalg.norm(yn)
        if nyn == 0:
            lr *= 0.5
            continue
        yn /= nyn
        new = float((Xr @ yn).max())
        if new < cur:
            y, cur = yn, new
            lr *= 1.03
        else:
            lr *= 0.55
        if lr < 1e-7:
            break
    return y

def solve():
    t0 = time.time()
    TLIM = 178.0
    rng = np.random.RandomState(0)
    np.random.seed(0)
    n, d = 82, 14
    Xinit = None
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            if a.shape == (n, d):
                Xinit = a.astype(float)
                break
        except Exception:
            continue
    if Xinit is None:
        Xinit = rng.randn(n, d)
    Xinit /= np.linalg.norm(Xinit, axis=1, keepdims=True)
    best = Xinit.copy()
    best_val = _mc(best)
    # quick jam of start
    if _HAS and (time.time() - t0 < 150):
        try:
            y = lp_polish(best.copy(), seconds=4.0)
            v = _mc(y)
            if v <= best_val:
                best, best_val = y, v
        except Exception:
            pass
    # baseline smooth on start
    try:
        if time.time() - t0 < 140:
            Y0 = smooth_polish(best.copy(), steps=2000, beta0=80.0, beta1=1500.0, lr0=0.02, lr1=0.0006, mu=0.85)
            v0 = _mc(Y0)
            if v0 < best_val:
                # squeeze it
                if _HAS:
                    try:
                        Y0 = lp_polish(Y0, seconds=2.0)
                        v0 = _mc(Y0)
                    except Exception:
                        pass
                if v0 < best_val:
                    best, best_val = Y0, v0
    except Exception:
        pass

    sigmas = [0.003, 0.01, 0.03, 0.07, 0.15]
    trial = 0
    while True:
        if time.time() - t0 > 148.0:
            break
        trial += 1
        mode = trial % 12
        Y = best.copy()
        if mode == 0:
            # fresh random
            Y = rng.randn(n, d); Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            steps, b0, b1, l0, l1 = 1800, 60.0, 1200.0, 0.03, 0.0008
        elif mode in (1, 2):
            # worst-point ruin & recreate (1 or 2 points)
            pm, G = _permax(Y)
            order = np.argsort(pm)
            k = 1 if mode == 1 else 2
            if trial % 24 == 14:
                k = 3
            Ws = list(order[-k:])
            # occasionally pick random jammed instead of worst
            if trial % 7 == 0:
                Ws = list(rng.choice(n, size=k, replace=False))
            for w_ in Ws:
                idx = [i for i in range(n) if i != w_]
                # for multi-point, use current Y with other Ws still present except w_
                # build rest excluding all Ws then add back already reinserted
                rest_idx = [i for i in range(n) if i not in Ws]
                # include already done Ws? to keep simple use rest of non-W plus keep current values of other Ws
                Xr_full = Y[rest_idx]
                K = 6000
                C = rng.randn(K, d); C /= np.linalg.norm(C, axis=1, keepdims=True)
                sc = (C @ Xr_full.T).max(axis=1)
                # also consider dots to other Ws (approx with current positions)
                if len(Ws) > 1:
                    other = np.array([Y[i] for i in Ws if i != w_])
                    if len(other) > 0:
                        sc = np.maximum(sc, (C @ other.T).max(axis=1))
                bi = int(np.argmin(sc))
                y0 = C[bi]
                Xr = Y[[i for i in range(n) if i != w_]]
                y0 = refine_point(y0, Xr, iters=350, lr0=0.05, rng=rng)
                Y[w_] = y0
            steps, b0, b1, l0, l1 = 1400, 70.0, 1400.0, 0.02, 0.0005
        elif mode == 3:
            # break the current max pair: relocate one endpoint to best hole
            pm, G = _permax(Y)
            G2 = Y @ Y.T; np.fill_diagonal(G2, -2.0)
            ij = np.unravel_index(int(np.argmax(G2)), G2.shape)
            w_ = int(ij[rng.randint(2)])
            Xr = Y[[i for i in range(n) if i != w_]]
            K = 8000
            C = rng.randn(K, d); C /= np.linalg.norm(C, axis=1, keepdims=True)
            sc = (C @ Xr.T).max(axis=1)
            bi = int(np.argmin(sc))
            Y[w_] = refine_point(C[bi], Xr, iters=400, lr0=0.06, rng=rng)
            steps, b0, b1, l0, l1 = 1400, 70.0, 1400.0, 0.02, 0.0005
        elif mode == 4:
            # rattler teleport
            pm, G = _permax(Y)
            m = pm.max()
            order = np.argsort(pm)
            loose = [i for i in order if pm[i] < m - 2e-4]
            if len(loose) < 3:
                loose = list(order[:6])
            loose = loose[:max(4, min(len(loose), 7))]
            for i in loose:
                v = rng.randn(d); v /= np.linalg.norm(v)
                Y[i] = v
            steps, b0, b1, l0, l1 = 1500, 70.0, 1400.0, 0.025, 0.0006
        else:
            sigma = sigmas[(trial + trial // 12) % len(sigmas)]
            if trial % 3 == 0:
                Y = Y + sigma * rng.randn(n, d)
            else:
                pm, _ = _permax(Y)
                o = np.argsort(pm)
                subset = list(o[-8:]) + list(o[:2])
                Y[subset] += (sigma * 1.6) * rng.randn(len(subset), d)
                Y += (sigma * 0.3) * rng.randn(n, d)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            steps, b0, b1, l0, l1 = 1250, 60.0, 1300.0, 0.025, 0.0006
        try:
            Y = smooth_polish(Y, steps=steps, beta0=b0, beta1=b1, lr0=l0, lr1=l1, mu=0.85)
        except Exception:
            continue
        v = _mc(Y)
        improved = False
        if v < best_val:
            # LP squeeze this improvement
            if _HAS and (time.time() - t0 < 155.0):
                try:
                    Ys = lp_polish(Y.copy(), seconds=2.0)
                    vs = _mc(Ys)
                    if vs <= v:
                        Y, v = Ys, vs
                except Exception:
                    pass
            if v < best_val:
                best, best_val = Y, v
                improved = True
        # occasional squeeze of near-miss to discover LP-only win
        if (not improved) and _HAS and v < best_val + 3e-4 and (trial % 10 == 0) and (time.time() - t0 < 145.0):
            try:
                Ys = lp_polish(Y.copy(), seconds=1.0)
                vs = _mc(Ys)
                if vs < best_val:
                    best, best_val = Ys, vs
            except Exception:
                pass
    # final long refinement
    try:
        if time.time() - t0 < TLIM - 30:
            best = smooth_polish(best.copy(), steps=4500, beta0=150.0, beta1=3000.0, lr0=0.008, lr1=0.00015, mu=0.88)
    except Exception:
        pass
    remain = TLIM - (time.time() - t0) - 1.0
    if remain > 3 and _HAS:
        try:
            Ylp = lp_polish(best.copy(), seconds=min(remain, 25.0))
            if _mc(Ylp) <= _mc(best):
                best = Ylp
        except Exception:
            pass
    best /= np.linalg.norm(best, axis=1, keepdims=True)
    return best
