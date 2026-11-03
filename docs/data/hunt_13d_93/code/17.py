import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def solve():
    t0 = time.time()
    LIMIT = 172.0
    HOP_END = t0 + 148.0
    rng = np.random.default_rng(0)
    np.random.seed(0)
    n, d = 93, 13
    iu0, iu1 = np.triu_indices(n, 1)
    iu = (iu0, iu1)

    def _norm(X):
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        return X / nr

    def _maxcos(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        return float(np.max(G))

    # ---------- load ----------
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (n, d):
                continue
            a = _norm(a)
            v = _maxcos(a)
            if v < best_val:
                best_val = v
                best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.standard_normal((n, d)))
        best_val = _maxcos(best)

    # ---------- annealed soft-max Adam ----------
    def _adam(X, steps, tau0, tau1, lr0, lr1):
        m = np.zeros_like(X)
        v = np.zeros_like(X)
        b1, b2, eps = 0.9, 0.999, 1e-8
        t = 0
        for s in range(steps):
            if time.time() - t0 > LIMIT:
                break
            f = s / max(1, steps - 1)
            tau = tau0 * (tau1 / tau0) ** f if tau1 != tau0 else tau0
            lr = lr0 * (lr1 / lr0) ** f if lr1 != lr0 else lr0
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            mg = float(G.max())
            # weights, stabilized
            W = np.exp((G - mg) / tau)
            np.fill_diagonal(W, 0.0)
            grad = W @ X
            dot = np.einsum('ij,ij->i', grad, X)[:, None]
            gt = grad - dot * X
            t += 1
            m = b1 * m + (1.0 - b1) * gt
            vv = b2 * v + (1.0 - b2) * (gt * gt)
            v[:] = vv
            mh = m / (1.0 - b1 ** t)
            vh = vv / (1.0 - b2 ** t)
            step = lr * mh / (np.sqrt(vh) + eps)
            X = X - step
            X = _norm(X)
        return X

    # ---------- trust-region LP polish ----------
    def _lp(X, seconds, r0=1e-3):
        X = _norm(X.copy())
        cur = _maxcos(X)
        radius = r0
        start = time.time()
        while time.time() - start < seconds and radius > 1e-14:
            if time.time() - t0 > LIMIT:
                break
            G = X @ X.T
            c = G[iu]
            m = float(c.max())
            thresh = m - max(20.0 * radius, 1e-9)
            sel = c > thresh
            cnt = int(sel.sum())
            if cnt == 0:
                radius *= 0.5
                continue
            if cnt > 4000:
                idx = np.argpartition(-c, 4000)[:4000]
                mask = np.zeros_like(sel)
                mask[idx] = True
                sel = mask
                cnt = 4000
            I = iu0[sel]
            J = iu1[sel]
            p = cnt
            nv = n * d + 1
            try:
                rows = np.repeat(np.arange(p), 2 * d + 1)
                cols = np.concatenate([I[:, None] * d + np.arange(d),
                                       J[:, None] * d + np.arange(d),
                                       np.full((p, 1), n * d)], 1).ravel()
                vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
                A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
                A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            except Exception:
                radius *= 0.5
                continue
            obj = np.zeros(nv)
            obj[-1] = 1.0
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            try:
                y = X + res.x[:-1].reshape(n, d)
            except Exception:
                radius *= 0.5
                continue
            y = _norm(y)
            vv = _maxcos(y)
            if vv < cur:
                X, cur = y, vv
                radius = min(radius * 1.5, 0.02)
            else:
                radius *= 0.5
            if time.time() - start > seconds - 0.3:
                break
        return X

    def _void_reinsert(X, nbad, Nc, jitter):
        Xn = X.copy()
        G = Xn @ Xn.T
        np.fill_diagonal(G, -2.0)
        rm = G.max(axis=1)
        order = np.argsort(-rm)
        for k in range(nbad):
            ibad = int(order[k])
            cand = _norm(rng.standard_normal((Nc, d)))
            D = cand @ Xn.T
            D[:, ibad] = -2.0
            worst = D.max(axis=1)
            ibest = int(np.argmin(worst))
            Xn[ibad] = cand[ibest]
        if jitter > 0:
            Xn = _norm(Xn + jitter * rng.standard_normal(Xn.shape))
        return Xn

    def _split_max_pair(X, push=0.02, jitter=0.003):
        Xn = X.copy()
        G = Xn @ Xn.T
        np.fill_diagonal(G, -2.0)
        k = int(np.argmax(G))
        ii = k // n
        jj = k % n
        if ii == jj:
            flat = int(np.argmax(G[iu]))
            ii, jj = int(iu0[flat]), int(iu1[flat])
        xi = Xn[ii].copy()
        xj = Xn[jj].copy()
        di = xj - (xj @ xi) * xi
        dj = xi - (xi @ xj) * xj
        ni = float(np.linalg.norm(di))
        nj = float(np.linalg.norm(dj))
        if ni > 1e-12:
            Xn[ii] = xi - push * di / ni
        if nj > 1e-12:
            Xn[jj] = xj - push * dj / nj
        Xn = _norm(Xn + jitter * rng.standard_normal(Xn.shape))
        return Xn

    # initial quick LP to reach true local optimum of incumbent
    try:
        if time.time() - t0 < LIMIT - 5:
            Xp = _lp(best, seconds=4.0, r0=8e-4)
            vp = _maxcos(Xp)
            if vp < best_val:
                best, best_val = Xp, vp
    except Exception:
        pass

    trial = 0
    while time.time() < HOP_END and time.time() - t0 < LIMIT - 25:
        mode = trial % 7
        try:
            if mode == 0:
                prop = _norm(best + 0.008 * rng.standard_normal(best.shape))
                s, e0, e1, l0, l1 = 850, 0.008, 0.0012, 0.006, 0.0015
            elif mode == 1:
                prop = _norm(best + 0.022 * rng.standard_normal(best.shape))
                s, e0, e1, l0, l1 = 950, 0.010, 0.0010, 0.007, 0.0012
            elif mode == 2:
                # targeted: kick worst 8 + split max pair
                G = best @ best.T
                np.fill_diagonal(G, -2.0)
                rm = G.max(axis=1)
                worst = np.argsort(-rm)[:8]
                prop = best.copy()
                prop[worst] += 0.030 * rng.standard_normal((len(worst), d))
                prop = _split_max_pair(_norm(prop), push=0.015, jitter=0.002)
                s, e0, e1, l0, l1 = 1000, 0.009, 0.0009, 0.007, 0.0012
            elif mode == 3:
                prop = _void_reinsert(best, nbad=1, Nc=7000, jitter=0.002)
                s, e0, e1, l0, l1 = 1000, 0.009, 0.0009, 0.007, 0.0012
            elif mode == 4:
                prop = _void_reinsert(best, nbad=2, Nc=5000, jitter=0.002)
                s, e0, e1, l0, l1 = 1050, 0.010, 0.0011, 0.008, 0.0013
            elif mode == 5:
                prop = best.copy()
                idx = rng.choice(n, size=16, replace=False)
                prop[idx] += 0.025 * rng.standard_normal((len(idx), d))
                prop = _norm(prop)
                s, e0, e1, l0, l1 = 900, 0.011, 0.0013, 0.008, 0.0015
            else:
                prop = _norm(best + 0.05 * rng.standard_normal(best.shape))
                s, e0, e1, l0, l1 = 1100, 0.012, 0.0015, 0.008, 0.0012

            prop = _adam(prop, steps=s, tau0=e0, tau1=e1, lr0=l0, lr1=l1)
            v = _maxcos(prop)
            if v < best_val:
                # immediate win without LP
                # short LP to squeeze further
                rem = min(3.0, max(0.5, (LIMIT - 8 - (time.time() - t0)) / 4))
                prop2 = _lp(prop, seconds=rem, r0=6e-4)
                v2 = _maxcos(prop2)
                if v2 < v:
                    prop, v = prop2, v2
                if v < best_val:
                    best, best_val = prop.copy(), v
            elif v < best_val + 0.0025:
                rem = min(2.2, max(0.4, (LIMIT - 8 - (time.time() - t0)) / 8))
                if rem > 0.35:
                    prop2 = _lp(prop, seconds=rem, r0=5e-4)
                    v2 = _maxcos(prop2)
                    if v2 < best_val:
                        best, best_val = prop2.copy(), v2
        except Exception:
            pass
        trial += 1

    # final squeeze
    try:
        remain = LIMIT - (time.time() - t0) - 1.0
        if remain > 1.0:
            Xf = _lp(best, seconds=min(remain * 0.7, 12.0), r0=4e-4)
            vf = _maxcos(Xf)
            if vf < best_val:
                best, best_val = Xf, vf
        remain = LIMIT - (time.time() - t0) - 0.5
        if remain > 1.0:
            best = _adam(best, steps=400, tau0=0.002, tau1=2e-4, lr0=0.004, lr1=0.0008)
            Xf2 = _lp(best, seconds=min(remain * 0.8, 8.0), r0=3e-4)
            vf2 = _maxcos(Xf2)
            if vf2 < best_val:
                best, best_val = Xf2, vf2
    except Exception:
        pass

    best = _norm(np.asarray(best, dtype=np.float64))
    return best
