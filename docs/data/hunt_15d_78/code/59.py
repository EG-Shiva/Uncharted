import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def _lp_polish(X, seconds, cap=2500):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    best_x = X.copy()
    radius = 8e-3
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    nv = n * d + 1
    obj = np.zeros(nv)
    obj[-1] = 1.0
    while time.time() - start < seconds and radius > 1e-13:
        G = X @ X.T
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        thresh = max(20.0 * radius, 1e-9)
        sel = c_all > m - thresh
        nnz = int(np.count_nonzero(sel))
        if nnz == 0:
            radius *= 0.5
            continue
        if nnz > cap:
            idx = np.where(sel)[0]
            order = np.argsort(-c_all[idx])[:cap]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I = iu0[sel]
        J = iu1[sel]
        p = len(I)
        c_sel = c_all[sel]
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], axis=1).ravel()
        rows = np.repeat(np.arange(p), 2 * d + 1)
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        bounds = [(-radius, radius)] * (n * d) + [(None, None)]
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=bounds, method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        if not np.all(np.isfinite(yn)):
            radius *= 0.5
            continue
        y = y / np.maximum(yn, 1e-12)
        v = _max_cos(y)
        if v < best:
            X = y
            best = v
            best_x = y.copy()
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return best_x


def _adam_quench(X0, iu0, iu1, schedule, b1=0.9, b2=0.999, eps=1e-8):
    n, d = X0.shape
    X = X0 / np.linalg.norm(X0, axis=1, keepdims=True)
    m1 = np.zeros_like(X)
    m2 = np.zeros_like(X)
    t = 0
    for beta, steps, lr in schedule:
        for s in range(steps):
            frac = s / max(1, steps)
            lrc = lr * (1.0 - 0.5 * frac)
            G = X @ X.T
            gv = G[iu0, iu1]
            mx = float(gv.max())
            e = np.exp(beta * (gv - mx))
            S = float(e.sum())
            if not np.isfinite(S) or S <= 0:
                break
            w = e / S
            W = np.zeros((n, n), dtype=np.float64)
            W[iu0, iu1] = w
            W = W + W.T
            Gr = W @ X
            dot = np.sum(Gr * X, axis=1, keepdims=True)
            Gt = Gr - dot * X
            t += 1
            m1 = b1 * m1 + (1.0 - b1) * Gt
            m2 = b2 * m2 + (1.0 - b2) * (Gt * Gt)
            c1 = 1.0 - b1 ** t
            c2 = 1.0 - b2 ** t
            mh = m1 / c1
            vh = m2 / c2
            upd = mh / (np.sqrt(vh) + eps)
            X = X - lrc * upd
            nr = np.linalg.norm(X, axis=1, keepdims=True)
            nr = np.maximum(nr, 1e-12)
            X = X / nr
    return X


def _stress_order(X, beta=250.0):
    n = X.shape[0]
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    E = np.exp(beta * (G - m))
    np.fill_diagonal(E, 0.0)
    s = E.sum(axis=1)
    return np.argsort(s, kind="stable"), s, m


def _void_pick(kept, n_pick, rng, n_cand=300):
    d = kept.shape[1] if kept.size else 15
    if kept.shape[0] == 0:
        c = rng.standard_normal((n_pick, d))
        c /= np.linalg.norm(c, axis=1, keepdims=True)
        return c
    out = np.empty((n_pick, d))
    for k in range(n_pick):
        cand = rng.standard_normal((n_cand, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        scores = (cand @ kept.T).max(axis=1)
        bi = int(np.argmin(scores))
        out[k] = cand[bi]
        kept = np.vstack([kept, out[k:k + 1]])
    return out


def solve():
    t0 = time.time()
    LIMIT = 168.0
    rng = np.random.default_rng(7)
    n, d = 78, 15
    iu0, iu1 = np.triu_indices(n, 1)

    bestX = None
    bestV = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.asarray(np.load(fn), dtype=np.float64)
            if a.shape != (78, 15):
                continue
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
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

    champ = bestX.copy()
    champV = float(bestV)

    sched = [(80.0, 180, 0.025), (200.0, 180, 0.014), (500.0, 220, 0.007)]

    def isotropise(X, iters=3, alpha=0.8):
        Y = X.copy()
        for _ in range(iters):
            mu = Y.mean(axis=0, keepdims=True)
            Y = Y - alpha * mu
            nr = np.linalg.norm(Y, axis=1, keepdims=True)
            nr = np.maximum(nr, 1e-12)
            Y = Y / nr
        return Y

    it = 0
    # reserve ~28s for final LP
    while time.time() - t0 < LIMIT - 30.0 and it < 80:
        it += 1
        mode = it % 5
        if mode == 0:
            # fresh isotropic random
            Xt = rng.standard_normal((n, d))
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = isotropise(Xt, 3, 0.9)
        elif mode == 1 or mode == 2:
            # global big shake of champ
            sigma = 0.35 if mode == 1 else 0.55
            Xt = champ + rng.standard_normal((n, d)) * sigma
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = isotropise(Xt, 2, 0.7)
        else:
            # core-targeted surgery: replace high-stress core + low-stress periphery
            order, stress, mcur = _stress_order(champ, beta=250.0)
            # high stress = tail of order
            k_core = int(rng.integers(3, 6))
            k_peri = int(rng.integers(2, 5))
            hi = order[-12:]
            lo = order[:12]
            ch_hi = rng.choice(hi, size=min(k_core, len(hi)), replace=False)
            ch_lo = rng.choice(lo, size=min(k_peri, len(lo)), replace=False)
            ch_idx = np.concatenate([ch_hi, ch_lo]).astype(int)
            mask = np.ones(n, dtype=bool)
            mask[ch_idx] = False
            kept = champ[mask]
            fresh = _void_pick(kept, len(ch_idx), rng, n_cand=300)
            Xt = champ.copy()
            Xt[ch_idx] = fresh
            # shake rest a bit to break symmetry, larger on core neighbours
            Xt = Xt + rng.standard_normal((n, d)) * 0.02
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = isotropise(Xt, 2, 0.6)
        try:
            Xq = _adam_quench(Xt, iu0, iu1, sched)
        except Exception:
            continue
        v = _max_cos(Xq)
        if v < champV - 1e-12:
            champ = Xq
            champV = float(v)
        # early exit if clearly new record by margin
        if time.time() - t0 > LIMIT - 31.0:
            break

    # selective quick LP snaps on a couple of near-best variants is skipped;
    # intensive polish of champion with remaining time
    try:
        remain = LIMIT - (time.time() - t0) - 1.0
        if remain > 2.0:
            champ = _lp_polish(champ, min(remain, 26.0))
            champV = _max_cos(champ)
    except Exception:
        pass
    champ = champ / np.linalg.norm(champ, axis=1, keepdims=True)
    champ = np.where(np.isnan(champ), 0.0, champ)
    nr = np.linalg.norm(champ, axis=1, keepdims=True)
    nr = np.maximum(nr, 1e-12)
    champ = champ / nr
    return champ
