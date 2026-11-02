import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def _lp_polish(X, seconds):
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
        sel = c_all > m - max(20.0 * radius, 1e-9)
        nnz = int(np.count_nonzero(sel))
        if nnz == 0:
            radius *= 0.5
            continue
        if nnz > 2500:
            idx = np.where(sel)[0]
            order = np.argsort(-c_all[idx])[:2500]
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
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        if not np.all(np.isfinite(yn)) or np.any(yn < 1e-12):
            radius *= 0.5
            continue
        y = y / yn
        v = _max_cos(y)
        if v < best:
            X = y
            best = v
            best_x = y.copy()
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return best_x, best


def _adam_quench(X0, steps=600, lr=0.015, b1=0.9, b2=0.999, eps=1e-8):
    n, d = X0.shape
    X = X0 / np.linalg.norm(X0, axis=1, keepdims=True)
    iu0, iu1 = np.triu_indices(n, 1)
    m1 = np.zeros_like(X)
    m2 = np.zeros_like(X)
    b1p = 1.0
    b2p = 1.0
    for t in range(steps):
        frac = t / max(1, steps - 1)
        beta = 50.0 * (8.0 ** frac)  # 50 -> 400
        G = X @ X.T
        gv = G[iu0, iu1]
        mx = float(gv.max())
        e = np.exp(beta * (gv - mx))
        S = float(e.sum())
        w = e / S
        W = np.zeros((n, n))
        W[iu0, iu1] = w
        W = W + W.T
        Gr = W @ X
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        b1p *= b1
        b2p *= b2
        m1 = b1 * m1 + (1.0 - b1) * Gt
        m2 = b2 * m2 + (1.0 - b2) * (Gt * Gt)
        m1h = m1 / (1.0 - b1p)
        m2h = m2 / (1.0 - b2p)
        step = lr * m1h / (np.sqrt(m2h) + eps)
        # keep tangential
        sd = np.sum(step * X, axis=1, keepdims=True)
        step = step - sd * X
        X = X - step
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X


def solve():
    t0 = time.time()
    DEADLINE = t0 + 150.0
    rng = np.random.default_rng(20261101)
    n, d = 78, 15

    bestX = None
    bestV = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (78, 15):
                continue
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            if not np.all(np.isfinite(a)):
                continue
            v = _max_cos(a)
            if v < bestV:
                bestV = v
                bestX = a.copy()
        except Exception:
            continue
    if bestX is None:
        bestX = rng.standard_normal((78, 15))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        bestV = _max_cos(bestX)
    incumbent = bestX.copy()
    incV = bestV

    def near_count(X, thr):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        return int(np.count_nonzero(G > thr))

    best_near = near_count(incumbent, incV - 1e-4)
    curX = incumbent.copy()
    curV = incV

    trial = 0
    tols = [3e-4, 5e-4, 8e-4, 1e-3]
    sigmas = [0.005, 0.008, 0.012, 0.015]
    while time.time() < DEADLINE - 2.0:
        trial += 1
        if time.time() > DEADLINE - 3.0:
            break
        base = incumbent if (trial % 2 == 1) else curX
        baseV = incV if (trial % 2 == 1) else curV
        tol = tols[(trial - 1) % len(tols)]
        sigma = sigmas[(trial - 1) % len(sigmas)]
        ssize = 10 + ((trial - 1) % 5)  # 10..14

        G = base @ base.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        adj = G > (m - tol)
        deg = adj.sum(axis=1)
        pmax = G.max(axis=1)
        # score: degree primarily, personal max secondarily
        order = np.lexsort((-pmax, -deg))
        # pick seed among top-5 (round-robin for diversity)
        top_pool = order[:5]
        seed = int(top_pool[(trial - 1) % len(top_pool)])
        # grow connected cluster by most-shared neighbours
        cluster = {seed}
        while len(cluster) < ssize:
            best_c = -1
            best_s = -1
            for v in range(n):
                if v in cluster:
                    continue
                # shared neighbours with cluster: count edges to cluster
                s = 0
                for u in cluster:
                    if adj[v, u]:
                        s += 1
                # require connectivity unless isolated
                if s == 0 and len(cluster) < n:
                    # allow only if no connected candidate exists
                    continue
                # tie-break by degree + pmax
                score = s * 1000 + int(deg[v]) + float(pmax[v])
                if score > best_s:
                    best_s = score
                    best_c = v
            if best_c < 0:
                # no connected candidate, pick highest-degree outside
                for v in order:
                    if int(v) not in cluster:
                        best_c = int(v)
                        break
                if best_c < 0:
                    break
            cluster.add(int(best_c))
        cluster = sorted(cluster)
        keep_idx = [i for i in range(n) if i not in set(cluster)]
        Xkept = base[keep_idx, :].copy()
        # jitter kept core to break exact ties
        noise = rng.standard_normal(Xkept.shape)
        Xkept = Xkept + sigma * noise
        Xkept /= np.linalg.norm(Xkept, axis=1, keepdims=True)

        # sequential greedy reinsertion from 8192 candidates
        K = len(keep_idx)
        cand = rng.standard_normal((8192, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        inserted = []
        Mk = Xkept.copy()
        for _ in range(len(cluster)):
            # max dot of each candidate vs Mk
            D = cand @ Mk.T
            vals = D.max(axis=1)
            bi = int(np.argmin(vals))
            chosen = cand[bi].copy()
            inserted.append(chosen)
            Mk = np.vstack([Mk, chosen[None, :]])
        Xinit = np.vstack([Xkept, np.array(inserted)])
        # quench
        Xq = _adam_quench(Xinit, steps=600, lr=0.015)
        vq = _max_cos(Xq)
        # update cur for diversity by near-contact count
        if vq < curV or (vq <= curV + 5e-6 and time.time() < DEADLINE - 10):
            # compute near count at its own level minus 1e-4
            nc = near_count(Xq, vq - 1e-4)
            cur_near = near_count(curX, curV - 1e-4)
            if vq < curV - 1e-12 or nc < cur_near:
                curX = Xq.copy()
                curV = vq
        if vq > 0.175:
            continue
        # LP polish only promising
        time_left = DEADLINE - time.time()
        if time_left < 4.0:
            break
        budget = min(7.0, time_left - 1.0)
        if budget < 1.0:
            break
        Xp, vp = _lp_polish(Xq, budget)
        if vp < incV - 1e-12:
            incumbent = Xp.copy()
            incV = vp
            curX = Xp.copy()
            curV = vp
            best_near = near_count(incumbent, incV - 1e-4)
        elif vp <= incV + 1e-9:
            nc = near_count(Xp, vp - 1e-4)
            if nc < best_near and vp < curV:
                curX = Xp.copy()
                curV = vp
        # early exit if excellent (well below record) to polish more
        if incV < 0.166 and time.time() < DEADLINE - 8:
            Xp2, vp2 = _lp_polish(incumbent, min(6.0, DEADLINE - time.time() - 1.0))
            if vp2 < incV:
                incumbent = Xp2
                incV = vp2
    incumbent = incumbent / np.linalg.norm(incumbent, axis=1, keepdims=True)
    return incumbent
