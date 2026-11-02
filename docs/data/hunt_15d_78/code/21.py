import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def _lp_polish(X, budget, deadline):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < budget and time.time() < deadline and radius > 1e-14:
        Gf = X @ X.T
        c = Gf[iu]
        m = float(c.max())
        thr = max(20.0 * radius, 1e-9)
        sel = c > m - thr
        if not np.any(sel):
            break
        I = iu[0][sel]; J = iu[1][sel]
        p = len(I)
        if p == 0:
            break
        # cap constraints for speed
        if p > 1500:
            # keep the tightest 1500
            idx = np.argpartition(-c[sel], 1500)[:1500]
            I = I[idx]; J = J[idx]
            p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel() if False else np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel][:p] if p != int(sel.sum()) else -c[sel],
                          A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"presolve": True})
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


def _fingerprint(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    C = (G > m - 1e-6)
    deg = C.sum(axis=1).astype(float)
    deg_sorted = np.sort(deg)
    # Gram spectrum (with unit diagonal)
    G2 = X @ X.T
    try:
        ev = np.linalg.eigvalsh(G2)
    except Exception:
        ev = np.zeros(X.shape[0])
    ev_sorted = np.sort(ev)[::-1]
    personal = G.max(axis=1)
    return deg_sorted, ev_sorted, m, personal, deg


def solve():
    t0 = time.time()
    LIMIT = 160.0
    deadline = t0 + LIMIT - 1.0
    n, d = 78, 15

    # incumbent checkpoint (never destroyed)
    bestX = None
    bestV = 1e9
    recX = None
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (78, 15):
                continue
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            v = _max_cos(a)
            if fn == "record.npy":
                recX = a.copy()
            if v < bestV:
                bestV = v
                bestX = a.copy()
        except Exception:
            continue
    if bestX is None:
        rng0 = np.random.default_rng(0)
        bestX = rng0.standard_normal((n, d))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        bestV = _max_cos(bestX)
    if recX is None:
        recX = bestX.copy()
    checkpointX = bestX.copy()
    checkpointV = float(bestV)

    try:
        R = 12
        STEPS = 3000
        rng = np.random.default_rng(20261101)
        Xs = rng.standard_normal((R, n, d))
        Xs /= np.linalg.norm(Xs, axis=2, keepdims=True)

        m_ad = np.zeros_like(Xs)
        v_ad = np.zeros_like(Xs)
        b1, b2, eps = 0.9, 0.999, 1e-8
        lr0 = 0.02
        T0, T1 = 0.03, 0.002
        logr = np.log(T1 / T0)

        # batched Adam quench, vectorised across replicas
        for k in range(1, STEPS + 1):
            if time.time() > deadline - 45.0:
                break
            frac = (k - 1) / max(1, STEPS - 1)
            T = float(T0 * np.exp(logr * frac))
            lr = float(lr0 * (1.0 - 0.4 * frac))
            # Gram batches
            G = np.matmul(Xs, Xs.transpose(0, 2, 1))
            # mask diagonal via -2
            # use advanced indexing to set diag
            for r in range(R):
                np.fill_diagonal(G[r], -2.0)
            m = G.max(axis=(1, 2))
            # stable weights
            # clip to avoid overflow: (G-m)/T <= 0
            W = np.exp((G - m[:, None, None]) / T)
            for r in range(R):
                np.fill_diagonal(W[r], 0.0)
            S = W.sum(axis=(1, 2)) / 2.0
            S = np.maximum(S, 1e-12)
            Gr = np.matmul(W, Xs) / S[:, None, None]
            # tangent projection
            dot = (Gr * Xs).sum(axis=2, keepdims=True)
            Gt = Gr - dot * Xs
            # Adam
            m_ad = b1 * m_ad + (1 - b1) * Gt
            v_ad = b2 * v_ad + (1 - b2) * (Gt * Gt)
            bc1 = 1 - b1 ** k
            bc2 = 1 - b2 ** k
            mh = m_ad / bc1
            vh = v_ad / bc2
            step = mh / (np.sqrt(vh) + eps)
            Xs = Xs - lr * step
            Xs /= np.linalg.norm(Xs, axis=2, keepdims=True)

        # scores
        scores = np.empty(R)
        for r in range(R):
            scores[r] = _max_cos(Xs[r])

        # fingerprints
        dr, er, mr, _, _ = _fingerprint(recX)
        deg_list = []
        ev_list = []
        pers_list = []
        dist_deg = np.empty(R)
        dist_spec = np.empty(R)
        for r in range(R):
            dd, ee, mm, pp, _raw = _fingerprint(Xs[r])
            deg_list.append(dd); ev_list.append(ee); pers_list.append(pp)
            dist_deg[r] = float(np.linalg.norm(dd - dr))
            dist_spec[r] = float(np.linalg.norm(ee - er))

        # discard record-like: need degree distance >= threshold
        THR = 5.0
        keep = np.where(dist_deg >= THR)[0]
        # also require spectral novelty: at least small distance
        # (record spectrum is distinctive; keep as secondary filter)
        # if too few, take most distant ones
        if len(keep) < 2:
            order = np.argsort(-dist_deg)
            keep = order[:2]
        # among keep, pick best two by score with novelty bonus
        # rank by score primarily
        korder = keep[np.argsort(scores[keep])]
        sel = korder[:2]
        # update checkpoint with best raw replica if better than incumbent
        # (even before polish, keep safety)
        for r in sel:
            if float(scores[r]) < checkpointV and time.time() < deadline:
                checkpointV = float(scores[r])
                checkpointX = (Xs[r] / np.linalg.norm(Xs[r], axis=1, keepdims=True)).copy()

        # LP-polish only the two distinct survivors
        polished = []
        polished_scores = []
        for s in sel:
            if time.time() > deadline - 25.0:
                break
            # budget <=20s each, leave room for crossover
            remain = deadline - time.time()
            if remain < 3.0:
                break
            budget = min(15.0, remain - 12.0 if len(polished) == 0 else remain - 6.0)
            budget = max(2.0, min(15.0, budget))
            try:
                Xp = _lp_polish(Xs[s], budget, deadline - 6.0)
            except Exception:
                Xp = Xs[s]
            vp = _max_cos(Xp)
            polished.append(Xp)
            polished_scores.append(vp)
            if vp < checkpointV:
                checkpointV = vp
                checkpointX = Xp.copy()

        # crossover: 39 lowest-personal-max from each after Procrustes
        if len(polished) >= 2 and time.time() < deadline - 6.0:
            try:
                # order by polished score
                o = np.argsort(np.array(polished_scores))
                A = polished[o[0]]
                B = polished[o[1]]
                # personal max
                GA = A @ A.T; np.fill_diagonal(GA, -2.0); pa = GA.max(axis=1)
                GB = B @ B.T; np.fill_diagonal(GB, -2.0); pb = GB.max(axis=1)
                idxA = np.argsort(pa)[:39]
                idxB = np.argsort(pb)[:39]
                # orthogonal Procrustes: align B to A globally
                M = B.T @ A
                U, _, Vt = np.linalg.svd(M)
                Q = U @ Vt
                Bal = B @ Q
                Xc = np.concatenate([A[idxA], Bal[idxB]], axis=0)
                Xc /= np.linalg.norm(Xc, axis=1, keepdims=True)
                remain = deadline - time.time()
                if remain > 3.0:
                    Xc = _lp_polish(Xc, min(12.0, remain - 1.0), deadline)
                    vc = _max_cos(Xc)
                    if vc < checkpointV:
                        checkpointV = vc
                        checkpointX = Xc.copy()
            except Exception:
                pass

        bestX = checkpointX
        bestV = checkpointV
    except Exception:
        # on any failure keep incumbent
        pass

    # sanitize + return
    try:
        bestX = np.asarray(bestX, dtype=np.float64)
        bestX = np.where(np.isnan(bestX), 0.0, bestX)
        nr = np.linalg.norm(bestX, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        bestX = bestX / nr
    except Exception:
        rngf = np.random.default_rng(1)
        bestX = rngf.standard_normal((78, 15))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX
