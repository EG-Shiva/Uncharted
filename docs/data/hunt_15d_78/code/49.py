import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lp_polish(X, deadline, r_init=0.006, r_max=0.03, cap=1500, max_it=40):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    best_x = X.copy()
    radius = r_init
    iu0, iu1 = np.triu_indices(n, 1)
    nv = n * d + 1
    obj = np.zeros(nv); obj[-1] = 1.0
    it = 0
    while time.time() + 0.5 < deadline and radius > 1e-13 and it < max_it:
        it += 1
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
        I = iu0[sel]; J = iu1[sel]
        p = len(I)
        c_sel = c_all[sel]
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], axis=1).ravel()
        rows = np.repeat(np.arange(p), 2 * d + 1)
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            bounds = [(-radius, radius)] * (n * d) + [(None, None)]
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
        if not np.all(np.isfinite(yn)) or np.any(yn < 1e-12):
            radius *= 0.5
            continue
        y = y / yn
        v = _max_cos(y)
        if v < best:
            X = y; best = v; best_x = y.copy()
            radius = min(radius * 1.5, r_max)
        else:
            radius *= 0.5
    return best_x, best

def _adam_quench(X0, steps=600, lr=0.02):
    n, d = X0.shape
    X = X0 / np.linalg.norm(X0, axis=1, keepdims=True)
    iu0, iu1 = np.triu_indices(n, 1)
    M = np.zeros_like(X)
    V = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    for t in range(1, steps + 1):
        frac = (t - 1) / max(1, steps - 1)
        beta = 50.0 + (400.0 - 50.0) * frac
        G = X @ X.T
        gv = G[iu0, iu1]
        m = float(gv.max())
        e = np.exp(beta * (gv - m))
        s = float(e.sum())
        w = e / s
        W = np.zeros((n, n), dtype=np.float64)
        W[iu0, iu1] = w
        W = W + W.T
        Gr = W @ X
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        b1p *= b1; b2p *= b2
        M = b1 * M + (1.0 - b1) * Gt
        V = b2 * V + (1.0 - b2) * (Gt * Gt)
        Mh = M / (1.0 - b1p)
        Vh = V / (1.0 - b2p)
        step = lr * Mh / (np.sqrt(Vh) + eps)
        # keep tangent: remove radial part of step
        sdot = np.sum(step * X, axis=1, keepdims=True)
        step = step - sdot * X
        # clip large steps
        sn = np.linalg.norm(step, axis=1, keepdims=True)
        clip = 0.05
        too = sn > clip
        step = np.where(too, step * (clip / np.maximum(sn, 1e-12)), step)
        X = X - step
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X

def solve():
    t0 = time.time()
    HARD = t0 + 150.0
    n, d = 78, 15
    bestX = None
    bestV = 1e9
    for fn in ("best.npy", "record.npy"):
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
        rng0 = np.random.default_rng(0)
        bestX = rng0.standard_normal((78, 15))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        bestV = _max_cos(bestX)
    incumbentX = bestX.copy()
    incumbentV = bestV
    iu0, iu1 = np.triu_indices(n, 1)

    n_trials = 10
    for trial in range(n_trials):
        if time.time() + 6.0 > HARD:
            break
        # parameters per trial
        k = 10 + (trial % 5)  # 10..14
        if trial % 3 == 0:
            tol = 1e-4
        elif trial % 3 == 1:
            tol = 5e-4
        else:
            tol = 1e-3
        sigma = 0.005 + 0.0025 * (trial % 5)  # 0.005..0.015
        # contact degrees with loose tolerance
        G = bestX @ bestX.T
        np.fill_diagonal(G, -2.0)
        mcur = float(G.max())
        adj = G > (mcur - tol)
        np.fill_diagonal(adj, False)
        deg = adj.sum(axis=1).astype(float)
        pmax = G.max(axis=1)
        # score: degree + small tie-break by personal max + deterministic jitter
        rng_s = np.random.default_rng(777 + trial * 131)
        tie = rng_s.standard_normal(n) * 1e-6
        score = deg + 2.0 * np.clip((pmax - (mcur - tol)) / max(tol, 1e-12), 0, 1) * 0.1 + tie
        # pick start among top-3 for diversity
        order = np.argsort(-score, kind="stable")
        start = int(order[trial % min(3, n)])
        cluster = [start]
        in_cl = np.zeros(n, dtype=bool)
        in_cl[start] = True
        while len(cluster) < k:
            # neighbours of cluster
            conn = np.zeros(n, dtype=int)
            for c in cluster:
                conn += adj[c].astype(int)
            conn[in_cl] = -1
            if np.all(conn < 0):
                # disconnected: pick highest score outside
                for o in order:
                    if not in_cl[o]:
                        cluster.append(int(o)); in_cl[int(o)] = True
                        break
                continue
            # shared count primary, degree secondary
            cand = np.where(conn >= 0)[0]
            # most-shared neighbours first
            best_c = -1; best_key = (-1, -1)
            # add small random tie break
            perm = rng_s.permutation(cand)
            for c in perm:
                key = (int(conn[c]), float(score[c]))
                if key > best_key:
                    best_key = key; best_c = int(c)
            cluster.append(best_c); in_cl[best_c] = True
        keep = np.where(~in_cl)[0]
        n_keep = len(keep)
        Xkeep = bestX[keep].copy()
        # jitter kept core to break exact ties
        rng = np.random.default_rng(1000 + trial * 977)
        Xkeep = Xkeep + sigma * rng.standard_normal(Xkeep.shape)
        Xkeep /= np.linalg.norm(Xkeep, axis=1, keepdims=True)
        # sequential greedy reinsertion from 8192 candidates
        Nc = 8192
        C = rng.standard_normal((Nc, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        placed = []
        # precompute kept for scoring
        cur = Xkeep.copy()  # will grow
        for s in range(k):
            # D = C @ cur.T -> (Nc, ncur), score = max dot
            D = C @ cur.T
            scores = D.max(axis=1)
            idx = int(np.argmin(scores))
            chosen = C[idx].copy()
            placed.append(chosen)
            cur = np.vstack([cur, chosen[None, :]])
            # optional: remove chosen to avoid duplicate (set its score high)
            # replace that candidate with a fresh one to keep diversity
            nr = rng.standard_normal(d)
            nr /= np.linalg.norm(nr)
            C[idx] = nr
            if time.time() + 4.0 > HARD:
                break
        X0 = np.vstack([Xkeep, np.array(placed)])
        # X0 order doesn't matter; keep size 78
        if X0.shape[0] != 78:
            continue
        # cheap tangent-Adam LSE quench
        try:
            Xq = _adam_quench(X0, steps=600, lr=0.02)
        except Exception:
            continue
        vq = _max_cos(Xq)
        if time.time() + 4.0 > HARD:
            # still track if better without polish
            if vq < bestV:
                bestV = vq; bestX = Xq.copy()
            continue
        # only LP-polish promising candidates
        if vq > 0.175:
            continue
        if vq > bestV + 0.01:
            continue
        # count near contacts at t-1e-4 for tie-break info (not strictly needed)
        budget = min(7.0, HARD - time.time() - 1.0)
        if budget < 1.0:
            if vq < bestV:
                bestV = vq; bestX = Xq.copy()
            continue
        Xp, vp = _lp_polish(Xq, time.time() + budget)
        if vp < bestV - 1e-12:
            bestV = vp; bestX = Xp.copy()
        elif vp < bestV:
            bestV = vp; bestX = Xp.copy()
        # also accept by reduced near-contact count if equal-ish
        # (light check, no extra polish)
    # safety: return incumbent if nothing improved (still return best found)
    if bestX is None:
        bestX = incumbentX
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    # final tiny shrink not needed; ensure finite
    bestX = np.where(np.isfinite(bestX), bestX, incumbentX)
    rn = np.linalg.norm(bestX, axis=1, keepdims=True)
    rn = np.maximum(rn, 1e-12)
    bestX = bestX / rn
    return bestX
