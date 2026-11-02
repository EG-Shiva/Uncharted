import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lp_polish(X, budget, cap=1800, max_it=25):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    best_x = X.copy()
    radius = 5e-3
    iu0, iu1 = np.triu_indices(n, 1)
    nv = n * d + 1
    obj = np.zeros(nv); obj[-1] = 1.0
    start = time.time()
    it = 0
    while time.time() - start < budget and radius > 1e-13 and it < max_it:
        it += 1
        G = X @ X.T
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        thr = max(20.0 * radius, 1e-9)
        sel = c_all > m - thr
        nnz = int(np.count_nonzero(sel))
        if nnz == 0:
            radius *= 0.5
            continue
        if nnz > cap:
            idx = np.where(sel)[0]
            part = np.argpartition(-c_all[idx], cap)[:cap]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[part]] = True
            sel = mask
        I = iu0[sel]; J = iu1[sel]
        p = len(I)
        c_sel = c_all[sel]
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], axis=1).ravel()
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
        if not np.all(np.isfinite(yn)) or np.any(yn < 1e-12):
            radius *= 0.5
            continue
        y = y / yn
        v = _max_cos(y)
        if v < best:
            X = y; best = v; best_x = y.copy()
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return best_x, best

def _adam_quench(X0, steps=550, beta0=50.0, beta1=400.0, lr0=0.02, lr1=0.003, deadline=None):
    n, d = X0.shape
    Y = X0.copy()
    M = np.zeros_like(Y)
    V = np.zeros_like(Y)
    iu0, iu1 = np.triu_indices(n, 1)
    b1, b2 = 0.9, 0.999
    eps = 1e-8
    lbr = np.log(beta1 / beta0)
    llr = np.log(lr1 / lr0)
    W = np.zeros((n, n), dtype=np.float64)
    for s in range(steps):
        if deadline is not None and s % 100 == 0 and time.time() > deadline:
            break
        frac = s / max(steps - 1, 1)
        beta = beta0 * np.exp(lbr * frac)
        lr = lr0 * np.exp(llr * frac)
        G = Y @ Y.T
        gv = G[iu0, iu1]
        m = float(gv.max())
        e = np.exp(beta * (gv - m))
        S = float(e.sum())
        if not np.isfinite(S) or S <= 0:
            break
        w = e / S
        W.fill(0.0)
        W[iu0, iu1] = w
        W[iu1, iu0] = w
        Gr = W @ Y
        dot = np.sum(Gr * Y, axis=1, keepdims=True)
        Gt = Gr - dot * Y
        M = b1 * M + (1.0 - b1) * Gt
        V = b2 * V + (1.0 - b2) * (Gt * Gt)
        c1 = 1.0 - b1 ** (s + 1)
        c2 = 1.0 - b2 ** (s + 1)
        Mh = M / c1
        Vh = V / c2
        step = lr * Mh / (np.sqrt(Vh) + eps)
        # clip extreme steps for stability
        sn = np.linalg.norm(step)
        if sn > 0.1:
            step *= 0.1 / sn
        Y = Y - step
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return Y

def solve():
    t0 = time.time()
    HARD = t0 + 150.0
    n, d = 78, 15
    # load incumbent
    bestX = None; bestV = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.asarray(np.load(fn), dtype=np.float64)
            if a.shape != (78, 15):
                continue
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            if not np.all(np.isfinite(a)):
                continue
            v = _max_cos(a)
            if v < bestV:
                bestV = v; bestX = a.copy()
        except Exception:
            continue
    if bestX is None:
        rng0 = np.random.default_rng(0)
        bestX = rng0.standard_normal((78, 15))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        bestV = _max_cos(bestX)
    init_bestV = float(bestV)
    # parameters per trial
    tols = [5e-4, 1e-3, 7e-4, 3e-4, 1e-3, 5e-4, 8e-4, 1e-3, 4e-4, 6e-4, 1e-3, 5e-4]
    Ks = [12, 10, 14, 11, 13, 12, 10, 14, 12, 11, 10, 13]
    sigs = [0.008, 0.012, 0.005, 0.010, 0.015, 0.007, 0.012, 0.006, 0.014, 0.009, 0.011, 0.008]
    ntrials = 12
    iu0, iu1 = np.triu_indices(n, 1)
    for trial in range(ntrials):
        if time.time() > HARD - 18:
            break
        tol = float(tols[trial % len(tols)])
        K = int(Ks[trial % len(Ks)])
        sigma = float(sigs[trial % len(sigs)])
        rng = np.random.default_rng(1000 + trial * 77)
        # score on current best
        G = bestX @ bestX.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        adj = G > (m - tol)
        deg = adj.sum(axis=1)
        pmax = G.max(axis=1)
        # order by degree then pmax
        # diversify start among top-5
        order = np.lexsort((-pmax, -deg))
        top = order[:5]
        start = int(rng.choice(top))
        cluster = set([start])
        # grow connected cluster greedily
        while len(cluster) < K:
            best_c = -1; best_gain = -1
            # candidates = neighbors of cluster not in cluster; fallback to global high-degree
            neigh = set()
            for c in list(cluster):
                ns = np.where(adj[c])[0]
                for q in ns:
                    if q not in cluster:
                        neigh.add(int(q))
            if not neigh:
                # take highest degree outside
                for q in order:
                    if int(q) not in cluster:
                        neigh.add(int(q))
                        break
            # pick member with most edges into cluster (+ small noise to diversify)
            cands = list(neigh)
            gains = np.array([int(adj[q, list(cluster)].sum()) for q in cands])
            # add tiny random tie-break
            tie = rng.random(len(cands)) * 1e-6
            pick = int(cands[int(np.argmax(gains + tie))])
            cluster.add(pick)
        cluster = sorted(cluster)
        keep = np.array([i for i in range(n) if i not in set(cluster)], dtype=int)
        Xkeep = bestX[keep].copy()
        # jitter kept core to break exact ties
        Xkeep = Xkeep + sigma * rng.standard_normal(Xkeep.shape)
        Xkeep /= np.linalg.norm(Xkeep, axis=1, keepdims=True)
        # sequential greedy reinsertion from 8192 candidates
        NC = 8192
        C = rng.standard_normal((NC, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        placed = [Xkeep[i] for i in range(len(keep))]
        # pre-stack for speed: iteratively compute
        import math
        for k in range(K):
            if time.time() > HARD - 15:
                break
            S = np.stack(placed, axis=0)  # (M, d)
            D = C @ S.T  # (NC, M)
            md = D.max(axis=1)
            bi = int(np.argmin(md))
            placed.append(C[bi].copy())
            # optional: remove chosen to avoid duplicates (not needed)
        if len(placed) != n:
            continue
        Xtry = np.stack(placed, axis=0)
        # cheap tangent-Adam LSE quench
        steps = 550
        # reduce steps if time short
        if HARD - time.time() < 30:
            steps = 350
        Y = _adam_quench(Xtry, steps=steps, beta0=50.0, beta1=400.0, lr0=0.02, lr1=0.003, deadline=HARD - 12)
        vq = _max_cos(Y)
        # only LP-polish promising candidates
        if vq > 0.190:
            continue
        # near-contact count criterion (secondary) computed but polish anyway if below thresh
        budget = min(5.0, max(1.0, (HARD - 10 - time.time()) / 3.0))
        if budget < 0.8:
            break
        Yp, vp = _lp_polish(Y, budget=budget)
        if vp < bestV - 1e-12:
            bestX = Yp; bestV = vp
            # immediate keep; continue searching from improved basin
        # also track near-contact reduction for diagnostics (no acceptance unless better)
    # final safety: ensure finite, normalised
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    # if we somehow got worse (should not), fall back is already best
    return bestX
