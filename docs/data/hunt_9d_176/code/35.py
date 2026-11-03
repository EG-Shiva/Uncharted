import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _repulse300(X, beta=80.0, lr=0.015, steps=300, deadline=1e18):
    n = X.shape[0]
    for s in range(steps):
        if (s % 50 == 0) and time.time() >= deadline:
            break
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        try:
            m = float(np.max(G))
        except Exception:
            break
        W = np.exp((G - m) * beta)
        np.fill_diagonal(W, 0.0)
        Grad = W @ X
        proj = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - proj * X
        try:
            gnorm = float(np.linalg.norm(Gt) / np.sqrt(n))
        except Exception:
            break
        if not np.isfinite(gnorm) or gnorm < 1e-16:
            continue
        X = X - (lr / gnorm) * Gt
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[~np.isfinite(nr)] = 1.0
        nr[nr < 1e-12] = 1.0
        X = X / nr
    return X

def _bfs_community(X, k, seed, thresh):
    n = X.shape[0]
    G = X @ X.T
    # threshold adjacency
    Adj = G >= thresh
    # degrees excluding self
    # BFS
    visited = np.zeros(n, dtype=bool)
    visited[seed] = True
    queue = [int(seed)]
    qpos = 0
    # precompute neighbor lists in order of closeness to seed? just index order
    # for speed get neighbor indices on fly
    while len(np.where(visited)[0]) < k and qpos < len(queue):
        u = queue[qpos]; qpos += 1
        nbrs = np.where(Adj[u])[0]
        # sort nbrs by G[u,nbrs] descending for more coherent cluster
        if len(nbrs) > 1:
            try:
                o = np.argsort(-G[u, nbrs])
                nbrs = nbrs[o]
            except Exception:
                pass
        for v in nbrs:
            v = int(v)
            if not visited[v]:
                visited[v] = True
                queue.append(v)
                if np.count_nonzero(visited) >= k:
                    break
    idx = np.where(visited)[0]
    if len(idx) > k:
        # keep seed + closest k-1 by G[seed]
        rest = idx[idx != seed]
        try:
            o = np.argsort(-G[seed, rest])
            rest = rest[o[:k-1]]
        except Exception:
            rest = rest[:k-1]
        idx = np.concatenate([[seed], rest])
    if len(idx) < k:
        # fill with highest contact counts
        deg = np.count_nonzero(Adj, axis=1)
        order = np.argsort(-deg)
        for v in order:
            if not visited[v]:
                visited[v] = True
                if np.count_nonzero(visited) >= k:
                    break
        idx = np.where(visited)[0][:k]
    return np.asarray(idx, dtype=int)

def _lp_polish_deadline(x, deadline, r_init=0.003, max_pairs=2500):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = r_init
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        try:
            m = float(c_all.max())
        except Exception:
            break
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > max_pairs:
            idx_all = np.where(sel)[0]
            # keep largest max_pairs
            try:
                order = np.argpartition(-c_all[idx_all], max_pairs)[:max_pairs]
            except Exception:
                break
            keep = np.zeros_like(sel)
            keep[idx_all[order]] = True
            sel = keep
        I = iu0[0][sel]; J = iu0[1][sel]
        c_sel = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.empty((p, 2 * d + 1), dtype=int)
        try:
            cols[:, :d] = I[:, None] * d + np.arange(d)
            cols[:, d:2*d] = J[:, None] * d + np.arange(d)
        except Exception:
            break
        cols[:, 2*d] = n * d
        cols = cols.ravel()
        try:
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        if not np.all(np.isfinite(yn)) or np.any(yn < 1e-12):
            radius *= 0.5
            continue
        y = y / yn
        try:
            v = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    SEARCH_END = t0 + 130.0
    HARD_END = t0 + 176.0
    rng = np.random.default_rng(0)
    n, d = 176, 9
    NCAND = 20000
    # load starts
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                nr = np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(nr > 1e-12):
                    cands.append(a / nr)
        except Exception:
            pass
    if len(cands) == 0:
        X0 = rng.standard_normal((n, d))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        cands.append(X0)
    vals = [_max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    best_X = cands[bi].copy()
    best_v = float(vals[bi])
    sizes = [12, 18, 24]
    trial = -1
    # search loop, no LP inside
    while True:
        if time.time() >= SEARCH_END:
            break
        trial += 1
        k = sizes[trial % 3]
        # contact graph at max-1e-5 on best_X
        try:
            Gb = best_X @ best_X.T
            np.fill_diagonal(Gb, -2.0)
            cur_max = float(np.max(Gb))
        except Exception:
            continue
        thresh = cur_max - 1e-5
        # degrees
        try:
            # need diag 1 for degree; recompute without fill? use Gb+diag
            deg = np.count_nonzero(Gb >= thresh, axis=1)
            order_deg = np.argsort(-deg)
        except Exception:
            order_deg = np.arange(n)
        rank = (trial // 3) % 8
        # vary seed among top-8 by stride to get different communities
        try:
            seed = int(order_deg[(rank * 3 + trial) % min(8, n)])
        except Exception:
            seed = int(order_deg[0])
        if time.time() >= SEARCH_END:
            break
        try:
            comm = _bfs_community(best_X, int(k), int(seed), float(thresh))
        except Exception:
            continue
        if len(comm) != k:
            try:
                k = len(comm)
            except Exception:
                continue
        mask = np.ones(n, dtype=bool)
        mask[comm] = False
        try:
            F0 = best_X[mask].copy()
        except Exception:
            continue
        F = F0  # (n-k) x d, will grow
        aborted = False
        # sequential reinsertion
        for pos in range(int(k)):
            if time.time() >= SEARCH_END:
                aborted = True
                break
            mcur = F.shape[0]
            # 20000 random sphere candidates
            try:
                R = rng.standard_normal((NCAND, d))
                rn = np.linalg.norm(R, axis=1, keepdims=True)
                rn[rn < 1e-12] = 1.0
                R /= rn
            except Exception:
                aborted = True
                break
            try:
                dots = R @ F.T  # NCAND x mcur
                cmax = dots.max(axis=1)
            except Exception:
                aborted = True
                break
            # cheap screening: best 100 by cmax
            N100 = 100
            try:
                idx100 = np.argpartition(cmax, N100)[:N100]
            except Exception:
                aborted = True
                break
            try:
                d100 = dots[idx100]  # 100 x mcur
                c100 = cmax[idx100]
                # tie-break sum exp(80c) shifted for stability
                E100 = np.exp(80.0 * (d100 - c100[:, None]))
                S100 = E100.sum(axis=1)
                lex = np.lexsort((S100, c100))
                idx20_rel = lex[:20]
                idx20 = idx100[idx20_rel]
                V = R[idx20].copy()  # 20 x d
            except Exception:
                try:
                    idx20 = np.argpartition(cmax, 20)[:20]
                    V = R[idx20].copy()
                except Exception:
                    aborted = True
                    break
            # local tangent refinement of best 20, vectorized, 25 steps
            try:
                lr_ref = 0.03
                for it in range(25):
                    D = V @ F.T  # 20 x mcur
                    M = D.max(axis=1, keepdims=True)
                    E = np.exp(80.0 * (D - M))
                    S = E.sum(axis=1, keepdims=True)
                    S[~np.isfinite(S)] = 1.0
                    S[S <= 0] = 1.0
                    Grad = (E @ F) / S  # 20 x d
                    dot = np.sum(Grad * V, axis=1, keepdims=True)
                    Gt = Grad - dot * V
                    V = V - lr_ref * Gt
                    vn = np.linalg.norm(V, axis=1, keepdims=True)
                    vn[~np.isfinite(vn)] = 1.0
                    vn[vn < 1e-12] = 1.0
                    V = V / vn
                Df = V @ F.T
                cf = Df.max(axis=1)
                Ef = np.exp(80.0 * (Df - cf[:, None]))
                Sf = Ef.sum(axis=1)
                lex2 = np.lexsort((Sf, cf))
                best_v_cand = V[lex2[0]]
            except Exception:
                # fallback to unrefined best
                try:
                    pick = int(np.argmin(cmax))
                    best_v_cand = R[pick]
                except Exception:
                    aborted = True
                    break
            try:
                F = np.vstack([F, best_v_cand[None, :]])
            except Exception:
                aborted = True
                break
            if time.time() >= SEARCH_END:
                aborted = True
                break
        if aborted:
            # discard incomplete trial, continue if time remains
            continue
        # F is now n x d
        try:
            Xnew = F.copy()
            # only 300 steps of normalised exponential-repulsion GD
            Xnew = _repulse300(Xnew, beta=80.0, lr=0.015, steps=300, deadline=SEARCH_END)
            vnew = _max_cos(Xnew)
        except Exception:
            continue
        if not np.isfinite(vnew):
            continue
        if vnew < best_v and time.time() < SEARCH_END + 1.0:
            best_v = float(vnew)
            best_X = Xnew.copy()
        # check time every iteration (while condition does too)
        if time.time() >= SEARCH_END:
            break
    # LP polish only single best screened candidate for remaining time
    try:
        if time.time() < HARD_END - 1.0:
            best_X = _lp_polish_deadline(best_X, HARD_END - 1.0, r_init=0.003)
    except Exception:
        pass
    try:
        best_X = best_X / np.linalg.norm(best_X, axis=1, keepdims=True)
    except Exception:
        pass
    return best_X
