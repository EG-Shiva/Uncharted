import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def lp_polish_deadline(x, deadline, r_init=0.004, cap=2600):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = max_cos(x)
    except Exception:
        return x
    radius = r_init
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        m = float(c_all.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > cap:
            idx = np.where(sel)[0]
            part = np.argpartition(-c_all[idx], cap)[:cap]
            keep = np.zeros_like(sel)
            keep[idx[part]] = True
            sel = keep
        I = iu0[0][sel]; J = iu0[1][sel]
        c_sel = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.empty((p, 2 * d + 1), dtype=int)
        cols[:, :d] = I[:, None] * d + np.arange(d)
        cols[:, d:2 * d] = J[:, None] * d + np.arange(d)
        cols[:, 2 * d] = n * d
        cols = cols.ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
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
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    SEARCH_END = t0 + 130.0
    HARD_END = t0 + 175.0
    rng = np.random.default_rng(0)
    n, d = 176, 9

    # load start
    Xb = None
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                nr = np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(nr > 1e-12):
                    Xb = (a / nr).astype(float)
                    break
        except Exception:
            pass
    if Xb is None:
        Xb = rng.standard_normal((n, d))
        Xb /= np.linalg.norm(Xb, axis=1, keepdims=True)
    base = Xb.copy()
    best_v = max_cos(base)
    best_X = base.copy()

    sizes = [12, 18, 24]
    it = 0
    # preallocate not needed
    while time.time() < SEARCH_END:
        # --- contact graph at max-1e-5 on current base ---
        G0 = base @ base.T
        np.fill_diagonal(G0, -2.0)
        cur_max = float(G0.max())
        thr = cur_max - 1e-5
        Adj = G0 > thr
        counts = Adj.sum(axis=1)
        order = np.argsort(-counts)
        # pick seed: cycle through top ranks + random among top 12
        ksize = sizes[it % len(sizes)]
        # deterministic varying seed
        if it % 4 == 0:
            seed = int(order[0])
        elif it % 4 == 1:
            seed = int(order[(it // 4) % min(15, n)])
        elif it % 4 == 2:
            top12 = order[:12]
            seed = int(top12[rng.integers(0, 12)])
        else:
            seed = int(rng.integers(0, n))
        # BFS community
        visited = np.zeros(n, dtype=bool)
        visited[seed] = True
        queue = [seed]
        # adjacency lists via boolean matrix rows
        while np.count_nonzero(visited) < ksize and len(queue) > 0:
            u = queue.pop(0)
            nbrs = np.where(Adj[u] & (~visited))[0]
            # visit higher-cosine neighbours first
            if len(nbrs) > 1:
                nbrs = nbrs[np.argsort(-G0[u, nbrs])]
            for w in nbrs:
                if np.count_nonzero(visited) >= ksize:
                    break
                visited[w] = True
                queue.append(int(w))
        # if component exhausted, grow by closest cosine
        while np.count_nonzero(visited) < ksize:
            vis_idx = np.where(visited)[0]
            unw_idx = np.where(~visited)[0]
            # score = max cosine to visited set
            sub = G0[np.ix_(unw_idx, vis_idx)].max(axis=1)
            pick = unw_idx[int(np.argmax(sub))]
            visited[pick] = True
        del_idx = np.where(visited)[0]
        keep_mask = ~visited
        F0 = base[keep_mask].copy()  # frozen, shape (n-k, d)

        # --- reinsert one at a time ---
        cur_list = [F0[i] for i in range(F0.shape[0])]
        # stack for fast matmul
        Fmat = F0.copy()
        new_pts = []
        aborted = False
        for ri in range(ksize):
            if time.time() >= SEARCH_END:
                aborted = True
                break
            # 20000 random candidates
            R = rng.standard_normal((20000, d))
            R /= np.linalg.norm(R, axis=1, keepdims=True)
            D = R @ Fmat.T  # 20000 x m
            mx = D.max(axis=1)
            # 20 smallest maxima
            if len(mx) > 20:
                pi = np.argpartition(mx, 20)[:20]
            else:
                pi = np.arange(len(mx))
            V = R[pi].copy()  # 20 x d
            # local tangent refinement of best 20, vectorized
            # 30 steps, alpha=80
            alpha = 80.0
            lr = 0.08
            for s in range(30):
                C = V @ Fmat.T  # 20 x m
                cm = C.max(axis=1, keepdims=True)
                E = np.exp(alpha * (C - cm))
                W = E / (E.sum(axis=1, keepdims=True) + 1e-300)
                Gr = W @ Fmat  # 20 x d
                dot = np.sum(Gr * V, axis=1, keepdims=True)
                Gt = Gr - dot * V
                V = V - lr * Gt
                V /= np.linalg.norm(V, axis=1, keepdims=True)
                lr *= 0.97
            # score refined: primary max, tie-break sum exp(80c)
            Cf = V @ Fmat.T
            mxf = Cf.max(axis=1)
            # tie-break
            tb = np.exp(80.0 * (Cf - mxf[:, None])).sum(axis=1)
            # lexicographic: pick min mxf, then min tb
            bi = int(np.argmin(mxf + 1e-9 * (tb / (tb.max() + 1e-300))))
            # more careful: sort by mxf then tb
            srt = np.lexsort((tb, mxf))
            bi = int(srt[0])
            vbest = V[bi] / np.linalg.norm(V[bi])
            new_pts.append(vbest)
            Fmat = np.vstack([Fmat, vbest[None, :]])
        if aborted:
            it += 1
            continue
        # assemble full config: frozen + new
        Xnew = np.vstack([F0, np.array(new_pts)])
        # restore approximate original ordering? not needed; but keep shape
        # shuffle to avoid ordering bias? keep as is
        # --- 300 steps normalized exponential-repulsion GD ---
        X = Xnew / np.linalg.norm(Xnew, axis=1, keepdims=True)
        beta0, beta1 = 25.0, 130.0
        lr0, lr1 = 0.02, 0.004
        for s in range(300):
            frac = s / 299.0
            beta = beta0 * (beta1 / beta0) ** frac
            lr_ = lr0 * (lr1 / lr0) ** frac
            G = X @ X.T
            np.fill_diagonal(G, -10.0)
            m = float(G.max())
            W = np.exp((G - m) * beta)
            np.fill_diagonal(W, 0.0)
            Gr = W @ X
            proj = np.sum(Gr * X, axis=1, keepdims=True)
            Gt = Gr - proj * X
            gn = float(np.linalg.norm(Gt) / np.sqrt(n))
            if not np.isfinite(gn) or gn < 1e-16:
                continue
            X = X - (lr_ / gn) * Gt
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        v = max_cos(X)
        if v < best_v and np.all(np.isfinite(X)):
            best_v = v
            best_X = X.copy()
            base = X.copy()
        # else keep base as best (elitist); occasionally move base to Xnew region?
        # every 25 iters without improvement, randomize base slightly from best to diversify seeds
        it += 1
        if it % 40 == 0 and time.time() < SEARCH_END:
            # small global jitter of base to get new contact graph (does not change best)
            J = base + 0.002 * rng.standard_normal((n, d))
            J /= np.linalg.norm(J, axis=1, keepdims=True)
            base = J
            # re-anchor base to best if jitter hurts a lot? keep jitter only for graph seeds;
            # reset base to best every other cycle
            if it % 80 == 0:
                base = best_X.copy()

    # LP polish only single best screened candidate
    if time.time() < HARD_END - 2.0:
        best_X = lp_polish_deadline(best_X, HARD_END - 1.0)
    best_X = best_X / np.linalg.norm(best_X, axis=1, keepdims=True)
    return best_X
