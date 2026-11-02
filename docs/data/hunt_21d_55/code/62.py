import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _refine_point(y0, Others, steps=250, lr0=0.012, lr1=0.0015):
    y = y0 / max(np.linalg.norm(y0), 1e-12)
    m = np.zeros_like(y)
    v = np.zeros_like(y)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        f = t / steps
        lr = lr0 * (lr1 / lr0) ** f
        dots = Others @ y
        z = (dots - 0.060) * 1000.0
        zm = z.max()
        e = np.exp(z - zm)
        s = e.sum()
        if not np.isfinite(s) or s <= 0:
            break
        w = e / s
        grad = w @ Others
        gt = grad - np.dot(grad, y) * y
        m = b1 * m + (1 - b1) * gt
        vv = b2 * v + (1 - b2) * gt * gt
        v = vv
        mhat = m / (1 - b1 ** t)
        vhat = v / (1 - b2 ** t)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        step = step - np.dot(step, y) * y
        y = y - step
        n = np.linalg.norm(y)
        if not np.isfinite(n) or n < 1e-12:
            break
        y = y / n
    return y

def _joint_quench(X0, steps=900, a0=40.0, a1=550.0, lr0=0.012, lr1=0.002):
    X = X0.copy()
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    n = X.shape[0]
    for t in range(1, steps + 1):
        f = t / steps
        alpha = a0 * (a1 / a0) ** f
        lr = lr0 * (lr1 / lr0) ** f
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        Mx = G.max()
        E = np.exp(alpha * (G - Mx))
        np.fill_diagonal(E, 0.0)
        S = E.sum()
        if not np.isfinite(S) or S <= 0:
            continue
        grad = (E @ X) / S
        proj = np.sum(grad * X, axis=1, keepdims=True)
        gt = grad - proj * X
        m = b1 * m + (1 - b1) * gt
        v = b2 * v + (1 - b2) * gt * gt
        mhat = m / (1 - b1 ** t)
        vhat = v / (1 - b2 ** t)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        # keep tangential
        sp = np.sum(step * X, axis=1, keepdims=True)
        step = step - sp * X
        X = X - step
        X = _norm(X)
    return X

def _lp_polish_deadline(x, deadline, radius_init=1e-3, max_it=60, cap=900):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = radius_init
    iu = np.triu_indices(n, 1)
    it = 0
    while time.time() < deadline and radius > 1e-14 and it < max_it:
        it += 1
        if time.time() >= deadline:
            break
        cfull = (x @ x.T)[iu]
        mm = cfull.max()
        sel = cfull > mm - max(20 * radius, 1e-9)
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > cap:
            idx = np.where(sel)[0]
            order = np.argsort(-cfull[idx])[:cap]
            nsel = np.zeros_like(sel)
            nsel[idx[order]] = True
            sel = nsel
        csel = cfull[sel]
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cidx = np.arange(d)
        cols = np.concatenate([I[:, None] * d + cidx, J[:, None] * d + cidx, np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        vv = _max_cos(y)
        if vv < best:
            x, best = y, vv
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    SEARCH_END = t0 + 132.0
    HARD_END = t0 + 170.0
    n, d = 55, 21
    base = None
    best_score = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (n, d):
                a = _norm(a)
                s = _max_cos(a)
                if s < best_score:
                    best_score = s
                    base = a.copy()
        except Exception:
            pass
    if base is None:
        rng0 = np.random.default_rng(0)
        base = _norm(rng0.standard_normal((n, d)))
        best_score = _max_cos(base)
    x_best = base.copy()
    cur_best = best_score

    # contact degrees of base for hub ordering
    def hub_order(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        mx = G.max()
        adj = G > mx - 1e-4
        deg = adj.sum(axis=1)
        return np.argsort(-deg), mx, adj
    hubs, _, _ = hub_order(x_best)
    hubs = hubs[:12].tolist()

    trial = 0
    # per-trial budget: 200k randoms in chunks
    CHUNK = 25000
    NCHUNK = 8  # 200k
    while time.time() < SEARCH_END and time.time() < HARD_END - 25:
        trial += 1
        if time.time() > SEARCH_END - 2:
            break
        hub = hubs[(trial - 1) % len(hubs)]
        K = 6 + ((trial - 1) % 5)  # 6..10
        # build BFS neighbourhood around hub using current x_best graph
        G = x_best @ x_best.T
        np.fill_diagonal(G, -2.0)
        mx = G.max()
        adj = G > mx - 1e-4
        # BFS
        seen = {int(hub)}
        queue = [int(hub)]
        order = [int(hub)]
        qi = 0
        # neighbor order by degree desc for determinism
        deg = adj.sum(axis=1)
        while len(order) < K and qi < len(queue):
            u = queue[qi]; qi += 1
            nbrs = np.where(adj[u])[0]
            nbrs = sorted(nbrs.tolist(), key=lambda v: -deg[v])
            for v in nbrs:
                if v not in seen:
                    seen.add(v)
                    queue.append(v)
                    order.append(v)
                    if len(order) >= K:
                        break
        if len(order) < K:
            # fill with highest-degree remaining
            for v in hubs:
                if v not in seen:
                    seen.add(v); order.append(v)
                    if len(order) >= K:
                        break
        delset = set(order[:K])
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[list(delset)] = False
        S = x_best[keep_mask]  # (Ns,d)
        Ns = S.shape[0]
        rng = np.random.default_rng(1000 + trial * 7919)
        # massive random candidates: keep best 2000 vs S
        pool_best_vals = None
        pool_best_cands = None
        # iterate chunks
        all_c = []
        all_v = []
        for ci in range(NCHUNK):
            cand = rng.standard_normal((CHUNK, d))
            cand = _norm(cand)
            dots = cand @ S.T
            md = dots.max(axis=1)
            # keep top 400 per chunk
            k = 400
            idx = np.argpartition(md, k)[:k]
            all_c.append(cand[idx])
            all_v.append(md[idx])
            if time.time() > SEARCH_END:
                break
        if len(all_c) == 0:
            continue
        C = np.vstack(all_c)
        # pairwise averages of survivors
        try:
            iu2 = np.triu_indices(Ns, 1)
            A = S[iu2[0]] + S[iu2[1]]
            A = _norm(A)
            C = np.vstack([C, A])
        except Exception:
            pass
        # small extra: survivors pairwise averages already; also add slightly perturbed survivors holes? skip
        # greedy insert
        Y_list = [S[i] for i in range(Ns)]
        Ymat = S.copy()
        # precompute C dots to Ymat will grow; do iterative
        used = np.zeros(C.shape[0], dtype=bool)
        for kk in range(K):
            if time.time() > SEARCH_END:
                break
            # score pool vs current Ymat
            dots = C @ Ymat.T
            md = dots.max(axis=1)
            md[used] = 1e9
            bi = int(np.argmin(md))
            used[bi] = True
            y0 = C[bi]
            # refine per-point Adam while keeping earlier fixed
            Yfixed = np.vstack(Y_list)
            steps = 220 + (kk % 3) * 60  # 220-340
            y1 = _refine_point(y0, Yfixed, steps=steps)
            Y_list.append(y1)
            Ymat = np.vstack([Ymat, y1[None, :]])
        # assemble full config: survivors + new points
        Xnew = np.vstack([S, np.vstack(Y_list[Ns:])])
        # light shuffle not needed; joint quench
        # vary quench params by trial
        if trial % 3 == 0:
            Xq = _joint_quench(Xnew, steps=700, a0=30.0, a1=450.0, lr0=0.010, lr1=0.002)
        elif trial % 3 == 1:
            Xq = _joint_quench(Xnew, steps=1000, a0=50.0, a1=600.0, lr0=0.012, lr1=0.0015)
        else:
            Xq = _joint_quench(Xnew, steps=850, a0=40.0, a1=500.0, lr0=0.015, lr1=0.002)
        s = _max_cos(Xq)
        if s < cur_best and np.isfinite(s):
            # gate LP: only if below 0.0675
            if s < 0.0675 and time.time() < HARD_END - 25:
                dl = min(time.time() + 2.5, HARD_END - 22)
                try:
                    Xp = _lp_polish_deadline(Xq, dl, radius_init=8e-4, max_it=25, cap=850)
                    sp = _max_cos(Xp)
                    if sp < s:
                        Xq, s = Xp, sp
                except Exception:
                    pass
            if s < cur_best:
                cur_best = s
                x_best = _norm(Xq)
                # refresh hub order
                try:
                    hubs, _, _ = hub_order(x_best)
                    hubs = hubs[:12].tolist()
                except Exception:
                    pass
        # early exit if big improvement? continue to use time
        if cur_best < 0.060:
            break

    # final LP polish at most ~20s but reserve hard end
    try:
        fin_dead = min(time.time() + 18.0, t0 + 176.0)
        if time.time() < fin_dead - 1:
            x_best = _lp_polish_deadline(x_best, fin_dead, radius_init=1e-3, max_it=60, cap=900)
    except Exception:
        pass
    x_best = _norm(x_best)
    return x_best
