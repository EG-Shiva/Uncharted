import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _refine_single(y0, Y, steps=300, a0=30.0, a1=500.0, lr0=0.03, lr1=0.003):
    d = y0.shape[0]
    y = y0.copy()
    m = np.zeros(d)
    v = np.zeros(d)
    b1, b2, eps = 0.9, 0.999, 1e-8
    la0 = np.log(a0); la1 = np.log(a1)
    llr0 = np.log(lr0); llr1 = np.log(lr1)
    for t in range(1, steps + 1):
        f = t / steps
        alpha = np.exp(la0 + (la1 - la0) * f)
        lr = np.exp(llr0 + (llr1 - llr0) * f)
        dots = Y @ y
        mx = np.max(dots)
        e = np.exp(alpha * (dots - mx))
        s = np.sum(e)
        if not np.isfinite(s) or s <= 0:
            continue
        w = e / s
        grad = w @ Y
        gd = np.dot(grad, y)
        gt = grad - gd * y
        m = b1 * m + (1.0 - b1) * gt
        v = b2 * v + (1.0 - b2) * (gt * gt)
        mhat = m / (1.0 - b1 ** t)
        vhat = v / (1.0 - b2 ** t)
        y = y - lr * mhat / (np.sqrt(vhat) + eps)
        ny = np.linalg.norm(y)
        if ny < 1e-12:
            y = y0
        else:
            y = y / ny
    return y

def _lp_polish_deadline(x, deadline):
    x = _norm(x)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = 1e-3
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        try:
            G = x @ x.T
            cfull = G[iu0]
            m = float(np.max(cfull))
            sel = cfull > m - max(20.0 * radius, 1e-9)
            cnt = int(np.count_nonzero(sel))
            if cnt == 0:
                break
            if cnt > 1400:
                idx = np.where(sel)[0]
                order = np.argsort(-cfull[idx])[:1400]
                nsel = np.zeros_like(sel)
                nsel[idx[order]] = True
                sel = nsel
            csel = cfull[sel]
            I = iu0[0][sel]; J = iu0[1][sel]
            p = len(I)
            if p == 0:
                break
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d),
                                   J[:, None] * d + np.arange(d),
                                   np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            y = x + res.x[:-1].reshape(n, d)
            y = _norm(y)
            vv = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if vv < best:
            x, best = y, vv
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    T_SEARCH_END = t0 + 145.0
    T_FINAL_DEADLINE = t0 + 165.0
    rng = np.random.default_rng(0)
    n, d = 55, 21
    xb = None
    bscore = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (n, d):
                a = _norm(a)
                s = _max_cos(a)
                if s < bscore:
                    bscore = s
                    xb = a.copy()
        except Exception:
            continue
    if xb is None:
        xb = _norm(rng.standard_normal((n, d)))
        bscore = _max_cos(xb)
    x_best = xb.copy()
    best = float(bscore)

    NCAND = 20000
    trial = 0
    # preallocate iu not needed
    while time.time() < T_SEARCH_END:
        trial += 1
        if time.time() > T_SEARCH_END - 0.5:
            break
        # contact graph with threshold best-1e-6
        try:
            G = x_best @ x_best.T
            np.fill_diagonal(G, -2.0)
            thr = best - 1e-6
            adj = G > thr
            np.fill_diagonal(adj, False)
            counts = np.sum(adj, axis=1)
            hub = int(np.argmax(counts))
            # cluster size 6-9
            k = 6 + int(rng.integers(0, 4))  # 6..9
            # BFS
            visited = [hub]
            in_vis = np.zeros(n, dtype=bool)
            in_vis[hub] = True
            queue = [hub]
            qi = 0
            while len(visited) < k and qi < len(queue):
                u = queue[qi]; qi += 1
                neigh = np.where(adj[u])[0]
                if len(neigh) > 1:
                    perm = rng.permutation(len(neigh))
                    neigh = neigh[perm]
                for vv in neigh:
                    if not in_vis[vv]:
                        in_vis[vv] = True
                        visited.append(int(vv))
                        queue.append(int(vv))
                        if len(visited) >= k:
                            break
                if len(visited) >= k:
                    break
            # if graph too sparse, fill randomly
            while len(visited) < k:
                c = int(rng.integers(0, n))
                if not in_vis[c]:
                    in_vis[c] = True
                    visited.append(c)
            visited = np.array(visited, dtype=int)
            # randomize hub choice occasionally for diversity
            # (mostly highest hub, sometimes second best)
            if trial % 7 == 0:
                # pick random hub among top-5
                top = np.argsort(-counts)[:5]
                hub2 = int(top[rng.integers(0, 5)])
                # rebuild quickly centered at hub2 with same k
                visited2 = [hub2]
                in2 = np.zeros(n, dtype=bool); in2[hub2] = True
                q2 = [hub2]; qi2 = 0
                while len(visited2) < k and qi2 < len(q2):
                    u = q2[qi2]; qi2 += 1
                    neigh = np.where(adj[u])[0]
                    if len(neigh) > 1:
                        neigh = neigh[rng.permutation(len(neigh))]
                    for vv in neigh:
                        if not in2[vv]:
                            in2[vv] = True
                            visited2.append(int(vv)); q2.append(int(vv))
                            if len(visited2) >= k:
                                break
                while len(visited2) < k:
                    c = int(rng.integers(0, n))
                    if not in2[c]:
                        in2[c] = True; visited2.append(c)
                visited = np.array(visited2, dtype=int)
                in_vis = in2
        except Exception:
            continue
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[visited] = False
        X_keep = x_best[keep_mask]  # (n-k, d)
        # reinsert one at a time
        Y_list = [X_keep[i] for i in range(X_keep.shape[0])]
        # order of reinsertion: random perm of visited
        order = visited[rng.permutation(len(visited))]
        new_pts = {}
        ok = True
        for _vi in order:
            if time.time() > T_SEARCH_END:
                ok = False
                break
            try:
                Y = np.stack(Y_list, axis=0) if len(Y_list) > 0 else X_keep[:0]
                # sample 20000
                cand = rng.standard_normal((NCAND, d))
                cand = _norm(cand)
                dots = cand @ Y.T
                md = np.max(dots, axis=1)
                bi = int(np.argmin(md))
                y0 = cand[bi]
                # refine 300 steps
                y1 = _refine_single(y0, Y, steps=300)
                Y_list.append(y1)
                new_pts[int(_vi)] = y1
            except Exception:
                ok = False
                break
        if not ok:
            continue
        try:
            x_cand = x_best.copy()
            for _vi, y1 in new_pts.items():
                x_cand[_vi] = y1
            x_cand = _norm(x_cand)
            cand_score = _max_cos(x_cand)
        except Exception:
            continue
        if cand_score < best:
            x_best = x_cand
            best = float(cand_score)
            # opportunistic LP if very good? still gate below 0.0675
            if cand_score < 0.0675:
                dl = min(time.time() + 3.0, T_SEARCH_END, T_FINAL_DEADLINE)
                if dl - time.time() > 0.5:
                    x_p = _lp_polish_deadline(x_best, dl)
                    try:
                        s_p = _max_cos(x_p)
                    except Exception:
                        s_p = best
                    if s_p < best:
                        x_best = x_p; best = float(s_p)
        elif cand_score < 0.0675:
            # gate LP only for promising
            if time.time() < T_SEARCH_END - 3.5:
                dl = min(time.time() + 3.0, T_SEARCH_END)
                x_p = _lp_polish_deadline(x_cand, dl)
                try:
                    s_p = _max_cos(x_p)
                except Exception:
                    s_p = cand_score
                if s_p < best:
                    x_best = x_p; best = float(s_p)
        # loop time check
    # final 15s polish best
    try:
        if time.time() < T_FINAL_DEADLINE:
            x_best = _lp_polish_deadline(x_best, T_FINAL_DEADLINE)
    except Exception:
        pass
    x_best = _norm(x_best)
    return x_best
