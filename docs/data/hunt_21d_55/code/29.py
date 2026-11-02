import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _lp_polish_deadline(x, deadline, radius_init=8e-4):
    # short trust-region LP squeeze, respects deadline
    x = _norm_rows(x)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = radius_init
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-13:
        try:
            G = x @ x.T
        except Exception:
            break
        cfull = G[iu]
        m = float(cfull.max())
        sel = cfull > m - max(20.0 * radius, 1e-9)
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 1200:
            idx = np.where(sel)[0]
            order = np.argsort(-cfull[idx])[:1200]
            nsel = np.zeros_like(sel)
            nsel[idx[order]] = True
            sel = nsel
        csel = cfull[sel]
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm_rows(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
        if time.time() >= deadline:
            break
    return x

def solve():
    t0 = time.time()
    BUDGET = 150.0
    T_SEARCH_END = t0 + 138.0
    rng = np.random.default_rng(0)
    n, d = 55, 21
    # load incumbent
    best = None
    best_score = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (55, 21):
                a = _norm_rows(a)
                s = _max_cos(a)
                if s < best_score:
                    best_score = s
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _norm_rows(rng.standard_normal((n, d)))
        best_score = _max_cos(best)
    cur = best.copy()
    cur_score = best_score

    N_CAND = 200000
    it = 0
    T0, T1 = 0.0025, 0.00008

    while time.time() < T_SEARCH_END:
        it += 1
        if time.time() + 4.0 > T_SEARCH_END and it % 2 == 0:
            pass
        elapsed = time.time() - t0
        frac = elapsed / BUDGET
        Temp = T0 * (1.0 - frac) + T1 * frac
        if Temp <= 1e-12:
            Temp = 1e-12
        # occasionally restart walk from best to avoid drift
        if it % 40 == 0 and cur_score > best_score + 0.004:
            cur = best.copy()
            cur_score = best_score

        k = int(rng.integers(4, 9))  # 4..8
        # contact graph of cur
        try:
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            mcur = float(G.max())
        except Exception:
            mcur = cur_score
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
        delta = 1e-3
        adj = G > (mcur - delta)
        # degree
        deg = np.sum(adj, axis=1)
        # seed: high-degree random among top
        order_deg = np.argsort(-deg)
        top_pool = order_deg[:12]
        seed = int(top_pool[rng.integers(len(top_pool))])
        cluster = [seed]
        in_cl = np.zeros(n, dtype=bool)
        in_cl[seed] = True
        # grow to size k keeping connectivity
        while len(cluster) < k:
            if time.time() >= T_SEARCH_END:
                break
            # frontier: neighbors of cluster
            front = []
            for c in cluster:
                nbrs = np.where(adj[c])[0]
                for nb in nbrs:
                    if not in_cl[int(nb)]:
                        front.append(int(nb))
            # dedup
            if len(front) > 0:
                front = list(dict.fromkeys(front))
                pick = front[int(rng.integers(len(front)))]
            else:
                # no contact frontier: pick closest (max dot) to cluster
                rest = np.where(~in_cl)[0]
                if len(rest) == 0:
                    break
                # sum dots to cluster
                dots = np.sum(G[np.ix_(rest, np.array(cluster, dtype=int))], axis=1) if len(cluster) > 1 else G[rest, cluster[0]]
                pick = int(rest[int(np.argmax(dots))])
            cluster.append(pick)
            in_cl[pick] = True
        cluster = np.array(cluster, dtype=int)
        frozen_mask = np.ones(n, dtype=bool)
        frozen_mask[cluster] = False
        F = cur[frozen_mask]  # (n-k, d)

        # ---- large sampling: deep holes ----
        # generate candidates in one go (200k) but compute scores chunked
        try:
            cand = rng.standard_normal((N_CAND, d))
        except Exception:
            break
        cand = _norm_rows(cand)
        # chunked max-dot to frozen
        s_all = np.empty(N_CAND, dtype=np.float64)
        CH = 25000
        ok = True
        for a in range(0, N_CAND, CH):
            if time.time() >= T_SEARCH_END:
                ok = False
                break
            b = min(a + CH, N_CAND)
            Cb = cand[a:b]
            try:
                D = Cb @ F.T
            except Exception:
                ok = False
                break
            s_all[a:b] = D.max(axis=1)
            # time check inside loop
            if time.time() >= T_SEARCH_END:
                ok = False
                break
        if not ok:
            # give up this iteration
            continue
        order = np.argsort(s_all)  # deepest first
        # greedy diverse pick of k holes
        sel = []
        thresh = 0.25
        scanned = 0
        oi = 0
        # to avoid O(N*k) scan, limit scan length
        while len(sel) < k and oi < N_CAND:
            if time.time() >= T_SEARCH_END:
                break
            idx = int(order[oi]); oi += 1
            scanned += 1
            c = cand[idx]
            if len(sel) == 0:
                sel.append(c)
            else:
                S = np.stack(sel)
                md = float(np.max(S @ c))
                if md < thresh:
                    sel.append(c)
            # relax thresh if scanning too far
            if oi % 8000 == 0 and len(sel) < k:
                thresh += 0.1
                if thresh > 0.9:
                    # just take next deepest
                    while len(sel) < k and oi < N_CAND:
                        sel.append(cand[int(order[oi])]); oi += 1
                    break
            if oi >= N_CAND:
                break
        if len(sel) < k:
            # fallback pad with deepest
            for oi2 in range(oi, N_CAND):
                if len(sel) >= k:
                    break
                sel.append(cand[int(order[oi2])])
        if len(sel) < k:
            continue
        Y = np.stack(sel[:k])
        # ---- local optimisation of each reinserted point (soft-min), cycling ----
        n_cycles = 4
        inner_steps = 110
        beta = 220.0
        lr = 0.012
        broken = False
        for cyc in range(n_cycles):
            for j in range(k):
                if time.time() >= T_SEARCH_END:
                    broken = True
                    break
                # others = frozen + other Y
                mask_j = np.ones(k, dtype=bool); mask_j[j] = False
                O = np.vstack([F, Y[mask_j]])  # (n-1, d)
                y = Y[j].copy()
                for tstep in range(inner_steps):
                    dts = O @ y
                    mm = float(np.max(dts))
                    # soft-max weights
                    w = np.exp(beta * (dts - mm))
                    sw = float(np.sum(w))
                    if not np.isfinite(sw) or sw <= 0:
                        break
                    w /= sw
                    grad = O.T @ w  # (d,)
                    gd = float(np.dot(grad, y))
                    gt = grad - gd * y
                    nn = float(np.linalg.norm(gt))
                    if nn < 1e-12:
                        break
                    gt /= nn
                    y = y - lr * gt
                    nn2 = float(np.linalg.norm(y))
                    if nn2 < 1e-12:
                        break
                    y /= nn2
                Y[j] = y
            if broken:
                break
        # build proposal
        Xnew = cur.copy()
        Xnew[cluster] = Y
        Xnew = _norm_rows(Xnew)
        try:
            ns = _max_cos(Xnew)
        except Exception:
            continue
        # SA acceptance, walk from accepted
        accept = False
        if ns < cur_score:
            accept = True
        else:
            try:
                p = float(np.exp((cur_score - ns) / Temp))
            except Exception:
                p = 0.0
            if rng.random() < p:
                accept = True
        if accept:
            cur = Xnew
            cur_score = ns
            if ns < best_score:
                best = Xnew.copy()
                best_score = ns
            # short LP squeeze after accepted repair (0.5-1s), only if promising
            if ns < best_score + 0.006:
                dl = min(0.6, max(0.3, T_SEARCH_END - (time.time())))
                if dl > 0.25 and time.time() + dl < t0 + BUDGET - 4.0:
                    deadline = time.time() + dl
                    # use 0.5-0.6s squeeze
                    Xpol = _lp_polish_deadline(cur, deadline)
                    try:
                        ps = _max_cos(Xpol)
                    except Exception:
                        ps = cur_score
                    if ps < cur_score:
                        cur = Xpol
                        cur_score = ps
                    if ps < best_score:
                        best = Xpol.copy()
                        best_score = ps
        else:
            # even if rejected, track best (should not happen since ns>=cur> =best? but keep)
            if ns < best_score:
                best = Xnew.copy()
                best_score = ns
                cur = Xnew.copy()
                cur_score = ns
        # time check
        if time.time() >= T_SEARCH_END:
            break

    # final short polish of best with remaining budget (cap ~10s)
    remain = (t0 + BUDGET) - time.time()
    if remain > 1.0:
        deadline = time.time() + min(remain - 0.5, 10.0)
        try:
            best = _lp_polish_deadline(best, deadline)
        except Exception:
            pass
    best = _norm_rows(np.asarray(best, dtype=np.float64))
    return best
