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

def _lp_squeeze(x, seconds, deadline):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < deadline and radius > 1e-14:
        cfull = (x @ x.T)[iu]
        m = cfull.max()
        sel = cfull > m - max(20 * radius, 1e-9)
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 900:
            idx = np.where(sel)[0]
            order = np.argsort(-cfull[idx])[:900]
            nsel = np.zeros_like(sel)
            nsel[idx[order]] = True
            sel = nsel
        csel = cfull[sel]
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
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
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x, best

def solve():
    t0 = time.time()
    DEADLINE = t0 + 150.0
    HARD = t0 + 165.0
    rng = np.random.default_rng(20261101)
    n, d = 55, 21
    xb = None
    bs = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (n, d):
                a = _norm(a)
                s = _max_cos(a)
                if s < bs:
                    bs = s
                    xb = a.copy()
        except Exception:
            pass
    if xb is None:
        xb = _norm(rng.standard_normal((n, d)))
        bs = _max_cos(xb)
    cur = xb.copy()
    cur_score = bs
    best = xb.copy()
    best_score = bs
    # SA schedule
    T_init, T_end = 0.0015, 0.00002
    it = 0
    # precompute iu not needed
    while time.time() < DEADLINE - 3.0:
        it += 1
        if time.time() > HARD - 8.0:
            break
        frac = (time.time() - t0) / 150.0
        frac = min(max(frac, 0.0), 1.0)
        Temp = T_init * (T_end / T_init) ** frac
        # --- choose connected cluster 4-8 ---
        G = cur @ cur.T
        np.fill_diagonal(G, -2.0)
        mx = float(np.max(G))
        thr = mx - 5e-4
        adj = G > thr
        deg = np.sum(adj, axis=1)
        k = int(rng.integers(4, 9))
        # seed biased to high degree
        order = np.argsort(-deg)
        top = order[:12]
        seed = int(top[rng.integers(len(top))]) if rng.random() < 0.85 else int(rng.integers(n))
        cluster = [seed]
        in_cl = np.zeros(n, dtype=bool); in_cl[seed] = True
        # grow connected cluster
        while len(cluster) < k:
            # frontier: neighbours of cluster not in cluster
            cand_mask = np.zeros(n, dtype=bool)
            for c in cluster:
                cand_mask |= adj[c]
            cand_mask[in_cl] = False
            if not np.any(cand_mask):
                # disconnected: add highest-degree outside
                rem = np.where(~in_cl)[0]
                j = rem[int(np.argmax(deg[rem]))]
                cluster.append(int(j)); in_cl[int(j)] = True
                continue
            cands = np.where(cand_mask)[0]
            # prefer high degree, add randomness
            if rng.random() < 0.7:
                j = cands[int(np.argmax(deg[cands]))]
            else:
                j = cands[int(rng.integers(len(cands)))]
            cluster.append(int(j)); in_cl[int(j)] = True
        cluster = np.array(cluster, dtype=int)
        keep_mask = np.ones(n, dtype=bool); keep_mask[cluster] = False
        Yf = cur[keep_mask]  # (m,d)
        m = Yf.shape[0]
        # --- sample >=200000 random sphere candidates for deep holes ---
        N = 200000
        B = 50000
        all_cand = np.empty((N, d), dtype=np.float64)
        all_md = np.empty(N, dtype=np.float64)
        pos = 0
        ok = True
        for b in range(N // B):
            if time.time() > HARD - 6.0:
                ok = False; break
            cb = rng.standard_normal((B, d))
            cb = _norm(cb)
            dots = cb @ Yf.T
            md = np.max(dots, axis=1)
            all_cand[pos:pos+B] = cb
            all_md[pos:pos+B] = md
            pos += B
        if not ok:
            break
        # greedy diverse selection of k holes
        picked = []
        # first: deepest hole
        i0 = int(np.argmin(all_md))
        picked.append(i0)
        # precompute dots between candidates and picked for diversity?
        # iterative greedy: score = max(md, max_dot_to_picked)
        # to avoid O(N*k) huge memory, loop
        for _ in range(1, k):
            if time.time() > HARD - 6.0:
                break
            best_s = 1e9
            best_i = -1
            # subsample to top pool for speed: consider 20000 best by md
            # recompute quickly each round on pool
            # use pool of 20000 smallest md not yet picked
            # simple: scan in chunks
            # maintain max dot to picked set
            P = all_cand[picked]  # (q,d)
            # chunk scan
            CH = 20000
            for s in range(0, N, CH):
                e = min(s+CH, N)
                Cd = all_cand[s:e] @ P.T
                mdq = np.max(Cd, axis=1) if Cd.size else np.zeros(e-s)
                score = np.maximum(all_md[s:e], mdq)
                # penalize already picked
                # find min in chunk
                j = int(np.argmin(score))
                if score[j] < best_s:
                    best_s = float(score[j])
                    best_i = s + j
            picked.append(int(best_i))
        V = all_cand[np.array(picked)].copy()  # (k,d)
        # --- locally optimise each reinserted point: soft-min, cycle several times ---
        beta = 120.0
        lr = 0.06
        n_sweep = 3
        n_inner = 60
        for sweep in range(n_sweep):
            if time.time() > HARD - 5.0:
                break
            for qi in range(k):
                if time.time() > HARD - 5.0:
                    break
                v = V[qi].copy()
                # others = frozen + other reinserted
                for step in range(n_inner):
                    # build others
                    mask_o = np.ones(k, dtype=bool); mask_o[qi] = False
                    O = np.vstack([Yf, V[mask_o]]) if k > 1 else Yf
                    dots = O @ v  # (m+k-1,)
                    dm = float(np.max(dots))
                    w = np.exp(beta * (dots - dm))
                    sw = float(np.sum(w))
                    if not np.finite(sw) or sw <= 0:
                        break
                    gw = (w[:, None] * O).sum(axis=0) / sw
                    gt = gw - float(np.dot(gw, v)) * v
                    gn = float(np.linalg.norm(gt))
                    if gn < 1e-12:
                        break
                    gt /= gn
                    # decay lr slightly
                    lrs = lr * (1.0 - 0.5 * (step / n_inner)) * (1.0 - 0.3 * (sweep / n_sweep))
                    v = v - lrs * gt
                    vn = float(np.linalg.norm(v))
                    if vn < 1e-12:
                        break
                    v = v / vn
                V[qi] = v
        Xnew = cur.copy()
        Xnew[cluster] = V
        Xnew = _norm(Xnew)
        ns = _max_cos(Xnew)
        # SA acceptance on true max, walk from accepted
        accept = False
        if ns < cur_score:
            accept = True
        else:
            if Temp > 0 and rng.random() < np.exp(-(ns - cur_score) / max(Temp, 1e-12)):
                accept = True
        if accept:
            cur = Xnew
            cur_score = ns
            if ns < best_score:
                best = Xnew.copy()
                best_score = ns
            # short LP squeeze 0.5-1s after repair
            if time.time() < DEADLINE - 4.0:
                sq = 0.6 if (best_score < 0.0667) else 0.9
                # cap squeeze if little time left
                rem = DEADLINE - time.time() - 2.0
                sq = min(sq, max(rem * 0.15, 0.3))
                if sq > 0.15:
                    try:
                        Xsq, vsq = _lp_squeeze(cur, sq, DEADLINE - 1.0)
                        if vsq < cur_score:
                            cur = Xsq; cur_score = vsq
                        if vsq < best_score:
                            best = Xsq.copy(); best_score = vsq
                    except Exception:
                        pass
        # occasional restart to best if wandering too high
        if it % 40 == 0 and cur_score > best_score + 0.004:
            cur = best.copy(); cur_score = best_score
    return _norm(best)
