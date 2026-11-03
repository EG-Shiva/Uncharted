import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _relax(x, steps, lr, T0, T1):
    n, d = x.shape
    m_ad = np.zeros_like(x)
    v_ad = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p = 1.0
    b2p = 1.0
    if steps > 1:
        logr = np.log(max(T1, 1e-12) / max(T0, 1e-12)) / (steps - 1)
    else:
        logr = 0.0
    for t in range(1, steps + 1):
        T = T0 * np.exp(logr * (t - 1))
        if T < 5e-5:
            T = 5e-5
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        mv = float(G.max())
        E = np.exp((G - mv) / T)
        np.fill_diagonal(E, 0.0)
        S = E.sum() * 0.5
        if S <= 0 or not np.isfinite(S):
            break
        Grad = (E @ x) / S
        dot = (Grad * x).sum(axis=1, keepdims=True)
        Gt = Grad - dot * x
        b1p *= b1
        b2p *= b2
        m_ad = b1 * m_ad + (1.0 - b1) * Gt
        v_ad = b2 * v_ad + (1.0 - b2) * (Gt * Gt)
        mhat = m_ad / (1.0 - b1p)
        vhat = v_ad / (1.0 - b2p)
        step = mhat / (np.sqrt(vhat) + eps)
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        step = step / np.maximum(sn / 1.0, 1.0)
        x = x - lr * step
        x = _norm(x)
    return x

def _refine_single(p0, X, steps=200, lr=0.03, T0=0.02, T1=0.002):
    p = p0 / (np.linalg.norm(p0) + 1e-12)
    m_ad = np.zeros_like(p)
    v_ad = np.zeros_like(p)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    if steps > 1:
        logr = np.log(max(T1, 1e-12) / max(T0, 1e-12)) / (steps - 1)
    else:
        logr = 0.0
    for t in range(1, steps + 1):
        T = T0 * np.exp(logr * (t - 1))
        if T < 8e-4:
            T = 8e-4
        s = X @ p
        mv = float(s.max())
        e = np.exp((s - mv) / T)
        S = float(e.sum())
        if not np.isfinite(S) or S <= 0:
            break
        w = e / S
        grad = w @ X
        gt = grad - np.dot(grad, p) * p
        b1p *= b1
        b2p *= b2
        m_ad = b1 * m_ad + (1.0 - b1) * gt
        v_ad = b2 * v_ad + (1.0 - b2) * (gt * gt)
        mhat = m_ad / (1.0 - b1p)
        vhat = v_ad / (1.0 - b2p)
        step = mhat / (np.sqrt(vhat) + eps)
        sn = float(np.linalg.norm(step)) + 1e-12
        if sn > 1.0:
            step = step / sn
        p = p - lr * step
        nrm = float(np.linalg.norm(p))
        if nrm == 0 or not np.isfinite(nrm):
            p = p0 / (np.linalg.norm(p0) + 1e-12)
        else:
            p = p / nrm
    return p

def _lp_polish(x, seconds, deadline=1e18):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() > deadline:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 2500:
            order = np.argsort(-c[sel])[:2500]
            I, J = I[order], J[order]
            p = 2500
            bvals = -c[sel][order]
        else:
            bvals = -c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=bvals, A_eq=A_eq, b_eq=np.zeros(n),
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
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _bfs_cluster(Gmax_minus, adj, seed, K, Gfull):
    n = adj.shape[0]
    seen = np.zeros(n, dtype=bool)
    seen[seed] = True
    queue = [int(seed)]
    order = [int(seed)]
    qi = 0
    while qi < len(queue) and len(order) < K:
        u = queue[qi]; qi += 1
        nbrs = np.where(adj[u])[0]
        # visit higher-contact neighbours first
        if len(nbrs) > 1:
            deg = adj[nbrs].sum(axis=1) if hasattr(adj[nbrs], 'sum') else None
            # simple sort by Gram value to seed cluster
            vals = Gfull[u, nbrs]
            nbrs = nbrs[np.argsort(-vals)]
        for w in nbrs:
            w = int(w)
            if not seen[w]:
                seen[w] = True
                queue.append(w)
                order.append(w)
                if len(order) >= K:
                    break
    # if graph disconnected and cluster too small, grow by closest point to cluster
    while len(order) < K:
        # max dot of outside points to cluster
        inside = np.array(order, dtype=int)
        mask = np.ones(n, dtype=bool); mask[inside] = False
        outside = np.where(mask)[0]
        if len(outside) == 0:
            break
        # score = max dot to inside
        sub = Gfull[np.ix_(outside, inside)].max(axis=1)
        pick = outside[int(np.argmax(sub))]
        seen[pick] = True
        order.append(int(pick))
        queue.append(int(pick))
    return np.array(order[:K], dtype=int)

def solve():
    t0 = time.time()
    DEAD = t0 + 170.0
    FINAL_RESERVE = 28.0
    HOP_END = DEAD - FINAL_RESERVE - 2.0
    rng = np.random.default_rng(12345)
    n, d = 154, 10
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            if a.shape == (n, d):
                a = _norm(a.astype(float))
                v = _max_cos(a)
                if v < best_val:
                    best_val = v
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.standard_normal((n, d)))
        best_val = _max_cos(best)
    base_best = best.copy()
    base_val = best_val

    trial = 0
    # time-budgeted trials
    while time.time() < HOP_END:
        trial += 1
        trial_start = time.time()
        trial_budget = min(32.0, HOP_END - time.time())
        if trial_budget < 8.0:
            break
        # choose cluster size 12..24 cycling
        Ks = [16, 20, 13, 24, 18, 12, 22, 15]
        K = int(Ks[(trial - 1) % len(Ks)])
        try:
            cur_best = best.copy()
            cur_val = float(best_val)
            G = cur_best @ cur_best.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            # contact graph at 1e-5, widen if empty
            thr = 1e-5
            adj = (G > m - thr)
            np.fill_diagonal(adj, False)
            if adj.sum() < n:
                thr = 5e-5
                adj = (G > m - thr)
                np.fill_diagonal(adj, False)
            if adj.sum() < n:
                thr = 2e-4
                adj = (G > m - thr)
                np.fill_diagonal(adj, False)
            deg = adj.sum(axis=1).astype(float)
            # high-degree seed
            maxdeg = float(deg.max())
            cands = np.where(deg >= maxdeg - 1e-9)[0]
            seed = int(cands[int(rng.integers(len(cands)))])
            Gfull = cur_best @ cur_best.T
            cluster = _bfs_cluster(m - thr, adj, seed, K, Gfull)
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[cluster] = False
            cur = cur_best[keep_mask].copy()
            # relax remaining with Adam until max drops (no LP here)
            # two short stages
            if time.time() - trial_start > trial_budget * 0.35:
                raise TimeoutError("abandon")
            cur = _relax(cur, steps=320, lr=0.015, T0=0.012, T1=0.0012)
            if time.time() - trial_start > trial_budget * 0.5:
                raise TimeoutError("abandon")
            cur = _relax(cur, steps=160, lr=0.010, T0=0.003, T1=0.0006)
            # regrow one at a time
            n_missing = n - cur.shape[0]
            for ins in range(n_missing):
                # check elapsed after every insertion (before next)
                elapsed_trial = time.time() - trial_start
                if elapsed_trial > trial_budget * 0.92:
                    raise TimeoutError("abandon insertion budget")
                if time.time() > HOP_END:
                    raise TimeoutError("hop end")
                # sample ~20000 candidates
                C = 20000
                cand = rng.standard_normal(size=(C, d))
                cand = _norm(cand)
                D = cand @ cur.T
                md = D.max(axis=1)
                Kb = 8
                idx = np.argpartition(md, Kb)[:Kb]
                idx = idx[np.argsort(md[idx])]
                # refine best few by single-point softmax descent
                refined = []
                for kk in range(min(4, len(idx))):
                    p0 = cand[int(idx[kk])]
                    pr = _refine_single(p0, cur, steps=180, lr=0.03, T0=0.02, T1=0.002)
                    # score
                    s = float((cur @ pr).max())
                    refined.append((s, pr))
                # also keep unrefined best as fallback
                s0 = float(md[idx[0]])
                refined.append((s0, cand[int(idx[0])]))
                refined.sort(key=lambda t: t[0])
                winner = refined[0][1]
                cur = np.vstack([cur, winner.reshape(1, d)])
                cur = _norm(cur)
                # short all-point relaxation
                if cur.shape[0] < n:
                    cur = _relax(cur, steps=90, lr=0.010, T0=0.004, T1=0.0007)
                else:
                    cur = _relax(cur, steps=160, lr=0.010, T0=0.0035, T1=0.0006)
            # restored to 154
            cur = _norm(cur)
            val = _max_cos(cur)
            if val < best_val and cur.shape == (n, d):
                best = cur.copy()
                best_val = val
        except TimeoutError:
            continue
        except Exception:
            continue
        # loop continues to try another cluster
    # final single capped active-set LP polish, reserving time
    remain = DEAD - time.time() - 1.0
    if remain > 3.0 and best is not None:
        lp_time = float(min(max(5.0, remain - 0.5), FINAL_RESERVE))
        try:
            best = _lp_polish(best, lp_time, deadline=DEAD - 0.5)
        except Exception:
            pass
    best = _norm(best)
    best[~np.isfinite(best)] = 0.0
    nr = np.linalg.norm(best, axis=1, keepdims=True)
    bad = (nr.squeeze() < 1e-12)
    if np.any(bad):
        rb = rng.standard_normal((int(np.sum(bad)), d))
        # fix shape handling
        if rb.ndim == 1:
            rb = rb.reshape(1, d)
        best[bad] = _norm(rb)
    best = _norm(best)
    return best
