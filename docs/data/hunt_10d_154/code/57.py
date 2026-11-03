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

def _relax_global(X, steps, lr, T0, T1, deadline=1e18):
    X = _norm(X.copy())
    n, d = X.shape
    m_ad = np.zeros_like(X)
    v_ad = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p = 1.0
    b2p = 1.0
    if steps > 1:
        logr = np.log(max(T1, 1e-12) / max(T0, 1e-12)) / (steps - 1)
    else:
        logr = 0.0
    for t in range(1, steps + 1):
        if t % 50 == 0 and time.time() > deadline:
            break
        T = T0 * np.exp(logr * (t - 1))
        if T < 1e-4:
            T = 1e-4
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        mv = float(G.max())
        E = np.exp((G - mv) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if not np.isfinite(S) or S <= 0:
            break
        Grad = (E @ X) / S
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        b1p *= b1
        b2p *= b2
        m_ad = b1 * m_ad + (1.0 - b1) * Gt
        v_ad = b2 * v_ad + (1.0 - b2) * (Gt * Gt)
        mhat = m_ad / (1.0 - b1p)
        vhat = v_ad / (1.0 - b2p)
        step = mhat / (np.sqrt(vhat) + eps)
        X = X - lr * step
        X = _norm(X)
    return X

def _refine_single(p0, X, steps=100, lr=0.03, T0=0.02, T1=0.0025):
    p = p0.copy()
    nrm = float(np.linalg.norm(p))
    if nrm == 0 or not np.isfinite(nrm):
        p = X[0] * -1.0
        nrm = float(np.linalg.norm(p))
    p = p / nrm
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
        gt = grad - float(np.dot(grad, p)) * p
        b1p *= b1
        b2p *= b2
        m_ad = b1 * m_ad + (1.0 - b1) * gt
        v_ad = b2 * v_ad + (1.0 - b2) * (gt * gt)
        mhat = m_ad / (1.0 - b1p)
        vhat = v_ad / (1.0 - b2p)
        step = mhat / (np.sqrt(vhat) + eps)
        sn = float(np.linalg.norm(step))
        if sn > 1.0:
            step = step / sn
        p = p - lr * step
        nrm = float(np.linalg.norm(p))
        if nrm == 0 or not np.isfinite(nrm):
            break
        p = p / nrm
    return p

def _pick_victims(base, k, rng):
    n, d = base.shape
    G = base @ base.T
    np.fill_diagonal(G, -2.0)
    M = float(G.max())
    # tight graph 1e-5 slack
    adj = (G > M - 1e-5)
    np.fill_diagonal(adj, False)
    # overload counts with wide band
    cnt = (G > M - 0.005).sum(axis=1)
    order_desc = np.argsort(-cnt, kind="stable")
    # seed: among top overloaded randomly
    top_pool = order_desc[:max(10, n // 8)].tolist()
    seed = int(top_pool[rng.integers(len(top_pool))])
    # BFS cluster target
    n_over = int(rng.integers(4, 9))  # overloaded separate
    if n_over > k - 4:
        n_over = max(2, k - 8)
    k_clu = k - n_over
    # BFS
    seen = set([seed])
    queue = [seed]
    qi = 0
    while len(seen) < k_clu and qi < len(queue):
        u = queue[qi]
        qi += 1
        nbrs = np.where(adj[u])[0].tolist()
        rng.shuffle(nbrs)
        for w in nbrs:
            if w not in seen:
                seen.add(int(w))
                queue.append(int(w))
                if len(seen) >= k_clu:
                    break
    # if cluster too small, expand with nearest by cosine
    if len(seen) < k_clu:
        # sort others by max dot to cluster
        others = [i for i in range(n) if i not in seen]
        # score = max G[i, seen]
        clu = list(seen)
        scores = G[np.ix_(others, clu)].max(axis=1) if len(clu) > 0 else cnt[others]
        o = np.argsort(-scores, kind="stable")
        for idx in o:
            seen.add(int(others[idx]))
            if len(seen) >= k_clu:
                break
    cluster = list(seen)
    # overloaded extras not in cluster
    extra = []
    for ii in order_desc:
        ii = int(ii)
        if ii not in seen:
            extra.append(ii)
        if len(extra) >= n_over:
            break
    victims = np.array(cluster + extra, dtype=int)
    # ensure exact k (truncate/pad)
    if len(victims) > k:
        victims = victims[:k]
    elif len(victims) < k:
        remain = [i for i in range(n) if i not in set(victims.tolist())]
        rng.shuffle(remain)
        victims = np.array(victims.tolist() + remain[:k - len(victims)], dtype=int)
    return victims

def _greedy_insert(cur, rng, C=30000):
    d = cur.shape[1]
    cand = _norm(rng.normal(size=(C, d)))
    # score in one matmul; cur size ~124-153
    D = cand @ cur.T
    md = D.max(axis=1)
    # diversity: mostly best, sometimes top-3
    if rng.random() < 0.75:
        j = int(np.argmin(md))
    else:
        K = 5
        idx = np.argpartition(md, K)[:K]
        # pick randomly among K weighted to smaller
        order = idx[np.argsort(md[idx])]
        j = int(order[int(rng.integers(min(3, K)))])
    return cand[j]

def _lp_polish(x, deadline):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline - 0.8 and radius > 1e-14:
        if time.time() + 1.0 > deadline:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
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
            x, best, radius = y, v, min(radius * 1.6, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    T_LIM = 172.0
    t_end = t0 + T_LIM
    hop_end = t_end - 20.0
    rng = np.random.default_rng(20261103)
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
        best = _norm(rng.normal(size=(n, d)))
        best_val = _max_cos(best)
    # surgery cycles, no intermediate LP
    cycle = 0
    while True:
        if time.time() > hop_end - 3.0:
            break
        cycle += 1
        k = int(rng.integers(18, 31))  # 18..30
        base = best.copy()
        try:
            victims = _pick_victims(base, k, rng)
        except Exception:
            continue
        keep = np.ones(n, dtype=bool)
        keep[victims] = False
        rem = base[keep].copy()
        # fixed small number of softmax-Adam steps on remainder
        try:
            rem = _relax_global(rem, steps=320, lr=0.020, T0=0.010, T1=0.0010, deadline=hop_end)
        except Exception:
            continue
        if time.time() > hop_end - 2.0:
            continue
        cur = rem.copy()
        aborted = False
        # reinsert one at a time
        for ii in range(k):
            if time.time() > hop_end - 1.0:
                aborted = True
                break
            try:
                p0 = _greedy_insert(cur, rng, C=30000)
            except Exception:
                aborted = True
                break
            try:
                p1 = _refine_single(p0, cur, steps=100, lr=0.03, T0=0.02, T1=0.0025)
            except Exception:
                p1 = p0
            cur = np.vstack([cur, p1[None, :]])
            cur = _norm(cur)
            if (ii + 1) % 5 == 0:
                try:
                    cur = _relax_global(cur, steps=120, lr=0.015, T0=0.008, T1=0.0008, deadline=hop_end)
                except Exception:
                    pass
            if time.time() > hop_end - 0.8:
                # if not finished all insertions, abort cycle
                if cur.shape[0] != n:
                    aborted = True
                break
        if aborted or cur.shape[0] != n:
            continue
        try:
            cur = _relax_global(cur, steps=120, lr=0.012, T0=0.006, T1=0.0007, deadline=hop_end)
        except Exception:
            pass
        try:
            v = _max_cos(cur)
        except Exception:
            continue
        if v < best_val and np.all(np.isfinite(cur)):
            best = _norm(cur.copy())
            best_val = float(v)
        # time check already each 50 iters inside relax; also here
        if time.time() > hop_end - 1.0:
            break
    # final 20s: LP polish only
    try:
        best = _lp_polish(best, deadline=t_end - 0.4)
    except Exception:
        pass
    best = _norm(best)
    return best
