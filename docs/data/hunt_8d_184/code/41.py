import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-300)

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _global_softmax(x, steps, T0, T1, lr0, lr1):
    x = _norm(x.copy())
    for s in range(steps):
        f = s / max(steps - 1, 1)
        T = T0 * (T1 / T0) ** f
        lr = lr0 * (lr1 / lr0) ** f
        G = x @ x.T
        np.fill_diagonal(G, -1.0)
        m = float(np.max(G))
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        rs = E.sum(axis=1, keepdims=True) + 1e-300
        D = (E @ x) / rs
        dot = (D * x).sum(axis=1, keepdims=True)
        Dt = D - dot * x
        x = _norm(x - lr * Dt)
    return x

def _refine_hole(Xf, y, steps=80):
    y = y / (np.linalg.norm(y) + 1e-300)
    la0, la1 = np.log(60.0), np.log(300.0)
    ll0, ll1 = np.log(0.06), np.log(0.005)
    for t in range(steps):
        f = t / max(steps - 1, 1)
        alpha = np.exp(la0 + (la1 - la0) * f)
        lr = np.exp(ll0 + (ll1 - ll0) * f)
        s = Xf @ y
        mx = float(np.max(s))
        w = np.exp(alpha * (s - mx))
        sw = float(np.sum(w)) + 1e-30
        g = w @ Xf
        gt = g - float(np.dot(g, y)) * y
        y = y - lr * gt / sw
        n = np.linalg.norm(y)
        if not np.isfinite(n) or n == 0:
            break
        y = y / n
    return y

def _find_hole(Xf, rng, n_sample=50000):
    d = Xf.shape[1]
    best_y = None
    best_m = 1e9
    batch = 10000
    rem = n_sample
    while rem > 0:
        b = batch if rem >= batch else rem
        rem -= b
        Y = rng.standard_normal(size=(b, d))
        Y /= (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-300)
        S = Y @ Xf.T
        m = S.max(axis=1)
        idx = int(np.argmin(m))
        if float(m[idx]) < best_m:
            best_m = float(m[idx])
            best_y = Y[idx].copy()
    return best_y, best_m

def _local_refine(cur, movable, steps=80):
    # movable: bool mask, only these move; repulsion from all points
    cur = _norm(cur.copy())
    n = cur.shape[0]
    mv = np.where(movable)[0]
    if len(mv) == 0:
        return cur
    for s in range(steps):
        f = s / max(steps - 1, 1)
        T = 0.02 * (0.004 / 0.02) ** f if steps > 1 else 0.02
        lr = 0.03 * (0.005 / 0.03) ** f if steps > 1 else 0.03
        G = cur @ cur.T
        np.fill_diagonal(G, -1.0)
        m = float(np.max(G))
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        # gradient for movable rows only
        D = (E[mv] @ cur) / (E[mv].sum(axis=1, keepdims=True) + 1e-300)
        xm = cur[mv]
        dot = (D * xm).sum(axis=1, keepdims=True)
        Dt = D - dot * xm
        cur[mv] = _norm(xm - lr * Dt)
    return cur

def _lp_polish(x, seconds, deadline):
    x = _norm(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    xb = x.copy()
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    ar = np.arange(d)
    pcap = 1500
    while time.time() - start < seconds and time.time() < deadline:
        if radius < 1e-13:
            break
        if time.time() > deadline - 0.5:
            break
        G = x @ x.T
        cf = G[iu0]
        m = float(cf.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = cf > thr
        I = iu0[0][sel]; J = iu0[1][sel]; cs = cf[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > pcap:
            idx = np.argpartition(-cs, pcap)[:pcap]
            I, J, cs = I[idx], J[idx], cs[idx]
            p = pcap
        nv = n * d + 1
        rows = np.concatenate([np.repeat(np.arange(p), d), np.repeat(np.arange(p), d), np.arange(p)])
        cols = np.concatenate([np.repeat(I * d, d) + np.tile(ar, p),
                               np.repeat(J * d, d) + np.tile(ar, p),
                               np.full(p, n * d)])
        vals = np.concatenate([x[J].ravel(), x[I].ravel(), -np.ones(p)])
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = _norm(x + res.x[:-1].reshape(n, d))
        v = _max_cos(y)
        if v < best:
            best = v; xb = y.copy(); x = y
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return xb, best

def solve():
    t0 = time.time()
    DEADLINE = t0 + 168.0
    rng = np.random.default_rng(12345)
    n, d = 184, 8
    cands = []
    for f in ("best.npy", "record.npy"):
        try:
            c = np.asarray(np.load(f), dtype=np.float64)
            if c.shape == (n, d):
                c = _norm(c)
                cands.append((_max_cos(c), c))
        except Exception:
            pass
    if cands:
        cands.sort(key=lambda t: t[0])
        gval, x0 = cands[0][0], cands[0][1].copy()
    else:
        x0 = _norm(rng.standard_normal(size=(n, d)))
        gval = _max_cos(x0)
    gbest = x0.copy()

    # contact graph
    G0 = x0 @ x0.T
    np.fill_diagonal(G0, -2.0)
    m0 = float(np.max(G0))
    tol = 1e-7
    mask = G0 > m0 - tol
    deg = mask.sum(axis=1).astype(int) // 1
    # mask is symmetric, sum counts each neighbor once (diag is False)
    deg = np.asarray(deg)
    iu = np.triu_indices(n, 1)
    cf = (x0 @ x0.T)[iu]
    # contact pairs
    cp_mask = cf > m0 - tol
    CI = iu[0][cp_mask]; CJ = iu[1][cp_mask]
    # adjacency for greedy
    Adj = np.zeros((n, n), dtype=bool)
    Adj[CI, CJ] = True; Adj[CJ, CI] = True

    rattler = int(np.argmin(deg))
    lows = np.where(deg <= 4)[0].tolist()
    highs = np.where(deg >= 22)[0].tolist()
    if len(highs) < 10:
        # fallback: top by degree
        order = np.argsort(-deg)
        highs = order[:20].tolist()
        if rattler in highs:
            highs.remove(rattler)

    # sort highs by degree desc
    highs = sorted(highs, key=lambda i: -deg[i])

    results = []  # (val, x)
    results.append((gval, gbest.copy()))

    for k in [1, 2, 3, 4, 6, 8]:
        if time.time() > DEADLINE - 45:
            break
        # choose removal set: rattler + (k-1) highs greedy max new contacts
        selected = [rattler]
        selset = set(selected)
        # candidate pool: highs not rattler
        pool = [h for h in highs if h not in selset]
        # if pool too small, add next highest degree points
        if len(pool) < k:
            order = np.argsort(-deg)
            for o in order:
                o = int(o)
                if o not in selset and o not in pool:
                    pool.append(o)
                if len(pool) >= k + 5:
                    break
        while len(selected) < k:
            best_gain = -1
            best_p = None
            for p in pool:
                if p in selset:
                    continue
                # gain = contacts eliminated newly: neighbors not already covered
                # covered pairs = pairs with >=1 endpoint in selset
                # incremental = degree[p] - edges to selset (those already covered) ... actually edges to selset are already covered, so gain = deg[p] - count(neighbors in selset)
                cnt = 0
                for s in selected:
                    if Adj[p, s]:
                        cnt += 1
                gain = int(deg[p]) - cnt
                # bonus for clustering? prefer connected to selected to open a hole cluster? use slight bonus if connected
                # but spec says maximize eliminated per point, so use gain
                if gain > best_gain:
                    best_gain = gain
                    best_p = p
            if best_p is None:
                break
            selected.append(best_p)
            selset.add(best_p)
        selected = sorted(selected)
        keep = np.array([i for i in range(n) if i not in selset], dtype=int)
        cur = x0[keep].copy()
        n0 = len(cur)
        # sequential reinsertion
        for r in range(k):
            if time.time() > DEADLINE - 40:
                break
            hole_y, _ = _find_hole(cur, rng, n_sample=50000)
            y = _refine_hole(cur, hole_y, steps=80)
            cur = np.vstack([cur, y.reshape(1, -1)])
            cur = _norm(cur)
            # movable: reinserted + neighbors with cos > 0.35
            ncur = cur.shape[0]
            n_re = ncur - n0  # number reinserted so far
            re_idx = np.arange(n0, ncur)
            # cosine of fixed to reinserted
            S = cur[:n0] @ cur[re_idx].T
            neigh = np.where(S.max(axis=1) > 0.35)[0]
            movable = np.zeros(ncur, dtype=bool)
            movable[re_idx] = True
            movable[neigh] = True
            cur = _local_refine(cur, movable, steps=60)
            # short global anneal
            cur = _global_softmax(cur, steps=80, T0=0.02, T1=0.004, lr0=0.02, lr1=0.004)
        # final short global anneal for this candidate
        cur = _global_softmax(cur, steps=120, T0=0.015, T1=0.002, lr0=0.015, lr1=0.003)
        v = _max_cos(cur)
        results.append((v, cur.copy()))
        if v < gval:
            gval = v
            gbest = cur.copy()

    # reserve LP for best two surgical candidates (excluding initial if it is best? include anyway top-2 distinct)
    results.sort(key=lambda t: t[0])
    # take top 2 that came from surgery (or overall) for polishing, but don't polish more than 2
    polished = []
    lp_budget_each = 12.0
    # ensure time left
    for i in range(min(2, len(results))):
        if time.time() > DEADLINE - 15:
            break
        v, x = results[i]
        # skip polishing the incumbent if already polished? just polish
        xp, vp = _lp_polish(x, seconds=lp_budget_each, deadline=DEADLINE - 2)
        polished.append((vp, xp))
        if vp < gval:
            gval = vp
            gbest = xp.copy()
    if polished:
        polished.sort(key=lambda t: t[0])
        gbest = polished[0][1].copy()
        gval = polished[0][0]
        # compare with unpolished results
        for v, x in results:
            if v < gval:
                gval = v; gbest = x.copy()
    else:
        # no time / no polish, keep best
        pass

    gbest = _norm(np.asarray(gbest, dtype=np.float64))
    # strict feasibility: shrink not needed but ensure finite
    gbest = np.where(np.isfinite(gbest), gbest, 0.0)
    # renormalize, avoid zero rows
    nr = np.linalg.norm(gbest, axis=1, keepdims=True)
    nr[~np.isfinite(nr)] = 1.0
    nr[nr == 0] = 1.0
    gbest = gbest / nr
    return gbest
