import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-300
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _global_anneal(X, steps, T0, T1, lr0, lr1, rng=None, mov_mask=None):
    n, d = X.shape
    vel = np.zeros_like(X)
    mu = 0.85
    for s in range(steps):
        f = s / max(steps - 1, 1)
        T = T0 * (T1 / T0) ** f
        lr = lr0 * (lr1 / lr0) ** f
        G = X @ X.T
        np.fill_diagonal(G, -1.0)
        m = float(np.max(G))
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        rs = E.sum(axis=1, keepdims=True) + 1e-300
        D = (E @ X) / rs
        dot = (D * X).sum(axis=1, keepdims=True)
        Gt = D - dot * X
        # normalize scale
        sc = np.sqrt((Gt*Gt).sum(axis=1, keepdims=True)) + 1e-12
        # cap huge grads
        Gn = Gt / np.maximum(sc, 1.0)
        if mov_mask is not None:
            Gn = Gn * mov_mask[:, None]
            vel = vel * mov_mask[:, None]
        vel = mu * vel + (1.0 - mu) * Gn
        # bias correction-ish
        X = X - lr * vel
        X = _norm(X)
    return X

def _hole_sample(Xf, rng, ns=50000):
    d = Xf.shape[1]
    best_y = None
    best_m = 1e99
    batch = 10000
    nb = ns // batch
    for _ in range(nb):
        Y = rng.standard_normal(size=(batch, d))
        Y = _norm(Y)
        S = Y @ Xf.T
        m = S.max(axis=1)
        i = int(np.argmin(m))
        if float(m[i]) < best_m:
            best_m = float(m[i])
            best_y = Y[i].copy()
    return best_y, best_m

def _single_refine(Xf, y, steps=160):
    la0 = np.log(60.0); la1 = np.log(320.0)
    ll0 = np.log(0.06); ll1 = np.log(0.005)
    for t in range(steps):
        f = t / max(steps - 1, 1)
        alpha = np.exp(la0 + (la1 - la0) * f)
        lr = np.exp(ll0 + (ll1 - ll0) * f)
        s = Xf @ y
        mx = float(np.max(s))
        w = np.exp(alpha * (s - mx))
        sw = float(np.sum(w)) + 1e-300
        g = w @ Xf
        gt = g - float(np.dot(g, y)) * y
        y = y - lr * gt / sw
        nrm = np.linalg.norm(y)
        if not np.isfinite(nrm) or nrm == 0:
            break
        y = y / nrm
    return y

def _lp_polish(X, deadline, max_iters=25):
    X = _norm(X.copy())
    n, d = X.shape
    best = _max_cos(X)
    Xb = X.copy()
    radius = 1e-3
    iu0 = np.triu_indices(n, 1)
    ar = np.arange(d)
    it = 0
    while it < max_iters:
        it += 1
        if time.time() > deadline - 1.5:
            break
        if radius < 1e-13:
            break
        G = X @ X.T
        cf = G[iu0]
        m = float(cf.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = cf > thr
        I = iu0[0][sel]; J = iu0[1][sel]; cs = cf[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > 1500:
            idx = np.argpartition(-cs, 1500)[:1500]
            I, J, cs = I[idx], J[idx], cs[idx]
            p = 1500
        nv = n * d + 1
        rows = np.concatenate([np.repeat(np.arange(p), d), np.repeat(np.arange(p), d), np.arange(p)])
        cols = np.concatenate([np.repeat(I * d, d) + np.tile(ar, p),
                               np.repeat(J * d, d) + np.tile(ar, p),
                               np.full(p, n * d)])
        vals = np.concatenate([X[J].ravel(), X[I].ravel(), -np.ones(p)])
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            tl = max(1.0, deadline - time.time() - 0.6)
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": min(4.0, tl)})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            y = _norm(X + res.x[:-1].reshape(n, d))
        except Exception:
            radius *= 0.5
            continue
        v = _max_cos(y)
        if v < best:
            best = v; Xb = y.copy(); X = y
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return Xb, best

def solve():
    t0 = time.time()
    HARD = t0 + 168.0
    HOP_END = t0 + 122.0
    rng = np.random.default_rng(7)
    n, d = 184, 8
    try:
        x0 = np.asarray(np.load("best.npy"), dtype=np.float64)
    except Exception:
        x0 = np.asarray(np.load("record.npy"), dtype=np.float64)
    if x0.shape != (n, d):
        x0 = x0.reshape(n, d)
    x0 = _norm(x0)
    gval = _max_cos(x0)
    gbest = x0.copy()

    # contact graph with tight threshold 1e-9
    G0 = x0 @ x0.T
    np.fill_diagonal(G0, -2.0)
    m0 = float(np.max(G0))
    eps = 1e-9
    CM = (G0 > m0 - eps)
    deg = CM.sum(axis=1).astype(int)
    permax = np.max(G0, axis=1)
    # rattler: degree 0 else argmin permax
    zero = np.where(deg == 0)[0]
    if len(zero) > 0:
        rattler = int(zero[np.argmin(permax[zero])])
    else:
        rattler = int(np.argmin(permax))
    order_hi = np.argsort(-deg, kind="stable")
    # high pool: top 40 by degree
    high_pool = [int(i) for i in order_hi[:40] if int(i) != rattler]
    # contact adjacency for greedy gain
    # precompute neighbor sets
    neigh = [set(np.where(CM[i])[0].tolist()) for i in range(n)]

    cand_ks = [1, 2, 3, 4, 6, 8]
    # store removal sets
    removals = []
    for k in cand_ks:
        rem = [rattler]
        # greedy add
        cur_set = set(rem)
        # eliminated edges count helper: we greedily maximize incremental new edges
        # precompute eliminated set implicitly via cur_set
        while len(rem) < k:
            best_c = -1
            best_gain = -1
            for c in high_pool:
                if c in cur_set:
                    continue
                # gain = neighbors of c outside cur_set (edges newly covered) + edges to cur_set? count all incident not yet covered
                # an edge (c,u) is newly covered if u not in cur_set and (c,u) is contact; plus edges among? simplified:
                cnt = 0
                for u in neigh[c]:
                    if u not in cur_set:
                        cnt += 1
                # tie-break by degree
                score = cnt * 1000 + int(deg[c])
                if score > best_gain:
                    best_gain = score
                    best_c = c
            if best_c == -1:
                # fallback: pick highest degree not in set
                for c in order_hi:
                    c = int(c)
                    if c not in cur_set:
                        best_c = c
                        break
            rem.append(best_c)
            cur_set.add(best_c)
        removals.append((k, rem))

    candidates = []  # list of (val, X)
    candidates.append((gval, gbest.copy()))

    for (k, rem) in removals:
        if time.time() > HOP_END:
            break
        rem_set = set(rem)
        keep_idx = [i for i in range(n) if i not in rem_set]
        Xcur = x0[keep_idx].copy()  # (n-k, d)
        # reinsert one at a time
        inserted = []  # list of vectors appended
        ok = True
        for j in range(k):
            if time.time() > HOP_END - 2:
                ok = False
                break
            Xfixed = np.vstack([Xcur] + inserted) if len(inserted) > 0 else Xcur
            y, _ = _hole_sample(Xfixed, rng, ns=50000)
            y = _single_refine(Xfixed, y, steps=140)
            inserted.append(y.copy())
            # local optimize: movable = reinserted + neighbors >0.35
            # build full current array
            Xfull = np.vstack([Xcur] + inserted)
            # find fixed points close to any inserted
            # compute max cos of each row to inserted block
            ncur = Xfull.shape[0]
            nins = len(inserted)
            # inserted are last nins rows
            # compute similarities between first part and inserted
            if nins > 0:
                S = Xfull @ np.array(inserted).T  # ncur x nins
                mx_to_ins = S.max(axis=1)
            else:
                mx_to_ins = np.zeros(ncur)
            mov = np.zeros(ncur, dtype=bool)
            mov[(ncur - nins):] = True
            mov[mx_to_ins > 0.35] = True
            mov_mask = mov.astype(float)
            # partial anneal: only movable move
            Xfull = _global_anneal(Xfull, steps=90, T0=0.015, T1=0.0025, lr0=0.02, lr1=0.004, mov_mask=mov_mask)
            # write back: update Xcur part that moved? and inserted
            # split
            Xcur_new = Xfull[:Xcur.shape[0]]
            ins_new = [Xfull[Xcur.shape[0] + t].copy() for t in range(nins)]
            Xcur = Xcur_new
            inserted = ins_new
        if not ok:
            continue
        Xfull = np.vstack([Xcur] + inserted)
        # short global Adam/softmax anneal
        # two-stage: hotter then colder
        Xfull = _global_anneal(Xfull, steps=140, T0=0.02, T1=0.003, lr0=0.025, lr1=0.005)
        Xfull = _global_anneal(Xfull, steps=80, T0=0.006, T1=0.0015, lr0=0.008, lr1=0.003)
        v = _max_cos(Xfull)
        candidates.append((v, Xfull.copy()))
        if v < gval and v < 0.6:
            gval = v
            gbest = Xfull.copy()

    # pick best two distinct candidates for LP polishing
    candidates.sort(key=lambda t: t[0])
    # deduplicate roughly: keep top few
    top = candidates[:4]
    # ensure gbest included
    # LP budget: reserve until HARD
    # give each of best two up to ~ (HARD-now)/2
    best_two = []
    seen = set()
    for v, X in candidates:
        # round val to avoid duplicates
        key = round(float(v), 12)
        if key in seen:
            continue
        seen.add(key)
        best_two.append((v, X))
        if len(best_two) >= 2:
            break
    for v, X in best_two:
        if time.time() > HARD - 4:
            break
        Xp, vp = _lp_polish(X, deadline=HARD - 1.0, max_iters=30)
        if vp < gval:
            gval = vp
            gbest = Xp.copy()

    gbest = _norm(gbest)
    return gbest
