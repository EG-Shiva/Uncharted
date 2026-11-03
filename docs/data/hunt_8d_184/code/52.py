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

def _degrees(X, eps=1e-9):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    cnt = np.sum(G > m - eps, axis=1)
    per = np.max(G, axis=1)
    return m, G, cnt.astype(int), per

def _hole_sample(Xf, rng, ns=50000):
    d = Xf.shape[1]
    best_y = None
    best_m = 1e9
    # chunk to bound memory
    chunk = 10000
    done = 0
    while done < ns:
        cs = min(chunk, ns - done)
        Y = rng.standard_normal(size=(cs, d))
        Y = _norm(Y)
        S = Y @ Xf.T
        m = S.max(axis=1)
        i = int(np.argmin(m))
        if float(m[i]) < best_m:
            best_m = float(m[i])
            best_y = Y[i].copy()
        done += cs
    return best_y, best_m

def _refine_hole(Xf, y, steps=160):
    # single-point softmax refinement
    y = y / (np.linalg.norm(y) + 1e-300)
    a0, a1 = 30.0, 350.0
    l0, l1 = 0.08, 0.004
    la0, la1 = np.log(a0), np.log(a1)
    ll0, ll1 = np.log(l0), np.log(l1)
    for t in range(steps):
        f = t / max(steps - 1, 1)
        alpha = np.exp(la0 + (la1 - la0) * f)
        lr = np.exp(ll0 + (ll1 - ll0) * f)
        s = Xf @ y
        mx = float(np.max(s))
        w = np.exp(alpha * (s - mx))
        sw = float(np.sum(w)) + 1e-300
        g = w @ Xf  # (d,)
        gt = g - np.dot(g, y) * y
        y = y - lr * gt / sw
        nrm = np.linalg.norm(y)
        if not np.isfinite(nrm) or nrm == 0:
            break
        y = y / nrm
    return y

def _global_smooth(X, steps, T0=0.022, T1=0.0022, lr0=0.030, lr1=0.003):
    X = _norm(X)
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
        Dt = D - dot * X
        X = X - lr * Dt
        X = _norm(X)
    return X

def _subset_smooth(Xfull, active, steps=130, T0=0.018, T1=0.0025, lr0=0.025, lr1=0.003):
    # active: bool array (n,)
    X = _norm(Xfull)
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
        Dt = D - dot * X
        X[active] = X[active] - lr * Dt[active]
        # renormalize active only
        X[active] = _norm(X[active])
    return X

def _lp_polish(X, deadline, max_iters=40):
    X = _norm(X.copy())
    n, d = X.shape
    best = _max_cos(X)
    xb = X.copy()
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
            tl = max(1.0, deadline - time.time() - 0.4)
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": tl})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = _norm(X + res.x[:-1].reshape(n, d))
        v = _max_cos(y)
        if v < best:
            best = v; xb = y.copy(); X = y
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return xb, best

def solve():
    t0 = time.time()
    HARD = t0 + 170.0
    HOP_END = t0 + 120.0
    rng = np.random.default_rng(0)
    n, d = 184, 8
    try:
        xb0 = np.load("best.npy")
    except Exception:
        xb0 = np.load("record.npy")
    xb0 = np.asarray(xb0, dtype=np.float64).reshape(n, d)
    xb0 = _norm(xb0)
    gval = _max_cos(xb0)
    gbest = xb0.copy()

    m0, G0, cnt0, per0 = _degrees(gbest, eps=1e-9)
    # rattler: min degree, tie-break by min per-max
    order = np.argsort(cnt0 * 10.0 + (per0 - per0.min()))
    rattler = int(order[0])
    low_idx = np.where(cnt0 <= 4)[0]
    high_idx = np.where(cnt0 >= 22)[0]
    # order high by degree desc
    high_order = high_idx[np.argsort(-cnt0[high_idx], kind="stable")]
    # fallback pool by degree desc excluding rattler
    all_order = np.argsort(-cnt0, kind="stable")
    all_order = [int(i) for i in all_order if int(i) != rattler]

    # contact pairs for max-coverage greedy
    iu = np.triu_indices(n, 1)
    cf = G0[iu]
    thr = m0 - 1e-9
    sel = cf > thr
    PI = iu[0][sel].astype(int); PJ = iu[1][sel].astype(int)
    # adjacency for fast gain: list of pairs
    # greedy removal sets for each k
    ks = [1, 2, 3, 4, 6, 8]
    cand_removals = []
    # candidate high pool: prefer >=22 then others
    pool = list(map(int, high_order))
    for a in all_order:
        if a not in pool:
            pool.append(a)
    for k in ks:
        S = set([rattler])
        covered = set()  # pairs covered not needed explicitly; compute gains by scan
        # greedy pick k-1 points from pool maximizing incremental coverage
        # precompute pair endpoints
        for _ in range(k - 1):
            best_gain = -1
            best_p = -1
            # current covered mask
            # for speed, compute covered boolean from S
            # S small, so gain(p) = #pairs incident to p with other endpoint not in S (new) + pairs with both in S U {p} but not yet covered
            # simpler: count pairs with (a in S U {p} or b in S U {p}) minus pairs with (a in S or b in S)
            # = pairs incident to p with neither endpoint in S, plus pairs (p,q) with q in S (already counted? no) -> actually all pairs incident to p with other not already covering? All pairs incident to p where no endpoint in S.
            # If pair has one endpoint in S already covered, adding p does not add new coverage. So gain = #pairs (p,q) with q not in S and pair is contact and (p not in S).
            # compute quickly
            Sarr = S
            for p in pool:
                if p in S:
                    continue
                # count contacts of p to outside S
                c = 0
                # scan PI,PJ: could be ~1100 pairs, pool ~184, k<=8 => trivial
                for a, b in zip(PI, PJ):
                    if a == p and (b not in Sarr):
                        # check if a already in S? p not in S so need b not in S and a(=p) not in S => uncovered
                        # but also need that pair not already covered by S: since p not in S, covered iff b in S. So b not in S => new.
                        c += 1
                    elif b == p and (a not in Sarr):
                        c += 1
                if c > best_gain:
                    best_gain = c
                    best_p = p
            if best_p == -1:
                break
            S.add(best_p)
            if len(S) >= k:
                break
        # if still short (e.g., gain zero everywhere), fill with top pool
        for p in pool:
            if len(S) >= k:
                break
            if p not in S:
                S.add(p)
        cand_removals.append((k, sorted(list(S))))

    results = []  # (val, X)
    # surgical loop
    for ci, (k, rem) in enumerate(cand_removals):
        if time.time() > HOP_END:
            break
        rem_set = set(rem)
        keep = [i for i in range(n) if i not in rem_set]
        X_keep = gbest[np.array(keep)].copy()
        # per-candidate rng for determinism
        rsub = np.random.default_rng(1000 + ci)
        Y_list = []
        for j in range(k):
            if time.time() > HOP_END:
                break
            if Y_list:
                Xf = np.vstack([X_keep] + Y_list)
            else:
                Xf = X_keep
            y0, _ = _hole_sample(Xf, rsub, ns=50000)
            y = _refine_hole(Xf, y0, steps=160)
            Y_list.append(y.reshape(1, d))
            # local optimize: active = all Y + keep points with cos>0.35 to any Y
            X_tmp = np.vstack([X_keep] + Y_list)
            ntmp = X_tmp.shape[0]
            nk = X_keep.shape[0]
            # compute max cos of keep points to Y_list
            Yall = np.vstack(Y_list)
            S = X_tmp @ Yall.T  # ntmp x len(Y)
            mx_to_Y = S.max(axis=1)
            active = np.zeros(ntmp, dtype=bool)
            active[nk:] = True  # all reinserted
            active[:nk] = mx_to_Y[:nk] > 0.35
            # if too few active, include top few nearest
            if np.sum(active) < len(Y_list) + 3:
                # add 5 nearest keep points to newest y
                s_new = X_keep @ y
                add = np.argsort(-s_new)[:5]
                active[add] = True
            X_tmp = _subset_smooth(X_tmp, active, steps=110, T0=0.016, T1=0.0025, lr0=0.022, lr1=0.003)
            # write back
            X_keep = X_tmp[:nk].copy()
            Y_list = [X_tmp[nk + t: nk + t + 1].copy() for t in range(len(Y_list))]
        X_cand = np.vstack([X_keep] + Y_list) if Y_list else X_keep
        # short global anneal
        X_cand = _global_smooth(X_cand, steps=260, T0=0.020, T1=0.0018, lr0=0.028, lr1=0.0028)
        v = _max_cos(X_cand)
        results.append((float(v), X_cand.copy()))
    # select best two surgical candidates for LP polishing
    if results:
        results.sort(key=lambda t: t[0])
        # reserve LP polishing for best two
        polished = []
        for b in range(min(2, len(results))):
            if time.time() > HARD - 8:
                break
            Xc = results[b][1]
            # skip LP if far worse than gval (no need)
            # still polish best two regardless if time allows, but cap budget
            budget_deadline = min(HARD - 1.0, time.time() + 22.0)
            Xp, vp = _lp_polish(Xc, deadline=budget_deadline, max_iters=60)
            polished.append((float(vp), Xp))
            if vp < gval:
                gval = float(vp)
                gbest = Xp.copy()
        # also consider unpolished results that may already beat
        for (v, Xc) in results:
            if v < gval:
                gval = float(v)
                gbest = Xc.copy()
        for (v, Xp) in polished:
            if v < gval:
                gval = float(v)
                gbest = Xp.copy()
    gbest = _norm(gbest)
    if not np.all(np.isfinite(gbest)):
        gbest = _norm(xb0)
    return gbest
