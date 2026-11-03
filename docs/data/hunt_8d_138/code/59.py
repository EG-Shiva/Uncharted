import time
import numpy as np

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _near_count(X, tol=1e-6):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    iu = np.triu_indices(X.shape[0], 1)
    c = G[iu]
    return int(np.count_nonzero(c > m - tol)), m

def _lse_jam(X0, beta, maxiter):
    from scipy.optimize import minimize
    n, d = X0.shape
    iu = np.triu_indices(n, 1)
    Y = _norm(X0.copy())
    def fun(Yf):
        Ym = Yf.reshape(n, d)
        nr = np.sqrt(np.sum(Ym*Ym, axis=1, keepdims=True)) + 1e-30
        X = Ym / nr
        G = X @ X.T
        c = G[iu]
        mv = c.max()
        E = np.exp(beta*(c-mv))
        S = E.sum() + 1e-300
        f = float(mv + np.log(S)/beta)
        W = np.zeros((n, n))
        W[iu] = E
        W = W + W.T
        GX = (W @ X) / S
        dot = np.sum(GX*X, axis=1, keepdims=True)
        Gt = GX - dot*X
        GY = Gt / nr
        return f, GY.ravel()
    try:
        res = minimize(fun, Y.ravel(), jac=True, method='L-BFGS-B',
                       options={'maxiter': int(maxiter), 'ftol':1e-14, 'gtol':1e-6, 'maxcor':12})
        Y = _norm(res.x.reshape(n, d))
    except Exception:
        pass
    return _norm(Y)

def _lp_polish(X, deadline, radius0=1e-2, active=1e-3, cap=3000):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    X = _norm(X)
    n, d = X.shape
    best = _max_cos(X)
    radius = radius0
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        Gfull = X @ X.T
        c = Gfull[iu]
        m = c.max()
        thr = m - active
        sel = c > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > cap:
            idx = np.where(sel)[0]
            order = np.argsort(-c[idx])[:cap]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n*d+1
        rows = np.repeat(np.arange(p), 2*d+1)
        cols = np.concatenate([I[:, None]*d + np.arange(d), J[:, None]*d + np.arange(d), np.full((p,1), n*d)],1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p,1))],1).ravel()
        try:
            A_ub = coo_matrix((vals,(rows,cols)), shape=(p,nv)).tocsr()
            A_eq = coo_matrix((X.ravel(),(np.repeat(np.arange(n),d), np.arange(n*d))), shape=(n,nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1]=1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius,radius)]*(n*d)+[(None,None)], method='highs')
        except Exception:
            radius*=0.5
            continue
        if res.status!=0:
            radius*=0.5
            continue
        y = X + res.x[:-1].reshape(n,d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius*1.5, 0.05)
        else:
            radius*=0.5
    return X

def _refine_holes(Y0, Xcur, beta=120.0, iters=220):
    Y = _norm(Y0.copy())
    B = Y.shape[0]
    lr0 = 0.35
    for it in range(iters):
        S = Y @ Xcur.T
        mx = S.max(axis=1, keepdims=True)
        E = np.exp(beta*(S-mx))
        Sm = E.sum(axis=1, keepdims=True) + 1e-300
        W = E / Sm
        G = W @ Xcur
        dot = np.sum(G*Y, axis=1, keepdims=True)
        Gt = G - dot*Y
        lr = lr0*(1.0 - it/iters) + 0.003
        Y = Y - lr*Gt
        Y = _norm(Y)
    return Y

def _subset_relax(cur, subset, beta=160.0, maxiter=60):
    from scipy.optimize import minimize
    cur = _norm(cur.copy())
    n, d = cur.shape
    s = len(subset)
    if s <= 1:
        return cur
    subset = np.array(subset, dtype=int)
    mask = np.ones(n, dtype=bool); mask[subset] = False
    fixed_idx = np.where(mask)[0]
    F = cur[fixed_idx]
    Z0 = cur[subset].copy()
    iu_s = np.triu_indices(s, 1)
    has_internal = len(iu_s[0]) > 0
    def fun(Yf):
        Ym = Yf.reshape(s, d)
        nr = np.sqrt(np.sum(Ym*Ym, axis=1, keepdims=True)) + 1e-30
        Z = Ym / nr
        Sc = Z @ F.T
        if has_internal:
            Gss = Z @ Z.T
            ci = Gss[iu_s]
            allc = np.concatenate([Sc.ravel(), ci])
        else:
            allc = Sc.ravel()
        mv = allc.max()
        E = np.exp(beta*(allc-mv))
        S = E.sum() + 1e-300
        f = float(mv + np.log(S)/beta)
        if has_internal:
            nc = Sc.size
            Ec = E[:nc].reshape(s, len(fixed_idx))
            Ei = np.zeros((s, s))
            Ei[iu_s] = E[nc:]
            Ei = Ei + Ei.T
            GZ = (Ei @ Z + Ec @ F) / S
        else:
            Ec = E.reshape(s, len(fixed_idx))
            GZ = (Ec @ F) / S
        dot = np.sum(GZ*Z, axis=1, keepdims=True)
        Gt = GZ - dot*Z
        GY = Gt / nr
        return f, GY.ravel()
    try:
        res = minimize(fun, Z0.ravel(), jac=True, method='L-BFGS-B',
                       options={'maxiter': int(maxiter), 'ftol':1e-14, 'gtol':1e-5, 'maxcor':8})
        Z1 = _norm(res.x.reshape(s, d))
        cur[subset] = Z1
    except Exception:
        pass
    return _norm(cur)

def solve():
    rng = np.random.default_rng(7)
    t0 = time.time()
    deadline = t0 + 170.0
    n, d = 138, 8
    best = None; best_val = 1e9
    for fn in ("best.npy","record.npy"):
        try:
            v = np.load(fn).astype(float)
            if v.shape == (n, d):
                v = _norm(v)
                c = _max_cos(v)
                if c < best_val:
                    best_val = c; best = v.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.standard_normal((n, d)))
        best_val = _max_cos(best)
    incumbent = best.copy()
    inc_val = best_val
    inc_near, _ = _near_count(incumbent, tol=1e-6)

    # degrees at m-1e-6
    try:
        G0 = incumbent @ incumbent.T
        np.fill_diagonal(G0, -2.0)
        m0 = float(G0.max())
        thr = m0 - 1e-6
        deg = np.count_nonzero(G0 > thr, axis=1)
        order_desc = np.argsort(-deg, kind='stable')
        order_asc = np.argsort(deg, kind='stable')
        rattlers = order_asc[:2].tolist()
    except Exception:
        deg = np.zeros(n, dtype=int)
        order_desc = np.arange(n)
        rattlers = [0, 1]

    ks = [3, 5, 8, 12, 18]
    # strict budget: leave 25s for final polish
    for k in ks:
        if time.time() > deadline - 28:
            break
        # two removal-set variants per k
        for variant in range(2):
            if time.time() > deadline - 28:
                break
            if variant == 0:
                topk = order_desc[:k].tolist()
            else:
                # different high-degree set: sample k from top 2k
                pool = order_desc[:max(k*2, k+8)].tolist()
                # deterministic pseudo-random choice seeded by k
                rs = np.random.default_rng(1000+k)
                # ensure diversity: shuffle pool and take k not identical to variant0
                perm = rs.permutation(len(pool))
                topk = [pool[i] for i in perm[:k]]
                if set(topk) == set(order_desc[:k].tolist()):
                    # force difference
                    if len(pool) > k:
                        topk[-1] = pool[k]
            removed = []
            for r in rattlers:
                if r not in topk:
                    removed.append(int(r))
                # if rattler overlaps topk (unlikely), still need 2 rattlers: add next lowest
            # ensure exactly 2 rattlers + k distinct
            if len(removed) < 2:
                for cand in order_asc:
                    cand = int(cand)
                    if cand not in topk and cand not in removed:
                        removed.append(cand)
                    if len(removed) >= 2:
                        break
            removed = removed + [int(x) for x in topk]
            # dedup preserve order
            seen = set(); rem2 = []
            for x in removed:
                if x not in seen:
                    seen.add(x); rem2.append(x)
            removed = rem2
            # core indices
            remset = set(removed)
            core_idx = [i for i in range(n) if i not in remset]
            Xc = incumbent[core_idx].copy()
            # relax remaining core (short)
            try:
                # small LSE
                if time.time() < deadline - 25:
                    Xc = _lse_jam(Xc, beta=180.0, maxiter=120)
                if time.time() < deadline - 25:
                    Xc = _lp_polish(Xc, min(deadline-24, time.time()+1.5), radius0=8e-3, active=8e-4, cap=2500)
            except Exception:
                pass
            cur = Xc.copy()
            # reinsert one at a time
            n_insert = len(removed)
            ok = True
            for step in range(n_insert):
                if time.time() > deadline - 24:
                    ok = False
                    break
                # 20k random candidates
                try:
                    Rc = rng.standard_normal((20000, d))
                    Rc = _norm(Rc)
                    # scores: max dot to cur
                    S = Rc @ cur.T
                    mx = S.max(axis=1)
                    # best indices
                    B = 6
                    part = np.argpartition(mx, B)[:B]
                    # sort them
                    part = part[np.argsort(mx[part])]
                    Y0 = Rc[part].copy()
                    Yr = _refine_holes(Y0, cur, beta=120.0, iters=200)
                    # evaluate pool: all random best + refined
                    # best random overall
                    best_rand_i = int(np.argmin(mx))
                    cand_pool = [Rc[best_rand_i]]
                    for b in range(B):
                        cand_pool.append(Y0[b])
                        cand_pool.append(Yr[b])
                    best_y = None; best_s = 1e9
                    for cp in cand_pool:
                        cp = cp / (np.linalg.norm(cp)+1e-30)
                        s = float(cur @ cp.reshape(d, 1).ravel() if False else (cur @ cp).max())
                        if s < best_s:
                            best_s = s; best_y = cp.copy()
                    # append
                    cur = np.vstack([cur, best_y.reshape(1, d)])
                    cur = _norm(cur)
                    # local relax with neighbours >0.35
                    dots = (cur[:-1] @ cur[-1])
                    nb = np.where(dots > 0.35)[0].tolist()
                    subset = nb + [cur.shape[0]-1]
                    # limit subset size for speed
                    if len(subset) > 40:
                        # keep closest 39 + new
                        o = np.argsort(-dots[nb])[:39]
                        subset = [nb[i] for i in o] + [cur.shape[0]-1]
                    if len(subset) > 1:
                        cur = _subset_relax(cur, subset, beta=160.0, maxiter=50)
                except Exception:
                    # fallback: random insert
                    try:
                        y = _norm(rng.standard_normal((1, d)))
                        cur = _norm(np.vstack([cur, y]))
                    except Exception:
                        ok = False
                        break
            if not ok:
                continue
            # after all insertions: LSE + LP polishing
            try:
                if time.time() > deadline - 22:
                    continue
                cur = _lse_jam(cur, beta=220.0, maxiter=150)
                v1 = _max_cos(cur)
                cur = _lse_jam(cur, beta=500.0, maxiter=100)
                v2 = _max_cos(cur)
                cur = _lp_polish(cur, min(deadline-20, time.time()+5.0), radius0=1e-2, active=1e-3, cap=3000)
                v3 = _max_cos(cur)
                v = min(v1, v2, v3)
                # near count of cur
                nc, mv = _near_count(cur, tol=1e-6)
                improved = False
                if v3 < inc_val - 1e-12:
                    improved = True
                elif v3 <= inc_val + 1e-9 and nc < inc_near:
                    improved = True
                # also consider v2 if LP hurt
                if not improved:
                    # check pre-LP value
                    pass
                if improved:
                    incumbent = _norm(cur.copy())
                    inc_val = v3
                    inc_near = nc
                    # update degrees/rattlers for next k
                    try:
                        G1 = incumbent @ incumbent.T
                        np.fill_diagonal(G1, -2.0)
                        mm = float(G1.max())
                        deg = np.count_nonzero(G1 > mm-1e-6, axis=1)
                        order_desc = np.argsort(-deg, kind='stable')
                        order_asc = np.argsort(deg, kind='stable')
                        rattlers = order_asc[:2].tolist()
                    except Exception:
                        pass
                    # keep going to try further improvement
                # else discard
            except Exception:
                continue
    # final polish of incumbent with remaining time
    try:
        remain = deadline - time.time()
        if remain > 4:
            Yf = _lp_polish(incumbent, deadline-0.8, radius0=1e-2, active=1e-3, cap=3000)
            vf = _max_cos(Yf)
            if vf < inc_val:
                incumbent = Yf
                inc_val = vf
    except Exception:
        pass
    incumbent = _norm(incumbent)
    return incumbent
