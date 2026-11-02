import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm_rows(x):
    n = np.sqrt(np.sum(x*x, axis=1, keepdims=True))
    n = np.maximum(n, 1e-12)
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _stats(x):
    n = x.shape[0]
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    m = float(np.max(g))
    mask = g > m - 1e-4
    # count pairs i<j
    iu = np.triu_indices(n, 1)
    active = int(np.sum(g[iu] > m - 1e-4))
    # per point contacts (count symmetric, divide by 2 since mask includes both)
    deg = np.sum(mask, axis=1)
    rattlers = int(np.sum(deg < 5))
    zero = int(np.sum(deg == 0))
    return m, active, rattlers, zero, deg

def _surrogate_adam(X0, steps, t1, t2, lam, tau, lr, rng=None):
    X = X0.copy()
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps+1):
        G = X @ X.T
        # upper computation, but need full symmetric F
        # stable sigmoid + softplus
        S1 = (G - t1) / tau
        S2 = (G - t2) / tau
        # sigmoid
        # clip for exp
        Sig1 = np.empty_like(S1)
        Sig2 = np.empty_like(S2)
        # use stable: sigmoid(s)=1/(1+exp(-s))
        # for large |s| clip
        np.clip(S1, -30, 30, out=Sig1)  # reuse as clipped
        np.clip(S2, -30, 30, out=Sig2)
        Sig1 = 1.0/(1.0+np.exp(-Sig1))
        Sig2 = 1.0/(1.0+np.exp(-Sig2))
        # softplus: log1p(exp(s)) stable
        # for s>20 -> s, for s<-20 -> exp(s)
        Sp1 = np.empty_like(S1)
        Sp2 = np.empty_like(S2)
        # vectorized with masks
        m1a = S1 > 20
        m1b = S1 < -20
        m1c = ~(m1a | m1b)
        Sp1[m1a] = S1[m1a]
        Sp1[m1b] = np.exp(S1[m1b])
        Sp1[m1c] = np.log1p(np.exp(S1[m1c]))
        m2a = S2 > 20
        m2b = S2 < -20
        m2c = ~(m2a | m2b)
        Sp2[m2a] = S2[m2a]
        Sp2[m2b] = np.exp(S2[m2b])
        Sp2[m2c] = np.log1p(np.exp(S2[m2c]))
        F = (2.0*Sp1*Sig1 + lam*2.0*Sp2*Sig2) / tau
        np.fill_diagonal(F, 0.0)
        grad = F @ X
        proj = np.sum(grad*X, axis=1, keepdims=True)
        gt = grad - proj*X
        # Adam
        m = b1*m + (1-b1)*gt
        v = b2*v + (1-b2)*(gt*gt)
        mhat = m/(1-b1**t)
        vhat = v/(1-b2**t)
        step = lr*mhat/(np.sqrt(vhat)+eps)
        X = X - step
        # project to sphere
        nr = np.sqrt(np.sum(X*X, axis=1, keepdims=True))
        nr = np.maximum(nr, 1e-12)
        X /= nr
        if not np.all(np.isfinite(X)):
            return X0
    return X

def _lse_adam(X0, steps, a0, a1, lr0, lr1):
    X = X0.copy()
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps+1):
        frac = t/steps
        alpha = a0*(a1/a0)**frac
        lr = lr0*(lr1/lr0)**frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        Mx = np.max(G)
        E = np.exp(alpha*(G-Mx))
        np.fill_diagonal(E, 0.0)
        S = np.sum(E)
        if not np.isfinite(S) or S<=0:
            continue
        grad = (E @ X)/S
        proj = np.sum(grad*X, axis=1, keepdims=True)
        gt = grad - proj*X
        m = b1*m+(1-b1)*gt
        v = b2*v+(1-b2)*(gt*gt)
        mhat = m/(1-b1**t)
        vhat = v/(1-b2**t)
        X = X - lr*mhat/(np.sqrt(vhat)+eps)
        nr = np.sqrt(np.sum(X*X, axis=1, keepdims=True))
        nr = np.maximum(nr, 1e-12)
        X /= nr
    return X

def _lp_polish(x, seconds, t0):
    x = _norm_rows(x)
    n, d = x.shape
    best = _max_cos(x)
    bx = x.copy()
    radius = 2e-3
    iu = np.triu_indices(n, 1)
    while time.time()-t0 < seconds and radius > 1e-14:
        cfull = (x @ x.T)[iu]
        mcur = float(np.max(cfull))
        thr = mcur - max(10*radius, 2e-7)
        sel = cfull > thr
        if np.sum(sel) > 600:
            idx = np.where(sel)[0]
            order = np.argsort(-cfull[idx])[:600]
            sel2 = np.zeros_like(sel)
            sel2[idx[order]] = True
            sel = sel2
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n*d+1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2*d+1)
        cols = np.concatenate([I[:,None]*d+np.arange(d), J[:,None]*d+np.arange(d), np.full((p,1), n*d)],1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p,1))],1).ravel()
        try:
            A_ub = coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
            A_eq = coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1]=1.0
        tl = seconds-(time.time()-t0)
        if tl <= 0.5:
            break
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cfull[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius,radius)]*(n*d)+[(None,None)], method="highs",
                          options={"time_limit": max(1.0, tl)})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n,d)
        y = _norm_rows(y)
        vv = _max_cos(y)
        if vv < best:
            best = vv
            bx = y.copy()
            x = y.copy()
            radius = min(radius*1.5, 0.02)
        else:
            radius *= 0.5
        if time.time()-t0 >= seconds:
            break
    return bx

def solve():
    t_start = time.time()
    TIME_LIMIT = 170.0
    SEARCH_END = t_start + 62.0
    REFINE_END = t_start + 140.0
    rng = np.random.default_rng(12345)
    n, d = 55, 21
    best = None
    best_score = 1e9
    for fn in ["best.npy","record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape==(55,21):
                a = _norm_rows(a)
                s = _max_cos(a)
                if s < best_score:
                    best_score = s
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _norm_rows(rng.standard_normal(size=(n,d)))
        best_score = _max_cos(best)
    incumbent = best.copy()
    incumbent_score = best_score
    # archive of (X, max, active, rattlers)
    cands = []
    # seed archive with incumbent perturbed slightly evaluated
    m0,a0,r0,_,_ = _stats(incumbent)
    cands.append((incumbent.copy(), m0, a0, r0))
    tau = 0.0015
    t1 = 0.052
    lam = 4.0
    trial = 0
    sigmas = [0.10, 0.20, 0.06, 0.30, 0.12, 0.04]
    while time.time() < SEARCH_END:
        trial += 1
        elapsed = time.time()-t_start
        prog = min(1.0, elapsed/60.0)
        t2 = 0.075 - (0.075-0.064)*prog
        sigma = sigmas[trial % len(sigmas)]
        # perturb incumbent (not drifting cur, to keep diversity near record)
        base = incumbent if (trial % 3 != 0) else (cands[rng.integers(len(cands))][0] if len(cands)>1 and rng.random()<0.4 else incumbent)
        prop = base + sigma*rng.standard_normal(size=(n,d))
        prop = _norm_rows(prop)
        # occasionally hole-reinsert one overloaded point for topology change
        if trial % 4 == 0:
            G = prop @ prop.T
            np.fill_diagonal(G, -2.0)
            mx = np.max(G)
            cnt = np.sum(G > mx-1e-3, axis=1)
            victim = int(np.argmax(cnt))
            # find hole
            cand_h = _norm_rows(rng.standard_normal(size=(800,d)))
            dots = cand_h @ prop.T
            # exclude victim column
            dots[:,victim] = -1e9
            md = np.max(dots, axis=1)
            prop[victim] = cand_h[int(np.argmin(md))]
            prop = _norm_rows(prop)
        steps = 550
        # shrink steps if near deadline
        if SEARCH_END - time.time() < 8:
            steps = 300
        lr = 0.008
        try:
            new = _surrogate_adam(prop, steps, t1, t2, lam, tau, lr)
        except Exception:
            continue
        try:
            m,a,r,_,_ = _stats(new)
        except Exception:
            continue
        cands.append((new.copy(), m, a, r))
        if m < best_score and np.all(np.isfinite(new)):
            # keep global best but do not let search collapse to jammed vertex alone;
            # still record it
            best_score = m
            best = new.copy()
        if len(cands) > 120:
            # prune worst max to bound memory (keep diverse)
            # keep incumbent + 119 most recent? simple truncate oldest non-incumbent
            # remove the worst-max candidate beyond 80
            if len(cands) > 120:
                # drop candidate with largest m among non-best
                ms = [c[1] for c in cands]
                idx_worst = int(np.argmax(ms))
                # don't drop current best
                if cands[idx_worst][1] > best_score + 1e-12:
                    del cands[idx_worst]
                else:
                    # drop second worst
                    order = np.argsort(ms)
                    del cands[int(order[-1])]
        if time.time() >= SEARCH_END:
            break
    # Pareto selection: require >=3 rattlers (deg<5), minimize composite
    # composite = max + w*active
    w = 2e-6
    feas = [c for c in cands if c[3] >= 3 and np.all(np.isfinite(c[0]))]
    if len(feas)==0:
        feas = cands
    # sort by composite
    feas_sorted = sorted(feas, key=lambda c: c[1]+w*c[2])
    # take top 3 slack-rich for quick LSE trial
    topK = feas_sorted[:3]
    refined = []
    for k,(Xk,mk,ak,rk) in enumerate(topK):
        if time.time() > REFINE_END-10:
            refined.append((Xk,mk,ak,rk))
            continue
        try:
            # short LSE descent to bring max down while staying loose
            Y = _lse_adam(Xk, 600, 30.0, 500.0, 0.015, 0.002)
            m,a,r,_,_ = _stats(Y)
            refined.append((Y,m,a,r))
            # also track global best
            if m < best_score:
                best_score = m
                best = Y.copy()
        except Exception:
            refined.append((Xk,mk,ak,rk))
        if time.time() > REFINE_END:
            break
    # choose best slack-rich parent among refined by composite (slightly favor low active)
    refined_sorted = sorted(refined, key=lambda c: c[1]+w*c[2])
    parent = refined_sorted[0][0]
    # extra LSE squeeze on parent if time
    if time.time() < REFINE_END:
        try:
            remain = REFINE_END - time.time()
            steps2 = 700 if remain > 25 else 400
            parent2 = _lse_adam(parent, steps2, 80.0, 800.0, 0.008, 0.0008)
            m2,_,_,_,_ = _stats(parent2)
            # accept if not much worse (allow +0.001 to keep slack)
            if m2 < refined_sorted[0][1] + 0.002:
                parent = parent2
        except Exception:
            pass
    # single LP polish on parent only
    remain_total = TIME_LIMIT - (time.time()-t_start)
    lp_time = min(max(remain_total-2.0, 4.0), 22.0)
    if lp_time > 4:
        try:
            polished = _lp_polish(parent, lp_time, t0=time.time())
            mp,_,_,_,_ = _stats(polished)
            # compare with best/incumbent
            if mp < best_score:
                best_score = mp
                best = polished.copy()
        except Exception:
            pass
    # final safety: return at least incumbent
    if best_score > incumbent_score and incumbent is not None:
        best = incumbent.copy()
    best = _norm_rows(np.asarray(best, dtype=np.float64))
    # guard NaN/zero
    best = np.where(np.isfinite(best), best, 0.0)
    nr = np.linalg.norm(best, axis=1, keepdims=True)
    nr = np.maximum(nr, 1e-12)
    # if any zero row fallback to incumbent row
    bad = (np.linalg.norm(best,axis=1)<1e-9) | (~np.all(np.isfinite(best),axis=1))
    if np.any(bad):
        for i in np.where(bad)[0]:
            best[i] = incumbent[i]
        best = _norm_rows(best)
    return best
