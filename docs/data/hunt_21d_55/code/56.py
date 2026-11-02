import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _metrics(X, tol=1e-4):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    iu = np.triu_indices(X.shape[0], 1)
    c = G[iu]
    active = int(np.sum(c > m - tol))
    # contacts per point
    cnt = np.sum(G > m - tol, axis=1)
    ratt = int(np.sum(cnt < 5))
    return m, active, cnt, ratt

def _adam_surrogate(X0, steps, thr1, thr2_start, thr2_end, tau, lam2, lr0, lr1, seed=0):
    X = X0.copy()
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    n = X.shape[0]
    iu = np.triu_indices(n, 1)
    for t in range(1, steps + 1):
        frac = t / steps
        thr2 = thr2_start + (thr2_end - thr2_start) * frac
        lr = lr0 * (lr1 / lr0) ** frac if lr0 != lr1 else lr0
        G = X @ X.T
        # upper triangle only
        g1 = (G - thr1) / tau
        g2 = (G - thr2) / tau
        # stable sigmoid
        z1c = np.clip(g1, -30, 30)
        z2c = np.clip(g2, -30, 30)
        sig1 = 1.0 / (1.0 + np.exp(-z1c))
        sig2 = 1.0 / (1.0 + np.exp(-z2c))
        # stable softplus
        def _sp(z):
            return np.where(z > 20, z, np.where(z < -20, np.exp(z), np.log1p(np.exp(z))))
        sp1 = _sp(g1)
        sp2 = _sp(g2)
        # dL/dG
        W = (2.0 * sp1 * sig1 / tau) + lam2 * (2.0 * sp2 * sig2 / tau)
        np.fill_diagonal(W, 0.0)
        # symmetrize / half? sum over i<j gives full W for i!=j
        grad = W @ X
        # tangent projection
        proj = np.sum(grad * X, axis=1, keepdims=True)
        gt = grad - proj * X
        # clip extreme
        # Adam
        m = b1 * m + (1 - b1) * gt
        v = b2 * v + (1 - b2) * (gt * gt)
        mhat = m / (1 - b1 ** t)
        vhat = v / (1 - b2 ** t)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        # limit step norm
        X = X - step
        X = _norm_rows(X)
        if not np.all(np.isfinite(X)):
            return _norm_rows(X0)
    return _norm_rows(X)

def _lp_polish_once(X, seconds, deadline_abs):
    X = _norm_rows(X)
    n, d = X.shape
    best = _max_cos(X)
    bestX = X.copy()
    radius = 8e-4
    iu = np.triu_indices(n, 1)
    it = 0
    while time.time() < deadline_abs and time.time() < deadline_abs and it < 40:
        if seconds is not None and it * 1.0 > seconds * 10:
            pass
        it += 1
        if time.time() >= deadline_abs - 0.5:
            break
        cfull = (X @ X.T)[iu]
        m = float(cfull.max())
        sel = cfull > m - max(20 * radius, 2e-9)
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 800:
            idx = np.where(sel)[0]
            order = np.argsort(-cfull[idx])[:800]
            nsel = np.zeros_like(sel)
            nsel[idx[order]] = True
            sel = nsel
        csel = cfull[sel]
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None]*d + np.arange(d),
                               J[:, None]*d + np.arange(d),
                               np.full((p,1), n*d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p,1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals,(rows,cols)), shape=(p,nv)).tocsr()
            A_eq = coo_matrix((X.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))), shape=(n,nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1]=1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius,radius)]*(n*d)+[(None,None)], method="highs",
                          options={"time_limit": 4.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            if radius < 1e-12:
                break
            continue
        y = X + res.x[:-1].reshape(n,d)
        y = _norm_rows(y)
        vv = _max_cos(y)
        if vv < best - 1e-12:
            X, best, bestX = y, vv, y.copy()
            radius = min(radius*1.5, 0.02)
        else:
            # revert to best region
            radius *= 0.5
            if radius < 1e-12:
                break
    return bestX

def solve():
    t0 = time.time()
    LIMIT = 150.0
    deadline = t0 + LIMIT
    t_end_search = t0 + 95.0
    tau = 0.0015
    thr1 = 0.052
    n, d = 55, 21
    xb = None; best_load = 1e9
    for fn in ["best.npy","record.npy"]:
        try:
            a = np.asarray(np.load(fn), dtype=np.float64)
            if a.shape == (55,21):
                a = _norm_rows(a)
                s = _max_cos(a)
                if s < best_load:
                    best_load = s; xb = a.copy()
        except Exception:
            pass
    if xb is None:
        rng0 = np.random.default_rng(0)
        xb = _norm_rows(rng0.standard_normal((n,d)))
        best_load = _max_cos(xb)
    X0 = xb.copy()
    m0, act0, cnt0, ratt0 = _metrics(X0)
    cands = [(m0, act0, ratt0, X0.copy())]
    rng = np.random.default_rng(12345)
    # perturbation scales - deterministic set
    scales = [0.03, 0.07, 0.12, 0.20, 0.30, 0.45, 0.10, 0.25]
    # global thr2 schedule 0.075 -> 0.064
    THR2_TOP = 0.075
    THR2_BOT = 0.064
    K = len(scales)
    for k, sc in enumerate(scales):
        if time.time() > t_end_search - 2.0:
            break
        # interpolate top for this run to gradually move down
        frac_g = k / max(1, K-1)
        # each run covers a small window sliding down
        w = 0.004
        center = THR2_TOP + (THR2_BOT - THR2_TOP) * frac_g
        thr2_s = center + w
        thr2_e = center - w
        # perturb
        noise = rng.standard_normal((n,d))
        Xp = _norm_rows(X0 + sc * noise)
        # choose steps/lr based on remaining time
        steps = 2500
        # larger kicks need more steps
        if sc >= 0.2:
            steps = 3000
        lam2 = 5.0
        # first threshold weight is 1, second lam2
        lr0, lr1 = 0.015, 0.0015
        if sc < 0.1:
            lr0, lr1 = 0.008, 0.001
        Xq = _adam_surrogate(Xp, steps, thr1, thr2_s, thr2_e, tau, lam2, lr0, lr1)
        if time.time() > deadline - 25.0:
            # still record
            try:
                mm, aa, _, rr = _metrics(Xq)
                cands.append((mm, aa, rr, Xq.copy()))
            except Exception:
                pass
            break
        try:
            mm, aa, _, rr = _metrics(Xq)
        except Exception:
            continue
        cands.append((mm, aa, rr, Xq.copy()))
        # also try a short second-stage quench at fixed low threshold for same Xq
        if time.time() < t_end_search - 5.0 and k % 2 == 0:
            Xq2 = _adam_surrogate(Xq, 1200, thr1, 0.066, 0.064, tau, 8.0, 0.005, 0.0008)
            try:
                mm2, aa2, _, rr2 = _metrics(Xq2)
                cands.append((mm2, aa2, rr2, Xq2.copy()))
            except Exception:
                pass
    # Pareto selection: prefer slack-rich (rattlers>=3)
    # sort candidates by max
    cands_sorted = sorted(cands, key=lambda t: t[0])
    best_overall = cands_sorted[0]
    # find slack-rich candidates
    slack = [c for c in cands if c[2] >= 3]
    # also consider sparse: active < 600
    sparse_slack = [c for c in slack if c[1] < 900]
    polish_target = None
    if sparse_slack:
        # best max among sparse slack, but allow slightly worse: choose minimal active among those within 0.004 of best slack max
        minmax = min(c[0] for c in sparse_slack)
        within = [c for c in sparse_slack if c[0] < minmax + 0.004]
        # pick sparsest
        polish_target = min(within, key=lambda t: (t[1], t[0]))
    elif slack:
        polish_target = min(slack, key=lambda t: (t[1], t[0]))
    else:
        # no rattlers: pick sparsest candidate within tolerance to create slack next time, but still polish sparsest-good
        # take candidates within 0.002 of best and pick minimal active
        b0 = best_overall[0]
        within = [c for c in cands if c[0] < b0 + 0.002]
        polish_target = min(within, key=lambda t: (t[1], t[0]))
    # polish only best slack-rich candidate with LP, sparingly
    remain = deadline - time.time()
    # reserve 3s for final
    pol_sec = min(20.0, max(remain - 3.0, 0.0))
    polished = None
    if polish_target is not None and pol_sec > 3.0:
        lp_deadline = time.time() + pol_sec
        # cap to 18s
        lp_deadline = min(lp_deadline, t0 + 148.0)
        try:
            Xp_pol = _lp_polish_once(polish_target[3], pol_sec, lp_deadline)
            mp, ap, _, rp = _metrics(Xp_pol)
            polished = (mp, ap, rp, Xp_pol)
            cands.append(polished)
        except Exception:
            pass
    # final choice: lowest max among all (ensures we never regress vs incumbent)
    final = min(cands, key=lambda t: t[0])
    Xbest = _norm_rows(final[3])
    # safety: if best is worse than loaded best, return loaded best
    if _max_cos(Xbest) > best_load + 1e-12:
        return _norm_rows(X0)
    # final normalize and tiny shrink not needed (we report exact)
    return _norm_rows(Xbest)
