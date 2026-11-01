import time
import numpy as np

def _norm_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds, deadline=None):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    x = _norm_rows(x.copy())
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if deadline is not None and time.time() > deadline:
            break
        if radius < 1e-14:
            break
        G = x @ x.T
        cv = G[iu]
        m = float(cv.max())
        thresh = m - max(20 * radius, 1e-9)
        sel = cv > thresh
        if np.count_nonzero(sel) == 0:
            break
        I = iu[0][sel]; J = iu[1][sel]
        csub = cv[sel]
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
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csub, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm_rows(y)
        if not np.all(np.isfinite(y)):
            radius *= 0.5
            continue
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _quench(X_init, steps, T0, T1, lr, mu=0.88, deadline=None):
    X = X_init.copy()
    n, d = X.shape
    V = np.zeros_like(X)
    lt0 = np.log(T0); lt1 = np.log(T1)
    G0 = X @ X.T
    np.fill_diagonal(G0, -2.0)
    best_m = float(G0.max())
    best_X = X.copy()
    for s in range(steps):
        if deadline is not None and (s % 200 == 0) and time.time() > deadline:
            break
        frac = s / max(1, steps - 1)
        T = float(np.exp(lt0 + (lt1 - lt0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -1e100)
        m = float(G.max())
        if m < best_m:
            best_m = m
            best_X = X.copy()
        Q = (G - m) / T
        np.clip(Q, -50, 0, out=Q)
        E = np.exp(Q)
        np.fill_diagonal(E, 0.0)
        rs = E.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        Grad = (E @ X) / rs
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        # clip large grads
        gn = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        sc = np.ones_like(gn)
        big = gn[:, 0] > 2.0
        sc[big] = 2.0 / gn[big]
        Gt = Gt * sc
        if not np.all(np.isfinite(Gt)):
            break
        V = mu * V + Gt
        X = X - lr * V
        X = _norm_rows(X)
    Gf = X @ X.T
    np.fill_diagonal(Gf, -2.0)
    mf = float(Gf.max())
    if mf < best_m:
        best_m = mf
        best_X = X.copy()
    return best_X

def _single_refine(s, R, steps=250, T=0.004, lr=0.12, mu=0.8):
    s = s / (np.linalg.norm(s) + 1e-12)
    v = np.zeros_like(s)
    for _ in range(steps):
        g = R @ s
        m = float(g.max())
        e = np.exp(np.clip((g - m) / T, -50, 0))
        den = e.sum() + 1e-30
        Grad = (R.T @ e) / den
        Gt = Grad - float(Grad @ s) * s
        v = mu * v + Gt
        s = s - lr * v
        nn = float(np.linalg.norm(s))
        if not np.isfinite(nn) or nn == 0:
            break
        s = s / nn
    return s

def _find_hole(R, rng, n_sample=10000, n_cand=3):
    n, d = R.shape
    Z = _norm_rows(rng.standard_normal(size=(n_sample, d)))
    S = Z @ R.T
    scores = S.max(axis=1)
    idx = np.argpartition(scores, n_cand)[:n_cand]
    best_s = None
    best_v = 1e100
    for k in idx:
        s0 = Z[k]
        s1 = _single_refine(s0, R, steps=250, T=0.005, lr=0.12, mu=0.8)
        s1 = _single_refine(s1, R, steps=250, T=0.0015, lr=0.07, mu=0.8)
        v = float((R @ s1).max())
        if v < best_v:
            best_v = v
            best_s = s1
    best_s = best_s / (np.linalg.norm(best_s) + 1e-12)
    return best_s

def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    EXPLORE_END = t0 + 144.0
    rng = np.random.default_rng(12345)
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=np.float64)
            if a.shape == (71, 12):
                a = _norm_rows(a)
                cands.append(a)
        except Exception:
            pass
    if len(cands) == 0:
        cands.append(_norm_rows(rng.standard_normal(size=(71, 12))))
    def mc(a):
        return _max_cos(a)
    bestX = min(cands, key=mc).copy()
    bestVal = float(mc(bestX))
    curX = bestX.copy()
    curVal = bestVal
    n, d = bestX.shape
    archive = [(bestVal, bestX.copy())]
    def push_archive(v, X):
        archive.append((v, X.copy()))
        archive.sort(key=lambda t: t[0])
        del archive[6:]
    trial = 0
    sigmas = [0.03, 0.08, 0.18, 0.05, 0.12, 0.025]
    noimp = 0
    while time.time() < EXPLORE_END and time.time() < HARD_END - 28 and trial < 500:
        trial += 1
        X0 = curX.copy()
        rtype = rng.random()
        if rtype < 0.40:
            sigma = sigmas[trial % len(sigmas)]
            G = X0 @ X0.T
            np.fill_diagonal(G, -2.0)
            mx = G.max(axis=1)
            cnt = (G > bestVal - 1e-3).sum(axis=1)
            order = np.argsort(-cnt)
            w = np.ones(n)
            # overloaded hubs get bigger kick
            topk = order[:10]
            w[topk] = 2.0
            w[mx > bestVal - 5e-4] *= 1.4
            noise = rng.standard_normal(size=X0.shape)
            X0 = X0 + sigma * noise * w[:, None]
            X0 = _norm_rows(X0)
        elif rtype < 0.72:
            G = X0 @ X0.T
            np.fill_diagonal(G, -2.0)
            mx = G.max(axis=1)
            cnt = (G > bestVal - 1e-3).sum(axis=1)
            score = mx + 0.0005 * cnt
            order = np.argsort(-score)
            cand_v = order[:6]
            victim = cand_v[rng.integers(0, len(cand_v))]
            R = np.delete(X0, victim, axis=0)
            hole = _find_hole(R, rng, n_sample=8000, n_cand=3)
            X0 = R.copy()
            X0 = np.vstack([X0[:victim], hole[None, :], X0[victim:]])
            X0 = X0 + 0.008 * rng.standard_normal(size=X0.shape)
            X0 = _norm_rows(X0)
        elif rtype < 0.88:
            # double kill
            G = X0 @ X0.T
            np.fill_diagonal(G, -2.0)
            mx = G.max(axis=1)
            cnt = (G > bestVal - 1e-3).sum(axis=1)
            score = mx + 0.0005 * cnt
            order = np.argsort(-score)
            k = 2 if rng.random() < 0.7 else 3
            victims = order[:8]
            victims = rng.choice(victims, size=k, replace=False)
            victims = np.sort(victims)
            R = np.delete(X0, victims, axis=0)
            newpts = []
            Rg = R.copy()
            for _ in range(k):
                hole = _find_hole(Rg, rng, n_sample=6000, n_cand=2)
                newpts.append(hole)
                Rg = np.vstack([Rg, hole[None, :]])
            X0 = R.copy()
            for vv, hh in zip(victims, newpts):
                # insert accounting shift: rebuild progressively
                pass
            # simpler: replace in place
            X0full = curX.copy()
            for vv, hh in zip(victims, newpts):
                X0full[vv] = hh
            X0 = X0full + 0.008 * rng.standard_normal(size=X0full.shape)
            X0 = _norm_rows(X0)
        else:
            sigma = 0.10 + 0.08 * rng.random()
            X0 = X0 + sigma * rng.standard_normal(size=X0.shape)
            X0 = _norm_rows(X0)
        # random quench params
        T0 = float(np.exp(rng.uniform(np.log(0.008), np.log(0.03))))
        T1 = float(np.exp(rng.uniform(np.log(0.0007), np.log(0.0035))))
        lr = float(rng.uniform(0.04, 0.11))
        mu = float(rng.uniform(0.82, 0.92))
        steps = int(rng.integers(600, 1300))
        Xopt = _quench(X0, steps=steps, T0=T0, T1=T1, lr=lr, mu=mu, deadline=HARD_END - 22)
        v = float(_max_cos(Xopt))
        if v < bestVal - 1e-12:
            Xp = _lp_polish(Xopt, seconds=2.0, deadline=HARD_END - 14)
            v2 = float(_max_cos(Xp))
            if v2 < v:
                Xopt, v = Xp, v2
            if v < bestVal:
                bestX, bestVal = Xopt.copy(), v
                curX, curVal = Xopt.copy(), v
                push_archive(v, Xopt)
                noimp = 0
            else:
                noimp += 1
        elif v < curVal:
            curX, curVal = Xopt.copy(), v
            noimp = 0
            if v < bestVal + 0.0006 and (trial % 3 == 0):
                Xp = _lp_polish(Xopt, seconds=1.0, deadline=HARD_END - 14)
                v2 = float(_max_cos(Xp))
                if v2 < bestVal:
                    bestX, bestVal = Xp.copy(), v2
                    curX, curVal = Xp.copy(), v2
                    push_archive(v2, Xp)
                    noimp = 0
        else:
            noimp += 1
            if v < curVal + 0.002 and rng.random() < 0.15:
                curX, curVal = Xopt.copy(), v
            if v < bestVal + 0.0004 and (trial % 4 == 0):
                Xp = _lp_polish(Xopt, seconds=0.8, deadline=HARD_END - 14)
                v2 = float(_max_cos(Xp))
                if v2 < bestVal:
                    bestX, bestVal = Xp.copy(), v2
                    curX, curVal = Xp.copy(), v2
                    push_archive(v2, Xp)
                    noimp = 0
        if noimp >= 18:
            # restart from best or archive
            noimp = 0
            if len(archive) > 1 and rng.random() < 0.5:
                pick = archive[rng.integers(0, min(len(archive), 4))][1]
                curX = pick + 0.06 * rng.standard_normal(size=pick.shape)
                curX = _norm_rows(curX)
                curVal = float(_max_cos(curX))
            else:
                curX = bestX + 0.05 * rng.standard_normal(size=bestX.shape)
                curX = _norm_rows(curX)
                curVal = float(_max_cos(curX))
        if time.time() > HARD_END - 28:
            break
    # guarded low-temp tightening of archive
    remain = HARD_END - time.time() - 2.0
    if remain > 4 and len(archive) > 0:
        # sort archive
        archive.sort(key=lambda t: t[0])
        for ai in range(min(len(archive), 3)):
            if time.time() > HARD_END - 8:
                break
            av, aX = archive[ai]
            Xt = _quench(aX, steps=700, T0=0.004, T1=0.0006, lr=0.035, mu=0.88, deadline=HARD_END - 7)
            vt = float(_max_cos(Xt))
            if vt < bestVal + 0.001:
                Xt2 = _lp_polish(Xt, seconds=2.5, deadline=HARD_END - 5)
                vt2 = float(_max_cos(Xt2))
                if vt2 < bestVal:
                    bestX, bestVal = Xt2.copy(), vt2
    remain2 = HARD_END - time.time() - 1.5
    if remain2 > 2:
        bestX = _lp_polish(bestX, seconds=min(remain2, 20.0), deadline=HARD_END - 1.0)
    bestX = _norm_rows(bestX)
    return bestX
