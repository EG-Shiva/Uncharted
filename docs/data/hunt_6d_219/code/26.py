import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _rowmax_G(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    rm = np.max(G, axis=1)
    return rm, float(np.max(rm)), G

def _lp_polish(x, seconds, deadline, iu):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    iuI, iuJ = iu
    best = _max_cos(x)
    bestX = x.copy()
    radius = 1.1e-3
    start = time.time()
    nv = n * d + 1
    while time.time() - start < seconds and time.time() < deadline:
        if radius < 1e-14:
            break
        G = x @ x.T
        c = G[iuI, iuJ]
        m = float(np.max(c))
        thr = m - max(20.0 * radius, 1e-9)
        sel = c > thr
        I = iuI[sel]; J = iuJ[sel]; csel = c[sel]
        p = len(I)
        if p == 0:
            break
        if p > 3500:
            k = 3500
            part = np.argpartition(-csel, k - 1)[:k]
            I, J, csel = I[part], J[part], csel[part]
            p = k
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            bestX = y.copy()
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return bestX, best

def _lbfgs_smooth(x0, iu, beta=600.0, maxiter=100):
    n, d = x0.shape
    iu0, iu1 = iu
    y0 = x0.ravel().copy()
    def fun(y):
        Y = y.reshape(n, d)
        rn = np.linalg.norm(Y, axis=1, keepdims=True)
        rn = np.maximum(rn, 1e-12)
        X = Y / rn
        G = X @ X.T
        g = G[iu0, iu1]
        m = float(np.max(g))
        e = np.exp(beta * (g - m))
        S = float(np.sum(e))
        loss = m + np.log(S) / beta
        w = e / S
        W = np.zeros((n, n), dtype=float)
        W[iu0, iu1] = w
        W = W + W.T
        GX = W @ X
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GY = (GX - dot * X) / rn
        return float(loss), GY.ravel()
    try:
        res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-8, "maxfun": 20000})
        Y = res.x.reshape(n, d)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        return Y
    except Exception:
        return x0

def _anneal(X, steps, T0, T1, lr0, momentum=0.0):
    cur = X.copy()
    n, d = cur.shape
    best = _max_cos(cur)
    bestX = cur.copy()
    if steps <= 1:
        return bestX, best
    dec = (T1 / T0) ** (1.0 / (steps - 1)) if steps > 1 else 1.0
    T = T0
    vel = np.zeros_like(cur)
    for it in range(steps):
        G = cur @ cur.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        if m < best:
            best = m
            bestX = cur.copy()
        # soft-max weights
        S = (G - m) / T
        # clip to avoid overflow, zero far pairs
        W = np.exp(S, out=np.empty_like(G))
        W[G < m - 8.0 * T] = 0.0
        np.fill_diagonal(W, 0.0)
        Grad = W @ cur
        proj = np.sum(Grad * cur, axis=1, keepdims=True)
        Gt = Grad - proj * cur
        gn = float(np.max(np.linalg.norm(Gt, axis=1)))
        if not np.isfinite(gn) or gn < 1e-14:
            T *= dec
            continue
        step = Gt * (lr0 / gn)
        if momentum > 0:
            vel = momentum * vel + step
            step = vel
        cur = cur - step
        cur /= np.linalg.norm(cur, axis=1, keepdims=True)
        T *= dec
    v = _max_cos(cur)
    if v < best:
        best = v
        bestX = cur.copy()
    return bestX, best

def _refine_single(v, others, iters=140, tau=0.002):
    v = v / np.linalg.norm(v)
    best = float(np.max(others @ v))
    bestv = v.copy()
    lr = 0.04
    for it in range(iters):
        dots = others @ v
        m = float(np.max(dots))
        w = np.exp((dots - m) / tau)
        w[dots < m - 6 * tau] = 0.0
        g = w @ others
        gt = g - np.dot(g, v) * v
        nn = float(np.linalg.norm(gt))
        if nn < 1e-14 or not np.isfinite(nn):
            break
        st = lr / (1.0 + it * 0.04) / nn
        vn = v - st * gt
        vn /= np.linalg.norm(vn)
        mn = float(np.max(others @ vn))
        if mn < m:
            v = vn
            if mn < best:
                best = mn
                bestv = vn.copy()
        else:
            lr *= 0.7
            if lr < 1e-5:
                break
    return bestv, best

def _void_reinsert(Y, idx_list, rng, n_cand=22000):
    n, d = Y.shape
    for idx in idx_list:
        others = np.delete(Y, idx, axis=0)
        cand = rng.standard_normal((n_cand, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        sc = cand @ others.T
        cm = np.max(sc, axis=1)
        j = int(np.argmin(cm))
        bv, _ = _refine_single(cand[j], others, iters=140)
        Y[idx] = bv
    return Y

def solve():
    t0 = time.time()
    DEAD = t0 + 172.0
    HOP_END = t0 + 136.0
    rng = np.random.default_rng(20261103)
    n, d = 219, 6
    best_val = 1e9
    bestX = None
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.load(fn)
            v = np.asarray(v, dtype=np.float64)
            if v.shape == (n, d):
                v = v / np.linalg.norm(v, axis=1, keepdims=True)
                c = _max_cos(v)
                if c < best_val:
                    best_val = c
                    bestX = v.copy()
        except Exception:
            pass
    if bestX is None:
        bestX = rng.standard_normal((n, d))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        best_val = _max_cos(bestX)
    iu = np.triu_indices(n, 1)
    # initial squeeze: LP + LBFGS + LP
    try:
        Xp, vp = _lp_polish(bestX, seconds=3.0, deadline=HOP_END, iu=iu)
        if vp < best_val:
            best_val, bestX = vp, Xp
        Y = _lbfgs_smooth(bestX, iu, beta=500.0, maxiter=80)
        vy = _max_cos(Y)
        if vy < best_val:
            best_val, bestX = vy, Y
            Xp2, vp2 = _lp_polish(bestX, seconds=2.0, deadline=HOP_END, iu=iu)
            if vp2 < best_val:
                best_val, bestX = vp2, Xp2
    except Exception:
        pass
    curX = bestX.copy()
    cur_val = best_val
    meta_T = 0.0009
    trial = 0
    sigmas = [0.006, 0.015, 0.035, 0.08]
    while time.time() < HOP_END and time.time() < DEAD - 34:
        trial += 1
        # base: mostly wander from cur, sometimes restart from best
        if trial % 7 == 0:
            base = bestX
            base_val = best_val
        else:
            base = curX
            base_val = cur_val
        Xtry = base.copy()
        try:
            rm, mcur, G = _rowmax_G(base)
            Ccnt = np.sum(G > mcur - 2e-4, axis=1)
            score = rm + 0.0004 * Ccnt + rng.random(n) * 0.0002
            order = np.argsort(-score)
            ratt = np.where(rm < mcur - 5e-4)[0]
            mode = int(rng.integers(0, 5))
            if mode == 0:
                # single worst void kill
                victims = [int(order[0])]
                if len(ratt) > 0 and rng.random() < 0.7:
                    victims = [int(ratt[0])]
                _void_reinsert(Xtry, victims, rng, n_cand=25000)
                Xa, va = _anneal(Xtry, steps=650, T0=0.0032, T1=0.00028, lr0=0.024, momentum=0.15)
            elif mode == 1:
                # multi kill of jammed points
                nv = int(rng.integers(2, 5))
                if trial % 11 == 0:
                    nv = 6
                victims = order[:nv].tolist()
                # ensure rattler included if exists
                for r in ratt[:1]:
                    if int(r) not in victims:
                        victims[-1] = int(r)
                _void_reinsert(Xtry, victims, rng, n_cand=22000)
                Xa, va = _anneal(Xtry, steps=700, T0=0.0042, T1=0.0003, lr0=0.026, momentum=0.2)
            elif mode == 2:
                # jammed cluster kick: worst + its contact neighbours
                i0 = int(order[0])
                thr = mcur - 0.004
                neigh = np.where(G[i0] > thr)[0]
                if len(neigh) < 6:
                    # expand to top dots
                    neigh = np.argsort(-G[i0])[:12]
                # add 1-hop
                if len(neigh) > 25:
                    neigh = neigh[np.argsort(-G[i0][neigh])[:25]]
                k = len(neigh)
                sigma = sigmas[trial % len(sigmas)]
                # correlated kick: common direction + individual
                common = rng.standard_normal(d) * sigma * 0.7
                for ii in neigh:
                    Xtry[ii] += common + rng.standard_normal(d) * sigma * 0.7
                Xtry[neigh] /= np.linalg.norm(Xtry[neigh], axis=1, keepdims=True)
                Xa, va = _anneal(Xtry, steps=600, T0=0.005, T1=0.00035, lr0=0.03, momentum=0.1)
            elif mode == 3:
                # random subset kick biased to bad points
                k = int(rng.integers(12, 55))
                hw = min(k // 2 + 2, n)
                sel = np.zeros(n, dtype=bool)
                sel[order[:hw]] = True
                extra = rng.choice(n, size=k, replace=False)
                for e in extra:
                    sel[e] = True
                    if np.sum(sel) >= k:
                        break
                sigma = 0.025 if rng.random() < 0.5 else 0.06
                noise = rng.standard_normal((n, d)) * sigma
                # project noise to tangent for stability
                noise -= np.sum(noise * Xtry, axis=1, keepdims=True) * Xtry
                Xtry[sel] += noise[sel]
                Xtry[sel] /= np.linalg.norm(Xtry[sel], axis=1, keepdims=True)
                Xa, va = _anneal(Xtry, steps=550, T0=0.006, T1=0.00035, lr0=0.03, momentum=0.0)
            else:
                # rattle-focused: reinsert rattlers to voids + light kick to worst
                if len(ratt) > 0:
                    _void_reinsert(Xtry, [int(r) for r in ratt[:2]], rng, n_cand=25000)
                k = 8 + int(rng.integers(0, 12))
                sel = order[:k]
                sigma = 0.012
                Xtry[sel] += rng.standard_normal((k, d)) * sigma
                Xtry[sel] /= np.linalg.norm(Xtry[sel], axis=1, keepdims=True)
                Xa, va = _anneal(Xtry, steps=600, T0=0.0035, T1=0.00025, lr0=0.022, momentum=0.2)
            # screening decisions
            improved_global = va < best_val
            promising = (va < best_val + 0.0015) or (va < cur_val + 0.0008)
            vp = None
            Xp = None
            if improved_global:
                # sharpen: LBFGS then short LP
                try:
                    Yl = _lbfgs_smooth(Xa, iu, beta=700.0, maxiter=60)
                    vl = _max_cos(Yl)
                    if vl < va:
                        Xa, va = Yl, vl
                except Exception:
                    pass
                Xp, vp = _lp_polish(Xa, seconds=2.0, deadline=DEAD - 28, iu=iu)
                if vp < best_val:
                    best_val, bestX = vp, Xp.copy()
                    cur_val, curX = vp, Xp.copy()
                elif va < best_val:
                    best_val, bestX = va, Xa.copy()
                    cur_val, curX = va, Xa.copy()
                else:
                    # accept to cur if close
                    if vp < cur_val:
                        cur_val, curX = vp, Xp.copy()
            elif promising and (trial % 2 == 0):
                Xp, vp = _lp_polish(Xa, seconds=0.9, deadline=DEAD - 28, iu=iu)
                if vp < best_val:
                    best_val, bestX = vp, Xp.copy()
                    cur_val, curX = vp, Xp.copy()
                else:
                    # Metropolis on wandering state
                    if vp < cur_val or rng.random() < np.exp((cur_val - vp) / meta_T):
                        cur_val, curX = vp, Xp.copy()
                    elif va < cur_val and rng.random() < 0.3:
                        cur_val, curX = va, Xa.copy()
            else:
                # pure Metropolis on annealed value without LP
                if va < cur_val or rng.random() < np.exp((cur_val - va) / (meta_T * 1.5 + 1e-12)):
                    # keep wandering but don't waste LP; store annealed
                    # only store best of anneal trajectory (Xa already best)
                    cur_val, curX = va, Xa.copy()
            # decay meta temperature slowly
            meta_T *= 0.9995
            if meta_T < 0.00015:
                meta_T = 0.00015
            # occasional reset of wanderer to best if too far
            if cur_val > best_val + 0.004 and rng.random() < 0.4:
                cur_val, curX = best_val, bestX.copy()
        except Exception:
            continue
    # final deep polish of global best
    try:
        left = DEAD - time.time()
        if left > 5:
            # alternate LBFGS (increasing beta) + LP
            for beta, mi in [(600.0, 90), (1500.0, 90)]:
                if time.time() > DEAD - 4:
                    break
                try:
                    Y = _lbfgs_smooth(bestX, iu, beta=beta, maxiter=mi)
                    vy = _max_cos(Y)
                    if vy < best_val:
                        best_val, bestX = vy, Y
                except Exception:
                    pass
                try:
                    Xp, vp = _lp_polish(bestX, seconds=min(8.0, max(1.0, (DEAD - time.time() - 2) / 3)), deadline=DEAD - 0.5, iu=iu)
                    if vp < best_val:
                        best_val, bestX = vp, Xp
                except Exception:
                    pass
            # final LP squeeze with remaining time
            try:
                rem = DEAD - time.time() - 1.0
                if rem > 2:
                    Xp, vp = _lp_polish(bestX, seconds=rem, deadline=DEAD - 0.5, iu=iu)
                    if vp < best_val:
                        best_val, bestX = vp, Xp
            except Exception:
                pass
    except Exception:
        pass
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX
