import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _row_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return np.max(G, axis=1), G

def _fast_relax(X, iters, lr0=0.03, margin0=0.003):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    cur = X.copy()
    best_val = _max_cos(cur)
    bestX = cur.copy()
    lr = lr0
    margin = margin0
    for it in range(iters):
        G = cur @ cur.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        tau = margin * 0.2 + 1e-12
        W = np.exp((G - m) / tau)
        W[G < m - margin * 5.0] = 0.0
        np.fill_diagonal(W, 0.0)
        Grad = W @ cur
        proj = np.sum(Grad * cur, axis=1, keepdims=True)
        Gt = Grad - proj * cur
        gn = float(np.max(np.linalg.norm(Gt, axis=1)))
        if not np.isfinite(gn) or gn < 1e-14:
            margin *= 0.5
            if margin < 1e-7:
                margin = 1e-7
                lr = lr0 * 0.5
            continue
        Xn = cur - (lr / gn) * Gt
        Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
        G2 = Xn @ Xn.T
        np.fill_diagonal(G2, -2.0)
        mn = float(np.max(G2))
        if mn < m:
            cur = Xn
            if mn < best_val:
                best_val = mn
                bestX = Xn.copy()
            if it % 200 == 0 and it > 0:
                margin *= 0.85
        else:
            lr *= 0.6
            if lr < 1e-6:
                margin *= 0.7
                lr = lr0 * 0.3
                if margin < 5e-7:
                    margin = 5e-7
                    break
    return bestX, best_val

def _lp_polish(x, seconds, t0, deadline, budget_end):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    bestX = x.copy()
    radius = 1.1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    iuI, iuJ = iu0
    while time.time() - start < seconds and time.time() < deadline and time.time() - t0 < budget_end:
        if radius < 1e-14:
            break
        Gfull = x @ x.T
        c = Gfull[iuI, iuJ]
        m = float(np.max(c))
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        I = iuI[sel]; J = iuJ[sel]; csel = c[sel]
        p = len(I)
        if p == 0:
            break
        if p > 3500:
            k = 3500
            part = np.argpartition(-csel, k - 1)[:k]
            I, J, csel = I[part], J[part], csel[part]
            p = k
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
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

def _refine_void(v, others, iters=200, lr0=0.05):
    v = v / np.linalg.norm(v)
    lr = lr0
    best = float(np.max(others @ v))
    bestv = v.copy()
    for _ in range(iters):
        dots = others @ v
        m = float(np.max(dots))
        tau = 0.002
        w = np.exp((dots - m) / tau)
        w[dots < m - 0.01] = 0.0
        g = w @ others
        gt = g - np.dot(g, v) * v
        nrm = float(np.linalg.norm(gt))
        if nrm < 1e-14 or not np.isfinite(nrm):
            break
        vn = v - (lr / nrm) * gt
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

def _lbfgs_smooth(x0, maxiter=100, beta=300.0):
    n, d = x0.shape
    iu0, iu1 = np.triu_indices(n, 1)
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
                       options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-7, "maxfun": 20000})
        Y = res.x.reshape(n, d)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        return Y
    except Exception:
        return x0

def solve():
    t0 = time.time()
    BUDGET = 168.0
    HOP_END = t0 + 120.0
    FINAL_END = t0 + BUDGET - 2.0
    rng = np.random.default_rng(11)
    n, d = 219, 6
    bestX = None; best_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.load(fn)
            v = np.asarray(v, dtype=np.float64)
            if v.shape == (n, d):
                v = v / np.linalg.norm(v, axis=1, keepdims=True)
                if np.all(np.isfinite(v)):
                    c = _max_cos(v)
                    if c < best_val:
                        best_val = c; bestX = v.copy()
        except Exception:
            pass
    if bestX is None:
        bestX = rng.standard_normal((n, d))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        best_val = _max_cos(bestX)
    # initial squeeze to local optimum
    try:
        bx, bv = _lp_polish(bestX, seconds=3.0, t0=t0, deadline=HOP_END, budget_end=BUDGET - 3)
        if bv < best_val and np.all(np.isfinite(bx)):
            best_val, bestX = bv, bx
    except Exception:
        pass
    curX = bestX.copy(); cur_val = best_val
    trial = 0
    stuck = 0
    while time.time() < HOP_END:
        trial += 1
        remain_hop = HOP_END - time.time()
        if remain_hop < 0.6:
            break
        try:
            rm, Gcur = _row_max(curX)
            mcur = float(np.max(rm))
            contacts = np.sum(Gcur > mcur - 1e-9, axis=1) - 1
            # -1 because diagonal was -2 so not counted; actually count only off-diag
            # correct: diagonal -2 never > m-1e-9, so fine
            worst_order = np.argsort(-rm)
            rattlers = np.where(contacts <= 0)[0]
        except Exception:
            worst_order = np.arange(n); rattlers = np.array([], dtype=int); Gcur = curX @ curX.T
        mode = trial % 6
        Xtry = None
        iters = 600; lr0 = 0.03; margin0 = 0.003
        try:
            if mode == 0:
                k = int(rng.integers(2, 6))
                sigma = 0.09 + float(rng.random()) * 0.09
                idx = worst_order[:k]
                Xtry = curX.copy()
                Xtry[idx] += sigma * rng.standard_normal((len(idx), d))
                Xtry[idx] /= np.linalg.norm(Xtry[idx], axis=1, keepdims=True)
                iters = 700; lr0 = 0.03; margin0 = 0.0035
            elif mode == 1:
                # jammed cluster kick: worst point + nearest neighbours
                p = int(worst_order[int(rng.integers(0, 3))])
                row = Gcur[p].copy()
                row[p] = -2
                nn = np.argsort(-row)[:6]
                idx = np.unique(np.concatenate([[p], nn]))
                sigma = 0.06 + float(rng.random()) * 0.08
                Xtry = curX.copy()
                Xtry[idx] += sigma * rng.standard_normal((len(idx), d))
                Xtry[idx] /= np.linalg.norm(Xtry[idx], axis=1, keepdims=True)
                iters = 800; lr0 = 0.03; margin0 = 0.004
            elif mode == 2:
                # void re-insert rattler / worst
                if len(rattlers) > 0 and rng.random() < 0.7:
                    idx = np.array([rattlers[int(rng.integers(0, len(rattlers)))]])
                else:
                    idx = np.array([worst_order[int(rng.integers(0, 3))]])
                Y = curX.copy()
                for j in idx.tolist():
                    others = np.delete(Y, j, axis=0)
                    cand = rng.standard_normal((15000, d))
                    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                    scores = np.max(cand @ others.T, axis=1)
                    order = np.argpartition(scores, 6)[:6]
                    bv2 = 1e9; bvv = None
                    for o in order:
                        rv, rs = _refine_void(cand[o], others, iters=160, lr0=0.05)
                        if rs < bv2:
                            bv2 = rs; bvv = rv
                    Y[j] = bvv / np.linalg.norm(bvv)
                Xtry = Y
                iters = 700; lr0 = 0.03; margin0 = 0.003
            elif mode == 3:
                sigma = 0.005 + float(rng.random()) * 0.012
                Xtry = curX + sigma * rng.standard_normal((n, d))
                Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
                iters = 500; lr0 = 0.02; margin0 = 0.0025
            elif mode == 4:
                k = int(rng.integers(4, 11))
                sigma = 0.04 + float(rng.random()) * 0.07
                idx = rng.choice(n, k, replace=False)
                Xtry = curX.copy()
                Xtry[idx] += sigma * rng.standard_normal((k, d))
                Xtry[idx] /= np.linalg.norm(Xtry[idx], axis=1, keepdims=True)
                iters = 700; lr0 = 0.028; margin0 = 0.004
            else:
                # mix worst + random medium kick
                k = int(rng.integers(6, 13))
                sigma = 0.02 + float(rng.random()) * 0.05
                a = worst_order[:k // 2]
                b = rng.choice(n, k - len(a), replace=False)
                idx = np.unique(np.concatenate([a, b]))
                Xtry = curX.copy()
                Xtry[idx] += sigma * rng.standard_normal((len(idx), d))
                Xtry[idx] /= np.linalg.norm(Xtry[idx], axis=1, keepdims=True)
                iters = 750; lr0 = 0.025; margin0 = 0.0035
        except Exception:
            continue
        try:
            Xr, vr = _fast_relax(Xtry, iters=iters, lr0=lr0, margin0=margin0)
        except Exception:
            continue
        # screen
        # allow LP if within threshold of cur or best
        thresh = min(cur_val + 0.002, best_val + 0.0015)
        if vr > thresh and vr > best_val:
            # metropolis on unpolished? just maybe move cur if very close
            if vr < cur_val and vr < best_val + 0.0008:
                curX, cur_val = Xr, vr
                stuck = 0
            else:
                stuck += 1
            # occasional reset from best if stuck
            if stuck > 25:
                try:
                    sigma = 0.02
                    curX = bestX + sigma * rng.standard_normal((n, d))
                    curX /= np.linalg.norm(curX, axis=1, keepdims=True)
                    cur_val = _max_cos(curX)
                except Exception:
                    pass
                stuck = 0
            continue
        lp_sec = min(2.0, max(0.6, remain_hop / 9.0))
        if vr < best_val:
            lp_sec = min(3.0, max(1.0, remain_hop / 6.0))
        try:
            Xp, vp = _lp_polish(Xr, seconds=lp_sec, t0=t0, deadline=HOP_END, budget_end=BUDGET - 3)
        except Exception:
            continue
        improved_best = False
        if vp < best_val - 1e-12:
            best_val, bestX = vp, Xp.copy()
            improved_best = True
        elif vr < best_val - 1e-12:
            best_val, bestX = vr, Xr.copy()
            improved_best = True
            vp, Xp = vr, Xr
        # L-BFGS sharpen if very promising but LP stalled
        if (not improved_best) and vp < best_val + 2e-05 and time.time() < HOP_END - 8:
            try:
                Xs = _lbfgs_smooth(Xr, maxiter=60, beta=350.0)
                vs = _max_cos(Xs)
                if vs < vp:
                    Xp, vp = Xs, vs
                    if vs < best_val - 1e-12:
                        best_val, bestX = vs, Xs.copy()
                        improved_best = True
            except Exception:
                pass
        # update wandering current with Metropolis
        cand_val = vp
        cand_X = Xp
        if cand_val < cur_val - 1e-12:
            curX, cur_val = cand_X.copy(), cand_val
            stuck = 0
        else:
            # temperature decaying
            prog = (time.time() - t0) / BUDGET
            T = 0.0004 * (1.0 - prog) + 0.00003
            if rng.random() < np.exp(-(cand_val - cur_val) / max(T, 1e-9)):
                curX, cur_val = cand_X.copy(), cand_val
                stuck = 0
            else:
                stuck += 1
                if improved_best:
                    # jump current to best region occasionally
                    curX, cur_val = bestX.copy(), best_val
                    stuck = 0
        if stuck > 30:
            try:
                sigma = 0.03 + float(rng.random()) * 0.03
                curX = bestX + sigma * rng.standard_normal((n, d))
                curX /= np.linalg.norm(curX, axis=1, keepdims=True)
                cur_val = _max_cos(curX)
            except Exception:
                pass
            stuck = 0
    # final intensive polish on best
    try:
        if time.time() < FINAL_END - 4:
            Xr, vr = _fast_relax(bestX, iters=1200, lr0=0.02, margin0=0.002)
            if vr < best_val:
                best_val, bestX = vr, Xr
    except Exception:
        pass
    try:
        if time.time() < FINAL_END - 3:
            Xs = _lbfgs_smooth(bestX, maxiter=150, beta=400.0)
            vs = _max_cos(Xs)
            if vs < best_val and np.all(np.isfinite(Xs)):
                best_val, bestX = vs, Xs
    except Exception:
        pass
    try:
        remain = FINAL_END - time.time()
        if remain > 2:
            Xp, vp = _lp_polish(bestX, seconds=remain, t0=t0, deadline=FINAL_END, budget_end=1e18)
            if vp < best_val and np.all(np.isfinite(Xp)):
                best_val, bestX = vp, Xp
    except Exception:
        pass
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    bad = ~np.all(np.isfinite(bestX), axis=1)
    if np.any(bad):
        brng = np.random.default_rng(0)
        for i in np.where(bad)[0]:
            v = brng.standard_normal(6); v /= np.linalg.norm(v); bestX[i] = v
    return bestX
