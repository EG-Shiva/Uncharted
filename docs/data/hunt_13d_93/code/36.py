import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _adam_balanced(X, steps, tau0, tau1, lr0, lr1):
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    ltau = np.log(tau1 / tau0) if tau0 != tau1 else 0.0
    llr = np.log(lr1 / lr0) if lr0 != lr1 else 0.0
    tt = 0
    for s in range(steps):
        frac = s / max(1, steps - 1)
        tau = tau0 * np.exp(ltau * frac)
        lr = lr0 * np.exp(llr * frac)
        if tau < 5e-05:
            tau = 5e-05
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        cmax = float(G.max())
        E = np.exp((G - cmax) / tau)
        np.fill_diagonal(E, 0.0)
        rs = E.sum(axis=1, keepdims=True) + 1e-12
        grad = (E @ X) / np.sqrt(rs)
        dot = (grad * X).sum(axis=1, keepdims=True)
        gt = grad - dot * X
        if not np.all(np.isfinite(gt)):
            continue
        tt += 1
        m *= 0.9
        m += 0.1 * gt
        v *= 0.999
        v += 0.001 * (gt * gt)
        mh = m / (1.0 - 0.9 ** tt)
        vh = v / (1.0 - 0.999 ** tt)
        step = lr * mh / (np.sqrt(vh) + 1e-8)
        X = X - step
        X = _norm(X)
    return X

def _topk_mean_descend(X, steps, K0, K1, lr0, lr1):
    n, d = X.shape
    iu0, iu1 = np.triu_indices(n, 1)
    for s in range(steps):
        frac = s / max(1, steps - 1)
        K = int(round(K0 + (K1 - K0) * frac))
        K = max(1, min(K, iu0.shape[0]))
        lr = lr0 * (lr1 / lr0) ** frac if lr0 != lr1 else lr0
        G = X @ X.T
        c = G[iu0, iu1]
        if K < c.shape[0]:
            idx = np.argpartition(-c, K)[:K]
        else:
            idx = np.arange(c.shape[0])
        I = iu0[idx]
        J = iu1[idx]
        grad = np.zeros_like(X)
        np.add.at(grad, I, X[J])
        np.add.at(grad, J, X[I])
        grad /= float(len(idx))
        dot = (grad * X).sum(axis=1, keepdims=True)
        gt = grad - dot * X
        rn = np.linalg.norm(gt, axis=1, keepdims=True)
        nz = (rn[:, 0] > 1e-14)
        direc = np.zeros_like(gt)
        direc[nz] = gt[nz] / rn[nz]
        X = _norm(X - lr * direc)
    return X

def _greedy_topk(X, iters, K, step0):
    best = _max_cos(X)
    step = step0
    n, d = X.shape
    iu0, iu1 = np.triu_indices(n, 1)
    for _ in range(iters):
        G = X @ X.T
        c = G[iu0, iu1]
        if K < c.shape[0]:
            idx = np.argpartition(-c, K)[:K]
        else:
            idx = np.arange(c.shape[0])
        I = iu0[idx]
        J = iu1[idx]
        grad = np.zeros_like(X)
        np.add.at(grad, I, X[J])
        np.add.at(grad, J, X[I])
        grad /= float(len(idx))
        dot = (grad * X).sum(axis=1, keepdims=True)
        gt = grad - dot * X
        rn = np.linalg.norm(gt, axis=1, keepdims=True)
        nz = (rn[:, 0] > 1e-14)
        if not np.any(nz):
            break
        direc = np.zeros_like(gt)
        direc[nz] = gt[nz] / rn[nz]
        Xt = _norm(X - step * direc)
        vv = _max_cos(Xt)
        if vv < best:
            X = Xt
            best = vv
            step = min(step * 1.2, 0.006)
        else:
            step *= 0.6
            if step < 1e-12:
                break
    return X

def _refine_hole(v, Y, iters=80):
    v = v / (np.linalg.norm(v) + 1e-18)
    lr = 0.05
    tau = 0.006
    for _ in range(iters):
        dots = Y @ v
        m0 = float(dots.max())
        w = np.exp((dots - m0) / tau)
        s = float(w.sum())
        if s <= 0 or not np.isfinite(s):
            break
        w /= s
        g = w @ Y
        gt = g - float(g @ v) * v
        nr = float(np.linalg.norm(gt))
        if nr < 1e-12 or not np.isfinite(nr):
            break
        gt /= nr
        v2 = v - lr * gt
        n2 = float(np.linalg.norm(v2))
        if n2 < 1e-12:
            lr *= 0.5
            continue
        v2 /= n2
        m2 = float((Y @ v2).max())
        if m2 < m0:
            v = v2
            lr = min(lr * 1.06, 0.08)
        else:
            lr *= 0.5
            if lr < 1e-6:
                break
    return v / (np.linalg.norm(v) + 1e-18)

def solve():
    t0 = time.time()
    BUDGET = 172.0
    rng = np.random.default_rng(7)
    np.random.seed(7)
    n, d = 93, 13
    best_x = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (93, 13):
                a = _norm(a)
                va = _max_cos(a)
                if va < best_val:
                    best_val = va
                    best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        best_x = _norm(rng.standard_normal((n, d)))
        best_val = _max_cos(best_x)

    sigmas = [0.006, 0.015, 0.032, 0.060, 0.105, 0.022, 0.045, 0.080]
    trial = 0
    # hopping phase
    while time.time() - t0 < 128.0 and trial < 2000:
        sigma = sigmas[trial % len(sigmas)]
        Xtry = best_x + sigma * rng.standard_normal((n, d))
        Xtry = _norm(Xtry)
        if trial % 4 == 2:
            c = int(rng.integers(0, n))
            Gc = Xtry @ Xtry[c]
            nn = np.argsort(-Gc)[:7]
            Xtry[nn] += 0.5 * sigma * rng.standard_normal((len(nn), d))
            Xtry = _norm(Xtry)
        # void re-insertion for larger kicks
        if sigma > 0.012:
            krep = 1 if sigma < 0.06 else 2
            for _ in range(krep):
                G = Xtry @ Xtry.T
                np.fill_diagonal(G, -2.0)
                per = G.max(axis=1)
                iw = int(np.argmax(per))
                others = np.delete(Xtry, iw, axis=0)
                C = rng.standard_normal((8000, d))
                C = _norm(C)
                dots = C @ others.T
                worst = dots.max(axis=1)
                ib = int(np.argmin(worst))
                vv = _refine_hole(C[ib], others, iters=80)
                Xtry[iw] = vv
        # two-stage balanced soft-max equalization
        if trial % 2 == 0:
            Xtry = _adam_balanced(Xtry, 450, 0.008, 0.001, 0.005, 0.0008)
            Xtry = _adam_balanced(Xtry, 450, 0.001, 0.00025, 0.0022, 0.0005)
        else:
            Xtry = _adam_balanced(Xtry, 550, 0.012, 0.0008, 0.006, 0.0006)
            Xtry = _adam_balanced(Xtry, 350, 0.0008, 0.0002, 0.0018, 0.0004)
        # persistent top-K mean to collapse cloud
        Xtry = _topk_mean_descend(Xtry, 180, 500, 60, 0.004, 0.0008)
        v = _max_cos(Xtry)
        if v < best_val - 1e-12 and np.all(np.isfinite(Xtry)):
            # quick greedy squeeze
            Xg = _greedy_topk(Xtry, 220, 80, 0.0025)
            vg = _max_cos(Xg)
            if vg < v and np.all(np.isfinite(Xg)):
                Xtry, v = Xg, vg
            if v < best_val:
                best_x, best_val = Xtry, v
        elif v < best_val + 0.0015:
            # opportunistic squeeze of near-miss
            try:
                Xg = _greedy_topk(Xtry, 120, 60, 0.0015)
                vg = _max_cos(Xg)
                if vg < best_val and np.all(np.isfinite(Xg)):
                    best_x, best_val = Xg, vg
            except Exception:
                pass
        trial += 1
        if time.time() - t0 > BUDGET - 45:
            break

    # final intensive equalization on best
    remain = BUDGET - (time.time() - t0) - 1.0
    if remain > 6:
        try:
            Xp = _adam_balanced(best_x.copy(), 700, 0.0008, 0.0002, 0.0015, 0.0003)
            vp = _max_cos(Xp)
            if vp < best_val:
                best_x, best_val = Xp, vp
        except Exception:
            pass
    remain = BUDGET - (time.time() - t0) - 0.8
    if remain > 4:
        try:
            Xp = _topk_mean_descend(best_x.copy(), 250, 400, 40, 0.002, 0.0004)
            Xp = _greedy_topk(Xp, 400, 100, 0.002)
            vp = _max_cos(Xp)
            if vp < best_val:
                best_x, best_val = Xp, vp
        except Exception:
            pass
    # micro single-point hill climb for last 1e-7
    deadline = t0 + BUDGET - 0.4
    try:
        X = best_x
        best = best_val
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        step = 2e-4
        it = 0
        while time.time() < deadline and it < 25000:
            it += 1
            if it % 60 == 1:
                G = X @ X.T
                np.fill_diagonal(G, -2.0)
                per = G.max(axis=1)
                order = np.argsort(-per)
                k = int(rng.integers(0, min(22, n)))
                i = int(order[k])
            else:
                i = int(rng.integers(0, n))
            r = rng.standard_normal(d)
            r -= float(r @ X[i]) * X[i]
            rn = float(np.linalg.norm(r))
            if rn < 1e-12:
                continue
            r /= rn
            improved = False
            for sgn in (1.0, -1.0):
                vv = X[i] + sgn * step * r
                nv = float(np.linalg.norm(vv))
                if nv < 1e-12:
                    continue
                vv /= nv
                old = X[i].copy()
                X[i] = vv
                # fast eval: full gram (tiny)
                D = X @ X.T
                np.fill_diagonal(D, -2.0)
                cur = float(D.max())
                if cur < best:
                    best = cur
                    best_val = cur
                    best_x = X.copy()
                    step = min(step * 1.12, 5e-4)
                    improved = True
                    break
                else:
                    X[i] = old
            if not improved and it % 35 == 0:
                step *= 0.92
                if step < 2e-6:
                    step = 2e-6
    except Exception:
        pass
    return _norm(best_x)
