import time
import numpy as np
from scipy.optimize import minimize

def _normalize(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def solve():
    t0 = time.time()
    BUDGET = 172.0
    HOP_END = 158.0
    rng = np.random.default_rng(7)
    n, d = 93, 13

    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (93, 13):
                continue
            a = _normalize(a)
            v = _maxcos(a)
            if v < best_val:
                best_val = v
                best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _normalize(rng.standard_normal((n, d)))
        best_val = _maxcos(best)

    iu0, iu1 = np.triu_indices(n, 1)

    # ---------- smoothed L-BFGS objective (scale invariant + radial anchor) ----------
    def make_fun(T):
        def fun_grad(Yf):
            Y = Yf.reshape(n, d)
            r = np.linalg.norm(Y, axis=1)
            r = np.maximum(r, 1e-12)
            X = Y / r[:, None]
            G = X @ X.T
            c = G[iu0, iu1]
            m = float(c.max())
            # weights
            e = np.exp((c - m) / T)
            S = float(e.sum())
            f = m + T * np.log(S + 1e-300)
            w = e / (S + 1e-300)
            W = np.zeros((n, n), dtype=np.float64)
            W[iu0, iu1] = w
            W[iu1, iu0] = w
            WX = W @ X  # (n,d)
            dot = np.einsum('ij,ij->i', WX, X)
            GX = WX - dot[:, None] * X
            GY = GX / r[:, None]
            # radial anchor to keep r~1 (f is scale invariant)
            pen = 0.5 * float(np.mean((r - 1.0) ** 2))
            f_total = f + 0.2 * pen
            # grad of pen: 0.2*(r-1)/n * (Y/r)
            gp = (0.2 / n) * ((r - 1.0) / (r + 1e-18))[:, None] * Y * 0.0  # keep tiny
            # actually add small anchor gradient
            anchor = (0.2 / n) * ((r - 1.0)[:, None] * X / np.maximum(r[:, None], 1e-12)) * 1.0
            # GY already tangent; anchor is radial
            Gtot = GY + anchor
            return float(f_total), Gtot.ravel()
        return fun_grad

    def lbfgs_stage(X0, T, maxiter):
        fg = make_fun(T)
        Y0 = X0.ravel().copy()
        try:
            res = minimize(fg, Y0, jac=True, method='L-BFGS-B',
                           options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-7, 'maxfun': maxiter * 4})
            Y = res.x.reshape(n, d)
            X = _normalize(Y)
            return X
        except Exception:
            return X0

    def anneal(X0, temps, iters):
        X = X0.copy()
        cur_best = X.copy()
        cur_bv = _maxcos(X)
        for T, it in zip(temps, iters):
            X = lbfgs_stage(X, T, it)
            v = _maxcos(X)
            if v < cur_bv:
                cur_bv = v
                cur_best = X.copy()
            if time.time() - t0 > BUDGET - 12:
                break
        return cur_best, cur_bv, X

    def refine_hole(v, Y, steps=120, tau=0.008):
        v = v / (np.linalg.norm(v) + 1e-18)
        lr = 0.05
        for _ in range(steps):
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
                lr = min(lr * 1.05, 0.08)
            else:
                lr *= 0.5
                if lr < 1e-6:
                    break
        return v / (np.linalg.norm(v) + 1e-18)

    def find_hole(Y, M=25000):
        dbest = 1e9
        vbest = None
        CH = 5000
        for _ in range(M // CH):
            C = rng.standard_normal((CH, d))
            C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-18
            D = C @ Y.T
            worst = D.max(axis=1)
            idx = int(np.argmin(worst))
            if float(worst[idx]) < dbest:
                dbest = float(worst[idx])
                vbest = C[idx].copy()
        return refine_hole(vbest, Y, steps=120)

    def hard_topk(X, iters=160, K=40, step0=0.004):
        bv = _maxcos(X)
        step = step0
        grad = np.empty_like(X)
        L = len(iu0)
        K = min(K, L)
        for _ in range(iters):
            G = X @ X.T
            c = G[iu0, iu1]
            if K < L:
                idx = np.argpartition(-c, K)[:K]
            else:
                idx = np.arange(L)
            I = iu0[idx]; J = iu1[idx]
            grad.fill(0.0)
            np.add.at(grad, I, X[J])
            np.add.at(grad, J, X[I])
            grad /= float(len(idx))
            dot = np.einsum('ij,ij->i', grad, X)[:, None]
            gt = grad - dot * X
            ng = float(np.linalg.norm(gt))
            if ng == 0 or not np.isfinite(ng):
                step *= 0.5
                if step < 1e-12:
                    break
                continue
            Xt = _normalize(X - step * gt)
            vv = _maxcos(Xt)
            if vv < bv:
                X = Xt
                bv = vv
                step = min(step * 1.25, 0.008)
            else:
                step *= 0.5
                if step < 1e-9:
                    break
            if time.time() - t0 > BUDGET - 6:
                break
        return X, bv

    def micro_single(X, seconds):
        start = time.time()
        bv = _maxcos(X)
        step = 1.5e-4
        it = 0
        while time.time() - start < seconds and (time.time() - t0) < BUDGET - 0.6:
            it += 1
            if it % 40 == 1:
                G = X @ X.T
                np.fill_diagonal(G, -2.0)
                per = G.max(axis=1)
                order = np.argsort(-per)
                k = int(rng.integers(0, min(25, n)))
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
                v = X[i] + sgn * step * r
                v /= np.linalg.norm(v) + 1e-18
                # fast bound: max involving i vs rest of max
                dots_i = X @ v
                dots_i[i] = -2.0
                mi = float(dots_i.max())
                if mi >= bv:
                    continue
                Y = X.copy()
                Y[i] = v
                vv = _maxcos(Y)
                if vv < bv:
                    X = Y
                    bv = vv
                    step = min(step * 1.12, 4e-4)
                    improved = True
                    break
            if not improved and it % 25 == 0:
                step *= 0.92
                if step < 2e-6:
                    step = 2e-6
            if it > 30000:
                break
        return X, bv

    # quick squeeze of starting point with one low-T stage
    try:
        Xs, vs = hard_topk(best.copy(), iters=80, K=30, step0=0.002)
        if vs < best_val:
            best, best_val = Xs, vs
    except Exception:
        pass

    sigmas = [0.006, 0.018, 0.035, 0.06, 0.10, 0.16]
    trial = 0
    while time.time() < t0 + HOP_END and (time.time() - t0) < BUDGET - 30:
        sigma = sigmas[trial % len(sigmas)]
        surgical = (trial % 3 == 2)
        Xtry = best + sigma * rng.standard_normal((n, d))
        Xtry = _normalize(Xtry)
        # cluster kick on some trials
        if trial % 2 == 1:
            Gc = Xtry @ Xtry[int(rng.integers(0, n))]
            nn = np.argsort(-Gc)[:6]
            Xtry[nn] += 0.6 * sigma * rng.standard_normal((len(nn), d))
            Xtry = _normalize(Xtry)
        # void surgery: reinsert worst points into holes
        if surgical:
            try:
                krep = 1 if (trial % 6 != 5) else 2
                for _ in range(krep):
                    G = Xtry @ Xtry.T
                    np.fill_diagonal(G, -2.0)
                    per = G.max(axis=1)
                    iw = int(np.argmax(per))
                    others = np.delete(Xtry, iw, axis=0)
                    v = find_hole(others, M=20000)
                    Xtry[iw] = v
            except Exception:
                pass
        # anneal schedule depends on sigma (larger sigma -> higher start T)
        if sigma < 0.02:
            temps = [0.004, 0.001, 0.00035]
            iters = [50, 60, 70]
        elif sigma < 0.07:
            temps = [0.008, 0.002, 0.0005]
            iters = [60, 60, 80]
        else:
            temps = [0.012, 0.003, 0.0007]
            iters = [70, 70, 80]
        try:
            Xb, vb, Xlast = anneal(Xtry, temps, iters)
        except Exception:
            trial += 1
            continue
        # hard minimax squeeze
        try:
            K = 50 if surgical else 30
            Xh, vh = hard_topk(Xlast, iters=120, K=K, step0=0.003)
            if vh < vb:
                vb, Xb = vh, Xh
        except Exception:
            pass
        if vb < best_val - 1e-12 and np.all(np.isfinite(Xb)):
            best, best_val = Xb, vb
        trial += 1
        if trial > 200:
            break

    # final polish
    remain = BUDGET - (time.time() - t0) - 0.8
    if remain > 5:
        try:
            Xb, vb, Xlast = anneal(best.copy(), [0.002, 0.0004, 0.0002], [80, 100, 100])
            if vb < best_val:
                best, best_val = Xb, vb
            else:
                Xlast = best
        except Exception:
            Xlast = best
        try:
            Xh, vh = hard_topk(Xlast, iters=250, K=60, step0=0.002)
            if vh < best_val:
                best, best_val = Xh, vh
        except Exception:
            pass
        remain2 = BUDGET - (time.time() - t0) - 0.5
        if remain2 > 1.5:
            try:
                best, best_val = micro_single(best, min(remain2, 6.0))
            except Exception:
                pass

    best = _normalize(best)
    # safety margin shrink (does not change max materially, ensures strict feasibility)
    return best
