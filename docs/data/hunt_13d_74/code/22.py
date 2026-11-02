import time
import numpy as np
from scipy.optimize import minimize

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _normalize(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def solve():
    t0 = time.time()
    LIM = 172.0
    rng = np.random.default_rng(0)
    n, d = 74, 13

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (74, 13):
                a = _normalize(a)
                cands.append(a)
        except Exception:
            pass
    if not cands:
        cands.append(_normalize(rng.standard_normal((74, 13))))
    vals = [_max_cos(a) for a in cands]
    bi = int(np.argmin(vals))
    best = cands[bi].copy()
    best_val = float(vals[bi])

    IU = np.triu_indices(n, 1)
    I0, J0 = IU[0].astype(np.int64), IU[1].astype(np.int64)

    def lbfgs_polish(X0, tau, maxiter):
        y0 = X0.ravel().copy()
        def fun(yf):
            Y = yf.reshape(n, d)
            r2 = (Y * Y).sum(axis=1, keepdims=True)
            r = np.sqrt(np.maximum(r2, 1e-24))
            r = np.maximum(r, 1e-12)
            X = Y / r
            G = X @ X.T
            g = G[I0, J0]
            mmax = float(g.max())
            e = np.exp((g - mmax) / tau)
            S = float(e.sum())
            if not np.isfinite(S) or S <= 0:
                return float(mmax), np.zeros_like(yf)
            F = mmax + tau * np.log(S + 1e-300)
            w = e / (S + 1e-300)
            Wx = np.zeros_like(X)
            # scatter: Wx[I0]+=w*X[J0]
            Wx[I0] += w[:, None] * X[J0]
            Wx[J0] += w[:, None] * X[I0]
            dot = (Wx * X).sum(axis=1, keepdims=True)
            gradY = (Wx - dot * X) / r
            return float(F), gradY.ravel()
        try:
            res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                           options={"maxiter": int(maxiter), "ftol": 1e-20,
                                    "gtol": 1e-8, "maxfun": int(maxiter * 6)})
            Y = res.x.reshape(n, d)
        except Exception:
            return X0
        rn = np.linalg.norm(Y, axis=1, keepdims=True)
        rn[rn == 0] = 1.0
        X = Y / rn
        if not np.all(np.isfinite(X)):
            return X0
        return _normalize(X)

    def min_norm_g(Grad, iters=150):
        # Grad: (p,N), return g = sum w grad min norm via Frank-Wolfe
        g = np.mean(Grad, axis=0)
        for _ in range(iters):
            dots = Grad @ g
            kmin = int(np.argmin(dots))
            gnorm2 = float(g @ g)
            dg = float(dots[kmin]) - gnorm2
            if dg >= -1e-16:
                break
            dk = Grad[kmin] - g
            dd = float(dk @ dk)
            if dd <= 1e-24:
                break
            gamma = -dg / dd
            if gamma > 1.0:
                gamma = 1.0
            if gamma < 1e-14:
                break
            g = g + gamma * dk
        return g

    def bundle_polish(X, deadline, K=500, lr_init=2e-5):
        cur = _max_cos(X)
        lr = float(lr_init)
        N = n * d
        outer = 0
        fails = 0
        while time.time() < deadline:
            outer += 1
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            gfull = G[I0, J0]
            # top K
            if K < len(gfull):
                part = np.argpartition(-gfull, K)[:K]
                # sort part by value descending for stability (small)
                # keep as is for speed except need values
                Ik = I0[part]; Jk = J0[part]; ck = gfull[part]
            else:
                Ik, Jk, ck = I0, J0, gfull
            p = len(ck)
            Grad = np.zeros((p, N), dtype=np.float64)
            # fill
            for k in range(p):
                i = int(Ik[k]); j = int(Jk[k]); c = float(ck[k])
                xi = X[i]; xj = X[j]
                gi = xj - c * xi
                gj = xi - c * xj
                Grad[k, i * d:(i + 1) * d] = gi
                Grad[k, j * d:(j + 1) * d] = gj
            gvec = min_norm_g(Grad, iters=150)
            ng = float(np.sqrt(gvec @ gvec))
            if not np.isfinite(ng) or ng < 1e-10:
                # stationary for this K, enlarge K or jitter lr
                if K < 900 and fails % 3 == 0:
                    K = min(900, K + 150)
                    continue
                # tiny random kick to change active set
                lr *= 0.5
                if lr < 1e-12:
                    break
                fails += 1
                if fails > 12:
                    break
                continue
            Gd = gvec.reshape(n, d)
            # backtrack
            accepted = False
            for _ in range(12):
                if time.time() >= deadline:
                    break
                Xt = X - lr * Gd
                Xt = _normalize(Xt)
                nv = _max_cos(Xt)
                if nv < cur - 1e-14:
                    X[:] = Xt
                    cur = nv
                    lr *= 1.35
                    if lr > 5e-4:
                        lr = 5e-4
                    accepted = True
                    fails = 0
                    break
                lr *= 0.5
                if lr < 1e-13:
                    break
            if not accepted:
                fails += 1
                if fails > 14:
                    # try larger active set
                    if K < 900:
                        K = min(900, K + 200)
                        fails = 0
                        lr = lr_init * 0.5
                        continue
                    break
            if outer > 5000:
                break
        return X, cur

    # Phase 1: annealed L-BFGS from current best
    Xcur = best.copy()
    # quick anneal; keep time in check
    sched = [(0.008, 80), (0.002, 100), (5e-4, 120), (1.2e-4, 150), (4e-5, 150)]
    for tau, mi in sched:
        if time.time() - t0 > 75:
            break
        Xcur = lbfgs_polish(Xcur, tau, mi)
        v = _max_cos(Xcur)
        if v < best_val:
            best_val = float(v)
            best = Xcur.copy()

    # Phase 2: bundle polish on Xcur (or best)
    # start from better of Xcur / best
    vc = _max_cos(Xcur)
    if vc < best_val + 1e-7:
        Xb = Xcur.copy()
    else:
        Xb = best.copy()
    deadline = t0 + min(LIM - 22, 118 if (time.time()-t0)<80 else LIM-22)
    # use remaining logic: give bundle decent time
    Xb, vb = bundle_polish(Xb, t0 + 118 if time.time()-t0 < 100 else t0+LIM-22)
    if vb < best_val:
        best_val = float(vb)
        best = Xb.copy()
    else:
        # keep Xb as current for hopping even if slightly worse (within 2e-7)
        if vb < best_val + 2e-7:
            Xcur = Xb.copy()
        else:
            Xcur = best.copy()

    # Phase 3: small basin hops + polish
    hop = 0
    while time.time() - t0 < LIM - 10:
        hop += 1
        remain = LIM - (time.time() - t0)
        if remain < 8:
            break
        Xtry = best.copy()
        mode = hop % 4
        if mode == 0:
            sig = 0.004
            Xtry = _normalize(Xtry + sig * rng.standard_normal(Xtry.shape))
        elif mode == 1:
            # shake jammed points harder
            G = Xtry @ Xtry.T
            np.fill_diagonal(G, -2.0)
            gmax = float(G.max())
            cntact = (G > (gmax - 5e-5)).sum(axis=1)
            order = np.argsort(-cntact)
            k = 10
            idx = order[:k]
            noise = 0.002 * rng.standard_normal(Xtry.shape)
            noise[idx] += 0.008 * rng.standard_normal((len(idx), d))
            Xtry = _normalize(Xtry + noise)
        elif mode == 2:
            sig = 0.0015
            Xtry = _normalize(Xtry + sig * rng.standard_normal(Xtry.shape))
        else:
            # farthest-hole reinsert of worst point
            G = Xtry @ Xtry.T
            np.fill_diagonal(G, -2.0)
            iw = int(np.argmax(G.max(axis=1)))
            cand = _normalize(rng.standard_normal((2500, d)))
            S = cand @ Xtry.T
            S[:, iw] = -10.0
            sc = S.max(axis=1)
            j = int(np.argmin(sc))
            Xtry[iw] = cand[j]
            Xtry = _normalize(Xtry + 0.001 * rng.standard_normal(Xtry.shape))
        # quick smooth + bundle
        if remain > 30:
            Xtry = lbfgs_polish(Xtry, 3e-4, 60)
            Xtry = lbfgs_polish(Xtry, 5e-5, 80)
        else:
            Xtry = lbfgs_polish(Xtry, 1e-4, 50)
        vq = _max_cos(Xtry)
        # short bundle
        bdead = min(time.time() + (12 if remain > 40 else 5), t0 + LIM - 6)
        Xtry, vq2 = bundle_polish(Xtry, bdead, K=450, lr_init=1.5e-5)
        if vq2 < best_val - 1e-13:
            best = Xtry.copy()
            best_val = float(vq2)
        if hop > 30:
            break

    # final squeeze
    if time.time() - t0 < LIM - 2:
        Xf = best.copy()
        Xf, vf = bundle_polish(Xf, t0 + LIM - 1.5, K=650, lr_init=8e-6)
        if vf < best_val:
            best = Xf.copy()
            best_val = float(vf)

    best = _normalize(best)
    return best
