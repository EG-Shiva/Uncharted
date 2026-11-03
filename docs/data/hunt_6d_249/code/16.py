import numpy as np
import time

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def solve():
    import time as _time
    import numpy as _np
    from scipy.optimize import minimize as _minimize
    from scipy.optimize import linprog as _linprog
    from scipy.sparse import coo_matrix as _coo
    t0 = _time.time()
    BUDGET = 172.0
    _np.random.seed(12345)
    n, d = 249, 6

    try:
        xb = _np.load("best.npy")
    except Exception:
        xb = None
    try:
        xr = _np.load("record.npy")
    except Exception:
        xr = None
    cands = []
    for a in (xb, xr):
        if a is None:
            continue
        try:
            a = _np.asarray(a, dtype=float)
            if a.shape == (n, d):
                an = a / _np.maximum(_np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
                if _np.all(_np.isfinite(an)):
                    cands.append((_max_cos(an), an))
        except Exception:
            pass
    if len(cands) == 0:
        X0 = _np.random.randn(n, d)
        X0 /= _np.linalg.norm(X0, axis=1, keepdims=True)
        bestX = X0
        bestVal = _max_cos(bestX)
    else:
        cands.sort(key=lambda t: t[0])
        bestVal, bestX = cands[0]
        bestX = bestX.copy()

    iu0 = _np.triu_indices(n, 1)
    I0, J0 = iu0

    def make_fg(tau, creg=0.05):
        def fg(xf):
            X = xf.reshape(n, d)
            r = _np.sqrt(_np.sum(X * X, axis=1))
            r = _np.maximum(r, 1e-12)
            Xn = X / r[:, None]
            G = Xn @ Xn.T
            g = G[I0, J0]
            gmax = _np.max(g)
            e = _np.exp((g - gmax) / tau)
            S = _np.sum(e)
            Lsoft = gmax + tau * _np.log(S)
            w = e / S
            W = _np.zeros((n, n), dtype=float)
            W[I0, J0] = w
            W = W + W.T
            A = W @ Xn
            proj = _np.sum(A * Xn, axis=1)
            Grad0 = (A - proj[:, None] * Xn) / r[:, None]
            diff = r - 1.0
            L = Lsoft + creg * _np.sum(diff * diff)
            Grad = Grad0 + (2.0 * creg * diff)[:, None] * Xn
            return float(L), Grad.ravel()
        return fg

    def soft_polish(Xinit, taus, iters):
        Xf = Xinit.ravel().copy()
        for tau, mi in zip(taus, iters):
            if _time.time() - t0 > BUDGET - 10:
                break
            fg = make_fg(float(tau))
            try:
                res = _minimize(fg, Xf, jac=True, method='L-BFGS-B',
                                options={'maxiter': int(mi), 'maxfun': int(mi * 4),
                                         'ftol': 1e-14, 'gtol': 1e-9})
                Xf = res.x
            except Exception:
                break
            Xm = Xf.reshape(n, d)
            rn = _np.linalg.norm(Xm, axis=1, keepdims=True)
            rn = _np.maximum(rn, 1e-12)
            Xm = Xm / rn
            Xf = Xm.ravel()
        Xout = Xf.reshape(n, d)
        Xout /= _np.maximum(_np.linalg.norm(Xout, axis=1, keepdims=True), 1e-12)
        return Xout

    def greedy_polish(Xinit, deadline, step0=2e-3):
        cur = Xinit.copy()
        cur /= _np.maximum(_np.linalg.norm(cur, axis=1, keepdims=True), 1e-12)
        curVal = _max_cos(cur)
        step = float(step0)
        while _time.time() < deadline and step > 1e-13:
            if _time.time() - t0 > BUDGET - 3:
                break
            G = cur @ cur.T
            _np.fill_diagonal(G, -2.0)
            m = float(G.max())
            window = max(10.0 * step, 4e-4)
            if step < 5e-5:
                window = max(2e-4, 10.0 * step)
            mask = G > (m - window)
            F = mask.astype(float) @ cur
            dot = _np.sum(F * cur, axis=1)
            Ft = F - dot[:, None] * cur
            trial = cur - step * Ft
            rn = _np.linalg.norm(trial, axis=1, keepdims=True)
            rn = _np.maximum(rn, 1e-12)
            trial /= rn
            v = _max_cos(trial)
            if v < curVal:
                cur = trial
                curVal = v
                step = min(step * 1.25, 0.01)
            else:
                step *= 0.5
                if step < 5e-12:
                    break
        return cur, curVal

    def lp_polish(Xinit, seconds):
        X = Xinit.copy()
        X /= _np.maximum(_np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        best = _max_cos(X)
        bestXl = X.copy()
        radius = 2e-3
        start = _time.time()
        iu = _np.triu_indices(n, 1)
        while _time.time() - start < seconds and radius > 1e-14:
            if _time.time() - t0 > BUDGET - 2:
                break
            Gfull = X @ X.T
            c = Gfull[iu]
            m = float(c.max())
            sel = c > m - max(20.0 * radius, 2e-7)
            p0 = int(_np.sum(sel))
            if p0 > 5500:
                idx = _np.argpartition(c, -5500)[-5500:]
                sel = _np.zeros_like(sel, dtype=bool)
                sel[idx] = True
                p0 = 5500
            if p0 < 10:
                sel = c > m - 1e-6
                p0 = int(_np.sum(sel))
            I = iu[0][sel]; J = iu[1][sel]; p = p0
            csel = c[sel]
            try:
                rows = _np.repeat(_np.arange(p), 2 * d + 1)
                colsA = _np.concatenate([I[:, None] * d + _np.arange(d),
                                         J[:, None] * d + _np.arange(d),
                                         _np.full((p, 1), n * d)], 1).ravel()
                vals = _np.concatenate([X[J], X[I], -_np.ones((p, 1))], 1).ravel()
                A_ub = _coo((vals, (rows, colsA)), shape=(p, n * d + 1)).tocsr()
                erows = _np.repeat(_np.arange(n), d)
                ecols = _np.arange(n * d)
                A_eq = _coo((X.ravel(), (erows, ecols)), shape=(n, n * d + 1)).tocsr()
                obj = _np.zeros(n * d + 1); obj[-1] = 1.0
                bounds = [(-radius, radius)] * (n * d) + [(None, None)]
                res = _linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq,
                               b_eq=_np.zeros(n), bounds=bounds, method="highs")
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            dx = res.x[:-1].reshape(n, d)
            y = X + dx
            y /= _np.maximum(_np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
            v = _max_cos(y)
            if v < best:
                X, best = y, v
                bestXl = y.copy()
                radius = min(radius * 1.5, 0.015)
            else:
                radius *= 0.5
        return bestXl, best

    def per_point_max(X):
        G = X @ X.T
        _np.fill_diagonal(G, -2.0)
        return G.max(axis=1), G

    def largest_holes(X, k, nsample=25000):
        R = _np.random.randn(nsample, d)
        R /= _np.maximum(_np.linalg.norm(R, axis=1, keepdims=True), 1e-12)
        S = R @ X.T
        m = S.max(axis=1)
        if k < nsample:
            idx = _np.argpartition(m, k)[:k]
        else:
            idx = _np.arange(nsample)
        idx = idx[_np.argsort(m[idx])]
        return R[idx]

    # initial squeeze of incumbent
    Xp = soft_polish(bestX, [0.010, 0.005, 0.0022, 0.0010], [80, 90, 100, 120])
    vp = _max_cos(Xp)
    if vp < bestVal:
        bestX, bestVal = Xp, vp
    Xe, ve = greedy_polish(bestX, min(t0 + 10, t0 + BUDGET - 15), step0=2e-3)
    if ve < bestVal:
        bestX, bestVal = Xe, ve
    if _time.time() - t0 < BUDGET - 25:
        Xl, vl = lp_polish(bestX, 5.0)
        if vl < bestVal:
            bestX, bestVal = Xl, vl

    taus_cheap = [0.006, 0.0022]
    iters_cheap = [45, 60]
    taus_full = [0.006, 0.0022, 0.0009]
    iters_full = [55, 75, 95]

    sigmas = [0.055, 0.028, 0.014, 0.007, 0.035, 0.018, 0.009]
    trial = 0
    while _time.time() - t0 < BUDGET - 24 and trial < 80:
        trial += 1
        mode = trial % 6
        sig_base = sigmas[(trial - 1) % len(sigmas)] * (0.92 ** ((trial - 1) // 8))
        Xt = None
        if mode == 0 or mode == 5:
            Xt = bestX + sig_base * _np.random.randn(n, d)
            Xt /= _np.maximum(_np.linalg.norm(Xt, axis=1, keepdims=True), 1e-12)
        elif mode == 1:
            ppm, Gcur = per_point_max(bestX)
            # overloaded hubs: top by contacts near max
            Gtmp = bestX @ bestX.T
            _np.fill_diagonal(Gtmp, -2.0)
            m0 = float(Gtmp.max())
            contacts = _np.sum(Gtmp > m0 - 1e-9, axis=1)
            iw = int(_np.argmax(contacts + 0.1 * (ppm - ppm.min())))
            dists = Gcur[iw].copy()
            neigh = _np.argsort(-dists)[:14]
            Xt = bestX.copy()
            Xt[neigh] += (sig_base * 2.4) * _np.random.randn(len(neigh), d)
            Xt += (sig_base * 0.22) * _np.random.randn(n, d)
            Xt /= _np.maximum(_np.linalg.norm(Xt, axis=1, keepdims=True), 1e-12)
        elif mode == 2:
            ppm, _ = per_point_max(bestX)
            nw = 3 if (trial % 3 == 0) else (2 if (trial % 2 == 0) else 1)
            worst = _np.argsort(-ppm)[:nw]
            holes = largest_holes(bestX, nw + 2, nsample=22000)
            Xt = bestX.copy()
            for k2, wi in enumerate(worst):
                Xt[wi] = holes[k2 % len(holes)] + 0.004 * _np.random.randn(d)
            # jitter its nearest neighbours a bit
            Gc = Xt @ Xt.T
            for wi in worst:
                nn = _np.argsort(-Gc[wi])[:6]
                Xt[nn] += (sig_base * 0.5) * _np.random.randn(len(nn), d)
            Xt /= _np.maximum(_np.linalg.norm(Xt, axis=1, keepdims=True), 1e-12)
        elif mode == 3:
            Gc = bestX @ bestX.T
            _np.fill_diagonal(Gc, -2.0)
            flat = _np.argmax(Gc)
            ii, jj = flat // n, flat % n
            Xt = bestX.copy()
            direction = Xt[ii] - Xt[jj]
            direction /= max(_np.linalg.norm(direction), 1e-12)
            amp = sig_base * 1.6
            Xt[ii] += amp * direction + sig_base * 0.5 * _np.random.randn(d)
            Xt[jj] -= amp * direction + sig_base * 0.5 * _np.random.randn(d)
            ri = _np.random.choice(n, size=8, replace=False)
            Xt[ri] += sig_base * 0.6 * _np.random.randn(8, d)
            Xt /= _np.maximum(_np.linalg.norm(Xt, axis=1, keepdims=True), 1e-12)
        else:
            # random-subspace avalanche: rotate small random subset
            ksub = 18
            sub = _np.random.choice(n, size=ksub, replace=False)
            Xt = bestX.copy()
            # random tangent perturbation with larger amp on subset
            Xt[sub] += (sig_base * 2.0) * _np.random.randn(ksub, d)
            Xt += (sig_base * 0.15) * _np.random.randn(n, d)
            Xt /= _np.maximum(_np.linalg.norm(Xt, axis=1, keepdims=True), 1e-12)

        Xq = soft_polish(Xt, taus_cheap, iters_cheap)
        vq = _max_cos(Xq)
        if vq > bestVal + 0.004:
            continue
        Xq2, vq2 = greedy_polish(Xq, min(_time.time() + 0.55, t0 + BUDGET - 22), step0=1.4e-3)
        if vq2 < vq:
            Xq, vq = Xq2, vq2
        if vq < bestVal - 1e-12:
            Xd = soft_polish(Xq, [0.0018, 0.0008], [85, 115])
            vd = _max_cos(Xd)
            if vd < vq:
                Xq, vq = Xd, vd
            Xg, vg = greedy_polish(Xq, min(_time.time() + 0.7, t0 + BUDGET - 22), step0=9e-4)
            if vg < vq:
                Xq, vq = Xg, vg
            Xl2, vl2 = lp_polish(Xq, 1.4)
            if vl2 < vq:
                Xq, vq = Xl2, vl2
            if vq < bestVal:
                bestX, bestVal = Xq, vq
        elif vq < bestVal + 0.0006 and (trial % 4 == 0):
            # second chance intensify for near-miss basins
            Xd = soft_polish(Xq, taus_full, iters_full)
            vd = _max_cos(Xd)
            Xg, vg = greedy_polish(Xd, min(_time.time() + 0.5, t0 + BUDGET - 22), step0=1e-3)
            vv = vg if vg < vd else vd
            XX = Xg if vg < vd else Xd
            if vv < bestVal:
                Xl2, vl2 = lp_polish(XX, 1.2)
                if vl2 < vv:
                    XX, vv = Xl2, vl2
                if vv < bestVal:
                    bestX, bestVal = XX, vv
        if _time.time() - t0 > BUDGET - 25:
            break

    if _time.time() - t0 < BUDGET - 3:
        Xf = soft_polish(bestX, [0.0026, 0.0011, 0.0005], [95, 125, 155])
        vf = _max_cos(Xf)
        if vf < bestVal:
            bestX, bestVal = Xf, vf
    if _time.time() - t0 < BUDGET - 2:
        remain = max(1.0, (t0 + BUDGET - 1.0) - _time.time())
        XlF, vlF = lp_polish(bestX, min(remain, 16.0))
        if vlF < bestVal:
            bestX, bestVal = XlF, vlF
        # final tiny greedy to lock
        XgF, vgF = greedy_polish(bestX, min(_time.time() + 1.5, t0 + BUDGET - 0.5), step0=6e-4)
        if vgF < bestVal:
            bestX, bestVal = XgF, vgF

    bestX /= _np.maximum(_np.linalg.norm(bestX, axis=1, keepdims=True), 1e-12)
    return bestX
