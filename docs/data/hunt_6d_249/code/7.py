import time
import numpy as np

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
    _np.random.seed(7)
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
    NPAIR = len(I0)

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
                                options={'maxiter': int(mi), 'maxfun': int(mi * 3),
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
            fn = _np.linalg.norm(Ft) + 1e-12
            # scale step to keep update bounded
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
                if step < 1e-7 and v >= curVal:
                    # small random tangent kick to escape flat
                    pass
                if step < 5e-12:
                    break
        return cur, curVal

    def lp_polish(Xinit, seconds):
        X = Xinit.copy()
        X /= _np.maximum(_np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        best = _max_cos(X)
        bestXl = X.copy()
        radius = 3e-3
        start = _time.time()
        iu = _np.triu_indices(n, 1)
        while _time.time() - start < seconds and radius > 1e-14:
            if _time.time() - t0 > BUDGET - 2:
                break
            Gfull = X @ X.T
            c = Gfull[iu]
            m = float(c.max())
            sel = c > m - max(20.0 * radius, 2e-7)
            # cap active set
            p0 = int(np.sum(sel))
            if p0 > 6000:
                # keep largest 6000
                idx = _np.argpartition(c, -6000)[-6000:]
                sel = _np.zeros_like(sel, dtype=bool)
                sel[idx] = True
                p0 = 6000
            if p0 < 10:
                sel = c > m - 1e-6
                p0 = int(np.sum(sel))
            I = iu[0][sel]; J = iu[1][sel]; p = p0
            csel = c[sel]
            try:
                rows = _np.repeat(_np.arange(p), 2 * d + 1)
                colsA = _np.concatenate([I[:, None] * d + _np.arange(d),
                                         J[:, None] * d + _np.arange(d),
                                         _np.full((p, 1), n * d)], 1).ravel()
                vals = _np.concatenate([X[J], X[I], -_np.ones((p, 1))], 1).ravel()
                A_ub = _coo((vals, (rows, colsA)), shape=(p, n * d + 1)).tocsr()
                # equality: X_i . dx_i = 0
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
                radius = min(radius * 1.5, 0.02)
            else:
                radius *= 0.5
            # track best even if current wandered? keep X as improved-or-same?
            # if not improved, keep X unchanged (trust region reject)
        return bestXl, best

    def per_point_max(X):
        G = X @ X.T
        _np.fill_diagonal(G, -2.0)
        return G.max(axis=1), G

    def largest_holes(X, k, nsample=20000):
        # random candidates, pick k farthest from set
        R = _np.random.randn(nsample, d)
        R /= _np.maximum(_np.linalg.norm(R, axis=1, keepdims=True), 1e-12)
        S = R @ X.T  # nsample x n
        m = S.max(axis=1)
        idx = _np.argpartition(m, k)[:k] if k < nsample else _np.arange(nsample)
        # sort by best (smallest max)
        idx = idx[_np.argsort(m[idx])]
        return R[idx]

    # initial polish
    Xp = soft_polish(bestX, [0.012, 0.006, 0.003, 0.0012], [90, 100, 110, 130])
    vp = _max_cos(Xp)
    if vp < bestVal:
        bestX, bestVal = Xp, vp
    Xe, ve = greedy_polish(bestX, min(t0 + 14, t0 + BUDGET - 10), step0=2e-3)
    if ve < bestVal:
        bestX, bestVal = Xe, ve
    # early LP squeeze (short)
    if _time.time() - t0 < BUDGET - 20:
        Xl, vl = lp_polish(bestX, 6.0)
        if vl < bestVal:
            bestX, bestVal = Xl, vl

    sigmas = [0.03, 0.015, 0.007, 0.05, 0.02, 0.009, 0.012]
    tausL = [0.007, 0.0028, 0.0011]
    itersL = [55, 70, 90]
    trial = 0
    while _time.time() - t0 < BUDGET - 22 and trial < 60:
        mode = trial % 5
        trial += 1
        sig_base = sigmas[(trial - 1) % len(sigmas)] * (0.9 ** ((trial - 1) // 7))
        Xt = None
        if mode == 0 or mode == 4:
            # global gaussian
            Xt = bestX + sig_base * _np.random.randn(n, d)
            Xt /= _np.maximum(_np.linalg.norm(Xt, axis=1, keepdims=True), 1e-12)
        elif mode == 1:
            # regional reshake around worst point
            ppm, Gcur = per_point_max(bestX)
            iw = int(_np.argmax(ppm))
            dists = Gcur[iw].copy()
            # nearest = largest cosine
            neigh = _np.argsort(-dists)[:16]
            Xt = bestX.copy()
            Xt[neigh] += (sig_base * 2.2) * _np.random.randn(len(neigh), d)
            # small global jitter
            Xt += (sig_base * 0.25) * _np.random.randn(n, d)
            Xt /= _np.maximum(_np.linalg.norm(Xt, axis=1, keepdims=True), 1e-12)
        elif mode == 2:
            # hole reinsertion: replace 1-2 worst points by holes
            ppm, _ = per_point_max(bestX)
            nw = 2 if (trial % 2 == 0) else 1
            worst = _np.argsort(-ppm)[:nw]
            holes = largest_holes(bestX, nw + 2, nsample=15000)
            Xt = bestX.copy()
            for k2, wi in enumerate(worst):
                Xt[wi] = holes[k2 % len(holes)] + 0.005 * _np.random.randn(d)
            # jitter neighbours slightly
            Xt /= _np.maximum(_np.linalg.norm(Xt, axis=1, keepdims=True), 1e-12)
        else:
            # worst-pair split: push max pair apart
            Gc = bestX @ bestX.T
            _np.fill_diagonal(Gc, -2.0)
            # find top pair
            flat = _np.argmax(Gc)
            ii, jj = flat // n, flat % n
            Xt = bestX.copy()
            direction = Xt[ii] - Xt[jj]
            direction /= max(_np.linalg.norm(direction), 1e-12)
            amp = sig_base * 1.5
            Xt[ii] += amp * direction + sig_base * 0.5 * _np.random.randn(d)
            Xt[jj] -= amp * direction + sig_base * 0.5 * _np.random.randn(d)
            # also jitter a few random points
            ri = _np.random.choice(n, size=6, replace=False)
            Xt[ri] += sig_base * 0.6 * _np.random.randn(6, d)
            Xt /= _np.maximum(_np.linalg.norm(Xt, axis=1, keepdims=True), 1e-12)

        Xq = soft_polish(Xt, tausL, itersL)
        vq = _max_cos(Xq)
        # quick greedy 0.7s
        Xq2, vq2 = greedy_polish(Xq, min(_time.time() + 0.7, t0 + BUDGET - 20), step0=1.5e-3)
        if vq2 < vq:
            Xq, vq = Xq2, vq2
        if vq < bestVal - 1e-12:
            # intensify
            Xd = soft_polish(Xq, [0.002, 0.0009], [90, 120])
            vd = _max_cos(Xd)
            if vd < vq:
                Xq, vq = Xd, vd
            Xg, vg = greedy_polish(Xq, min(_time.time() + 0.8, t0 + BUDGET - 20), step0=1e-3)
            if vg < vq:
                Xq, vq = Xg, vg
            # short LP confirm
            Xl2, vl2 = lp_polish(Xq, 1.5)
            if vl2 < vq:
                Xq, vq = Xl2, vl2
            if vq < bestVal:
                bestX, bestVal = Xq, vq
        if _time.time() - t0 > BUDGET - 23:
            break

    # final intensification
    if _time.time() - t0 < BUDGET - 3:
        Xf = soft_polish(bestX, [0.003, 0.0012, 0.00055], [100, 130, 160])
        vf = _max_cos(Xf)
        if vf < bestVal:
            bestX, bestVal = Xf, vf
    if _time.time() - t0 < BUDGET - 2:
        remain = max(1.0, (t0 + BUDGET - 1.0) - _time.time())
        # cap LP to remain
        XlF, vlF = lp_polish(bestX, min(remain, 18.0))
        if vlF < bestVal:
            bestX, bestVal = XlF, vlF
    if _time.time() - t0 < BUDGET - 1:
        XeF, veF = greedy_polish(bestX, t0 + BUDGET - 0.5, step0=5e-4)
        if veF < bestVal:
            bestX, bestVal = XeF, veF

    bestX = bestX / _np.maximum(_np.linalg.norm(bestX, axis=1, keepdims=True), 1e-12)
    bestX = _np.nan_to_num(bestX)
    bestX = bestX / _np.maximum(_np.linalg.norm(bestX, axis=1, keepdims=True), 1e-12)
    return bestX
