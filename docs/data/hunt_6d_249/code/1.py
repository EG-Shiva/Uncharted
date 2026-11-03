import time
import numpy as np

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def solve():
    import time as _time
    import numpy as _np
    from scipy.optimize import minimize as _minimize
    t0 = _time.time()
    BUDGET = 172.0

    _np.random.seed(0)
    n, d = 249, 6
    # load start
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
        if a is not None:
            try:
                a = _np.asarray(a, dtype=float)
                if a.shape == (n, d):
                    an = a / _np.linalg.norm(a, axis=1, keepdims=True)
                    cands.append(( _max_cos(an), an))
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

    def make_fg(tau, creg=0.1):
        def fg(xf):
            X = xf.reshape(n, d)
            r = _np.sqrt(_np.sum(X*X, axis=1))
            r = _np.maximum(r, 1e-12)
            Xn = X / r[:, None]
            G = Xn @ Xn.T
            g = G[I0, J0]
            gmax = _np.max(g)
            # exp
            e = _np.exp((g - gmax) / tau)
            S = _np.sum(e)
            Lsoft = gmax + tau * _np.log(S)
            w = e / S
            # dense W
            W = _np.zeros((n, n), dtype=float)
            W[I0, J0] = w
            W = W + W.T
            A = W @ Xn
            proj = _np.sum(A * Xn, axis=1)
            Grad0 = (A - proj[:, None] * Xn) / r[:, None]
            # reg
            diff = r - 1.0
            L = Lsoft + creg * _np.sum(diff*diff)
            Grad = Grad0 + (2.0*creg*diff)[:, None] * Xn
            return float(L), Grad.ravel()
        return fg

    def soft_polish(Xinit, taus, iters):
        Xf = Xinit.ravel().copy()
        # keep norms ~1
        for tau, mi in zip(taus, iters):
            if _time.time() - t0 > BUDGET - 8:
                break
            fg = make_fg(tau)
            try:
                res = _minimize(fg, Xf, jac=True, method='L-BFGS-B',
                                options={'maxiter': int(mi), 'maxfun': int(mi*3),
                                         'ftol': 1e-14, 'gtol': 1e-9})
                Xf = res.x
            except Exception:
                break
            # renormalize to unit rows to remove radial drift
            Xm = Xf.reshape(n, d)
            rn = _np.linalg.norm(Xm, axis=1, keepdims=True)
            rn = _np.maximum(rn, 1e-12)
            Xm = Xm / rn
            Xf = Xm.ravel()
        Xout = Xf.reshape(n, d)
        Xout /= _np.linalg.norm(Xout, axis=1, keepdims=True)
        return Xout

    def exact_polish(Xinit, deadline):
        cur = Xinit.copy()
        cur /= _np.linalg.norm(cur, axis=1, keepdims=True)
        curVal = _max_cos(cur)
        step = 2e-3
        # preallocate
        while _time.time() < deadline and step > 1e-13:
            if _time.time() - t0 > BUDGET - 2:
                break
            G = cur @ cur.T
            _np.fill_diagonal(G, -2.0)
            m = float(G.max())
            window = max(10.0*step, 5e-4)
            # shrink window as we converge to focus
            if step < 5e-5:
                window = max(2e-4, 10.0*step)
            mask = G > (m - window)
            # build force: F = mask @ cur  (mask symmetric, diag False)
            # mask bool -> float via @ (will cast)
            F = mask.astype(float) @ cur
            # tangent projection
            dot = _np.sum(F*cur, axis=1)
            Ft = F - dot[:, None]*cur
            # normalize step scale by typical force norm
            fn = _np.linalg.norm(Ft) + 1e-12
            # trial
            trial = cur - step * Ft
            # guard against zero
            rn = _np.linalg.norm(trial, axis=1, keepdims=True)
            rn = _np.maximum(rn, 1e-12)
            trial /= rn
            v = _max_cos(trial)
            if v < curVal:
                cur = trial
                curVal = v
                step = min(step*1.25, 0.01)
            else:
                step *= 0.5
                # occasional random tangent kick if stuck with large step?
                if step < 1e-7:
                    break
        return cur, curVal

    # initial deep polish of incumbent
    taus0 = [0.015, 0.008, 0.004, 0.002, 0.001]
    iters0 = [120, 150, 150, 150, 200]
    Xp = soft_polish(bestX, taus0, iters0)
    vp = _max_cos(Xp)
    if vp < bestVal:
        bestX, bestVal = Xp, vp
    # short exact polish
    Xe, ve = exact_polish(bestX, t0 + 12 if _time.time() < t0+12 else _time.time()+2)
    if ve < bestVal:
        bestX, bestVal = Xe, ve

    # basin hopping
    sigmas = [0.025, 0.012, 0.006, 0.04, 0.018, 0.008]
    trial = 0
    # light schedule for hops
    tausL = [0.008, 0.003, 0.0012]
    itersL = [90, 110, 130]
    while _time.time() - t0 < BUDGET - 14 and trial < 40:
        sig = sigmas[trial % len(sigmas)] * (0.85 ** (trial // len(sigmas)))
        trial += 1
        Xt = bestX + sig * _np.random.randn(n, d)
        Xt /= _np.linalg.norm(Xt, axis=1, keepdims=True)
        Xq = soft_polish(Xt, tausL, itersL)
        # exact short, 1.2 sec each
        dl = min(_time.time()+1.5, t0+BUDGET-5)
        Xq2, vq2 = exact_polish(Xq, dl)
        # also consider Xq itself
        vq = _max_cos(Xq)
        if vq < vq2:
            Xq2, vq2 = Xq, vq
        if vq2 < bestVal - 1e-12:
            bestX, bestVal = Xq2, vq2
            # intensify around new best immediately
            Xp2 = soft_polish(bestX, [0.002, 0.0009], [150, 200])
            vpp = _max_cos(Xp2)
            if vpp < bestVal:
                bestX, bestVal = Xp2, vpp
            Xe2, ve2 = exact_polish(bestX, min(_time.time()+2.0, t0+BUDGET-5))
            if ve2 < bestVal:
                bestX, bestVal = Xe2, ve2
        # timeout check
        if _time.time() - t0 > BUDGET - 6:
            break

    # final intensification with remaining time
    if _time.time() - t0 < BUDGET - 3:
        Xf = soft_polish(bestX, [0.004, 0.0015, 0.0007], [120, 150, 200])
        vf = _max_cos(Xf)
        if vf < bestVal:
            bestX, bestVal = Xf, vf
    if _time.time() - t0 < BUDGET - 2:
        XeF, veF = exact_polish(bestX, t0 + BUDGET - 1)
        if veF < bestVal:
            bestX, bestVal = XeF, veF

    bestX = bestX / _np.linalg.norm(bestX, axis=1, keepdims=True)
    # safety: remove NaN
    bestX = _np.nan_to_num(bestX)
    bestX = bestX / _np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX
