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
    t0 = _time.time()
    BUDGET = 172.0
    t_end = t0 + BUDGET
    n, d = 249, 6
    rng = _np.random.default_rng(0)

    # load incumbent
    bestX = None
    bestVal = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = _np.load(fn).astype(float)
            if a.shape == (n, d):
                a = a / _np.maximum(_np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
                if _np.all(_np.isfinite(a)):
                    v = _max_cos(a)
                    if v < bestVal:
                        bestVal = v
                        bestX = a.copy()
        except Exception:
            pass
    if bestX is None:
        bestX = rng.standard_normal(size=(n, d))
        bestX /= _np.linalg.norm(bestX, axis=1, keepdims=True)
        bestVal = _max_cos(bestX)

    iu0 = _np.triu_indices(n, 1)
    I0, J0 = iu0

    def make_fg(tau, creg=0.05):
        tau = float(tau)
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
            if _time.time() > t_end - 12:
                break
            fg = make_fg(float(tau))
            try:
                res = _minimize(fg, Xf, jac=True, method='L-BFGS-B',
                                options={'maxiter': int(mi), 'maxfun': int(mi * 3),
                                         'ftol': 1e-14, 'gtol': 1e-7, 'maxcor': 10})
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

    def greedy_polish(Xinit, max_steps=400, step0=2e-3):
        cur = Xinit.copy()
        cur /= _np.maximum(_np.linalg.norm(cur, axis=1, keepdims=True), 1e-12)
        curVal = _max_cos(cur)
        step = float(step0)
        for _ in range(int(max_steps)):
            if _time.time() > t_end - 6:
                break
            if step < 1e-13:
                break
            G = cur @ cur.T
            _np.fill_diagonal(G, -2.0)
            m = float(G.max())
            window = max(10.0 * step, 3e-4)
            if step < 5e-5:
                window = max(1.5e-4, 10.0 * step)
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
        return cur, curVal

    def find_holes(Xkeep, m, n_samp=22000):
        # sample random directions, score by max cos to keep set
        R = rng.standard_normal(size=(n_samp, d))
        R /= _np.maximum(_np.linalg.norm(R, axis=1, keepdims=True), 1e-12)
        # chunk to save memory
        best_scores = _np.empty(n_samp, dtype=float)
        cs = 4000
        for s in range(0, n_samp, cs):
            e = min(s + cs, n_samp)
            Gc = R[s:e] @ Xkeep.T
            best_scores[s:e] = Gc.max(axis=1)
        order = _np.argsort(best_scores)  # smallest max-cos = largest hole
        chosen = []
        for idx in order:
            if len(chosen) >= m:
                break
            cand = R[idx]
            ok = True
            for c in chosen:
                if float(cand @ c) > 0.75:
                    ok = False
                    break
            if ok:
                chosen.append(cand)
            if len(chosen) < m and idx == order[-1]:
                break
        # fallback fill
        k = 0
        while len(chosen) < m:
            chosen.append(R[order[k]])
            k += 1
        return _np.stack(chosen, axis=0)

    curX = bestX.copy()
    curVal = bestVal

    # basin hopping with topology moves
    # time for exploration: until ~ t0+135
    explore_end = t0 + 135.0
    if explore_end > t_end - 35:
        explore_end = t_end - 35
    it = 0
    temp0, temp1 = 4e-4, 1.5e-5
    while _time.time() < explore_end:
        it += 1
        elapsed = _time.time() - t0
        frac = elapsed / max(explore_end - t0, 1e-9)
        frac = min(max(frac, 0.0), 1.0)
        temp = temp0 * (temp1 / max(temp0, 1e-12)) ** frac
        u = rng.random()
        trial = None
        if u < 0.28:
            # global kick of current
            sig = float(rng.choice([0.012, 0.007, 0.02, 0.004]))
            # sometimes kick best instead
            base = curX if rng.random() < 0.7 else bestX
            trial = base + sig * rng.standard_normal(size=(n, d))
            trial /= _np.maximum(_np.linalg.norm(trial, axis=1, keepdims=True), 1e-12)
            taus = [0.0035, 0.0012]
            iters = [70, 90]
        elif u < 0.60:
            # regional reshake around random center
            base = curX if rng.random() < 0.8 else bestX
            Gc = base @ base.T
            _np.fill_diagonal(Gc, -2.0)
            c = int(rng.integers(0, n))
            k = int(rng.integers(12, 32))
            nbr = _np.argsort(-Gc[c])[:k]
            # include center
            sig = float(rng.choice([0.06, 0.10, 0.16, 0.04]))
            trial = base.copy()
            trial[nbr] += sig * rng.standard_normal(size=(k, d))
            # occasionally also jitter center a bit more
            trial[c] += 0.5 * sig * rng.standard_normal(size=(d,))
            trial /= _np.maximum(_np.linalg.norm(trial, axis=1, keepdims=True), 1e-12)
            taus = [0.003, 0.001]
            iters = [80, 100]
        else:
            # hole reinsertion of worst points
            base = curX if rng.random() < 0.8 else bestX
            G = base @ base.T
            _np.fill_diagonal(G, -2.0)
            rowmax = G.max(axis=1)
            # contact overload count within 2e-4 of global max
            gm = float(rowmax.max())
            cnt = _np.sum(G > (gm - 2e-4), axis=1)
            score = rowmax + 0.0004 * cnt  # penalize overloaded
            m = int(rng.choice([2, 3, 3, 4, 5, 1, 6]))
            m = min(m, 8)
            # pick m among top 3m worst at random
            cand_top = _np.argsort(-score)[:max(m * 3, m + 2)]
            sel = rng.choice(cand_top, size=m, replace=False)
            keep_mask = _np.ones(n, dtype=bool)
            keep_mask[sel] = False
            Xkeep = base[keep_mask]
            holes = find_holes(Xkeep, m)
            trial = base.copy()
            trial[sel] = holes + 1e-3 * rng.standard_normal(size=(m, d))
            trial /= _np.maximum(_np.linalg.norm(trial, axis=1, keepdims=True), 1e-12)
            taus = [0.004, 0.0012]
            iters = [80, 110]
        # polish trial
        try:
            Y = soft_polish(trial, taus, iters)
        except Exception:
            continue
        try:
            Y2, v2 = greedy_polish(Y, max_steps=220, step0=1.5e-3)
        except Exception:
            Y2 = Y
            v2 = _max_cos(Y2)
        # acceptance
        if v2 < bestVal - 1e-12:
            bestVal = v2
            bestX = Y2.copy()
            curX = Y2.copy()
            curVal = v2
        elif v2 < curVal:
            curX = Y2.copy()
            curVal = v2
        else:
            # Metropolis sideways
            try:
                p = _np.exp((curVal - v2) / max(temp, 1e-12))
            except Exception:
                p = 0.0
            if rng.random() < p:
                curX = Y2.copy()
                curVal = v2
        # opportunistic restart of cur to best if cur too bad
        if curVal > bestVal + 8e-4 and rng.random() < 0.3:
            curX = bestX.copy()
            curVal = bestVal

    # final intensive polish of best
    if _time.time() < t_end - 8:
        try:
            Y = soft_polish(bestX, [0.002, 0.0008, 0.0003, 0.00012], [140, 160, 200, 220])
            v = _max_cos(Y)
            if v < bestVal:
                bestVal = v
                bestX = Y.copy()
        except Exception:
            pass
    if _time.time() < t_end - 5:
        try:
            Y2, v2 = greedy_polish(bestX, max_steps=800, step0=2e-3)
            if v2 < bestVal:
                bestVal = v2
                bestX = Y2.copy()
        except Exception:
            pass
    # trust-region LP jam if time remains
    if _time.time() < t_end - 3:
        try:
            from scipy.optimize import linprog as _linprog
            from scipy.sparse import coo_matrix as _coo
            X = bestX / _np.maximum(_np.linalg.norm(bestX, axis=1, keepdims=True), 1e-12)
            best = _max_cos(X)
            radius = 5e-4
            iu = _np.triu_indices(n, 1)
            while _time.time() < t_end - 1.0 and radius > 1e-14:
                Gf = X @ X.T
                c = Gf[iu]
                m = float(c.max())
                sel = c > m - max(20 * radius, 4e-7)
                if _np.count_nonzero(sel) > 6000:
                    idx = _np.argsort(c)[-6000:]
                    msel = _np.zeros_like(sel)
                    msel[idx] = True
                    sel = msel
                I, J = iu[0][sel], iu[1][sel]
                p = len(I)
                if p == 0:
                    break
                nv = n * d + 1
                rows = _np.repeat(_np.arange(p), 2 * d + 1)
                cols = _np.concatenate([I[:, None] * d + _np.arange(d), J[:, None] * d + _np.arange(d), _np.full((p, 1), n * d)], 1).ravel()
                vals = _np.concatenate([X[J], X[I], -_np.ones((p, 1))], 1).ravel()
                A_ub = _coo((vals, (rows, cols)), shape=(p, nv)).tocsr()
                A_eq = _coo((X.ravel(), (_np.repeat(_np.arange(n), d), _np.arange(n * d))), shape=(n, nv)).tocsr()
                obj = _np.zeros(nv)
                obj[-1] = 1.0
                try:
                    res = _linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=_np.zeros(n),
                                   bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                                   options={"time_limit": max(1.0, t_end - _time.time() - 0.5)})
                except Exception:
                    radius *= 0.5
                    continue
                if res.status != 0:
                    radius *= 0.5
                    continue
                y = X + res.x[:-1].reshape(n, d)
                y /= _np.maximum(_np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
                v = _max_cos(y)
                if v < best:
                    X, best = y, v
                    radius = min(radius * 1.5, 0.02)
                    if best < bestVal:
                        bestVal = best
                        bestX = X.copy()
                else:
                    radius *= 0.5
        except Exception:
            pass

    bestX = bestX / _np.maximum(_np.linalg.norm(bestX, axis=1, keepdims=True), 1e-12)
    return bestX
