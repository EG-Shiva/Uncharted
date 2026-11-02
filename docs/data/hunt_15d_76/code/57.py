import time
import numpy as np

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def solve():
    import time
    import numpy as np
    t0 = time.time()
    deadline = t0 + 172.0

    def normed(a):
        a = np.asarray(a, dtype=float)
        a = a / np.linalg.norm(a, axis=1, keepdims=True)
        return a

    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    fb = None
    fbv = 1e9
    if xb is not None and xb.shape == (76, 15):
        try:
            xb = normed(xb)
            fb = xb.copy()
            fbv = _maxcos(fb)
        except Exception:
            pass
    if xr is not None and xr.shape == (76, 15):
        try:
            xr = normed(xr)
            v = _maxcos(xr)
            if v < fbv:
                fb = xr.copy()
                fbv = v
        except Exception:
            pass
    if fb is None:
        rng0 = np.random.default_rng(0)
        fb = normed(rng0.normal(size=(76, 15)))
        fbv = _maxcos(fb)

    n, d = 76, 15
    ONES = np.ones(15, dtype=float) / np.sqrt(15.0)
    iu0, iu1 = np.triu_indices(76, 1)
    ar = np.arange(15)
    ROLL = (ar[None, :] - ar[:, None]) % 15
    ADD = (ar[None, :] + ar[:, None]) % 15

    from scipy.optimize import minimize

    def expand_Gn(Gn):
        X = np.empty((76, 15), dtype=float)
        for k in range(5):
            X[k*15:(k+1)*15] = Gn[k][ROLL]
        X[75] = ONES
        return X

    # symmetric LSE fun with analytic grad, beta fixed per stage
    def make_fun(beta):
        def fun(flat):
            Pm = flat.reshape(5, 15)
            norms = np.linalg.norm(Pm, axis=1)
            ns = np.maximum(norms, 1e-12)
            Gn = Pm / ns[:, None]
            X = expand_Gn(Gn)
            G = X @ X.T
            s = G[iu0, iu1]
            m = float(s.max())
            E = np.exp(beta * (s - m))
            S = float(E.sum())
            if not np.isfinite(S) or S <= 0:
                return 1e6, np.zeros_like(flat)
            f = m + np.log(S) / beta
            W = E / S
            Wm = np.zeros((76, 76), dtype=float)
            Wm[iu0, iu1] = W
            Wm = Wm + Wm.T
            GX = Wm @ X  # 76x15
            GG = np.zeros((5, 15), dtype=float)
            for k in range(5):
                base = k * 15
                acc = np.zeros(15, dtype=float)
                for tt in range(15):
                    acc += GX[base + tt][ADD[tt]]
                GG[k] = acc
            dot = np.sum(GG * Gn, axis=1)
            GP = (GG - dot[:, None] * Gn) / ns[:, None]
            # mild penalty to avoid collapse to 1
            if m > 0.92:
                pen = 5.0 * (m - 0.92) ** 2
                f = f + pen
            gflat = GP.ravel()
            # clip for stability
            gflat = np.clip(gflat, -10.0, 10.0)
            return float(f), gflat.astype(float)
        return fun

    rng = np.random.default_rng(7)
    sym_cands = []  # list of (true_max, X)
    sym_deadline = t0 + 60.0
    n_restart = 6
    betas_seq = [200.0, 400.0]
    for r in range(n_restart):
        if time.time() > sym_deadline - 2.0:
            break
        P0 = rng.normal(size=(5, 15))
        # normalize rows to avoid scale issues
        P0 = P0 / np.linalg.norm(P0, axis=1, keepdims=True)
        P0 = P0.ravel()
        cur = P0
        # two beta stages, short L-BFGS-B
        for bi, beta in enumerate(betas_seq):
            if time.time() > sym_deadline - 1.0:
                break
            fun = make_fun(beta)
            try:
                res = minimize(fun, cur, method='L-BFGS-B', jac=True,
                               options={'maxiter': 120, 'maxfun': 400, 'ftol': 1e-12, 'gtol': 1e-7})
                cur = np.asarray(res.x, dtype=float)
            except Exception:
                break
            if time.time() > sym_deadline - 0.5:
                break
        try:
            Pm = cur.reshape(5, 15)
            ns = np.maximum(np.linalg.norm(Pm, axis=1, keepdims=True), 1e-12)
            Gn = Pm / ns
            X = expand_Gn(Gn)
            X = _norm(X)
            v = _maxcos(X)
            sym_cands.append((v, X.copy()))
        except Exception:
            continue

    # sort symmetric candidates
    sym_cands.sort(key=lambda z: z[0])
    # tangent Adam release
    def adam_release(X0, steps=1500, beta=300.0, lr=0.02, seed=0):
        X = X0.copy()
        m1 = np.zeros_like(X)
        m2 = np.zeros_like(X)
        b1, b2, eps = 0.9, 0.999, 1e-8
        for k in range(steps):
            G = X @ X.T
            s = G[iu0, iu1]
            m = float(s.max())
            E = np.exp(beta * (s - m))
            S = float(E.sum())
            if not np.isfinite(S) or S <= 0:
                break
            W = E / S
            Wm = np.zeros((76, 76), dtype=float)
            Wm[iu0, iu1] = W
            Wm = Wm + Wm.T
            GX = Wm @ X
            dot = np.sum(GX * X, axis=1, keepdims=True)
            Gt = GX - dot * X
            m1 = b1 * m1 + (1 - b1) * Gt
            m2 = b2 * m2 + (1 - b2) * (Gt * Gt)
            bc1 = 1 - b1 ** (k + 1)
            bc2 = 1 - b2 ** (k + 1)
            mh = m1 / bc1
            vh = m2 / bc2
            step = lr * mh / (np.sqrt(vh) + eps)
            # trust cap
            X = X - step
            X = _norm(X)
        return X

    bestX = fb.copy()
    bestV = float(fbv)

    # release top candidates (at most 3) with small asymmetric noise
    rel_deadline = t0 + 150.0
    nrel = min(len(sym_cands), 3)
    if nrel == 0:
        # create random symmetric fallback to still try release path
        nrel = 0
    rng2 = np.random.default_rng(123)
    for ci in range(nrel):
        if time.time() > rel_deadline - 3.0:
            break
        _, Xs = sym_cands[ci]
        # two noise levels, keep better
        for nl, st in [(0.002, 1500), (0.008, 800)]:
            if time.time() > rel_deadline - 2.0:
                break
            Xn = Xs + nl * rng2.normal(size=(76, 15))
            # keep pole close to ones? allow move (fully released)
            Xn = _norm(Xn)
            Xr = adam_release(Xn, steps=st, beta=320.0, lr=0.02)
            v = _maxcos(Xr)
            if v < bestV:
                bestV = float(v)
                bestX = Xr.copy()
        # also try direct release without noise (will stay symmetric, likely worse, skip if no time)
        if time.time() > rel_deadline - 2.0:
            break

    # final 15s capped LP polish on current best released (or fb if no improvement yet, polish a released cand)
    try:
        lp_budget = 15.0
        lp_start = time.time()
        lp_deadline = min(lp_start + lp_budget, deadline - 8.0)
        # choose base for LP: bestX (could be fb). Prefer best released sym cand if bestX is fb
        base = bestX.copy()
        # if bestX == fb and we have sym cands, polish the best released attempt instead? use base as is.
        from scipy.optimize import linprog
        delta = 0.004
        for it in range(6):
            if time.time() > lp_deadline - 1.0:
                break
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            cur_max = float(G.max())
            # select near-max pairs
            thr = cur_max - 0.012
            pairs = np.argwhere(G > thr)
            pairs = pairs[pairs[:, 0] < pairs[:, 1]]
            if pairs.shape[0] > 420:
                # keep tightest 420
                vals = G[pairs[:, 0], pairs[:, 1]]
                idx = np.argsort(-vals)[:420]
                pairs = pairs[idx]
            if pairs.shape[0] < 10:
                break
            P = pairs.shape[0]
            Nvar = 76 * 15 + 1
            # Build A_ub: P x Nvar (dense, ~420x1141 ~ 480k entries, ok)
            try:
                A = np.zeros((P, Nvar), dtype=float)
                b = np.zeros(P, dtype=float)
                for pi in range(P):
                    i, j = int(pairs[pi, 0]), int(pairs[pi, 1])
                    A[pi, i*15:(i+1)*15] = base[j]
                    A[pi, j*15:(j+1)*15] = base[i]
                    A[pi, Nvar-1] = -1.0
                    b[pi] = -float(G[i, j])
                # equalities: <Xi, dxi>=0
                Aeq = np.zeros((76, Nvar), dtype=float)
                beq = np.zeros(76, dtype=float)
                for i in range(76):
                    Aeq[i, i*15:(i+1)*15] = base[i]
                c = np.zeros(Nvar, dtype=float)
                c[Nvar-1] = 1.0
                bounds = [(-delta, delta)] * (76*15) + [(cur_max - 0.03, cur_max)]
                res = linprog(c, A_ub=A, b_ub=b, A_eq=Aeq, b_eq=beq, bounds=bounds, method='highs',
                              options={'presolve': True})
            except Exception:
                break
            if not hasattr(res, 'success') or not res.success:
                delta *= 0.5
                if delta < 5e-4:
                    break
                continue
            try:
                dx = np.asarray(res.x[:76*15], dtype=float).reshape(76, 15)
                # project to tangent (remove normal comp)
                dot = np.sum(dx * base, axis=1, keepdims=True)
                dx = dx - dot * base
                # trust cap
                nr = np.linalg.norm(dx, axis=1, keepdims=True)
                mx = float(nr.max())
                if mx > delta * 3:
                    dx = dx * (delta * 3 / mx)
                cand = _norm(base + dx)
                v = _maxcos(cand)
                if v < _maxcos(base) - 1e-12:
                    base = cand
                    delta = min(delta * 1.2, 0.006)
                    if v < bestV - 1e-12:
                        bestV = float(v)
                        bestX = base.copy()
                else:
                    delta *= 0.5
                    if delta < 5e-4:
                        break
            except Exception:
                break
            if time.time() > lp_deadline - 0.5:
                break
    except Exception:
        pass

    # only return if beats fallback; else return fallback
    try:
        bestX = np.asarray(bestX, dtype=float)
        if bestX.shape != (76, 15):
            return _norm(fb)
        bestX = _norm(bestX)
        if not np.all(np.isfinite(bestX)):
            return _norm(fb)
        # final check
        v = _maxcos(bestX)
        if v <= bestV + 1e-12 and v < fbv - 1e-12:
            return bestX
        else:
            # if we improved return it, else fallback (which may equal bestX)
            if v < fbv:
                return bestX
            return _norm(fb)
    except Exception:
        return _norm(fb)
