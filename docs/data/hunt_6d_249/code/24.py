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
    _np.random.seed(20261103)
    n, d = 249, 6

    # load incumbents
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = _np.load(fn)
            a = _np.asarray(a, dtype=float)
            if a.shape == (n, d):
                an = a / _np.maximum(_np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
                if _np.all(_np.isfinite(an)):
                    cands.append((_max_cos(an), an))
        except Exception:
            pass
    if not cands:
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
    Npair = len(I0)

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
            W = _np.zeros((n, n))
            W[I0, J0] = w
            W = W + W.T
            A = W @ Xn
            proj = _np.sum(A * Xn, axis=1)
            Grad0 = (A - proj[:, None] * Xn) / r[:, None]
            diff = r - 1.0
            L = Lsoft + creg * _np.sum(diff * diff)
            Grad = Grad0 + (2.0 * creg * diff)[:, None] * (X / r[:, None])
            return float(L), Grad.ravel()
        return fg

    def soft_polish(Xinit, taus, iters):
        Xf = Xinit.ravel().copy()
        for tau, mi in zip(taus, iters):
            if _time.time() - t0 > BUDGET - 8:
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
            if _time.time() - t0 > BUDGET - 2:
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
        return cur, curVal

    def lp_polish(Xinit, seconds, radius0=2e-3):
        X = Xinit.copy()
        X /= _np.maximum(_np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        best = _max_cos(X)
        bestXl = X.copy()
        radius = float(radius0)
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

    # tangent bases via eigh of projector
    def tangent_bases(X):
        T = _np.empty((n, d, d - 1))
        for i in range(n):
            xi = X[i]
            M = _np.eye(d) - _np.outer(xi, xi)
            w, V = _np.linalg.eigh(M)
            # smallest eigenvalue ~0 corresponds to xi; take largest 5
            T[i] = V[:, 1:6] if w[0] < w[1] else V[:, 0:5]
            # ensure orthonormal & orthogonal to xi (numerical)
            # quick re-orthogonalize
            # T already orthonormal
        return T

    def forced_flex_step(X, hub, radius, delta, Kclus, capP=3000):
        # contact list with slack below 2e-4 (plus cap)
        G = X @ X.T
        _np.fill_diagonal(G, -2.0)
        m = float(G.max())
        cfull = G[iu0]
        slack = m - cfull
        # select tight set for Jacobian: slack < 2e-4, else top capP
        sel = slack < 2e-4
        if int(_np.sum(sel)) < 800:
            # take tightest capP to have enough constraints
            k = min(capP, Npair)
            idx = _np.argpartition(slack, k - 1)[:k]
            # keep only those with slack < 2e-3 to avoid far pairs? keep all k
            sel = _np.zeros(Npair, dtype=bool)
            sel[idx] = True
        elif int(_np.sum(sel)) > capP:
            # keep tightest capP among them
            idxs = _np.where(sel)[0]
            order = _np.argsort(slack[idxs])[:capP]
            sel = _np.zeros(Npair, dtype=bool)
            sel[idxs[order]] = True
        I = I0[sel]; J = J0[sel]; csel = cfull[sel]; p = len(I)
        if p < 20:
            return None, None
        # SVD diagnostic (flex identification) - compute smallest singular directions
        # Build cluster around hub: top-K nearest by cosine
        row = G[hub].copy()
        row[hub] = -2.0
        neigh = _np.argsort(-row)[:int(Kclus)]
        nodeset = _np.zeros(n, dtype=bool)
        nodeset[hub] = True
        nodeset[neigh] = True
        inS = nodeset[I] & nodeset[J]
        s_idx = _np.where(inS)[0]
        if len(s_idx) < 100:
            # enlarge cluster
            neigh2 = _np.argsort(-row)[:int(Kclus * 1.6 + 10)]
            nodeset[:] = False
            nodeset[hub] = True
            nodeset[neigh2] = True
            inS = nodeset[I] & nodeset[J]
            s_idx = _np.where(inS)[0]
        if len(s_idx) > 300:
            # keep tightest 300 (smallest slack)
            sl = m - csel[s_idx]
            o = _np.argsort(sl)[:300]
            s_idx = s_idx[o]
        if len(s_idx) < 30:
            # fallback: take contacts incident to hub
            s_idx = _np.where((I == hub) | (J == hub))[0]
            if len(s_idx) == 0:
                return None, None
        is_forced = _np.zeros(p, dtype=bool)
        is_forced[s_idx] = True

        T = tangent_bases(X)  # n x 6 x 5
        # Build LP: vars u (n*5) + t .  a_k^T u - t <= -c_k (-delta if forced)
        # a for pair k: block I: x_J^T T_I , block J: x_I^T T_J
        dp = d - 1
        Nv = n * dp
        # sparse construction
        # each row has 2*dp +1 nonzeros
        nnz = p * (2 * dp + 1)
        rows = _np.empty(nnz, dtype=_np.int64)
        cols = _np.empty(nnz, dtype=_np.int64)
        vals = _np.empty(nnz, dtype=float)
        b = -csel.copy()
        b[is_forced] -= float(delta)
        pos = 0
        # precompute projections
        # for speed loop in python over p (<=3000) fine
        for k in range(p):
            i = int(I[k]); j = int(J[k])
            ai = T[i].T @ X[j]  # 5
            aj = T[j].T @ X[i]  # 5
            base = k * (2 * dp + 1)
            # fill rows
            r0 = k
            for s in range(dp):
                rows[base + s] = r0
                cols[base + s] = i * dp + s
                vals[base + s] = ai[s]
            for s in range(dp):
                rows[base + dp + s] = r0
                cols[base + dp + s] = j * dp + s
                vals[base + dp + s] = aj[s]
            rows[base + 2 * dp] = r0
            cols[base + 2 * dp] = Nv
            vals[base + 2 * dp] = -1.0
        A_ub = _coo((vals, (rows, _np.arange(nnz) * 0 + rows)), shape=(p, Nv + 1))
        # The above coo trick wrong: need (rows, cols). Fix:
        A_ub = _coo((vals, (rows, cols)), shape=(p, Nv + 1)).tocsr()
        obj = _np.zeros(Nv + 1); obj[-1] = 1.0
        bounds = [(-float(radius), float(radius))] * Nv + [(None, None)]
        try:
            res = _linprog(obj, A_ub=A_ub, b_ub=b, bounds=bounds, method="highs")
        except Exception:
            return None, None
        if res.status != 0:
            return None, None
        u = res.x[:-1].reshape(n, dp)
        t_pred = float(res.x[-1])
        # step along displacement even if worsens
        dX = _np.einsum('nik,nk->ni', T, u)
        Y = X + dX
        Y /= _np.maximum(_np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
        return Y, dict(t_pred=t_pred, m=m, nforced=int(_np.sum(is_forced)), p=p, hub=int(hub))

    # ---- initial squeeze ----
    Xp = soft_polish(bestX, [0.008, 0.003, 0.0012], [60, 70, 80])
    vp = _max_cos(Xp)
    if vp < bestVal:
        bestX, bestVal = Xp, vp
    Xe, ve = greedy_polish(bestX, min(t0 + 8, t0 + BUDGET - 20), step0=2e-3)
    if ve < bestVal:
        bestX, bestVal = Xe, ve
    if _time.time() - t0 < BUDGET - 40:
        Xl, vl = lp_polish(bestX, 4.0)
        if vl < bestVal:
            bestX, bestVal = Xl, vl

    cur = bestX.copy()
    curVal = bestVal

    taus_cheap = [0.006, 0.0022]
    iters_cheap = [40, 55]
    taus_full = [0.005, 0.002, 0.0008]
    iters_full = [50, 70, 90]

    radii = [0.01, 0.02, 0.035, 0.05, 0.015, 0.03]
    deltas = [1e-4, 5e-4, 1e-3, 5e-4, 1e-4, 1e-3]
    Kclus_list = [26, 30, 34, 28, 32, 24]

    it = 0
    # precompute hub list refresh every few iters
    while _time.time() - t0 < BUDGET - 22 and it < 60:
        it += 1
        # identify overloaded hubs using slack < 2e-4 contact counts (Jacobian set)
        Gc = cur @ cur.T
        _np.fill_diagonal(Gc, -2.0)
        mc = float(Gc.max())
        # contact counts for hub detection: use tight threshold 2e-4 to match Jacobian
        Ccnt = _np.sum(Gc > mc - 2e-4, axis=1)
        # also ultra-tight for tie-break
        Ccnt0 = _np.sum(Gc > mc - 1e-9, axis=1)
        order_hubs = _np.argsort(-(Ccnt * 10 + Ccnt0))
        # cycle hubs: pick among top 8
        hub = int(order_hubs[(it - 1) % 8])
        radius = radii[(it - 1) % len(radii)]
        delta = deltas[(it - 1) % len(deltas)]
        Kclus = Kclus_list[(it - 1) % len(Kclus_list)]
        # vary slightly
        if it % 7 == 0:
            Kclus = 40
            radius = 0.045
        if it % 11 == 0:
            delta = 1e-3
            radius = 0.05

        Y, info = forced_flex_step(cur, hub, radius, delta, Kclus)
        if Y is None:
            # fallback random shake on hub neighbourhood
            Xt = cur.copy()
            Xt[hub] += 0.02 * _np.random.randn(d)
            Xt /= _np.maximum(_np.linalg.norm(Xt, axis=1, keepdims=True), 1e-12)
            Y = Xt
            info = dict(t_pred=curVal, m=curVal, nforced=0, p=0, hub=hub)
        vY = _max_cos(Y)
        # step even if worsens: polish from Y
        Xq = soft_polish(Y, taus_cheap, iters_cheap)
        vq = _max_cos(Xq)
        Xq2, vq2 = greedy_polish(Xq, min(_time.time() + 0.5, t0 + BUDGET - 20), step0=1.2e-3)
        if vq2 < vq:
            Xq, vq = Xq2, vq2
        # update best with full equalization if promising
        if vq < bestVal - 1e-12:
            Xd = soft_polish(Xq, [0.0016, 0.0007], [70, 100])
            vd = _max_cos(Xd)
            if vd < vq:
                Xq, vq = Xd, vd
            Xg, vg = greedy_polish(Xq, min(_time.time() + 0.6, t0 + BUDGET - 18), step0=8e-4)
            if vg < vq:
                Xq, vq = Xg, vg
            Xl2, vl2 = lp_polish(Xq, 1.2)
            if vl2 < vq:
                Xq, vq = Xl2, vl2
            if vq < bestVal:
                bestX, bestVal = Xq.copy(), vq
                cur, curVal = Xq.copy(), vq
                continue
        # non-improving: move cur to floppy pocket if not too bad (allow worsening)
        # accept if within 3e-3 of best (temporary worsening allowed)
        if vq < bestVal + 0.003:
            cur, curVal = Xq.copy(), vq
        else:
            # too bad: restart from best with forced step of different hub next time
            # small jitter around best to diversify
            if it % 3 == 0:
                cur = bestX.copy()
                curVal = bestVal
            else:
                cur, curVal = Xq.copy(), vq
                # if far worse, pull back toward best partially
                if vq > bestVal + 0.006:
                    cur = bestX + 0.5 * (cur - bestX)
                    cur /= _np.maximum(_np.linalg.norm(cur, axis=1, keepdims=True), 1e-12)
                    curVal = _max_cos(cur)
        # occasional intensify near-miss basins
        if vq < bestVal + 0.0005 and (it % 5 == 0) and (_time.time() - t0 < BUDGET - 30):
            Xd = soft_polish(Xq, taus_full, iters_full)
            vd = _max_cos(Xd)
            Xg, vg = greedy_polish(Xd, min(_time.time() + 0.5, t0 + BUDGET - 20), step0=9e-4)
            vv = vg if vg < vd else vd
            XX = Xg if vg < vd else Xd
            if vv < bestVal:
                Xl2, vl2 = lp_polish(XX, 1.2)
                if vl2 < vv:
                    XX, vv = Xl2, vl2
                if vv < bestVal:
                    bestX, bestVal = XX.copy(), vv
                    cur, curVal = XX.copy(), vv

    # final intensive polish of best
    if _time.time() - t0 < BUDGET - 5:
        Xf = soft_polish(bestX, [0.002, 0.0009, 0.00045], [70, 90, 80])
        vf = _max_cos(Xf)
        if vf < bestVal:
            bestX, bestVal = Xf, vf
        Xg, vg = greedy_polish(bestX, min(_time.time() + 1.5, t0 + BUDGET - 3), step0=7e-4)
        if vg < bestVal:
            bestX, bestVal = Xg, vg
        if _time.time() - t0 < BUDGET - 8:
            Xl, vl = lp_polish(bestX, min(4.0, max(0.5, t0 + BUDGET - 2 - _time.time())))
            if vl < bestVal:
                bestX, bestVal = Xl, vl

    bestX /= _np.maximum(_np.linalg.norm(bestX, axis=1, keepdims=True), 1e-12)
    return bestX
