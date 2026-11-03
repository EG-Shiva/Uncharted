import time
import numpy as np

def solve():
    import time as _time
    import numpy as _np
    t0 = _time.time()
    TIME_BUDGET = 168.0
    t_end = t0 + TIME_BUDGET
    rng = _np.random.default_rng(0)
    n, d = 59, 22
    p = d - 1

    def _normalize(X):
        nr = _np.linalg.norm(X, axis=1, keepdims=True)
        nr = _np.maximum(nr, 1e-12)
        return X / nr

    def _max_cos(X):
        G = X @ X.T
        _np.fill_diagonal(G, -2.0)
        return float(G.max())

    # load incumbents
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = _np.load(fn)
            a = _np.asarray(a, dtype=float)
            if a.shape == (n, d) and _np.all(_np.isfinite(a)):
                nr = _np.linalg.norm(a, axis=1, keepdims=True)
                nr = _np.maximum(nr, 1e-12)
                a = a / nr
                if _np.all(_np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
        cands.append(_normalize(rng.standard_normal((n, d))))
    vals = [_max_cos(c) for c in cands]
    gbestX = cands[int(_np.argmin(vals))].copy()
    gbest = float(min(vals))

    def tangent_bases(X):
        B = _np.empty((n, d, p), dtype=float)
        for i in range(n):
            xi = X[i]
            piv = int(_np.argmax(_np.abs(xi)))
            M = _np.empty((d, p), dtype=float)
            col = 0
            for k in range(d):
                if k == piv:
                    continue
                v = _np.zeros(d)
                v[k] = 1.0
                v = v - float(v @ xi) * xi
                M[:, col] = v
                col += 1
            Q, _ = _np.linalg.qr(M, mode='reduced')
            # ensure orthogonality to xi (span already tangent)
            B[i] = Q
        return B

    def adam_polish(X0, steps, beta0, beta1, lr, deadline):
        X = X0.copy()
        m_ad = _np.zeros_like(X)
        v_ad = _np.zeros_like(X)
        b1 = 0.9; b2 = 0.999; eps = 1e-8
        best = _max_cos(X)
        bestX = X.copy()
        pow1 = b1; pow2 = b2
        for t in range(1, steps + 1):
            if (t & 31) == 0 and _time.time() > deadline:
                break
            beta = beta0 + (beta1 - beta0) * (t / steps)
            G = X @ X.T
            _np.fill_diagonal(G, -2.0)
            gm = float(G.max())
            W = _np.exp(beta * (G - gm))
            _np.fill_diagonal(W, 0.0)
            rs = W.sum(axis=1, keepdims=True) + 1e-16
            D = (W @ X) / rs
            dot = (D * X).sum(axis=1, keepdims=True)
            Gt = D - dot * X
            m_ad *= b1; m_ad += (1.0 - b1) * Gt
            v_ad *= b2; v_ad += (1.0 - b2) * (Gt * Gt)
            mhat = m_ad / (1.0 - pow1)
            vhat = v_ad / (1.0 - pow2)
            pow1 *= b1; pow2 *= b2
            step = lr * mhat / (_np.sqrt(vhat) + eps)
            rn = _np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
            sc = _np.minimum(1.0, 0.05 / rn)
            X = X - step * sc
            X = _normalize(X)
            if (t % 25) == 0:
                cur = _max_cos(X)
                if cur < best:
                    best = cur; bestX = X.copy()
        return bestX, best

    # try import linprog once
    try:
        from scipy.optimize import linprog as _linprog
        has_lp = True
    except Exception:
        has_lp = False

    def trust_lp_for_marginals(X, delta, thresh=1e-5):
        # minimize t s.t. lin pair <= t for pairs within thresh
        G = X @ X.T
        _np.fill_diagonal(G, -2.0)
        m0 = float(G.max())
        pairs = []
        for i in range(n):
            Gi = G[i]
            for j in range(i + 1, n):
                if Gi[j] > m0 - thresh:
                    pairs.append((i, j))
        if len(pairs) == 0:
            return None, [], None, m0
        B = tangent_bases(X)
        Nvar = n * p + 1
        R = len(pairs)
        A = _np.zeros((R, Nvar), dtype=float)
        b = _np.empty(R, dtype=float)
        for r, (i, j) in enumerate(pairs):
            A[r, i * p:(i + 1) * p] = B[i].T @ X[j]
            A[r, j * p:(j + 1) * p] = B[j].T @ X[i]
            A[r, -1] = -1.0
            b[r] = -G[i, j]
        c = _np.zeros(Nvar); c[-1] = 1.0
        bounds = [(-delta, delta)] * (Nvar - 1) + [(-2.0, 2.0)]
        try:
            res = _linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs',
                           options={'time_limit': 3.0, 'presolve': True})
        except Exception:
            return None, pairs, None, m0
        duals = None
        try:
            if hasattr(res, 'ineqlin') and res.ineqlin is not None:
                # dict-like or object
                try:
                    duals = _np.asarray(res.ineqlin.marginals, dtype=float)
                except Exception:
                    try:
                        duals = _np.asarray(res.ineqlin['marginals'], dtype=float)
                    except Exception:
                        duals = None
            elif hasattr(res, 'marginals'):
                duals = _np.asarray(res.marginals, dtype=float)
        except Exception:
            duals = None
        return res, pairs, duals, m0

    def forced_sequence(Xstart, Sset, m0, delta, K, allow_rise=1e-4, drop=2e-4):
        X = Xstart.copy()
        target = m0 - drop
        Nvar = n * p + 1
        c = _np.zeros(Nvar); c[-1] = 1.0
        for _k in range(K):
            if _time.time() > t_end - 3.0:
                break
            G = X @ X.T
            _np.fill_diagonal(G, -2.0)
            B = tangent_bases(X)
            # other pairs within window
            others = []
            # use threshold 0.002 window to limit size
            thr = m0 - 0.002
            for i in range(n):
                Gi = G[i]
                for j in range(i + 1, n):
                    if (i, j) in Sset:
                        continue
                    if Gi[j] > thr:
                        others.append((i, j))
            # cap others size: keep worst 1200
            if len(others) > 1200:
                vals_o = _np.array([G[i, j] for (i, j) in others])
                idx = _np.argsort(-vals_o)[:1200]
                others = [others[k] for k in idx]
            Slist = list(Sset)
            R = len(Slist) + len(others)
            if R == 0:
                break
            A = _np.zeros((R, Nvar), dtype=float)
            b = _np.empty(R, dtype=float)
            for r, (i, j) in enumerate(Slist):
                A[r, i * p:(i + 1) * p] = B[i].T @ X[j]
                A[r, j * p:(j + 1) * p] = B[j].T @ X[i]
                # last col 0
                b[r] = target - G[i, j]
            for r2, (i, j) in enumerate(others):
                r = len(Slist) + r2
                A[r, i * p:(i + 1) * p] = B[i].T @ X[j]
                A[r, j * p:(j + 1) * p] = B[j].T @ X[i]
                A[r, -1] = -1.0
                b[r] = -G[i, j]
            bounds = [(-delta, delta)] * (Nvar - 1) + [(-2.0, m0 + allow_rise)]
            try:
                res = _linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs',
                               options={'time_limit': 3.0, 'presolve': True})
            except Exception:
                return None
            if res.status != 0:
                return None
            y = _np.asarray(res.x, dtype=float).ravel()
            dy = y[:n * p].reshape(n, p)
            V = _np.empty((n, d))
            for i in range(n):
                V[i] = B[i] @ dy[i]
            # limit step norm for stability
            vn = _np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
            # trust already bounds, but clip extreme
            V = V * _np.minimum(1.0, 0.05 / vn)
            Xn = _normalize(X + V)
            # check linearization sanity
            if not _np.all(_np.isfinite(Xn)):
                return None
            X = Xn
        return X

    # ---- forced contact-breaking trials ----
    if has_lp:
        # estimate active pairs of gbest for trial count
        G0 = gbestX @ gbestX.T
        _np.fill_diagonal(G0, -2.0)
        m_init = float(G0.max())
        # one marginal LP to rank
        try:
            _res0, _pairs0, _duals0, _m0 = trust_lp_for_marginals(gbestX, 0.006, thresh=1e-5)
        except Exception:
            _res0, _pairs0, _duals0, _m0 = None, [], None, m_init
        n_trials = 14
        for trial in range(n_trials):
            if _time.time() > t_end - 12.0:
                break
            # refresh pairs/duals every few trials from current gbest
            if trial % 4 == 0 and trial > 0:
                try:
                    _res0, _pairs0, _duals0, _m0 = trust_lp_for_marginals(gbestX, 0.006, thresh=1e-5)
                except Exception:
                    pass
            cur_m0 = gbest
            # build candidate pair pool: pairs within 1e-5 of cur max on gbest
            Gg = gbestX @ gbestX.T
            _np.fill_diagonal(Gg, -2.0)
            pool = [(i, j) for i in range(n) for j in range(i + 1, n) if Gg[i, j] > cur_m0 - 1e-5]
            if len(pool) < 20:
                # widen
                pool = [(i, j) for i in range(n) for j in range(i + 1, n) if Gg[i, j] > cur_m0 - 1e-4]
            if len(pool) == 0:
                break
            ksel = int(rng.integers(20, 51))
            ksel = min(ksel, len(pool))
            Sset = set()
            # use marginals if available and aligned
            use_dual = False
            if _duals0 is not None and _pairs0 is not None and len(_pairs0) == len(pool):
                try:
                    w = _np.abs(_np.asarray(_duals0).ravel())
                    if w.size == len(pool) and _np.all(_np.isfinite(w)) and w.max() > 0:
                        use_dual = True
                except Exception:
                    use_dual = False
            if use_dual:
                # weighted sampling favouring large duals + some randomness
                w = _np.abs(_np.asarray(_duals0).ravel()) + 1e-12
                # mix uniform to diversify
                w = 0.7 * (w / w.sum()) + 0.3 * (1.0 / len(pool))
                w = w / w.sum()
                chosen = rng.choice(len(pool), size=ksel, replace=False, p=w)
                for cidx in chosen:
                    Sset.add(pool[int(cidx)])
            else:
                perm = rng.permutation(len(pool))[:ksel]
                for cidx in perm:
                    Sset.add(pool[int(cidx)])
            # start point: gbest with tiny jitter to help feasibility
            Xstart = gbestX + 0.002 * rng.standard_normal((n, d))
            Xstart = _normalize(Xstart)
            delta = float(rng.choice([0.008, 0.010, 0.012]))
            K = int(rng.integers(5, 11))  # 5-10
            Xf = forced_sequence(Xstart, Sset, cur_m0, delta, K)
            if Xf is None:
                continue
            # short Adam polish after forced move
            rem = t_end - _time.time()
            if rem < 4.0:
                break
            steps = 700 if rem > 40 else 400
            Xp, vp = adam_polish(Xf, steps, 35.0, 260.0, 0.012, min(t_end - 1.0, _time.time() + 6.0))
            if vp < gbest:
                gbest = vp
                gbestX = Xp.copy()
                # refresh marginal info
                try:
                    _res0, _pairs0, _duals0, _m0 = trust_lp_for_marginals(gbestX, 0.006, thresh=1e-5)
                except Exception:
                    pass
            else:
                # second finer polish attempt from same forced point with higher beta
                if rem > 30:
                    Xp2, vp2 = adam_polish(Xf, 400, 150.0, 350.0, 0.007, min(t_end - 1.0, _time.time() + 3.0))
                    if vp2 < gbest:
                        gbest = vp2
                        gbestX = Xp2.copy()
            # early exit if big win? keep going to push further
        # end trials
    # ---- fallback / extra basin hopping with remaining time ----
    rem = t_end - _time.time()
    if rem > 5.0:
        # a few Adam restarts from shakes to exploit new basin
        for _r in range(30):
            if _time.time() > t_end - 2.0:
                break
            amp = float(rng.choice([0.03, 0.06, 0.10, 0.18, 0.30]))
            if rng.random() < 0.6:
                X0 = gbestX + amp * rng.standard_normal((n, d))
                X0 = _normalize(X0)
            else:
                X0 = _normalize(rng.standard_normal((n, d)))
            X1, v1 = adam_polish(X0, 600 if rem > 20 else 300, 35.0, 250.0, 0.012, min(t_end - 0.8, _time.time() + 4.0))
            if v1 < gbest:
                gbest = v1
                gbestX = X1.copy()
            rem = t_end - _time.time()
            if rem < 3.0:
                break
    gbestX = _normalize(gbestX)
    return gbestX
