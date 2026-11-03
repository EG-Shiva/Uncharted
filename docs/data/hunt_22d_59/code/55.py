import time
import numpy as np

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _adam_polish(X0, steps, beta0, beta1, lr0, lr1, forbidden=None, seed=0):
    # fast Riemannian Adam on log-sum-exp, optionally ignoring forbidden pairs
    n, d = X0.shape
    X = X0.copy()
    M = np.zeros_like(X)
    V = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1t, b2t = 1.0, 1.0
    best = _max_cos(X)
    bestX = X.copy()
    # forbidden mask
    Fmask = None
    if forbidden is not None and len(forbidden) > 0:
        Fmask = np.zeros((n, n), dtype=bool)
        for (a, b) in forbidden:
            Fmask[a, b] = True
            Fmask[b, a] = True
    lb0, lb1 = np.log(beta0), np.log(beta1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    Eye = np.eye(n)
    for s in range(steps):
        frac = s / max(1, steps - 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * frac))
        lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
        G = X @ X.T
        # mask forbidden for objective/gradient (allow them to rise)
        if Fmask is not None:
            G[Fmask] = -10.0
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        E = G - m
        E *= beta
        np.exp(E, out=E)
        np.fill_diagonal(E, 0.0)
        if Fmask is not None:
            E[Fmask] = 0.0
        S = float(E.sum())
        if not np.isfinite(S) or S <= 0:
            continue
        E /= S
        Gr = E @ X
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gr -= dot * X
        b1t *= b1; b2t *= b2
        M *= b1; M += (1.0 - b1) * Gr
        V *= b2; V += (1.0 - b2) * (Gr * Gr)
        Mh = M / (1.0 - b1t); Vh = V / (1.0 - b2t)
        X -= lr * Mh / (np.sqrt(Vh) + eps)
        X = _norm(X)
        if (s % 200 == 199) or s == steps - 1:
            v = _max_cos(X)
            if v < best:
                best = v
                bestX = X.copy()
    return bestX, best

def _tangent_bases(X):
    n, d = X.shape
    Bs = np.empty((n, d, d - 1))
    for i in range(n):
        x = X[i]
        # V_j = e_j - x_j * x for j != k
        k = int(np.argmax(np.abs(x)))
        V = np.zeros((d, d - 1))
        col = 0
        for j in range(d):
            if j == k:
                continue
            e = np.zeros(d); e[j] = 1.0
            V[:, col] = e - x[j] * x
            col += 1
        Q, _ = np.linalg.qr(V)
        Bs[i] = Q
    return Bs

def _solve_rigidity_lp(X, m, thresh=1e-4, delta=1e-3, cap=1200):
    # Build sparse rigidity LP: min t s.t. dot + (B_i v_i).x_j + (B_j v_j).x_i <= t
    n, d = X.shape
    dm = d - 1
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    # select tightest pairs within thresh of max
    # get upper triangle indices sorted by G desc
    iu = np.triu_indices(n, 1)
    vals = G[iu]
    order = np.argsort(-vals)
    sel = []
    for k in order:
        if vals[k] < m - thresh:
            break
        sel.append(k)
        if len(sel) >= cap:
            break
    P = len(sel)
    if P == 0:
        return None, None, None
    pairs = [(int(iu[0][order[k]]), int(iu[1][order[k]])) for k in range(P)]
    dots = np.array([float(G[a, b]) for (a, b) in pairs])
    Bs = _tangent_bases(X)
    Nv = n * dm
    N = Nv + 1
    # A_ub: P x N, b_ub = -dots + m? Let's formulate with t' = t - m (shift) for stability
    # constraint: coeff@v - t' <= m - dots  (since dots+coeff@v <= m + t')
    # Use variables [v; t'] with t' free, objective min t'
    A = np.zeros((P, N), dtype=np.float64)
    b = m - dots  # >=0 small
    for r, (a, bpair) in enumerate(pairs):
        # coeff for v_a: Bs[a].T @ X[bpair]
        ca = Bs[a].T @ X[bpair]  # dm
        cb = Bs[bpair].T @ X[a]
        A[r, a * dm:(a + 1) * dm] = ca
        A[r, bpair * dm:(bpair + 1) * dm] = cb
        A[r, Nv] = -1.0
    c = np.zeros(N); c[Nv] = 1.0
    bounds = [(-delta, delta)] * Nv + [(None, None)]
    try:
        import scipy.optimize as so
        res = so.linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs',
                         options={'presolve': True})
    except Exception:
        return None, None, None
    if not hasattr(res, 'x') or res.x is None:
        return None, None, None
    if not np.all(np.finite(res.x)):
        return None, pairs, None
    tpred = m + float(res.x[Nv])
    Vall = res.x[:Nv].reshape(n, dm)
    U = np.einsum('ijk,ik->ij', Bs, Vall)  # n x d
    # duals (stress)
    dual = None
    try:
        dual = np.asarray(res.ineqlin.marginals, dtype=float).ravel()
    except Exception:
        try:
            dual = np.asarray(res.get('ineqlin', {}).get('marginals', None))
        except Exception:
            dual = None
    if dual is not None:
        dual = np.abs(dual)
        if dual.shape[0] != P:
            dual = None
    return (tpred, U), pairs, dual

def solve():
    import time as _tm
    rng = np.random.default_rng(0)
    t0 = _tm.time()
    deadline = t0 + 168.0

    def left(sec=5.0):
        return _tm.time() < deadline - sec

    X0 = None
    best0 = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (59, 22):
                continue
            if not np.all(np.isfinite(a)):
                continue
            a = _norm(a)
            v = _max_cos(a)
            if v < best0:
                best0 = v
                X0 = a.copy()
        except Exception:
            continue
    if X0 is None:
        X0 = _norm(rng.standard_normal((59, 22)))
        best0 = _max_cos(X0)

    x_best = X0.copy()
    best_val = best0

    # Stage 0: quick annealed Adam to settle into basin center
    if left(30):
        y, v = _adam_polish(x_best, steps=2500, beta0=60.0, beta1=500.0,
                            lr0=0.008, lr1=0.0008)
        if v < best_val:
            best_val = v; x_best = y.copy()

    forbidden = []  # list of (a,b) to ignore temporarily
    # LP-guided loop: at most a few dozen, but time-checked to few
    max_lp_iters = 10
    for it in range(max_lp_iters):
        if not left(25.0):
            break
        G = x_best @ x_best.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        # one sparse HiGHS LP
        out, pairs, dual = _solve_rigidity_lp(x_best, m, thresh=1e-4, delta=1e-3, cap=1200)
        if out is None:
            break
        tpred, U = out
        # if LP predicts improvement, take step with backtracking
        if np.isfinite(tpred) and tpred < m - 5e-9:
            improved = False
            for alpha in (1.0, 0.5, 0.25, 0.12):
                Xt = _norm(x_best + alpha * U)
                v = _max_cos(Xt)
                if v < m - 1e-12:
                    # short repolish from stepped point (fast, masked if forbidden)
                    y, vy = _adam_polish(Xt, steps=1200, beta0=150.0, beta1=700.0,
                                        lr0=0.004, lr1=0.0005,
                                        forbidden=forbidden if forbidden else None)
                    if vy < best_val:
                        best_val = vy; x_best = y.copy(); improved = True
                    elif v < best_val:
                        best_val = v; x_best = Xt.copy(); improved = True
                    break
            if improved:
                # clear old forbiddens if we improved
                if len(forbidden) > 4:
                    forbidden = forbidden[-2:]
                continue
            # predicted but backtracking failed -> fall through to break
        # LP certifies local optimality (or step failed): extract stress, break one contact
        # choose victim: most heavily stressed or most redundant
        victim = None
        if dual is not None and pairs is not None and len(pairs) == len(dual):
            try:
                victim = pairs[int(np.argmax(dual))]
            except Exception:
                victim = None
        if victim is None:
            # fallback: highest degree-sum contact among tight pairs
            rowmax = G.max(axis=1)
            # degrees within 1e-4
            W = (G >= (m - 1e-4))
            deg = W.sum(axis=1)
            best_s = -1; victim = pairs[0] if pairs else None
            if pairs:
                for (a, b) in pairs[:200]:
                    s = int(deg[a] + deg[b])
                    if s > best_s:
                        best_s = s; victim = (a, b)
        if victim is None:
            break
        # forbid victim (require it to rise): add to forbidden set
        a, b = int(victim[0]), int(victim[1])
        if (a, b) not in forbidden and (b, a) not in forbidden:
            forbidden.append((a, b))
        # keep forbidden small
        if len(forbidden) > 3:
            forbidden = forbidden[-3:]
        # resolve / repolish with victim masked out so others can fall
        # first force separation a bit to ensure topology change
        xa, xb = x_best[a].copy(), x_best[b].copy()
        # tangent push apart
        d = xa - xb
        d -= (d @ xa) * xa
        dn = np.linalg.norm(d)
        if dn > 1e-12:
            d /= dn
            Xt = x_best.copy()
            Xt[a] = _norm((xa + 0.004 * d).reshape(1, -1))[0]
            # opposite for b
            d2 = xb - xa
            d2 -= (d2 @ xb) * xb
            n2 = np.linalg.norm(d2)
            if n2 > 1e-12:
                Xt[b] = _norm((xb + 0.004 * d2 / n2).reshape(1, -1))[0]
        else:
            Xt = x_best.copy()
        y, vy_masked_ignored = _adam_polish(Xt, steps=1800, beta0=80.0, beta1=600.0,
                                            lr0=0.006, lr1=0.0006,
                                            forbidden=forbidden)
        # true value including forbidden
        vy_true = _max_cos(y)
        # also try unmasked short polish from y to settle victim back
        if left(15):
            z, vz = _adam_polish(y, steps=1200, beta0=200.0, beta1=800.0,
                                 lr0=0.003, lr1=0.0004, forbidden=None)
            vz_true = vz  # _adam returns true max already
            if vz_true < vy_true:
                y, vy_true = z, vz_true
        if vy_true < best_val - 1e-12:
            best_val = vy_true; x_best = y.copy()
            # success: drop forbidden (new basin) to fully exploit
            forbidden = []
        else:
            # keep y as current for next LP even if slightly worse, to explore new topology
            # only if not much worse (within 1e-3) to avoid losing best
            if vy_true < m + 5e-4:
                x_best = y.copy()
            # else keep x_best and try different victim next round (forbidden accumulates)
            # to avoid infinite loop on same victim, continue
            pass
        # safety: if forbidden victim rose too much (>2e-2), drop it
        Gf = x_best @ x_best.T
        if Gf[a, b] > best_val + 0.02:
            forbidden = [p for p in forbidden if not (p == (a, b) or p == (b, a))]

    # Final ultra-fine polish on best basin (fast L-BFGS if time, else Adam)
    if left(12.0):
        try:
            import scipy.optimize as so
            n, d = x_best.shape
            Y0 = x_best.copy().reshape(-1)
            beta = 900.0
            def fg(Yf):
                Y = Yf.reshape(n, d)
                rn = np.linalg.norm(Y, axis=1, keepdims=True)
                rn = np.maximum(rn, 1e-12)
                X = Y / rn
                G = X @ X.T
                np.fill_diagonal(G, -2.0)
                mm = float(G.max())
                E = (G - mm) * beta
                np.exp(E, out=E)
                np.fill_diagonal(E, 0.0)
                S = float(E.sum())
                if S <= 0 or not np.isfinite(S):
                    S = 1.0
                f = mm + np.log(S) / beta
                W = E / S
                GX = (W @ X) * 2.0
                dot = np.sum(GX * X, axis=1, keepdims=True)
                GXt = GX - dot * X
                GY = GXt / rn
                return f, GY.reshape(-1)
            opts = {"maxiter": 250, "ftol": 1e-22, "gtol": 1e-14,
                    "maxfun": 800, "maxcor": 20}
            res = so.minimize(fg, Y0, jac=True, method="L-BFGS-B", options=opts)
            Yf = res.x.reshape(n, d)
            Xf = _norm(Yf)
            vf = _max_cos(Xf)
            if vf < best_val:
                best_val = vf; x_best = Xf.copy()
        except Exception:
            pass
    if left(4.0):
        y, v = _adam_polish(x_best, steps=1200, beta0=400.0, beta1=1500.0,
                            lr0=0.002, lr1=0.0003)
        if v < best_val:
            best_val = v; x_best = y.copy()

    x_best = _norm(x_best)
    return x_best
