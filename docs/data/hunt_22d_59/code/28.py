import time
import numpy as np

def _max_cos(Y):
    G = Y @ Y.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _normalize(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _adam_refine(Y0, steps, beta0, beta1, lr0, lr1):
    n, d = Y0.shape
    Y = Y0.copy()
    M = np.zeros_like(Y)
    V = np.zeros_like(Y)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    best_val = _max_cos(Y)
    best_Y = Y.copy()
    lb0 = np.log(beta0); lb1 = np.log(beta1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    b1t = 1.0; b2t = 1.0
    for t in range(1, steps + 1):
        frac = (t - 1) / max(1, steps - 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * frac))
        lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
        G = Y @ Y.T
        np.fill_diagonal(G, -10.0)
        gmax = float(G.max())
        G -= gmax
        G *= beta
        np.exp(G, out=G)
        np.fill_diagonal(G, 0.0)
        s = float(G.sum())
        if s > 0 and np.isfinite(s):
            G /= s
            Gr = G @ Y
            dot = np.sum(Gr * Y, axis=1, keepdims=True)
            Gr -= Y * dot
            b1t *= b1; b2t *= b2
            M *= b1; M += (1.0 - b1) * Gr
            V *= b2; V += (1.0 - b2) * (Gr * Gr)
            Mh = M / (1.0 - b1t)
            Vh = V / (1.0 - b2t)
            Y -= lr * Mh / (np.sqrt(Vh) + eps)
            rn = np.linalg.norm(Y, axis=1, keepdims=True)
            rn[rn == 0] = 1.0
            Y /= rn
        if (t % 200 == 0) or (t == steps):
            v = _max_cos(Y)
            if v < best_val:
                best_val = v
                best_Y = Y.copy()
    return best_Y, best_val

def _get_marginals(res, P):
    try:
        obj = getattr(res, "ineqlin", None)
        if obj is None:
            return None
        if hasattr(obj, "marginals"):
            m = np.asarray(obj.marginals, dtype=float).ravel()
            if m.size == P:
                return m
        if isinstance(obj, dict) and "marginals" in obj:
            m = np.asarray(obj["marginals"], dtype=float).ravel()
            if m.size == P:
                return m
        # tuple style?
        try:
            m = np.asarray(obj, dtype=float).ravel()
            if m.size == P:
                return m
        except Exception:
            pass
    except Exception:
        pass
    try:
        m = getattr(res, "marginals", None)
        if m is not None:
            m = np.asarray(m, dtype=float).ravel()
            if m.size == P:
                return m
    except Exception:
        pass
    return None

def _lp_solve(c, A_ub, b_ub, A_eq, b_eq, bounds, timelim=4.0):
    try:
        import scipy.optimize as _opt
        try:
            res = _opt.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                               bounds=bounds, method="highs",
                               options={"presolve": True, "time_limit": timelim})
        except Exception:
            res = _opt.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                               bounds=bounds, method="highs")
        return res
    except Exception:
        return None

def _standard_lp(Y, r):
    N, D = Y.shape
    G = Y @ Y.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    iu, ju = np.where(np.triu(G > (m - 1e-5), k=1))
    P = len(iu)
    if P == 0:
        return None, None, m, []
    nvar = N * D + 1
    A_ub = np.zeros((P, nvar), dtype=np.float64)
    b_ub = np.zeros(P, dtype=np.float64)
    for k in range(P):
        i = int(iu[k]); j = int(ju[k])
        A_ub[k, i*D:(i+1)*D] = Y[j]
        A_ub[k, j*D:(j+1)*D] = Y[i]
        A_ub[k, -1] = -1.0
        b_ub[k] = -float(G[i, j])
    c = np.zeros(nvar); c[-1] = 1.0
    A_eq = np.zeros((N, nvar), dtype=np.float64)
    for i in range(N):
        A_eq[i, i*D:(i+1)*D] = Y[i]
    b_eq = np.zeros(N)
    bounds = [(-r, r)] * (N*D) + [(m-0.02, m+0.02)]
    res = _lp_solve(c, A_ub, b_ub, A_eq, b_eq, bounds)
    pairs = list(zip([int(x) for x in iu], [int(x) for x in ju]))
    return res, _get_marginals(res, P) if res is not None else None, m, pairs

def _forced_sequence(Y0, m0, sel_set, r, steps, drop, rise):
    N, D = Y0.shape
    Y = Y0.copy()
    sel = set(sel_set)
    for _ in range(steps):
        G = Y @ Y.T
        np.fill_diagonal(G, -2.0)
        # candidate pool: pairs above m0-0.003 plus selected
        thr = m0 - 0.003
        mask = np.triu(G > thr, k=1)
        # ensure selected included
        for (a, b) in sel:
            mask[a, b] = True
        iu, ju = np.where(mask)
        P = len(iu)
        if P == 0:
            break
        # cap size to top 1200 by G
        if P > 1200:
            vals = G[iu, ju]
            idx = np.argsort(-vals)[:1200]
            iu = iu[idx]; ju = ju[idx]
            P = 1200
        # split selected / unselected within this list
        is_sel = np.array([((int(iu[k]), int(ju[k])) in sel) or ((int(ju[k]), int(iu[k])) in sel) for k in range(P)])
        n_unsel = int(np.sum(~is_sel))
        n_con = P + 1  # +1 for t<=m0+rise
        nvar = N * D + 1
        A_ub = np.zeros((n_con, nvar), dtype=np.float64)
        b_ub = np.zeros(n_con, dtype=np.float64)
        row = 0
        for k in range(P):
            i = int(iu[k]); j = int(ju[k])
            g = float(G[i, j])
            if is_sel[k]:
                A_ub[row, i*D:(i+1)*D] = Y[j]
                A_ub[row, j*D:(j+1)*D] = Y[i]
                A_ub[row, -1] = 0.0
                b_ub[row] = (m0 - drop) - g
            else:
                A_ub[row, i*D:(i+1)*D] = Y[j]
                A_ub[row, j*D:(j+1)*D] = Y[i]
                A_ub[row, -1] = -1.0
                b_ub[row] = -g
            row += 1
        A_ub[row, -1] = 1.0
        b_ub[row] = m0 + rise
        c = np.zeros(nvar); c[-1] = 1.0
        A_eq = np.zeros((N, nvar), dtype=np.float64)
        for i in range(N):
            A_eq[i, i*D:(i+1)*D] = Y[i]
        b_eq = np.zeros(N)
        bounds = [(-r, r)] * (N*D) + [(m0-0.02, m0+0.02)]
        res = _lp_solve(c, A_ub, b_ub, A_eq, b_eq, bounds, timelim=4.0)
        if res is None or getattr(res, "status", 2) != 0 or getattr(res, "x", None) is None:
            break
        d = np.asarray(res.x, dtype=float).ravel()
        if d.size != nvar or not np.all(np.finite(d)):
            break
        dv = d[:N*D].reshape(N, D)
        # limit step norm for safety
        Yn = _normalize(Y + dv)
        if not np.all(np.finite(Yn)):
            break
        Y = Yn
    return Y

def solve():
    rng = np.random.default_rng(0)
    t0 = time.time()
    T_END = t0 + 172.0

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
            a = _normalize(a)
            v = _max_cos(a)
            if v < best0:
                best0 = v
                X0 = a.copy()
        except Exception:
            continue
    if X0 is None:
        X0 = _normalize(rng.standard_normal((59, 22)))
        best0 = _max_cos(X0)

    x_best = X0.copy()
    best_val = best0
    m0 = best_val

    # quick standard LP info for marginals
    # run a few forced-facet trials with different subsets
    n_trials = 7
    for trial in range(n_trials):
        if time.time() > T_END - 35.0:
            break
        # standard LP to get marginals (r small)
        try:
            res_std, marg, m_cur, pairs = _standard_lp(x_best, r=0.006)
        except Exception:
            res_std, marg, m_cur, pairs = None, None, _max_cos(x_best), []
        m0 = _max_cos(x_best)
        P = len(pairs)
        # select 20-50 dominant contacts
        ssize = int(rng.integers(20, 51))
        sel = []
        if P == 0:
            sel = []
        elif marg is not None and np.all(np.finite(marg)) and float(np.abs(marg).max()) > 0:
            order = np.argsort(np.abs(marg))[::-1]
            # take top ssize with a bit of randomization: mix top-80
            pool = order[:min(P, 90)].tolist()
            # always include some of top-10
            must = order[:min(P, 10)].tolist()
            chosen = set()
            for q in must[:5]:
                chosen.add(int(q))
            rng.shuffle(pool)
            for q in pool:
                if len(chosen) >= ssize:
                    break
                chosen.add(int(q))
            # fill if needed
            for q in order:
                if len(chosen) >= ssize:
                    break
                chosen.add(int(q))
            sel = [pairs[int(q)] for q in list(chosen)[:ssize]]
        else:
            # randomized subset weighted to tightest: pairs already tight (within 1e-5), pick random
            # also add near-tight pool within 2e-4 for diversity on some trials
            if trial % 3 == 2:
                G = x_best @ x_best.T
                np.fill_diagonal(G, -2.0)
                iu2, ju2 = np.where(np.triu(G > (m0 - 2e-4), k=1))
                pool2 = list(zip([int(x) for x in iu2], [int(x) for x in ju2]))
                if len(pool2) > ssize:
                    idx = rng.choice(len(pool2), size=ssize, replace=False)
                    sel = [pool2[int(k)] for k in idx]
                else:
                    sel = pool2
            else:
                if P > ssize:
                    idx = rng.choice(P, size=ssize, replace=False)
                    sel = [pairs[int(k)] for k in idx]
                else:
                    sel = pairs
        # normalize pair order
        sel_set = set()
        for (a, b) in sel:
            if a > b:
                a, b = b, a
            sel_set.add((a, b))
        # forced sequence params vary per trial
        drop = 2e-4 if (trial % 2 == 0) else 3e-4
        rise = 1e-4
        r = 0.010 if (trial % 3 != 1) else 0.014
        steps = 6 if (trial < 5) else 5
        try:
            Yf = _forced_sequence(x_best, m0, sel_set, r=r, steps=steps, drop=drop, rise=rise)
        except Exception:
            continue
        if time.time() > T_END - 30.0:
            break
        # check forced actually broke contacts (optional) then polish
        try:
            Yp, vp = _adam_refine(Yf, steps=1800, beta0=80.0, beta1=600.0, lr0=0.006, lr1=0.0009)
        except Exception:
            continue
        if vp < best_val - 1e-12:
            best_val = vp
            x_best = Yp.copy()
        else:
            # also try a second polish from slightly perturbed forced point once in a while
            if trial % 4 == 3 and time.time() < T_END - 40.0:
                try:
                    Yn = _normalize(Yf + 0.003 * rng.standard_normal((59, 22)))
                    Yp2, vp2 = _adam_refine(Yn, steps=1200, beta0=90.0, beta1=500.0, lr0=0.005, lr1=0.001)
                    if vp2 < best_val - 1e-12:
                        best_val = vp2
                        x_best = Yp2.copy()
                except Exception:
                    pass
        # firm clock: stop trials if little time
        if time.time() > T_END - 32.0:
            break

    # final polish of best
    if time.time() < T_END - 25.0:
        try:
            y, v = _adam_refine(x_best, steps=3500, beta0=200.0, beta1=1000.0, lr0=0.0035, lr1=0.0004)
            if v < best_val:
                best_val = v
                x_best = y.copy()
        except Exception:
            pass
    if time.time() < T_END - 8.0:
        try:
            import scipy.optimize as _opt
            N, D = x_best.shape
            X0f = x_best.copy().reshape(-1)
            beta = 900.0
            def fun(x):
                X = x.reshape(N, D)
                rsq = np.sum(X*X, axis=1, keepdims=True)
                rn = np.sqrt(np.maximum(rsq, 1e-24))
                Y = X / rn
                G = Y @ Y.T
                np.fill_diagonal(G, -2.0)
                m = float(G.max())
                Gs = G - m
                Gs *= beta
                np.exp(Gs, out=Gs)
                np.fill_diagonal(Gs, 0.0)
                Sp = float(Gs.sum()*0.5)
                if Sp <= 0 or not np.isfinite(Sp):
                    return 1e9, np.zeros_like(x)
                L = m + np.log(Sp)/beta
                GY = (Gs @ Y)/Sp
                dot = np.sum(GY*Y, axis=1, keepdims=True)
                GX = (GY - Y*dot)/rn
                return float(L), GX.reshape(-1)
            try:
                res = _opt.minimize(fun, X0f, jac=True, method="L-BFGS-B",
                                    options={"maxiter": 80, "ftol": 1e-22, "gtol": 1e-14})
                Yo = _normalize(res.x.reshape(N, D))
                vo = _max_cos(Yo)
                if vo < best_val:
                    best_val = vo
                    x_best = Yo.copy()
            except Exception:
                pass
        except Exception:
            pass

    x_best = _normalize(x_best)
    return x_best
