import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def solve():
    import time as _tm
    import numpy as _np
    np = _np
    t0 = _tm.time()
    DEADLINE = t0 + 168.0
    rng = np.random.default_rng(0)
    n, d = 59, 22

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                a = _norm(a)
                cands.append(a)
        except Exception:
            pass
    if not cands:
        cands.append(_norm(rng.standard_normal((n, d))))
    vals = [_max_cos(c) for c in cands]
    best = cands[int(np.argmin(vals))].copy()
    best_val = float(min(vals))

    def adam_polish(X0, steps, a0, a1, lr0, lr1):
        X = X0.copy()
        m = np.zeros_like(X)
        v = np.zeros_like(X)
        b1, b2, eps = 0.9, 0.999, 1e-8
        la0, la1 = np.log(a0), np.log(a1)
        ll0, ll1 = np.log(lr0), np.log(lr1)
        cur_best = X.copy()
        cur_val = _max_cos(X)
        for t in range(1, steps + 1):
            if (t & 511) == 0 and _tm.time() > DEADLINE - 4:
                break
            frac = t / steps
            alpha = float(np.exp(la0 + (la1 - la0) * frac))
            lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            M = float(np.max(G))
            G -= M
            G *= alpha
            np.exp(G, out=G)
            np.fill_diagonal(G, 0.0)
            S = float(np.sum(G))
            if not np.isfinite(S) or S <= 0:
                continue
            Gr = (G @ X) / S
            dot = np.sum(Gr * X, axis=1, keepdims=True)
            Gt = Gr - dot * X
            m *= b1; m += (1.0 - b1) * Gt
            v *= b2; v += (1.0 - b2) * (Gt * Gt)
            c1 = 1.0 - (b1 ** t); c2 = 1.0 - (b2 ** t)
            mc = m / c1; vc = v / c2
            X -= lr * mc / (np.sqrt(vc) + eps)
            nn = np.linalg.norm(X, axis=1, keepdims=True)
            nn[nn == 0] = 1.0
            X /= nn
            if (t % 600 == 0) or (t == steps):
                cv = _max_cos(X)
                if cv < cur_val:
                    cur_val = cv; cur_best = X.copy()
        fv = _max_cos(X)
        if fv < cur_val:
            cur_val = fv; cur_best = X.copy()
        return cur_best, cur_val

    def lbfgs_polish(X0, beta, maxiter):
        try:
            from scipy.optimize import minimize
        except Exception:
            return X0.copy(), _max_cos(X0)
        Z0 = X0.copy().ravel()
        nn0, dd0 = X0.shape
        def fun_g(zf):
            Z = zf.reshape(nn0, dd0)
            rn = np.sqrt(np.sum(Z * Z, axis=1, keepdims=True))
            rn[rn == 0] = 1.0
            X = Z / rn
            G = X @ X.T
            np.fill_diagonal(G, -10.0)
            gm = float(np.max(G))
            E = G - gm
            E *= beta
            np.exp(E, out=E)
            np.fill_diagonal(E, 0.0)
            S = float(np.sum(E))
            if S <= 0 or not np.isfinite(S):
                return 1e9, np.zeros_like(zf)
            L = gm + np.log(S) / beta
            E /= S
            GX = (E @ X) * 2.0
            dot = np.sum(GX * X, axis=1, keepdims=True)
            GXt = GX - dot * X
            DZ = GXt / rn
            return float(L), DZ.ravel()
        try:
            res = minimize(fun_g, Z0, jac=True, method='L-BFGS-B',
                           options={'maxiter': int(maxiter), 'ftol': 1e-22, 'gtol': 1e-14, 'maxfun': 100000})
            Z = res.x.reshape(nn0, dd0)
            X = _norm(Z)
            return X, _max_cos(X)
        except Exception:
            return X0.copy(), _max_cos(X0)

    def deep_polish(X, light=True):
        if light:
            Y, vy = adam_polish(X, steps=1800, a0=120, a1=900, lr0=0.004, lr1=0.0004)
            if _tm.time() > DEADLINE - 8:
                return Y, vy
            # quick lbfgs
            Y2, v2 = lbfgs_polish(Y, beta=800.0, maxiter=80)
            if v2 < vy:
                return Y2, v2
            return Y, vy
        else:
            Y, vy = adam_polish(X, steps=4000, a0=80, a1=1200, lr0=0.005, lr1=0.0003)
            for be, mi in ((500, 100), (1500, 100)):
                if _tm.time() > DEADLINE - 8:
                    break
                Y2, v2 = lbfgs_polish(Y, beta=be, maxiter=mi)
                if v2 < vy:
                    vy = v2; Y = Y2
            return Y, vy

    def hole_replace(X, kill_idx, n_cand=3000):
        Y = X.copy()
        killset = set([int(k) for k in kill_idx])
        keep = [i for i in range(n) if i not in killset]
        Xk = X[keep].copy()
        for k in kill_idx:
            k = int(k)
            C = rng.standard_normal((n_cand, d))
            C /= np.linalg.norm(C, axis=1, keepdims=True)
            sc = C @ Xk.T
            mx = np.max(sc, axis=1)
            bi = int(np.argmin(mx))
            Y[k] = C[bi]
            Xk = np.vstack([Xk, Y[k:k+1]])
        return Y

    def coherent_shift(X, cap=None, step=None):
        Y = X.copy()
        v = rng.standard_normal(d); v /= np.linalg.norm(v)
        proj = X @ v
        if cap is None:
            cap = int(rng.integers(4, 11))
        idx = np.argsort(-proj)[:cap]
        w = rng.standard_normal(d)
        w -= (w @ v) * v
        nw = np.linalg.norm(w)
        if nw == 0:
            return _norm(Y + 0.01 * rng.standard_normal((n, d)))
        w /= nw
        if step is None:
            step = float(rng.uniform(0.08, 0.28))
        for k in idx:
            Y[k] = Y[k] + step * w + 0.008 * rng.standard_normal(d)
        # light jitter rest
        Y += 0.002 * rng.standard_normal((n, d))
        return _norm(Y)

    def twist_move(X, theta=None):
        # random plane a,b
        a = rng.standard_normal(d); a /= np.linalg.norm(a)
        b = rng.standard_normal(d); b -= (b @ a) * a
        b /= (np.linalg.norm(b) + 1e-12)
        if theta is None:
            theta = float(rng.uniform(0.08, 0.35)) * (1 if rng.random() < 0.5 else -1)
        proj = X @ a
        # hemisphere subset (about half) to rotate
        if rng.random() < 0.5:
            sel = np.where(proj > np.median(proj))[0]
        else:
            # cap subset
            v = rng.standard_normal(d); v /= np.linalg.norm(v)
            pr2 = X @ v
            cap = int(rng.integers(6, 16))
            sel = np.argsort(-pr2)[:cap]
        c, s = np.cos(theta), np.sin(theta)
        Y = X.copy()
        for k in sel:
            xa = float(X[k] @ a); xb = float(X[k] @ b)
            rest = X[k] - xa * a - xb * b
            xa2 = xa * c - xb * s
            xb2 = xa * s + xb * c
            Y[k] = rest + xa2 * a + xb2 * b
        Y += 0.002 * rng.standard_normal((n, d))
        return _norm(Y)

    # ---- Phase 0: diverse pool, light polish to find new basins ----
    pool = [(best, best_val)]
    # heavily perturbed copies (leave basin)
    for sig in (0.35, 0.55, 0.8):
        Y = _norm(best + sig * rng.standard_normal((n, d)))
        pool.append((Y, _max_cos(Y)))
    # pure random seeds
    for _ in range(6):
        Y = _norm(rng.standard_normal((n, d)))
        pool.append((Y, _max_cos(Y)))
    # cross-polytope + random completion (symmetry-broken structured seed)
    try:
        CP = np.zeros((44, d))
        for i in range(22):
            CP[2*i, i] = 1.0; CP[2*i+1, i] = -1.0
        R = rng.standard_normal((15, d)); R = _norm(R)
        Y0 = np.vstack([CP, R])
        # shuffle rows
        perm = rng.permutation(n)
        Y0 = _norm(Y0[perm] + 0.3 * rng.standard_normal((n, d)))
        pool.append((Y0, _max_cos(Y0)))
    except Exception:
        pass

    # light polish of pool (fast triage)
    triaged = []
    for (Y, _) in pool:
        if _tm.time() > DEADLINE - 60:
            break
        Yp, Yv = adam_polish(Y, steps=1500, a0=60, a1=600, lr0=0.006, lr1=0.0006)
        triaged.append((Yp, Yv))
        if Yv < best_val - 1e-12:
            best, best_val = Yp, Yv
    # keep elites: sort triaged, keep top 4 distinct
    triaged.sort(key=lambda t: t[1])
    elites = []
    for Yp, Yv in triaged[:6]:
        elites.append([Yp.copy(), Yv])
    # ensure best is elite[0]
    if elites:
        bi = int(np.argmin([e[1] for e in elites]))
        if elites[bi][1] < best_val:
            best, best_val = elites[bi][0].copy(), elites[bi][1]
        # move best to front
        elites[0], elites[bi] = elites[bi], elites[0]
    else:
        elites = [[best.copy(), best_val]]
    # quick lbfgs on current best
    if _tm.time() < DEADLINE - 40:
        Y2, v2 = lbfgs_polish(best, beta=900.0, maxiter=100)
        if v2 < best_val:
            best, best_val = Y2, v2
            elites[0] = [best.copy(), best_val]

    it = 0
    while _tm.time() < DEADLINE - 10:
        it += 1
        remain = DEADLINE - _tm.time()
        if remain < 9:
            break
        # pick source elite (round robin, bias to best)
        src_idx = 0 if (it % 3 == 0) else int(rng.integers(0, len(elites)))
        src = elites[src_idx][0]
        G = src @ src.T
        np.fill_diagonal(G, -2.0)
        rowmax = np.max(G, axis=1)
        order_worst = np.argsort(-rowmax)
        mode = it % 6
        Y = None
        if mode == 0:
            Y = coherent_shift(src)
        elif mode == 1:
            Y = twist_move(src)
        elif mode == 2:
            k = int(rng.integers(3, 8))
            # kill worst + random cap members for heterogeneity
            kill = order_worst[:min(k, 3)].tolist()
            while len(kill) < k:
                kill.append(int(rng.integers(0, n)))
            kill = np.unique(np.array(kill))[:k]
            Y = hole_replace(src, kill, n_cand=4000)
            Y = _norm(Y + 0.002 * rng.standard_normal((n, d)))
        elif mode == 3:
            # large global jump from best but coherent-biased
            sig = float(rng.uniform(0.15, 0.4))
            Y = _norm(src + sig * rng.standard_normal((n, d)))
        elif mode == 4:
            # single hole + twist combo
            k = int(rng.integers(0, n))
            Y = hole_replace(src, np.array([k]), n_cand=6000)
            Y = twist_move(Y, theta=float(rng.uniform(0.05, 0.15)))
        else:
            # cap reseed: replace a cap with fresh mutually-spread points
            v = rng.standard_normal(d); v /= np.linalg.norm(v)
            pr = src @ v
            cap = int(rng.integers(5, 10))
            kill = np.argsort(-pr)[:cap]
            Y = hole_replace(src, kill, n_cand=3000)

        Yp, Yv = deep_polish(Y, light=True)
        # update elites / best
        if Yv < best_val - 1e-12:
            best, best_val = Yp.copy(), float(Yv)
            elites[0] = [best.copy(), best_val]
        else:
            # diversity: replace worst elite if sufficiently different and decent
            # keep pool size <=4
            if len(elites) < 4 and Yv < best_val + 0.01:
                # check distance to existing
                elites.append([Yp.copy(), float(Yv)])
            elif len(elites) >= 1:
                # simulated-annealing style: occasionally accept worse as new elite slot
                wi = int(np.argmax([e[1] for e in elites]))
                if Yv < elites[wi][1] and Yv < best_val + 0.005:
                    elites[wi] = [Yp.copy(), float(Yv)]
        if (it % 14) == 0 and _tm.time() < DEADLINE - 22:
            Yd, vd = deep_polish(best, light=False)
            if vd < best_val:
                best, best_val = Yd, vd
                elites[0] = [best.copy(), best_val]

    if _tm.time() < DEADLINE - 2:
        Y, vy = adam_polish(best, steps=2200, a0=900, a1=2500, lr0=0.0012, lr1=0.00012)
        if vy < best_val:
            best, best_val = Y, vy
    if _tm.time() < DEADLINE - 1:
        for be in (1200, 2500):
            Y2, v2 = lbfgs_polish(best, beta=be, maxiter=120)
            if v2 < best_val:
                best, best_val = Y2, v2
            if _tm.time() > DEADLINE - 1:
                break
    best = _norm(best)
    return best
