import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _anneal(X, steps, tau0, tau1, lr0, lr1, mu, t_end, rng, best_holder=None):
    n, d = X.shape
    v = np.zeros_like(X)
    ltau = np.log(tau1 / tau0)
    llr = np.log(lr1 / lr0)
    for it in range(steps):
        if (it & 31) == 0 and time.time() > t_end:
            break
        frac = it / max(steps - 1, 1)
        tau = tau0 * np.exp(ltau * frac)
        lr = lr0 * np.exp(llr * frac)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        if best_holder is not None and m < best_holder[0]:
            best_holder[0] = m
            best_holder[1] = X.copy()
        inv = 1.0 / tau
        # softmax weights: exp((G-m)/tau)
        with np.errstate(over='ignore'):
            P = np.exp((G - m) * inv)
        # P diag ~ exp((-2-m)/tau) ~ 0
        np.fill_diagonal(P, 0.0)
        s = float(np.sum(P))
        if not np.isfinite(s) or s <= 0:
            break
        P /= s
        Grad = P @ X
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        mn = float(np.max(np.sqrt(np.sum(Gt * Gt, axis=1))))
        if not np.isfinite(mn) or mn < 1e-18:
            break
        Gt /= mn
        v = mu * v + (1.0 - mu) * Gt
        X = X - lr * v
        X /= np.sqrt(np.sum(X * X, axis=1, keepdims=True))
    return X

def _shave(X, iters, eps_init, t_end, rng, best_holder=None):
    n, d = X.shape
    if best_holder is not None:
        bestV = best_holder[0]
    else:
        bestV = _max_cos(X)
        best_holder = [bestV, X.copy()]
    eps = float(eps_init)
    Gbuf = np.empty((n, n))
    for it in range(iters):
        if (it & 63) == 0 and time.time() > t_end:
            break
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        if m < best_holder[0]:
            best_holder[0] = m
            best_holder[1] = X.copy()
        flat = int(np.argmax(G))
        i = flat // n
        j = flat % n
        xi = X[i].copy()
        xj = X[j].copy()
        cij = float(xi @ xj)
        if cij > 1.0: cij = 1.0
        if cij < -1.0: cij = -1.0
        ti = xj - cij * xi
        ni = float(np.linalg.norm(ti))
        if ni < 1e-12:
            ti = rng.standard_normal(d)
            ti -= float(ti @ xi) * xi
            ni = float(np.linalg.norm(ti)) + 1e-30
        ti /= ni
        tj = xi - cij * xj
        nj = float(np.linalg.norm(tj))
        if nj < 1e-12:
            tj = rng.standard_normal(d)
            tj -= float(tj @ xj) * xj
            nj = float(np.linalg.norm(tj)) + 1e-30
        tj /= nj
        # rest max excluding i,j
        G[i, :] = -2.0; G[:, i] = -2.0
        G[j, :] = -2.0; G[:, j] = -2.0
        m_rest = float(np.max(G))
        improved = False
        for sc in (eps, eps * 0.5, eps * 0.25):
            vi = xi - sc * ti
            vi /= float(np.linalg.norm(vi))
            vj = xj - sc * tj
            vj /= float(np.linalg.norm(vj))
            di = X @ vi
            dj = X @ vj
            # exclude self rows
            a = float(np.max(np.delete(di, [i, j])))
            b = float(np.max(np.delete(dj, [i, j])))
            c2 = float(vi @ vj)
            mc = m_rest
            if a > mc: mc = a
            if b > mc: mc = b
            if c2 > mc: mc = c2
            if mc < m - 1e-15:
                X[i] = vi; X[j] = vj
                if mc < best_holder[0]:
                    best_holder[0] = mc
                    best_holder[1] = X.copy()
                eps = min(eps * 1.03, 3e-3)
                improved = True
                break
        if not improved:
            # try single-sided
            for sc in (eps,):
                vi = xi - sc * ti
                vi /= float(np.linalg.norm(vi))
                di = X @ vi
                a = float(np.max(np.delete(di, [i, j])))
                mc = m_rest
                if a > mc: mc = a
                # vj old vs? cij approx; use max of rest and j-pair
                cj = float(vi @ xj)
                if cj > mc: mc = cj
                if mc < m - 1e-15:
                    X[i] = vi
                    if mc < best_holder[0]:
                        best_holder[0] = mc
                        best_holder[1] = X.copy()
                    improved = True
                    break
                vj = xj - sc * tj
                vj /= float(np.linalg.norm(vj))
                dj = X @ vj
                b = float(np.max(np.delete(dj, [i, j])))
                mc2 = m_rest
                if b > mc2: mc2 = b
                ci = float(vj @ xi)
                if ci > mc2: mc2 = ci
                if mc2 < m - 1e-15:
                    X[j] = vj
                    if mc2 < best_holder[0]:
                        best_holder[0] = mc2
                        best_holder[1] = X.copy()
                    improved = True
                    break
            if not improved:
                eps *= 0.85
                if eps < 3e-10:
                    k = i if (it & 1) == 0 else j
                    rnd = rng.standard_normal(d)
                    rnd -= float(rnd @ X[k]) * X[k]
                    rn = float(np.linalg.norm(rnd))
                    if rn > 1e-12:
                        rnd /= rn
                        X[k] = X[k] + 8e-10 * rnd
                        X[k] /= float(np.linalg.norm(X[k]))
                    eps = 3e-10
    return X

def _hole_replace(X, k, rng, n_cand=6000):
    n, d = X.shape
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    per = np.max(G, axis=1)
    worst = np.argsort(per)[-k:]
    keep_mask = np.ones(n, dtype=bool)
    keep_mask[worst] = False
    Xkeep = X[keep_mask]
    C = rng.standard_normal((n_cand, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    # score = max dot to keep set; want minimal
    D = C @ Xkeep.T
    md = np.max(D, axis=1)
    order = np.argsort(md)
    X2 = X.copy()
    used = set()
    oi = 0
    for w in worst:
        # pick next unused candidate far from already inserted too
        while oi < n_cand:
            ci = int(order[oi]); oi += 1
            if ci in used:
                continue
            used.add(ci)
            break
        else:
            ci = int(order[0])
        X2[w] = C[ci]
    X2 /= np.linalg.norm(X2, axis=1, keepdims=True)
    return X2

def solve():
    t0 = time.time()
    DEAD = t0 + 172.0
    np.random.seed(0)
    rng = np.random.default_rng(0)
    n, d = 453, 5
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            cands.append(np.asarray(a, dtype=np.float64))
        except Exception:
            pass
    bestV = float("inf")
    bestX = None
    for cc in cands:
        try:
            if cc.shape != (n, d):
                continue
            cc = cc / np.linalg.norm(cc, axis=1, keepdims=True)
            if not np.all(np.isfinite(cc)):
                continue
            vv = _max_cos(cc)
            if vv < bestV:
                bestV = vv
                bestX = cc.copy()
        except Exception:
            continue
    if bestX is None:
        bestX = rng.standard_normal((n, d))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        bestV = _max_cos(bestX)
    incumbentX = bestX.copy()
    holder = [bestV, bestX.copy()]

    # ---- C151 symmetric search ----
    sym_best_X = None
    sym_best_V = float("inf")
    try:
        from scipy.optimize import differential_evolution, minimize
        TWO_PI = 2.0 * np.pi
        J = np.arange(151)
        BASE1 = TWO_PI * J / 151.0
        def build_sym(vec, a):
            X = np.empty((453, 5), dtype=np.float64)
            for k in range(3):
                w = float(vec[k*4+0]); th = float(vec[k*4+1])
                p1 = float(vec[k*4+2]); p2 = float(vec[k*4+3])
                if w > 0.95: w = 0.95
                if w < -0.95: w = -0.95
                rr = np.sqrt(max(1e-12, 1.0 - w*w))
                r1 = rr * np.cos(th); r2 = rr * np.sin(th)
                a1 = BASE1 + p1
                a2 = float(a) * BASE1 + p2
                c1 = np.cos(a1); s1 = np.sin(a1)
                c2 = np.cos(a2); s2 = np.sin(a2)
                s = k*151; e = s+151
                X[s:e, 0] = r1 * c1
                X[s:e, 1] = r1 * s1
                X[s:e, 2] = r2 * c2
                X[s:e, 3] = r2 * s2
                X[s:e, 4] = w
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            return X
        bounds = [(-0.9, 0.9), (0.05, 1.52), (0.0, TWO_PI), (0.0, TWO_PI)] * 3
        SYM_END = t0 + 32.0
        for a in [1, 10, 20, 37, 50]:
            if time.time() > SYM_END:
                break
            if time.time() > DEAD - 135.0:
                break
            try:
                res = differential_evolution(
                    lambda v: float(np.max(np.fill_diagonal(np.zeros((1,1)),0) or 0) if False else _max_cos(build_sym(np.asarray(v), a))),
                    bounds, maxiter=7, popsize=8, seed=1000+a,
                    polish=False, updating='deferred', workers=1, disp=False, tol=1e-9)
                v0 = np.asarray(res.x, dtype=np.float64)
                curV = float(res.fun)
            except Exception:
                continue
            # LSE refine
            try:
                beta = 900.0
                def lse_obj(vec):
                    Xc = build_sym(vec, a)
                    G = Xc @ Xc.T
                    np.fill_diagonal(G, -np.inf)
                    m = float(np.max(G))
                    with np.errstate(over='ignore'):
                        W = np.exp(beta * (G - m))
                    np.fill_diagonal(W, 0.0)
                    s = float(np.sum(W)) * 0.5
                    if not np.isfinite(s) or s <= 0:
                        return m
                    return m + np.log(s) / beta
                r2 = minimize(lse_obj, v0, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 60, 'ftol': 1e-12, 'gtol': 1e-8})
                v1 = np.asarray(r2.x, dtype=np.float64)
                X1 = build_sym(v1, a)
                V1 = _max_cos(X1)
                if V1 < curV:
                    v0 = v1; curV = V1
            except Exception:
                pass
            try:
                Xc = build_sym(v0, a)
                Vc = _max_cos(Xc)
            except Exception:
                continue
            if Vc < sym_best_V:
                sym_best_V = Vc
                sym_best_X = Xc.copy()
    except Exception:
        pass

    # broken-symmetry alt basin
    altX = None
    if sym_best_X is not None and np.all(np.isfinite(sym_best_X)):
        try:
            tmp = sym_best_X + 0.003 * rng.standard_normal((n, d))
            tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
            tmp = _anneal(tmp, steps=120, tau0=0.004, tau1=0.0008, lr0=0.02, lr1=0.004,
                          mu=0.85, t_end=min(time.time()+5.0, DEAD-120), rng=rng, best_holder=holder)
            altX = tmp
            # hybrid: inject 25 sym points into worst rattlers of incumbent
            try:
                G0 = incumbentX @ incumbentX.T
                np.fill_diagonal(G0, -2.0)
                per = np.max(G0, axis=1)
                worst = np.argsort(per)[-25:]
                keep = np.ones(n, dtype=bool); keep[worst] = False
                Xkeep = incumbentX[keep]
                dots = sym_best_X @ Xkeep.T
                md = np.max(dots, axis=1)
                order = np.argsort(md)[:25]
                Xhyb = incumbentX.copy()
                Xhyb[worst] = sym_best_X[order]
                Xhyb /= np.linalg.norm(Xhyb, axis=1, keepdims=True)
                Vhyb = _max_cos(Xhyb)
                # anneal hybrid briefly and consider
                Xhyb2 = _anneal(Xhyb, steps=80, tau0=0.003, tau1=0.0008, lr0=0.015, lr1=0.004,
                                mu=0.85, t_end=min(time.time()+4.0, DEAD-115), rng=rng, best_holder=holder)
                hv = _max_cos(Xhyb2)
                av = _max_cos(altX)
                altX = Xhyb2 if hv < av else altX
            except Exception:
                pass
        except Exception:
            altX = None

    # sync holder best
    if holder[0] < bestV:
        bestV = holder[0]; bestX = holder[1].copy()
    # polish incumbent + alt with anneal+shave
    starts = [incumbentX]
    if altX is not None:
        starts.append(altX)
    # also current best
    starts.append(bestX.copy())
    for si, S in enumerate(starts):
        if time.time() > DEAD - 95.0:
            break
        Xw = S.copy()
        Xw = _anneal(Xw, steps=220, tau0=0.0035, tau1=0.00035, lr0=0.025, lr1=0.003,
                     mu=0.88, t_end=min(time.time()+7.0, DEAD-90), rng=rng, best_holder=holder)
        Xw = _shave(Xw, iters=600, eps_init=1.2e-3, t_end=min(time.time()+5.0, DEAD-88), rng=rng, best_holder=holder)
        if holder[0] < bestV:
            bestV = holder[0]; bestX = holder[1].copy()

    # elitist basin hopping with hole surgery + shakes
    it = 0
    sigmas = [0.004, 0.008, 0.015, 0.006, 0.012]
    while time.time() < DEAD - 4.0:
        it += 1
        base = holder[1].copy()
        r = rng.random()
        if r < 0.45:
            k = int(rng.integers(1, 4))
            Xp = _hole_replace(base, k=k, rng=rng, n_cand=6000)
            # small jitter on rest
            Xp = Xp + 0.0015 * rng.standard_normal((n, d))
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
            Xp = _anneal(Xp, steps=160, tau0=0.003, tau1=0.0004, lr0=0.02, lr1=0.003,
                         mu=0.87, t_end=min(time.time()+4.5, DEAD-3.5), rng=rng, best_holder=holder)
            Xp = _shave(Xp, iters=350, eps_init=8e-4, t_end=min(time.time()+2.5, DEAD-3.0), rng=rng, best_holder=holder)
        elif r < 0.8:
            sig = sigmas[it % len(sigmas)]
            Xp = base + sig * rng.standard_normal((n, d))
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
            Xp = _anneal(Xp, steps=140, tau0=0.0025, tau1=0.00035, lr0=0.018, lr1=0.003,
                         mu=0.87, t_end=min(time.time()+4.0, DEAD-3.5), rng=rng, best_holder=holder)
            Xp = _shave(Xp, iters=300, eps_init=8e-4, t_end=min(time.time()+2.0, DEAD-3.0), rng=rng, best_holder=holder)
        else:
            # targeted shake of worst 8 points only
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            per = np.max(G, axis=1)
            worst = np.argsort(per)[-8:]
            Xp = base.copy()
            Xp[worst] += 0.01 * rng.standard_normal((8, d))
            Xp[worst] /= np.linalg.norm(Xp[worst], axis=1, keepdims=True)
            Xp = _anneal(Xp, steps=120, tau0=0.002, tau1=0.0003, lr0=0.015, lr1=0.003,
                         mu=0.86, t_end=min(time.time()+3.5, DEAD-3.5), rng=rng, best_holder=holder)
            Xp = _shave(Xp, iters=300, eps_init=6e-4, t_end=min(time.time()+2.0, DEAD-3.0), rng=rng, best_holder=holder)
        if holder[0] < bestV:
            bestV = holder[0]; bestX = holder[1].copy()
        # occasional restart from best to avoid drift
        if it % 25 == 0 and time.time() < DEAD - 10:
            pass

    Xout = holder[1].copy()
    Xout /= np.linalg.norm(Xout, axis=1, keepdims=True)
    # strict finite check
    Xout[~np.isfinite(Xout)] = 0.0
    nn = np.linalg.norm(Xout, axis=1, keepdims=True)
    nn[nn == 0] = 1.0
    # fix any zero rows deterministically
    for i in range(n):
        if float(np.linalg.norm(Xout[i])) < 1e-12:
            v = rng.standard_normal(d); v /= float(np.linalg.norm(v))
            Xout[i] = v
    Xout /= np.linalg.norm(Xout, axis=1, keepdims=True)
    return Xout
