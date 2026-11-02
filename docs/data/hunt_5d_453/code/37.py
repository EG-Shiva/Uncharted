import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _build_sym(p, a, W1, W2a):
    z0, t0, ph0, z1, t1, ph1, ps1, z2, t2, ph2, ps2 = p
    zs = (z0, z1, z2)
    ts = (t0, t1, t2)
    phs = (ph0, ph1, ph2)
    pss = (0.0, ps1, ps2)
    n151 = W1.shape[0]
    X = np.empty((3 * n151, 5), dtype=np.float64)
    for k in range(3):
        z = float(zs[k])
        if z > 0.999:
            z = 0.999
        if z < -0.999:
            z = -0.999
        s = np.sqrt(max(0.0, 1.0 - z * z))
        r1 = s * np.cos(float(ts[k]))
        r2 = s * np.sin(float(ts[k]))
        psi = float(pss[k])
        phi = float(phs[k])
        c1 = np.cos(psi + W1)
        s1 = np.sin(psi + W1)
        c2 = np.cos(phi + W2a)
        s2 = np.sin(phi + W2a)
        blk = np.empty((n151, 5), dtype=np.float64)
        blk[:, 0] = r1 * c1
        blk[:, 1] = r1 * s1
        blk[:, 2] = r2 * c2
        blk[:, 3] = r2 * s2
        blk[:, 4] = z
        X[k * n151:(k + 1) * n151, :] = blk
    # renormalize (safety)
    X /= np.sqrt(np.sum(X * X, axis=1, keepdims=True)) + 1e-30
    return X

def _softmax_relax(X, beta, steps, lr0, lr1, mu, t_end):
    n, d = X.shape
    v = np.zeros_like(X)
    for it in range(steps):
        if time.time() > t_end:
            break
        G = X @ X.T
        np.fill_diagonal(G, -np.inf)
        m = float(np.max(G))
        if not np.isfinite(m):
            break
        with np.errstate(over='ignore', invalid='ignore'):
            W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        Z = float(np.sum(W) * 0.5)
        if not np.isfinite(Z) or Z <= 0:
            break
        grad = (W @ X) / Z
        dot = np.sum(grad * X, axis=1, keepdims=True)
        Gt = grad - dot * X
        mn = float(np.max(np.sqrt(np.sum(Gt * Gt, axis=1))))
        if not np.isfinite(mn) or mn < 1e-14:
            break
        Gt /= mn
        v = mu * v + (1.0 - mu) * Gt
        frac = it / max(1, steps - 1)
        lr = lr0 + (lr1 - lr0) * frac
        Y = X - lr * v
        yn = np.sqrt(np.sum(Y * Y, axis=1, keepdims=True))
        yn[yn == 0] = 1.0
        X = Y / yn
    return X

def _fine_greedy(x, bestV, t_end, eps_init=8e-8, max_iters=20000):
    n, d = x.shape
    eps = float(eps_init)
    it = 0
    check = 0
    while it < max_iters:
        it += 1
        check += 1
        if check >= 256:
            check = 0
            if time.time() > t_end:
                break
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        if m < bestV:
            bestV = m
        flat = int(np.argmax(G))
        i = flat // n
        j = flat % n
        xi = x[i].copy()
        xj = x[j].copy()
        cij = float(xi @ xj)
        if cij > 1.0:
            cij = 1.0
        if cij < -1.0:
            cij = -1.0
        ti = xj - cij * xi
        ni = float(np.linalg.norm(ti))
        if ni < 1e-12:
            ti = np.random.randn(d)
            ti -= float(ti @ xi) * xi
            ni = float(np.linalg.norm(ti)) + 1e-30
        ti /= ni
        tj = xi - cij * xj
        nj = float(np.linalg.norm(tj))
        if nj < 1e-12:
            tj = np.random.randn(d)
            tj -= float(tj @ xj) * xj
            nj = float(np.linalg.norm(tj)) + 1e-30
        tj /= nj
        G[i, :] = -2.0
        G[:, i] = -2.0
        G[j, :] = -2.0
        G[:, j] = -2.0
        m_rest = float(np.max(G))
        ri = x @ xi
        rj = x @ xj
        ri[i] = -2.0
        ri[j] = -2.0
        rj[i] = -2.0
        rj[j] = -2.0
        m_ri_max = float(np.max(ri))
        m_rj_max = float(np.max(rj))
        m_rest_i = m_rest if m_rest > m_rj_max else m_rj_max
        m_rest_j = m_rest if m_rest > m_ri_max else m_ri_max
        best_m2 = m
        best_vi = None
        best_vj = None
        best_single = 0
        best_single_v = None
        for scale in (eps, eps * 0.5, eps * 0.22):
            vi = xi - scale * ti
            nv = float(np.linalg.norm(vi))
            if nv == 0 or not np.isfinite(nv):
                continue
            vi /= nv
            vj = xj - scale * tj
            nv2 = float(np.linalg.norm(vj))
            if nv2 == 0 or not np.isfinite(nv2):
                continue
            vj /= nv2
            di = x @ vi
            dj = x @ vj
            di[i] = -2.0
            di[j] = -2.0
            dj[i] = -2.0
            dj[j] = -2.0
            mc = m_rest
            a1 = float(np.max(di))
            if a1 > mc:
                mc = a1
            b1 = float(np.max(dj))
            if b1 > mc:
                mc = b1
            c2 = float(vi @ vj)
            if c2 > mc:
                mc = c2
            if mc < best_m2 - 1e-15:
                best_m2 = mc
                best_vi = vi
                best_vj = vj
                best_single = 0
        for scale in (eps, eps * 0.5):
            vi = xi - scale * ti
            nv = float(np.linalg.norm(vi))
            if nv == 0 or not np.isfinite(nv):
                continue
            vi /= nv
            di = x @ vi
            di[i] = -2.0
            mc = m_rest_i
            a1 = float(np.max(di))
            if a1 > mc:
                mc = a1
            if mc < best_m2 - 1e-15:
                best_m2 = mc
                best_single = 1
                best_single_v = vi
                best_vi = vi
                best_vj = None
        for scale in (eps, eps * 0.5):
            vj = xj - scale * tj
            nv2 = float(np.linalg.norm(vj))
            if nv2 == 0 or not np.isfinite(nv2):
                continue
            vj /= nv2
            dj = x @ vj
            dj[j] = -2.0
            mc = m_rest_j
            a1 = float(np.max(dj))
            if a1 > mc:
                mc = a1
            if mc < best_m2 - 1e-15:
                best_m2 = mc
                best_single = 2
                best_single_v = vj
                best_vi = None
                best_vj = vj
        if best_m2 < m - 1e-15:
            if best_single == 0:
                x[i] = best_vi
                x[j] = best_vj
            elif best_single == 1:
                x[i] = best_single_v
            else:
                x[j] = best_single_v
            bestV = best_m2
            eps *= 0.98
            if eps < 1e-13:
                eps = 1e-13
        else:
            eps *= 0.55
            if eps < 5e-14:
                break
    fv = _max_cos(x)
    if fv < bestV:
        bestV = fv
    return x, bestV

def solve():
    t0 = time.time()
    T_END = t0 + 172.0
    T_SYM_END = t0 + 48.0
    T_SYM_POL = t0 + 82.0
    np.random.seed(0)
    rng = np.random.default_rng(0)
    n, d = 453, 5
    N151 = 151
    # load incumbent
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            cands.append(np.asarray(a, dtype=np.float64))
        except Exception:
            pass
    bestX = None
    bestV = float("inf")
    for cc in cands:
        try:
            if cc.shape != (453, 5):
                continue
            nn = np.sqrt(np.sum(cc * cc, axis=1, keepdims=True))
            nn[nn == 0] = 1.0
            cc = cc / nn
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
    incumbentV = bestV

    # ---- symmetric C151 search ----
    sym_best_val = float("inf")
    sym_best_p = None
    sym_best_a = None
    sym_best_W1 = None
    sym_best_W2 = None
    TW = 2.0 * np.pi / 151.0
    jidx = np.arange(N151, dtype=np.float64)
    W1_base = jidx * TW
    alist = [1, 10, 20, 37, 50]
    PI2 = 2.0 * np.pi
    psi_max = TW
    for aidx, a in enumerate(alist):
        if time.time() > T_SYM_END - 1.0:
            break
        W1 = W1_base
        W2 = (jidx * (TW * float(a)))
        W2 = np.mod(W2, PI2)
        bounds = [(-0.9, 0.9), (0.05, 1.5208), (0.0, PI2),
                  (-0.9, 0.9), (0.05, 1.5208), (0.0, PI2), (0.0, psi_max),
                  (-0.9, 0.9), (0.05, 1.5208), (0.0, PI2), (0.0, psi_max)]
        def f(p, W1=W1, W2=W2, aa=a):
            try:
                X = _build_sym(p, aa, W1, W2)
                return _max_cos(X)
            except Exception:
                return 2.0
        try:
            res = differential_evolution(
                f, bounds, maxiter=22, popsize=9, seed=aidx,
                polish=False, updating='immediate', init='latinhypercube',
                workers=1, atol=0.0, tol=0.01)
            v = float(res.fun)
            if v < sym_best_val:
                sym_best_val = v
                sym_best_p = np.array(res.x, dtype=np.float64)
                sym_best_a = a
                sym_best_W1 = W1
                sym_best_W2 = W2
        except Exception:
            continue
    # local refine best symmetric with Nelder-Mead on exact max
    if sym_best_p is not None and time.time() < T_SYM_END:
        try:
            W1 = sym_best_W1
            W2 = sym_best_W2
            aa = sym_best_a
            def f2(p):
                try:
                    X = _build_sym(p, aa, W1, W2)
                    return _max_cos(X)
                except Exception:
                    return 2.0
            rem = max(0.5, T_SYM_END - time.time())
            # limit evaluations by time: ~600 max
            r2 = minimize(f2, sym_best_p, method='Nelder-Mead',
                          options={'maxfev': 600, 'xatol': 1e-4, 'fatol': 1e-9,
                                   'adaptive': True})
            try:
                v2 = float(r2.fun)
                if v2 < sym_best_val and np.all(np.isfinite(r2.x)):
                    sym_best_val = v2
                    sym_best_p = np.array(r2.x, dtype=np.float64)
            except Exception:
                pass
        except Exception:
            pass
    Xsym = None
    if sym_best_p is not None:
        try:
            Xsym = _build_sym(sym_best_p, sym_best_a, sym_best_W1, sym_best_W2)
            vs = _max_cos(Xsym)
            sym_best_val = vs
        except Exception:
            Xsym = None

    # ---- break symmetry and polish symmetric basin ----
    if Xsym is not None and time.time() < T_SYM_POL:
        try:
            # break symmetry with small tangent noise
            noise = rng.standard_normal(Xsym.shape) * 0.0025
            # project noise to tangent
            dotn = np.sum(noise * Xsym, axis=1, keepdims=True)
            noise -= dotn * Xsym
            Y = Xsym + noise
            Y /= np.linalg.norm(Y, axis=1, keepdims=True) + 1e-30
            # also partially symmetrised incumbent: snap nearest points onto orbits (cheap alternative basin)
            # quick softmax anneal of broken-symmetry code
            Y = _softmax_relax(Y, 500.0, 220, 0.0022, 0.00022, 0.88, min(time.time() + 8.0, T_SYM_POL))
            if time.time() < T_SYM_POL:
                Y, vy = _fine_greedy(Y, _max_cos(Y), min(time.time() + 6.0, T_SYM_POL), eps_init=6e-8, max_iters=8000)
                if vy < bestV and np.all(np.isfinite(Y)):
                    bestV = vy
                    bestX = Y.copy()
            # keep Y for further joint hopping even if not best
            # store as alternate start
            try:
                sym_polished = Y.copy()
                sym_polished_v = _max_cos(sym_polished)
            except Exception:
                sym_polished = None
        except Exception:
            sym_polished = None
    else:
        sym_polished = None

    # ---- polish incumbent / global best with remaining time ----
    # quick initial squeeze
    try:
        if time.time() < T_END - 8.0:
            bestX, bestV = _fine_greedy(bestX, bestV, min(time.time() + 6.0, T_END - 8.0), eps_init=1.0e-7, max_iters=20000)
    except Exception:
        pass
    betas = [600.0, 1200.0, 350.0, 900.0]
    sigmas = [0.004, 0.0015, 0.007, 0.0025]
    k = 0
    # basin hopping mixing incumbent best and symmetric polished basin
    sources = []
    if sym_polished is not None:
        try:
            sources.append(sym_polished.copy())
        except Exception:
            pass
    while time.time() < T_END - 6.0:
        k += 1
        if T_END - 6.0 - time.time() < 1.0:
            break
        # choose base: mostly global best, occasionally symmetric basin
        use_sym = (sym_polished is not None) and ((k % 4) == 0)
        if use_sym:
            base = sym_polished
        else:
            base = bestX
        n0 = base.shape[0]
        G = base @ base.T
        np.fill_diagonal(G, -2.0)
        flat = int(np.argmax(G))
        wi = flat // n0
        wj = flat % n0
        beta = float(betas[k % len(betas)])
        sigma = float(sigmas[k % len(sigmas)])
        y = base.copy()
        try:
            # neighborhood shake around worst pair
            row_i = G[wi].copy()
            row_j = G[wj].copy()
            ti_idx = np.argpartition(row_i, -8)[-8:]
            tj_idx = np.argpartition(row_j, -8)[-8:]
            sset = set([int(wi), int(wj)])
            for t_ in list(ti_idx) + list(tj_idx):
                sset.add(int(t_))
            for _ in range(4):
                sset.add(int(rng.integers(0, n0)))
            sidx = np.array(list(sset), dtype=int)
            noise = rng.standard_normal((len(sidx), d)) * sigma
            for kk2, idx in enumerate(sidx):
                vv = noise[kk2] - float(np.dot(noise[kk2], y[idx])) * y[idx]
                y[idx] = y[idx] + vv
            y /= np.linalg.norm(y, axis=1, keepdims=True) + 1e-30
            inner_end = min(time.time() + 2.5, T_END - 6.0)
            y = _softmax_relax(y, beta, 200, 0.0020, 0.00020, 0.88, inner_end)
        except Exception:
            continue
        if not np.all(np.isfinite(y)):
            continue
        try:
            y, vy = _fine_greedy(y, _max_cos(y), min(time.time() + 0.8, T_END - 6.0), eps_init=5e-8, max_iters=2200)
        except Exception:
            continue
        if vy < bestV:
            bestX = y
            bestV = vy
            if use_sym:
                sym_polished = y.copy()
    # final long squeeze
    try:
        bestX, bestV = _fine_greedy(bestX, bestV, T_END - 0.4, eps_init=7e-8, max_iters=1000000)
    except Exception:
        pass
    bestX = bestX / (np.linalg.norm(bestX, axis=1, keepdims=True) + 1e-30)
    return bestX
