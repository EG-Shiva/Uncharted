import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _softmax_relax(X, beta, steps, lr, mu, t_end):
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
        s = float(np.sum(W))
        if not np.isfinite(s) or s <= 0:
            break
        W /= s
        grad = W @ X
        dot = np.sum(grad * X, axis=1, keepdims=True)
        Gt = grad - dot * X
        mn = float(np.max(np.sqrt(np.sum(Gt * Gt, axis=1))))
        if not np.isfinite(mn) or mn < 1e-16:
            break
        Gt /= mn
        v = mu * v + (1.0 - mu) * Gt
        Y = X - lr * v
        yn = np.sqrt(np.sum(Y * Y, axis=1, keepdims=True))
        yn[yn == 0] = 1.0
        X = Y / yn
    return X

def _micro_shave(X, eps_init, t_end, bestV=None):
    n, d = X.shape
    if bestV is None:
        bestV = _max_cos(X)
    eps = float(eps_init)
    it = 0
    while time.time() < t_end:
        it += 1
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        if m < bestV:
            bestV = m
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
        Ri = G[i, :].copy()
        Rj = G[j, :].copy()
        G[i, :] = -2.0; G[:, i] = -2.0
        G[j, :] = -2.0; G[:, j] = -2.0
        m_rest = float(np.max(G))
        r_imax = float(np.max(np.delete(Ri, [i, j]))) if n > 2 else -2.0
        r_jmax = float(np.max(np.delete(Rj, [i, j]))) if n > 2 else -2.0
        base_i_only = m_rest if m_rest > r_jmax else r_jmax
        base_j_only = m_rest if m_rest > r_imax else r_imax
        best_m2 = m
        best_vi = None
        best_vj = None
        mode = 0
        for sc in (eps, eps * 0.5, eps * 0.25):
            vi = xi - sc * ti
            nvi = float(np.linalg.norm(vi))
            if nvi == 0 or not np.isfinite(nvi):
                continue
            vi /= nvi
            vj = xj - sc * tj
            nvj = float(np.linalg.norm(vj))
            if nvj == 0 or not np.isfinite(nvj):
                continue
            vj /= nvj
            di = X @ vi
            dj = X @ vj
            di[i] = -2.0; di[j] = -2.0
            dj[i] = -2.0; dj[j] = -2.0
            mc = m_rest
            a = float(np.max(di))
            if a > mc: mc = a
            b = float(np.max(dj))
            if b > mc: mc = b
            c2 = float(vi @ vj)
            if c2 > mc: mc = c2
            if mc < best_m2 - 1e-15:
                best_m2 = mc
                best_vi = vi
                best_vj = vj
                mode = 0
                break
        if best_m2 >= m - 1e-15:
            for sc in (eps, eps * 0.5):
                vi = xi - sc * ti
                nvi = float(np.linalg.norm(vi))
                if nvi == 0 or not np.isfinite(nvi):
                    continue
                vi /= nvi
                di = X @ vi
                di[i] = -2.0
                mc = base_i_only
                a = float(np.max(di))
                if a > mc: mc = a
                if mc < best_m2 - 1e-15:
                    best_m2 = mc
                    best_vi = vi
                    best_vj = None
                    mode = 1
                    break
            for sc in (eps, eps * 0.5):
                vj = xj - sc * tj
                nvj = float(np.linalg.norm(vj))
                if nvj == 0 or not np.isfinite(nvj):
                    continue
                vj /= nvj
                dj = X @ vj
                dj[j] = -2.0
                mc = base_j_only
                a = float(np.max(dj))
                if a > mc: mc = a
                if mc < best_m2 - 1e-15:
                    best_m2 = mc
                    best_vi = None
                    best_vj = vj
                    mode = 2
                    break
        if best_vi is not None or best_vj is not None:
            if mode == 0:
                X[i] = best_vi; X[j] = best_vj
            elif mode == 1:
                X[i] = best_vi
            else:
                X[j] = best_vj
            if best_m2 < bestV:
                bestV = best_m2
            eps *= 1.02
            if eps > 2e-6:
                eps = 2e-6
        else:
            eps *= 0.82
            if eps < 2e-10:
                k = i if (it % 2 == 0) else j
                rnd = np.random.randn(d)
                rnd -= float(rnd @ X[k]) * X[k]
                rn = float(np.linalg.norm(rnd))
                if rn > 1e-12:
                    rnd /= rn
                    X[k] = X[k] + 5e-10 * rnd
                    X[k] /= float(np.linalg.norm(X[k]))
                eps = 2e-10
        if it % 200 == 0 and time.time() > t_end:
            break
    return X, bestV, eps

def solve():
    import time as _tm
    t0 = _tm.time()
    DEAD = t0 + 172.0
    np.random.seed(0)
    rng = np.random.default_rng(0)
    # load incumbent
    cands = []
    try:
        b = np.load("best.npy")
        cands.append(np.asarray(b, dtype=np.float64))
    except Exception:
        pass
    try:
        r = np.load("record.npy")
        cands.append(np.asarray(r, dtype=np.float64))
    except Exception:
        pass
    bestX = None
    bestV = float("inf")
    for cc in cands:
        try:
            if cc.shape != (453, 5):
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
        bestX = rng.standard_normal((453, 5))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        bestV = _max_cos(bestX)
    n, d = bestX.shape
    incumbentX = bestX.copy()
    incumbentV = bestV

    # ---------- symmetric C151 search ----------
    sym_best_X = None
    sym_best_V = float("inf")
    sym_best_a = None
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
                a2 = (a * BASE1) + p2
                c1 = np.cos(a1); s1 = np.sin(a1)
                c2 = np.cos(a2); s2 = np.sin(a2)
                s = k*151; e = s+151
                X[s:e, 0] = r1 * c1
                X[s:e, 1] = r1 * s1
                X[s:e, 2] = r2 * c2
                X[s:e, 3] = r2 * s2
                X[s:e, 4] = w
            # renorm for safety
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            return X
        def sym_obj(vec, a):
            try:
                X = build_sym(vec, a)
                return _max_cos(X)
            except Exception:
                return 2.0
        bounds = [(-0.9, 0.9), (0.05, 1.52), (0.0, TWO_PI), (0.0, TWO_PI)] * 3
        for a in [1, 10, 20, 37, 50]:
            if _tm.time() > t0 + 48.0:
                break
            try:
                res = differential_evolution(
                    lambda v: sym_obj(v, a),
                    bounds, maxiter=12, popsize=10, seed=1000+a,
                    polish=False, updating='deferred', workers=1,
                    disp=False, tol=1e-9)
                v0 = np.asarray(res.x, dtype=np.float64)
                curV = float(res.fun)
            except Exception:
                continue
            # smooth LSE refinement with L-BFGS-B
            try:
                beta = 800.0
                def lse_obj(vec):
                    X = build_sym(vec, a)
                    G = X @ X.T
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
                              options={'maxiter': 80, 'ftol': 1e-12, 'gtol': 1e-8})
                if r2.success or r2.fun < 2.0:
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
                sym_best_a = a
            if _tm.time() > DEAD - 120.0:
                break
    except Exception:
        pass

    # partially symmetrised incumbent: snap rattlers toward symmetric template if useful
    # build a noisy broken-symmetry version of sym optimum as alternative basin
    altX = None
    altV = float("inf")
    if sym_best_X is not None and np.all(np.isfinite(sym_best_X)):
        try:
            # break symmetry with small noise
            tmp = sym_best_X + 0.0025 * rng.standard_normal((n, d))
            tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
            # quick anneal to leave symmetric manifold
            tmp = _softmax_relax(tmp, beta=400.0, steps=80, lr=0.002, mu=0.85, t_end=min(_tm.time()+6.0, DEAD-110))
            altX = tmp
            altV = _max_cos(altX)
        except Exception:
            altX = None
    # also make partially symmetrised incumbent: replace worst 30 rattlers by sym points far from all
    try:
        if sym_best_X is not None and altX is not None:
            G0 = incumbentX @ incumbentX.T
            np.fill_diagonal(G0, -2.0)
            per = np.max(G0, axis=1)
            worst = np.argsort(per)[-30:]
            cand_pool = sym_best_X.copy()
            # greedily pick pool points with small dot to incumbent (excluding worst)
            keep = np.ones(n, dtype=bool); keep[worst] = False
            Xkeep = incumbentX[keep]
            dots = cand_pool @ Xkeep.T
            md = np.max(dots, axis=1)
            order = np.argsort(md)[:30]
            Xhyb = incumbentX.copy()
            Xhyb[worst] = cand_pool[order]
            Xhyb /= np.linalg.norm(Xhyb, axis=1, keepdims=True)
            Vhyb = _max_cos(Xhyb)
            # keep hybrid as second alt if better than alt
            if Vhyb < altV:
                altX = Xhyb; altV = Vhyb
    except Exception:
        pass

    # ---------- polish incumbent briefly (keep record basin competitive) ----------
    try:
        if _tm.time() < DEAD - 100.0:
            X1, V1, _ = _micro_shave(incumbentX.copy(), 5e-7, min(t0+58.0, DEAD-100.0), incumbentV)
            if V1 < bestV and np.all(np.isfinite(X1)):
                bestV = V1; bestX = X1.copy()
            incumbentX = bestX.copy(); incumbentV = bestV
    except Exception:
        pass

    # ---------- polish symmetric/alternative basin ----------
    sym_polished = None
    sym_polished_V = float("inf")
    if altX is not None:
        try:
            t_allow = min(_tm.time()+22.0, DEAD-85.0)
            if _tm.time() < t_allow:
                Xa = _softmax_relax(altX.copy(), beta=1500.0, steps=80, lr=0.0006, mu=0.9, t_end=t_allow)
                Xa, Va, _ = _micro_shave(Xa, 4e-7, t_allow, float("inf"))
                Va = _max_cos(Xa)
                sym_polished = Xa; sym_polished_V = Va
                if Va < bestV - 1e-14 and np.all(np.isfinite(Xa)):
                    bestV = Va; bestX = Xa.copy()
        except Exception:
            pass

    # ---------- joint basin hopping ----------
    it = 0
    hop_end = DEAD - 38.0
    # ensure we have time
    while _tm.time() < hop_end and _tm.time() < DEAD - 40.0:
        it += 1
        # alternate seeds: mostly best, sometimes symmetric polished
        if sym_polished is not None and (it % 4 == 0):
            X = sym_polished.copy()
            sigma = 0.0015
            X += sigma * rng.standard_normal((n, d))
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            try:
                X = _softmax_relax(X, beta=300.0, steps=60, lr=0.002, mu=0.85, t_end=hop_end)
                X = _softmax_relax(X, beta=1500.0, steps=40, lr=0.0006, mu=0.9, t_end=hop_end)
            except Exception:
                pass
            try:
                X, V, _ = _micro_shave(X, 4e-7, min(_tm.time()+4.0, hop_end), float("inf"))
            except Exception:
                continue
            V = _max_cos(X)
            if np.all(np.isfinite(X)) and V < sym_polished_V - 1e-14:
                sym_polished = X.copy(); sym_polished_V = V
            if np.all(np.isfinite(X)) and V < bestV - 1e-14:
                bestV = V; bestX = X.copy()
        elif (it % 5 == 0):
            X = bestX.copy()
            sigma = 0.004 if (it % 10 == 0) else 0.0015
            X += sigma * rng.standard_normal((n, d))
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            try:
                X = _softmax_relax(X, beta=300.0, steps=80, lr=0.002, mu=0.85, t_end=hop_end)
            except Exception:
                pass
            try:
                X, V, _ = _micro_shave(X, 4e-7, min(_tm.time()+4.0, hop_end), float("inf"))
            except Exception:
                continue
            V = _max_cos(X)
            if np.all(np.isfinite(X)) and V < bestV - 1e-14:
                bestV = V; bestX = X.copy()
        elif (it % 5 == 1) or (it % 5 == 2):
            X = bestX.copy()
            sigma = 2.5e-7 if (it % 5 == 1) else 8e-7
            X += sigma * rng.standard_normal((n, d))
            try:
                Gtmp = X @ X.T
                np.fill_diagonal(Gtmp, -2.0)
                flat = int(np.argmax(Gtmp))
                wi = flat // n; wj = flat % n
                X[wi] += sigma * 3.0 * rng.standard_normal(d)
                X[wj] += sigma * 3.0 * rng.standard_normal(d)
            except Exception:
                pass
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            try:
                X = _softmax_relax(X, beta=3000000.0, steps=30, lr=2.5e-7, mu=0.8, t_end=hop_end)
            except Exception:
                pass
            try:
                X, V, _ = _micro_shave(X, 3.5e-7, min(_tm.time()+5.0, hop_end), float("inf"))
            except Exception:
                continue
            V = _max_cos(X)
            if np.all(np.isfinite(X)) and V < bestV - 1e-14:
                bestV = V; bestX = X.copy()
        else:
            X = bestX.copy()
            try:
                Gtmp = X @ X.T
                np.fill_diagonal(Gtmp, -2.0)
                per = np.max(Gtmp, axis=1)
                w = int(np.argmax(per))
            except Exception:
                w = int(rng.integers(0, n))
            M = 8000
            cand = rng.standard_normal((M, d))
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            dots = cand @ X.T
            dots[:, w] = -2.0
            scores = np.max(dots, axis=1)
            bi = int(np.argmin(scores))
            X[w] = cand[bi]
            X[w] += 1e-7 * rng.standard_normal(d)
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            try:
                X = _softmax_relax(X, beta=800.0, steps=60, lr=0.001, mu=0.85, t_end=hop_end)
            except Exception:
                pass
            try:
                X, V, _ = _micro_shave(X, 4e-7, min(_tm.time()+4.0, hop_end), float("inf"))
            except Exception:
                continue
            V = _max_cos(X)
            if np.all(np.isfinite(X)) and V < bestV - 1e-14:
                bestV = V; bestX = X.copy()

    # ---------- final long polish ----------
    try:
        remain = DEAD - _tm.time() - 1.0
        if remain < 5.0:
            remain = 5.0
        if remain > 36.0:
            remain = 36.0
        XP, VP, _ = _micro_shave(bestX.copy(), 4e-7, _tm.time()+remain, bestV)
        if VP < bestV and np.all(np.isfinite(XP)):
            bestV = VP; bestX = XP.copy()
    except Exception:
        pass
    # strict normalize, guard nan
    try:
        bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
        bestX = np.where(np.isfinite(bestX), bestX, 0.0)
        nn = np.linalg.norm(bestX, axis=1, keepdims=True)
        nn[nn == 0] = 1.0
        # fix any zero row deterministically
        for i in range(n):
            if float(np.linalg.norm(bestX[i])) < 1e-12:
                bestX[i] = np.array([1.0, 0, 0, 0, 0])
        bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    except Exception:
        pass
    return bestX
