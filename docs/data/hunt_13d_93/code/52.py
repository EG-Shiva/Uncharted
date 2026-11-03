import time
import numpy as np

def solve():
    t0 = time.time()
    BUDGET = 168.0
    n, d = 93, 13
    rng = np.random.default_rng(12345)

    def _norm(X):
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        return X / nr

    def _maxcos(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        return float(np.max(G))

    iu0, iu1 = np.triu_indices(n, 1)

    base_best = None
    base_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (n, d):
                continue
            a = _norm(a)
            v = _maxcos(a)
            if v < base_val:
                base_val = v
                base_best = a.copy()
        except Exception:
            pass
    if base_best is None:
        base_best = _norm(rng.standard_normal((n, d)))
        base_val = _maxcos(base_best)

    # symmetric parametrization: 3 x C31 orbits
    A_mult = np.array([1, 5, 7, 11, 13, 17], dtype=np.float64)
    K31 = 31
    ks = np.arange(K31, dtype=np.float64)
    ALPHA = (2.0 * np.pi / 31.0) * (ks[:, None] * A_mult[None, :])  # (31,6)

    def build_X(params):
        params = np.asarray(params, dtype=np.float64)
        X = np.empty((93, 13), dtype=np.float64)
        for o in range(3):
            v = params[o*13:o*13+7]
            ph = params[o*13+7:o*13+13]
            nv = float(np.sqrt(np.dot(v, v)))
            if nv < 1e-12:
                w = np.full(7, 1.0/np.sqrt(7))
            else:
                w = v / nv
            r = np.abs(w[:6])
            c = float(w[6])
            TH = ALPHA + ph[None, :]
            CT = np.cos(TH); ST = np.sin(TH)
            blk = np.empty((31, 13), dtype=np.float64)
            for j in range(6):
                blk[:, 2*j] = r[j]*CT[:, j]
                blk[:, 2*j+1] = r[j]*ST[:, j]
            blk[:, 12] = c
            X[o*31:(o+1)*31, :] = blk
        return _norm(X)

    def obj_grad(params, tau):
        params = np.asarray(params, dtype=np.float64)
        ws = []
        rs = []
        cs = []
        THs = []
        CTS = []
        STS = []
        for o in range(3):
            v = params[o*13:o*13+7]
            ph = params[o*13+7:o*13+13]
            nv = float(np.sqrt(np.dot(v, v) + 1e-300))
            if nv < 1e-12:
                w = np.full(7, 1.0/np.sqrt(7))
                nv = 1e-12
            else:
                w = v / nv
            ws.append((w, v, nv))
            r = np.abs(w[:6]); c = float(w[6])
            rs.append(r); cs.append(c)
            TH = ALPHA + ph[None, :]
            THs.append(TH)
            CTS.append(np.cos(TH)); STS.append(np.sin(TH))
        X = np.empty((93, 13), dtype=np.float64)
        for o in range(3):
            r = rs[o]; c = cs[o]
            CT = CTS[o]; ST = STS[o]
            blk = np.empty((31, 13), dtype=np.float64)
            for j in range(6):
                blk[:, 2*j] = r[j]*CT[:, j]
                blk[:, 2*j+1] = r[j]*ST[:, j]
            blk[:, 12] = c
            X[o*31:(o+1)*31, :] = blk
        G = X @ X.T
        c_all = G[iu0, iu1]
        m = float(np.max(c_all))
        E = np.exp((c_all - m) / tau)
        S = float(np.sum(E))
        if not np.isfinite(S) or S <= 0:
            return float(m), np.zeros_like(params)
        L = m + tau*np.log(S)
        # grad wrt X
        Wmat = np.zeros((n, n), dtype=np.float64)
        Wmat[iu0, iu1] = E
        Wmat[iu1, iu0] = E
        gradX = (Wmat @ X) / S
        grad = np.zeros_like(params)
        for o in range(3):
            w, v, nv = ws[o]
            r = rs[o]
            CT = CTS[o]; ST = STS[o]
            Gblk = gradX[o*31:(o+1)*31, :]
            dLdr = np.zeros(6); dLdphi = np.zeros(6)
            for j in range(6):
                gx = Gblk[:, 2*j]; gy = Gblk[:, 2*j+1]
                cj = CT[:, j]; sj = ST[:, j]
                dLdr[j] = float(np.sum(gx*cj + gy*sj))
                dLdphi[j] = float(np.sum(r[j]*(-gx*sj + gy*cj)))
            dLdc = float(np.sum(Gblk[:, 12]))
            sgn = np.sign(w[:6])
            sgn[sgn == 0] = 1.0
            h = np.empty(7, dtype=np.float64)
            h[:6] = dLdr*sgn
            h[6] = dLdc
            wdot = float(np.dot(w, h))
            gv = (h - wdot*w) / max(nv, 1e-12)
            grad[o*13:o*13+7] = gv
            grad[o*13+7:o*13+13] = dLdphi
        return float(L), grad

    # annealed L-BFGS-B over symmetric params
    try:
        from scipy.optimize import minimize
        has_sp = True
    except Exception:
        has_sp = False

    best_sym_val = base_val
    best_sym_params = None
    best_sym_X = None

    if has_sp:
        taus = [0.02, 0.01, 0.005, 0.002]
        n_restart = 70
        for ri in range(n_restart):
            if time.time() - t0 > 105.0:
                break
            vr = rng.standard_normal(21)
            phr = rng.uniform(-np.pi, np.pi, size=18)
            p0 = np.empty(39, dtype=np.float64)
            for o in range(3):
                p0[o*13:o*13+7] = vr[o*7:(o+1)*7]
                p0[o*13+7:o*13+13] = phr[o*6:(o+1)*6]
            x = p0
            try:
                for ti, tau in enumerate(taus):
                    if time.time() - t0 > 108.0:
                        break
                    def fun(p, tau=tau):
                        f, g = obj_grad(p, tau)
                        return f, g
                    res = minimize(fun, x, method='L-BFGS-B', jac=True,
                                   options={'maxiter': 120, 'ftol': 1e-12, 'gtol': 1e-8})
                    x = np.asarray(res.x, dtype=np.float64)
                    if time.time() - t0 > 108.0:
                        break
            except Exception:
                continue
            try:
                Xs = build_X(x)
                v = _maxcos(Xs)
            except Exception:
                continue
            if v < best_sym_val:
                best_sym_val = float(v)
                best_sym_params = x.copy()
                best_sym_X = Xs.copy()
    # fallback if symmetric failed
    if best_sym_X is None:
        best_sym_X = base_best.copy()
        best_sym_val = float(base_val)

    # full-dimensional polish helpers
    def _smooth_opt(X, steps, T0, T1, lr0, lr1, mu):
        nn, dd = X.shape
        V = np.zeros_like(X)
        W = np.empty((nn, nn), dtype=X.dtype)
        for k in range(steps):
            f = k / max(steps-1, 1)
            T = T0 * (T1/T0)**f
            lr = lr0 * (lr1/lr0)**f
            G = X @ X.T
            np.fill_diagonal(G, -10.0)
            m = float(G.max())
            # use upper indices via full matrix for speed
            wfull = np.exp((G - m)/T)
            # zero diagonal weight
            np.fill_diagonal(wfull, 0.0)
            s = float(wfull.sum())
            if not np.isfinite(s) or s == 0:
                continue
            wfull /= s
            grad = wfull @ X
            dot = np.einsum('ij,ij->i', grad, X)[:, None]
            grad_t = grad - dot*X
            V = mu*V + (1.0-mu)*grad_t
            dotv = np.einsum('ij,ij->i', V, X)[:, None]
            Vt = V - dotv*X
            X = X - lr*Vt
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            if time.time() - t0 > BUDGET - 4.0:
                break
        return X

    def _hard_polish(X, iters, step0, K):
        best = _maxcos(X)
        step = step0
        grad = np.empty_like(X)
        for _ in range(iters):
            if time.time() - t0 > BUDGET - 2.0:
                break
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            # top-K pairs via partition on flattened? use iu
            c = G[iu0, iu1]
            if K < len(c):
                idx = np.argpartition(-c, K)[:K]
            else:
                idx = np.arange(len(c))
            I = iu0[idx]; J = iu1[idx]
            grad.fill(0.0)
            np.add.at(grad, I, X[J])
            np.add.at(grad, J, X[I])
            grad /= float(len(idx))
            dot = np.einsum('ij,ij->i', grad, X)[:, None]
            gt = grad - dot*X
            ng = float(np.linalg.norm(gt))
            if ng == 0 or not np.isfinite(ng):
                step *= 0.5
                if step < 1e-13:
                    break
                continue
            Xt = X - step*gt
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            v = _maxcos(Xt)
            if v < best:
                X = Xt
                best = v
                step = min(step*1.3, 0.02)
            else:
                step *= 0.5
                if step < 1e-13:
                    break
        return X, best

    overall_best = base_best.copy()
    overall_val = float(base_val)
    if best_sym_val < overall_val:
        overall_best = best_sym_X.copy()
        overall_val = float(best_sym_val)

    # build candidate list: symmetric + perturbations + base
    cands = []
    cands.append(best_sym_X.copy())
    # perturbations
    scales = [1e-4, 1e-3, 1e-2]
    for s in scales:
        for rep in range(3):
            if time.time() - t0 > BUDGET - 25.0:
                break
            pert = best_sym_X + s*rng.standard_normal(best_sym_X.shape)
            pert = _norm(pert)
            cands.append(pert)
    # also include base_best perturbed slightly to not lose record
    cands.append(base_best.copy())

    # polish candidates (quick then keep best)
    polished_best = overall_best
    polished_val = overall_val
    for ci, cand in enumerate(cands):
        if time.time() - t0 > BUDGET - 3.0:
            break
        X = cand.copy()
        # lighter polish for many, heavier for first
        if ci == 0:
            steps = 2200
        else:
            steps = 1400
        X = _smooth_opt(X, steps=steps, T0=0.004, T1=0.00015, lr0=0.05, lr1=0.004, mu=0.8)
        X, v = _hard_polish(X, iters=250, step0=0.003, K=250)
        if v < polished_val - 1e-12:
            polished_val = float(v)
            polished_best = X.copy()
        if time.time() - t0 > BUDGET - 3.0:
            break

    # final tight polish on winner
    if time.time() - t0 < BUDGET - 2.0:
        X = polished_best
        X, polished_val = _hard_polish(X, iters=600, step0=0.002, K=200)
        if time.time() - t0 < BUDGET - 1.0:
            X = _smooth_opt(X, steps=600, T0=3e-4, T1=3e-5, lr0=0.01, lr1=0.0015, mu=0.8)
            X, polished_val = _hard_polish(X, iters=600, step0=0.001, K=150)
        polished_best = X

    polished_best = _norm(polished_best)
    return polished_best
