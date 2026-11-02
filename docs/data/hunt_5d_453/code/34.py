import time
import numpy as np

def solve():
    t0 = time.time()
    T_END = t0 + 172.0
    T_SYM = t0 + 28.0
    T_ANNEAL = t0 + 158.0
    rng = np.random.default_rng(0)
    np.random.seed(0)
    n, d = 453, 5
    N = 151
    JJ = np.arange(N)
    ANG = 2.0 * np.pi * JJ / float(N)
    iu_r, iu_c = np.triu_indices(n, 1)
    TAU_S = 0.003

    def max_cos(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        return float(np.max(G))

    # ---- load incumbent ----
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (453, 5):
                nn = np.linalg.norm(a, axis=1, keepdims=True)
                nn[nn == 0] = 1.0
                a = a / nn
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if len(cands) == 0:
        a = rng.standard_normal((n, d))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    vals = [(max_cos(c), c) for c in cands]
    vals.sort(key=lambda t: t[0])
    best_val = float(vals[0][0])
    best_X = vals[0][1].copy()

    # ---- symmetric family ----
    # params: [z0,s0,p20, z1,s1,p11,p21, z2,s2,p12,p22], phi10=0 fixed
    lb = np.array([-0.98, 0.02, 0.0, -0.98, 0.02, 0.0, 0.0, -0.98, 0.02, 0.0, 0.0])
    ub = np.array([0.98, 0.98, 2*np.pi, 0.98, 0.98, 2*np.pi, 2*np.pi, 0.98, 0.98, 2*np.pi, 2*np.pi])

    def gen_params(p, aval):
        z0, s0, p20, z1, s1, p11, p21, z2, s2, p12, p22 = p
        zs = [z0, z1, z2]
        ss = [s0, s1, s2]
        ph1 = [0.0, p11, p12]
        ph2 = [p20, p21, p22]
        X = np.empty((453, 5), dtype=np.float64)
        for k in range(3):
            z = float(np.clip(zs[k], -0.999, 0.999))
            r2 = max(0.0, 1.0 - z*z)
            r = np.sqrt(r2)
            s = float(np.clip(ss[k], 0.001, 0.999))
            rho1 = r * np.sqrt(s)
            rho2 = r * np.sqrt(1.0 - s)
            a1 = ANG + float(ph1[k])
            a2 = aval * ANG + float(ph2[k])
            blk = np.empty((N, 5))
            blk[:, 0] = rho1 * np.cos(a1)
            blk[:, 1] = rho1 * np.sin(a1)
            blk[:, 2] = rho2 * np.cos(a2)
            blk[:, 3] = rho2 * np.sin(a2)
            blk[:, 4] = z
            X[k*N:(k+1)*N, :] = blk
        # renormalize (safety)
        nn = np.linalg.norm(X, axis=1, keepdims=True)
        nn[nn == 0] = 1.0
        return X / nn

    def exact_of_params(p, aval):
        try:
            X = gen_params(p, aval)
        except Exception:
            return 2.0, None
        G = X @ X.T
        c = G[iu_r, iu_c]
        m = float(np.max(c))
        return m, X

    def smooth_of_params(p, aval, tau=TAU_S):
        X = gen_params(p, aval)
        G = X @ X.T
        c = G[iu_r, iu_c]
        m = float(np.max(c))
        # logsumexp
        e = np.exp((c - m) / tau)
        s = float(np.sum(e))
        if not np.isfinite(s) or s <= 0:
            return m
        return m + tau * np.log(s)

    best_sym_val = float("inf")
    best_sym_X = None
    best_sym_a = None
    try:
        from scipy.optimize import minimize as _min
        has_min = True
    except Exception:
        has_min = False

    avals = [1, 10, 20, 37, 50]
    for aval in avals:
        if time.time() > T_SYM - 1.0:
            break
        # random sampling
        NR = 220
        # vectorized per-sample loop (cheap Gram)
        top = []  # (val, params)
        for _ in range(NR):
            if time.time() > T_SYM - 1.0:
                break
            u = rng.random(11)
            p = lb + (ub - lb) * u
            m, _ = exact_of_params(p, aval)
            if not np.isfinite(m):
                continue
            if len(top) < 4:
                top.append((m, p.copy()))
                top.sort(key=lambda t: t[0])
            elif m < top[-1][0]:
                top[-1] = (m, p.copy())
                top.sort(key=lambda t: t[0])
        # update global sym best from samples
        for m, p in top:
            if m < best_sym_val:
                _, Xg = exact_of_params(p, aval)
                best_sym_val = m
                best_sym_X = Xg
                best_sym_a = aval
        # local polish top 2 with L-BFGS-B on smooth surrogate
        if has_min:
            for k in range(min(2, len(top))):
                if time.time() > T_SYM - 0.5:
                    break
                p0 = top[k][1]
                try:
                    def _obj(pp, _a=aval):
                        return smooth_of_params(pp, _a, tau=TAU_S)
                    res = _min(_obj, p0, method="L-BFGS-B", bounds=list(zip(lb, ub)),
                               options={"maxiter": 60, "maxfun": 500, "ftol": 1e-12, "gtol": 1e-8})
                    pf = np.asarray(res.x, dtype=np.float64)
                    m2, X2 = exact_of_params(pf, aval)
                    if m2 < best_sym_val:
                        best_sym_val = m2
                        best_sym_X = X2
                        best_sym_a = aval
                    # also try exact value even if smooth worse
                except Exception:
                    continue
    # compare stats (contacts/rattlers) implicitly via max; keep sym as second basin
    sym_start = None
    if best_sym_X is not None and np.all(np.isfinite(best_sym_X)):
        # break symmetry with small noise
        sym_start = best_sym_X.copy()
        nz = rng.standard_normal(sym_start.shape) * 0.008
        # tangential
        dot = np.sum(nz * sym_start, axis=1, keepdims=True)
        nz = nz - dot * sym_start
        sym_start = sym_start + nz
        sym_start /= np.linalg.norm(sym_start, axis=1, keepdims=True)
        sv = max_cos(sym_start)
        # keep track; don't overwrite best unless better
        if sv < best_val:
            best_val = sv
            best_X = sym_start.copy()
    else:
        sv = float("inf")

    # ---- full-space softmax annealing ----
    def anneal_run(X, steps, tau0, tau1, lr0, lr1, mu, t_dead):
        nonlocal_best = [best_val, best_X]
        # use closure variables via lists
        x = X.copy()
        v = np.zeros_like(x)
        ltau = np.log(tau1 / tau0)
        llr = np.log(lr1 / lr0)
        bv = float(nonlocal_best[0])
        bx = nonlocal_best[1]
        for it in range(steps):
            if it % 40 == 0 and time.time() > t_dead:
                break
            frac = it / max(steps - 1, 1)
            tau = tau0 * np.exp(ltau * frac)
            lr = lr0 * np.exp(llr * frac)
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            if m < bv:
                bv = m
                bx = x.copy()
                nonlocal_best[0] = bv
                # update outer best via mutable? do below
            # softmax weights
            # clip (G-m)/tau to avoid overflow (<=0)
            P = np.exp((G - m) / tau)
            # P diag ~ exp((-2-m)/tau) tiny; zero it
            np.fill_diagonal(P, 0.0)
            Grad = P @ x
            dot = np.sum(Grad * x, axis=1, keepdims=True)
            Gt = Grad - dot * x
            # normalize step by mean row norm to keep lr stable
            # momentum
            v = mu * v + (1.0 - mu) * Gt
            # scale: divide by max row norm occasionally
            x = x - lr * v
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            if m > bv + 0.02:
                # reset diverged
                x = bx.copy()
                v *= 0.5
        return bx, bv

    # helper to update global best
    def consider(Xc):
        nonlocal best_val, best_X
        vv = max_cos(Xc)
        if vv < best_val:
            best_val = vv
            best_X = Xc.copy()
        return vv

    # initial anneals of both basins
    if time.time() < T_ANNEAL:
        X0 = best_X.copy()
        # short anneal incumbent
        steps = 1200
        tau0, tau1 = 0.008, 0.0008
        lr0, lr1 = 6e-4, 6e-5
        # inline anneal with global update
        x = X0.copy()
        v = np.zeros_like(x)
        for it in range(steps):
            if it % 50 == 0 and time.time() > T_ANNEAL:
                break
            frac = it / max(steps - 1, 1)
            tau = tau0 * np.exp(np.log(tau1/tau0) * frac)
            lr = lr0 * np.exp(np.log(lr1/lr0) * frac)
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            if m < best_val:
                best_val = m
                best_X = x.copy()
            P = np.exp((G - m) / tau)
            np.fill_diagonal(P, 0.0)
            Grad = P @ x
            Gt = Grad - np.sum(Grad*x, axis=1, keepdims=True)*x
            v = 0.8*v + 0.2*Gt
            x = x - lr*v
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            if m > best_val + 0.02:
                x = best_X.copy()
                v *= 0.5
    if sym_start is not None and time.time() < T_ANNEAL:
        x = sym_start.copy()
        v = np.zeros_like(x)
        steps = 1000
        tau0, tau1 = 0.006, 0.0006
        lr0, lr1 = 5e-4, 5e-5
        for it in range(steps):
            if it % 50 == 0 and time.time() > T_ANNEAL:
                break
            frac = it / max(steps - 1, 1)
            tau = tau0 * np.exp(np.log(tau1/tau0) * frac)
            lr = lr0 * np.exp(np.log(lr1/lr0) * frac)
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            if m < best_val:
                best_val = m
                best_X = x.copy()
            P = np.exp((G - m) / tau)
            np.fill_diagonal(P, 0.0)
            Grad = P @ x
            Gt = Grad - np.sum(Grad*x, axis=1, keepdims=True)*x
            v = 0.8*v + 0.2*Gt
            x = x - lr*v
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            if m > best_val + 0.02:
                x = best_X.copy()
                v *= 0.5

    # basin-hopping: shake + hole surgery + re-anneal
    sigmas = [0.002, 0.005, 0.010, 0.020]
    cyc = 0
    while time.time() < T_ANNEAL:
        cyc += 1
        # build shaken candidate from best
        y = best_X.copy()
        G = y @ y.T
        np.fill_diagonal(G, -2.0)
        flat = int(np.argmax(G))
        wi = flat // n
        wj = flat % n
        sigma = sigmas[cyc % len(sigmas)]
        # shake worst pair neighbourhood + random subset
        idx_set = {int(wi), int(wj)}
        # neighbours of worst pair
        for kk in (int(wi), int(wj)):
            row = G[kk].copy()
            row[kk] = -2.0
            topn = np.argpartition(row, -6)[-6:]
            for t_ in topn:
                idx_set.add(int(t_))
        # random points
        for _ in range(20):
            idx_set.add(int(rng.integers(0, n)))
        idx_arr = np.array(list(idx_set), dtype=int)
        noise = rng.standard_normal((len(idx_arr), d)) * sigma
        for k2, ii in enumerate(idx_arr):
            vv = noise[k2] - float(np.dot(noise[k2], y[ii])) * y[ii]
            y[ii] = y[ii] + vv
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        # occasional hole surgery: relocate single worst point to largest hole
        if (cyc % 4) == 0 and time.time() < T_ANNEAL - 5.0:
            try:
                # row maxima
                Gr = y @ y.T
                np.fill_diagonal(Gr, -2.0)
                rmax = Gr.max(axis=1)
                worst = int(np.argmax(rmax))
                # candidate holes
                C = rng.standard_normal((6000, d))
                C /= np.linalg.norm(C, axis=1, keepdims=True)
                # max cos to y excluding worst: compute dots
                D = C @ y.T  # 6000x453
                # ignore column worst by setting -2
                D[:, worst] = -2.0
                cmax = D.max(axis=1)
                bi = int(np.argmin(cmax))
                y[worst] = C[bi]
            except Exception:
                pass
        # short re-anneal
        steps = 450
        # vary tau/lr slightly per cycle
        tau0 = 0.004 if (cyc % 2 == 0) else 0.007
        tau1 = 0.0006
        lr0 = 3e-4 if (cyc % 3 == 0) else 5e-4
        lr1 = 5e-5
        mu = 0.8
        x = y
        v = np.zeros_like(x)
        ltau = np.log(tau1 / tau0)
        llr = np.log(lr1 / lr0)
        for it in range(steps):
            if it % 60 == 0 and time.time() > T_ANNEAL:
                break
            frac = it / max(steps - 1, 1)
            tau = tau0 * np.exp(ltau * frac)
            lr = lr0 * np.exp(llr * frac)
            G2 = x @ x.T
            np.fill_diagonal(G2, -2.0)
            m2 = float(np.max(G2))
            if m2 < best_val:
                best_val = m2
                best_X = x.copy()
            P = np.exp((G2 - m2) / tau)
            np.fill_diagonal(P, 0.0)
            Grad = P @ x
            Gt = Grad - np.sum(Grad*x, axis=1, keepdims=True)*x
            v = mu*v + (1.0-mu)*Gt
            x = x - lr*v
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            if m2 > best_val + 0.015:
                x = best_X.copy()
                v *= 0.5
                break

    # final micro worst-pair shaving
    x = best_X.copy()
    cur = best_val
    # tiny tangential moves on worst pair
    it = 0
    while time.time() < T_END - 0.5:
        it += 1
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        if m < cur:
            cur = m
            if m < best_val:
                best_val = m
                best_X = x.copy()
        flat = int(np.argmax(G))
        i = flat // n
        j = flat % n
        # directions away from each other tangentially
        cij = float(np.dot(x[i], x[j]))
        ti = x[j] - cij * x[i]
        nti = float(np.linalg.norm(ti))
        if nti < 1e-12 or not np.isfinite(nti):
            r_ = rng.standard_normal(d)
            r_ = r_ - float(np.dot(r_, x[i])) * x[i]
            nti = float(np.linalg.norm(r_)) + 1e-12
            ti = r_ / nti
        else:
            ti = ti / nti
        tj = x[i] - cij * x[j]
        ntj = float(np.linalg.norm(tj))
        if ntj < 1e-12 or not np.isfinite(ntj):
            r_ = rng.standard_normal(d)
            r_ = r_ - float(np.dot(r_, x[j])) * x[j]
            ntj = float(np.linalg.norm(r_)) + 1e-12
            tj = r_ / ntj
        else:
            tj = tj / ntj
        # scale cycles
        sc = 2e-4 if (it % 3 == 0) else (5e-5 if (it % 3 == 1) else 1e-5)
        improved = False
        for s_ in (sc, sc*0.5):
            yi = x[i] - s_ * ti
            yi /= np.linalg.norm(yi)
            yj = x[j] - s_ * tj
            yj /= np.linalg.norm(yj)
            oxi = x[i].copy(); oxj = x[j].copy()
            x[i] = yi; x[j] = yj
            G2 = x @ x.T
            np.fill_diagonal(G2, -2.0)
            m2 = float(np.max(G2))
            if m2 < m - 1e-12:
                if m2 < best_val:
                    best_val = m2
                    best_X = x.copy()
                cur = m2
                improved = True
                break
            else:
                x[i] = oxi; x[j] = oxj
        if not improved and (it % 20) == 0:
            # tiny random jitter on a random point to escape
            k_ = int(rng.integers(0, n))
            r_ = rng.standard_normal(d) * 3e-6
            r_ = r_ - float(np.dot(r_, x[k_])) * x[k_]
            xn = x[k_] + r_
            xn /= np.linalg.norm(xn)
            x[k_] = xn

    best_X = best_X / np.linalg.norm(best_X, axis=1, keepdims=True)
    return best_X
