import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _rand_rot(rng):
    A = rng.randn(4, 4)
    Q, R = np.linalg.qr(A)
    d = np.diag(R)
    Q = Q * np.sign(d + 1e-12)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q

def _lp_polish(x, deadline, r0=0.004, cap=1400):
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x, 1.0
    radius = r0
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        try:
            G = x @ x.T
            c = G[iu0, iu1]
            m = float(c.max())
            sel = c > m - max(20.0 * radius, 1e-9)
            cnt = int(np.count_nonzero(sel))
            if cnt == 0:
                sel[np.argmax(c)] = True
                cnt = 1
            if cnt > cap:
                idx = np.argsort(-c)[:cap]
                sel2 = np.zeros_like(sel, dtype=bool)
                sel2[idx] = True
                sel = sel2
            I = iu0[sel]; J = iu1[sel]
            csel = c[sel]
            p = len(I)
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.empty((p, 2 * d + 1), dtype=int)
            ar = np.arange(d)
            cols[:, :d] = I[:, None] * d + ar
            cols[:, d:2 * d] = J[:, None] * d + ar
            cols[:, 2 * d] = n * d
            cols = cols.ravel()
            vals = np.empty((p, 2 * d + 1), dtype=float)
            vals[:, :d] = x[J]
            vals[:, d:2 * d] = x[I]
            vals[:, 2 * d] = -1.0
            vals = vals.ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
            if res.status != 0:
                radius *= 0.5
                continue
            y = x + res.x[:-1].reshape(n, d)
            y /= np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
            v = _max_cos(y)
            if v < best:
                x, best = y, v
                radius = min(radius * 1.5, 0.05)
            else:
                radius *= 0.5
        except Exception:
            radius *= 0.5
            continue
    return x, best

def _full_gd(x0, steps, T0, T1, lr0, mu, rng, noise0=0.0, deadline=None):
    x = x0.copy()
    n, d = x.shape
    v = np.zeros_like(x)
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    bv = float(G.max())
    bx = x.copy()
    lratio = np.log(T1 / T0) if T0 > 0 and T1 > 0 else 0.0
    for k in range(steps):
        if deadline is not None and (k & 127) == 0 and time.time() > deadline:
            break
        frac = k / max(1, steps - 1)
        T = T0 * np.exp(lratio * frac)
        if T < 1e-12:
            T = 1e-12
        lr = lr0 * (1.0 - 0.6 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if S < 1e-300 or not np.isfinite(S):
            S = 1e-300
        grad = (E @ x) / S
        grad -= (grad * x).sum(axis=1, keepdims=True) * x
        v = mu * v + (1.0 - mu) * grad
        x = x - lr * v
        if noise0 > 0:
            amp = noise0 * (1.0 - frac)
            if amp > 0:
                nz = rng.randn(n, d) * amp
                nz -= (nz * x).sum(axis=1, keepdims=True) * x
                x = x + nz
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        if m < bv:
            bv = m
            bx = x.copy()
    return bx, bv

def _expand_cyclic(thetas, alphas, betas, K, m, free_pts):
    L = len(thetas)
    ct = np.cos(thetas); st = np.sin(thetas)
    nstd = L * K
    X = np.empty((nstd + (free_pts.shape[0] if free_pts is not None else 0), 4))
    twopi = 2 * np.pi
    for j in range(L):
        c = float(ct[j]); s = float(st[j])
        a0 = float(alphas[j]); b0 = float(betas[j])
        for kk in range(K):
            a = a0 + twopi * kk / K
            b = b0 + twopi * m * kk / K
            X[j * K + kk, 0] = c * np.cos(a)
            X[j * K + kk, 1] = c * np.sin(a)
            X[j * K + kk, 2] = s * np.cos(b)
            X[j * K + kk, 3] = s * np.sin(b)
    if free_pts is not None and free_pts.shape[0] > 0:
        F = free_pts / np.maximum(np.linalg.norm(free_pts, axis=1, keepdims=True), 1e-12)
        X[nstd:] = F
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    return X

def _sym_anneal(thetas0, alphas0, betas0, free0, K, m, steps, T0, T1, lr, mu, deadline=None):
    thetas = thetas0.copy(); alphas = alphas0.copy(); betas = betas0.copy()
    free = free0.copy()
    L = len(thetas)
    R = free.shape[0]
    vth = np.zeros(L); val = np.zeros(L); vbe = np.zeros(L)
    vfr = np.zeros_like(free)
    twopi = 2 * np.pi
    # best tracking via expanded
    best_val = 1e9
    best = (thetas.copy(), alphas.copy(), betas.copy(), free.copy())
    lratio = np.log(T1 / T0) if T0 > 0 and T1 > 0 else 0.0
    kk_arr = np.arange(K)
    for k in range(steps):
        if deadline is not None and (k & 127) == 0 and time.time() > deadline:
            break
        frac = k / max(1, steps - 1)
        T = T0 * np.exp(lratio * frac)
        if T < 8e-5:
            T = 8e-5
        lr_k = lr * (1.0 - 0.5 * frac)
        ct = np.cos(thetas); st = np.sin(thetas)
        X = _expand_cyclic(thetas, alphas, betas, K, m, free)
        n = X.shape[0]
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        cur_m = float(G.max())
        if cur_m < best_val:
            best_val = cur_m
            best = (thetas.copy(), alphas.copy(), betas.copy(), free.copy())
        E = np.exp((G - cur_m) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if not np.isfinite(S) or S < 1e-300:
            S = 1e-300
        Grad = (E @ X) / S
        Grad -= (Grad * X).sum(axis=1, keepdims=True) * X
        gth = np.zeros(L); gal = np.zeros(L); gbe = np.zeros(L)
        for j in range(L):
            c = float(ct[j]); s = float(st[j])
            a0 = float(alphas[j]); b0 = float(betas[j])
            acc_th = 0.0; acc_al = 0.0; acc_be = 0.0
            for t, kk in enumerate(kk_arr):
                idx = j * K + t
                g = Grad[idx]
                a = a0 + twopi * kk / K
                b = b0 + twopi * m * kk / K
                ca = np.cos(a); sa = np.sin(a); cb = np.cos(b); sb = np.sin(b)
                dth0 = -s * ca; dth1 = -s * sa; dth2 = c * cb; dth3 = c * sb
                acc_th += g[0]*dth0 + g[1]*dth1 + g[2]*dth2 + g[3]*dth3
                dal0 = -c * sa; dal1 = c * ca
                acc_al += g[0]*dal0 + g[1]*dal1
                dbe2 = -s * sb; dbe3 = s * cb
                acc_be += g[2]*dbe2 + g[3]*dbe3
            gth[j] = acc_th; gal[j] = acc_al; gbe[j] = acc_be
        if R > 0:
            gfr = Grad[L*K:]
        else:
            gfr = vfr
        vth = mu * vth + (1.0 - mu) * gth
        val = mu * val + (1.0 - mu) * gal
        vbe = mu * vbe + (1.0 - mu) * gbe
        # scale phase steps by radius to avoid freezing near poles
        csc = 1.0 / np.maximum(ct**2, 0.03)
        ssc = 1.0 / np.maximum(st**2, 0.03)
        csc = np.minimum(csc, 8.0); ssc = np.minimum(ssc, 8.0)
        thetas = thetas - lr_k * vth
        alphas = alphas - lr_k * np.minimum(csc, 3.0) * val * 0.5
        betas = betas - lr_k * np.minimum(ssc, 3.0) * vbe * 0.5
        thetas = np.clip(thetas, 0.06, 1.5108)
        if R > 0:
            vfr = mu * vfr + (1.0 - mu) * gfr
            # project free velocity to tangent
            Fn = free / np.maximum(np.linalg.norm(free, axis=1, keepdims=True), 1e-12)
            vfr = vfr - (vfr * Fn).sum(axis=1, keepdims=True) * Fn
            free = free - lr_k * 1.2 * vfr
            free /= np.maximum(np.linalg.norm(free, axis=1, keepdims=True), 1e-12)
    thb, alb, beb, frb = best
    Xb = _expand_cyclic(thb, alb, beb, K, m, frb)
    return Xb, best_val

def _lbfgs_smooth(x_init, deadline, taus=(0.003, 0.001), iters=(80, 80)):
    x = x_init / np.maximum(np.linalg.norm(x_init, axis=1, keepdims=True), 1e-12)
    n, d = x.shape
    iu0, iu1 = np.triu_indices(n, 1)
    Np = len(iu0)
    y = x.ravel().copy()
    for tau, maxiter in zip(taus, iters):
        if time.time() > deadline - 0.4:
            break
        def fg(yflat, _tau=tau):
            Y = yflat.reshape(n, d)
            rn = np.linalg.norm(Y, axis=1, keepdims=True)
            rn = np.maximum(rn, 1e-12)
            X = Y / rn
            G = X @ X.T
            g = G[iu0, iu1]
            m = float(g.max())
            e = np.exp((g - m) / _tau)
            s = float(e.sum())
            if s <= 0 or not np.isfinite(s):
                return float(m), np.zeros_like(yflat)
            L = m + _tau * np.log(s / Np)
            w = e / s
            W = np.zeros((n, n))
            W[iu0, iu1] = w
            W = W + W.T
            GX = W @ X
            dot = (GX * X).sum(axis=1, keepdims=True)
            GY = (GX - dot * X) / rn
            return float(L), GY.ravel()
        try:
            res = minimize(fg, y, method="L-BFGS-B", jac=True,
                           options={"maxiter": int(maxiter), "maxfun": int(maxiter * 6),
                                    "ftol": 1e-14, "gtol": 1e-9})
            y = res.x
            Yt = y.reshape(n, d)
            nt = np.linalg.norm(Yt, axis=1, keepdims=True)
            nt[nt < 1e-12] = 1.0
            y = (Yt / nt).ravel()
        except Exception:
            break
    Xf = y.reshape(n, d)
    Xf /= np.maximum(np.linalg.norm(Xf, axis=1, keepdims=True), 1e-12)
    return Xf

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    SEARCH_END = t0 + 128.0
    rng = np.random.RandomState(20261102)
    n = 163
    best_x = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (n, 4) or not np.all(np.isfinite(a)):
                continue
            a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
            v = _max_cos(a)
            if v < best_val:
                best_val = v
                best_x = a.copy()
        except Exception:
            continue
    if best_x is None:
        best_x = rng.randn(n, 4)
        best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
        best_val = _max_cos(best_x)
    # symmetric configs: (K,m)
    configs = [(6, 1), (6, 5), (8, 3), (8, 5), (10, 3), (10, 7), (12, 5), (12, 7), (12, 1), (10, 1)]
    trial = 0
    # quick sym trials budget
    while time.time() < SEARCH_END - 20.0:
        trial += 1
        if time.time() > DEADLINE - 52:
            break
        mode = trial % 5
        try:
            if mode in (0, 1, 2):
                # symmetric orbit construction
                ci = int(rng.randint(len(configs)))
                K, mm = configs[ci]
                L = n // K
                R = n - L * K
                thetas0 = rng.uniform(0.18, 1.39, size=L)
                alphas0 = rng.uniform(0, 2 * np.pi, size=L)
                betas0 = rng.uniform(0, 2 * np.pi, size=L)
                free0 = rng.randn(R, 4) if R > 0 else np.zeros((0, 4))
                if R > 0:
                    free0 /= np.linalg.norm(free0, axis=1, keepdims=True)
                # stage 1: exact symmetry anneal
                Xs, vs = _sym_anneal(thetas0, alphas0, betas0, free0, K, mm,
                                    steps=1100, T0=0.016, T1=0.0009, lr=0.07, mu=0.85,
                                    deadline=SEARCH_END)
                # stage 2: orbit-wise split (alternating theta jitter keeps collective mode)
                Xsplit = Xs + rng.randn(n, 4) * 0.004
                Xsplit /= np.linalg.norm(Xsplit, axis=1, keepdims=True)
                Xm, vm = _full_gd(Xsplit, steps=2200, T0=0.009, T1=0.0005, lr0=0.036, mu=0.87,
                                  rng=rng, noise0=0.0012, deadline=SEARCH_END)
                # stage 3: full release with bigger kick if promising or randomly
                if vm < best_val + 0.004 or rng.rand() < 0.25:
                    Xrel = Xm + rng.randn(n, 4) * 0.008
                    Xrel /= np.linalg.norm(Xrel, axis=1, keepdims=True)
                    Xf, vf = _full_gd(Xrel, steps=1800, T0=0.006, T1=0.00035, lr0=0.030, mu=0.88,
                                      rng=rng, noise0=0.0005, deadline=SEARCH_END)
                else:
                    Xf, vf = Xm, vm
                if vf < best_val:
                    # quick LBFGS + light LP before accepting
                    if time.time() < SEARCH_END - 8:
                        Xf = _lbfgs_smooth(Xf, SEARCH_END, taus=(0.002, 0.0007), iters=(60, 60))
                        vf = _max_cos(Xf)
                    if vf < best_val:
                        best_val = float(vf)
                        best_x = Xf.copy()
            elif mode == 3:
                # splice two symmetric halves with random quaternion rotation
                K1, m1 = configs[int(rng.randint(len(configs)))]
                K2, m2 = configs[int(rng.randint(len(configs)))]
                n1 = n // 2
                n2 = n - n1
                L1 = n1 // K1; R1 = n1 - L1 * K1
                L2 = n2 // K2; R2 = n2 - L2 * K2
                th1 = rng.uniform(0.18, 1.39, size=max(L1, 1))[:L1]
                al1 = rng.uniform(0, 2 * np.pi, size=max(L1, 1))[:L1]
                be1 = rng.uniform(0, 2 * np.pi, size=max(L1, 1))[:L1]
                th2 = rng.uniform(0.18, 1.39, size=max(L2, 1))[:L2]
                al2 = rng.uniform(0, 2 * np.pi, size=max(L2, 1))[:L2]
                be2 = rng.uniform(0, 2 * np.pi, size=max(L2, 1))[:L2]
                fr1 = rng.randn(R1, 4) if R1 > 0 else np.zeros((0, 4))
                fr2 = rng.randn(R2, 4) if R2 > 0 else np.zeros((0, 4))
                X1, _ = _sym_anneal(th1, al1, be1, fr1 if R1 > 0 else np.zeros((0, 4)), K1, m1,
                                    steps=700, T0=0.014, T1=0.001, lr=0.07, mu=0.85, deadline=SEARCH_END)
                X2, _ = _sym_anneal(th2, al2, be2, fr2 if R2 > 0 else np.zeros((0, 4)), K2, m2,
                                    steps=700, T0=0.014, T1=0.001, lr=0.07, mu=0.85, deadline=SEARCH_END)
                Q = _rand_rot(rng)
                X2 = X2 @ Q.T
                Xc = np.vstack([X1, X2])
                # fix size mismatch due to integer division quirks
                if Xc.shape[0] > n:
                    Xc = Xc[:n]
                elif Xc.shape[0] < n:
                    ex = rng.randn(n - Xc.shape[0], 4)
                    ex /= np.linalg.norm(ex, axis=1, keepdims=True)
                    Xc = np.vstack([Xc, ex])
                Xc = Xc + rng.randn(n, 4) * 0.003
                Xc /= np.linalg.norm(Xc, axis=1, keepdims=True)
                Xm, vm = _full_gd(Xc, steps=2600, T0=0.010, T1=0.0005, lr0=0.038, mu=0.86,
                                  rng=rng, noise0=0.0015, deadline=SEARCH_END)
                if vm < best_val:
                    if time.time() < SEARCH_END - 8:
                        Xm = _lbfgs_smooth(Xm, SEARCH_END, taus=(0.002, 0.0007), iters=(60, 60))
                        vm = _max_cos(Xm)
                    if vm < best_val:
                        best_val = float(vm)
                        best_x = Xm.copy()
            else:
                # perturb incumbent to guarantee competitiveness
                sig = [0.015, 0.04, 0.09][trial % 3]
                cur = best_x + rng.randn(n, 4) * sig
                cur /= np.linalg.norm(cur, axis=1, keepdims=True)
                Xm, vm = _full_gd(cur, steps=2400, T0=0.010, T1=0.00045, lr0=0.036, mu=0.87,
                                  rng=rng, noise0=0.001, deadline=SEARCH_END)
                if vm < best_val:
                    best_val = float(vm)
                    best_x = Xm.copy()
        except Exception:
            continue
    # final polish of global best
    try:
        if time.time() < DEADLINE - 12:
            best_x = _lbfgs_smooth(best_x, DEADLINE - 10, taus=(0.0025, 0.0009, 0.0004), iters=(70, 70, 60))
            best_val = _max_cos(best_x)
        best_x, best_val = _lp_polish(best_x, DEADLINE - 1.0, r0=0.004)
    except Exception:
        pass
    best_x = np.asarray(best_x, dtype=np.float64)
    best_x /= np.maximum(np.linalg.norm(best_x, axis=1, keepdims=True), 1e-12)
    # tiny shrink guard against boundary ties (no-op normalisation safe)
    return best_x
