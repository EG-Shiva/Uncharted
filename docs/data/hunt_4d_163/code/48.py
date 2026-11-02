import time
import numpy as np
from scipy.optimize import minimize, linprog
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

def _lp_polish(x, deadline, r0=0.003, cap=1400):
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x, 1e9
    radius = r0
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        try:
            G = x @ x.T
            c = G[iu0, iu1]
            m = float(c.max())
            sel = c > m - max(20.0 * radius, 1e-9)
            if np.count_nonzero(sel) == 0:
                sel[np.argmax(c)] = True
            if np.count_nonzero(sel) > cap:
                idx = np.argsort(-c)[:cap]
                s2 = np.zeros_like(sel); s2[idx] = True; sel = s2
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

def _anneal(x0, steps, T0, T1, lr0, mu, rng, noise0=0.0, deadline=None):
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
        lr = lr0 * (1.0 - 0.55 * frac)
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
            bv = m; bx = x.copy()
    return bx, bv

def _lbfgs_smooth(x_init, deadline, taus=(0.003, 0.001, 0.0004), iters=(70, 70, 80)):
    x = x_init / np.maximum(np.linalg.norm(x_init, axis=1, keepdims=True), 1e-12)
    n, d = x.shape
    iu0, iu1 = np.triu_indices(n, 1)
    Np = len(iu0)
    logNp = np.log(Np)
    y = x.ravel().copy()
    for tau, maxiter in zip(taus, iters):
        if time.time() > deadline - 0.3:
            break
        def fg(yflat, _tau=tau):
            Y = yflat.reshape(n, d)
            rn = np.linalg.norm(Y, axis=1, keepdims=True)
            rn = np.maximum(rn, 1e-12)
            X = Y / rn
            g = (X @ X.T)[iu0, iu1]
            m = float(g.max())
            e = np.exp((g - m) / _tau)
            s = float(e.sum())
            if s <= 0 or not np.isfinite(s):
                return float(m), np.zeros_like(yflat)
            L = m + _tau * (np.log(s) - logNp)
            w = e / s
            W = np.zeros((n, n)); W[iu0, iu1] = w; W = W + W.T
            GX = W @ X
            dot = (GX * X).sum(axis=1, keepdims=True)
            GY = (GX - dot * X) / rn
            return float(L), GY.ravel()
        try:
            res = minimize(fg, y, method="L-BFGS-B", jac=True,
                           options={"maxiter": int(maxiter), "maxfun": int(maxiter * 5)})
            y = res.x
        except Exception:
            break
    Xf = y.reshape(n, d)
    Xf /= np.maximum(np.linalg.norm(Xf, axis=1, keepdims=True), 1e-12)
    return Xf, _max_cos(Xf)

def _x_to_hopf(X):
    r1 = np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)
    r2 = np.sqrt(X[:, 2] ** 2 + X[:, 3] ** 2)
    th = np.arctan2(r2, r1)
    a = np.arctan2(X[:, 1], X[:, 0])
    b = np.arctan2(X[:, 3], X[:, 2])
    return th, a, b

def _hopf_build(th, a, b):
    c = np.cos(th); s = np.sin(th)
    return np.stack([c * np.cos(a), c * np.sin(a), s * np.cos(b), s * np.sin(b)], axis=1)

def _hopf_refine(X0, deadline, tau=0.0025, maxiter=60):
    n = X0.shape[0]
    th0, a0, b0 = _x_to_hopf(X0)
    p0 = np.concatenate([th0, a0, b0])
    iu0, iu1 = np.triu_indices(n, 1)
    Np = len(iu0); logNp = np.log(Np)
    def fg(p, _tau=tau):
        th = p[0:n]; a = p[n:2 * n]; b = p[2 * n:3 * n]
        c = np.cos(th); s = np.sin(th)
        ca = np.cos(a); sa = np.sin(a); cb = np.cos(b); sb = np.sin(b)
        X = np.stack([c * ca, c * sa, s * cb, s * sb], axis=1)
        g = (X @ X.T)[iu0, iu1]
        m = float(g.max())
        e = np.exp((g - m) / _tau)
        es = float(e.sum())
        if es <= 0 or not np.isfinite(es):
            return float(m), np.zeros_like(p)
        L = m + _tau * (np.log(es) - logNp)
        w = e / es
        W = np.zeros((n, n)); W[iu0, iu1] = w; W = W + W.T
        GX = W @ X
        dth = GX[:, 0] * (-s * ca) + GX[:, 1] * (-s * sa) + GX[:, 2] * (c * cb) + GX[:, 3] * (c * sb)
        da = GX[:, 0] * (-c * sa) + GX[:, 1] * (c * ca)
        db = GX[:, 2] * (-s * sb) + GX[:, 3] * (s * cb)
        gp = np.concatenate([dth, da, db])
        return float(L), gp
    try:
        res = minimize(fg, p0, method="L-BFGS-B", jac=True,
                       options={"maxiter": int(maxiter), "maxfun": int(maxiter * 5)})
        p = res.x
    except Exception:
        p = p0
    th = p[0:n]; a = p[n:2 * n]; b = p[2 * n:3 * n]
    X = _hopf_build(th, a, b)
    return X, _max_cos(X)

def _sym_opt_cyclic(k, m, rng, tau_list=(0.008, 0.003), iter_list=(60, 60)):
    n_tot = 163
    t = n_tot // k
    r = n_tot - t * k
    sh_a = 2 * np.pi * np.arange(k) / k
    sh_b = 2 * np.pi * np.arange(k) * m / k
    iu0, iu1 = np.triu_indices(n_tot, 1)
    Np = len(iu0); logNp = np.log(Np)
    th0 = np.arccos(np.clip(rng.rand(t) * 2 - 1, -1, 1)) / 2
    th0 = np.clip(th0, 0.08, np.pi / 2 - 0.08)
    al0 = rng.rand(t) * 2 * np.pi
    be0 = rng.rand(t) * 2 * np.pi
    rem0 = rng.randn(max(r, 1), 4)
    p = np.concatenate([th0, al0, be0, rem0.ravel()]) if r > 0 else np.concatenate([th0, al0, be0])
    def build_X(pv):
        th = pv[0:t]; al = pv[t:2 * t]; be = pv[2 * t:3 * t]
        c = np.cos(th); s = np.sin(th)
        X = np.empty((n_tot, 4))
        for j in range(t):
            a = al[j] + sh_a; b = be[j] + sh_b
            X[j * k:(j + 1) * k, 0] = c[j] * np.cos(a)
            X[j * k:(j + 1) * k, 1] = c[j] * np.sin(a)
            X[j * k:(j + 1) * k, 2] = s[j] * np.cos(b)
            X[j * k:(j + 1) * k, 3] = s[j] * np.sin(b)
        if r > 0:
            Y = pv[3 * t:].reshape(r, 4)
            nr = np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
            X[t * k:] = Y / nr
        return X
    for tau, maxiter in zip(tau_list, iter_list):
        def fg(pv, _tau=tau):
            th = pv[0:t]; al = pv[t:2 * t]; be = pv[2 * t:3 * t]
            X = build_X(pv)
            g = (X @ X.T)[iu0, iu1]
            mm = float(g.max())
            e = np.exp((g - mm) / _tau)
            es = float(e.sum())
            if es <= 0 or not np.isfinite(es):
                return float(mm), np.zeros_like(pv)
            L = mm + _tau * (np.log(es) - logNp)
            w = e / es
            W = np.zeros((n_tot, n_tot)); W[iu0, iu1] = w; W = W + W.T
            GX = W @ X
            c = np.cos(th); s = np.sin(th)
            gth = np.zeros(t); gal = np.zeros(t); gbe = np.zeros(t)
            for j in range(t):
                a = al[j] + sh_a; b = be[j] + sh_b
                ca = np.cos(a); sa = np.sin(a); cb = np.cos(b); sb = np.sin(b)
                Gblk = GX[j * k:(j + 1) * k]
                dth0 = -s[j] * ca; dth1 = -s[j] * sa; dth2 = c[j] * cb; dth3 = c[j] * sb
                gth[j] = float((Gblk[:, 0] * dth0 + Gblk[:, 1] * dth1 + Gblk[:, 2] * dth2 + Gblk[:, 3] * dth3).sum())
                gal[j] = float((Gblk[:, 0] * (-c[j] * sa) + Gblk[:, 1] * (c[j] * ca)).sum())
                gbe[j] = float((Gblk[:, 2] * (-s[j] * sb) + Gblk[:, 3] * (s[j] * cb)).sum())
            gp = np.zeros_like(pv)
            gp[0:t] = gth; gp[t:2 * t] = gal; gp[2 * t:3 * t] = gbe
            if r > 0:
                Y = pv[3 * t:].reshape(r, 4)
                nr = np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
                Xn = Y / nr
                Gr = GX[t * k:]
                dot = (Gr * Xn).sum(axis=1, keepdims=True)
                GY = (Gr - dot * Xn) / nr
                gp[3 * t:] = GY.ravel()
            return float(L), gp
        try:
            res = minimize(fg, p, method="L-BFGS-B", jac=True,
                           options={"maxiter": int(maxiter), "maxfun": int(maxiter * 5)})
            p = res.x
        except Exception:
            break
    return build_X(p)

def _sign_group_mats():
    mats = []
    for s0 in (-1.0, 1.0):
        for s1 in (-1.0, 1.0):
            for s2 in (-1.0, 1.0):
                s3 = s0 * s1 * s2  # det +1
                mats.append(np.diag([s0, s1, s2, s3]))
    return mats

def _sym_opt_sign(rng, tau=0.004, maxiter=70):
    Gs = _sign_group_mats()
    Gg = len(Gs)  # 8
    n_tot = 163
    u = n_tot // Gg
    r = n_tot - u * Gg
    n = n_tot
    iu0, iu1 = np.triu_indices(n, 1)
    Np = len(iu0); logNp = np.log(Np)
    S0 = rng.randn(u, 4); R0 = rng.randn(max(r, 1), 4)
    p0 = np.concatenate([S0.ravel(), R0.ravel()]) if r > 0 else S0.ravel()
    GG = np.stack(Gs, axis=0)
    def build_X(pv):
        S = pv[:u * 4].reshape(u, 4)
        ns = np.maximum(np.linalg.norm(S, axis=1, keepdims=True), 1e-12)
        Sn = S / ns
        X = np.empty((n, 4))
        for i in range(u):
            for gi in range(Gg):
                X[i * Gg + gi] = GG[gi] @ Sn[i]
        if r > 0:
            Y = pv[u * 4:].reshape(r, 4)
            ny = np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
            X[u * Gg:] = Y / ny
        return X
    def fg(pv, _tau=tau):
        X = build_X(pv)
        g = (X @ X.T)[iu0, iu1]
        mm = float(g.max())
        e = np.exp((g - mm) / _tau)
        es = float(e.sum())
        if es <= 0 or not np.isfinite(es):
            return float(mm), np.zeros_like(pv)
        L = mm + _tau * (np.log(es) - logNp)
        w = e / es
        W = np.zeros((n, n)); W[iu0, iu1] = w; W = W + W.T
        GX = W @ X
        S = pv[:u * 4].reshape(u, 4)
        ns = np.maximum(np.linalg.norm(S, axis=1, keepdims=True), 1e-12)
        Sn = S / ns
        gS = np.zeros_like(S)
        for i in range(u):
            H = np.zeros(4)
            for gi in range(Gg):
                H += GG[gi].T @ GX[i * Gg + gi]
            gS[i] = (H - float(H @ Sn[i]) * Sn[i]) / float(ns[i, 0])
        gp = np.zeros_like(pv)
        gp[:u * 4] = gS.ravel()
        if r > 0:
            Y = pv[u * 4:].reshape(r, 4)
            ny = np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
            Yn = Y / ny
            Gr = GX[u * Gg:]
            dot = (Gr * Yn).sum(axis=1, keepdims=True)
            gp[u * 4:] = ((Gr - dot * Yn) / ny).ravel()
        return float(L), gp
    try:
        res = minimize(fg, p0, method="L-BFGS-B", jac=True,
                       options={"maxiter": int(maxiter), "maxfun": int(maxiter * 5)})
        p0 = res.x
    except Exception:
        pass
    return build_X(p0)

def solve():
    t0 = time.time()
    DEADLINE = t0 + 173.0
    SEARCH_END = t0 + 140.0
    rng = np.random.RandomState(20261102)
    n = 163
    try:
        bx = np.load("best.npy")
    except Exception:
        try:
            bx = np.load("record.npy")
        except Exception:
            bx = rng.randn(n, 4)
    bx = np.asarray(bx, dtype=float)
    if bx.shape != (n, 4) or not np.all(np.isfinite(bx)):
        bx = rng.randn(n, 4)
    bx = bx / np.maximum(np.linalg.norm(bx, axis=1, keepdims=True), 1e-12)
    best_x = bx.copy()
    best_val = _max_cos(best_x)
    configs = [(6, 1), (6, 5), (8, 1), (8, 3), (10, 1), (10, 3), (12, 1), (12, 5)]
    trial = 0
    while time.time() < SEARCH_END:
        if time.time() > DEADLINE - 35:
            break
        trial += 1
        mode = trial % 6
        cur = None
        try:
            if mode in (0, 1, 2):
                k, mm = configs[(trial // 6) % len(configs)]
                cur = _sym_opt_cyclic(k, mm, rng)
                # stage 1: Hopf split (orbit-wise release)
                cur, _ = _hopf_refine(cur, SEARCH_END, tau=0.003, maxiter=50)
                if time.time() > SEARCH_END:
                    pass
                # stage 2: full Cartesian release
                cur, _ = _anneal(cur, 1600, 0.008, 0.0005, 0.035, 0.86, rng, noise0=0.0015, deadline=SEARCH_END)
                cur, _ = _lbfgs_smooth(cur, SEARCH_END, taus=(0.0025, 0.0008), iters=(60, 60))
            elif mode == 3:
                # quaternionic sign orbit start then release
                cur = _sym_opt_sign(rng, tau=0.004, maxiter=60)
                cur, _ = _hopf_refine(cur, SEARCH_END, tau=0.003, maxiter=40)
                cur, _ = _anneal(cur, 1400, 0.009, 0.0005, 0.035, 0.86, rng, noise0=0.0015, deadline=SEARCH_END)
                cur, _ = _lbfgs_smooth(cur, SEARCH_END, taus=(0.0025, 0.0008), iters=(60, 60))
            elif mode == 4:
                # splice two symmetry halves with random quaternion rotation
                k1, m1 = configs[rng.randint(len(configs))]
                k2, m2 = configs[rng.randint(len(configs))]
                A = _sym_opt_cyclic(k1, m1, rng)
                B = _sym_opt_cyclic(k2, m2, rng)
                R = _rand_rot(rng)
                # take halves
                n1 = n // 2
                A1 = A[:n1]
                B1 = (B[:(n - n1)] @ R.T)
                cur = np.vstack([A1, B1])
                cur /= np.maximum(np.linalg.norm(cur, axis=1, keepdims=True), 1e-12)
                # deduplicate crowded pairs greedily
                for _ in range(4):
                    G = cur @ cur.T
                    np.fill_diagonal(G, -2.0)
                    if float(G.max()) < best_val + 0.01:
                        break
                    i, j = np.unravel_index(int(np.argmax(G)), G.shape)
                    cur[j] = rng.randn(4); cur[j] /= np.linalg.norm(cur[j])
                cur, _ = _anneal(cur, 1800, 0.010, 0.0005, 0.038, 0.85, rng, noise0=0.002, deadline=SEARCH_END)
                cur, _ = _lbfgs_smooth(cur, SEARCH_END, taus=(0.0025, 0.0008), iters=(60, 60))
            else:
                # direct basin perturbation of incumbent to stay competitive
                sig = [0.02, 0.08, 0.2][trial % 3]
                cur = best_x + rng.randn(n, 4) * sig
                cur /= np.maximum(np.linalg.norm(cur, axis=1, keepdims=True), 1e-12)
                cur, _ = _anneal(cur, 1600, 0.009, 0.0005, 0.035, 0.87, rng, noise0=0.001, deadline=SEARCH_END)
                cur, _ = _lbfgs_smooth(cur, SEARCH_END, taus=(0.0025, 0.0008), iters=(60, 60))
        except Exception:
            continue
        try:
            v = _max_cos(cur)
        except Exception:
            continue
        # LP polish promising candidates
        if v < best_val + 0.004 and time.time() < DEADLINE - 20:
            try:
                cur2, v2 = _lp_polish(cur, min(DEADLINE - 8, time.time() + 6.0), r0=0.002)
                if v2 < v:
                    cur, v = cur2, v2
            except Exception:
                pass
        if v < best_val:
            best_val = v
            best_x = cur.copy()
    best_x = best_x / np.maximum(np.linalg.norm(best_x, axis=1, keepdims=True), 1e-12)
    return best_x
