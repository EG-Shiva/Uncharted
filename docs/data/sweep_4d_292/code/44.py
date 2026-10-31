import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _normed(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _tangent_bases(X):
    n = X.shape[0]
    Us = np.zeros((n, 4, 3), dtype=np.float64)
    E = np.eye(4, dtype=np.float64)
    for i in range(n):
        xi = X[i]
        m = int(np.argmin(np.abs(xi)))
        Vs = []
        for k in range(4):
            if k == m:
                continue
            ek = E[k]
            v = ek - float(ek @ xi) * xi
            Vs.append(v)
        V = np.stack(Vs, axis=1)  # 4x3
        Q, _ = np.linalg.qr(V)
        Us[i] = Q
    return Us

def _equilibrium_stresses(X, I, J, reg=1e-6):
    n = X.shape[0]
    p = len(I)
    try:
        Us = _tangent_bases(X)
        Nv = n * 3
        Jt = np.zeros((p, Nv), dtype=np.float64)
        for k in range(p):
            i = int(I[k]); j = int(J[k])
            ai = Us[i].T @ X[j]  # 3
            aj = Us[j].T @ X[i]  # 3
            Jt[k, i * 3:(i + 1) * 3] = ai
            Jt[k, j * 3:(j + 1) * 3] = aj
        A = Jt.T  # Nv x p
        Q = A.T @ A  # p x p  == Jt @ Jt.T
        # regularize
        tr = float(np.trace(Q)) / max(1, p)
        Q = Q + (reg * max(1e-12, tr) + 1e-12) * np.eye(p)
        ones = np.ones(p, dtype=np.float64)
        try:
            qinv1 = np.linalg.solve(Q, ones)
        except Exception:
            qinv1 = np.linalg.lstsq(Q, ones, rcond=None)[0]
        den = float(ones @ qinv1)
        if not np.isfinite(den) or abs(den) < 1e-18:
            return np.ones(p)
        s = p * qinv1 / den
        s = np.asarray(s, dtype=np.float64)
        # clip absurd values for stability
        s = np.clip(s, -10.0, 10.0)
        return s
    except Exception:
        return np.ones(p)

def _lp_polish_deadline(x, deadline, r0=1e-3):
    x = _normed(x)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    xb = x.copy()
    radius = r0
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        if not np.any(sel):
            break
        I = iu[0][sel]; J = iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d),
                                   J[:, None] * d + np.arange(d),
                                   np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            break
        if getattr(res, "status", 1) != 0:
            radius *= 0.5
            continue
        try:
            y = x + res.x[:-1].reshape(n, d)
        except Exception:
            radius *= 0.5
            continue
        y = _normed(y)
        try:
            vv = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if vv < best:
            best = vv; xb = y.copy(); x = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return xb

def _hinge_opt(X0, Teff, wvec, beta, maxiter, iu0, iu1):
    n = X0.shape[0]
    # init Y = X0 (already normed)
    yflat = X0.ravel().copy()
    def fg(yf):
        Y = yf.reshape(n, 4)
        nr = np.linalg.norm(Y, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        S = Y / nr
        Gr = S @ S.T
        c = Gr[iu0, iu1]
        d = c - Teff
        z = beta * d
        zc = np.clip(z, -60.0, 60.0)
        # softplus
        Sp = np.logaddexp(0.0, z)  # stable even for large z
        h = Sp / beta
        # sigmoid
        sig = 1.0 / (1.0 + np.exp(-zc))
        loss = float(np.sum(wvec * h * h))
        g = wvec * (2.0 * h * sig)  # dL/dc
        Gmat = np.zeros((n, n), dtype=np.float64)
        Gmat[iu0, iu1] = g
        Gmat = Gmat + Gmat.T
        gradS = Gmat @ S
        dot = (gradS * S).sum(axis=1, keepdims=True)
        gradY = (gradS - dot * S) / nr
        return loss, gradY.ravel()
    try:
        res = minimize(fg, yflat, jac=True, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-7, 'maxfun': maxiter * 8})
        yf = res.x
    except Exception:
        yf = yflat
    Y = yf.reshape(n, 4)
    X = _normed(Y)
    return X

def solve():
    t0 = time.time()
    TLIM = 172.0
    deadline = t0 + TLIM
    rng = np.random.default_rng(292031)
    n, d = 292, 4
    # load incumbent
    xb = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (292, 4):
                a = _normed(a)
                v = _max_cos(a)
                if v < best_val:
                    best_val = v
                    xb = a.copy()
        except Exception:
            continue
    if xb is None:
        xb = _normed(rng.standard_normal((n, d)))
        best_val = _max_cos(xb)
    x_best = xb.copy()
    iu0, iu1 = np.triu_indices(n, 1)
    P = len(iu0)
    # active set ~870
    p_active = 3 * n - 6
    try:
        G0 = xb @ xb.T
        c0 = G0[iu0, iu1]
        order = np.argsort(-c0)
        top = order[:p_active]
        Ia = iu0[top]; Ja = iu1[top]
        s = _equilibrium_stresses(xb, Ia, Ja)
    except Exception:
        Ia = None; Ja = None
        s = np.ones(p_active)
    # select deactivation subsets
    K = 70
    try:
        idx_sorted = np.argsort(s)
        small_idx = idx_sorted[:K]
        large_idx = idx_sorted[-K:]
        mix_idx = np.concatenate([idx_sorted[:K // 2], idx_sorted[-K // 2:]])
    except Exception:
        small_idx = np.arange(K)
        large_idx = np.arange(K)
        mix_idx = np.arange(K)
    # map active position -> pair; build deact pair lists
    deact_lists = []
    try:
        pairs_active = list(zip(Ia.tolist(), Ja.tolist()))
        for sel in (large_idx, small_idx, mix_idx):
            lst = [pairs_active[int(k)] for k in sel]
            deact_lists.append(lst)
    except Exception:
        deact_lists = [[], [], []]
    Ts = [0.90340, 0.90335, 0.90325]
    sigmas = [0.012, 0.022, 0.035]
    Wdeact = 25.0
    GAP = 1e-5
    # iterate targets
    for ti in range(3):
        if time.time() > deadline - 30.0:
            break
        T = Ts[ti]
        deact = deact_lists[ti] if ti < len(deact_lists) else []
        # build Teff / wvec over all pairs
        Teff = np.full(P, T, dtype=np.float64)
        wvec = np.ones(P, dtype=np.float64)
        if len(deact) > 0:
            dmat = np.zeros((n, n), dtype=bool)
            for (i, j) in deact:
                dmat[i, j] = True; dmat[j, i] = True
            mask = dmat[iu0, iu1]
            Teff[mask] = T - GAP
            wvec[mask] = Wdeact
        # perturb incumbent (use x_best as base to keep progress, occasionally xb)
        base = x_best if ti > 0 else xb
        sigma = sigmas[ti]
        try:
            pert = rng.standard_normal((n, d))
            pert -= (pert * base).sum(axis=1, keepdims=True) * base
            pn = np.linalg.norm(pert, axis=1, keepdims=True) + 1e-12
            pert = pert / pn * sigma
            X0 = _normed(base + pert)
        except Exception:
            X0 = base.copy()
        # two-stage beta hinge
        for beta, mi in ((1500.0, 90), (4000.0, 110)):
            if time.time() > deadline - 28.0:
                break
            try:
                X0 = _hinge_opt(X0, Teff, wvec, beta, mi, iu0, iu1)
            except Exception:
                break
        try:
            v = _max_cos(X0)
        except Exception:
            continue
        # check deactivated pairs satisfy gap?
        if v < best_val:
            best_val = v
            x_best = X0.copy()
        # near-feasible -> LP polish to squeeze to local optimum
        # success at any T immediately gives better code; also polish near misses
        if v < T + 8e-06:
            lp_dead = min(deadline - 12.0, time.time() + 6.0)
            if lp_dead > time.time() + 1.0:
                try:
                    Xp = _lp_polish_deadline(X0, lp_dead)
                    vp = _max_cos(Xp)
                    if vp < best_val:
                        best_val = vp
                        x_best = Xp.copy()
                except Exception:
                    pass
        else:
            # even failures give diverse start: brief LP (2s) to settle different topology, keep if competitive
            if v < best_val + 3e-04 and time.time() < deadline - 35.0:
                try:
                    Xp = _lp_polish_deadline(X0, time.time() + 2.0)
                    vp = _max_cos(Xp)
                    if vp < best_val:
                        best_val = vp
                        x_best = Xp.copy()
                    # chain: use Xp as base for next target if it has different active set
                    # compare overlap of top contacts
                    try:
                        Gp = Xp @ Xp.T
                        cp = Gp[iu0, iu1]
                        order_p = set(np.argsort(-cp)[:p_active].tolist())
                        order_b = set(order[:p_active].tolist())
                        inter = len(order_p.intersection(order_b)) / float(p_active)
                        if inter < 0.85:
                            # structurally different: keep Xp for next iteration base by swapping
                            # store as xb-diverse for next loop (reuse x_best already)
                            pass
                    except Exception:
                        pass
                except Exception:
                    pass
    # final polish of best
    try:
        if time.time() < deadline - 3.0:
            x_best = _lp_polish_deadline(x_best, min(deadline - 1.0, time.time() + 12.0))
            best_val = _max_cos(x_best)
    except Exception:
        pass
    x_best = _normed(x_best)
    return x_best
