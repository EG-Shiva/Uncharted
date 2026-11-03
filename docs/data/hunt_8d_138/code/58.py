import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm_rows(X):
    nrm = np.sqrt(np.sum(X * X, axis=1, keepdims=True)) + 1e-30
    X = X / nrm
    X[~np.isfinite(X)] = 0.0
    nrm2 = np.sqrt(np.sum(X * X, axis=1, keepdims=True))
    nrm2[nrm2 == 0] = 1.0
    return X / nrm2

def _lse_jam(X0, betas, maxiters):
    from scipy.optimize import minimize
    n, d = X0.shape
    iu = np.triu_indices(n, 1)
    Y = _norm_rows(X0.copy())
    for beta, mi in zip(betas, maxiters):
        def fun(Yf):
            Ym = Yf.reshape(n, d)
            nr = np.sqrt(np.sum(Ym * Ym, axis=1, keepdims=True)) + 1e-30
            X = Ym / nr
            G = X @ X.T
            c = G[iu]
            mv = c.max()
            E = np.exp(beta * (c - mv))
            S = E.sum() + 1e-300
            f = float(mv + np.log(S) / beta)
            W = np.zeros((n, n))
            W[iu] = E
            W = W + W.T
            GX = (W @ X) / S
            dot = np.sum(GX * X, axis=1, keepdims=True)
            Gt = GX - dot * X
            GY = Gt / nr
            return f, GY.ravel()
        try:
            res = minimize(fun, Y.ravel(), jac=True, method='L-BFGS-B',
                           options={'maxiter': int(mi), 'ftol': 1e-14, 'gtol': 1e-6, 'maxcor': 12,
                                    'maxfun': int(mi * 30)})
            Y = _norm_rows(res.x.reshape(n, d))
        except Exception:
            pass
    return _norm_rows(Y)

def _lp_once(X, deadline, radius0=8e-3):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    X = _norm_rows(X)
    n, d = X.shape
    best = _max_cos(X)
    Xb = X.copy()
    radius = radius0
    iu = np.triu_indices(n, 1)
    cf = X @ X.T
    while time.time() < deadline and radius > 1e-14:
        c = (X @ X.T)[iu]
        mval = float(c.max())
        sel = c > mval - max(20.0 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        # cap to keep LP tractable, keep tightest
        if p > 3500:
            idx = np.argsort(-c[sel])[:3500]
            I, J = I[idx], J[idx]
            p = len(I)
            csv = c[sel][idx]
        else:
            csv = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csv, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = _norm_rows(y)
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            Xb = y.copy()
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() >= deadline:
            break
    return Xb

def _lp_restarted(X, deadline):
    # two restarts with different initial radii, keep best
    X = _norm_rows(X)
    best = _max_cos(X)
    Xb = X.copy()
    # split remaining time
    now = time.time()
    rem = deadline - now
    if rem <= 0.3:
        return Xb
    # first restart
    mid = now + rem * 0.55
    try:
        X1 = _lp_once(Xb, mid, radius0=8e-3)
        v1 = _max_cos(X1)
        if v1 < best:
            best, Xb = v1, X1
    except Exception:
        pass
    try:
        X2 = _lp_once(Xb, deadline, radius0=2e-3)
        v2 = _max_cos(X2)
        if v2 < best:
            best, Xb = v2, X2
    except Exception:
        pass
    return Xb

def solve():
    import time
    import numpy as np
    t0 = time.time()
    DEADLINE = t0 + 176.0
    rng = np.random.RandomState(20261102)
    n, d = 138, 8
    # load incumbent
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    if xb is not None and xb.shape == (n, d):
        cands.append(_norm_rows(xb))
    if xr is not None and xr.shape == (n, d):
        cands.append(_norm_rows(xr))
    if len(cands) == 0:
        cands.append(_norm_rows(rng.randn(n, d)))
    vals = [_max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    x0 = cands[bi].copy()
    best_val = float(vals[bi])
    x_best = x0.copy()

    # lightly jam starting point if time allows (baseline)
    # build Fiedler split from best jammed-ish point; first get graph
    Xref = x_best.copy()
    G = Xref @ Xref.T
    np.fill_diagonal(G, -2.0)
    m0 = float(G.max())
    A = (G > m0 - 1e-5).astype(float)
    np.fill_diagonal(A, 0.0)
    # symmetrize
    A = np.maximum(A, A.T)
    deg = A.sum(axis=1)
    L = np.diag(deg) - A
    try:
        w, V = np.linalg.eigh(L)
        order = np.argsort(w)
        w = w[order]; V = V[:, order]
        # pick first eigenvector with eigenvalue > 1e-9 as Fiedler
        fied = None
        for k in range(n):
            if w[k] > 1e-9:
                fied = V[:, k].copy()
                break
        if fied is None:
            fied = V[:, 1].copy()
    except Exception:
        fied = rng.randn(n)
    # balanced median split by rank
    rank = np.argsort(fied)
    side = np.ones(n)
    side[rank[:n // 2]] = -1.0
    # side as +/-1
    # global coherent vector from sign pattern
    # v_coh = sum s_i x_i normalized
    v_coh = (side[:, None] * Xref).sum(axis=0)
    nv = np.linalg.norm(v_coh)
    if nv < 1e-12:
        v_coh = rng.randn(d); v_coh /= np.linalg.norm(v_coh)
    else:
        v_coh = v_coh / nv
    # centroid difference vector
    m_plus = Xref[side > 0].mean(axis=0)
    m_minus = Xref[side < 0].mean(axis=0)
    v_sep = m_plus - m_minus
    nv2 = np.linalg.norm(v_sep)
    if nv2 < 1e-12:
        v_sep = v_coh.copy()
    else:
        v_sep = v_sep / nv2
    # direction pool: coherent, separation, + randoms (deterministic)
    v_pool = [v_coh, v_sep]
    for _ in range(2):
        v = rng.randn(d); v /= np.linalg.norm(v)
        v_pool.append(v)

    amplitudes = [0.01, 0.03, 0.08, 0.15, 0.25]

    def tangent_displace(X, sidev, vglob, amp):
        Tn = np.empty_like(X)
        dots = X @ vglob  # n
        T = vglob[None, :] - dots[:, None] * X
        tn = np.sqrt(np.sum(T * T, axis=1))
        # fallback for degenerate rows
        for i in range(X.shape[0]):
            if tn[i] < 1e-12:
                # random tangent
                r = rng.randn(X.shape[1])
                r = r - np.dot(r, X[i]) * X[i]
                rn = np.linalg.norm(r)
                if rn < 1e-12:
                    r = np.zeros_like(r); r[0] = 1.0
                    r = r - np.dot(r, X[i]) * X[i]
                    rn = np.linalg.norm(r) + 1e-30
                T[i] = r / rn
                tn[i] = 1.0
            else:
                T[i] = T[i] / tn[i]
        Y = X + amp * sidev[:, None] * T
        return _norm_rows(Y)

    # main Fiedler-split sweep
    for amp in amplitudes:
        for vi, vglob in enumerate(v_pool):
            if time.time() > DEADLINE - 22.0:
                break
            # displace
            try:
                Xprop = tangent_displace(Xref, side, vglob, amp)
            except Exception:
                continue
            v_after_disp = _max_cos(Xprop)
            if v_after_disp < best_val - 1e-12:
                best_val = v_after_disp
                x_best = Xprop.copy()
            # short symmetry-free LSE re-jam
            try:
                # smaller budget for small amps, same for all to stay in time
                Xjam = _lse_jam(Xprop, [60.0, 220.0], [70, 70])
            except Exception:
                Xjam = Xprop
            vjam = _max_cos(Xjam)
            if vjam < best_val - 1e-12:
                best_val = vjam
                x_best = Xjam.copy()
            # one restarted-trust-region LP polish with limited time
            try:
                # allocate ~2.5s per polish, less near deadline
                lp_dead = min(time.time() + 2.8, DEADLINE - 18.0)
                if lp_dead > time.time() + 0.5:
                    Xlp = _lp_restarted(Xjam, lp_dead)
                else:
                    Xlp = Xjam
            except Exception:
                Xlp = Xjam
            try:
                vlp = _max_cos(Xlp)
            except Exception:
                continue
            if vlp < best_val - 1e-12:
                best_val = vlp
                x_best = Xlp.copy()
            # also update Xref to best for subsequent splits? keep original core
            # but if we found much better, re-anchor splits to new best occasionally
            # do not change side/v_pool mid-sweep to keep deterministic
        if time.time() > DEADLINE - 22.0:
            break

    # rebuild Fiedler on current best and try one more round at best amplitude if time
    try:
        if time.time() < DEADLINE - 25.0:
            G2 = x_best @ x_best.T
            np.fill_diagonal(G2, -2.0)
            m2 = float(G2.max())
            A2 = (G2 > m2 - 1e-5).astype(float)
            np.fill_diagonal(A2, 0.0)
            A2 = np.maximum(A2, A2.T)
            deg2 = A2.sum(axis=1)
            L2 = np.diag(deg2) - A2
            w2, V2 = np.linalg.eigh(L2)
            o2 = np.argsort(w2); w2 = w2[o2]; V2 = V2[:, o2]
            f2 = None
            for k in range(n):
                if w2[k] > 1e-9:
                    f2 = V2[:, k].copy()
                    break
            if f2 is None:
                f2 = V2[:, 1].copy()
            rank2 = np.argsort(f2)
            side2 = np.ones(n); side2[rank2[:n // 2]] = -1.0
            v2 = (side2[:, None] * x_best).sum(axis=0)
            v2 = v2 / (np.linalg.norm(v2) + 1e-30)
            for amp in [0.03, 0.08]:
                if time.time() > DEADLINE - 20.0:
                    break
                Xprop = tangent_displace(x_best, side2, v2, amp)
                Xjam = _lse_jam(Xprop, [80.0, 300.0], [80, 80])
                vjam = _max_cos(Xjam)
                if vjam < best_val - 1e-12:
                    best_val = vjam; x_best = Xjam.copy()
                lp_dead = min(time.time() + 3.0, DEADLINE - 14.0)
                if lp_dead > time.time() + 0.5:
                    Xlp = _lp_restarted(Xjam, lp_dead)
                    vlp = _max_cos(Xlp)
                    if vlp < best_val - 1e-12:
                        best_val = vlp; x_best = Xlp.copy()
    except Exception:
        pass

    # final intensive polish of global best with remaining time
    try:
        rem = DEADLINE - time.time()
        if rem > 5.0:
            # medium LSE at high beta
            Xf = _lse_jam(x_best, [250.0, 700.0], [120, 120] if rem > 14 else [60, 60])
            vf = _max_cos(Xf)
            if vf < best_val:
                best_val = vf; x_best = Xf.copy()
    except Exception:
        pass
    try:
        rem2 = DEADLINE - time.time()
        if rem2 > 3.0:
            x_best = _lp_restarted(x_best, time.time() + max(1.0, rem2 - 1.0))
            best_val = _max_cos(x_best)
    except Exception:
        pass

    x_best = _norm_rows(x_best)
    if not np.all(np.isfinite(x_best)):
        x_best = _norm_rows(x0)
    return x_best
