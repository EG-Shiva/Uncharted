import time
import numpy as np

def _norm_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _anneal_lse_adam(X, steps=1500, beta0=40.0, beta1=400.0,
                     lr0=0.003, lr1=0.0003, deadline=None):
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for k in range(steps):
        if (k & 31) == 0 and deadline is not None and time.time() > deadline:
            break
        f = k / max(1, steps - 1)
        beta = beta0 * (beta1 / beta0) ** f
        lr = lr0 * (lr1 / lr0) ** f
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gm = float(G.max())
        E = np.exp((G - gm) * beta)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum())
        if not np.isfinite(S) or S <= 0:
            break
        Gr = (E @ X) / S
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        m = b1 * m + (1.0 - b1) * Gt
        v = b2 * v + (1.0 - b2) * (Gt * Gt)
        mhat = m / (1.0 - b1 ** (k + 1))
        vhat = v / (1.0 - b2 ** (k + 1))
        step = mhat / (np.sqrt(vhat) + eps)
        X = X - lr * step
        X = _norm_rows(X)
    return X

def _lp_polish(X, seconds, t_end):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    X = _norm_rows(X.copy())
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() < t_end and radius > 1e-14:
        if time.time() > t_end - 0.4:
            break
        G = X @ X.T
        c = G[iu0, iu1]
        mcur = float(c.max())
        thr = max(20.0 * radius, 1e-9)
        sel = c > mcur - thr
        if int(sel.sum()) > 1500:
            idx = np.argpartition(-c, 1500)[:1500]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = mask
        I, J = iu0[sel], iu1[sel]
        p = len(I)
        if p == 0:
            break
        csel = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            break
        if res.status != 0:
            radius *= 0.5
            continue
        Y = X + res.x[:-1].reshape(n, d)
        Y = _norm_rows(Y)
        vv = _max_cos(Y)
        if vv < best:
            X, best = Y, vv
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
        if time.time() - (t_end - seconds) > seconds:
            break
    return X, best

def solve():
    import time
    t0 = time.time()
    deadline = t0 + 165.0
    rng = np.random.default_rng(0)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    for a in (xb, xr):
        if a is not None and a.shape == (76, 15):
            a = _norm_rows(a.astype(float))
            if np.all(np.isfinite(a)):
                cands.append(a)
    if not cands:
        R = rng.normal(size=(76, 15))
        return _norm_rows(R)
    bv = 1e9
    X0 = cands[0]
    for c in cands:
        vv = _max_cos(c)
        if vv < bv:
            bv, X0 = vv, c.copy()
    n, d = 76, 15
    N = n * d
    # active set slack < 1e-7
    G0 = X0 @ X0.T
    np.fill_diagonal(G0, -2.0)
    m0 = float(G0.max())
    iu0, iu1 = np.triu_indices(n, 1)
    c_all = (X0 @ X0.T)[iu0, iu1]
    slack = m0 - c_all
    act = slack < 1e-7
    Iact, Jact = iu0[act], iu1[act]
    p = len(Iact)
    # contact counts with 1e-9 for rattlers
    Gall = X0 @ X0.T
    np.fill_diagonal(Gall, -2.0)
    mall = float(Gall.max())
    Cmask = Gall > mall - 1e-9
    cnt = Cmask.sum(axis=1)
    rattlers = np.where(cnt == 0)[0]
    # build Jacobian
    Kmax = 20
    modes = None
    if p > 0:
        cact = c_all[act]
        Jd = np.zeros((p, N), dtype=float)
        # vectorized fill via loop (p~500, fast)
        for k in range(p):
            i, j = int(Iact[k]), int(Jact[k])
            ck = float(cact[k])
            Jd[k, i * d:(i + 1) * d] = X0[j] - ck * X0[i]
            Jd[k, j * d:(j + 1) * d] = X0[i] - ck * X0[j]
        # smallest singular vectors via svds
        try:
            from scipy.sparse import csr_matrix
            from scipy.sparse.linalg import svds
            kk = int(min(Kmax, min(p, N) - 1))
            if kk < 1:
                kk = 1
            Js = csr_matrix(Jd)
            u, s, vt = svds(Js, k=kk, which='SM', random_state=0, tol=1e-6, maxiter=20000)
            order = np.argsort(s)
            s = s[order]; vt = vt[order]
            modes = vt.copy()
        except Exception:
            modes = None
        if modes is None:
            try:
                U, S, Vt = np.linalg.svd(Jd, full_matrices=False)
                order = np.argsort(S)
                # smallest last; take up to Kmax smallest
                take = min(Kmax, Vt.shape[0])
                modes = Vt[order[:take]] if False else Vt[np.argsort(S)[-take:]]
                # sort ascending
                # recompute norms order by S
                ss = S[np.argsort(S)[-take:]]
                oo = np.argsort(ss)
                modes = modes[oo]
            except Exception:
                modes = None
    if modes is None or modes.shape[0] < 5:
        # fallback random tangent modes
        Z = rng.normal(size=(20, N))
        for k in range(20):
            M = Z[k].reshape(n, d)
            dot = np.sum(M * X0, axis=1, keepdims=True)
            M = M - dot * X0
            Z[k] = (M.ravel() / (np.linalg.norm(M.ravel()) + 1e-12))
        modes = Z
    Kavail = modes.shape[0]
    Xbest = X0.copy()
    best = bv
    amps = [0.03, 0.06, 0.12, 0.24]
    Ks = [5, 10, 15, 20]
    lp_used = 0.0
    lp_budget = 20.0
    trial = 0
    max_trials = 60
    while trial < max_trials and time.time() < deadline - 3.0:
        # reserve lp budget + margin
        if time.time() > deadline - (lp_budget - lp_used) - 8.0:
            break
        K = Ks[trial % len(Ks)]
        K = min(K, Kavail)
        amp = amps[(trial // len(Ks)) % len(amps)]
        coeff = rng.standard_normal(K)
        cn = np.linalg.norm(coeff)
        if cn == 0 or not np.isfinite(cn):
            trial += 1
            continue
        coeff /= cn
        Dvec = coeff @ modes[:K]
        Dmat = Dvec.reshape(n, d).copy()
        # project radial (should already be tangent)
        dot = np.sum(Dmat * X0, axis=1, keepdims=True)
        Dmat = Dmat - dot * X0
        dn = np.linalg.norm(Dmat.ravel())
        if dn > 1e-12:
            Dmat = Dmat / dn
        Y = Xbest + amp * Dmat
        # optionally larger motion to rattlers
        if len(rattlers) > 0 and (trial % 2 == 0):
            extra = 0.15 + 0.05 * (trial % 3)
            Rr = rng.normal(size=(len(rattlers), d))
            # tangentize
            Xr = Xbest[rattlers]
            dotr = np.sum(Rr * Xr, axis=1, keepdims=True)
            Rr = Rr - dotr * Xr
            rn = np.linalg.norm(Rr, axis=1, keepdims=True) + 1e-12
            Rr /= rn
            Y[rattlers] = Y[rattlers] + extra * Rr
        Y = _norm_rows(Y)
        # fast vectorized log-sum-exp Adam anneal beta 40->400 ~1500 steps
        Y = _anneal_lse_adam(Y, steps=1500, beta0=40.0, beta1=400.0,
                             lr0=0.003, lr1=0.0003, deadline=deadline)
        if time.time() > deadline - 2.0:
            vv = _max_cos(Y)
            if vv < best:
                Xbest, best = Y, vv
            break
        vv = _max_cos(Y)
        # only polish if promising (within 0.01 of best or better) to save LP budget
        do_polish = False
        if vv < best:
            do_polish = True
        elif vv < best + 0.002 and (trial % 4 == 0) and (lp_budget - lp_used) > 3.0:
            do_polish = True
        if do_polish and (lp_budget - lp_used) > 0.8 and time.time() < deadline - 3.0:
            sec = min(2.0, lp_budget - lp_used, max(0.5, deadline - time.time() - 2.0))
            t_end = time.time() + sec
            t_s = time.time()
            Yp, vp = _lp_polish(Y, sec, t_end)
            lp_used += time.time() - t_s
            if vp < best - 1e-12 and np.all(np.isfinite(Yp)):
                Xbest, best = Yp, vp
                # keep X0 as original for modes per spec; but also allow hopping from new best?
                # stay on original Jacobian to keep exploring different basin combos
            elif vv < best - 1e-12:
                Xbest, best = Y, vv
        else:
            if vv < best - 1e-12:
                Xbest, best = Y, vv
        trial += 1
    # final short LP polish if budget remains and time allows
    if (lp_budget - lp_used) > 0.8 and time.time() < deadline - 2.0:
        sec = min(lp_budget - lp_used, max(0.5, deadline - time.time() - 1.0))
        t_end = time.time() + sec
        try:
            Yp, vp = _lp_polish(Xbest, sec, t_end)
            if vp < best and np.all(np.isfinite(Yp)):
                Xbest, best = Yp, vp
        except Exception:
            pass
    Xbest = _norm_rows(Xbest)
    return Xbest
