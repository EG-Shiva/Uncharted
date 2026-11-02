import time
import numpy as np

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _tangent_bases(X):
    n, d = X.shape
    dm = d - 1
    Ts = np.zeros((n, d, dm))
    Eye = np.eye(d)
    for i in range(n):
        x = X[i]
        k = int(np.argmax(np.abs(x)))
        idx = [j for j in range(d) if j != k]
        E = Eye[:, idx]  # d x dm
        # orthogonalize against x
        coeff = x @ E  # dm
        Eo = E - np.outer(x, coeff)
        # QR to orthonormalize
        Q, _ = np.linalg.qr(Eo)
        # Q is d x dm, ensure tangent (Q.T @ x ~ 0); fix sign if needed
        # qr may flip if rank deficient (should not be)
        Ts[i] = Q
    return Ts

def _build_modes(X, slack_thr=1e-7, K=20):
    n, d = X.shape
    dm = d - 1
    Nt = n * dm
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    iu = np.triu_indices(n, 1)
    c = G[iu]
    sel = (m - c) < slack_thr
    I = iu[0][sel]
    J = iu[1][sel]
    p = len(I)
    # contacts per point for rattlers
    cnt = np.zeros(n, dtype=int)
    if p > 0:
        np.add.at(cnt, I, 1)
        np.add.at(cnt, J, 1)
    rattlers = np.where(cnt == 0)[0]
    if p == 0:
        return None, None, rattlers, cnt, m
    # dense Jacobian p x (n*d)
    # build block-wise to save time
    Jfull_blocks = []  # not full, build Jr directly
    Ts = _tangent_bases(X)
    # J_block per point: p x d, start zero
    # For efficiency build Jr = zeros(p, Nt)
    Jr = np.zeros((p, Nt), dtype=np.float64)
    # c_act
    cact = c[sel]
    # X rows
    # for each active pair k: row has X[J]-c*X[I] at I, X[I]-c*X[J] at J
    # Jr col block i = Jblock_i @ Ti
    # So we can accumulate row by row using Ti
    # vectorized: for each i, find rows where I==i or J==i
    # use loop over p (p~450, cheap)
    # Precompute Ti for multiply
    for k in range(p):
        i = int(I[k]); j = int(J[k])
        cc = float(cact[k])
        vi = X[j] - cc * X[i]  # d
        vj = X[i] - cc * X[j]  # d
        # project onto tangent bases: vi @ Ti , vj @ Tj
        Jr[k, i * dm:(i + 1) * dm] = vi @ Ts[i]
        Jr[k, j * dm:(j + 1) * dm] = vj @ Ts[j]
    # smallest singular vectors via svds
    Vt = None
    try:
        from scipy.sparse.linalg import svds
        kk = int(min(K, min(Jr.shape) - 1))
        if kk >= 1:
            # svds needs k < min dim
            u, s, vt = svds(Jr, k=kk, which='SM', maxiter=20000, tol=1e-4)
            # svds returns ascending? sort by s ascending to get smallest first
            order = np.argsort(s)
            Vt = vt[order, :]  # kk x Nt
    except Exception:
        Vt = None
    if Vt is None:
        # fallback: economy SVD + random null projection basis
        try:
            U, S, Vth = np.linalg.svd(Jr, full_matrices=False)
            # row space = Vth (p x Nt). Generate random null vectors
            rngf = np.random.default_rng(0)
            R = rngf.normal(size=(K, Nt))
            # project out row space: R_null = R - (R @ Vth.T) @ Vth
            coeff = R @ Vth.T  # K x p
            Rnull = R - coeff @ Vth
            # normalize
            nr = np.linalg.norm(Rnull, axis=1, keepdims=True) + 1e-12
            Vt = Rnull / nr
        except Exception:
            Vt = None
    return Vt, Ts, rattlers, cnt, m

def _modes_to_dx(Vt, Ts):
    # Vt: K x Nt, Ts: n x d x dm -> Dx: K x n x d
    K, Nt = Vt.shape
    n, d, dm = Ts.shape
    Dx = np.zeros((K, n, d))
    for k in range(K):
        v = Vt[k]
        for i in range(n):
            vb = v[i * dm:(i + 1) * dm]
            Dx[k, i, :] = Ts[i] @ vb
    # each Dx[k] has Fro norm 1 (isometry)
    return Dx

def _anneal(X, steps, beta0, beta1, lr, t_deadline):
    n, d = X.shape
    m1 = np.zeros_like(X)
    v1 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    lr0 = lr
    logr = np.log(beta1 / beta0)
    # preallocate
    for k in range(steps):
        if (k & 31) == 0 and time.time() > t_deadline:
            break
        f = k / max(1, steps - 1)
        beta = beta0 * np.exp(logr * f)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        mval = float(G.max())
        # stable exp
        E = np.exp(beta * (G - mval))
        np.fill_diagonal(E, 0.0)
        S = float(E.sum())
        if not np.isfinite(S) or S <= 0:
            break
        # grad of LSE (mean over pairs)
        # S_half = S/2
        Gr = (E @ X) / (S * 0.5)
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        b1p *= b1
        b2p *= b2
        m1 *= b1
        m1 += (1.0 - b1) * Gt
        v1 *= b2
        v1 += (1.0 - b2) * (Gt * Gt)
        mhat = m1 / (1.0 - b1p)
        vhat = v1 / (1.0 - b2p)
        X = X - lr0 * mhat / (np.sqrt(vhat) + eps)
        # retract
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X /= nr
    return X

def _lp_polish(X, time_budget, t_deadline):
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except Exception:
        return X, _maxcos(X)
    X = _norm(X)
    n, d = X.shape
    best = _maxcos(X)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    t0 = time.time()
    while time.time() < t_deadline and (time.time() - t0) < time_budget and radius > 1e-14:
        if time.time() > t_deadline - 0.5:
            break
        G = X @ X.T
        c = G[iu]
        mcur = float(c.max())
        thr = max(20 * radius, 1e-9)
        sel = c > mcur - thr
        if int(sel.sum()) > 1500:
            idx = np.argpartition(-c, 1500)[:1500]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = mask
        I = iu[0][sel]; J = iu[1][sel]
        p = int(sel.sum())
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            break
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _maxcos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return X, best

def solve():
    t0 = time.time()
    t_dead = t0 + 165.0
    # load
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
            a = _norm(a.astype(float))
            # sanitize
            a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
            a = _norm(a)
            cands.append(a)
    if len(cands) == 0:
        rng0 = np.random.default_rng(0)
        X = rng0.normal(size=(76, 15))
        return _norm(X)
    bestX = cands[0].copy()
    bv = _maxcos(bestX)
    for c in cands[1:]:
        v = _maxcos(c)
        if v < bv:
            bestX, bv = c.copy(), v
    Xbase = bestX.copy()
    base_v = bv
    n, d = 76, 15

    # build floppy modes from base
    Vt, Ts, rattlers, cnt, mcur = _build_modes(Xbase, slack_thr=1e-7, K=20)
    if Vt is None:
        # fallback to random tangent modes
        Ts = _tangent_bases(Xbase)
        rngf = np.random.default_rng(1)
        Vt = rngf.normal(size=(20, n * (d - 1)))
        Vt /= np.linalg.norm(Vt, axis=1, keepdims=True) + 1e-12
        rattlers = np.array([], dtype=int)
    K = Vt.shape[0]
    Dx_modes = _modes_to_dx(Vt, Ts)  # K x n x d

    amps = [0.03, 0.06, 0.12, 0.24]
    lp_used = 0.0
    lp_budget_total = 20.0

    rng = np.random.default_rng(20261101)
    trial = 0
    # reserve time for final polish
    while True:
        if time.time() > t_dead - 8.0:
            break
        # need enough time for an anneal (~0.4s) + check
        if time.time() > t_dead - 3.0:
            break
        trial += 1
        amp = amps[(trial - 1) % len(amps)]
        # cycle K_use 5..20
        Kuse = 5 + ((trial - 1) % 4) * 5
        Kuse = min(Kuse, K)
        coeff = rng.normal(size=Kuse)
        nc = float(np.linalg.norm(coeff))
        if nc == 0 or not np.isfinite(nc):
            continue
        coeff /= nc
        combo = np.zeros((n, d))
        for k in range(Kuse):
            combo += coeff[k] * Dx_modes[k]
        # combo Fro=1
        Xp = Xbase + amp * combo
        # extra rattler motion
        if len(rattlers) > 0:
            extra_amp = amp * 1.5 + 0.05
            # deterministic per trial extra noise
            Rr = rng.normal(size=(len(rattlers), d))
            # project to tangent at Xbase rattler points
            for ii, idx in enumerate(rattlers):
                x = Xbase[idx]
                r = Rr[ii]
                r = r - float(r @ x) * x
                nr = float(np.linalg.norm(r))
                if nr > 1e-12 and np.isfinite(nr):
                    r /= nr
                    Xp[idx] += extra_amp * r
                else:
                    Xp[idx] += 0.0
        Xp = _norm(Xp)
        # fast vectorized LSE Adam anneal beta 40->400, ~1500 steps
        # shorten steps if low on time
        steps = 1500
        if time.time() > t_dead - 25:
            steps = 900
        Xann = _anneal(Xp, steps=steps, beta0=40.0, beta1=400.0, lr=0.015, t_deadline=t_dead - 6.0)
        av = _maxcos(Xann)
        # only polish promising ones to save LP budget
        # threshold: within 0.004 of base (annealed is unpolished so higher)
        if av < base_v + 0.004 and lp_used < lp_budget_total and time.time() < t_dead - 6.0:
            budget = min(2.5, lp_budget_total - lp_used, max(0.5, t_dead - 6.0 - time.time()))
            if budget >= 0.4:
                lp_s = time.time()
                Xp2, pv = _lp_polish(Xann, time_budget=budget, t_deadline=t_dead - 4.0)
                lp_used += time.time() - lp_s
                if pv < bv - 1e-12 and np.isfinite(pv):
                    bv = pv
                    bestX = Xp2.copy()
                    # update base to new optimum and rebuild modes (topology changed)
                    Xbase = bestX.copy()
                    base_v = bv
                    # rebuild if enough time left
                    if time.time() < t_dead - 30:
                        try:
                            Vt2, Ts2, ratt2, cnt2, _ = _build_modes(Xbase, slack_thr=1e-7, K=20)
                            if Vt2 is not None:
                                Vt, Ts, rattlers = Vt2, Ts2, ratt2
                                K = Vt.shape[0]
                                Dx_modes = _modes_to_dx(Vt, Ts)
                        except Exception:
                            pass
                    continue
        # also track best annealed even without polish? keep for final polish
        # if annealed alone beats bv (rare since bv polished), accept
        if av < bv - 1e-12:
            bv = av
            bestX = Xann.copy()
            Xbase = bestX.copy()
            base_v = bv
        if trial > 200:
            break
    # final short LP polish on best if time and LP budget left
    if time.time() < t_dead - 2.0 and lp_used < lp_budget_total:
        budget = min(lp_budget_total - lp_used, max(0.5, t_dead - 2.5 - time.time()))
        if budget > 0.3:
            try:
                Xf, vf = _lp_polish(bestX, time_budget=budget, t_deadline=t_dead - 1.0)
                if vf < bv and np.isfinite(vf):
                    bestX, bv = Xf, vf
            except Exception:
                pass
    bestX = _norm(bestX)
    return bestX
