import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _normed(X):
    X = np.asarray(X, dtype=np.float64)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _adam_anneal(X0, steps, beta0, beta1, lr, deadline):
    X = X0.copy()
    n, d = X.shape
    m1 = np.zeros_like(X)
    m2 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for k in range(steps):
        if (k & 31) == 0 and time.time() > deadline:
            break
        f = k / max(1, steps - 1)
        beta = beta0 * (beta1 / beta0) ** f
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        gm = float(G.max())
        E = np.exp((G - gm) * beta)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum())
        if S <= 0 or not np.isfinite(S):
            break
        Gr = (E @ X) / S
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        m1 = b1 * m1 + (1 - b1) * Gt
        m2 = b2 * m2 + (1 - b2) * (Gt * Gt)
        c1 = 1 - b1 ** (k + 1)
        c2 = 1 - b2 ** (k + 1)
        mh = m1 / c1
        vh = m2 / c2
        step = mh / (np.sqrt(vh) + eps)
        X = X - lr * step
        X = _normed(X)
    return X

def _lp_polish(X, budget, deadline, cap=1500):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    X = _normed(X)
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    iu0, iu1 = np.triu_indices(n, 1)
    t0 = time.time()
    while time.time() < deadline and (time.time() - t0) < budget and radius > 1e-14:
        if time.time() > deadline - 0.4:
            break
        G = X @ X.T
        c = G[iu0, iu1]
        m = float(c.max())
        thr = max(20 * radius, 1e-9)
        sel = c > m - thr
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > cap:
            idx = np.argpartition(-c, cap)[:cap]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = mask
        I, J = iu0[sel], iu1[sel]
        csel = c[sel]
        p = len(I)
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            break
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = _normed(y)
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return X, best

def solve():
    import time
    t0 = time.time()
    deadline = t0 + 165.0
    lp_budget_total = 20.0
    lp_used = 0.0
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy")
    except Exception:
        xr = None
    cands = []
    for a in (xb, xr):
        if a is not None:
            try:
                a = np.asarray(a, dtype=np.float64)
                if a.shape == (76, 15):
                    cands.append(_normed(a))
            except Exception:
                pass
    if len(cands) == 0:
        rng0 = np.random.default_rng(0)
        X = rng0.normal(size=(76, 15))
        return _normed(X)
    best = cands[0]
    bv = _max_cos(best)
    for c in cands[1:]:
        v = _max_cos(c)
        if v < bv:
            best, bv = c.copy(), v
    Xbase = best.copy()
    n, d = 76, 15
    # active set slack < 1e-7
    G0 = Xbase @ Xbase.T
    np.fill_diagonal(G0, -2.0)
    m0 = float(G0.max())
    iu0, iu1 = np.triu_indices(n, 1)
    call = G0[iu0, iu1]
    slack = m0 - call
    sel = np.where(slack < 1e-7)[0]
    if sel.size < 10:
        sel = np.argsort(slack)[:500]
    if sel.size > 2500:
        part = np.argpartition(slack, 2500)[:2500]
        sel = part[np.argsort(slack[part])]
    I = iu0[sel]; J = iu1[sel]
    p = len(I)
    cij = call[sel]
    # tangent bases Ui (d x d-1)
    Ub = []
    for i in range(n):
        xi = Xbase[i]
        k = int(np.argmax(np.abs(xi)))
        cols = []
        for j in range(d):
            if j == k:
                continue
            e = np.zeros(d); e[j] = 1.0
            v = e - float(e @ xi) * xi
            cols.append(v)
        V = np.stack(cols, axis=1)  # d x d-1
        Q, _ = np.linalg.qr(V)
        # ensure orthogonal to xi (qr already spans correct space)
        Ub.append(Q)
    Ub = np.stack(Ub, axis=0)  # n x d x d-1
    td = d - 1
    Ntan = n * td
    Jt = np.zeros((p, Ntan), dtype=np.float64)
    for r in range(p):
        i, j = int(I[r]), int(J[r])
        ci = float(cij[r])
        bi = Xbase[j] - ci * Xbase[i]  # d
        bj = Xbase[i] - ci * Xbase[j]
        Jt[r, i * td:(i + 1) * td] = bi @ Ub[i]
        Jt[r, j * td:(j + 1) * td] = bj @ Ub[j]
    K = 20
    modes_tan = None
    try:
        from scipy.sparse.linalg import svds
        import scipy.sparse as sp
        Js = sp.csr_matrix(Jt)
        # smallest singular vectors
        u, s, vt = svds(Js, k=K, which='SM', maxiter=20000, tol=1e-6)
        order = np.argsort(s)
        vt = vt[order, :]
        modes_tan = vt  # K x Ntan
    except Exception:
        modes_tan = None
    if modes_tan is None:
        try:
            C = Jt.T @ Jt
            C = 0.5 * (C + C.T)
            w, V = np.linalg.eigh(C)
            idx = np.argsort(w)[:K]
            modes_tan = V[:, idx].T  # K x Ntan
        except Exception:
            modes_tan = None
    if modes_tan is None:
        rngf = np.random.default_rng(0)
        modes_tan = rngf.normal(size=(K, Ntan))
    # convert to full displacements K x n x d
    Dm = np.zeros((K, n, d), dtype=np.float64)
    for k in range(K):
        v = modes_tan[k]
        for i in range(n):
            y = v[i * td:(i + 1) * td]
            Dm[k, i, :] = Ub[i] @ y
    # contacts per point for rattlers (same threshold)
    cnt = np.zeros(n, dtype=int)
    for r in range(p):
        cnt[int(I[r])] += 1
        cnt[int(J[r])] += 1
    rattlers = np.where(cnt == 0)[0]
    rng = np.random.default_rng(0)
    amps = [0.03, 0.06, 0.12, 0.24]
    Ks = [5, 10, 20]
    Xbest = Xbase.copy()
    bbest = bv
    trial = 0
    sqrtn = float(np.sqrt(n))
    # reserve time for final polish: keep 8s margin
    while time.time() < deadline - 8.0 and trial < 200:
        if time.time() > deadline - 8.0:
            break
        amp = amps[trial % len(amps)]
        Kcur = Ks[(trial // len(amps)) % len(Ks)]
        Kcur = min(Kcur, K)
        coeff = rng.normal(size=Kcur)
        nc = float(np.linalg.norm(coeff))
        if nc == 0 or not np.isfinite(nc):
            coeff = np.ones(Kcur) / np.sqrt(Kcur)
        else:
            coeff = coeff / nc
        Dcomb = np.zeros((n, d), dtype=np.float64)
        for k in range(Kcur):
            Dcomb += coeff[k] * Dm[(trial + k) % K, :, :]
        fn = float(np.linalg.norm(Dcomb))
        if fn > 1e-12 and np.isfinite(fn):
            Dcomb *= (sqrtn / fn)
        else:
            Dcomb = rng.normal(size=(n, d))
            Dcomb -= np.sum(Dcomb * Xbase, axis=1, keepdims=True) * Xbase
            Dcomb *= (sqrtn / max(1e-12, float(np.linalg.norm(Dcomb))))
        # optionally larger motion to rattlers on alternating trials
        if len(rattlers) > 0 and (trial % 2 == 1):
            R = rng.normal(size=(n, d))
            R -= np.sum(R * Xbase, axis=1, keepdims=True) * Xbase
            boost = amp * 2.5
            Dcomb[rattlers, :] += boost * R[rattlers, :] * 2.0
        Xt0 = Xbase + amp * Dcomb
        Xt0 = _normed(Xt0)
        # fast vectorized log-sum-exp Adam anneal, ~1500 steps
        steps = 1500
        # shorten if low on time
        if time.time() > deadline - 30.0:
            steps = 800
        Xt = _adam_anneal(Xt0, steps, 40.0, 400.0, 0.01, deadline - 8.0)
        va = _max_cos(Xt)
        # hill check every trial is cheap; polish only if promising and LP budget left
        need_polish = (va < bbest + 0.004) and (lp_used < lp_budget_total) and (time.time() < deadline - 8.0)
        if need_polish:
            bdg = min(2.5, lp_budget_total - lp_used, max(0.5, (deadline - 8.0 - time.time()) / 2.0))
            tlp0 = time.time()
            Xp, vp = _lp_polish(Xt, bdg, deadline - 5.0)
            lp_used += (time.time() - tlp0)
            if vp < bbest - 1e-12 and np.all(np.isfinite(Xp)):
                Xbest, bbest = Xp, vp
                # keep Xbase at improved? keep original modes per spec, but update best; do not switch base to stay in surgery regime
                # also try to keep annealing from improved point once
                if time.time() < deadline - 10.0:
                    Xextra = _adam_anneal(Xbest, 600, 200.0, 800.0, 0.003, deadline - 8.0)
                    ve = _max_cos(Xextra)
                    if ve < bbest:
                        Xbest, bbest = _normed(Xextra), ve
        else:
            # without polish, accept if strictly better than polished best (rare)
            if va < bbest - 1e-9:
                # quick tiny polish to verify
                if lp_used < lp_budget_total and time.time() < deadline - 8.0:
                    bdg = min(2.0, lp_budget_total - lp_used)
                    tlp0 = time.time()
                    Xp, vp = _lp_polish(Xt, bdg, deadline - 5.0)
                    lp_used += (time.time() - tlp0)
                    if vp < bbest:
                        Xbest, bbest = Xp, vp
                else:
                    Xbest, bbest = Xt, va
        trial += 1
        # time-check every 32 iterations is done inside anneal; outer check each trial is finer
    # final short polish of best if budget left
    if lp_used < lp_budget_total and time.time() < deadline - 4.0:
        bdg = min(lp_budget_total - lp_used, deadline - 4.0 - time.time(), 5.0)
        if bdg > 0.4:
            try:
                Xp, vp = _lp_polish(Xbest, bdg, deadline - 1.0)
                if vp < bbest:
                    Xbest, bbest = Xp, vp
            except Exception:
                pass
    Xbest = _normed(Xbest)
    return Xbest
