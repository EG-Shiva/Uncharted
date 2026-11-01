import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm_rows(X):
    X = np.asarray(X, dtype=float)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _quench(X_init, steps, T0, T1, lr, mu=0.9, deadline=None):
    X = np.array(X_init, dtype=float, copy=True)
    n, d = X.shape
    V = np.zeros_like(X)
    lt0, lt1 = np.log(T0), np.log(T1)
    G0 = X @ X.T
    np.fill_diagonal(G0, -2.0)
    best_m = float(G0.max())
    best_X = X.copy()
    for s in range(steps):
        if deadline is not None and (s % 200 == 0) and time.time() > deadline:
            break
        frac = s / max(1, steps - 1)
        T = float(np.exp(lt0 + (lt1 - lt0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -1e100)
        m = float(G.max())
        Q = (G - m) / T
        np.clip(Q, -50, 0, out=Q)
        E = np.exp(Q)
        np.fill_diagonal(E, 0.0)
        rs = E.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        Grad = (E @ X) / rs
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        gn = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        bad = gn[:, 0] > 2.0
        if np.any(bad):
            Gt[bad] *= (2.0 / gn[bad])
        V = mu * V + Gt
        X = X - lr * V
        X = _norm_rows(X)
        if (s % 150 == 0) or (s == steps - 1):
            Gc = X @ X.T
            np.fill_diagonal(Gc, -2.0)
            vc = float(Gc.max())
            if vc < best_m:
                best_m = vc
                best_X = X.copy()
    return best_X, best_m

def _lp_polish(X0, seconds, deadline=None, r0=2e-3, cap=1600):
    X = _norm_rows(np.array(X0, dtype=float, copy=True))
    try:
        best = _max_cos(X)
    except Exception:
        return X
    bestX = X.copy()
    n, d = X.shape
    radius = float(r0)
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if deadline is not None and time.time() > deadline:
            break
        if radius < 1e-14:
            break
        cmat = (X @ X.T)[iu]
        m = float(cmat.max())
        band = max(25.0 * radius, 2e-4)
        sel = cmat > m - band
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > cap:
            order = np.argsort(-cmat[sel])[:cap]
            I, J = I[order], J[order]
            csel = cmat[sel][order]
        else:
            csel = cmat[sel]
            p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cc = np.concatenate([I[:, None] * d + np.arange(d),
                             J[:, None] * d + np.arange(d),
                             np.full((p, 1), n * d)], 1).ravel()
        vv = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vv, (rows, cc)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
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
        if v < best - 1e-15:
            X, best = y, v
            bestX = y.copy()
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return bestX

def solve():
    t0 = time.time()
    HARD_END = t0 + 173.0
    rng = np.random.default_rng(717171)
    n, d = 71, 12
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=float)
            if a.shape == (71, 12) and np.all(np.isfinite(a)):
                a = _norm_rows(a)
                cands.append((_max_cos(a), a))
        except Exception:
            pass
    if not cands:
        a = _norm_rows(rng.standard_normal((71, 12)))
        cands.append((_max_cos(a), a))
    cands.sort(key=lambda t: t[0])
    best_score, best_X = cands[0][0], cands[0][1].copy()
    cur_X, cur_score = best_X.copy(), best_score
    # ---- cyclic C71 single-orbit search ----
    # cos table
    K = 71
    ctab = np.cos(2.0 * np.pi * np.arange(K) / K)
    cyc_cands = []  # (t, f, w)
    # quick LP for weights given frequencies f (6,)
    def eval_freq(f):
        # f shape (6,) values 1..35
        S = np.arange(1, 36)[:, None] * np.asarray(f, dtype=int)[None, :]
        S = np.mod(S, K)
        C = ctab[S]  # 35x6
        # min t s.t. C w <= t, sum w=1, w>=0
        c_obj = np.zeros(7); c_obj[6] = 1.0
        A_ub = np.hstack([C, -np.ones((35, 1))])
        b_ub = np.zeros(35)
        A_eq = np.zeros((1, 7)); A_eq[0, :6] = 1.0
        b_eq = np.ones(1)
        bounds = [(0, None)] * 6 + [(None, None)]
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                          bounds=bounds, method="highs")
        except Exception:
            return None, None
        if res.status != 0:
            return None, None
        return float(res.x[6]), np.array(res.x[:6])
    CY_END = t0 + 28.0
    ntry = 0
    # include a few structured frequency sets
    struct = []
    struct.append(np.array([1, 2, 4, 8, 16, 32]) % 36)
    struct[-1][struct[-1] == 0] = 5
    struct.append(np.array([1, 3, 5, 9, 17, 27]))
    struct.append(np.array([1, 5, 11, 19, 29, 33]))
    for f0 in struct:
        f0 = np.clip(f0, 1, 35).astype(int)
        if time.time() > CY_END:
            break
        t, w = eval_freq(f0)
        ntry += 1
        if t is not None:
            cyc_cands.append((t, f0.copy(), w))
    # random sampling
    while time.time() < CY_END and ntry < 2600:
        f = rng.integers(1, 36, size=6)
        # enforce distinct for diversity
        if len(np.unique(f)) < 5:
            continue
        t, w = eval_freq(f)
        ntry += 1
        if t is not None and np.isfinite(t):
            cyc_cands.append((t, f.copy(), w))
    cyc_cands.sort(key=lambda t: t[0])
    # build symmetric codes for top few and quench them
    def build_cyclic(f, w):
        a = np.sqrt(np.maximum(w, 0))
        ssum = float((a * a).sum())
        if ssum <= 0:
            a = np.full(6, 1.0 / np.sqrt(6.0))
        else:
            a = a / np.sqrt(ssum)
        phi = 2.0 * np.pi * np.asarray(f, dtype=float) / 71.0
        X = np.zeros((71, 12))
        for k in range(71):
            for j in range(6):
                ang = k * phi[j]
                X[k, 2 * j] = a[j] * np.cos(ang)
                X[k, 2 * j + 1] = a[j] * np.sin(ang)
        return _norm_rows(X)
    # evaluate top cyclic raw scores (for info) and quench best 4
    topK = cyc_cands[:6]
    for idx, (tt, ff, ww) in enumerate(topK):
        if time.time() > HARD_END - 120:
            break
        try:
            Xs = build_cyclic(ff, ww)
            sc0 = _max_cos(Xs)
        except Exception:
            continue
        # break symmetry with tiny noise then quench
        try:
            N = rng.standard_normal(size=Xs.shape) * 1e-3
            N -= Xs * np.sum(N * Xs, axis=1, keepdims=True)
            Xs2 = _norm_rows(Xs + N)
            # two-stage quench: hot then cold
            Xq, _ = _quench(Xs2, steps=900, T0=0.008, T1=0.0004, lr=0.012, deadline=HARD_END - 95)
            Xq = _lp_polish(Xq, seconds=1.2, deadline=HARD_END - 90, r0=2.5e-3)
            Xq2, _ = _quench(Xq, steps=900, T0=0.002, T1=0.00008, lr=0.008, deadline=HARD_END - 80)
            Xq2 = _lp_polish(Xq2, seconds=1.2, deadline=HARD_END - 75, r0=1.5e-3)
            sc = _max_cos(Xq2)
            if sc < best_score:
                best_score, best_X = sc, Xq2.copy()
                cur_X, cur_score = Xq2.copy(), sc
            elif sc < cur_score:
                cur_X, cur_score = Xq2.copy(), sc
        except Exception:
            continue
    # ---- basin hopping with subset shakes from best/cur ----
    it = 0
    HOP_END = HARD_END - 16.0
    while time.time() < HOP_END:
        it += 1
        if time.time() > HOP_END - 1.0:
            break
        base = cur_X if rng.random() < 0.6 else best_X
        # shake size: mix small and large to change topology
        r = rng.random()
        if r < 0.35:
            m = int(rng.integers(2, 6))
            sigma = float(rng.uniform(0.05, 0.18))
        elif r < 0.75:
            m = int(rng.integers(5, 13))
            sigma = float(rng.uniform(0.15, 0.45))
        else:
            m = int(rng.integers(12, 24))
            sigma = float(rng.uniform(0.3, 0.7))
        idx = rng.choice(n, size=m, replace=False)
        Y = base.copy()
        N = rng.standard_normal(size=(m, d)) * sigma
        # tangent projection
        for ii, jj in enumerate(idx):
            v = N[ii]
            v = v - float(v @ Y[jj]) * Y[jj]
            Y[jj] = Y[jj] + v
        Y[idx] = _norm_rows(Y[idx])
        # occasionally apply coherent rotation to a block to jump basin
        if rng.random() < 0.18:
            # random orthogonal via QR on subset mean-removed? just random tangent drift
            m2 = int(rng.integers(8, 20))
            idx2 = rng.choice(n, size=m2, replace=False)
            Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
            # rotate points partially toward Q applied
            alpha = float(rng.uniform(0.2, 0.6))
            Yr = Y[idx2] @ Q.T
            # keep hemisphere continuity: flip sign if negative dot
            dots = np.sum(Yr * Y[idx2], axis=1)
            flip = dots < 0
            Yr[flip] = -Yr[flip]
            Y[idx2] = _norm_rows((1 - alpha) * Y[idx2] + alpha * Yr)
        T0 = float(rng.uniform(0.003, 0.010))
        T1 = float(rng.uniform(0.00008, 0.0006))
        lr = float(rng.uniform(0.005, 0.014))
        steps = 700 if m > 10 else 500
        try:
            Yq, _ = _quench(Y, steps=steps, T0=T0, T1=T1, lr=lr, deadline=HOP_END)
        except Exception:
            continue
        try:
            # cheap LP only every few iters or if promising
            gtmp = Yq @ Yq.T
            np.fill_diagonal(gtmp, -2.0)
            sc_q = float(gtmp.max())
        except Exception:
            continue
        # threshold to spend LP time
        thresh = min(cur_score, best_score) + (0.004 if (it % 3 == 0) else 0.0015)
        if sc_q < thresh:
            try:
                Yq = _lp_polish(Yq, seconds=0.9, deadline=HOP_END, r0=2e-3)
                sc_q = _max_cos(Yq)
            except Exception:
                pass
        if sc_q < best_score - 1e-12:
            best_score, best_X = sc_q, Yq.copy()
            cur_X, cur_score = Yq.copy(), sc_q
        elif sc_q < cur_score - 1e-12 or rng.random() < 0.08:
            cur_X, cur_score = Yq.copy(), sc_q
    # final polish of best
    try:
        best_X = _lp_polish(best_X, seconds=min(9.0, max(2.0, HARD_END - time.time() - 1.0)), deadline=HARD_END - 0.5, r0=2e-3)
        best_score = _max_cos(best_X)
    except Exception:
        pass
    best_X = _norm_rows(best_X)
    return best_X
