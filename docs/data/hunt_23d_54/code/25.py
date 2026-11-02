import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _proj_A(G, t):
    # symmetric cap + unit diag
    G2 = G.copy()
    # cap off-diagonal
    # keep symmetry: G symmetric so min preserves it
    np.fill_diagonal(G2, 0.0)
    np.minimum(G2, t, out=G2)
    np.fill_diagonal(G2, 1.0)
    # ensure exact symmetry
    # (minimum already symmetric)
    return G2

def _proj_B(G, rank=23):
    # rank-rank PSD projection via eigh
    # G symmetric
    w, V = np.linalg.eigh((G + G.T) * 0.5)
    # w ascending, take largest `rank`
    idx = np.argsort(w)[::-1]
    w = w[idx]
    V = V[:, idx]
    wt = np.maximum(w[:rank], 0.0)
    # reconstruct: V_rank * diag(wt) * V_rank.T
    Vr = V[:, :rank]
    # scale columns by wt
    # G = (Vr*sqrt(wt)) @ (Vr*sqrt(wt)).T
    s = np.sqrt(wt)
    Y = Vr * s[np.newaxis, :]
    Gb = Y @ Y.T
    # symmetrize
    Gb = (Gb + Gb.T) * 0.5
    return Gb, Y, w

def _extract_points(Y):
    # Y is 54x23 factor, rows may not be unit; normalize
    n = Y.shape[0]
    nr = np.linalg.norm(Y, axis=1, keepdims=True)
    nr[nr < 1e-12] = 1.0
    X = Y / nr
    # fix possible NaN / zero
    X = np.where(np.isfinite(X), X, 0.0)
    nr2 = np.linalg.norm(X, axis=1, keepdims=True)
    nr2[nr2 < 1e-12] = 1.0
    X = X / nr2
    return X

def _bm_smooth(X_init, t_target, steps, k0, k1, lr0, lr1, window):
    X = X_init.copy()
    n, d = X.shape
    lk0 = np.log(k0); lk1 = np.log(k1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    for s in range(steps):
        f = s / max(steps - 1, 1)
        k = np.exp(lk0 + (lk1 - lk0) * f)
        lr = np.exp(ll0 + (ll1 - ll0) * f)
        G = X @ X.T
        # mask for near violators
        # window shrinks? use fixed
        sel = G > (t_target - window)
        np.fill_diagonal(sel, False)
        if np.any(sel):
            M = G[sel] - t_target
            # k*M clipped to avoid overflow
            KM = k * M
            np.clip(KM, -30.0, 8.0, out=KM)
            W = np.zeros_like(G)
            W[sel] = np.exp(KM)
            F = W @ X
        else:
            F = np.zeros_like(X)
        dot = np.sum(F * X, axis=1, keepdims=True)
        Ft = F - dot * X
        deg = np.sum(np.abs(Ft), axis=1, keepdims=True) / d + 1.0
        # normalize step by mean norm to keep lr meaningful
        X = X - lr * Ft / deg
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X /= nr
    return X

def _lp_polish(X, seconds, t_deadline):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if time.time() > t_deadline:
            break
        if radius < 1e-14:
            break
        Gfull = X @ X.T
        c_all = Gfull[iu0]
        m = float(c_all.max())
        thr = m - max(20.0 * radius, 2e-6)
        sel = c_all > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 700:
            idx = np.where(sel)[0]
            order = np.argsort(c_all[idx])[::-1][:700]
            nsel = np.zeros_like(sel, dtype=bool)
            nsel[idx[order]] = True
            sel = nsel
        I = iu0[0][sel]; J = iu0[1][sel]
        c = c_all[sel]
        p = len(I)
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate(
                [I[:, None] * d + np.arange(d),
                 J[:, None] * d + np.arange(d),
                 np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix(
                (X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
        # early stop if radius tiny and no progress
        if time.time() - start > seconds:
            break
    return X

def solve():
    t0 = time.time()
    BUDGET = 172.0
    t_deadline = t0 + BUDGET
    rng = np.random.RandomState(12345)
    n, d = 54, 23
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = rng.randn(n, d)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    if xb.shape != (n, d):
        tmp = rng.randn(n, d)
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        xb = tmp
    try:
        xr = np.load("record.npy").astype(float)
        xr /= np.linalg.norm(xr, axis=1, keepdims=True)
        if xr.shape == (n, d):
            if _max_cos(xr) < _max_cos(xb):
                xb = xr.copy()
    except Exception:
        pass
    best = _max_cos(xb)
    X_best = xb.copy()

    # ---- Phase A: Gram SDP alternating projections ----
    targets = [0.04755, 0.04745, 0.0473, 0.0470]
    pool = []  # list of (val, X)
    pool.append((best, X_best.copy()))
    best_resid = 1e9
    best_resid_info = None

    # time for AP: at most ~35s
    ap_deadline = t0 + 38.0
    for ti, t in enumerate(targets):
        if time.time() > ap_deadline:
            break
        if time.time() > t_deadline - 60.0:
            break
        for restart in range(2):
            if time.time() > ap_deadline:
                break
            if restart == 0:
                G = X_best @ X_best.T
                # symmetric noise to leave basin
                Ns = rng.randn(n, n) * (0.008 if ti < 2 else 0.015)
                Ns = (Ns + Ns.T) * 0.5
                G = G + Ns
                G = (G + G.T) * 0.5
            else:
                Yr = rng.randn(n, d)
                Yr /= np.linalg.norm(Yr, axis=1, keepdims=True)
                G = Yr @ Yr.T
            Niter = 350 if ti < 2 else 250
            lam = 1.0
            for k in range(Niter):
                if time.time() > ap_deadline:
                    break
                Ga = _proj_A(G, t)
                Gb, Y, w = _proj_B(Ga, rank=23)
                # residual: gap + tail + cap violation of Gb
                # gap
                gap = float(np.linalg.norm(Ga - Gb) / n)
                tail = float(np.sum(np.maximum(-w, 0.0)) + np.sum(w[23:][w[23:] > 0]) if len(w) > 23 else 0.0)
                # max off-diag of Gb
                tmp = Gb.copy()
                np.fill_diagonal(tmp, -2.0)
                mo = float(tmp.max())
                viol = max(0.0, mo - t)
                resid = gap + viol
                if resid < best_resid:
                    best_resid = resid
                    best_resid_info = (t, restart, k, gap, viol, mo)
                # relaxed update
                G = (1.0 - lam) * G + lam * Gb
                # periodic extraction
                if (k % 60 == 59) or (k == Niter - 1):
                    Xtry = _extract_points(Y)
                    v = _max_cos(Xtry)
                    if v < best - 1e-12:
                        best = v
                        X_best = Xtry.copy()
                    # keep in pool if reasonable
                    if v < 0.09:
                        pool.append((v, Xtry.copy()))
                    # also try row-normalized Ga factor? Ga not PSD, skip
            # end iters
    # keep pool small: sort and keep top 6
    pool = sorted(pool, key=lambda z: z[0])[:6]

    # ---- Phase B: Burer-Monteiro penalty homotopy + tangential smooth ----
    # use remaining time up to deadline-18s
    bm_deadline = t_deadline - 18.0
    # build candidate inits: pool + noisy variants of X_best
    cands = []
    for v, X in pool:
        cands.append(X.copy())
    # add a couple random Gram-extracted inits
    for _ in range(2):
        Yr = rng.randn(n, d)
        Yr /= np.linalg.norm(Yr, axis=1, keepdims=True)
        # quick smooth at loose target to diversify
        cands.append(Yr)

    # homotopy targets below current best
    # recompute best
    # iterate over candidates round-robin with different t_target
    it = 0
    # predefine t_targets ladder relative to best at start of phase
    base_best = best
    ladder = [base_best - 0.00015, 0.04745, 0.04730, base_best - 0.0006]
    # ensure ladder values are sensible (< base_best+0.001)
    while time.time() < bm_deadline and it < 28:
        if time.time() > bm_deadline:
            break
        Xc = cands[it % len(cands)].copy()
        # add kick that changes topology: small global + stronger on few points
        # use contact-guided choice on X_best to pick points? use random subset
        if it % 3 == 0:
            Xc = Xc + rng.randn(n, d) * 0.006
        elif it % 3 == 1:
            # kick 4 random points harder
            idx = rng.choice(n, size=4, replace=False)
            Xc[idx] += rng.randn(4, d) * 0.06
            Xc += rng.randn(n, d) * 0.003
        else:
            Xc += rng.randn(n, d) * 0.012
        Xc /= np.linalg.norm(Xc, axis=1, keepdims=True)
        t_target = ladder[it % len(ladder)]
        # clamp t_target to be below current best but not absurdly low
        # if best improved, lower ladder a bit
        if t_target > best - 0.00005:
            t_target = best - 0.00025
        if t_target < 0.0465:
            t_target = 0.0468
        steps = 500
        # shorter steps when time low
        if bm_deadline - time.time() < 20:
            steps = 300
        k0, k1 = 60.0, 2500.0
        lr0, lr1 = 0.06, 0.004
        window = 0.06 if it % 2 == 0 else 0.03
        try:
            Xs = _bm_smooth(Xc, t_target, steps, k0, k1, lr0, lr1, window)
        except Exception:
            it += 1
            continue
        vs = _max_cos(Xs)
        # second short anneal centered at actual max for polishing without LP
        try:
            if vs < best + 0.005:
                Xs2 = _bm_smooth(Xs, vs - 0.0004, 200, 200.0, 4000.0, 0.010, 0.0015, 0.015)
                v2 = _max_cos(Xs2)
                if v2 < vs:
                    Xs, vs = Xs2, v2
        except Exception:
            pass
        if vs < best - 1e-12:
            best = vs
            X_best = Xs.copy()
            # update pool
            cands.append(Xs.copy())
            if len(cands) > 10:
                cands = cands[-10:]
        elif vs < best + 0.002 and len(pool) < 8:
            pass
        it += 1

    # ---- Phase C: short LP polish of champion ----
    try:
        left = t_deadline - time.time()
        # reserve 1s for return
        if left > 3.0:
            lp_sec = min(left - 1.0, 14.0)
            X_best = _lp_polish(X_best, lp_sec, t_deadline - 0.8)
            best = _max_cos(X_best)
    except Exception:
        pass
    X_best /= np.linalg.norm(X_best, axis=1, keepdims=True)
    # final safety: replace non-finite
    if not np.all(np.isfinite(X_best)):
        X_best = xb.copy()
    return X_best
