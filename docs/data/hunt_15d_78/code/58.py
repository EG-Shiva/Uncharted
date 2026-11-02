import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _pair_vec(X, iu0, iu1):
    G = X @ X.T
    return G[iu0, iu1]

def _whiten_seed(rng, n=78, d=15):
    X = rng.standard_normal((n, d))
    for _ in range(4):
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        X = X / nr
        mu = X.mean(axis=0, keepdims=True)
        X = X - mu
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        X = X / nr
        C = X.T @ X + 1e-9 * np.eye(d)
        try:
            w, V = np.linalg.eigh(C)
        except Exception:
            break
        w = np.maximum(w, 1e-8)
        invs = V @ np.diag(1.0 / np.sqrt(w)) @ V.T
        X = X @ invs * np.sqrt(n / d)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    return X

def _basis_seed(rng, n=78, d=15):
    X = np.zeros((n, d))
    for b in range(5):
        A = rng.standard_normal((d, d))
        Q, _ = np.linalg.qr(A)
        # random signs
        s = np.where(rng.random(d) < 0.5, -1.0, 1.0)
        Q = Q * s[None, :]
        X[b*d:(b+1)*d, :] = Q
    E = rng.standard_normal((3, d))
    E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    X[75:, :] = E
    perm = rng.permutation(n)
    X = X[perm]
    X = X + 0.004 * rng.standard_normal((n, d))
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    return X

def _adam_quench(X0, iu0, iu1, stages, lr0, deadline):
    n, d = X0.shape
    a = n / d
    X = X0.copy()
    m1 = np.zeros_like(X)
    m2 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    step = 0
    bestX = X.copy()
    bestV = _max_cos(X)
    for beta, nsteps, lam_m, lam_c, lr in stages:
        for _ in range(nsteps):
            if time.time() > deadline:
                return bestX, bestV, X
            step += 1
            G = X @ X.T
            gv = G[iu0, iu1]
            mx = float(gv.max())
            e = np.exp(beta * (gv - mx))
            S = float(e.sum())
            w = e / max(S, 1e-300)
            # repulsion grad W@X
            # build dense W on the fly without full zeros each time
            W = np.zeros((n, n), dtype=np.float64)
            W[iu0, iu1] = w
            W = W + W.T
            Gr = W @ X
            # isotropy penalties
            if lam_m != 0.0:
                s = X.sum(axis=0)  # (d,)
                Gr = Gr + (2.0 * lam_m) * np.broadcast_to(s[None, :], (n, d))
            if lam_c != 0.0:
                C = X.T @ X - a * np.eye(d)
                Gr = Gr + (4.0 * lam_c) * (X @ C)
            # tangential projection
            dot = (Gr * X).sum(axis=1, keepdims=True)
            Gt = Gr - dot * X
            m1 = b1 * m1 + (1 - b1) * Gt
            m2 = b2 * m2 + (1 - b2) * (Gt * Gt)
            m1h = m1 / (1 - b1 ** step)
            m2h = m2 / (1 - b2 ** step)
            upd = lr * m1h / (np.sqrt(m2h) + eps)
            # keep update tangential
            ud = (upd * X).sum(axis=1, keepdims=True)
            upd = upd - ud * X
            # clip large steps
            un = np.linalg.norm(upd, axis=1, keepdims=True)
            # limit per-row step to 0.05
            scale = np.minimum(1.0, 0.05 / np.maximum(un, 1e-12))
            upd = upd * scale
            X = X - upd
            X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
            if (step % 100) == 0:
                v = _max_cos(X)
                if v < bestV:
                    bestV = v
                    bestX = X.copy()
        # evaluate at stage end
        v = _max_cos(X)
        if v < bestV:
            bestV = v
            bestX = X.copy()
    return bestX, bestV, X

def _lp_polish(X, deadline, cap=2000, max_it=40):
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except Exception:
        return X, _max_cos(X)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    bestX = X.copy()
    radius = 4e-3
    iu0, iu1 = np.triu_indices(n, 1)
    nv = n * d + 1
    obj = np.zeros(nv)
    obj[-1] = 1.0
    it = 0
    while time.time() + 0.6 < deadline and radius > 1e-13 and it < max_it:
        it += 1
        G = X @ X.T
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        thr = max(20.0 * radius, 2e-9)
        sel = c_all > m - thr
        nnz = int(np.count_nonzero(sel))
        if nnz == 0:
            radius *= 0.5
            continue
        if nnz > cap:
            idx = np.where(sel)[0]
            part = np.argpartition(-c_all[idx], cap)[:cap]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[part]] = True
            sel = mask
        I = iu0[sel]; J = iu1[sel]
        p = len(I)
        c_sel = c_all[sel]
        cols = np.concatenate([I[:, None]*d + np.arange(d), J[:, None]*d + np.arange(d), np.full((p,1), n*d)], axis=1).ravel()
        rows = np.repeat(np.arange(p), 2*d+1)
        vals = np.concatenate([X[J], X[I], -np.ones((p,1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n*d))), shape=(n, nv)).tocsr()
        bounds = [(-radius, radius)]*(n*d) + [(None, None)]
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=bounds, method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        if not np.all(np.finite(yn)) or np.any(yn < 1e-12):
            radius *= 0.5
            continue
        y = y / yn
        v = _max_cos(y)
        if v < best:
            X = y; best = v; bestX = y.copy()
            radius = min(radius*1.5, 0.02)
        else:
            radius *= 0.5
    return bestX, best

def solve():
    t0 = time.time()
    HARD = t0 + 172.0
    n, d = 78, 15
    iu0, iu1 = np.triu_indices(n, 1)
    # load fallback base
    base = None
    baseV = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.asarray(np.load(fn), dtype=np.float64)
            if a.shape != (78, 15):
                continue
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            v = _max_cos(a)
            if v < baseV:
                baseV = v
                base = a.copy()
        except Exception:
            continue
    # build 8 independent isotropic seeds
    seeds = []
    for i in range(8):
        rng = np.random.default_rng(1000 + i)
        if i % 2 == 0:
            seeds.append(_basis_seed(rng))
        else:
            seeds.append(_whiten_seed(rng))
    # quench budget: ~10.5s each
    results = []  # (val, Xbest, gv_sorted)
    quench_end = t0 + 105.0
    per = 10.5
    for i, s0 in enumerate(seeds):
        now = time.time()
        if now > quench_end - 1.0 or now > HARD - 55.0:
            break
        dl = min(now + per, quench_end, HARD - 50.0)
        # stages: beta increasing, isotropy released gradually
        stages = [
            (40.0, 500, 2.0e-3, 5.0e-4, 0.008),
            (120.0, 500, 5.0e-4, 1.0e-4, 0.005),
            (350.0, 500, 0.0, 0.0, 0.003),
        ]
        # adjust steps to fit time: estimate ~0.004s/step -> 1500 steps ~6s
        try:
            Xb, vb, _ = _adam_quench(s0, iu0, iu1, stages, 0.008, dl)
        except Exception:
            continue
        gv = np.sort(_pair_vec(Xb, iu0, iu1))
        results.append((vb, Xb, gv))
    # sort by value
    results.sort(key=lambda t: t[0])
    # keep diverse: Procrustes/chordal proxy via sorted Gram vec distance
    diverse = []
    for vb, Xb, gv in results:
        ok = True
        for _, _, gv2 in diverse:
            dist = float(np.linalg.norm(gv - gv2) / np.sqrt(len(gv)))
            if dist < 2e-4:
                ok = False
                break
        if ok:
            diverse.append((vb, Xb, gv))
        if len(diverse) >= 4:
            break
    if len(diverse) == 0 and len(results) > 0:
        diverse = [(results[0][0], results[0][1], results[0][2])]
    # release phase: refine top-2 with zero penalty at high beta
    refined = []
    rel_deadline = min(HARD - 35.0, time.time() + 20.0)
    for k in range(min(2, len(diverse))):
        vb, Xb, gv = diverse[k]
        if time.time() > rel_deadline - 1.0:
            refined.append((vb, Xb))
            continue
        stages2 = [(600.0, 400, 0.0, 0.0, 0.002), (1000.0, 200, 0.0, 0.0, 0.001)]
        try:
            Xr, vr, _ = _adam_quench(Xb, iu0, iu1, stages2, 0.002, rel_deadline)
        except Exception:
            refined.append((vb, Xb))
            continue
        refined.append((vr, Xr))
    # also consider base as fallback
    if base is not None:
        refined.append((baseV, base))
    refined.sort(key=lambda t: t[0])
    # LP polish best two with remaining time
    bestX = refined[0][1].copy()
    bestV = refined[0][0]
    # polish top two, keep best
    cands_lp = refined[:2]
    # allocate lp time
    lp_total = max(5.0, min(35.0, HARD - time.time() - 3.0))
    per_lp = lp_total / max(1, len(cands_lp))
    polished = []
    for vb, Xc in cands_lp:
        if time.time() > HARD - 4.0:
            polished.append((vb, Xc))
            continue
        dl = time.time() + per_lp
        dl = min(dl, HARD - 2.0)
        try:
            Xp, vp = _lp_polish(Xc, dl)
        except Exception:
            polished.append((vb, Xc))
            continue
        polished.append((vp, Xp))
    polished.sort(key=lambda t: t[0])
    bestV, bestX = polished[0]
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    # safety: if somehow worse than base and base exists, return base polished? return best found
    # ensure no NaN
    bestX = np.where(np.isfinite(bestX), bestX, 0.0)
    nr = np.linalg.norm(bestX, axis=1, keepdims=True)
    nr = np.where(nr < 1e-12, 1.0, nr)
    bestX = bestX / nr
    return bestX
