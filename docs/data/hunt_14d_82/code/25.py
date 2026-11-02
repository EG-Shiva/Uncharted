import time
import numpy as np
from scipy.optimize import linprog, linear_sum_assignment
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lp_polish(X, seconds, radius_init=2e-3, cap=1100):
    t_end = time.time() + seconds
    X = _norm(X.copy())
    n, d = X.shape
    best = _max_cos(X)
    radius = radius_init
    iu = np.triu_indices(n, 1)
    it = 0
    while time.time() < t_end and radius > 1e-14 and it < 20:
        it += 1
        G = X @ X.T
        c_all = G[iu]
        m = float(c_all.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = c_all > thr
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > cap:
            # keep largest
            idx = np.where(sel)[0]
            vals = c_all[idx]
            part = np.argpartition(-vals, cap - 1)[:cap]
            m2 = np.zeros_like(sel, dtype=bool)
            m2[idx[part]] = True
            sel = m2
        I = iu[0][sel]; J = iu[1][sel]
        c = c_all[sel]
        p = len(I); nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        Y = X + res.x[:-1].reshape(n, d)
        Y = _norm(Y)
        v = _max_cos(Y)
        if v < best:
            X, best = Y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return X

def _adam_quench(X, T, steps, lr0, lr1, t_end=None):
    X = _norm(X.copy())
    n, d = X.shape
    m1 = np.zeros_like(X); m2 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    l0, l1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        if t_end is not None and s % 50 == 0 and time.time() > t_end:
            break
        frac = s / max(steps - 1, 1)
        lr = float(np.exp(l0 + (l1 - l0) * frac))
        p = 2.0 + 4.0 * frac
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        D = G - T
        W = np.zeros_like(G)
        pos = D > 0
        if np.any(pos):
            W[pos] = p * (D[pos] ** (p - 1.0))
            # cap to avoid explosion
            np.clip(W, 0, 1e4, out=W)
        Gr = W @ X
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        b1p *= b1; b2p *= b2
        m1 = b1 * m1 + (1.0 - b1) * Gt
        m2 = b2 * m2 + (1.0 - b2) * (Gt * Gt)
        mh = m1 / (1.0 - b1p); vh = m2 / (1.0 - b2p)
        upd = mh / (np.sqrt(vh) + eps)
        X = X - lr * upd
        X = _norm(X)
    return X

def _greedy_init(n, d, rng, n_cand_pool=5000):
    X = []
    # first random
    v = rng.standard_normal(d); v /= np.linalg.norm(v)
    X.append(v)
    while len(X) < n:
        A = np.stack(X, axis=0)
        cand = rng.standard_normal((n_cand_pool, d))
        cand = _norm(cand)
        dots = cand @ A.T
        worst = dots.max(axis=1)
        X.append(cand[int(np.argmin(worst))])
    return np.stack(X, axis=0)

def _best_hole_insert(X_keep, rng, n_total=200000, batch=20000):
    n_keep, d = X_keep.shape
    best_v = None; best_worst = 1e9
    nb = n_total // batch
    for _ in range(nb):
        cand = _norm(rng.standard_normal((batch, d)))
        dots = cand @ X_keep.T
        worst = dots.max(axis=1)
        i = int(np.argmin(worst))
        if worst[i] < best_worst:
            best_worst = float(worst[i])
            best_v = cand[i].copy()
    # local refine of new point on sphere: greedy hill climb
    v = best_v.copy()
    cur = best_worst
    step = 0.02
    for _ in range(300):
        prop = v + step * rng.standard_normal(d)
        prop /= np.linalg.norm(prop)
        w = float(np.max(X_keep @ prop))
        if w < cur:
            v, cur = prop, w
        else:
            step *= 0.97
            if step < 1e-6:
                break
    return v

def _deletion_victim(X):
    n = X.shape[0]
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    # m_i = max excluding i
    best_i, best_m, best_deg = 0, 1e9, 1e9
    m_global = float(G.max())
    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        sub = G[np.ix_(mask, mask)]
        mi = float(sub.max())
        if mi < best_m - 1e-12:
            best_m = mi; best_i = i
            best_deg = float(np.sum(G[i] > m_global - 1e-3))
        elif abs(mi - best_m) < 1e-12:
            deg = float(np.sum(G[i] > m_global - 1e-3))
            if deg > best_deg:
                best_i = i; best_deg = deg
    return best_i

def _spectrum(X):
    G = X @ X.T
    try:
        w = np.linalg.eigvalsh(G)
    except Exception:
        w = np.zeros(X.shape[0])
    return np.sort(w)

def _fingerprint(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    cnt = np.sum(G > m - 1e-3, axis=1)
    return np.sort(cnt.astype(float))

def _align(B, A):
    # align B to A: find perm + rotation, B,A (n,d)
    n, d = A.shape
    R = np.eye(d)
    perm = np.arange(n)
    for _ in range(3):
        Br = B @ R
        D = A @ Br.T
        try:
            r, c = linear_sum_assignment(-D)
            P = np.empty(n, dtype=int)
            P[r] = c
            # P maps A-index -> B-index; reorder B to match A
            Bp = B[P]
        except Exception:
            Bp = B
            P = np.arange(n)
        M = Bp.T @ A
        try:
            U, _, Vt = np.linalg.svd(M, full_matrices=False)
            Rnew = U @ Vt
        except Exception:
            Rnew = R
        # update B permuted? keep original B but update R to include perm effect next iter
        # compute rotation aligning Bp to A then fold: simpler set B = Bp and reset perm
        B = Bp
        R = np.eye(d) if _ == 0 else R
        # recompute R for new B
        M2 = B.T @ A
        try:
            U2, _, Vt2 = np.linalg.svd(M2, full_matrices=False)
            R = U2 @ Vt2
        except Exception:
            pass
        perm = P
    return _norm(B @ R)

def solve():
    t0 = time.time()
    BUDGET = 168.0
    def remain():
        return BUDGET - (time.time() - t0)
    rng = np.random.default_rng(0)
    nT, d = 82, 14
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.standard_normal((nT, d))
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (nT, d):
        z = rng.standard_normal((nT, d))
        m = min(z.size, xb.size)
        z.ravel()[:m] = xb.ravel()[:m]
        xb = z
    xb = _norm(xb)
    best_val = _max_cos(xb)
    X_best = xb.copy()
    # quick initial polish if time
    if remain() > 30:
        X_best = _lp_polish(X_best, min(2.0, remain() - 25))
        best_val = _max_cos(X_best)
    archive = [(best_val, X_best.copy(), _spectrum(X_best), _fingerprint(X_best))]
    def add_archive(X):
        nonlocal archive
        v = _max_cos(X)
        s = _spectrum(X); f = _fingerprint(X)
        for (vv, _, ss, _) in archive:
            if abs(vv - v) < 5e-5 and float(np.linalg.norm(ss - s)) < 0.5:
                # duplicate basin, keep better
                if v < vv:
                    # replace
                    for k in range(len(archive)):
                        if abs(archive[k][0] - vv) < 1e-12:
                            archive[k] = (v, X.copy(), s, f)
                            break
                return v
        archive.append((v, X.copy(), s, f))
        archive.sort(key=lambda t: t[0])
        archive = archive[:6]
        return v
    # Phase A: upward continuation from N=76..79
    N0_list = [76, 77, 78, 79]
    for si, N0 in enumerate(N0_list):
        if remain() < 28:
            break
        seed_deadline = time.time() + min(20.0, remain() - 20)
        # build init
        if si % 2 == 0:
            X0 = _greedy_init(N0, d, rng)
        else:
            X0 = _norm(rng.standard_normal((N0, d)))
        v0 = _max_cos(X0)
        X0 = _adam_quench(X0, v0 - 3e-4, 180, 0.02, 0.002, t_end=seed_deadline)
        Xcur = X0
        # grow
        ok = True
        for nn in range(N0, nT):
            if time.time() > seed_deadline:
                ok = False; break
            v = _hole_vector if False else None
            h = _best_hole_insert(Xcur, rng, n_total=200000, batch=20000)
            Xcur = np.vstack([Xcur, h[None, :]])
            Xcur = _norm(Xcur)
            mcur = _max_cos(Xcur)
            Xcur = _adam_quench(Xcur, mcur - 5e-4, 150, 0.015, 0.002, t_end=seed_deadline)
            if remain() < 5:
                ok = False; break
            # short polish after each insertion
            Xcur = _lp_polish(Xcur, min(0.45, max(0.2, seed_deadline - time.time()) * 0.3))
            if time.time() > seed_deadline:
                ok = False; break
        if not ok:
            continue
        # final polish
        if time.time() < seed_deadline:
            Xcur = _lp_polish(Xcur, min(1.2, seed_deadline - time.time()))
        v = _max_cos(Xcur)
        add_archive(Xcur)
        if v < best_val - 1e-12:
            best_val = v; X_best = Xcur.copy()
        # update archive ordering var
        # refresh archive list reference (add_archive uses closure, need rebinding? use global)
        # archive already updated inside
    # Phase B: downward from N=84,86
    for Nbig in [86, 84]:
        if remain() < 18:
            break
        seed_deadline = time.time() + min(18.0, remain() - 10)
        Xb = _norm(rng.standard_normal((Nbig, d)))
        mb = _max_cos(Xb)
        Xb = _adam_quench(Xb, mb - 5e-4, 200, 0.02, 0.002, t_end=seed_deadline)
        Xb = _lp_polish(Xb, min(0.6, max(0.2, (seed_deadline - time.time()) * 0.2)))
        cur = Xb
        while cur.shape[0] > nT and time.time() < seed_deadline:
            vic = _deletion_victim(cur)
            mask = np.ones(cur.shape[0], dtype=bool); mask[vic] = False
            cur = cur[mask]
            mc = _max_cos(cur)
            cur = _adam_quench(cur, mc - 5e-4, 120, 0.015, 0.002, t_end=seed_deadline)
            cur = _lp_polish(cur, min(0.4, max(0.15, (seed_deadline - time.time()) * 0.25)))
        if cur.shape[0] != nT:
            continue
        cur = _lp_polish(cur, min(1.0, max(0.2, seed_deadline - time.time())))
        v = _max_cos(cur)
        add_archive(cur)
        if v < best_val - 1e-12:
            best_val = v; X_best = cur.copy()
    # Phase C: recombine distant survivors
    # rebuild sorted archive copy
    # ensure archive variable accessible (closure may have shadowed); fetch via sorting
    # we maintained archive list in closure; use it
    nrec = 0
    while remain() > 8 and nrec < 10:
        nrec += 1
        if len(archive) < 2:
            break
        # pick most distant pair by spectrum distance
        best_pair = None; best_d = -1
        for i in range(len(archive)):
            for j in range(i + 1, len(archive)):
                dd = float(np.linalg.norm(archive[i][2] - archive[j][2])) + 200.0 * abs(archive[i][0] - archive[j][0])
                if dd > best_d:
                    best_d = dd; best_pair = (i, j)
        i, j = best_pair
        A = archive[i][1]; Bc = archive[j][1]
        try:
            Bal = _align(Bc.copy(), A.copy())
        except Exception:
            Bal = Bc.copy()
        # point exchange masks
        for trial in range(2):
            if remain() < 6:
                break
            mask = rng.random(nT) < 0.5
            # bias: take low-contact points from each? simple random is fine
            Ch = A.copy()
            Ch[~mask] = Bal[~mask]
            Ch = _norm(Ch + 0.005 * rng.standard_normal(Ch.shape))
            mc = _max_cos(Ch)
            Ch = _adam_quench(Ch, mc - 4e-4, 160, 0.015, 0.002)
            Ch = _lp_polish(Ch, 0.5)
            v = _max_cos(Ch)
            add_archive(Ch)
            if v < best_val - 1e-12:
                best_val = v; X_best = Ch.copy()
    # Final polish
    if remain() > 1.5:
        X_best = _lp_polish(X_best, min(3.0, remain() - 0.8))
        best_val = _max_cos(X_best)
    X_best = _norm(X_best)
    return X_best
