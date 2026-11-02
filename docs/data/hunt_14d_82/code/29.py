import time
import numpy as np
from scipy.optimize import linprog, linear_sum_assignment
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _lp_polish(X, deadline, radius_init=2e-3, cap=1500, max_it=25):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = float(radius_init)
    iu = np.triu_indices(n, 1)
    it = 0
    while it < max_it and radius > 1e-13:
        it += 1
        if time.time() >= deadline:
            break
        G = X @ X.T
        c = G[iu]
        m = float(np.max(c))
        thr = m - max(20.0 * radius, 1e-9)
        sel = c > thr
        ns = int(np.sum(sel))
        if ns == 0:
            break
        if ns > cap:
            idx = np.argpartition(-c, cap)[:cap]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx] = True
            sel = sel & mask
        I = iu[0][sel]; J = iu[1][sel]
        cc = c[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cc, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True); yn[yn == 0] = 1.0
        y = y / yn
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return X

def _quench(X, steps, beta0, beta1, lr0, lr1):
    X = X.copy()
    n, d = X.shape
    m1 = np.zeros_like(X); m2 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    p1 = 1.0; p2 = 1.0
    l0, l1 = np.log(lr0), np.log(lr1)
    e0, e1 = np.log(beta0), np.log(beta1)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        lr = float(np.exp(l0 + (l1 - l0) * frac))
        beta = float(np.exp(e0 + (e1 - e0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        gm = float(G.max())
        E = np.exp(np.clip(beta * (G - gm), -30.0, 0.0))
        np.fill_diagonal(E, 0.0)
        Gr = E @ X
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        p1 *= b1; p2 *= b2
        m1 = b1 * m1 + (1.0 - b1) * Gt
        m2 = b2 * m2 + (1.0 - b2) * (Gt * Gt)
        mh = m1 / (1.0 - p1); vh = m2 / (1.0 - p2)
        upd = mh / (np.sqrt(vh) + eps)
        X = X - lr * upd
        nr = np.linalg.norm(X, axis=1, keepdims=True); nr[nr == 0] = 1.0
        X = X / nr
    return X

def _greedy(n, d, rng, C=4000):
    x0 = rng.standard_normal(d); x0 /= np.linalg.norm(x0)
    L = [x0]
    for k in range(1, n):
        Z = rng.standard_normal((C, d))
        Z /= np.linalg.norm(Z, axis=1, keepdims=True)
        A = np.stack(L, axis=0)
        S = Z @ A.T
        w = S.max(axis=1)
        L.append(Z[int(np.argmin(w))].copy())
    return np.stack(L, axis=0)

def _best_hole(fixed, rng, n_try=200000, chunk=25000):
    d = fixed.shape[1]
    best_h = None; best_v = 1e9
    n_chunks = n_try // chunk
    for _ in range(n_chunks):
        Z = rng.standard_normal((chunk, d))
        Z /= np.linalg.norm(Z, axis=1, keepdims=True)
        S = Z @ fixed.T
        w = S.max(axis=1)
        j = int(np.argmin(w))
        if float(w[j]) < best_v:
            best_v = float(w[j]); best_h = Z[j].copy()
    return best_h, best_v

def _refine_hole(h, fixed, steps=80, beta=120.0, lr=0.08):
    h = h / np.linalg.norm(h)
    for s in range(steps):
        g = fixed @ h
        mm = float(np.max(g))
        w = np.exp(np.clip(beta * (g - mm), -20.0, 0.0))
        grad = w @ fixed
        grad = grad - (grad @ h) * h
        nn = float(np.linalg.norm(grad))
        if not np.isfinite(nn) or nn < 1e-12:
            break
        st = lr / (1.0 + s * 0.05)
        h = h - (st / (nn + 1e-12)) * grad * (float(np.sum(w)) / max(len(w), 1) * 10.0 + 1.0) * 0.3
        h /= (np.linalg.norm(h) + 1e-18)
    return h / np.linalg.norm(h)

def _spectrum(X):
    G = X @ X.T
    try:
        w = np.linalg.eigvalsh(G)
        w = np.sort(w)[::-1]
    except Exception:
        w = np.zeros(X.shape[0])
    return w

def _fingerprint(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    cnt = np.sum(G > m - 1e-3, axis=1)
    hist, _ = np.histogram(cnt, bins=np.arange(-0.5, 8.5))
    spec = _spectrum(X / np.linalg.norm(X, axis=1, keepdims=True))
    return m, cnt, hist.astype(float), spec

def solve():
    t0 = time.time()
    BUDGET = 166.0
    DEAD = t0 + BUDGET
    rng = np.random.default_rng(7)
    n, d = 82, 14
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if len(cands) == 0:
        z = rng.standard_normal((n, d)); z /= np.linalg.norm(z, axis=1, keepdims=True)
        cands.append(z)
    cands.sort(key=lambda c: _max_cos(c))
    Xrec = cands[0].copy()
    archive = []
    def add_arch(X):
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        v = _max_cos(X)
        _, _, hist, spec = _fingerprint(X)
        # diversity check: spectrum distance
        for e in archive:
            ds = float(np.linalg.norm(spec - e["spec"]) / np.sqrt(n))
            if ds < 1e-4 and abs(v - e["val"]) < 2e-6:
                if v < e["val"] - 1e-12:
                    e["X"] = X.copy(); e["val"] = v; e["spec"] = spec; e["hist"] = hist
                return
        archive.append({"X": X.copy(), "val": v, "spec": spec, "hist": hist})
        archive.sort(key=lambda e: e["val"])
        while len(archive) > 8:
            archive.pop()
    add_arch(Xrec)
    # ---- ascending continuation N=76..81 -> 82 ----
    for chain in range(2):
        if time.time() > DEAD - 70:
            break
        cdead = min(time.time() + 19.0, DEAD - 55)
        try:
            N0 = 76 + (chain % 2)
            X = _greedy(N0, d, rng, C=4000)
            if time.time() >= cdead:
                continue
            X = _quench(X, steps=350, beta0=30.0, beta1=600.0, lr0=0.02, lr1=0.002)
            if time.time() + 1.0 < cdead:
                X = _lp_polish(X, min(cdead, time.time() + 0.7), radius_init=2e-3)
            cur = N0
            ok = True
            while cur < 82:
                if time.time() >= cdead - 1.0:
                    ok = False; break
                Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
                h, _ = _best_hole(Xn, rng, n_try=200000, chunk=25000)
                h = _refine_hole(h, Xn, steps=60)
                X = np.vstack([Xn, h.reshape(1, d)])
                X = _quench(X, steps=220, beta0=50.0, beta1=800.0, lr0=0.015, lr1=0.002)
                if time.time() + 0.4 < cdead:
                    X = _lp_polish(X, min(cdead, time.time() + 0.5), radius_init=1.5e-3, cap=1200, max_it=12)
                cur += 1
            if ok and X.shape[0] == 82:
                X = X / np.linalg.norm(X, axis=1, keepdims=True)
                add_arch(X)
        except Exception:
            continue
    # ---- descending continuation N=85/86 -> 82 ----
    for chain in range(2):
        if time.time() > DEAD - 45:
            break
        cdead = min(time.time() + 17.0, DEAD - 30)
        try:
            base = archive[0]["X"].copy()
            kadd = 4 if chain == 0 else 3
            Xn = base / np.linalg.norm(base, axis=1, keepdims=True)
            for _ in range(kadd):
                h, _ = _best_hole(Xn, rng, n_try=30000, chunk=15000)
                h = _refine_hole(h, Xn, steps=40)
                Xn = np.vstack([Xn, h.reshape(1, d)])
            Xn = _quench(Xn, steps=300, beta0=30.0, beta1=500.0, lr0=0.02, lr1=0.002)
            if time.time() + 0.5 < cdead:
                Xn = _lp_polish(Xn, min(cdead, time.time() + 0.6), radius_init=1.5e-3, cap=1200, max_it=10)
            while Xn.shape[0] > 82:
                if time.time() >= cdead - 1.0:
                    break
                mN = Xn.shape[0]
                G = Xn @ Xn.T
                np.fill_diagonal(G, -2.0)
                best_i = -1; best_key = None
                for i in range(mN):
                    mask = np.ones(mN, dtype=bool); mask[i] = False
                    sub = G[np.ix_(mask, mask)]
                    mi = float(np.max(sub))
                    ci = int(np.sum(sub > mi - 1e-7))
                    key = (mi, ci)
                    if best_key is None or key < best_key:
                        best_key = key; best_i = i
                keep = np.ones(mN, dtype=bool); keep[best_i] = False
                Xn = Xn[keep]
                Xn = _quench(Xn, steps=200, beta0=50.0, beta1=800.0, lr0=0.015, lr1=0.002)
                if time.time() + 0.4 < cdead:
                    Xn = _lp_polish(Xn, min(cdead, time.time() + 0.45), radius_init=1.2e-3, cap=1200, max_it=10)
            if Xn.shape[0] == 82:
                add_arch(Xn)
        except Exception:
            continue
    # ---- recombination of distant survivors ----
    ncross = 0
    while time.time() < DEAD - 14 and len(archive) >= 2 and ncross < 5:
        ncross += 1
        try:
            # most distant pair by spectrum + value
            bi = bj = -1; bd = -1.0
            for a in range(len(archive)):
                for b in range(a + 1, len(archive)):
                    ds = float(np.linalg.norm(archive[a]["spec"] - archive[b]["spec"]))
                    if ds > bd:
                        bd = ds; bi = a; bj = b
            A = archive[bi]["X"].copy(); B = archive[bj]["X"].copy()
            # iterative alignment: assignment + Procrustes
            Bal = B.copy()
            for _ in range(2):
                C = A @ Bal.T
                ri, ci = linear_sum_assignment(-C)
                Aperm = np.zeros_like(A)
                Aperm[ri] = A[ri]
                # build matched: row i of Bal matches A[perm[i]]; find perm
                inv = np.zeros(n, dtype=int)
                for r_, c_ in zip(ri, ci):
                    inv[c_] = r_
                Ap = A[inv]
                M = Bal.T @ Ap
                U, _, Vt = np.linalg.svd(M)
                R = U @ Vt
                Bal = Bal @ R
            # point exchange: half from A, fill rest from Bal far from chosen
            idx = rng.choice(n, size=n // 2, replace=False)
            child_pts = [A[i] for i in idx]
            have = np.stack(child_pts, axis=0)
            # greedily add 41 from Bal maximizing distance to have
            order = list(range(n)); rng.shuffle(order)
            cand_pool = Bal[order]
            while len(child_pts) < n:
                S = np.stack(child_pts, axis=0) @ cand_pool.T
                # for each candidate, worst dot to chosen
                w = S.max(axis=0)
                # also penalize closeness among pool? just take min
                j = int(np.argmin(w))
                child_pts.append(cand_pool[j].copy())
                cand_pool = np.delete(cand_pool, j, axis=0)
                if cand_pool.shape[0] == 0:
                    break
            Ch = np.stack(child_pts, axis=0)
            Ch = Ch / np.linalg.norm(Ch, axis=1, keepdims=True)
            Ch = _quench(Ch, steps=300, beta0=60.0, beta1=900.0, lr0=0.012, lr1=0.002)
            if time.time() + 0.5 < DEAD - 8:
                Ch = _lp_polish(Ch, min(DEAD - 8, time.time() + 1.0), radius_init=1.2e-3, cap=1400, max_it=14)
            add_arch(Ch)
        except Exception:
            continue
    # ---- final selection + polish ----
    archive.sort(key=lambda e: e["val"])
    best = archive[0]["X"].copy()
    bv = archive[0]["val"]
    if time.time() < DEAD - 1.0:
        try:
            best = _quench(best, steps=250, beta0=100.0, beta1=1500.0, lr0=0.008, lr1=0.001)
            vq = _max_cos(best)
            if vq < bv + 0.002:
                best = _lp_polish(best, min(DEAD - 0.3, time.time() + max(1.0, DEAD - time.time() - 0.6)), radius_init=8e-4, cap=2000, max_it=30)
        except Exception:
            pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
