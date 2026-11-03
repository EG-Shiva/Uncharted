import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _per_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1), G

def _anneal(X, steps, b0, b1, lr0, lr1):
    n, d = X.shape
    for s in range(steps):
        f = s / max(steps - 1, 1)
        beta = b0 * (b1 / b0) ** f
        lr = lr0 * (lr1 / lr0) ** f
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = np.exp(beta * (G - m))
        np.fill_diagonal(E, 0.0)
        S = float(E.sum())
        if not np.isfinite(S) or S == 0:
            break
        Grad = (E @ X) / S
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Grad = Grad - dot * X
        # clip huge steps
        gn = np.sqrt((Grad * Grad).sum()) + 1e-30
        if gn > 5.0:
            Grad = Grad * (5.0 / gn)
        X = X - lr * Grad
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _max_descent(X, steps=200, lr=0.01, tol=3e-3):
    for _ in range(steps):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        A = (G > (m - tol)).astype(float)
        np.fill_diagonal(A, 0.0)
        AX = A @ X
        coeff = (A * G).sum(axis=1, keepdims=True)
        F = AX - coeff * X
        fn = np.sqrt((F * F).sum()) + 1e-30
        if fn > 10.0:
            F = F * (10.0 / fn)
        X = X - lr * F
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _refine_void(v0, Xrem, beta=150.0, steps=120, lr=0.08):
    v = v0 / (np.linalg.norm(v0) + 1e-30)
    for t in range(steps):
        dots = Xrem @ v
        m = float(dots.max())
        W = np.exp(beta * (dots - m))
        S = float(W.sum()) + 1e-300
        g = (W @ Xrem) / S
        gt = g - float(np.dot(g, v)) * v
        st = lr / (1.0 + t * 0.02)
        v = v - st * gt
        nrm = np.linalg.norm(v) + 1e-30
        v = v / nrm
    return v

def _best_void(Xrem, rng, ncand=20000):
    n1, d = Xrem.shape
    cand = rng.normal(size=(ncand, d))
    cand = cand / np.linalg.norm(cand, axis=1, keepdims=True)
    S = cand @ Xrem.T
    smax = S.max(axis=1)
    bi = int(np.argmin(smax))
    v0 = cand[bi]
    try:
        v0 = _refine_void(v0, Xrem, beta=150.0, steps=120, lr=0.08)
    except Exception:
        pass
    return v0

def _e8_roots():
    d = 8
    roots = []
    for a in range(d):
        for b in range(a + 1, d):
            for s1 in (-1.0, 1.0):
                for s2 in (-1.0, 1.0):
                    v = np.zeros(d)
                    v[a] = s1
                    v[b] = s2
                    roots.append(v)
    # 2^8 sign combos with even parity
    for mask in range(256):
        bits = [(1.0 if ((mask >> k) & 1 else -1.0) for k in range(d)]
        # even number of -1  <=> product +1
        prod = 1.0
        for b in bits:
            prod *= b
        if prod > 0:
            v = 0.5 * np.array(bits)
            roots.append(v)
    R = np.array(roots, dtype=float)
    R = R / np.linalg.norm(R, axis=1, keepdims=True)
    return R  # 240x8

def _lp_polish(X, deadline):
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except Exception:
        return X
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        Gfull = X @ X.T
        c = Gfull[iu]
        m = float(c.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = c > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 2500:
            idx = np.where(sel)[0]
            order = np.argsort(-c[idx])[:2500]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            y = X + res.x[:-1].reshape(n, d)
        except Exception:
            radius *= 0.5
            continue
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def solve():
    t0 = time.time()
    deadline = t0 + 172.0
    rng = np.random.default_rng(0)
    n, d = 138, 8
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            v = np.load(fn).astype(float)
            if v.shape == (n, d):
                v = v / np.linalg.norm(v, axis=1, keepdims=True)
                c = _max_cos(v)
                if c < best_val:
                    best_val = c
                    best = v.copy()
        except Exception:
            pass
    if best is None:
        best = rng.normal(size=(n, d))
        best = best / np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    archive = [(best_val, best.copy())]

    def consider(Y):
        nonlocal best, best_val, archive
        Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
        v = _max_cos(Y)
        if v < best_val - 1e-12:
            best_val = v
            best = Y.copy()
            archive.append((v, Y.copy()))
            archive.sort(key=lambda t: t[0])
            archive = archive[:8]
            return True
        elif v < best_val + 0.004 and len(archive) < 8:
            # diverse keeper if not duplicate
            if all(_max_cos((Y + A[1]) * 0.5) > 1e-6 or abs(v - A[0]) > 1e-7 for A in archive):
                archive.append((v, Y.copy()))
                archive.sort(key=lambda t: t[0])
                archive = archive[:8]
        return False

    # E8 roots for seeding
    try:
        E8 = _e8_roots()
    except Exception:
        E8 = None

    # Phase 1: diverse basins (fast, no LP)
    n_init = 10
    for k in range(n_init):
        if time.time() > deadline - 40:
            break
        if k == 0 and E8 is not None:
            idx = rng.choice(E8.shape[0], size=n, replace=False)
            Y = E8[idx].copy() + 0.02 * rng.normal(size=(n, d))
            Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
            Y = _anneal(Y, 1200, 15.0, 300.0, 0.03, 0.003)
            Y = _max_descent(Y, 150, 0.008, 3e-3)
        elif k == 1:
            Y = best + 0.5 * rng.normal(size=(n, d))
            Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
            Y = _anneal(Y, 1200, 12.0, 280.0, 0.03, 0.003)
            Y = _max_descent(Y, 150, 0.008, 3e-3)
        else:
            Y = rng.normal(size=(n, d))
            Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
            Y = _anneal(Y, 1500, 10.0, 250.0, 0.03, 0.002)
            Y = _max_descent(Y, 150, 0.008, 3e-3)
        consider(Y)

    # Phase 2: topology-changing basin hopping
    it = 0
    while time.time() < deadline - 28.0 and it < 300:
        it += 1
        # pick base from archive (bias to best but keep diversity)
        rpick = rng.random()
        if rpick < 0.6:
            base = archive[0][1]
        else:
            base = archive[rng.integers(len(archive))][1]
        op = it % 4
        Y = base.copy()
        if op == 0:
            # core-targeted heavy shake of worst points
            per, _ = _per_max(Y)
            order = np.argsort(-per)
            kcore = 20
            victims = order[:kcore]
            Y[victims] += 0.45 * rng.normal(size=(kcore, d))
            mask = np.ones(n, dtype=bool)
            mask[victims] = False
            Y[mask] += 0.02 * rng.normal(size=(int(mask.sum()), d))
            Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
            Y = _anneal(Y, 800, 20.0, 300.0, 0.025, 0.002)
            Y = _max_descent(Y, 120, 0.007, 3e-3)
        elif op == 1:
            # peel worst and reinsert into best void
            per, _ = _per_max(Y)
            victim = int(np.argmax(per + rng.normal(size=n) * 0.002))
            Xrem = np.delete(Y, victim, axis=0)
            # relax remainder briefly
            Xrem = _anneal(Xrem, 300, 20.0, 200.0, 0.02, 0.003)
            vnew = _best_void(Xrem, rng, ncand=20000)
            Y = np.vstack([Xrem, vnew[np.newaxis, :]])
            # shuffle order to avoid bias (deterministic perm)
            perm = rng.permutation(n)
            Y = Y[perm]
            Y = _anneal(Y, 700, 20.0, 300.0, 0.025, 0.002)
            Y = _max_descent(Y, 120, 0.007, 3e-3)
        elif op == 2:
            # multi-void rebirth: replace 5 worst with voids
            per, _ = _per_max(Y)
            order = np.argsort(-per)
            victims = order[:5]
            keep = np.delete(Y, victims, axis=0)
            # find 5 voids sequentially
            newpts = []
            cur = keep.copy()
            for _ in range(5):
                vnew = _best_void(cur, rng, ncand=12000)
                newpts.append(vnew)
                cur = np.vstack([cur, vnew[np.newaxis, :]])
            Y[victims] = np.array(newpts)
            Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
            Y = _anneal(Y, 800, 20.0, 320.0, 0.025, 0.002)
            Y = _max_descent(Y, 120, 0.007, 3e-3)
        else:
            # global moderate shake
            Y = Y + 0.12 * rng.normal(size=(n, d))
            Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
            Y = _anneal(Y, 800, 15.0, 300.0, 0.025, 0.002)
            Y = _max_descent(Y, 120, 0.007, 3e-3)
        improved = consider(Y)
        # opportunistic LP on clear improvements
        if improved and time.time() < deadline - 35.0:
            try:
                Yp = _lp_polish(Y, min(deadline - 28.0, time.time() + 2.5))
                consider(Yp)
            except Exception:
                pass

    # Phase 3: final tight polish of top candidates
    archive.sort(key=lambda t: t[0])
    # polish best two
    for k in range(min(2, len(archive))):
        if time.time() >= deadline - 1.0:
            break
        Y = archive[k][1].copy()
        try:
            budget = (deadline - time.time()) / (2 - k)
            budget = min(max(budget, 1.0), 14.0)
            Yp = _lp_polish(Y, time.time() + budget)
            consider(Yp)
        except Exception:
            pass
    archive.sort(key=lambda t: t[0])
    best = archive[0][1].copy()
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
