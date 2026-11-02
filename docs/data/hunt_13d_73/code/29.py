import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _row_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def _rand_sphere(rng, n, d):
    Y = rng.normal(size=(n, d))
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return Y

def _find_hole(X_keep, rng, n_cand=4000):
    d = X_keep.shape[1]
    C = rng.normal(size=(n_cand, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    dots = C @ X_keep.T
    m = dots.max(axis=1)
    return C[int(np.argmin(m))]

def _refine(X0, steps, a0, a1, lr0, lr1, b1=0.9, b2=0.999, eps=1e-8, t_end=1e30):
    X = X0.copy()
    m1 = np.zeros_like(X)
    v1 = np.zeros_like(X)
    b1p = b1
    b2p = b2
    la0 = np.log(a0); la1 = np.log(a1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    n = X.shape[0]
    for k in range(steps):
        if k % 64 == 0 and time.time() > t_end:
            break
        frac = k / max(1, steps - 1)
        alpha = float(np.exp(la0 + (la1 - la0) * frac))
        lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        cmax = float(G.max())
        # softmax weights, symmetric
        E = np.exp(alpha * (G - cmax))
        np.fill_diagonal(E, 0.0)
        s = float(E.sum() * 0.5)
        if not np.isfinite(s) or s <= 0:
            # fallback to hard max
            M = (G >= cmax - 1e-12).astype(float)
            np.fill_diagonal(M, 0.0)
            tot = float(M.sum() * 0.5)
            if tot <= 0:
                break
            W = M / tot
        else:
            W = E / s
            # W sums to 2 (both triangles); scale to 1 for gradient scale invariance
            W = W * 0.5
        Gt = W @ X
        dot = (Gt * X).sum(axis=1, keepdims=True)
        gt = Gt - dot * X
        m1 = b1 * m1 + (1 - b1) * gt
        v1 = b2 * v1 + (1 - b2) * (gt * gt)
        mhat = m1 / (1 - b1p)
        vhat = v1 / (1 - b2p)
        b1p *= b1
        b2p *= b2
        step = mhat / (np.sqrt(vhat) + eps)
        # keep step tangential for stability / equitable motion
        sd = (step * X).sum(axis=1, keepdims=True)
        step = step - sd * X
        # limit per-point step to avoid blowup
        X = X - lr * step
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X /= nr
    return X

def _lp_polish(X, seconds, t_end):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    it = 0
    while time.time() - start < seconds and time.time() < t_end and radius > 1e-14:
        it += 1
        if it > 60:
            break
        G = X @ X.T
        c = G[iu0]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I = iu0[0][sel]; J = iu0[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 3000:
            # subsample tightest to keep LP cheap
            idx = np.argsort(-c[sel])[:3000]
            I = I[idx]; J = J[idx]
            csel = c[sel][idx]
            p = len(I)
        else:
            csel = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
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
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    SEARCH_END = t0 + 138.0
    rng = np.random.default_rng(12345)
    n, d = 73, 13
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = _rand_sphere(rng, n, d)
    if xb.shape != (n, d):
        xb = _rand_sphere(rng, n, d)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = _max_cos(xb)
    Xbest = xb.copy()

    # quick diverse pool: mostly fresh random (new basins), few big kicks from record
    candidates = []
    # fresh random seeds
    for s in range(34):
        candidates.append(_rand_sphere(rng, n, d))
    # big kicks far from jam (leave graph)
    for sig in [0.5, 0.7, 0.9, 1.0, 0.6, 0.8]:
        Y = xb + sig * rng.normal(size=(n, d))
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        candidates.append(Y)
    # mirrored / structured: random + orthogonal transform of best chunk
    for _ in range(4):
        Y = _rand_sphere(rng, n, d)
        # implant a random 20-point cap from xb rotated randomly to mix topologies
        Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
        Y[:20] = (xb[rng.choice(n, 20, replace=False)] @ Q.T)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        candidates.append(Y)

    # stage-1 cheap anneal from low sharpness (balanced) then select
    scored = []
    for ci, C0 in enumerate(candidates):
        if time.time() > SEARCH_END:
            break
        # coherent cap shake for some to expand core together
        if ci % 7 == 6:
            u = rng.normal(size=(d,)); u /= np.linalg.norm(u)
            proj = C0 @ u
            cap = proj > np.quantile(proj, 0.75)
            ang = 0.12 + 0.1 * rng.random()
            # small rotation of cap points around random tangent axis
            w = rng.normal(size=(d,)); w = w - (w @ u) * u
            nw = np.linalg.norm(w)
            if nw > 1e-9:
                w /= nw
                C0 = C0.copy()
                C0[cap] = np.cos(ang) * C0[cap] + np.sin(ang) * w[None, :]
                C0 /= np.linalg.norm(C0, axis=1, keepdims=True)
        X1 = _refine(C0, steps=650, a0=12.0, a1=140.0, lr0=0.035, lr1=0.004, t_end=SEARCH_END)
        v1 = _max_cos(X1)
        scored.append((v1, X1))
        if v1 < best_val - 1e-12:
            best_val = float(v1)
            Xbest = X1.copy()
        # early aggressive second-stage for very promising fresh basins
        if v1 < best_val + 0.006 and time.time() < SEARCH_END - 10:
            X2 = _refine(X1, steps=900, a0=140.0, a1=550.0, lr0=0.008, lr1=0.0012, t_end=SEARCH_END)
            v2 = _max_cos(X2)
            if v2 < best_val - 1e-12:
                best_val = float(v2)
                Xbest = X2.copy()
                scored.append((v2, X2))
            else:
                # rattler / overload repair: reinsert worst points into holes then short re-anneal
                try:
                    rm = _row_max(X2)
                    cm = float(rm.max())
                    # overloaded = top-2, rattlers = far below max
                    order = np.argsort(-rm)
                    victims = list(order[:1])
                    for i2 in range(n):
                        if rm[i2] < cm - 5e-3:
                            victims.append(i2)
                            if len(victims) >= 3:
                                break
                    victims = victims[:3]
                    Z = X2.copy()
                    keep = np.ones(n, dtype=bool); keep[victims] = False
                    cur_keep = Z[keep].copy()
                    for vi in victims:
                        h = _find_hole(cur_keep, rng, n_cand=5000)
                        Z[vi] = h
                        cur_keep = np.vstack([cur_keep, h[None, :]])
                    Z += 0.005 * rng.normal(size=Z.shape)
                    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
                    Z = _refine(Z, steps=500, a0=60.0, a1=400.0, lr0=0.008, lr1=0.0015, t_end=SEARCH_END)
                    v3 = _max_cos(Z)
                    scored.append((v3, Z))
                    if v3 < best_val - 1e-12:
                        best_val = float(v3)
                        Xbest = Z.copy()
                except Exception:
                    pass

    # final sharpening of top few distinct basins to enforce equitable contacts
    try:
        scored.sort(key=lambda t: t[0])
        seen = scored[:6]
        for v_, X_ in seen:
            if time.time() > SEARCH_END:
                break
            if v_ > best_val + 0.004:
                continue
            Xs = _refine(X_, steps=1100, a0=180.0, a1=700.0, lr0=0.006, lr1=0.0008, t_end=SEARCH_END + 6)
            vs = _max_cos(Xs)
            if vs < best_val - 1e-12:
                best_val = float(vs)
                Xbest = Xs.copy()
    except Exception:
        pass

    # brief LP squeeze on incumbent only, time-guarded
    remain = DEADLINE - time.time() - 2.0
    if remain > 5.0:
        try:
            Xp = _lp_polish(Xbest, seconds=min(remain - 1.0, 18.0), t_end=DEADLINE - 1.0)
            vp = _max_cos(Xp)
            if vp <= best_val:
                Xbest = Xp
                best_val = float(vp)
        except Exception:
            pass
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest
