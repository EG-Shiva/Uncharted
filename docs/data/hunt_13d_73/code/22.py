import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def lp_polish(x, seconds, deadline=1e30):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = max_cos(x)
    except Exception:
        return x
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    it = 0
    while True:
        if time.time() - start >= seconds:
            break
        if time.time() >= deadline:
            break
        if radius < 1e-14:
            break
        if it > 40:
            break
        it += 1
        Gfull = (x @ x.T)[iu0]
        m = float(Gfull.max())
        thresh = max(20.0 * radius, 1e-9)
        sel = Gfull > m - thresh
        # cap constraints to closest ones for speed
        cnt = int(np.sum(sel))
        if cnt > 900:
            # keep top 900
            idx = np.where(sel)[0]
            order = np.argsort(-Gfull[idx])[:900]
            sel2 = np.zeros_like(sel, dtype=bool)
            sel2[idx[order]] = True
            sel = sel2
            cnt = 900
        elif cnt > 500 and cnt <= 1200:
            pass
        # if too many, shrink thresh
        for _ in range(4):
            if cnt <= 1100 or thresh < 1e-12:
                break
            thresh *= 0.5
            sel = Gfull > m - thresh
            cnt = int(np.sum(sel))
        I = iu0[0][sel]
        J = iu0[1][sel]
        csub = Gfull[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csub, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        y = y / nr
        if not np.all(np.isfinite(y)):
            radius *= 0.5
            continue
        v = max_cos(y)
        if v < best:
            x = y
            best = v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def smooth_adam(X0, steps, s0, s1, lr0, lr1, deadline=1e30):
    X = X0.copy()
    n, d = X.shape
    m1 = np.zeros_like(X)
    v1 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = b1, b2
    ls0, ls1 = np.log(s0), np.log(s1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for k in range(steps):
        if k % 50 == 0 and time.time() >= deadline:
            break
        frac = k / (steps - 1) if steps > 1 else 1.0
        s = float(np.exp(ls0 + (ls1 - ls0) * frac))
        lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        cmax = float(G.max())
        # softmax weights, avoid overflow
        E = np.exp(s * (G - cmax))
        np.fill_diagonal(E, 0.0)
        sumE = float(E.sum() * 0.5)
        if sumE == 0 or not np.isfinite(sumE):
            # fallback to hard max
            Mask = (G >= cmax - 1e-12).astype(float)
            np.fill_diagonal(Mask, 0.0)
            tot = float(Mask.sum() * 0.5)
            if tot <= 0:
                continue
            W = Mask / tot
        else:
            W = E / sumE
        Gt = W @ X
        dot = np.sum(Gt * X, axis=1, keepdims=True)
        gt = Gt - dot * X
        m1 *= b1
        m1 += (1 - b1) * gt
        v1 *= b2
        v1 += (1 - b2) * (gt * gt)
        mhat = m1 / (1 - b1p + 1e-30)
        vhat = v1 / (1 - b2p + 1e-30)
        b1p *= b1
        b2p *= b2
        step = mhat / (np.sqrt(vhat) + eps)
        X = X - lr * step
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X = X / nr
    return X

def hole_vec(X, idx, rng, n_cand=3000):
    n, d = X.shape
    C = rng.randn(n_cand, d)
    cn = np.linalg.norm(C, axis=1, keepdims=True)
    cn[cn == 0] = 1.0
    C = C / cn
    others = np.delete(X, idx, axis=0)
    dots = C @ others.T
    md = np.max(dots, axis=1)
    return C[int(np.argmin(md))]

def solve():
    t0 = time.time()
    DEADLINE = t0 + 176.0
    HOP_END = t0 + 132.0
    rng = np.random.RandomState(7)
    n, d = 73, 13
    best = None
    best_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.load(fn).astype(np.float64)
            if v.shape != (n, d):
                continue
            v = v / np.linalg.norm(v, axis=1, keepdims=True)
            if not np.all(np.isfinite(v)):
                continue
            c = max_cos(v)
            if c < best_val:
                best_val = c
                best = v.copy()
        except Exception:
            continue
    if best is None:
        best = rng.randn(n, d)
        best = best / np.linalg.norm(best, axis=1, keepdims=True)
        best_val = max_cos(best)
    sigmas = [0.04, 0.09, 0.02, 0.16, 0.06, 0.012, 0.07, 0.03]
    it = 0
    while time.time() < HOP_END and time.time() < DEADLINE - 10:
        if HOP_END - time.time() < 2.5:
            break
        sigma = sigmas[it % len(sigmas)]
        # perturb from global best
        pert = rng.randn(n, d)
        X = best + sigma * pert
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        # pick jammed index to kill
        try:
            if it % 2 == 0:
                Gg = best @ best.T
                np.fill_diagonal(Gg, -np.inf)
                mg = float(np.max(Gg))
                cnt = np.sum(Gg > mg - 5e-4, axis=1)
                order = np.argsort(-cnt)
                k = min(3, n)
                idx = int(order[rng.randint(0, k)])
            else:
                idx = int(rng.randint(0, n))
        except Exception:
            idx = int(rng.randint(0, n))
        try:
            X[idx] = hole_vec(X, idx, rng, n_cand=3500)
            # occasionally second kill for large sigma
            if sigma > 0.06 and rng.rand() < 0.5:
                idx2 = int(rng.randint(0, n))
                if idx2 != idx:
                    X[idx2] = hole_vec(X, idx2, rng, n_cand=2500)
            # coherent cap shake: rotate small cluster around random direction
            if rng.rand() < 0.35:
                c0 = X[rng.randint(0, n)]
                dots = X @ c0
                cap = np.where(dots > 0.85)[0]
                if len(cap) >= 2 and len(cap) <= 12:
                    ax = rng.randn(d); ax /= np.linalg.norm(ax)
                    ang = rng.randn() * 0.08
                    # rotate cap points in plane (c0-ish)? simple tangent shift
                    shift = (rng.randn(len(cap), d) - np.sum(rng.randn(len(cap), d) * X[cap], axis=1, keepdims=True) * X[cap] if False else None)
                    # simpler: add small tangent noise
                    nz = rng.randn(len(cap), d) * 0.05
                    proj = np.sum(nz * X[cap], axis=1, keepdims=True)
                    nz -= proj * X[cap]
                    X[cap] = X[cap] + nz
                    X[cap] /= np.linalg.norm(X[cap], axis=1, keepdims=True)
        except Exception:
            pass
        # annealed smooth phase: soft -> hard
        try:
            # vary sharpness per iteration
            if it % 3 == 0:
                X = smooth_adam(X, steps=320, s0=35.0, s1=350.0, lr0=0.02, lr1=0.002, deadline=DEADLINE-8)
            elif it % 3 == 1:
                X = smooth_adam(X, steps=320, s0=60.0, s1=600.0, lr0=0.015, lr1=0.0015, deadline=DEADLINE-8)
            else:
                X = smooth_adam(X, steps=260, s0=25.0, s1=250.0, lr0=0.025, lr1=0.003, deadline=DEADLINE-8)
        except Exception:
            pass
        # short LP screen
        tl = min(1.6, HOP_END - (time.time() - t0 + t0) if False else HOP_END - time.time())
        # compute remaining correctly
        tl = HOP_END - time.time()
        tl = min(1.8, tl)
        if tl > 0.6:
            try:
                X = lp_polish(X, seconds=tl, deadline=DEADLINE-6)
            except Exception:
                pass
        try:
            v = max_cos(X)
        except Exception:
            it += 1
            continue
        if np.all(np.isfinite(X)) and v < best_val:
            best = X
            best_val = v
        it += 1
        if time.time() > HOP_END + 5:
            break
    # final polish
    remain = DEADLINE - time.time() - 1.5
    if remain < 4:
        remain = 4
    remain = min(remain, 34.0)
    try:
        best = lp_polish(best, seconds=remain, deadline=DEADLINE-1.0)
    except Exception:
        pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
