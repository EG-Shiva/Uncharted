import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _lp_polish(X, seconds, t_end=1e30):
    X = _norm(X)
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < t_end and radius > 1e-14:
        C = X @ X.T
        c = C[iu0]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I = iu0[0][sel]; J = iu0[1][sel]
        p = len(I)
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
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = _norm(X + res.x[:-1].reshape(n, d))
        v = _max_cos(y)
        if v < best:
            X, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def _refine_vec(v, Xa, alpha=130.0, steps=55, lr=0.2):
    v = v.copy()
    for _ in range(steps):
        s = Xa @ v
        m = float(s.max())
        e = np.exp(alpha * (s - m))
        e /= e.sum()
        g = e @ Xa
        gt = g - float(g @ v) * v
        v = v - lr * gt
        nv = np.linalg.norm(v)
        if nv == 0 or not np.isfinite(nv):
            break
        v /= nv
        lr *= 0.995
    return v

def _block_opt(Xf, Xb0, alpha=130.0, maxiter=150):
    M, d = Xf.shape
    K = Xb0.shape[0]
    Y0 = Xb0.ravel().copy()
    def fun(Yflat):
        Ym = Yflat.reshape(K, d)
        rn = np.linalg.norm(Ym, axis=1, keepdims=True)
        rn = np.maximum(rn, 1e-12)
        Xb = Ym / rn
        D1 = Xb @ Xf.T
        G2 = Xb @ Xb.T
        iu = np.triu_indices(K, 1)
        d2 = G2[iu] if K > 1 else np.empty(0)
        # max
        m1 = float(D1.max()) if D1.size else -2.0
        m2 = float(d2.max()) if d2.size else -2.0
        Mv = m1 if m1 > m2 else m2
        e1 = np.exp(alpha * (D1 - Mv))
        e2 = np.exp(alpha * (d2 - Mv)) if d2.size else np.empty(0)
        S = float(e1.sum() + e2.sum())
        loss = Mv + np.log(max(S, 1e-300)) / alpha
        p1 = e1 / S
        # grad wrt Xb
        grad = p1 @ Xf  # K x d
        if d2.size:
            p2 = e2 / S
            W = np.zeros((K, K))
            W[iu] = p2
            W = W + W.T
            grad = grad + W @ Xb
        dg = (grad * Xb).sum(axis=1, keepdims=True)
        grad_t = grad - dg * Xb
        grad_Y = (grad_t / rn).ravel()
        return float(loss), grad_Y.astype(float)
    try:
        res = minimize(fun, Y0, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-13, "gtol": 1e-7, "maxcor": 8})
        Ym = res.x.reshape(K, d)
    except Exception:
        Ym = Y0.reshape(K, d)
    return _norm(Ym)

def _relax_adam(X0, budget, alpha=160.0, lr=0.004, maxsteps=600):
    X = X0.copy()
    n, d = X.shape
    m = np.zeros_like(X); v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = b1, b2
    st = time.time()
    for k in range(maxsteps):
        if time.time() - st > budget:
            break
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        cm = float(G.max())
        E = np.exp(alpha * (G - cm))
        np.fill_diagonal(E, 0.0)
        s = E.sum() * 0.5
        if s == 0 or not np.isfinite(s):
            break
        W = E / s
        Gt = W @ X
        dot = (Gt * X).sum(axis=1, keepdims=True)
        gt = Gt - dot * X
        m *= b1; m += (1 - b1) * gt
        v *= b2; v += (1 - b2) * (gt * gt)
        mh = m / (1 - b1p); vh = v / (1 - b2p)
        b1p *= b1; b2p *= b2
        X -= lr * mh / (np.sqrt(vh) + eps)
        X = _norm(X)
    return X

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    SEARCH_END = t0 + 150.0
    rng = np.random.default_rng(20261102)
    n, d = 73, 13
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.normal(size=(n, d))
    if xb.shape != (n, d):
        tmp = rng.normal(size=(n, d))
        a = min(xb.shape[0], n); b = min(xb.shape[1], d)
        tmp[:a, :b] = xb[:a, :b]
        xb = tmp
    xb = _norm(xb)
    best_val = _max_cos(xb)
    Xbest = xb.copy()
    sizes = [12, 14, 10, 16, 11, 13, 15, 10, 12]
    modes = ["rattler", "random", "loaded", "pca"]
    it = 0
    NCAND = 80000
    while time.time() < SEARCH_END and time.time() < DEADLINE - 12:
        it += 1
        mode = modes[(it - 1) % len(modes)]
        K = sizes[(it - 1) % len(sizes)]
        # contacts of incumbent
        try:
            G = Xbest @ Xbest.T
            np.fill_diagonal(G, -2.0)
            cm = float(G.max())
            contacts = (G >= cm - 1e-7).sum(axis=1)
        except Exception:
            contacts = np.zeros(n)
        if mode == "rattler":
            idx_c = int(np.argmin(contacts))
            center = Xbest[idx_c]
        elif mode == "random":
            idx_c = int(rng.integers(0, n))
            center = Xbest[idx_c]
        elif mode == "loaded":
            idx_c = int(np.argmax(contacts))
            center = Xbest[idx_c]
        else:  # pca outlier
            try:
                U, S, Vt = np.linalg.svd(Xbest, full_matrices=False)
                P = Xbest @ Vt[:2].T
                cen = P.mean(axis=0, keepdims=True)
                dd = ((P - cen) ** 2).sum(axis=1)
                idx_c = int(np.argmax(dd))
                center = Xbest[idx_c]
            except Exception:
                idx_c = int(rng.integers(0, n))
                center = Xbest[idx_c]
        dots_c = Xbest @ center
        order = np.argsort(-dots_c)
        del_idx = order[:K]
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[del_idx] = False
        Xf = Xbest[keep_mask]
        rebuilt = []
        Xa = Xf
        ok = True
        for kk in range(K):
            if time.time() > DEADLINE - 10:
                ok = False
                break
            if len(rebuilt) > 0:
                Xa = np.vstack([Xf, np.stack(rebuilt)])
            else:
                Xa = Xf
            try:
                C = rng.normal(size=(NCAND, d))
                C /= np.linalg.norm(C, axis=1, keepdims=True)
            except Exception:
                ok = False
                break
            Dm = C @ Xa.T
            md = Dm.max(axis=1)
            bi = np.argsort(md)[:5]
            best_v = None; best_s = 1e30
            for bii in bi:
                v0 = C[int(bii)]
                vr = _refine_vec(v0, Xa)
                s = float(Xa @ vr.reshape(-1, 1).ravel() if False else (Xa @ vr).max())
                if s < best_s:
                    best_s = s; best_v = vr
            if best_v is None:
                best_v = C[int(bi[0])]
            rebuilt.append(best_v)
        if not ok or len(rebuilt) != K:
            continue
        Xb0 = np.stack(rebuilt)
        # block-only L-BFGS
        try:
            if time.time() < DEADLINE - 10:
                Xb1 = _block_opt(Xf, Xb0, alpha=130.0, maxiter=160)
            else:
                Xb1 = Xb0
        except Exception:
            Xb1 = Xb0
        Xcand = np.vstack([Xf, Xb1])
        Xcand = _norm(Xcand)
        # 1-sec smooth relaxation unfreezing everything
        try:
            Xcand = _relax_adam(Xcand, budget=1.0, alpha=170.0, lr=0.004, maxsteps=500)
        except Exception:
            pass
        vnew = _max_cos(Xcand)
        # LP polish only if within 0.001 or better
        if vnew < best_val + 0.001 and time.time() < DEADLINE - 8:
            try:
                Xp = _lp_polish(Xcand, seconds=min(6.0, max(0.5, DEADLINE - 4 - time.time())), t_end=DEADLINE - 2)
                vp = _max_cos(Xp)
                if vp < vnew:
                    Xcand, vnew = Xp, vp
            except Exception:
                pass
        if vnew < best_val - 1e-12 and np.isfinite(vnew):
            Xbest = _norm(Xcand).copy()
            best_val = float(vnew)
    return _norm(Xbest)
