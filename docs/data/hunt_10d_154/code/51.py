import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.sqrt(np.sum(X * X, axis=1, keepdims=True))
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _lp_polish(X, seconds, deadline=1e18):
    X = _norm(X)
    n, d = X.shape
    best = _max_cos(X)
    Xb = X.copy()
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < deadline:
        if radius < 1e-14:
            break
        G = X @ X.T
        iu = iu0
        c = G[iu]
        m = float(np.max(c))
        sel = c > m - max(20.0 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        p = len(I)
        if p == 0:
            break
        # cap constraints for speed
        if p > 4000:
            # keep most tight
            order = np.argsort(-c[sel])[:4000]
            I = I[order]; J = J[order]
            csv = c[sel][order]
            p = 4000
        else:
            csv = c[sel]
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csv, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            Xb = y.copy()
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return _norm(Xb) if _max_cos(Xb) <= _max_cos(X) else _norm(X)

def _adam(X, steps, a0, a1, lr0):
    n, d = X.shape
    m = np.zeros_like(X); v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    la0, la1 = np.log(a0), np.log(a1)
    for t in range(1, steps + 1):
        frac = t / max(steps, 1)
        alpha = np.exp(la0 + (la1 - la0) * frac)
        lr = lr0 * (1.0 - 0.7 * frac)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = float(np.max(G))
        E = np.exp(alpha * (G - M))
        np.fill_diagonal(E, 0.0)
        S = float(np.sum(E))
        if not np.isfinite(S) or S == 0:
            break
        Grad = (2.0 / S) * (E @ X)
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        m *= b1; m += (1.0 - b1) * Gt
        v *= b2; v += (1.0 - b2) * (Gt * Gt)
        c1 = 1.0 - b1 ** t; c2 = 1.0 - b2 ** t
        step = (m / c1) / (np.sqrt(v / c2) + eps)
        X = X - lr * step
        X = _norm(X)
    return X

def _tangent_bases(X):
    n, d = X.shape
    kdim = d - 1
    Ts = np.zeros((n, d, kdim))
    for i in range(n):
        xi = X[i]
        k = int(np.argmax(np.abs(xi)))
        cols = []
        for j in range(d):
            if j == k:
                continue
            v = np.zeros(d); v[j] = 1.0
            v = v - float(xi[j]) * xi
            cols.append(v)
        M = np.stack(cols, axis=1)  # d x 9
        try:
            Q, _ = np.linalg.qr(M)
            Ts[i] = Q
        except Exception:
            # fallback: random
            R = np.random.randn(d, kdim)
            R = R - np.outer(xi, (xi @ R))
            Q, _ = np.linalg.qr(R)
            Ts[i] = Q
    return Ts

def _soft_modes(X, thresh=1e-5, nmodes=30):
    n, d = X.shape
    kdim = d - 1
    N = n * kdim
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    mcur = float(np.max(G))
    iu = np.triu_indices(n, 1)
    c = G[iu]
    sel = c > mcur - thresh
    I = iu[0][sel]; J = iu[1][sel]
    p = len(I)
    if p < 10:
        return None, None, mcur, p
    # cap p for eig build
    if p > 5000:
        order = np.argsort(-c[sel])[:5000]
        I = I[order]; J = J[order]
        p = 5000
    try:
        Ts = _tangent_bases(X)
    except Exception:
        return None, None, mcur, p
    # Build R (p x N) then Gram Gmat = R.T @ R via accumulation to save memory
    try:
        R = np.zeros((p, N), dtype=float)
        # vectorize blocks: for each contact fill
        # loop in python (p up to 5000) is okay
        for l in range(p):
            i = int(I[l]); j = int(J[l])
            R[l, i * kdim:(i + 1) * kdim] = Ts[i].T @ X[j]
            R[l, j * kdim:(j + 1) * kdim] = Ts[j].T @ X[i]
        Gmat = R.T @ R
        # symmetrize
        Gmat = 0.5 * (Gmat + Gmat.T)
        w, V = np.linalg.eigh(Gmat)
        # w ascending, V columns
        K = min(nmodes, N)
        modes = V[:, :K].copy()  # N x K softest
        return modes, Ts, mcur, p
    except Exception:
        return None, None, mcur, p

def solve():
    t0 = time.time()
    DEAD = t0 + 172.0
    HOP_END = t0 + 128.0
    rng = np.random.default_rng(0)
    n, d = 154, 10
    kdim = d - 1
    best = None; best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = _norm(a)
                v = _max_cos(a)
                if v < best_val:
                    best_val = v
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.standard_normal((n, d)))
        best_val = _max_cos(best)
    Xbest = _norm(best.copy())

    # initial soft modes
    thresh = 1e-5
    modes, Ts, _, _ = _soft_modes(Xbest, thresh=thresh, nmodes=30)
    last_recompute = 0
    trial = 0
    # anneal params cycle
    while time.time() < HOP_END and time.time() < DEAD - 30:
        trial += 1
        # recompute modes periodically or after improvement
        if modes is None or (trial - last_recompute) >= 10:
            # alternate thresholds for diversity
            th = 1e-5 if (trial // 10) % 2 == 0 else 1e-6
            try:
                if time.time() < DEAD - 30:
                    modes, Ts, _, _ = _soft_modes(Xbest, thresh=th, nmodes=30)
            except Exception:
                modes = None
            last_recompute = trial
            # refresh Ts if modes None
            if modes is None:
                try:
                    Ts = _tangent_bases(Xbest)
                except Exception:
                    Ts = None
        # build trial displacement
        Xtry = None
        try:
            if modes is not None and Ts is not None and rng.random() < 0.85:
                K = int(rng.integers(10, 31))
                K = min(K, modes.shape[1])
                cols = rng.choice(modes.shape[1], size=K, replace=False)
                coeff = rng.standard_normal(K)
                u = modes[:, cols] @ coeff  # N,
                U = u.reshape(n, kdim)
                # scale to target radians
                fro = float(np.linalg.norm(U)) + 1e-12
                s = float(np.exp(rng.uniform(np.log(0.03), np.log(0.25))))
                U = U / fro * s * np.sqrt(float(n))
                # sign choice to reduce current top pairs
                Gc = Xbest @ Xbest.T
                np.fill_diagonal(Gc, -2.0)
                # find top ~40 pairs
                iu = np.triu_indices(n, 1)
                c = Gc[iu]
                topk = min(40, len(c))
                idx = np.argpartition(-c, topk - 1)[:topk]
                It = iu[0][idx]; Jt = iu[1][idx]
                # predicted delta for +U
                # dX_i = Ts[i] @ U[i]
                dX = np.zeros((n, d))
                for i in range(n):
                    dX[i] = Ts[i] @ U[i]
                ssum = 0.0
                for ii, jj in zip(It, Jt):
                    ssum += float(Xbest[ii] @ dX[jj] + Xbest[jj] @ dX[ii])
                if ssum > 0:
                    U = -U
                    dX = -dX
                # occasionally flip randomly for diversity
                if rng.random() < 0.15:
                    dX = -dX
                Xtry = Xbest + dX
                # small extra isotropic noise to break symmetry
                Xtry = Xtry + 0.005 * rng.standard_normal((n, d))
                Xtry = _norm(Xtry)
            else:
                # fallback random / hole kick
                Xtry = Xbest + 0.08 * rng.standard_normal((n, d))
                Xtry = _norm(Xtry)
        except Exception:
            Xtry = _norm(Xbest + 0.08 * rng.standard_normal((n, d)))
        # annealed softmax Adam short
        try:
            steps = int(rng.integers(300, 601))
            a0 = float(rng.uniform(20.0, 35.0))
            a1 = float(rng.uniform(400.0, 650.0))
            lr0 = float(rng.uniform(0.010, 0.018))
            Xref = _adam(Xtry, steps, a0, a1, lr0)
        except Exception:
            continue
        try:
            val = _max_cos(Xref)
        except Exception:
            continue
        # keep only if promising then short LP snap
        if val < best_val + 0.004:
            # short active-set LP with explicit deadline
            sec = 1.2
            if time.time() + sec > DEAD - 12:
                sec = max(0.4, (DEAD - 12) - time.time())
            try:
                Xpol = _lp_polish(Xref, sec, deadline=DEAD - 8)
                vpol = _max_cos(Xpol)
            except Exception:
                vpol = val
                Xpol = Xref
            if vpol < best_val - 1e-12:
                Xbest = _norm(Xpol.copy())
                best_val = float(vpol)
                last_recompute = trial + 100  # force recompute next iter
                # immediate recompute attempt
                try:
                    modes, Ts, _, _ = _soft_modes(Xbest, thresh=thresh, nmodes=30)
                    last_recompute = trial
                except Exception:
                    pass
            elif val < best_val - 1e-12:
                Xbest = _norm(Xref.copy())
                best_val = float(val)
        if time.time() > HOP_END:
            break

    # final polish with remaining budget
    remaining = (t0 + 172.0) - time.time() - 1.0
    if remaining > 3:
        try:
            Xbest = _lp_polish(Xbest, min(remaining, 12.0), deadline=t0 + 173.0)
            best_val = _max_cos(Xbest)
        except Exception:
            pass
    # micro shake squeeze if time left
    t_left = (t0 + 174.0) - time.time()
    if t_left > 5:
        for _ in range(2):
            if time.time() > t0 + 173.0:
                break
            try:
                Xt = _norm(Xbest + 2e-4 * rng.standard_normal((n, d)))
                Xt = _lp_polish(Xt, 0.9, deadline=t0 + 174.0)
                vt = _max_cos(Xt)
                if vt < best_val:
                    Xbest, best_val = Xt, vt
            except Exception:
                break
    Xbest = _norm(Xbest)
    return Xbest
