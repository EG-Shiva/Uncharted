import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _lp_polish(X, budget, init_radius=1e-3):
    t0 = time.time()
    x = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = init_radius
    iu = np.triu_indices(n, 1)
    while True:
        if time.time() - t0 + 0.25 >= budget:
            break
        if radius < 1e-14:
            break
        Gfull = x @ x.T
        c_all = Gfull[iu]
        m = float(np.max(c_all))
        thresh = max(20.0 * radius, 1e-9)
        sel = c_all > m - thresh
        cnt = int(np.sum(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 2500:
            kth = np.partition(c_all, -2500)[-2500]
            sel = c_all >= kth
            # trim to 2500 tightest if still many
            idx = np.where(sel)[0]
            if len(idx) > 2700:
                order = np.argsort(-c_all[idx])[:2500]
                mask = np.zeros_like(sel)
                mask[idx[order]] = True
                sel = mask
        I = iu[0][sel]
        J = iu[1][sel]
        csel = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            y = x + res.x[:-1].reshape(n, d)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            v = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _smooth_full(Y, steps, T0, T1, lr0, lr1, mu=0.6):
    n, d = Y.shape
    V = np.zeros_like(Y)
    lT0, lT1 = np.log(T0), np.log(T1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        T = float(np.exp(lT0 + (lT1 - lT0) * frac))
        lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
        G = Y @ Y.T
        np.fill_diagonal(G, -1.0)
        m = float(np.max(G))
        W = (G - m) / T
        np.clip(W, -50.0, 0.0, out=W)
        np.exp(W, out=W)
        np.fill_diagonal(W, 0.0)
        rs = np.sum(W, axis=1)
        WY = W @ Y
        r = np.sum(W * G, axis=1)
        inv = np.zeros_like(rs)
        mk = rs > 1e-12
        inv[mk] = 1.0 / rs[mk]
        avgWY = WY * inv[:, None]
        avgR = r * inv
        Dir = avgWY - avgR[:, None] * Y
        Dir[~mk] = 0.0
        V *= mu
        V += Dir
        Y = Y - lr * V
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return Y

def _build_perms(kind="D9"):
    if kind == "D9":
        perms = []
        for k in range(9):
            perms.append(np.array([(j + k) % 9 for j in range(9)], dtype=int))
        for k in range(9):
            perms.append(np.array([(k - j) % 9 for j in range(9)], dtype=int))
        perms = np.array(perms, dtype=int)  # 18x9
    else:  # C9
        perms = []
        for k in range(9):
            perms.append(np.array([(j + k) % 9 for j in range(9)], dtype=int))
        perms = np.array(perms, dtype=int)  # 9x9
    inv = np.empty_like(perms)
    for g in range(perms.shape[0]):
        iv = np.empty(9, dtype=int)
        iv[perms[g]] = np.arange(9)
        inv[g] = iv
    return perms, inv

def _expand(Theta, perms, fixed):
    R = Theta.shape[0]
    Gsz = perms.shape[0]
    n = R * Gsz + 1
    d = Theta.shape[1]
    X = np.empty((n, d))
    idx = 0
    for r in range(R):
        t = Theta[r]
        for g in range(Gsz):
            X[idx] = t[perms[g]]
            idx += 1
    X[idx] = fixed
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _optimize_symmetric(R, kind, steps, seed, t_limit, t_start):
    perms, inv = _build_perms(kind)
    Gsz = perms.shape[0]
    rng = np.random.default_rng(seed)
    Theta = rng.standard_normal((R, 9))
    Theta /= np.linalg.norm(Theta, axis=1, keepdims=True)
    fixed = np.ones(9) / 3.0
    # Adam state
    Mt = np.zeros_like(Theta)
    Vt = np.zeros_like(Theta)
    b1, b2, eps = 0.9, 0.999, 1e-8
    T0, T1 = 0.04, 0.004
    lr0, lr1 = 0.05, 0.008
    lT0, lT1 = np.log(T0), np.log(T1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    best_val = np.inf
    best_Theta = Theta.copy()
    # map idx -> (r,g)
    for s in range(steps):
        if time.time() - t_start > t_limit:
            break
        frac = s / max(steps - 1, 1)
        T = float(np.exp(lT0 + (lT1 - lT0) * frac))
        lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
        X = _expand(Theta, perms, fixed)
        G = X @ X.T
        np.fill_diagonal(G, -1.0)
        m = float(np.max(G))
        if m < best_val:
            best_val = m
            best_Theta = Theta.copy()
        W = (G - m) / T
        np.clip(W, -50.0, 0.0, out=W)
        np.exp(W, out=W)
        np.fill_diagonal(W, 0.0)
        sW = float(np.sum(W)) / 2.0 + 1e-30
        GradX = (W @ X) / sW
        # tangential
        GradX -= np.sum(GradX * X, axis=1, keepdims=True) * X
        # pullback to Theta
        GradT = np.zeros_like(Theta)
        idx = 0
        for r in range(R):
            acc = np.zeros(9)
            for g in range(Gsz):
                gx = GradX[idx]
                # dT = gx[inv[g]]
                acc += gx[inv[g]]
                idx += 1
            GradT[r] = acc
        # idx for fixed point: ignore (fixed not optimized), but its GradX contributes? no
        # Riemannian projection for Theta
        GradT -= np.sum(GradT * Theta, axis=1, keepdims=True) * Theta
        # Adam
        Mt = b1 * Mt + (1 - b1) * GradT
        Vt = b2 * Vt + (1 - b2) * GradT * GradT
        mhat = Mt / (1 - b1 ** (s + 1))
        vhat = Vt / (1 - b2 ** (s + 1))
        step = lr * mhat / (np.sqrt(vhat) + eps)
        # project step tangentially to avoid radial drift
        step -= np.sum(step * Theta, axis=1, keepdims=True) * Theta
        # limit step norm
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        cap = 0.1
        scale = np.minimum(1.0, cap / (sn + 1e-12))
        step = step * scale
        Theta = Theta - step
        Theta /= np.linalg.norm(Theta, axis=1, keepdims=True)
    Xb = _expand(best_Theta, perms, fixed)
    return Xb, best_val, perms

def solve():
    t_start = time.time()
    BUDGET = 168.0
    rng = np.random.default_rng(7)
    n, d = 181, 9
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
    if not cands:
        a = rng.standard_normal((n, d))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    vals = [_max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    x_best = cands[bi].copy()
    b_best = float(vals[bi])

    # --- Phase A: symmetry-constrained orbit search ---
    # D9: 10*18+1=181, C9: 20*9+1=181
    sym_cands = []
    try:
        Xs, vs, _ = _optimize_symmetric(10, "D9", steps=900, seed=123, t_limit=22.0, t_start=t_start)
        sym_cands.append((vs, Xs))
    except Exception:
        pass
    try:
        Xs2, vs2, _ = _optimize_symmetric(20, "C9", steps=600, seed=456, t_limit=38.0, t_start=t_start)
        sym_cands.append((vs2, Xs2))
    except Exception:
        pass
    x_sym = None
    if sym_cands:
        sym_cands.sort(key=lambda t: t[0])
        x_sym = sym_cands[0][1].copy()
        # break symmetry: smooth full-dimensional anneal
        try:
            rem = BUDGET - (time.time() - t_start)
            if rem > 30:
                y = _smooth_full(x_sym.copy(), steps=350, T0=0.02, T1=0.002, lr0=0.06, lr1=0.015)
                rem2 = BUDGET - (time.time() - t_start)
                if rem2 > 12:
                    y = _lp_polish(y, budget=min(9.0, rem2 - 4.0), init_radius=2e-3)
                v = _max_cos(y)
                if v < b_best:
                    x_best, b_best = y, v
                # keep smoothed sym as seed for perturbations even if not better
                x_sym = y.copy()
        except Exception:
            pass

    # --- Phase B: perturbations of symmetric seed 1e-4 .. 1e-2 + LP ---
    if x_sym is not None:
        sigmas = [1e-4, 5e-4, 1e-3, 3e-3, 6e-3, 1e-2]
        for si, sig in enumerate(sigmas):
            rem = BUDGET - (time.time() - t_start)
            if rem < 8.0:
                break
            try:
                y = x_sym + rng.standard_normal((n, d)) * sig
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                # short smooth
                y = _smooth_full(y, steps=120, T0=0.01, T1=0.002, lr0=0.04, lr1=0.01)
                rem2 = BUDGET - (time.time() - t_start)
                if rem2 < 4.0:
                    v = _max_cos(y)
                    if v < b_best:
                        x_best, b_best = y, v
                    break
                y = _lp_polish(y, budget=min(4.0, rem2 - 2.0), init_radius=3e-3)
                v = _max_cos(y)
                if v < b_best:
                    x_best, b_best = y, v
                    x_sym = y.copy()
            except Exception:
                continue

    # --- Phase C: ensure baseline polished + small hops around best ---
    while True:
        rem = BUDGET - (time.time() - t_start)
        if rem < 6.0:
            break
        try:
            # alternate: polish current best briefly, then try a hop
            y = _lp_polish(x_best, budget=min(5.0, rem - 1.5), init_radius=1e-3)
            v = _max_cos(y)
            if v < b_best:
                x_best, b_best = y, v
                continue
            # hop with moderate sigma
            sig = 0.004 if b_best > 0.462 else 0.002
            yh = x_best + rng.standard_normal((n, d)) * sig
            yh /= np.linalg.norm(yh, axis=1, keepdims=True)
            yh = _smooth_full(yh, steps=100, T0=0.008, T1=0.002, lr0=0.03, lr1=0.008)
            rem2 = BUDGET - (time.time() - t_start)
            if rem2 < 3.0:
                v2 = _max_cos(yh)
                if v2 < b_best:
                    x_best, b_best = yh, v2
                break
            yh = _lp_polish(yh, budget=min(3.0, rem2 - 1.0), init_radius=2e-3)
            v2 = _max_cos(yh)
            if v2 < b_best:
                x_best, b_best = yh, v2
            else:
                break
        except Exception:
            break

    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
