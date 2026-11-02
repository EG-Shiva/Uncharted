import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        c_full = G[iu]
        m = c_full.max()
        sel = c_full > m - max(20.0 * radius, 1e-9)
        if np.count_nonzero(sel) == 0:
            radius *= 0.5
            continue
        I = iu[0][sel]
        J = iu[1][sel]
        p = len(I)
        c_sel = c_full[sel]
        nv = n * d + 1
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], axis=1).ravel()
        rows = np.repeat(np.arange(p), 2 * d + 1)
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
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        if not np.all(np.isfinite(yn)) or np.any(yn < 1e-12):
            radius *= 0.5
            continue
        y /= yn
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _adam_lse(X0, steps, T0, T1, lr, rng, deadline, b1=0.9, b2=0.999, eps=1e-8):
    X = X0.copy()
    n, d = X.shape
    Mt = np.zeros_like(X)
    Vt = np.zeros_like(X)
    for s in range(steps):
        if (s & 31) == 0 and time.time() > deadline:
            break
        frac = s / max(steps - 1, 1)
        T = T0 * (T1 / T0) ** frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        # normalized softmax weights
        W = np.exp((G - m) / T)
        np.fill_diagonal(W, 0.0)
        Z = W.sum() * 0.5
        if Z < 1e-300 or not np.isfinite(Z):
            break
        Grad = (W @ X) / Z
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        # Adam
        Mt = b1 * Mt + (1.0 - b1) * Gt
        Vt = b2 * Vt + (1.0 - b2) * (Gt * Gt)
        bc1 = 1.0 - b1 ** (s + 1)
        bc2 = 1.0 - b2 ** (s + 1)
        Mh = Mt / bc1
        Vh = Vt / bc2
        Upd = Mh / (np.sqrt(Vh) + eps)
        # project update to tangent
        ud = (Upd * X).sum(axis=1, keepdims=True)
        Upd = Upd - ud * X
        # clip
        un = np.linalg.norm(Upd, axis=1, keepdims=True)
        # avoid huge steps
        mx = 3.0
        scale = np.minimum(1.0, mx / (un + 1e-12))
        Upd = Upd * scale
        X = X - lr * Upd
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _largest_void(Xfixed, rng, n_cand=12000):
    # find direction minimizing max dot to Xfixed
    C = rng.normal(size=(n_cand, 4))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    S = C @ Xfixed.T
    mx = S.max(axis=1)
    j = int(np.argmin(mx))
    return C[j]

def _random_rotation4(rng, angle):
    # random small rotation in 4D: exp of random skew
    A = rng.normal(size=(4, 4))
    A = A - A.T
    # scale to get approx angle
    nrm = np.linalg.norm(A, 'fro') + 1e-12
    A = A / nrm * angle * 1.41421356
    # Cayley approx or matrix exp via Taylor? use QR of I+A/2 approx
    # simple: use scipy-free Rodrigues via expm series
    R = np.eye(4) + A + 0.5 * (A @ A)
    # orthonormalize
    Q, _ = np.linalg.qr(R)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q

def solve():
    t0 = time.time()
    BUDGET = 172.0
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            rng0 = np.random.default_rng(0)
            xb = rng0.normal(size=(153, 4))
    xb = np.asarray(xb, dtype=np.float64)
    if xb.shape != (153, 4):
        rng0 = np.random.default_rng(0)
        tmp = rng0.normal(size=(153, 4))
        try:
            flat = xb.ravel()
            tmp.ravel()[:min(flat.size, tmp.size)] = flat[:min(flat.size, tmp.size)]
        except Exception:
            pass
        xb = tmp
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = _max_cos(xb)
    bestX = xb.copy()
    rng = np.random.default_rng(20261102)
    n, d = 153, 4

    trial = 0
    while time.time() - t0 < BUDGET - 10.0:
        trial += 1
        mode = trial % 5
        # analyze best
        Gb = bestX @ bestX.T
        np.fill_diagonal(Gb, -2.0)
        rowmax = Gb.max(axis=1)
        order_crowd = np.argsort(-rowmax)
        order_free = np.argsort(rowmax)
        Xp = bestX.copy()
        if mode == 1:
            sigma = 0.015 + 0.015 * rng.random()
            Xp = Xp + sigma * rng.normal(size=(n, d))
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
            T0, T1, lr, steps = 0.018, 0.0009, 0.015, 2200
        elif mode == 2:
            # patch rotation: pick cap and rotate collectively
            center = rng.normal(size=(4,))
            center /= np.linalg.norm(center)
            cos_thr = 0.88 + 0.09 * rng.random()  # 0.88..0.97
            dots = bestX @ center
            mask = dots > cos_thr
            if mask.sum() < 2 or mask.sum() > 60:
                # fallback to random subset near a random point
                ci = int(rng.integers(0, n))
                dots = bestX @ bestX[ci]
                cos_thr = float(np.sort(dots)[-int(rng.integers(5, 20))])
                mask = dots >= cos_thr
            ang = 0.03 + 0.15 * rng.random()
            R = _random_rotation4(rng, ang)
            Xp[mask] = Xp[mask] @ R.T
            Xp = Xp + 0.008 * rng.normal(size=(n, d))
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
            T0, T1, lr, steps = 0.025, 0.0011, 0.018, 2400
        elif mode == 3:
            # jammed reinsertion to voids
            k = int(rng.integers(3, 9))
            victims = order_crowd[:k].copy()
            # optionally add random jammed
            if rng.random() < 0.5:
                victims[int(rng.integers(0, k))] = order_crowd[int(rng.integers(k, 30))]
            keep = np.ones(n, dtype=bool)
            keep[victims] = False
            Xkeep = Xp[keep]
            for vi in victims:
                v = _largest_void(Xkeep, rng, n_cand=8000)
                # local refine void: hill climb
                for _ in range(25):
                    cand = v + 0.02 * rng.normal(size=(4,))
                    cand /= np.linalg.norm(cand)
                    if (cand @ Xkeep.T).max() < (v @ Xkeep.T).max():
                        v = cand
                Xp[vi] = v
                Xkeep = np.vstack([Xkeep, v[None, :]])
            Xp = Xp + 0.006 * rng.normal(size=(n, d))
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
            T0, T1, lr, steps = 0.03, 0.001, 0.02, 2600
        elif mode == 4:
            sigma = 0.05 + 0.06 * rng.random()
            Xp = Xp + sigma * rng.normal(size=(n, d))
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
            T0, T1, lr, steps = 0.045, 0.0015, 0.022, 2600
        else:
            # rattler + jammed swap / shake free points
            k = int(rng.integers(2, 6))
            free = order_free[:10]
            crowd = order_crowd[:15]
            sel_f = rng.choice(free, size=min(k, len(free)), replace=False)
            sel_c = rng.choice(crowd, size=min(k, len(crowd)), replace=False)
            # move crowded to near free voids with jitter
            for a, b in zip(sel_c, sel_f):
                Xp[a] = Xp[b] + 0.15 * rng.normal(size=(d,))
            Xp = Xp + 0.012 * rng.normal(size=(n, d))
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
            T0, T1, lr, steps = 0.028, 0.0009, 0.018, 2400
        T0 *= 1.0 + 0.25 * (rng.random() - 0.5)
        T1 *= 1.0 + 0.4 * (rng.random() - 0.5)
        T1 = float(np.clip(T1, 0.0006, 0.004))
        T0 = float(np.clip(T0, 0.012, 0.06))
        inner_deadline = t0 + BUDGET - 8.0
        Xd = _adam_lse(Xp, steps, T0, T1, lr, rng, inner_deadline)
        v_desc = _max_cos(Xd)
        # only LP-polish if promising (within tolerance) to save time
        if v_desc < best_val + 0.004:
            lp_deadline = min(time.time() + 0.9, t0 + BUDGET - 7.0)
            try:
                Xp2 = _lp_polish(Xd, lp_deadline)
            except Exception:
                Xp2 = Xd
            v = _max_cos(Xp2)
            if v < best_val - 1e-12 and np.all(np.isfinite(Xp2)):
                best_val = v
                bestX = Xp2.copy()
                # intensify: extra polish immediately
                try:
                    xt_deadline = min(time.time() + 1.2, t0 + BUDGET - 6.5)
                    bestX = _lp_polish(bestX, xt_deadline)
                    best_val = _max_cos(bestX)
                except Exception:
                    pass
        if time.time() - t0 > BUDGET - 10.0:
            break
    try:
        final_deadline = t0 + BUDGET - 0.5
        bestX = _lp_polish(bestX, final_deadline)
    except Exception:
        pass
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX
