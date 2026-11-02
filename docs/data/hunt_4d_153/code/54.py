import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def random_sphere(n, d, rng):
    z = rng.standard_normal((n, d))
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    return z

def contact_counts(x, tol=1e-9):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    B = G > (m - tol)
    return B.sum(axis=1), m, G

def smooth_adam(X, steps, a0, a1, lr0, lr1, t_deadline=1e30):
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    lrat = lr1 / lr0 if lr0 != 0 else 1.0
    arat = a1 / a0 if a0 != 0 else 1.0
    for t in range(1, steps + 1):
        if t % 300 == 0 and time.time() > t_deadline:
            break
        frac = t / steps
        alpha = a0 * (arat ** frac)
        lr = lr0 * (lrat ** frac)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m0 = np.max(G)
        W = np.exp(alpha * (G - m0))
        np.fill_diagonal(W, 0.0)
        grad_raw = W @ X
        dot = np.sum(grad_raw * X, axis=1, keepdims=True)
        grad_t = grad_raw - dot * X
        m *= b1; m += (1.0 - b1) * grad_t
        v *= b2; v += (1.0 - b2) * (grad_t * grad_t)
        mhat = m / (1.0 - b1 ** t)
        vhat = v / (1.0 - b2 ** t)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        sdot = np.sum(step * X, axis=1, keepdims=True)
        step -= sdot * X
        nrm = np.linalg.norm(step, axis=1, keepdims=True)
        scale = np.where(nrm > 0.02, 0.02 / (nrm + 1e-12), 1.0)
        step *= scale
        X = X - step
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def riesz_adam(X, steps, s0, s1, lr0, lr1, t_deadline=1e30):
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    lrat = lr1 / lr0 if lr0 != 0 else 1.0
    for t in range(1, steps + 1):
        if t % 300 == 0 and time.time() > t_deadline:
            break
        frac = t / steps
        s = s0 + (s1 - s0) * frac
        lr = lr0 * ((lrat) ** frac)
        G = X @ X.T
        np.fill_diagonal(G, -1.0)
        R2 = np.clip(2.0 - 2.0 * G, 1e-12, 4.0)
        np.fill_diagonal(R2, np.inf)
        # weight = R2^{-s/2 -1}
        W = np.exp((-s / 2.0 - 1.0) * np.log(R2))
        np.fill_diagonal(W, 0.0)
        # normalize to keep step stable
        mx = np.max(W)
        if mx > 0 and np.isfinite(mx):
            W = W / mx
        grad_raw = W @ X
        dot = np.sum(grad_raw * X, axis=1, keepdims=True)
        grad_t = grad_raw - dot * X
        # clip extreme
        gn = np.linalg.norm(grad_t, axis=1, keepdims=True)
        grad_t *= np.where(gn > 10, 10 / (gn + 1e-12), 1.0)
        m *= b1; m += (1.0 - b1) * grad_t
        v *= b2; v += (1.0 - b2) * (grad_t * grad_t)
        mhat = m / (1.0 - b1 ** t)
        vhat = v / (1.0 - b2 ** t)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        sdot = np.sum(step * X, axis=1, keepdims=True)
        step -= sdot * X
        nrm = np.linalg.norm(step, axis=1, keepdims=True)
        scale = np.where(nrm > 0.02, 0.02 / (nrm + 1e-12), 1.0)
        step *= scale
        X = X - step
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    best_x = x.copy()
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    I0 = iu[0]; J0 = iu[1]
    while time.time() - start < seconds and radius > 1e-14:
        cfull = (x @ x.T)[iu]
        m = float(np.max(cfull))
        thr = m - max(20.0 * radius, 1e-9)
        sel = cfull > thr
        cnt = int(np.sum(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 2500:
            idx = np.where(sel)[0]
            order = np.argsort(cfull[idx])[-2500:]
            keep = np.zeros_like(sel)
            keep[idx[order]] = True
            sel = keep
            cnt = 2500
        I = I0[sel]; J = J0[sel]
        csel = cfull[sel]
        p = cnt
        nv = n * d + 1
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], axis=1).ravel()
        rows = np.repeat(np.arange(p), 2 * d + 1)
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        if np.any(~np.isfinite(yn)) or np.any(yn < 1e-12):
            radius *= 0.5
            continue
        y /= yn
        vv = max_cos(y)
        if vv < best:
            best = vv
            best_x = y.copy()
            x = y
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
        if time.time() - start >= seconds:
            break
    return best_x

def random_rotation_near_id(d, rng, eps):
    A = rng.standard_normal((d, d))
    A = A - A.T
    A *= eps
    # Cayley-ish: Q = (I + A/2)(I - A/2)^{-1} approx; use QR for safety
    M = np.eye(d) + A
    Q, R = np.linalg.qr(M)
    # fix signs to be near identity
    s = np.sign(np.diag(R))
    s[s == 0] = 1.0
    Q = Q * s[None, :]
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q

def patch_rotate(X, rng):
    n, d = X.shape
    u = rng.standard_normal(d); u /= np.linalg.norm(u)
    # angular radius 15-45 deg -> cos threshold
    ang = rng.uniform(15, 45) * np.pi / 180.0
    thr = np.cos(ang)
    dots = X @ u
    sel = np.where(dots > thr)[0]
    # need nontrivial patch
    if len(sel) < 3 or len(sel) > n - 3:
        # try complement or random subset via second center
        if len(sel) > n - 3:
            return None
        # if too small, lower threshold
        thr2 = np.quantile(dots, 0.85)
        sel = np.where(dots > thr2)[0]
        if len(sel) < 3 or len(sel) > n - 3:
            return None
    eps = float(rng.uniform(0.05, 0.35))
    Q = random_rotation_near_id(d, rng, eps)
    Y = X.copy()
    Y[sel] = (Y[sel] @ Q.T)
    Y[sel] /= np.linalg.norm(Y[sel], axis=1, keepdims=True)
    return Y

def largest_holes(X, k, rng, npool=20000):
    d = X.shape[1]
    P = rng.standard_normal((npool, d))
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    G = P @ X.T
    closest = np.max(G, axis=1)
    idx = np.argsort(closest)[:k*4]
    picks = []
    chosen = []
    for ii in idx:
        v = P[ii]
        ok = True
        for c in chosen:
            if float(v @ c) > 0.9:
                ok = False
                break
        if ok:
            picks.append(v)
            chosen.append(v)
        if len(picks) >= k:
            break
    while len(picks) < k:
        v = rng.standard_normal(d); v /= np.linalg.norm(v)
        picks.append(v)
    return np.array(picks)

def solve():
    t0 = time.time()
    DEADLINE = 178.0
    rng = np.random.default_rng(7)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    if xb is not None and xb.shape == (153, 4):
        cands.append(xb / np.linalg.norm(xb, axis=1, keepdims=True))
    if xr is not None and xr.shape == (153, 4):
        cands.append(xr / np.linalg.norm(xr, axis=1, keepdims=True))
    if len(cands) == 0:
        cands.append(random_sphere(153, 4, rng))
    bestX = None; bestV = 1e30
    for c in cands:
        v = max_cos(c)
        if v < bestV:
            bestV = v; bestX = c.copy()
    n, d = bestX.shape
    cc, _, _ = contact_counts(bestX)
    order_hi = np.argsort(-cc)
    EXPLORE_END = t0 + 148.0
    trial = 0
    while time.time() < EXPLORE_END and (time.time() - t0) < DEADLINE - 28:
        trial += 1
        mode = trial % 7
        Xtry = None
        dl = EXPLORE_END
        if mode == 0:
            sigma = 0.01
            Xs = bestX + sigma * rng.standard_normal((n, d))
            Xs /= np.linalg.norm(Xs, axis=1, keepdims=True)
            Xtry = smooth_adam(Xs, steps=1500, a0=50, a1=320, lr0=0.004, lr1=0.0004, t_deadline=dl)
        elif mode == 1:
            # Riesz broad then softmax sharpen: encourages uniform second shell
            Xs = bestX + 0.05 * rng.standard_normal((n, d))
            Xs /= np.linalg.norm(Xs, axis=1, keepdims=True)
            Xs = riesz_adam(Xs, steps=900, s0=2.0, s1=8.0, lr0=0.005, lr1=0.002, t_deadline=dl)
            Xtry = smooth_adam(Xs, steps=1200, a0=60, a1=350, lr0=0.004, lr1=0.0004, t_deadline=dl)
        elif mode == 2:
            # cooperative patch rotation
            Y = None
            for _ in range(5):
                Y = patch_rotate(bestX, rng)
                if Y is not None:
                    break
            if Y is None:
                Y = bestX + 0.03 * rng.standard_normal((n, d))
                Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            else:
                Y = Y + 0.008 * rng.standard_normal((n, d))
                Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            Xtry = smooth_adam(Y, steps=1800, a0=45, a1=300, lr0=0.005, lr1=0.0004, t_deadline=dl)
        elif mode == 3:
            # kill overloaded + refill largest holes (load balancing)
            k = int(rng.integers(3, 9))
            idx = set()
            top = order_hi[:35]
            perm = rng.permutation(top)[:min(k, len(top))]
            for a in perm:
                idx.add(int(a))
            # also kill rattlers (0 contacts) to relocate them usefully
            low = np.where(cc <= 1)[0]
            if len(low) > 0:
                idx.add(int(rng.choice(low)))
            while len(idx) < k + 1:
                idx.add(int(rng.integers(0, n)))
            idx = np.array(sorted(idx), dtype=int)
            holes = largest_holes(bestX, len(idx), rng)
            Xs = bestX.copy()
            Xs[idx] = holes[:len(idx)]
            Xs += 0.01 * rng.standard_normal((n, d))
            Xs /= np.linalg.norm(Xs, axis=1, keepdims=True)
            Xs = riesz_adam(Xs, steps=600, s0=3.0, s1=6.0, lr0=0.005, lr1=0.002, t_deadline=dl)
            Xtry = smooth_adam(Xs, steps=1400, a0=50, a1=330, lr0=0.004, lr1=0.0004, t_deadline=dl)
        elif mode == 4:
            # large kick to jump basin
            sigma = float(rng.uniform(0.15, 0.35))
            Xs = bestX + sigma * rng.standard_normal((n, d))
            Xs /= np.linalg.norm(Xs, axis=1, keepdims=True)
            Xs = riesz_adam(Xs, steps=800, s0=1.5, s1=7.0, lr0=0.006, lr1=0.002, t_deadline=dl)
            Xtry = smooth_adam(Xs, steps=1500, a0=35, a1=300, lr0=0.005, lr1=0.0005, t_deadline=dl)
        elif mode == 5:
            # fresh random with Riesz -> softmax (new topology)
            Xs = random_sphere(n, d, rng)
            Xs = riesz_adam(Xs, steps=1200, s0=2.0, s1=9.0, lr0=0.006, lr1=0.002, t_deadline=dl)
            Xtry = smooth_adam(Xs, steps=1500, a0=40, a1=320, lr0=0.005, lr1=0.0004, t_deadline=dl)
        else:
            # double-patch / splice: two patches rotated differently
            Y = bestX.copy()
            for _ in range(2):
                Pp = patch_rotate(Y, rng)
                if Pp is not None:
                    # blend: take moved points from Pp
                    diff = np.linalg.norm(Pp - Y, axis=1)
                    sel = diff > 1e-12
                    Y[sel] = Pp[sel]
            Y += 0.008 * rng.standard_normal((n, d))
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            Xtry = smooth_adam(Y, steps=1600, a0=55, a1=340, lr0=0.004, lr1=0.0004, t_deadline=dl)
        if Xtry is None:
            continue
        v = max_cos(Xtry)
        need_lp = False
        lp_time = 1.0
        if v < bestV - 1e-12:
            need_lp = True; lp_time = 2.5
        elif v < bestV + 0.003:
            if mode in (2, 3, 4) or (trial % 2 == 0):
                need_lp = True; lp_time = 1.2
        if need_lp:
            remain = EXPLORE_END - time.time()
            if remain < 0.6:
                if v < bestV:
                    bestV = v; bestX = Xtry.copy()
                    cc, _, _ = contact_counts(bestX)
                    order_hi = np.argsort(-cc)
                break
            lp_time = min(lp_time, remain - 0.1)
            if lp_time < 0.3:
                if v < bestV:
                    bestV = v; bestX = Xtry.copy()
                    cc, _, _ = contact_counts(bestX)
                    order_hi = np.argsort(-cc)
                continue
            Xp = lp_polish(Xtry, lp_time)
            vp = max_cos(Xp)
            if vp < bestV:
                bestV = vp; bestX = Xp.copy()
                cc, _, _ = contact_counts(bestX)
                order_hi = np.argsort(-cc)
            elif v < bestV:
                bestV = v; bestX = Xtry.copy()
                cc, _, _ = contact_counts(bestX)
                order_hi = np.argsort(-cc)
        else:
            if v < bestV:
                bestV = v; bestX = Xtry.copy()
                cc, _, _ = contact_counts(bestX)
                order_hi = np.argsort(-cc)
    remain = DEADLINE - (time.time() - t0) - 1.0
    if remain > 1.0:
        bestX = lp_polish(bestX, min(remain, 25.0))
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX
