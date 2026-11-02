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


def make_hopf(n_fibers, pts_per_fiber, rng, left=False):
    # parallel Hopf fibers: representatives p_i uniform on S^3,
    # fiber = p_i * (cos t, sin t, 0, 0) (right) or (cos t,sin t,0,0)*p_i (left chirality)
    p = random_sphere(n_fibers, 4, rng)
    offs = rng.uniform(0.0, 2.0 * np.pi, size=n_fibers)
    pts = np.empty((n_fibers * pts_per_fiber, 4))
    ks = np.arange(pts_per_fiber)
    r = 0
    for i in range(n_fibers):
        w0, x0, y0, z0 = p[i]
        tj = 2.0 * np.pi * ks / pts_per_fiber + offs[i]
        c = np.cos(tj)
        s = np.sin(tj)
        if not left:
            # p * q
            w = w0 * c - x0 * s
            x = w0 * s + x0 * c
            y = y0 * c + z0 * s
            z = -y0 * s + z0 * c
        else:
            # q * p
            w = c * w0 - s * x0
            x = c * x0 + s * w0
            y = c * y0 - s * z0
            z = c * z0 + s * y0
        pts[r:r + pts_per_fiber, 0] = w
        pts[r:r + pts_per_fiber, 1] = x
        pts[r:r + pts_per_fiber, 2] = y
        pts[r:r + pts_per_fiber, 3] = z
        r += pts_per_fiber
    # normalize (exact already)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def smooth_adam(X, steps, a0, a1, lr0, lr1, t_deadline=1e30):
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    lrat = lr1 / lr0 if lr0 != 0 else 1.0
    arat = a1 / a0 if a0 != 0 else 1.0
    for t in range(1, steps + 1):
        if (t & 255) == 0 and time.time() > t_deadline:
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
        m *= b1
        m += (1.0 - b1) * grad_t
        v *= b2
        v += (1.0 - b2) * (grad_t * grad_t)
        b1t = 1.0 - b1 ** t
        b2t = 1.0 - b2 ** t
        mhat = m / b1t
        vhat = v / b2t
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
    iu0 = np.triu_indices(n, 1)
    I0 = iu0[0]
    J0 = iu0[1]
    while time.time() - start < seconds and radius > 1e-14:
        cfull = (x @ x.T)[iu0]
        m = float(np.max(cfull))
        thr = m - max(20.0 * radius, 1e-9)
        sel = cfull > thr
        cnt = int(np.sum(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 3000:
            idx = np.where(sel)[0]
            order = np.argsort(cfull[idx])[-3000:]
            keep = np.zeros_like(sel)
            keep[idx[order]] = True
            sel = keep
            cnt = 3000
        I = I0[sel]
        J = J0[sel]
        csel = cfull[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
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


def contact_counts(x, tol=1e-9):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    B = G > (m - tol)
    return B.sum(axis=1), m


def largest_void_dir(x, rng, s=12000):
    R = random_sphere(s, 4, rng)
    D = R @ x.T
    mx = np.max(D, axis=1)
    return R[np.argmin(mx)]


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
    bestX = None
    bestV = 1e30
    for c in cands:
        v = max_cos(c)
        if v < bestV:
            bestV = v
            bestX = c.copy()
    n, d = 153, 4
    EXPLORE_END = t0 + 150.0

    def try_update(Xtry, do_lp_time):
        nonlocal bestX, bestV
        v = max_cos(Xtry)
        if v < bestV + 0.004 and do_lp_time is not None and do_lp_time > 0.35:
            remain = EXPLORE_END - time.time()
            if remain > 0.5:
                lt = min(do_lp_time, remain - 0.1)
                if lt >= 0.35:
                    Xp = lp_polish(Xtry, lt)
                    vp = max_cos(Xp)
                    if vp < bestV:
                        bestV = vp
                        bestX = Xp.copy()
                        return True
                    if v < bestV:
                        bestV = v
                        bestX = Xtry.copy()
                        return True
                    return False
        if v < bestV:
            bestV = v
            bestX = Xtry.copy()
            return True
        return False

    # Phase A: Hopf balanced basins (17 fibers x 9 pts). k=17 intra cos=cos(40deg)=0.766 feasible.
    hopf_tries = 7
    for h in range(hopf_tries):
        if time.time() > EXPLORE_END - 2.0 or time.time() - t0 > DEADLINE - 30:
            break
        left = (h % 2 == 1)
        X0 = make_hopf(17, 9, rng, left=left)
        # tiny symmetry-breaking shake so Adam can diverge
        X0 = X0 + 0.002 * rng.standard_normal((n, d))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        X1 = smooth_adam(X0, steps=2500, a0=28, a1=300, lr0=0.006, lr1=0.0005, t_deadline=EXPLORE_END)
        # second anneal at higher sharpness
        X1 = smooth_adam(X1, steps=1200, a0=120, a1=420, lr0=0.002, lr1=0.0003, t_deadline=EXPLORE_END)
        try_update(X1, 1.4)
        # also a jittered-hop from hopf without LP if far (to save time, LP only if promising)
    # also try mixed fiber sizes (e.g. 16x9 + 1x9 jitter) implicitly covered; try 19-fiber-like random cap init
    # Phase B: surgery + basin hopping around incumbent
    cc, _ = contact_counts(bestX)
    order_hi = np.argsort(-cc)
    order_lo = np.argsort(cc)
    trial = 0
    while time.time() < EXPLORE_END and time.time() - t0 < DEADLINE - 25:
        trial += 1
        # refresh contact ordering every 5 trials from best
        if trial % 5 == 1:
            cc, _ = contact_counts(bestX)
            order_hi = np.argsort(-cc)
            order_lo = np.argsort(cc)
        rattler = int(order_lo[0])
        mode = trial % 6
        Xtry = None
        if mode == 0:
            # cage expansion around rattler hole
            dots = bestX @ bestX[rattler]
            k = int(rng.integers(12, 19))
            idx = np.argsort(-dots)[1:k + 1]
            Xs = bestX.copy()
            alpha = float(rng.uniform(0.06, 0.15))
            Xs[idx] = Xs[idx] - alpha * bestX[rattler]
            Xs[idx] += 0.025 * rng.standard_normal((len(idx), d))
            # relocate rattler to largest void (50%) or jitter
            if trial % 2 == 0:
                vd = largest_void_dir(bestX, rng, s=8000)
                Xs[rattler] = vd + 0.01 * rng.standard_normal(d)
            else:
                Xs[rattler] += 0.05 * rng.standard_normal(d)
            Xs += 0.004 * rng.standard_normal((n, d))
            Xs /= np.linalg.norm(Xs, axis=1, keepdims=True)
            Xtry = smooth_adam(Xs, steps=2200, a0=35, a1=320, lr0=0.005, lr1=0.0004, t_deadline=EXPLORE_END)
            try_update(Xtry, 1.2)
        elif mode == 1:
            # cap kill: kill neighbourhood of rattler or overloaded point
            if rng.random() < 0.6:
                center = rattler
            else:
                center = int(order_hi[int(rng.integers(0, 10))])
            dots = bestX @ bestX[center]
            k = int(rng.integers(10, 20))
            idx = np.argsort(-dots)[:k]
            Xs = bestX.copy()
            Xs[idx] = random_sphere(k, d, rng)
            Xs += 0.008 * rng.standard_normal((n, d))
            Xs /= np.linalg.norm(Xs, axis=1, keepdims=True)
            Xtry = smooth_adam(Xs, steps=2400, a0=30, a1=300, lr0=0.006, lr1=0.0005, t_deadline=EXPLORE_END)
            try_update(Xtry, 1.2)
        elif mode == 2:
            # overload kill global refill
            k = int(rng.integers(4, 9))
            idx = order_hi[:k].copy()
            # add a couple random to break symmetry
            extra = int(rng.integers(0, 3))
            for _ in range(extra):
                idx = np.append(idx, int(rng.integers(0, n)))
            idx = np.unique(idx)
            Xs = bestX.copy()
            Xs[idx] = random_sphere(len(idx), d, rng)
            Xs += 0.010 * rng.standard_normal((n, d))
            Xs /= np.linalg.norm(Xs, axis=1, keepdims=True)
            Xtry = smooth_adam(Xs, steps=2000, a0=40, a1=320, lr0=0.005, lr1=0.0004, t_deadline=EXPLORE_END)
            try_update(Xtry, 1.0)
        elif mode == 3:
            # medium Gaussian kick fresh direction
            sigma = float(rng.choice([0.02, 0.04, 0.08, 0.12]))
            Xs = bestX + sigma * rng.standard_normal((n, d))
            Xs /= np.linalg.norm(Xs, axis=1, keepdims=True)
            Xtry = smooth_adam(Xs, steps=2000, a0=40, a1=300, lr0=0.005, lr1=0.0004, t_deadline=EXPLORE_END)
            # only LP if close
            if max_cos(Xtry) < bestV + 0.002:
                try_update(Xtry, 1.0)
            else:
                try_update(Xtry, None)
        elif mode == 4:
            # fresh Hopf restart (keeps balanced-basin pressure)
            left = (trial % 2 == 0)
            X0 = make_hopf(17, 9, rng, left=left)
            # shake harder to get different basin each time
            X0 = X0 + 0.01 * rng.standard_normal((n, d))
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
            Xtry = smooth_adam(X0, steps=2200, a0=30, a1=300, lr0=0.006, lr1=0.0005, t_deadline=EXPLORE_END)
            try_update(Xtry, 1.0)
        else:
            # void relocation + cage jitter: move 2 lowest-contact points to voids
            lows = order_lo[:2]
            Xs = bestX.copy()
            for li in lows:
                vd = largest_void_dir(Xs, rng, s=6000)
                Xs[int(li)] = vd
            Xs += 0.012 * rng.standard_normal((n, d))
            Xs /= np.linalg.norm(Xs, axis=1, keepdims=True)
            Xtry = smooth_adam(Xs, steps=2000, a0=45, a1=340, lr0=0.004, lr1=0.0004, t_deadline=EXPLORE_END)
            try_update(Xtry, 1.2)
    remain = DEADLINE - (time.time() - t0) - 1.0
    if remain > 1.0:
        bestX = lp_polish(bestX, min(remain, 25.0))
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX
