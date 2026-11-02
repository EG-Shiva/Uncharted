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


def smooth_adam_w(X, steps, a0, a1, lr0, lr1, w_new=None, idx_new=-1, t_deadline=1e30):
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    lrat = lr1 / lr0 if lr0 != 0 else 1.0
    arat = a1 / a0 if a0 != 0 else 1.0
    has_w = (w_new is not None) and (0 <= idx_new < n)
    for t in range(1, steps + 1):
        if t % 200 == 0 and time.time() > t_deadline:
            break
        frac = t / steps
        alpha = a0 * (arat ** frac)
        lr = lr0 * (lrat ** frac)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m0 = np.max(G)
        W = np.exp(alpha * (G - m0))
        np.fill_diagonal(W, 0.0)
        if has_w and abs(w_new - 1.0) > 1e-12:
            W[idx_new, :] *= w_new
            W[:, idx_new] *= w_new
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
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], axis=1).ravel()
        rows = np.repeat(np.arange(p), 2 * d + 1)
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
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
    fallbackX = bestX.copy()
    fallbackV = bestV
    n, d = 153, 4

    # identify rattler (fewest contacts)
    cc, mm = contact_counts(bestX, tol=1e-9)
    rattler = int(np.argmin(cc))

    # ---- N=152: delete rattler and polish ----
    X152_0 = np.delete(bestX, rattler, axis=0)
    # short polish of 152 core (should drop well below 0.8491)
    remain = DEADLINE - (time.time() - t0)
    X152 = lp_polish(X152_0, min(4.0, max(0.5, remain - 100.0)) if remain > 20 else 2.0)
    v152 = max_cos(X152)

    # ---- largest voids over dense random sample ----
    M = 60000
    S = random_sphere(M, d, rng)
    # max dot to core
    dots = S @ X152.T
    mdots = np.max(dots, axis=1)
    # pick several well-separated voids
    nvoid = 4
    voids = []
    order = np.argsort(mdots)  # smallest first = largest void
    taken = []
    for idx in order:
        cand = S[idx]
        ok = True
        for t in taken:
            if float(cand @ t) > 0.90:
                ok = False
                break
        if ok:
            voids.append(cand.copy())
            taken.append(cand.copy())
            if len(voids) >= nvoid:
                break
    if len(voids) == 0:
        voids = [S[order[0]]]

    # ---- adiabatic insertion: ramp lambda 0->1 ----
    STAGES = 8
    # time split: half budget to insertions
    INSERT_END = t0 + 95.0
    for vi, voidv in enumerate(voids):
        if time.time() > INSERT_END:
            break
        if time.time() - t0 > DEADLINE - 60:
            break
        X = np.vstack([X152, voidv.reshape(1, d)])
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        idx_new = 152
        for s in range(STAGES):
            if time.time() > INSERT_END:
                break
            lam = (s + 1) / STAGES  # 0->1
            # soften early lambdas
            lam_eff = lam ** 0.8
            # soft-max Adam at this lambda
            X = smooth_adam_w(X, steps=350, a0=30, a1=250, lr0=0.005, lr1=0.0005,
                              w_new=lam_eff, idx_new=idx_new, t_deadline=INSERT_END)
            # 1s LP polish (trust region keeps it gradual)
            rem = INSERT_END - time.time()
            if rem < 0.4:
                break
            X = lp_polish(X, min(0.9, rem - 0.05))
        v = max_cos(X)
        # final short extra adam at full weight + polish if promising
        if v < bestV + 0.003:
            rem = INSERT_END - time.time()
            if rem > 1.2:
                X = smooth_adam_w(X, steps=500, a0=50, a1=320, lr0=0.003, lr1=0.0004,
                                  w_new=1.0, idx_new=-1, t_deadline=INSERT_END)
                rem2 = INSERT_END - time.time()
                if rem2 > 0.4:
                    X = lp_polish(X, min(1.2, rem2 - 0.05))
                v = max_cos(X)
        if v < bestV:
            bestV = v
            bestX = X.copy()

    # ---- symmetric variant: 154 via duplicated rattler ----
    if time.time() - t0 < DEADLINE - 30:
        # build 154
        rvec = fallbackX[rattler].copy()
        dup = rvec + 0.025 * rng.standard_normal(d)
        # make offset mostly tangential
        dup -= (dup @ rvec) * rvec * 0.0
        dup /= np.linalg.norm(dup)
        X154_0 = np.vstack([fallbackX, dup.reshape(1, d)])
        X154_0 /= np.linalg.norm(X154_0, axis=1, keepdims=True)
        # optimize briefly
        DELET_START = time.time()
        X154 = smooth_adam_w(X154_0, steps=1400, a0=40, a1=280, lr0=0.005, lr1=0.0005,
                             w_new=1.0, idx_new=-1, t_deadline=t0 + DEADLINE - 25)
        rem = (t0 + DEADLINE - 25) - time.time()
        if rem > 0.8:
            X154 = lp_polish(X154, min(2.0, rem * 0.3))
        # candidate deletions: ~20 choices
        # include high-contact points, random, and the two rattler copies
        cc154, _ = contact_counts(X154, tol=1e-9)
        order_hi = np.argsort(-cc154)
        cand_del = set()
        cand_del.add(153)  # duplicate
        cand_del.add(rattler)
        # top overloaded
        for a in order_hi[:8]:
            cand_del.add(int(a))
        while len(cand_del) < 20:
            cand_del.add(int(rng.integers(0, 154)))
        cand_del = sorted(cand_del)[:20]
        # raw scores without polish
        raw = []
        G154 = X154 @ X154.T
        for k in cand_del:
            mask = np.ones(154, dtype=bool)
            mask[k] = False
            Gm = G154[np.ix_(mask, mask)]
            np.fill_diagonal(Gm, -2.0)
            raw.append((float(np.max(Gm)), k))
        raw.sort()
        # polish top few
        topK = 5
        for (rv, k) in raw[:topK]:
            if time.time() - t0 > DEADLINE - 12:
                break
            mask = np.ones(154, dtype=bool)
            mask[k] = False
            X153 = X154[mask].copy()
            # quick adam + LP
            X153 = smooth_adam_w(X153, steps=600, a0=50, a1=320, lr0=0.003, lr1=0.0004,
                                 w_new=1.0, idx_new=-1, t_deadline=t0 + DEADLINE - 10)
            rem = (t0 + DEADLINE - 10) - time.time()
            if rem > 0.5:
                X153 = lp_polish(X153, min(1.5, rem))
            v = max_cos(X153)
            if v < bestV:
                bestV = v
                bestX = X153.copy()
        # also try polishing best raw without adam if time low
        # (already covered)

    # fallback safety
    if bestX is None or bestV > fallbackV + 1e-12:
        bestX = fallbackX.copy()
        bestV = fallbackV
    # final thorough polish with remaining time
    rem = DEADLINE - (time.time() - t0) - 1.0
    if rem > 1.0:
        Xp = lp_polish(bestX, min(rem, 12.0))
        vp = max_cos(Xp)
        if vp < bestV:
            bestX = Xp
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    # strict feasibility: ensure shape
    if bestX.shape != (153, 4):
        bestX = fallbackX.copy()
    return bestX
