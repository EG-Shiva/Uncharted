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


def anneal_adam(X, steps, a0, a1, lr0, lr1, t_deadline=1e30):
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    lrat = lr1 / lr0 if lr0 != 0 else 1.0
    arat = a1 / a0 if a0 != 0 else 1.0
    for t in range(1, steps + 1):
        if (t & 127) == 0 and time.time() > t_deadline:
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
        if cnt > 2500:
            idx = np.where(sel)[0]
            order = np.argsort(cfull[idx])[-2500:]
            keep = np.zeros_like(sel)
            keep[idx[order]] = True
            sel = keep
            cnt = 2500
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


def contact_mask_flat(x, tol=1e-6):
    n = x.shape[0]
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    iu = np.triu_indices(n, 1)
    return (G[iu] > (m - tol)), m


def contact_counts_vec(x, tol=1e-6):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    B = G > (m - tol)
    return B.sum(axis=1), m


def void_fill(C, need, rng, trials=3000):
    # greedily add points far from C
    lst = [C]
    cur = C
    for _ in range(need):
        R = random_sphere(trials, 4, rng)
        D = R @ cur.T
        mx = np.max(D, axis=1)
        best = R[np.argmin(mx)]
        cur = np.vstack([cur, best[None, :]])
    return cur


def crossover_child(A, B, rng):
    from scipy.linalg import orthogonal_procrustes
    try:
        R, _ = orthogonal_procrustes(A, B)
        Bal = B @ R
    except Exception:
        Bal = B
    u = rng.standard_normal(4)
    u /= np.linalg.norm(u) + 1e-12
    da = A @ u
    db = Bal @ u
    # randomize side choice
    if rng.random() < 0.5:
        partA = A[da >= 0]
        partB = Bal[db < 0]
    else:
        partA = A[da < 0]
        partB = Bal[db >= 0]
    if len(partA) == 0:
        partA = A[:77]
    if len(partB) == 0:
        partB = Bal[:76]
    C = np.vstack([partA, partB])
    # resolve overlaps / duplicates: while closest pair too close, drop one
    # also trim to 153 if too many by dropping closest pairs
    rng2 = rng
    # iterative trim
    while len(C) > 153:
        G = C @ C.T
        np.fill_diagonal(G, -2.0)
        ij = np.argmax(G)
        i = ij // len(C)
        j = ij % len(C)
        # drop the one with larger mean dot (more crowded)
        si = np.sum(G[i])
        sj = np.sum(G[j])
        drop = i if si > sj else j
        C = np.delete(C, drop, axis=0)
    # remove near-duplicates even if len<=153 (dot>0.985)
    for _ in range(8):
        if len(C) < 2:
            break
        G = C @ C.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        if m < 0.985:
            break
        ij = np.argmax(G)
        i = ij // len(C)
        j = ij % len(C)
        si = np.sum(G[i])
        sj = np.sum(G[j])
        drop = i if si > sj else j
        C = np.delete(C, drop, axis=0)
    if len(C) < 153:
        need = 153 - len(C)
        C = void_fill(C, need, rng2, trials=2500)
    # small jitter to break exact symmetry
    C = C + 0.002 * rng.standard_normal(C.shape)
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    return C


def delete_to_153(Xbig, rng):
    X = Xbig.copy()
    while len(X) > 153:
        cc, _ = contact_counts_vec(X, tol=1e-5)
        # remove highest load; tie-break random
        mx = np.max(cc)
        cand = np.where(cc >= mx - 0.5)[0]
        rm = int(rng.choice(cand))
        X = np.delete(X, rm, axis=0)
    return X


def solve():
    t0 = time.time()
    DEADLINE = 176.0
    rng = np.random.default_rng(20261102)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    inits = []
    if xb is not None and xb.shape == (153, 4):
        inits.append(xb / np.linalg.norm(xb, axis=1, keepdims=True))
    if xr is not None and xr.shape == (153, 4):
        inits.append(xr / np.linalg.norm(xr, axis=1, keepdims=True))
    if len(inits) == 0:
        inits.append(random_sphere(153, 4, rng))
    bestX = None
    bestV = 1e30
    for c in inits:
        v = max_cos(c)
        if v < bestV:
            bestV = v
            bestX = c.copy()
    base_record = inits[-1].copy() if len(inits) > 0 else random_sphere(153, 4, rng)

    pool = []  # list of dicts X,V,mask

    def try_add(X):
        nonlocal bestX, bestV
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        v = max_cos(X)
        if v < bestV - 1e-12:
            bestV = v
            bestX = X.copy()
        if v > 0.86:
            return False
        mask, _ = contact_mask_flat(X, tol=1e-6)
        # distinctness: Hamming distance
        best_h = None
        best_idx = -1
        for k, e in enumerate(pool):
            h = int(np.sum(mask != e["mask"]))
            if best_h is None or h < best_h:
                best_h = h
                best_idx = k
        if best_h is not None and best_h < 80:
            # same basin: replace only if strictly better
            if v < pool[best_idx]["V"] - 2e-10:
                pool[best_idx] = {"X": X.copy(), "V": v, "mask": mask.copy()}
                pool.sort(key=lambda e: e["V"])
                return True
            return False
        pool.append({"X": X.copy(), "V": v, "mask": mask.copy()})
        pool.sort(key=lambda e: e["V"])
        # keep at most 10, drop worst
        while len(pool) > 10:
            pool.pop()
        return True

    # seed pool with incumbent
    try_add(bestX.copy())

    POOL_END = t0 + 95.0
    # Phase 1: diverse optima
    sigmas = [0.15, 0.22, 0.30, 0.45]
    big_ns = [157, 159, 160, 156]
    attempt = 0
    # plan ~8 attempts
    while time.time() < POOL_END and time.time() - t0 < DEADLINE - 80 and len(pool) < 10:
        attempt += 1
        mode = attempt % 5
        X0 = None
        if mode in (0, 1):
            X0 = random_sphere(153, 4, rng)
        elif mode in (2, 3):
            s = sigmas[(attempt // 5) % len(sigmas)] if attempt > 5 else sigmas[attempt % len(sigmas)]
            X0 = base_record + s * rng.standard_normal((153, 4))
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        else:
            nb = big_ns[attempt % len(big_ns)]
            Xb0 = random_sphere(nb, 4, rng)
            # short anneal on overloaded problem
            Xb1 = anneal_adam(Xb0, steps=1400, a0=25, a1=250, lr0=0.006, lr1=0.0006, t_deadline=POOL_END)
            X0 = delete_to_153(Xb1, rng)
        # short softmax anneal
        X1 = anneal_adam(X0, steps=1700, a0=30, a1=300, lr0=0.006, lr1=0.0005, t_deadline=POOL_END)
        # quick LP polish if promising
        v1 = max_cos(X1)
        if v1 < 0.866:
            remain = POOL_END - time.time()
            lt = min(1.4, max(0.0, remain - 0.2))
            if lt >= 0.4:
                X1 = lp_polish(X1, lt)
        try_add(X1)
        if time.time() >= POOL_END:
            break
    # ensure at least a few members: if pool small, add perturbed ones without polish
    while len(pool) < 4 and time.time() - t0 < DEADLINE - 60:
        X0 = base_record + 0.2 * rng.standard_normal((153, 4))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        X1 = anneal_adam(X0, steps=1200, a0=30, a1=280, lr0=0.006, lr1=0.0006, t_deadline=t0 + DEADLINE - 50)
        try_add(X1)

    CROSS_END = t0 + DEADLINE - 12.0
    it = 0
    while time.time() < CROSS_END and time.time() - t0 < DEADLINE - 12:
        it += 1
        if len(pool) < 2:
            # fallback random restart
            X0 = random_sphere(153, 4, rng)
            X1 = anneal_adam(X0, steps=1500, a0=30, a1=300, lr0=0.006, lr1=0.0005, t_deadline=CROSS_END)
            v1 = max_cos(X1)
            if v1 < 0.86:
                lt = min(1.2, max(0.0, CROSS_END - time.time() - 0.2))
                if lt > 0.4:
                    X1 = lp_polish(X1, lt)
            try_add(X1)
            continue
        # select parents: bias to best but keep diversity
        # pick first among top-4, second random among pool
        topk = min(4, len(pool))
        ia = int(rng.integers(0, topk))
        ib = int(rng.integers(0, len(pool)))
        if ib == ia:
            ib = (ib + 1 + int(rng.integers(0, len(pool) - 1))) % len(pool) if len(pool) > 1 else ia
        A = pool[ia]["X"]
        B = pool[ib]["X"]
        # occasionally cross best with overloaded-deleted fresh member
        if it % 7 == 0:
            nb = int(rng.choice([157, 158, 160]))
            Xb0 = random_sphere(nb, 4, rng)
            Xb1 = anneal_adam(Xb0, steps=900, a0=25, a1=220, lr0=0.006, lr1=0.0007, t_deadline=CROSS_END)
            B = delete_to_153(Xb1, rng)
            # light anneal B before crossover
            B = anneal_adam(B, steps=600, a0=60, a1=280, lr0=0.003, lr1=0.0005, t_deadline=CROSS_END)
        C = crossover_child(A, B, rng)
        C = anneal_adam(C, steps=1600, a0=35, a1=320, lr0=0.005, lr1=0.0004, t_deadline=CROSS_END)
        v = max_cos(C)
        if v < 0.862:
            remain = CROSS_END - time.time()
            lt = 1.2 if v < bestV + 0.002 else 0.7
            lt = min(lt, max(0.0, remain - 0.15))
            if lt >= 0.4:
                C = lp_polish(C, lt)
        try_add(C)
        # also try reciprocal cut occasionally (second child same pair opposite randomness)
        if time.time() < CROSS_END - 2 and it % 3 == 0:
            C2 = crossover_child(B, A, rng)
            C2 = anneal_adam(C2, steps=1200, a0=40, a1=320, lr0=0.005, lr1=0.0004, t_deadline=CROSS_END)
            v2 = max_cos(C2)
            if v2 < bestV + 0.001 and v2 < 0.862:
                remain = CROSS_END - time.time()
                lt = min(1.0, max(0.0, remain - 0.15))
                if lt >= 0.4:
                    C2 = lp_polish(C2, lt)
            try_add(C2)

    # final polish of global best with remaining time
    remain = (t0 + DEADLINE) - time.time()
    if remain > 1.0:
        try:
            bestX = lp_polish(bestX, min(remain - 0.5, 9.0))
            bestV = max_cos(bestX)
        except Exception:
            pass
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX
