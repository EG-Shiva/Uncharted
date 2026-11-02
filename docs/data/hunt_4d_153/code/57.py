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


def contact_info(x, tol=1e-9):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    B = G > (m - tol)
    cc = B.sum(axis=1)
    return cc, m, G


def smooth_adam_free(X, free_idx, steps, a0, a1, lr0, lr1, t_deadline=1e30):
    n, d = X.shape
    free_idx = np.asarray(free_idx, dtype=int)
    if len(free_idx) == 0:
        return X
    fmask = np.zeros(n, dtype=bool)
    fmask[free_idx] = True
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    lrat = lr1 / lr0 if lr0 != 0 else 1.0
    arat = a1 / a0 if a0 != 0 else 1.0
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
        grad_raw = W @ X
        dot = np.sum(grad_raw * X, axis=1, keepdims=True)
        grad_t = grad_raw - dot * X
        # zero grad for frozen
        grad_t[~fmask] = 0.0
        m *= b1
        m += (1.0 - b1) * grad_t
        v *= b2
        v += (1.0 - b2) * (grad_t * grad_t)
        b1t = 1.0 - b1 ** t
        b2t = 1.0 - b2 ** t
        mhat = m / b1t
        vhat = v / b2t
        step = lr * mhat / (np.sqrt(vhat) + eps)
        # project step tangential
        sdot = np.sum(step * X, axis=1, keepdims=True)
        step -= sdot * X
        step[~fmask] = 0.0
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
    I0 = iu[0]
    J0 = iu[1]
    while time.time() - start < seconds and radius > 1e-14:
        cfull = (x @ x.T)[iu]
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


def greedy_new_vecs(X_keep, k_add, rng, S_total):
    # sequentially sample S_total random vectors per insertion, keep minimax
    X_cur = X_keep.copy()
    new_vecs = []
    chunk = 25000
    n_chunks = max(1, S_total // chunk)
    for _ in range(k_add):
        best_val = 1e30
        best_vec = None
        for __ in range(n_chunks):
            C = rng.standard_normal((chunk, 4))
            C /= np.linalg.norm(C, axis=1, keepdims=True)
            D = C @ X_cur.T
            vals = np.max(D, axis=1)
            i = int(np.argmin(vals))
            if float(vals[i]) < best_val:
                best_val = float(vals[i])
                best_vec = C[i].copy()
        # small random jitter
        best_vec = best_vec + 0.005 * rng.standard_normal(4)
        best_vec /= np.linalg.norm(best_vec)
        new_vecs.append(best_vec)
        X_cur = np.vstack([X_cur, best_vec.reshape(1, -1)])
    return np.array(new_vecs)


def solve():
    t0 = time.time()
    DEADLINE = 178.0
    EXPLORE_END = t0 + 148.0
    rng = np.random.default_rng(20261102)
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
    k_cycle = [4, 6, 8, 12]
    trial = 0
    while time.time() < EXPLORE_END and (time.time() - t0) < DEADLINE - 30:
        trial += 1
        k = k_cycle[(trial - 1) % len(k_cycle)]
        # identify rattler + hubs + neighbours from incumbent
        cc, mcur, Gfull = contact_info(bestX, tol=1e-9)
        rattler = int(np.argmin(cc))
        # nearest neighbours of rattler (largest cosine)
        Gr = bestX @ bestX[rattler]
        Gr[rattler] = -2.0
        nn_order = np.argsort(-Gr)
        hubs = np.where(cc >= 7)[0]
        # build delete set: rattler + k others
        delete_set = set()
        delete_set.add(rattler)
        # numbers from neighbours vs hubs
        n_from_nn = (k + 1) // 2
        n_from_hub = k - n_from_nn
        # take neighbours (skip rattler itself), with some randomisation
        # consider top 25 neighbours, sample
        pool_nn = [int(a) for a in nn_order[:25] if int(a) != rattler]
        rng.shuffle(pool_nn)
        for a in pool_nn[:n_from_nn]:
            delete_set.add(a)
        # hubs
        if len(hubs) > 0:
            hub_list = [int(a) for a in hubs if int(a) not in delete_set]
            rng.shuffle(hub_list)
            for a in hub_list[:n_from_hub]:
                delete_set.add(a)
        # fill remainder randomly
        while len(delete_set) < k + 1:
            delete_set.add(int(rng.integers(0, n)))
        del_idx = np.array(sorted(delete_set), dtype=int)
        k_add = len(del_idx)
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[del_idx] = False
        X_keep = bestX[keep_mask]
        # refill: 100k-200k samples per insertion (use 100k, 60k for large k to save time)
        if k_add <= 7:
            S_total = 100000
        elif k_add <= 9:
            S_total = 80000
        else:
            S_total = 60000
        # alternate larger sampling every 4th trial
        if trial % 4 == 0:
            S_total = min(200000 if k_add <= 5 else 100000, S_total + 20000)
        try:
            new_vecs = greedy_new_vecs(X_keep, k_add, rng, S_total)
        except Exception:
            continue
        if time.time() > EXPLORE_END:
            break
        Xtry = bestX.copy()
        # shuffle assignment to break symmetry
        perm = rng.permutation(k_add)
        for ii, di in enumerate(del_idx):
            Xtry[di] = new_vecs[perm[ii]]
        Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
        # neighbourhood within cosine > 0.75 of refilled points
        # compute max dot to deleted set
        Dmat = Xtry @ Xtry[del_idx].T
        maxd = np.max(Dmat, axis=1)
        neigh = np.where(maxd > 0.75)[0]
        freeA = np.union1d(del_idx, neigh)
        # phase A: optimise only refilled + neighbours, 500-1000 steps beta 50->300
        stepsA = 700 if k_add <= 9 else 500
        Xtry = smooth_adam_free(Xtry, freeA, steps=stepsA, a0=50, a1=300,
                                lr0=0.005, lr1=0.0005, t_deadline=EXPLORE_END)
        if time.time() > EXPLORE_END:
            # still evaluate?
            pass
        # phase B: unfreeze all for another 1000 steps
        Xtry = smooth_adam_free(Xtry, np.arange(n), steps=1000, a0=50, a1=300,
                                lr0=0.004, lr1=0.0004, t_deadline=EXPLORE_END)
        v = max_cos(Xtry)
        # keep candidate even if pre-LP up to 0.005 worse (topology change)
        need_lp = False
        lp_time = 1.0
        if v < bestV - 1e-12:
            need_lp = True
            lp_time = 2.5
        elif v < bestV + 0.005:
            need_lp = True
            lp_time = 1.2
        else:
            # too bad, skip LP
            continue
        remain = EXPLORE_END - time.time()
        if remain < 0.4:
            # no time for LP, just update if improved
            if v < bestV:
                bestV = v
                bestX = Xtry.copy()
            continue
        lp_time = min(lp_time, remain - 0.05)
        if lp_time < 0.3:
            if v < bestV:
                bestV = v
                bestX = Xtry.copy()
            continue
        try:
            Xp = lp_polish(Xtry, lp_time)
        except Exception:
            continue
        vp = max_cos(Xp)
        if vp < bestV:
            bestV = vp
            bestX = Xp.copy()
        elif v < bestV:
            bestV = v
            bestX = Xtry.copy()
        # loop continues; rattler recomputed next trial
    remain = DEADLINE - (time.time() - t0) - 1.0
    if remain > 1.0:
        try:
            bestX = lp_polish(bestX, min(remain, 25.0))
        except Exception:
            pass
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX
