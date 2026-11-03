import time
import itertools
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def top_k_sorted(x, k=50):
    n = x.shape[0]
    iu = np.triu_indices(n, 1)
    c = (x @ x.T)[iu]
    if len(c) <= k:
        return np.sort(c)[::-1]
    part = np.argpartition(c, -k)[-k:]
    return np.sort(c[part])[::-1]


def lp_polish(x, seconds, radius_init=1e-3, t_deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = float(radius_init)
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if t_deadline is not None and time.time() > t_deadline:
            break
        if time.time() - start > seconds - 0.05:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        thresh = m - max(20 * radius, 1e-9)
        sel = c > thresh
        if int(sel.sum()) > 1500:
            idx = np.argpartition(c, -1500)[-1500:]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def build_candidates():
    d = 13
    rows = []
    for k in (3, 4):
        for supp in itertools.combinations(range(d), k):
            # fix sign of first support entry to +1 for hemisphere
            s0 = supp[0]
            rest = supp[1:]
            for mask in range(1 << (k - 1)):
                v = np.zeros(d)
                v[s0] = 1.0
                for t, pos in enumerate(rest):
                    v[pos] = 1.0 if ((mask >> t) & 1) else -1.0
                v = v / np.sqrt(k)
                rows.append(v)
    C = np.stack(rows, axis=0)
    return C


def greedy_build(C, rng):
    M = C.shape[0]
    n_target = 95
    # start from index 0
    chosen = [int(rng.integers(0, M))]
    # pre-normalized already
    for k in range(1, n_target):
        S = C[chosen]  # k x d
        D = C @ S.T  # M x k
        # mask chosen rows
        D[chosen, :] = -2.0  # so they are not selected? need per-row max
        # for chosen rows max will be ~ large dot to S (includes self=1) but we masked to -2; still max may be from others; force to inf to exclude
        mx = D.max(axis=1)
        mx[np.array(chosen)] = np.inf
        # count of attainment within 1e-12
        # to avoid full exp cost, first find minimal mx
        mmin = mx.min()
        cand_pool = np.where(mx <= mmin + 1e-12)[0]
        # among pool compute counts and sumexp
        if len(cand_pool) > 1:
            Dc = D[cand_pool]
            cnt = (Dc >= mmin - 1e-12).sum(axis=1)
            cmin = cnt.min()
            cand_pool2 = cand_pool[cnt == cmin]
            if len(cand_pool2) > 1:
                Dc2 = D[cand_pool2]
                # sum exp(20*dot)
                s = np.exp(20.0 * Dc2).sum(axis=1)
                pick = cand_pool2[int(np.argmin(s))]
            else:
                pick = int(cand_pool2[0])
        else:
            pick = int(cand_pool[0])
        chosen.append(pick)
    return np.array(chosen, dtype=int)


def swap_improve(C, sel_idx, time_limit, rng):
    # sel_idx: (95,) indices into C
    n = len(sel_idx)
    start = time.time()
    X = C[sel_idx].copy()
    cur_top = top_k_sorted(X, 50)
    # build unused pool ranking: 200 best unused
    M = C.shape[0]
    is_sel = np.zeros(M, dtype=bool)
    is_sel[sel_idx] = True
    unused = np.where(~is_sel)[0]
    # rank unused by max dot
    D = C[unused] @ X.T
    umax = D.max(axis=1)
    order = np.argsort(umax, kind="stable")
    K = min(200, len(unused))
    pool = unused[order[:K]].copy()
    # passes
    improved = True
    order_p = np.arange(n)
    while improved and (time.time() - start < time_limit):
        improved = False
        rng.shuffle(order_p)
        for p in order_p:
            if time.time() - start >= time_limit:
                break
            # internal pairs of S without p
            mask = np.ones(n, dtype=bool)
            mask[p] = False
            Srest = X[mask]
            # internal top50
            G = Srest @ Srest.T
            iu = np.triu_indices(n - 1, 1)
            c_int = G[iu]
            if len(c_int) >= 50:
                idxp = np.argpartition(c_int, -50)[-50:]
                Itop = np.sort(c_int[idxp])[::-1]
            else:
                Itop = np.sort(c_int)[::-1]
                Itop = np.pad(Itop, (0, 50 - len(Itop)), constant_values=-2.0)
            # dots of pool to Srest
            Cp = C[pool]  # K x d
            B = Cp @ Srest.T  # K x d_rest
            # for each candidate compute merged top50 and compare
            # pre-sort each row top50 descending
            # use partition for speed
            if B.shape[1] >= 50:
                # partition each row
                part = np.argpartition(B, -50, axis=1)[:, -50:]
                Brow = np.take_along_axis(B, part, axis=1)
                Brow = np.sort(Brow, axis=1)[:, ::-1]
            else:
                Brow = np.sort(B, axis=1)[:, ::-1]
                Brow = np.pad(Brow, ((0, 0), (0, 50 - Brow.shape[1])), constant_values=-2.0)
            best_merge = None
            best_k = -1
            # current top for comparison
            # iterate candidates, merge Itop and Brow[k]
            for kk in range(K):
                # skip if this pool entry already selected (after swaps)
                # merge two sorted 50-lists
                a = Itop
                b = Brow[kk]
                # merge first 50
                m = np.empty(50)
                ia = ib = im = 0
                while im < 50:
                    if ib >= 50 or (ia < 50 and a[ia] >= b[ib]):
                        m[im] = a[ia]
                        ia += 1
                    else:
                        m[im] = b[ib]
                        ib += 1
                    im += 1
                # lexicographic compare to cur_top (tolerance 1e-12)
                diff = m - cur_top
                # find first index with |diff|>1e-12
                nz = np.where(np.abs(diff) > 1e-12)[0]
                if len(nz) == 0:
                    continue
                j = nz[0]
                if m[j] < cur_top[j] - 1e-12:
                    if best_merge is None or (m[best_merge.nonzero()[0][0]] if False else True):
                        # compare to best_merge
                        if best_merge is None:
                            best_merge = m
                            best_k = kk
                        else:
                            d2 = m - best_merge
                            nz2 = np.where(np.abs(d2) > 1e-12)[0]
                            if len(nz2) > 0 and m[nz2[0]] < best_merge[nz2[0]]:
                                best_merge = m
                                best_k = kk
            if best_merge is not None:
                # apply swap
                # check lexicographically better than cur_top (already)
                X[p] = C[pool[best_k]]
                sel_idx[p] = pool[best_k]
                # remove used pool entry, replace with next best unused if available
                # mark used; refill from unused ranking beyond K if possible
                # simple: drop it
                pool = np.delete(pool, best_k)
                K -= 1
                if K < 10:
                    break
                cur_top = best_merge
                improved = True
        # end for p
    return X, sel_idx, cur_top


def smooth_relax(X, steps, b0, b1, lr0, lr1, mu=0.9, penalty_C=None, lam0=0.0, half_steps=None, rng=None, t_deadline=None):
    n, d = X.shape
    V = np.zeros_like(X)
    lB0, lB1 = np.log(b0), np.log(b1)
    lL0, lL1 = np.log(lr0), np.log(lr1)
    # for penalty: nearest lattice indices
    nn_idx = None
    if penalty_C is not None:
        # initial assignment
        D = X @ penalty_C.T
        nn_idx = np.argmax(D, axis=1)
    for s in range(steps):
        if t_deadline is not None and time.time() > t_deadline:
            break
        f = s / max(1, steps - 1)
        beta = float(np.exp(lB0 + (lB1 - lB0) * f))
        lr = float(np.exp(lL0 + (lL1 - lL0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        mx = G.max()
        # softmax
        E = np.exp(beta * (G - mx))
        np.fill_diagonal(E, 0.0)
        sw = E.sum()
        if not np.isfinite(sw) or sw == 0:
            break
        W = E / sw
        Grad = W @ X
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        # symmetry penalty (pull to nearest lattice) decaying to zero in first half
        if penalty_C is not None and lam0 > 0:
            if half_steps is None:
                half_steps = steps // 2
            if s < half_steps:
                lam = lam0 * (1.0 - s / half_steps)
                # refresh nn every 100 steps
                if s % 100 == 0 and s > 0:
                    D = X @ penalty_C.T
                    nn_idx = np.argmax(D, axis=1)
                Cn = penalty_C[nn_idx]
                # tangent pull toward Cn
                cdot = np.sum(Cn * X, axis=1, keepdims=True)
                Pull = Cn - cdot * X
                # Gt includes repulsion; penalty adds -lam*Pull to gradient? we want to minimize distance so move toward Cn
                # Our update is X <- X - lr*V where V accumulates Gt. To pull toward Cn, subtract lam*Pull from Gt with negative sign:
                Gt = Gt - lam * Pull
        V = mu * V + Gt
        X = X - lr * V
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def solve():
    t0 = time.time()
    BUDGET = 172.0
    rng = np.random.default_rng(12345)
    # load incumbents
    try:
        xb = np.load("best.npy").astype(float)
        xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
        xr = xr / np.linalg.norm(xr, axis=1, keepdims=True)
    except Exception:
        xr = None
    best_overall = None
    best_val = np.inf
    for cand in (xb, xr):
        if cand is not None and cand.shape == (95, 13):
            v = max_cos(cand)
            if v < best_val:
                best_val = v
                best_overall = cand.copy()

    # structured lattice basin
    C = build_candidates()  # ~6864 x 13
    # greedy build (deterministic with rng)
    try:
        sel0 = greedy_build(C, rng)
    except Exception:
        # fallback random subset
        sel0 = rng.choice(C.shape[0], size=95, replace=False)
    Xd = C[sel0].copy()
    # small deterministic jitter to break exact ties before swaps? keep lattice exact for swaps
    # swap improve ~ up to 25s but leave time for relax
    remain = BUDGET - (time.time() - t0)
    swap_sec = min(28.0, max(8.0, remain - 120.0))
    # ensure at least some swap time if plenty remains
    if remain > 140:
        swap_sec = 25.0
    try:
        Xd, sel0, _ = swap_improve(C, sel0.copy(), swap_sec, rng)
    except Exception:
        Xd = C[sel0].copy()

    # release lattice constraint: two branches
    # add tiny jitter to break symmetry before continuous relax
    rng2 = np.random.default_rng(7)
    Xj = Xd + 1e-4 * rng2.standard_normal(Xd.shape)
    Xj /= np.linalg.norm(Xj, axis=1, keepdims=True)

    # time split for smooth phases
    # reserve ~30s for final LP polishes
    t_now = time.time() - t0
    time_left = BUDGET - t_now - 32.0
    if time_left < 20:
        time_left = max(10.0, BUDGET - t_now - 10.0)
    # split equally
    half = time_left / 2.0

    # Branch A: plain smooth relax
    Xa = Xj.copy()
    # estimate steps from time: ~ 0.004s per step? calibrate: 95x95 matmul ~ fast ~0.0002s. Use 1500 steps default.
    # adapt steps to half time: assume ~800 steps per 10s
    stepsA = int(np.clip(half * 80, 400, 2500))
    t_deadA = t0 + (time.time() - t0) + half
    try:
        Xa = smooth_relax(Xa, stepsA, 30.0, 500.0, 0.15, 0.012, mu=0.9, penalty_C=None, rng=rng, t_deadline=t_deadA)
    except Exception:
        pass
    va = max_cos(Xa)

    # Branch B: symmetry-penalised first half
    Xb2 = Xj.copy()
    stepsB = int(np.clip(half * 80, 400, 2500))
    t_deadB = t_deadA + half
    # if little time, reuse Xa
    try:
        Xb2 = smooth_relax(Xb2, stepsB, 30.0, 500.0, 0.15, 0.012, mu=0.9, penalty_C=C, lam0=0.35, half_steps=stepsB // 2, rng=rng, t_deadline=t_deadB)
    except Exception:
        pass
    vb = max_cos(Xb2)

    # LP polish both branches + incumbent with remaining time
    t_rem = BUDGET - (time.time() - t0)
    # allocate: polish better branch longer, plus quick polish of best_overall? best_overall already jammed, short polish enough
    # order by current value
    branches = [(va, Xa), (vb, Xb2)]
    branches.sort(key=lambda t: t[0])
    # polish time split: 60% to best branch, 30% to second, 10% to incumbent
    if t_rem < 8:
        t_rem = 8.0
    t1 = t_rem * 0.55
    t2 = t_rem * 0.30
    t3 = t_rem * 0.15
    try:
        Xp0 = lp_polish(branches[0][1], t1)
        vp0 = max_cos(Xp0)
    except Exception:
        Xp0, vp0 = branches[0][1], branches[0][0]
    try:
        Xp1 = lp_polish(branches[1][1], t2)
        vp1 = max_cos(Xp1)
    except Exception:
        Xp1, vp1 = branches[1][1], branches[1][0]

    cands = [(vp0, Xp0), (vp1, Xp1), (va, Xa), (vb, Xb2)]
    if best_overall is not None:
        # light re-polish incumbent to squeeze (short)
        try:
            Xi = lp_polish(best_overall, min(t3, 6.0))
            vi = max_cos(Xi)
        except Exception:
            Xi, vi = best_overall, best_val
        cands.append((vi, Xi))
        cands.append((best_val, best_overall))
    cands.sort(key=lambda t: t[0])
    Xbest = cands[0][1]
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest
