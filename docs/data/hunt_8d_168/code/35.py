import time
import numpy as np
from scipy.optimize import linprog, linear_sum_assignment
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lp_polish(x, seconds, t_end=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while True:
        if time.time() - start > seconds:
            break
        if t_end is not None and time.time() > t_end:
            break
        if radius < 1e-14:
            break
        Gfull = x @ x.T
        cv = Gfull[iu]
        m = cv.max()
        sel = cv > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 2600:
            idx = np.argsort(cv[sel])[-2600:]
            I, J = I[idx], J[idx]
            cc = cv[sel][idx]
            p = 2600
        else:
            cc = cv[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cc, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def _soft_run(x_init, steps, a0, a1, lr0, lr1, mu, rng):
    x = x_init.copy()
    n, d = x.shape
    vel = np.zeros_like(x)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        alpha = a0 + (a1 - a0) * frac
        lr = lr0 * (lr1 / lr0) ** frac if (lr0 > 0 and lr1 > 0) else lr0
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        S = W.sum(axis=1, keepdims=True) + 1e-12
        A = (W @ x) / S
        dot = np.sum(A * x, axis=1, keepdims=True)
        At = A - dot * x
        vel = mu * vel + At
        x = x - lr * vel
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _kabsch_R(M):
    U, _, Vt = np.linalg.svd(M)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt2 = Vt.copy()
        Vt2[-1, :] *= -1
        R = Vt2.T @ U.T
    return R

def _align_info(A, B, iters=3):
    # returns R, perm, rmsd, B_aligned ; B_aligned[i] corresponds to A[i]
    n, d = A.shape
    R = np.eye(d)
    perm = np.arange(n)
    for _ in range(iters):
        C = (A @ R) @ B.T  # A_rot? actually x^T R y -> (A R) B^T? we use A@R vs B
        # Our derivation: Cij = x_i^T R y_j ; with R current
        # First iter R=I gives X Y^T
        ri, cj = linear_sum_assignment(-C)
        # build perm: perm[i]=j matched to i ; ri should be arange
        P = np.empty(n, dtype=int)
        P[ri] = cj
        perm = P
        Yp = B[perm]
        M = Yp.T @ A  # d x d ; wait objective trace(R M)? M=Yp^T X
        # Actually we maximize trace(R M) with M=Yp^T A
        Rn = _kabsch_R(M)
        R = Rn
    # final assignment with final R
    C = (A @ R) @ B.T
    # careful: Cij defined as? We used A@R dot B : (A R) . B = A^T? This corresponds to x^T R^T? sign.
    # To be consistent, define aligned B' = B @ R.T (?) Let's just recompute properly:
    # We want R acting on y: score = x^T (R y) = (x^T R) y^T? row x @ R @ y^T.
    # So C = A @ R @ B.T is correct for R acting on y columns.
    ri, cj = linear_sum_assignment(-C)
    P = np.empty(n, dtype=int)
    P[ri] = cj
    perm = P
    Yp = B[perm]
    M = Yp.T @ A
    R = _kabsch_R(M)
    # aligned B:
    Bal = (R @ Yp.T).T  # each row = R y
    # rmsd
    dots = np.sum(A * Bal, axis=1)
    dots = np.clip(dots, -1, 1)
    rmsd = float(np.sqrt(np.maximum(0, 2 - 2 * dots).mean()))
    return R, perm, rmsd, Bal

def _rmsd_only(A, B):
    _, _, r, _ = _align_info(A, B, iters=3)
    return r

def _farthest_refill(cur, need, rng, n_try=4000):
    d = cur.shape[1]
    out = []
    C = cur.copy()
    for _ in range(need):
        cand = rng.normal(size=(n_try, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        D = cand @ C.T
        md = D.max(axis=1)
        bi = int(np.argmin(md))
        best = cand[bi]
        out.append(best)
        C = np.vstack([C, best.reshape(1, -1)])
    if len(out) == 0:
        return cur
    return np.vstack([cur, np.array(out)])

def _crossover_child(A, B, rng):
    n, d = A.shape
    R, perm, _, Bal = _align_info(A, B, iters=3)
    # random cap / half-space direction
    u = rng.normal(size=d); u /= np.linalg.norm(u)
    r = rng.random()
    if r < 0.4:
        thr = 0.0  # half-space
    elif r < 0.7:
        thr = float(rng.uniform(-0.4, 0.6))
    else:
        thr = float(rng.uniform(0.1, 0.7))
    # randomly choose which parent provides cap interior to add variety
    swap = rng.random() < 0.5
    projA = A @ u
    # Bal corresponds to A index-wise
    if not swap:
        mask = projA > thr
    else:
        mask = projA <= thr
        # equivalent to taking B inside cap defined by -u; keep complementary logic
    child = np.empty_like(A)
    child[mask] = A[mask]
    child[~mask] = Bal[~mask]
    # delete near-duplicates (cos>0.9): keep first occurrence
    G = child @ child.T
    np.fill_diagonal(G, -2.0)
    # find pairs >0.9 greedily
    keep = np.ones(n, dtype=bool)
    # order random to avoid bias
    order = rng.permutation(n)
    # simple greedy: iterate order, keep if not too close to already kept
    kept_list = []
    keep[:] = False
    cur = []
    for idx in order:
        v = child[idx]
        if len(cur) == 0:
            keep[idx] = True
            cur.append(v)
        else:
            Cmat = np.array(cur)
            md = float((Cmat @ v).max())
            if md > 0.9:
                pass  # drop duplicate
            else:
                keep[idx] = True
                cur.append(v)
    kept = child[keep]
    need = n - kept.shape[0]
    if need > 0:
        child2 = _farthest_refill(kept, need, rng, n_try=3000)
    else:
        child2 = kept
    # small jitter to break exact symmetry
    child2 = child2 + rng.normal(size=child2.shape) * 1e-4
    child2 /= np.linalg.norm(child2, axis=1, keepdims=True)
    return child2

def solve():
    t0 = time.time()
    BUDGET = 172.0
    rng = np.random.default_rng(12345)
    n, d = 168, 8
    try:
        best0 = np.load("best.npy").astype(float)
    except Exception:
        best0 = None
    try:
        rec0 = np.load("record.npy").astype(float)
    except Exception:
        rec0 = None
    cands = []
    if best0 is not None and best0.shape == (n, d):
        cands.append(best0 / np.linalg.norm(best0, axis=1, keepdims=True))
    if rec0 is not None and rec0.shape == (n, d):
        rn = rec0 / np.linalg.norm(rec0, axis=1, keepdims=True)
        cands.append(rn)
    if not cands:
        c = rng.normal(size=(n, d))
        c /= np.linalg.norm(c, axis=1, keepdims=True)
        cands.append(c)
    best_x = min(cands, key=_max_cos)
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    best_val = _max_cos(best_x)
    # baseline tighten
    if time.time() - t0 < BUDGET - 60:
        bx, bv = _lp_polish(best_x, 2.0, t_end=t0 + BUDGET - 20)
        if bv < best_val:
            best_x, best_val = bx, bv

    archive = []  # list of (x, val)
    archive.append((best_x.copy(), best_val))

    def is_novel(x, thresh=0.05):
        for (y, _) in archive:
            try:
                r = _rmsd_only(x, y)
            except Exception:
                continue
            if r < thresh:
                return False, r
        return True, 1e9

    # Phase 1: generate 15-30 LP-polished optima from large kicks
    n_seed = 14
    for s in range(n_seed):
        if time.time() - t0 > BUDGET - 90:
            break
        base = best_x
        # large kick
        if s % 3 == 0:
            sigma = float(np.exp(rng.uniform(np.log(0.08), np.log(0.35))))
            x0 = base + rng.normal(size=base.shape) * sigma
        elif s % 3 == 1:
            # kill subset and farthest refill
            krep = int(rng.integers(8, 24))
            kill = rng.choice(n, size=krep, replace=False)
            m = np.ones(n, dtype=bool); m[kill] = False
            kept = base[m]
            x0 = _farthest_refill(kept, krep, rng, n_try=3000)
        else:
            # random orthogonal + jitter (different orientation same basin but kick)
            Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
            if np.linalg.det(Q) < 0:
                Q[:, 0] *= -1
            x0 = (base @ Q.T) + rng.normal(size=base.shape) * float(rng.uniform(0.05, 0.2))
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        steps = int(rng.choice([1000, 1500, 2000]))
        a0 = float(rng.uniform(15, 50)); a1 = float(rng.uniform(150, 320))
        lr0 = float(np.exp(rng.uniform(np.log(0.03), np.log(0.12))))
        lr1 = float(np.exp(rng.uniform(np.log(0.001), np.log(0.015))))
        mu = float(rng.uniform(0.75, 0.90))
        xq = _soft_run(x0, steps, a0, a1, lr0, lr1, mu, rng)
        xp, vp = _lp_polish(xq, 1.2, t_end=t0 + BUDGET - 20)
        if vp < best_val:
            best_x, best_val = xp.copy(), vp
            archive.append((xp.copy(), vp))
        else:
            nov, _ = is_novel(xp, thresh=0.05)
            if nov and vp < best_val + 0.01:
                archive.append((xp.copy(), vp))
                if len(archive) > 30:
                    # drop worst non-best
                    worst = max(range(len(archive)), key=lambda i: archive[i][1])
                    # don't drop best
                    if archive[worst][1] > best_val + 1e-12:
                        archive.pop(worst)
            elif len(archive) < 6:
                # force diversity even if close
                # check with looser thresh
                nov2, _ = is_novel(xp, thresh=0.02)
                if nov2:
                    archive.append((xp.copy(), vp))

    # Phase 2: geometric crossover + anneal + polish
    it = 0
    while time.time() - t0 < BUDGET - 22:
        it += 1
        if len(archive) >= 2:
            ai, bi = rng.choice(len(archive), size=2, replace=False)
            A = archive[ai][0]; B = archive[bi][0]
        else:
            A = best_x; B = best_x + rng.normal(size=best_x.shape) * 0.2
            B /= np.linalg.norm(B, axis=1, keepdims=True)
        try:
            child0 = _crossover_child(A, B, rng)
        except Exception:
            continue
        # anneal
        steps = int(rng.choice([1200, 1800, 2500]))
        a0 = float(rng.uniform(15, 60)); a1 = float(rng.uniform(140, 330))
        lr0 = float(np.exp(rng.uniform(np.log(0.02), np.log(0.12))))
        lr1 = float(np.exp(rng.uniform(np.log(0.001), np.log(0.015))))
        mu = float(rng.uniform(0.75, 0.92))
        if time.time() - t0 > BUDGET - 24:
            break
        xq = _soft_run(child0, steps, a0, a1, lr0, lr1, mu, rng)
        vq = _max_cos(xq)
        # only LP-polish if promising or occasionally for diversity
        do_polish = False
        if vq < best_val + 0.002:
            do_polish = True
        elif rng.random() < 0.10:
            do_polish = True
        if do_polish:
            xp, vp = _lp_polish(xq, 1.2, t_end=t0 + BUDGET - 20)
        else:
            xp, vp = xq, vq
        if vp < best_val:
            best_x, best_val = xp.copy(), vp
            archive.append((xp.copy(), vp))
            if len(archive) > 30:
                # remove worst that is not best
                vals = [v for (_, v) in archive]
                worst = int(np.argmax(vals))
                if vals[worst] > best_val + 1e-12:
                    archive.pop(worst)
        else:
            # accept structurally novel even when 1e-7..1e-5 worse (and up to 1e-3 for exploration)
            if vp < best_val + 1e-3:
                try:
                    nov, _ = is_novel(xp, thresh=0.05)
                except Exception:
                    nov = False
                if nov:
                    archive.append((xp.copy(), vp))
                    if len(archive) > 30:
                        vals = [v for (_, v) in archive]
                        worst = int(np.argmax(vals))
                        archive.pop(worst)
        # occasional large-kick injection to keep archive fresh
        if it % 9 == 0 and time.time() - t0 < BUDGET - 40:
            sigma = float(np.exp(rng.uniform(np.log(0.05), np.log(0.3))))
            x0 = best_x + rng.normal(size=best_x.shape) * sigma
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            xq2 = _soft_run(x0, 1000, 30, 200, 0.06, 0.005, 0.85, rng)
            xp2, vp2 = _lp_polish(xq2, 1.0, t_end=t0 + BUDGET - 20)
            if vp2 < best_val:
                best_x, best_val = xp2.copy(), vp2
                archive.append((xp2.copy(), vp2))
            elif vp2 < best_val + 1e-3:
                try:
                    nov, _ = is_novel(xp2, thresh=0.05)
                except Exception:
                    nov = False
                if nov and len(archive) < 30:
                    archive.append((xp2.copy(), vp2))

    remaining = BUDGET - (time.time() - t0) - 1.0
    if remaining > 2:
        xp, vp = _lp_polish(best_x, min(remaining, 20.0))
        if vp < best_val:
            best_x, best_val = xp, vp
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
