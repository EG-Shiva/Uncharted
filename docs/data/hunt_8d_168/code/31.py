import time
import numpy as np
from scipy.optimize import linprog, linear_sum_assignment
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lp_polish(x, seconds, t_end_limit=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while True:
        if time.time() - start > seconds:
            break
        if t_end_limit is not None and time.time() > t_end_limit:
            break
        if radius < 1e-14:
            break
        Gfull = x @ x.T
        iu_c = Gfull[iu]
        m = iu_c.max()
        sel = iu_c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        if p > 3000:
            idx = np.argsort(iu_c[sel])[-3000:]
            I, J = I[idx], J[idx]
            p = 3000
            cc = iu_c[sel][idx]
        else:
            cc = iu_c[sel]
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
        lr = lr0 * (lr1 / lr0) ** frac if lr0 > 0 and lr1 > 0 else lr0
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
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x /= nr
    return x

def _align_match(A, B, iters=3):
    # align B to A (both n x d unit rows), return B_aligned permuted to A order + rmsd
    n, d = A.shape
    Bc = B.copy()
    for _ in range(iters):
        C = A @ Bc.T
        _, col = linear_sum_assignment(-C)
        Bp = Bc[col]
        H = Bp.T @ A
        try:
            U, S, Vt = np.linalg.svd(H)
        except Exception:
            break
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt = Vt.copy(); Vt[-1, :] *= -1
            R = Vt.T @ U.T
        Bc = Bp @ R
    diff = A - Bc
    rmsd = float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))
    return Bc, rmsd

def _rmsd(A, B):
    _, r = _align_match(A, B, iters=3)
    return r

def _contact_hash_deg(x, tol=1e-9):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = G.max()
    deg = (G > m - tol).sum(axis=1)
    return tuple(sorted(deg.tolist())), float(m), deg

def _crossover_child(A, B, rng):
    n, d = A.shape
    Bal, _ = _align_match(A, B, iters=3)
    u = rng.normal(size=d); u /= np.linalg.norm(u)
    mode = rng.random()
    if mode < 0.5:
        # spherical cap with random fraction
        f = float(rng.uniform(0.3, 0.7))
        da = A @ u
        thr = np.quantile(da, f)
        # randomly take inside from A or B
        if rng.random() < 0.5:
            mask = da > thr
        else:
            mask = da <= thr
    else:
        da = A @ u
        if rng.random() < 0.5:
            mask = da > 0
        else:
            mask = da <= 0
        # random shift for unbalanced halves
        if rng.random() < 0.3:
            thr = float(rng.uniform(-0.3, 0.3))
            mask = da > thr
    child = np.where(mask[:, None], A, Bal)
    # delete near-duplicates cosine>0.9 greedily
    keep = []
    bad = np.zeros(n, dtype=bool)
    cur_kept = []
    order = rng.permutation(n)
    kept_mat = []
    for idx in order:
        pt = child[idx]
        if len(kept_mat) == 0:
            kept_mat.append(pt); keep.append(idx)
        else:
            K = np.stack(kept_mat)
            md = float((K @ pt).max())
            if md > 0.9:
                bad[idx] = True
            else:
                kept_mat.append(pt); keep.append(idx)
    nbad = int(bad.sum())
    if nbad > 0:
        good_idx = np.where(~bad)[0]
        cur = child[good_idx].copy()
        # farthest-point probing refill
        new_pts = []
        for _ in range(nbad):
            cand = rng.normal(size=(400, d))
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            D = cand @ cur.T
            md = D.max(axis=1)
            bi = int(np.argmin(md))
            bp = cand[bi]
            new_pts.append(bp)
            cur = np.vstack([cur, bp.reshape(1, -1)])
        child = cur  # (n,d) reordered but fine
        # restore order randomness
        child = child[rng.permutation(n)]
    # small jitter
    child = child + rng.normal(size=child.shape) * 1e-3
    child /= np.linalg.norm(child, axis=1, keepdims=True)
    return child

def _greedy_fill(x_keep, k, rng, n_try=400):
    d = x_keep.shape[1]
    cur = x_keep.copy()
    new_pts = []
    for _ in range(k):
        cand = rng.normal(size=(n_try, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        dots = cand @ cur.T
        md = dots.max(axis=1)
        bi = int(np.argmin(md))
        bp = cand[bi]
        new_pts.append(bp)
        cur = np.vstack([cur, bp.reshape(1, -1)])
    return cur

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
        cands.append(rec0 / np.linalg.norm(rec0, axis=1, keepdims=True))
    if not cands:
        c = rng.normal(size=(n, d))
        c /= np.linalg.norm(c, axis=1, keepdims=True)
        cands.append(c)
    best_x = min(cands, key=_max_cos)
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    best_val = _max_cos(best_x)
    # baseline tighten
    if time.time() - t0 < BUDGET - 60:
        bx, bv = _lp_polish(best_x, 1.5, t_end_limit=t0 + BUDGET - 55)
        if bv < best_val:
            best_x, best_val = bx, bv
    archive = []
    archive_vals = []
    def try_add(x, v):
        # returns True if added/replaced
        for k, ax in enumerate(archive):
            r = _rmsd(x, ax)
            if r < 0.05:
                if v < archive_vals[k] - 1e-12:
                    archive[k] = x.copy(); archive_vals[k] = v
                    return True
                return False
        archive.append(x.copy()); archive_vals.append(v)
        return True
    try_add(best_x, best_val)
    # Phase 1: generate 15-30 LP-polished optima from large kicks
    n_init_target = 14
    it_c = 0
    while len(archive) < n_init_target and time.time() - t0 < BUDGET - 90:
        it_c += 1
        base = best_x
        r = rng.random()
        if r < 0.5:
            sigma = float(np.exp(rng.uniform(np.log(0.08), np.log(0.5))))
            x0 = base + rng.normal(size=base.shape) * sigma
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        elif r < 0.8:
            ksub = int(rng.integers(10, 50))
            idx = rng.choice(n, size=ksub, replace=False)
            sigma = float(np.exp(rng.uniform(np.log(0.05), np.log(0.5))))
            x0 = base.copy()
            x0[idx] += rng.normal(size=(ksub, d)) * sigma
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        else:
            # kill-and-fill random
            krep = int(rng.integers(4, 14))
            kill = rng.choice(n, size=krep, replace=False)
            msk = np.ones(n, dtype=bool); msk[kill] = False
            x0 = _greedy_fill(base[msk], krep, rng)
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        steps = int(rng.choice([1200, 1800, 2500]))
        a0 = float(rng.uniform(15, 60)); a1 = float(rng.uniform(120, 350))
        if a1 < a0: a0, a1 = a1, a0
        lr0 = float(np.exp(rng.uniform(np.log(0.02), np.log(0.15))))
        lr1 = float(np.exp(rng.uniform(np.log(0.001), np.log(0.02))))
        mu = float(rng.uniform(0.75, 0.92))
        if time.time() - t0 > BUDGET - 95:
            break
        xopt = _soft_run(x0, steps, a0, a1, lr0, lr1, mu, rng)
        vsoft = _max_cos(xopt)
        # polish if promising or periodically for diversity
        do_polish = (vsoft < best_val + 0.01) or (it_c % 2 == 0)
        if do_polish:
            xp, vp = _lp_polish(xopt, 1.2, t_end_limit=t0 + BUDGET - 88)
        else:
            xp, vp = xopt, vsoft
        if vp < best_val:
            best_x, best_val = xp.copy(), vp
        # accept into archive even if worse (novelty decides)
        try_add(xp, vp)
        # keep looping even if archive full of similar; force diversity by bigger kicks
        if it_c > 30:
            break
    # Phase 2: geometric crossover rounds using archive members as parents
    nrounds = 0
    while time.time() - t0 < BUDGET - 22:
        nrounds += 1
        if len(archive) < 2:
            # fallback large kick from best
            A = best_x; B = best_x + rng.normal(size=best_x.shape) * 0.3
            B /= np.linalg.norm(B, axis=1, keepdims=True)
        else:
            ia, ib = rng.choice(len(archive), size=2, replace=False)
            A = archive[ia]; B = archive[ib]
            # occasionally use incumbent as one parent
            if rng.random() < 0.35:
                A = best_x
            if rng.random() < 0.15:
                # mutate one parent with kick before crossover for extra diversity
                sigma = float(np.exp(rng.uniform(np.log(0.02), np.log(0.3))))
                B = B + rng.normal(size=B.shape) * sigma
                B /= np.linalg.norm(B, axis=1, keepdims=True)
        try:
            child0 = _crossover_child(A, B, rng)
        except Exception:
            continue
        steps = int(rng.choice([1000, 1500, 2200]))
        a0 = float(rng.uniform(20, 70)); a1 = float(rng.uniform(150, 400))
        if a1 < a0: a0, a1 = a1, a0
        lr0 = float(np.exp(rng.uniform(np.log(0.02), np.log(0.12))))
        lr1 = float(np.exp(rng.uniform(np.log(0.001), np.log(0.015))))
        mu = float(rng.uniform(0.78, 0.92))
        if time.time() - t0 > BUDGET - 26:
            break
        xc = _soft_run(child0, steps, a0, a1, lr0, lr1, mu, rng)
        vc = _max_cos(xc)
        # polish if competitive or novel-candidate
        thresh = best_val + 0.004
        if vc < thresh or (nrounds % 4 == 0):
            xp, vp = _lp_polish(xc, 1.2, t_end_limit=t0 + BUDGET - 20)
        else:
            xp, vp = xc, vc
        if vp < best_val:
            best_x, best_val = xp.copy(), vp
            try_add(xp, vp)
        else:
            # accept into archive if structurally novel even when 1e-7..1e-5 worse (or more)
            # novelty check inside try_add (RMSD>0.05)
            if vp < best_val + 0.01:
                try_add(xp, vp)
        # cap archive size to avoid O(n^2) rmsd cost
        if len(archive) > 30:
            # drop worst non-best entry
            worst = int(np.argmax(archive_vals))
            # don't drop best
            if archive_vals[worst] > best_val + 1e-12:
                del archive[worst]; del archive_vals[worst]
            else:
                # find second worst
                order = np.argsort(archive_vals)
                del archive[order[-1]]; del archive_vals[order[-1]]
    remaining = BUDGET - (time.time() - t0) - 1.0
    if remaining > 2:
        xp, vp = _lp_polish(best_x, min(remaining, 20.0))
        if vp < best_val:
            best_x, best_val = xp, vp
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
