import time
import numpy as np
from scipy.optimize import linprog, linear_sum_assignment
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _adam_quench(X, steps, beta0, beta1, lr0, lr1, rownorm, rng=None):
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    lb0 = np.log(beta0); lb1 = np.log(beta1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    for t in range(1, steps + 1):
        frac = t / steps
        beta = np.exp(lb0 + (lb1 - lb0) * frac)
        lr = np.exp(ll0 + (ll1 - ll0) * frac)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        gmax = G.max()
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        if rownorm:
            S = W.sum(axis=1, keepdims=True) + 1e-12
            Grad = (W @ X) / S
        else:
            Z = W.sum()
            if Z == 0:
                continue
            Grad = (W @ X) / Z
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        m *= b1; m += (1.0 - b1) * Gt
        v *= b2; v += (1.0 - b2) * (Gt * Gt)
        bc1 = 1.0 - b1 ** t; bc2 = 1.0 - b2 ** t
        mhat = m / bc1; vhat = v / bc2
        step = mhat / (np.sqrt(vhat) + eps)
        X = X - lr * step
        nrm = np.sqrt((X * X).sum(axis=1, keepdims=True))
        nrm[nrm == 0] = 1.0
        X /= nrm
    return X

def _hard_polish(X, steps=300, lr0=0.0035, lr1=0.00012, tol0=3e-4, tol1=2e-6):
    n, d = X.shape
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        tol = tol0 + (tol1 - tol0) * frac
        lr = lr0 * (lr1 / lr0) ** frac if lr0 > 0 and lr1 > 0 else lr0
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        M = G > (m - tol)
        cnt = M.sum(axis=1, keepdims=True) + 1e-12
        Grad = (M.astype(np.float64) @ X) / cnt
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        nn = np.sqrt((Gt * Gt).sum(axis=1, keepdims=True)) + 1e-12
        active = (cnt[:, 0] > 0.5)
        step = np.zeros_like(Gt)
        step[active] = Gt[active] / nn[active]
        X = X - lr * step
        nrm = np.sqrt((X * X).sum(axis=1, keepdims=True))
        nrm[nrm == 0] = 1.0
        X /= nrm
    return X

def _lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        vv = _max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _align_perm(A, B, iters=3):
    # A,B n x d unit rows. Find R in O(d) and perm to minimize RMSD.
    n, d = A.shape
    R = np.eye(d)
    perm = np.arange(n)
    for _ in range(iters):
        Brot = B @ R
        S = A @ Brot.T  # n x n similarity
        _, col = linear_sum_assignment(-S)
        # col[j] ? linear_sum_assignment returns row, col; rows sorted
        # S rows = A index, cols = Brot index, so perm maps A_i -> B_{perm[i]}
        # row_ind is 0..n-1, col_ind gives assignment
        perm = col
        Bp = B[perm, :]
        M = Bp.T @ A  # d x d
        U, _, Vt = np.linalg.svd(M)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
    Brot = B @ R
    Bp_rot = Brot[perm, :]
    diff = A - Bp_rot
    rmsd = float(np.sqrt((diff * diff).sum() / n))
    return rmsd, R, perm

def _contact_hash(X, tol=1e-7):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = G.max()
    C = G > (m - tol)
    deg = C.sum(axis=1).astype(int)
    nc = int(C.sum() // 2)
    return nc, tuple(sorted(deg.tolist())), m

def solve():
    t0 = time.time()
    BUDGET = 175.0
    n, d = 168, 8
    rng = np.random.RandomState(777)
    best = None; best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape != (n, d):
                continue
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            if not np.all(np.isfinite(a)):
                continue
            va = _max_cos(a)
            if va < best_val:
                best_val = va; best = a.copy()
        except Exception:
            pass
    if best is None:
        best = rng.randn(n, d)
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)

    # diversity archive: list of dicts {X, val}
    archive = [{'X': best.copy(), 'val': best_val}]

    def is_novel(Xc, thresh=0.05):
        for e in archive:
            try:
                rmsd, _, _ = _align_perm(Xc, e['X'], iters=2)
            except Exception:
                continue
            if rmsd < thresh:
                return False
        return True

    def rmsd_to_all(Xc):
        mn = 1e9
        for e in archive:
            try:
                rmsd, _, _ = _align_perm(Xc, e['X'], iters=2)
                mn = min(mn, rmsd)
            except Exception:
                pass
        return mn

    def random_kick_quench(src):
        X = src.copy()
        u = rng.rand()
        if u < 0.5:
            sigma = [0.06, 0.09, 0.13, 0.18][rng.randint(4)]
            X = X + rng.randn(n, d) * sigma
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        else:
            # block void replacement of 4..12 points
            k = 4 + int(rng.randint(9))
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            rm = G.max(axis=1)
            # pick worst + random jammed mix
            idx_worst = np.argsort(rm)[-k:]
            # randomize half of them
            for j in idx_worst:
                if rng.rand() < 0.6:
                    v = rng.randn(d); v /= np.linalg.norm(v)
                    # farthest-ish: try few random and keep most distant
                    cand = rng.randn(8, d)
                    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                    # score by max dot to rest
                    rest = np.delete(X, j, axis=0)
                    best_c = cand[0]
                    best_s = 1e9
                    for c in cand:
                        s = float((rest @ c).max())
                        if s < best_s:
                            best_s = s; best_c = c
                    X[j] = best_c
                else:
                    X[j] = X[j] + rng.randn(d) * 0.3
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            X = X + rng.randn(n, d) * 0.008
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        beta0 = 20.0 + rng.rand() * 40.0
        beta1 = 220.0 + rng.rand() * 380.0
        steps = 550 + int(rng.randint(500))
        lr0 = 0.003 + rng.rand() * 0.005
        lr1 = lr0 * (0.06 + rng.rand() * 0.1)
        rownorm = (rng.rand() < 0.6)
        Xq = _adam_quench(X, steps, beta0, beta1, lr0, lr1, rownorm)
        Xq = _hard_polish(Xq, steps=250, lr0=0.0032, lr1=0.00012)
        return Xq

    # Phase A: build archive of LP-ish polished optima from large kicks
    t_phaseA = t0 + 55.0
    target_archive = 14
    tries = 0
    while time.time() < t_phaseA and len(archive) < target_archive and time.time() < t0 + BUDGET - 30:
        tries += 1
        if tries > 60:
            break
        src = best.copy()
        # occasionally kick from non-best archive member to diversify
        if len(archive) > 1 and rng.rand() < 0.35:
            src = archive[rng.randint(len(archive))]['X']
        Xq = random_kick_quench(src)
        vq = _max_cos(Xq)
        if not np.all(np.isfinite(Xq)):
            continue
        if vq < best_val - 1e-12:
            best = Xq.copy(); best_val = vq
            # best always added (even if close, keep as separate)
            archive.append({'X': Xq.copy(), 'val': vq})
            # keep sorted, cap size 22
            archive = sorted(archive, key=lambda e: e['val'])
            if len(archive) > 22:
                # drop worst that is not best
                archive = archive[:22]
        elif vq < best_val + 2e-5:
            # novelty check RMSD > 0.05
            if is_novel(Xq, thresh=0.05):
                archive.append({'X': Xq.copy(), 'val': vq})
                archive = sorted(archive, key=lambda e: e['val'])
                if len(archive) > 22:
                    archive = archive[:22]
        # if too strict and archive small, relax
        if tries % 12 == 0 and len(archive) < 5:
            # accept next with looser threshold
            if vq < best_val + 5e-5 and rmsd_to_all(Xq) > 0.03:
                archive.append({'X': Xq.copy(), 'val': vq})

    def crossover_child(PA, PB):
        # align PB to PA
        try:
            rmsd, R, perm = _align_perm(PA, PB, iters=3)
        except Exception:
            R = np.eye(d); perm = np.arange(n)
        PBm = (PB @ R)[perm, :]  # matched & rotated, PBm[i] <-> PA[i]
        # random cap / half-space mask on PA
        u = rng.randn(d); u /= np.linalg.norm(u)
        proj = PA @ u
        mode = rng.rand()
        mask = np.zeros(n, dtype=bool)
        if mode < 0.5:
            # half-space with random offset near median
            q = np.median(proj) + rng.randn() * 0.15
            mask = proj > q
            # ensure not degenerate
            if mask.sum() < 20 or mask.sum() > n - 20:
                mask = proj > np.median(proj)
        else:
            # spherical cap: take top-k from PA
            k = 40 + int(rng.randint(89))  # 40..128
            order = np.argsort(proj)
            mask = np.zeros(n, dtype=bool)
            mask[order[-k:]] = True
            # randomly flip which parent is "cap"
            if rng.rand() < 0.5:
                mask = ~mask
        C = np.empty_like(PA)
        C[mask] = PA[mask]
        C[~mask] = PBm[~mask]
        # delete near-duplicates cos>0.9 and refill by farthest-point probing
        G = C @ C.T
        np.fill_diagonal(G, -2.0)
        # find pairs with very high cosine (should be rare across seam)
        # instead check each point's max to others
        dups = set()
        iu0 = np.triu_indices(n, 1)
        # quick detect: any pair >0.9
        high = np.where(G[iu0] > 0.9)[0]
        if len(high) > 0:
            for hh in high:
                i = int(iu0[0][hh]); j = int(iu0[1][hh])
                # mark j for replacement (keep one)
                dups.add(j)
        # also if child has internal clash > current best, repair worst few
        # repair dups
        if len(dups) > 0:
            for j in list(dups):
                # farthest probe vs all other points
                P = rng.randn(3000, d)
                P /= np.linalg.norm(P, axis=1, keepdims=True)
                rest = np.delete(C, j, axis=0)
                D = P @ rest.T
                mx = D.max(axis=1)
                bi = int(np.argmin(mx))
                C[j] = P[bi]
        # small jitter to seam points to break exact ties
        # seam = points whose nearest opposite-parent neighbor is close
        # approximate: jitter all slightly
        C = C + rng.randn(n, d) * 0.004
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        return C

    # Phase B: geometric crossover rounds using archive members as parents
    t_phaseB = t0 + 135.0
    it = 0
    while time.time() < t_phaseB and time.time() < t0 + BUDGET - 28:
        it += 1
        if len(archive) < 2:
            # need at least 2 parents; generate more
            Xq = random_kick_quench(best)
            vq = _max_cos(Xq)
            if np.all(np.isfinite(Xq)) and vq < best_val + 2e-5 and is_novel(Xq, 0.05):
                archive.append({'X': Xq.copy(), 'val': vq})
            continue
        # pick two distinct parents uniformly (archive members, not only incumbent)
        a_idx = rng.randint(len(archive))
        b_idx = rng.randint(len(archive))
        if a_idx == b_idx:
            continue
        PA = archive[a_idx]['X']
        PB = archive[b_idx]['X']
        C0 = crossover_child(PA, PB)
        # anneal + polish
        beta0 = 25.0 + rng.rand() * 35.0
        beta1 = 250.0 + rng.rand() * 350.0
        steps = 500 + int(rng.randint(500))
        lr0 = 0.003 + rng.rand() * 0.005
        lr1 = lr0 * (0.06 + rng.rand() * 0.1)
        rownorm = (rng.rand() < 0.6)
        Cq = _adam_quench(C0, steps, beta0, beta1, lr0, lr1, rownorm)
        Cq = _hard_polish(Cq, steps=250, lr0=0.0032, lr1=0.00012)
        vq = _max_cos(Cq)
        if not np.all(np.isfinite(Cq)):
            continue
        if vq < best_val - 1e-12:
            best = Cq.copy(); best_val = vq
            archive.append({'X': Cq.copy(), 'val': vq})
            archive = sorted(archive, key=lambda e: e['val'])
            if len(archive) > 22:
                archive = archive[:22]
        elif vq < best_val + 1.2e-5:
            # accept if structurally novel even when worse
            if is_novel(Cq, thresh=0.05):
                archive.append({'X': Cq.copy(), 'val': vq})
                archive = sorted(archive, key=lambda e: e['val'])
                if len(archive) > 22:
                    # keep best + diverse: drop worst
                    archive = archive[:22]
        else:
            # occasionally keep very novel but slightly worse (up to 3e-5) with prob
            if vq < best_val + 3e-5 and rng.rand() < 0.15 and is_novel(Cq, thresh=0.07):
                archive.append({'X': Cq.copy(), 'val': vq})
                archive = sorted(archive, key=lambda e: e['val'])
                if len(archive) > 22:
                    archive = archive[:22]
        # every ~15 iters, also try pure kick from random archive member to keep diversity flowing
        if it % 15 == 0 and time.time() < t_phaseB - 5:
            src = archive[rng.randint(len(archive))]['X']
            Xq = random_kick_quench(src)
            vq2 = _max_cos(Xq)
            if np.all(np.isfinite(Xq)):
                if vq2 < best_val - 1e-12:
                    best = Xq.copy(); best_val = vq2
                    archive.append({'X': Xq.copy(), 'val': vq2})
                elif vq2 < best_val + 1.2e-5 and is_novel(Xq, 0.05):
                    archive.append({'X': Xq.copy(), 'val': vq2})
                archive = sorted(archive, key=lambda e: e['val'])
                if len(archive) > 22:
                    archive = archive[:22]

    # Final LP polish on incumbent
    remaining = (t0 + BUDGET) - time.time()
    if remaining > 8.0:
        secs = min(remaining - 2.0, 38.0)
        try:
            polished = _lp_polish(best, secs)
            vp = _max_cos(polished)
            if vp < best_val and np.all(np.isfinite(polished)):
                best = polished; best_val = vp
        except Exception:
            pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    bad = ~np.all(np.isfinite(best), axis=1)
    if np.any(bad):
        rr = np.random.RandomState(1).randn(bad.sum(), d)
        rr /= np.linalg.norm(rr, axis=1, keepdims=True)
        best[bad] = rr
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
