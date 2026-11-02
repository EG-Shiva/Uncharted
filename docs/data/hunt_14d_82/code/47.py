import time
import numpy as np
from scipy.optimize import linprog, linear_sum_assignment
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _jam_stats(X):
    G = X @ X.T
    iu = np.triu_indices(X.shape[0], 1)
    c = G[iu]
    m = float(np.max(c))
    sel = c > (m - 1e-3)
    ncon = int(np.count_nonzero(sel))
    deg = np.zeros(X.shape[0], dtype=int)
    I = iu[0][sel]; J = iu[1][sel]
    np.add.at(deg, I, 1)
    np.add.at(deg, J, 1)
    nratt = int(np.count_nonzero(deg == 0))
    deg_sorted = np.sort(deg)[::-1].astype(float)
    # spectrum: top eigenvalues of Gram
    try:
        ev = np.linalg.eigvalsh(G)
        ev = np.sort(ev)[::-1]
    except Exception:
        ev = np.zeros(X.shape[0])
    return m, ncon, nratt, deg_sorted, ev

def _quench(X0, stages, steps_per_stage, lr0, lr1, mu, t_deadline, rng=None):
    X = _norm(X0.copy())
    n, d = X.shape
    V = np.zeros((n, d))
    for si, (T, p) in enumerate(stages):
        # linear lr decay within stage
        S = steps_per_stage[si] if si < len(steps_per_stage) else steps_per_stage[-1]
        for s in range(S):
            if (s & 63) == 0 and time.time() >= t_deadline:
                return X
            frac = s / max(1, S - 1)
            lr = lr0[si] + frac * (lr1[si] - lr0[si]) if isinstance(lr0, list) else lr0 + frac * (lr1 - lr0)
            G = X @ X.T
            # hinge residual
            # compute M = G - T clipped
            # avoid diag
            np.fill_diagonal(G, -10.0)
            M = G - T
            # quick exit if none violate
            if np.max(M) <= 0:
                V *= 0.5
                # small random tap to keep moving? no
                continue
            np.maximum(M, 0.0, out=M)
            # W = p * M^{p-1}
            if p == 2:
                W = M * 2.0
            else:
                # clip to avoid overflow: M<=~1.2
                W = np.power(M, p - 1.0) * float(p)
            np.fill_diagonal(W, 0.0)
            Grad = W @ X
            dot = np.sum(Grad * X, axis=1, keepdims=True)
            Grad -= dot * X
            rms = float(np.sqrt(np.mean(Grad * Grad)))
            if not np.isfinite(rms) or rms < 1e-15:
                continue
            Dir = Grad / rms
            # heavy ball
            V = mu * V + Dir
            # re-project velocity to tangent
            vdot = np.sum(V * X, axis=1, keepdims=True)
            V -= vdot * X
            vrms = float(np.sqrt(np.mean(V * V)))
            if not np.isfinite(vrms) or vrms < 1e-15:
                V[:] = Dir
                vrms = 1.0
            # scale step so that rms step = lr
            step = V * (lr / (vrms + 1e-12))
            # cap per-point move
            # (avoid blow up)
            mx = float(np.max(np.linalg.norm(step, axis=1)))
            if mx > 0.08:
                step *= (0.08 / mx)
            X = X - step
            # renormalize
            rn = np.linalg.norm(X, axis=1, keepdims=True)
            rn[rn == 0] = 1.0
            X /= rn
            # occasional tiny noise to escape symmetric saddle
            if rng is not None and (s % 400 == 399):
                X = _norm(X + 1e-4 * rng.standard_normal((n, d)))
    return _norm(X)

def _lp_polish(X, t_deadline, radius_init=8e-5, cap=1500, max_it=25):
    X = _norm(X.copy())
    n, d = X.shape
    best = _max_cos(X)
    radius = float(radius_init)
    iu0 = np.triu_indices(n, 1)
    it = 0
    while it < max_it and time.time() < t_deadline and radius > 1e-13:
        it += 1
        G = X @ X.T
        c_all = G[iu0]
        m = float(np.max(c_all))
        thr = m - max(20.0 * radius, 2e-7)
        sel = c_all > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > cap:
            idx = np.where(sel)[0]
            vals = c_all[idx]
            part = np.argpartition(-vals, cap - 1)[:cap]
            keep = np.zeros_like(sel, dtype=bool)
            keep[idx[part]] = True
            sel = keep
        I = iu0[0][sel]; J = iu0[1][sel]
        c = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": 1.5})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        Y = X + res.x[:-1].reshape(n, d)
        Y = _norm(Y)
        # time check
        v = _max_cos(Y)
        if v < best:
            X, best = Y, v
            radius = min(radius * 1.5, 5e-3)
        else:
            radius *= 0.5
    return X

def _greedy_seed(rng, n=82, d=14, cand=1500):
    Xlist = []
    # first point
    x = rng.standard_normal(d); x /= np.linalg.norm(x)
    Xlist.append(x)
    Xmat = np.array(Xlist)
    for k in range(1, n):
        C = _norm(rng.standard_normal((cand, d)))
        # max dot to existing
        D = C @ Xmat.T
        md = np.max(D, axis=1)
        bi = int(np.argmin(md))
        Xlist.append(C[bi])
        Xmat = np.array(Xlist)
    return np.array(Xlist)

def _lattice_seed(rng, n=82, d=14, noise=0.12):
    # stacked random orthonormal frames + sign
    pts = []
    while len(pts) < n:
        A = rng.standard_normal((d, d))
        Q, _ = np.linalg.qr(A)
        for k in range(d):
            pts.append(Q[:, k])
            if len(pts) >= n: break
            pts.append(-Q[:, k])
            if len(pts) >= n: break
    pts = np.array(pts[:n])
    pts = pts + noise * rng.standard_normal(pts.shape)
    return _norm(pts)

def solve():
    t0 = time.time()
    HARD = t0 + 135.0
    LP_START = HARD - 20.0
    n, d = 82, 14
    # fallback
    fb = None; fbv = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = _norm(a)
                if np.all(np.isfinite(a)):
                    v = _max_cos(a)
                    if v < fbv:
                        fbv = v; fb = a.copy()
        except Exception:
            pass
    if fb is None:
        rng0 = np.random.default_rng(999)
        fb = _norm(rng0.standard_normal((n, d)))
        fbv = _max_cos(fb)
    Xbest = fb.copy(); bestv = fbv
    rng = np.random.default_rng(12345)

    stages = [(0.205, 2), (0.200, 3), (0.197, 4), (0.194, 6)]
    steps_short = [700, 700, 800, 900]
    lr0s = [0.020, 0.014, 0.010, 0.008]
    lr1s = [0.006, 0.004, 0.003, 0.002]

    seeds = []
    # 4 pure random
    for s in range(4):
        r = np.random.default_rng(100 + s)
        seeds.append(_norm(r.standard_normal((n, d))))
        if time.time() >= HARD - 30: break
    # 3 greedy far-hole
    for s in range(3):
        if time.time() >= HARD - 30: break
        r = np.random.default_rng(200 + s)
        try:
            seeds.append(_greedy_seed(r, n, d, cand=1200))
        except Exception:
            seeds.append(_norm(r.standard_normal((n, d))))
    # 3 lattice-like
    for s in range(3):
        r = np.random.default_rng(300 + s)
        seeds.append(_lattice_seed(r, n, d, noise=0.08 + 0.05 * s))
    # 2 signed-cube greedy: sample from +-1 cube
    for s in range(2):
        r = np.random.default_rng(400 + s)
        # random subset of hypercube vertices, greedy far
        pool = np.sign(r.standard_normal((6000, d)))
        pool[pool == 0] = 1.0
        pool = _norm(pool)
        # greedy pick 82 from pool
        try:
            perm = r.permutation(len(pool))
            chosen = [0]
            # start random
            start = int(perm[0]) % len(pool)
            sel_idx = [start]
            sel_mat = pool[start:start+1]
            avail = pool
            # iterative: sample subset for speed
            for k in range(1, n):
                if time.time() >= HARD - 25: break
                sub = avail[r.choice(len(avail), size=min(1500, len(avail)), replace=False)]
                D = sub @ sel_mat.T
                md = np.max(D, axis=1)
                bi = int(np.argmin(md))
                # find actual index (approx) - just take sub vector
                sel_mat = np.vstack([sel_mat, sub[bi:bi+1]])
            seeds.append(_norm(sel_mat[:n] + 0.02 * r.standard_normal((n, d))))
        except Exception:
            seeds.append(_norm(r.standard_normal((n, d))))

    archive = []  # list of dicts
    def fp_dist(fp1, fp2):
        return float(np.linalg.norm(fp1 - fp2))

    def try_add_archive(X):
        m, ncon, nratt, deg_s, ev = _jam_stats(X)
        # fingerprint: normalized spectrum top20 + deg sorted normalized
        fp = np.concatenate([ev[:20] / 82.0, deg_s / 82.0])
        for a in archive:
            if fp_dist(fp, a['fp']) < 0.08:
                # same basin: keep better jam
                if ncon > a['ncon'] or (ncon == a['ncon'] and m < a['m']):
                    a['X'] = X.copy(); a['m'] = m; a['ncon'] = ncon
                    a['nratt'] = nratt; a['fp'] = fp
                return False
        archive.append({'X': X.copy(), 'm': m, 'ncon': ncon, 'nratt': nratt, 'fp': fp})
        # cap size
        if len(archive) > 8:
            # drop least jammed
            archive.sort(key=lambda a: (-a['ncon'], a['nratt'], a['m']))
            del archive[-1]
        return True

    # quench seeds, short 2-5s each but our steps are ~1s; enforce per-seed deadline
    for si, s0 in enumerate(seeds):
        if time.time() >= LP_START - 12.0:
            break
        per_deadline = min(time.time() + 4.0, LP_START - 12.0 + (12.0 / max(1, len(seeds))))
        # give at least 0.8s
        if per_deadline - time.time() < 0.5:
            break
        Xq = _quench(s0, stages, steps_short, lr0s, lr1s, 0.9, per_deadline, rng)
        m = _max_cos(Xq)
        if m < bestv:
            bestv = float(m); Xbest = Xq.copy()
        # jamming promotion even if max slightly worse (within 0.01)
        _, ncon, nratt, _, _ = _jam_stats(Xq)
        # require some rigidity signal: hundreds of near contacts
        # still archive all diverse ones to allow crossover
        try_add_archive(Xq)
        # quick extra micro-quench at low T for jammed ones if time allows
        if ncon > 250 and time.time() < per_deadline - 0.3:
            Xq2 = _quench(Xq, [(0.194, 6)], [400], [0.004], [0.001], 0.9, per_deadline, None)
            m2 = _max_cos(Xq2)
            if m2 < bestv:
                bestv = float(m2); Xbest = Xq2.copy()
            try_add_archive(Xq2)

    # crossover archive members with different fingerprints
    if len(archive) >= 2 and time.time() < LP_START - 4.0:
        # sort by jam
        archive.sort(key=lambda a: (-a['ncon'], a['nratt'], a['m']))
        # pairwise distances
        pairs = []
        for i in range(len(archive)):
            for j in range(i+1, len(archive)):
                dd = fp_dist(archive[i]['fp'], archive[j]['fp'])
                pairs.append((dd, i, j))
        pairs.sort(reverse=True)
        nchild = min(6, len(pairs))
        for ci in range(nchild):
            if time.time() >= LP_START - 1.0:
                break
            _, ii, jj = pairs[ci]
            A = archive[ii]['X']; B = archive[jj]['X']
            try:
                Mmat = B.T @ A
                U, _, Vt = np.linalg.svd(Mmat, full_matrices=False)
                R = U @ Vt
                Bb = B @ R
                # point matching
                Ddot = A @ Bb.T
                ri, cj = linear_sum_assignment(-Ddot)
                # reorder Bb to A order
                inv = np.empty(n, dtype=int)
                inv[ri] = cj
                Br = Bb[inv]
                mask = rng.random(n) < 0.5
                Ch = A.copy()
                Ch[mask] = Br[mask]
                Ch = _norm(Ch + 0.004 * rng.standard_normal((n, d)))
            except Exception:
                continue
            per_deadline = min(time.time() + 2.5, LP_START)
            Xc = _quench(Ch, stages[-2:], steps_short[-2:], lr0s[-2:], lr1s[-2:], 0.9, per_deadline, rng)
            m = _max_cos(Xc)
            if m < bestv:
                bestv = float(m); Xbest = Xc.copy()
            try_add_archive(Xc)

    # pick most jammed candidate for final LP
    cand_final = Xbest
    if len(archive) > 0:
        # most jammed: max ncon with penalty for rattlers, allow slightly worse max
        # filter to those within 0.008 of best max to avoid crazy
        filt = [a for a in archive if a['m'] < bestv + 0.008]
        if len(filt) == 0:
            filt = archive
        filt.sort(key=lambda a: (-a['ncon'] + 8 * a['nratt'], a['m']))
        cand_final = filt[0]['X'].copy()
        # if LP candidate much worse in max than global best, still polish jammed one
        # but remember global best for fallback
    # final 20s LP polish
    if time.time() < HARD - 0.8:
        try:
            Xp = _lp_polish(cand_final, HARD - 0.4, radius_init=8e-5, cap=1500, max_it=25)
            vp = _max_cos(Xp)
            if vp < bestv:
                bestv = vp; Xbest = Xp.copy()
            else:
                # also try polishing global best briefly
                if time.time() < HARD - 3.0 and np.max(np.abs(Xp - Xbest)) > 1e-12:
                    Xp2 = _lp_polish(Xbest, HARD - 0.4, radius_init=3e-5, cap=1500, max_it=10)
                    v2 = _max_cos(Xp2)
                    if v2 < bestv:
                        bestv = v2; Xbest = Xp2.copy()
        except Exception:
            pass
    # ensure returnable
    try:
        Xbest = _norm(np.asarray(Xbest, dtype=float))
        if not np.all(np.isfinite(Xbest)) or Xbest.shape != (82, 14):
            Xbest = _norm(fb.copy())
    except Exception:
        Xbest = _norm(fb.copy())
    return Xbest
