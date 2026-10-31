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
    return float(G.max())

def _anneal(X0, steps, a0, a1, s0, s1, mu=0.85, seed=0):
    X = _norm(X0.copy())
    vel = np.zeros_like(X)
    bv = _max_cos(X)
    bX = X.copy()
    for it in range(steps):
        frac = it / max(steps - 1, 1)
        alpha = a0 * (a1 / a0) ** frac if a0 != a1 else a0
        smax = s0 * (s1 / s0) ** frac if s0 != s1 else s0
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        if m < bv:
            bv = m
            bX = X.copy()
        W = np.exp(np.clip(alpha * (G - m), -50, 0))
        np.fill_diagonal(W, 0.0)
        grad = W @ X
        dot = np.sum(grad * X, axis=1, keepdims=True)
        gt = grad - dot * X
        vel = mu * vel + gt
        nrm = float(np.linalg.norm(vel, axis=1).max())
        if nrm > 1e-14:
            X = X - (smax / nrm) * vel
            X = _norm(X)
    fv = _max_cos(X)
    if fv < bv:
        return X, fv
    return bX, bv

def _lp_polish(x, deadline, cap=4500):
    x = _norm(x.copy())
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x, 2.0
    radius = 1e-3
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        m = float(c_all.max())
        sel = c_all > m - max(20.0 * radius, 1e-9)
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > cap:
            idx = np.where(sel)[0]
            order = np.argpartition(-c_all[idx], cap - 1)[:cap]
            keep = np.zeros_like(sel, dtype=bool)
            keep[idx[order]] = True
            sel = keep
        c = c_all[sel]
        I = iu0[0][sel]; J = iu0[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def _contact_set(X, slack=2e-6):
    n = X.shape[0]
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    iu0 = np.triu_indices(n, 1)
    mask = G[iu0] >= m - slack
    ii = iu0[0][mask]; jj = iu0[1][mask]
    s = set(int(a) * n + int(b) for a, b in zip(ii, jj))
    return s, m

def _jaccard(A, B):
    u = len(A | B)
    if u == 0:
        return 1.0
    return len(A & B) / u

def _kabsch(A, B):
    # min || A @ R.T? we want R (4x4) max trace(R H), H=A^T B
    H = A.T @ B
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T
    d = np.linalg.det(V @ U.T)
    D = np.eye(4)
    if d < 0:
        D[3, 3] = -1.0
    R = V @ D @ U.T
    if np.linalg.det(R) < 0:
        R[:, 3] *= -1
    return R

def _align_and_match(A, B, iters=6):
    # A,B (n,4). Find R in SO(4) and perm to maximize matched cosines.
    # ICP alternating Kabsch + assignment. R applied to A.
    R = np.eye(4)
    n = A.shape[0]
    perm = np.arange(n)
    Ar = A.copy()
    for _ in range(iters):
        C = Ar @ B.T
        # maximize -> minimize -C
        try:
            r, c = linear_sum_assignment(-C)
        except Exception:
            break
        # r should be arange
        Bp = B[c, :]
        # reorder so Bp aligns with Ar order: Bp[i] matched to Ar[i]
        # linear_sum_assignment returns row_ind sorted; ensure mapping
        # r is permutation of rows; build perm
        tmp = np.empty(n, dtype=int)
        tmp[r] = c
        perm = tmp
        Bp_ordered = B[perm, :]
        Rn = _kabsch(A, Bp_ordered)  # R maps A -> Bp
        R = Rn
        Ar = A @ R.T  # apply row-vector convention: (R @ a_col) => a_row @ R.T
        # check convergence: loop again
    # final assignment with final Ar
    try:
        C = Ar @ B.T
        r, c = linear_sum_assignment(-C)
        tmp = np.empty(n, dtype=int)
        tmp[r] = c
        perm = tmp
    except Exception:
        pass
    Ar = A @ R.T
    return R, perm, Ar

def _random_so4(rng):
    Q, _ = np.linalg.qr(rng.standard_normal((4, 4)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q

def _hopf_grid(rng, n, Q=None):
    # randomized Hopf/toroidal grid: torus coords
    if Q is None:
        Q = _random_so4(rng)
    # grid sizes ~ sqrt
    m1 = int(np.ceil(np.sqrt(n * 1.0)))
    # generate n points: uniform-ish on Clifford torus family + jitter
    # sample theta via arccos uniform, phi1,phi2 uniform grid + jitter
    X = np.zeros((n, 4))
    k = 0
    # make grid
    g = int(np.ceil(np.sqrt(n)))
    for i in range(g):
        for j in range(g):
            if k >= n:
                break
            # stratified
            u = (i + rng.rand()) / g
            v = (j + rng.rand()) / g
            w = (k + 0.5) / n
            # theta from w? cycle through
            th = np.arccos(2.0 * ((k * 0.61803398875) % 1.0) - 1.0) / 2.0  # cos^2 distribution-ish
            # Actually want cos(th),sin(th) uniform on sphere of radii: use uniform
            th = np.arcsin(np.sqrt((k * 0.7548776662 + 0.3) % 1.0))
            p1 = 2 * np.pi * u
            p2 = 2 * np.pi * v
            X[k, 0] = np.cos(th) * np.cos(p1)
            X[k, 1] = np.cos(th) * np.sin(p1)
            X[k, 2] = np.sin(th) * np.cos(p2)
            X[k, 3] = np.sin(th) * np.sin(p2)
            k += 1
        if k >= n:
            break
    X = X @ Q.T
    return _norm(X)

def _find_hole(Y, rng, n_samples=12000):
    d = Y.shape[1]
    best_s = 2.0
    best_c = None
    bs = 3000
    for a in range(0, n_samples, bs):
        nb = min(bs, n_samples - a)
        Cb = rng.standard_normal((nb, d))
        Cb = Cb / np.linalg.norm(Cb, axis=1, keepdims=True)
        S = Cb @ Y.T
        sm = S.max(axis=1)
        k2 = int(np.argmin(sm))
        if sm[k2] < best_s:
            best_s = float(sm[k2])
            best_c = Cb[k2].copy()
    if best_c is None:
        best_c = rng.standard_normal(d)
        best_c /= np.linalg.norm(best_c)
    # quick refine
    z = best_c / np.linalg.norm(best_c)
    vel = np.zeros(d)
    for it in range(120):
        dots = Y @ z
        m = float(dots.max())
        w = np.exp(np.clip(60.0 * (dots - m), -50, 0))
        sw = float(w.sum()) + 1e-300
        g = (w @ Y) / sw
        gt = g - float(g @ z) * z
        nn = float(np.linalg.norm(gt))
        if nn < 1e-14:
            break
        gt = gt / nn
        vel = 0.6 * vel + gt
        vn = float(np.linalg.norm(vel))
        if vn < 1e-14:
            continue
        step = 0.04 / (1.0 + it / 60.0)
        z = z - (step / vn) * vel
        nz = float(np.linalg.norm(z))
        if nz < 1e-12 or not np.isfinite(nz):
            break
        z = z / nz
    return z / np.linalg.norm(z)

def solve():
    t0 = time.time()
    HARD_END = t0 + 174.0
    POP_END = t0 + 55.0
    RECOMB_END = t0 + 150.0
    n, d = 272, 4
    rng = np.random.RandomState(20261031)
    best_X = None
    best_val = 2.0
    rec_X = None
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (n, d):
                a = _norm(a)
                if np.all(np.isfinite(a)):
                    v = _max_cos(a)
                    if fn == "record.npy":
                        rec_X = a.copy()
                    if v < best_val:
                        best_val = v
                        best_X = a.copy()
        except Exception:
            pass
    if best_X is None:
        best_X = _norm(rng.randn(n, d))
        best_val = _max_cos(best_X)
    if rec_X is None:
        rec_X = best_X.copy()
    init_val = float(best_val)
    # ---------- population ----------
    pop = []
    pop.append((best_X.copy(), float(best_val)))
    vrec = _max_cos(rec_X)
    # add record if distinct
    if abs(vrec - best_val) > 1e-12 or np.linalg.norm(np.sort((best_X @ rec_X.T).max(axis=1)) - 1.0) > 1e-9:
        # check not duplicate array
        if np.linalg.norm(best_X - rec_X) > 1e-8:
            pop.append((rec_X.copy(), float(vrec)))
    # target size 8
    TARGET = 8
    # helper to anneal quickly
    def quick_from(X0, steps, a0, a1, s0, s1):
        Xp, vp = _anneal(X0, steps, a0, a1, s0, s1)
        return Xp, vp
    # add jittered best
    njit = 3
    for k in range(njit):
        if time.time() > POP_END - 8:
            break
        amp = [0.02, 0.05, 0.09][k % 3]
        Xn = best_X + amp * rng.randn(n, d)
        Xn = _norm(Xn)
        Xp, vp = quick_from(Xn, 350, 60.0, 350.0, 0.008, 0.002)
        pop.append((Xp, vp))
        if vp < best_val:
            best_val = vp; best_X = Xp.copy()
    # random starts
    nrand = 2
    for k in range(nrand):
        if time.time() > POP_END - 5:
            break
        Xr = _norm(rng.randn(n, d))
        Xp, vp = quick_from(Xr, 500, 40.0, 400.0, 0.01, 0.002)
        pop.append((Xp, vp))
        if vp < best_val:
            best_val = vp; best_X = Xp.copy()
    # Hopf grids
    nhopf = 2
    for k in range(nhopf):
        if time.time() > POP_END - 2:
            break
        Xh = _hopf_grid(rng, n)
        # small random perturbation
        Xh = _norm(Xh + 0.01 * rng.randn(n, d))
        Xp, vp = quick_from(Xh, 450, 50.0, 380.0, 0.009, 0.002)
        pop.append((Xp, vp))
        if vp < best_val:
            best_val = vp; best_X = Xp.copy()
    # brief LP polish each (very short)
    polished = []
    for (Xp, vp) in pop:
        if time.time() > POP_END:
            polished.append((Xp, vp))
            continue
        remain = POP_END - time.time()
        # give at most 2s each, share
        dl = time.time() + min(2.0, max(0.4, remain / max(1, len(pop)))))
        try:
            Xl, vl = _lp_polish(Xp, dl, cap=3500)
        except Exception:
            Xl, vl = Xp, vp
        polished.append((Xl, vl))
        if vl < best_val:
            best_val = vl; best_X = Xl.copy()
    pop = polished
    # sort by value
    pop.sort(key=lambda t: t[1])
    # keep diverse: limit size
    if len(pop) > TARGET:
        pop = pop[:TARGET]
    # contact sets for diversity
    try:
        cs_best, _ = _contact_set(best_X, slack=2e-6)
    except Exception:
        cs_best = set()
    archive_sets = []
    for (Xp, _) in pop:
        try:
            s, _ = _contact_set(Xp, slack=2e-6)
        except Exception:
            s = set()
        archive_sets.append(s)
    # ---------- recombination ----------
    rng2 = np.random.RandomState(777123)
    nchild = 0
    # elite pool: top 5
    while time.time() < RECOMB_END:
        if len(pop) < 2:
            break
        # pick two elite-but-different parents
        # sort pop
        pop.sort(key=lambda t: t[1])
        E = min(5, len(pop))
        # choose first among top E randomly weighted, second different maximizing difference
        i1 = rng2.randint(E)
        # pick i2 != i1 maximizing (1-jaccard) + value penalty
        best_pick = -1
        best_score = -1e99
        for cand in range(len(pop)):
            if cand == i1:
                continue
            try:
                j = _jaccard(archive_sets[i1] if i1 < len(archive_sets) else cs_best,
                             archive_sets[cand] if cand < len(archive_sets) else cs_best)
            except Exception:
                j = 0.5
            div = 1.0 - j
            # prefer good values too
            score = div * 2.0 - 0.5 * (pop[cand][1] - pop[0][1]) * 500.0 + rng2.rand() * 0.1
            if score > best_score:
                best_score = score
                best_pick = cand
        if best_pick < 0:
            best_pick = (i1 + 1) % len(pop)
        A = pop[i1][0].copy()
        B = pop[best_pick][0].copy()
        # align: optimize SO(4) (pair of unit quats equiv = Kabsch) + assignment
        try:
            R, perm, Ar = _align_and_match(A, B, iters=5)
        except Exception:
            R = np.eye(4); perm = np.arange(n); Ar = A.copy()
        Bp = B[perm, :]
        # choose half-space type
        mode = rng2.rand()
        if mode < 0.4:
            # random spatial half-space
            u = rng2.randn(4); u /= np.linalg.norm(u)
            mid = _norm((Ar + Bp) * 0.5 + 1e-12)
            # guard zero rows (antipodal): fallback to Ar
            # compute mask
            try:
                proj = mid @ u
            except Exception:
                proj = Ar @ u
            mask = proj >= 0
        elif mode < 0.7:
            # Hopf half-space: random torus frame
            Qh = _random_so4(rng2)
            At = Ar @ Qh.T
            Bt = Bp @ Qh.T
            # toroidal coordinate: difference of phases? use sign of (a0*a3 - a1*a2)? use mid
            midt = (At + Bt) * 0.5
            # Hopf invariant-ish: h = x0*x3 - x1*x2 (one component of Hopf map)
            h = midt[:, 0] * midt[:, 3] - midt[:, 1] * midt[:, 2]
            # threshold at median to balance
            thr = float(np.median(h))
            mask = h >= thr
        else:
            # random spatial half-space with offset (cap) for diversity
            u = rng2.randn(4); u /= np.linalg.norm(u)
            mid = _norm((Ar + Bp) * 0.5 + 1e-12)
            proj = mid @ u
            thr = float(np.median(proj))
            mask = proj >= thr
        child = np.where(mask[:, None], Ar, Bp)
        child = _norm(child)
        # resolve near-duplicates by hole insertion
        try:
            for _ in range(6):
                G = child @ child.T
                np.fill_diagonal(G, -2.0)
                m = float(G.max())
                if m < 0.96:
                    break
                ij = np.unravel_index(int(np.argmax(G)), G.shape)
                # replace second with hole
                keep = np.delete(child, ij[1], axis=0)
                hz = _find_hole(keep, rng2, n_samples=8000)
                child[ij[1]] = hz
        except Exception:
            pass
        # anneal child
        try:
            Xc, vc0 = _anneal(child, 380, 70.0, 420.0, 0.007, 0.0018)
        except Exception:
            Xc, vc0 = child, _max_cos(child)
        # LP polish briefly
        dl2 = time.time() + min(3.0, max(0.5, (RECOMB_END - time.time()) / 3.0))
        try:
            Xc2, vc = _lp_polish(Xc, dl2, cap=4000)
        except Exception:
            Xc2, vc = Xc, vc0
        nchild += 1
        # retain?
        try:
            s_c, _ = _contact_set(Xc2, slack=2e-6)
        except Exception:
            s_c = set()
        # diversity vs pop
        maxj = 0.0
        for s in archive_sets:
            try:
                j = _jaccard(s_c, s)
            except Exception:
                j = 0.0
            if j > maxj:
                maxj = j
        is_diverse = maxj < 0.85
        if vc < best_val - 1e-12 or (is_diverse and vc < best_val + 0.002 and len(pop) < 10):
            # insert
            pop.append((Xc2.copy(), float(vc)))
            archive_sets.append(set(s_c))
            pop.sort(key=lambda t: t[1])
            # trim to 10 keeping best + diverse
            if len(pop) > 10:
                # keep best, drop worst non-diverse? simply drop worst
                # but ensure diversity: drop the one with highest jaccard among worst half
                pop = pop[:10]
                archive_sets = archive_sets[:10] if len(archive_sets) >= 10 else archive_sets
                # recompute sets for trimmed pop to stay aligned
                archive_sets = []
                for (Xp, _) in pop:
                    try:
                        ss, _ = _contact_set(Xp, slack=2e-6)
                    except Exception:
                        ss = set()
                    archive_sets.append(ss)
            if vc < best_val:
                best_val = float(vc); best_X = Xc2.copy()
                cs_best = set(s_c)
        else:
            # even if not retained, maybe it beats pop worst and is decent?
            if vc < pop[-1][1] - 5e-7 and maxj < 0.97:
                pop[-1] = (Xc2.copy(), float(vc))
                if len(archive_sets) == len(pop):
                    archive_sets[-1] = set(s_c)
                if vc < best_val:
                    best_val = float(vc); best_X = Xc2.copy()
        if time.time() > RECOMB_END:
            break
        # early exit if huge improvement? continue to refine
    # ---------- final long LP polish of best ----------
    try:
        dlF = HARD_END - 1.0
        Xf, vf = _lp_polish(best_X, dlF, cap=5000)
        if vf < best_val:
            best_X, best_val = Xf, vf
    except Exception:
        pass
    # fallback
    if best_val > init_val + 1e-12:
        try:
            a = np.load("best.npy")
            a = _norm(np.asarray(a, dtype=np.float64))
            if a.shape == (n, d):
                return a
        except Exception:
            pass
    best_X = _norm(best_X)
    return best_X
