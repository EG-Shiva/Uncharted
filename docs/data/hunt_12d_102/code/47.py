import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _lp_polish(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        cmat = x @ x.T
        c = cmat[iu]
        m = float(c.max())
        thr = m - max(20 * radius, 1e-9)
        sel = c > thr
        if int(sel.sum()) == 0:
            break
        I = iu[0][sel]
        J = iu[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols_arr = np.empty((p, 2 * d + 1), dtype=np.int64)
        cols_arr[:, :d] = I[:, None] * d + np.arange(d)
        cols_arr[:, d:2 * d] = J[:, None] * d + np.arange(d)
        cols_arr[:, 2 * d] = n * d
        cols = cols_arr.ravel()
        vals_arr = np.empty((p, 2 * d + 1))
        vals_arr[:, :d] = x[J]
        vals_arr[:, d:2 * d] = x[I]
        vals_arr[:, 2 * d] = -1.0
        vals = vals_arr.ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix(
            (np.tile(x.ravel(), 1), (np.repeat(np.arange(n), d), np.arange(n * d))),
            shape=(n, nv),
        ).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(
                obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                method="highs",
            )
        except Exception:
            break
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def _smooth_adam(x, steps, T0, T1, lr0, lr1, deadline):
    n, d = x.shape
    m1 = np.zeros_like(x)
    m2 = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for s in range(steps):
        if (s & 63) == 0 and time.time() > deadline:
            break
        frac = s / max(steps - 1, 1)
        T = T0 * (T1 / T0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        mv = float(G.max())
        E = np.exp((G - mv) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if S <= 0 or not np.isfinite(S):
            break
        W = E / S
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad = grad - dot * x
        m1 = b1 * m1 + (1 - b1) * grad
        m2 = b2 * m2 + (1 - b2) * (grad * grad)
        c1 = 1 - b1 ** (s + 1)
        c2 = 1 - b2 ** (s + 1)
        mh = m1 / c1
        vh = m2 / c2
        step = lr * mh / (np.sqrt(vh) + eps)
        x = x - step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def _build_pool(rng):
    d = 12
    sq2 = 1.0 / np.sqrt(2.0)
    pool = []
    # D12-type roots: two coords +-1/sqrt2
    for i in range(d):
        for j in range(i + 1, d):
            for s1 in (1.0, -1.0):
                for s2 in (1.0, -1.0):
                    v = np.zeros(d)
                    v[i] = s1 * sq2
                    v[j] = s2 * sq2
                    pool.append(v)
    # coordinate vectors +-e_i
    for i in range(d):
        for s in (1.0, -1.0):
            v = np.zeros(d)
            v[i] = s
            pool.append(v)
    # all 3-sparse sign vectors
    from itertools import combinations
    sq3 = 1.0 / np.sqrt(3.0)
    for comb in combinations(range(d), 3):
        for mask in range(8):
            v = np.zeros(d)
            for k, idx in enumerate(comb):
                s = 1.0 if ((mask >> k) & 1) else -1.0
                v[idx] = s * sq3
            pool.append(v)
    # all 4-sparse sign vectors
    sq4 = 0.5
    for comb in combinations(range(d), 4):
        for mask in range(16):
            v = np.zeros(d)
            for k, idx in enumerate(comb):
                s = 1.0 if ((mask >> k) & 1) else -1.0
                v[idx] = s * sq4
            pool.append(v)
    P0 = np.array(pool, dtype=np.float64)
    # sampled {0,+-1/2,+-sqrt3/2} patterns
    s3 = np.sqrt(3.0) / 2.0
    vals_set = np.array([0.5, -0.5, s3, -s3], dtype=np.float64)
    extra = []
    n_sparse = 7500
    for _ in range(n_sparse):
        k = int(rng.integers(2, 7))  # 2..6
        idx = rng.choice(d, size=k, replace=False)
        v = np.zeros(d)
        picks = rng.integers(0, 4, size=k)
        v[idx] = vals_set[picks]
        nrm = np.linalg.norm(v)
        if nrm > 0:
            v /= nrm
            extra.append(v)
    n_dense = 7500
    for _ in range(n_dense):
        v = np.zeros(d)
        for j in range(d):
            r = rng.random()
            if r < 0.45:
                v[j] = 0.0
            elif r < 0.60:
                v[j] = 0.5
            elif r < 0.70:
                v[j] = -0.5
            elif r < 0.85:
                v[j] = s3
            else:
                v[j] = -s3
        nrm = np.linalg.norm(v)
        if nrm < 1e-12:
            v = rng.standard_normal(d)
            nrm = np.linalg.norm(v)
            v /= nrm
            extra.append(v)
        else:
            v /= nrm
            extra.append(v)
    P1 = np.array(extra, dtype=np.float64)
    P = np.vstack([P0, P1])
    # normalise (safety)
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    return P


def solve():
    t0 = time.time()
    HARD_END = t0 + 168.0
    rng = np.random.default_rng(7)
    n, d = 102, 12
    TAU = 0.02
    GAP = 1e-4

    # fallback incumbent
    try:
        xb0 = np.load("best.npy")
    except Exception:
        try:
            xb0 = np.load("record.npy")
        except Exception:
            xb0 = rng.standard_normal((n, d))
    xb0 = np.asarray(xb0, dtype=np.float64)
    if xb0.shape != (n, d):
        tmp = rng.standard_normal((n, d))
        m = min(xb0.size, tmp.size)
        tmp.ravel()[:m] = xb0.ravel()[:m]
        xb0 = tmp
    xb0 = xb0 / np.linalg.norm(xb0, axis=1, keepdims=True)
    best_fallback_val = _max_cos(xb0)

    # ---- large discrete pool ----
    Pool = _build_pool(rng)
    P = Pool.shape[0]

    used = np.zeros(P, dtype=bool)
    S_idx = []
    # D matrix columns incrementally
    Dmat = np.empty((P, 0), dtype=np.float64)
    M_S = -2.0
    # greedy build
    # seed with index 0
    seed = 0
    S_idx.append(seed)
    used[seed] = True
    col = Pool @ Pool[seed]
    Dmat = np.column_stack([Dmat, col]) if Dmat.size else col.reshape(P, 1)
    # update M_S (single point -> -2)
    M_S = -2.0

    iu_small = None
    for k in range(1, n):
        if time.time() > HARD_END - 120.0:
            break
        # current S Gram for threshold counting
        Sm = Pool[np.array(S_idx)]
        kk = len(S_idx)
        Gsmall = Sm @ Sm.T
        iu2 = np.triu_indices(kk, 1)
        s_vals = Gsmall[iu2] if kk >= 2 else np.array([-2.0])
        if kk >= 2:
            M_S = float(s_vals.max())
        else:
            M_S = -2.0
        s_sorted = np.sort(s_vals)  # ascending
        D = Dmat  # P x kk
        # max to set
        mc = D.max(axis=1)
        Mp = np.maximum(M_S, mc)
        # mask unused
        unused = ~used
        Mp_u = Mp[unused]
        bestM = Mp_u.min()
        # candidates within 1e-12 of bestM
        cand_pos = np.flatnonzero(unused)
        m1mask = Mp[cand_pos] <= bestM + 1e-12
        c1 = cand_pos[m1mask]
        if len(c1) == 1:
            best = int(c1[0])
        else:
            # count criterion for c1 only
            D1 = D[c1]  # L x kk
            Mp1 = Mp[c1]
            thr1 = Mp1 - GAP
            # count_S above thr: via searchsorted
            # number > thr = len - idx_right
            idx = np.searchsorted(s_sorted, thr1 + 1e-18, side='right')
            cntS = len(s_sorted) - idx
            # cnt_c: compare per row
            cntC = np.sum(D1 > thr1[:, None], axis=1)
            Cp = cntS + cntC
            bestC = Cp.min()
            c2 = c1[Cp <= bestC]
            if len(c2) == 1:
                best = int(c2[0])
            else:
                D2 = D[c2]
                soft = np.sum(np.exp((D2 - 1.0) / TAU), axis=1)
                bi = int(np.argmin(soft))
                best = int(c2[bi])
        S_idx.append(best)
        used[best] = True
        col = Pool @ Pool[best]
        Dmat = np.column_stack([Dmat, col])
        # early M_S update (approx)
        mc_new = float((Pool[best] @ Sm.T).max()) if kk > 0 else -2.0
        if mc_new > M_S:
            M_S = mc_new

    # if greedy aborted early, fill randomly
    while len(S_idx) < n:
        cand = int(rng.integers(0, P))
        if not used[cand]:
            S_idx.append(cand)
            used[cand] = True
            col = Pool @ Pool[cand]
            Dmat = np.column_stack([Dmat, col])

    S_idx = np.array(S_idx, dtype=np.int64)
    S_mat = Pool[S_idx].copy()

    # ---- one-for-one swaps under lexicographic objective ----
    iu = np.triu_indices(n, 1)
    pI = iu[0]
    pJ = iu[1]

    def build_struct(Sm):
        G = Sm @ Sm.T
        vals = G[iu]
        M = float(vals.max())
        C = int(np.sum(vals > M - GAP))
        L = float(np.sum(np.exp((vals - 1.0) / TAU)))
        # per-i structures
        others_vals = []
        Mwo = np.empty(n)
        Swo = np.empty(n)
        for i in range(n):
            mask = (pI != i) & (pJ != i)
            ov = vals[mask]
            others_vals.append(ov)
            Mwo[i] = float(ov.max()) if ov.size else -2.0
            Swo[i] = float(np.sum(np.exp((ov - 1.0) / TAU))) if ov.size else 0.0
        return G, vals, M, C, L, others_vals, Mwo, Swo

    G, vals, Mb, Cb, Lb, others_vals, Mwo, Swo = build_struct(S_mat)
    unused_idx = np.flatnonzero(~used)
    # time box swaps ~ up to 20s but leave time for continuous
    SWAP_END = min(HARD_END - 100.0, t0 + 28.0)
    attempts = 0
    max_attempts = 60000
    # refresh unused list occasionally
    while attempts < max_attempts and time.time() < SWAP_END:
        # batch of 500 without time check each iter? check every 500
        for _ in range(500):
            if attempts >= max_attempts:
                break
            attempts += 1
            i = int(rng.integers(0, n))
            # pick random unused
            ri = int(rng.integers(0, len(unused_idx)))
            c = int(unused_idx[ri])
            cvec = Pool[c]
            dots = S_mat @ cvec  # n,
            # exclude i
            # m_c without i
            # fast max excluding i
            mc = -2.0
            # manual max to avoid alloc? use numpy
            # create copy without i via masking
            # np.max on masked: use partition
            # simple:
            m1 = float(dots[np.arange(n) != i].max())
            Mp_ = m1 if m1 > Mwo[i] else float(Mwo[i])
            thr = Mp_ - GAP
            ov = others_vals[i]
            cntO = int(np.sum(ov > thr)) if ov.size else 0
            # cnt_c
            cntc = 0
            # loop over j != i count dots>thr (n=102, python loop fine? use numpy)
            cntc = int(np.sum(dots[np.arange(n) != i] > thr))
            Cp_ = cntO + cntc
            better = False
            if Mp_ < Mb - 1e-12:
                better = True
            elif abs(Mp_ - Mb) <= 1e-12:
                if Cp_ < Cb:
                    better = True
                elif Cp_ == Cb:
                    # soft-max
                    mask2 = np.arange(n) != i
                    sc = float(np.sum(np.exp((dots[mask2] - 1.0) / TAU)))
                    Lp_ = float(Swo[i] + sc)
                    if Lp_ < Lb - 1e-10:
                        better = True
            if better:
                # accept
                old = int(S_idx[i])
                S_idx[i] = c
                S_mat[i] = cvec
                used[old] = False
                used[c] = True
                G, vals, Mb, Cb, Lb, others_vals, Mwo, Swo = build_struct(S_mat)
                # refresh unused list
                unused_idx = np.flatnonzero(~used)
                break
        if time.time() >= SWAP_END:
            break

    # ---- continuous release ----
    x = S_mat.copy()
    # tiny jitter to break exact symmetry
    x = x + 1e-4 * rng.standard_normal(x.shape)
    x /= np.linalg.norm(x, axis=1, keepdims=True)

    # annealed descent, leave ~12s for LP
    x = _smooth_adam(x, 2200, 0.02, 0.002, 0.02, 0.004, HARD_END - 12.0)
    v = _max_cos(x)
    # short LP if promising or time remains
    if time.time() < HARD_END - 4.0:
        # only polish if within margin of fallback to save time? always polish
        x = _lp_polish(x, HARD_END - 1.5)
        v = _max_cos(x)
    # extra smoothing+polish cycle if time remains
    if time.time() < HARD_END - 25.0:
        x = _smooth_adam(x, 1200, 0.008, 0.001, 0.008, 0.002, HARD_END - 12.0)
        if time.time() < HARD_END - 4.0:
            x = _lp_polish(x, HARD_END - 1.5)

    x /= np.linalg.norm(x, axis=1, keepdims=True)
    v = _max_cos(x)
    # keep fallback if discrete path worse
    if v > best_fallback_val:
        # polish fallback a bit and return it
        y = xb0.copy()
        if time.time() < HARD_END - 3.0:
            y = _lp_polish(y, HARD_END - 1.0)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            vy = _max_cos(y)
            if vy < best_fallback_val:
                return y
            # if discrete still worse, return better of x,y
            if vy < v:
                return y
        # if no time, return original fallback if it beats x
        if best_fallback_val < v:
            return xb0 / np.linalg.norm(xb0, axis=1, keepdims=True)
    return x
