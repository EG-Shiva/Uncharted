import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _smooth_refine(X, iters, a0=80.0, a1=400.0, s0=0.03, s1=0.003):
    n, d = X.shape
    for it in range(iters):
        frac = it / max(1, iters - 1)
        alpha = a0 * (a1 / a0) ** frac if iters > 1 else a0
        step = s0 * (s1 / s0) ** frac if iters > 1 else s0
        G = X @ X.T
        np.fill_diagonal(G, -np.inf)
        gmax = float(G.max())
        W = np.exp(alpha * (G - gmax))
        np.fill_diagonal(W, 0.0)
        W[W < 1e-12] = 0.0
        F = W @ X
        dot = np.sum(F * X, axis=1, keepdims=True)
        Ft = F - dot * X
        m = float(np.linalg.norm(Ft, axis=1).mean()) + 1e-12
        X = X - (step / m) * Ft
        X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-30
    return X

def _lp_polish(X, seconds, window=None, r0=1e-3):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    try:
        best = _max_cos(X)
    except Exception:
        return X
    radius = r0
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start > seconds - 0.4:
            break
        Gf = X @ X.T
        c = Gf[iu]
        m = float(c.max())
        if window is None:
            sel = c > m - max(20 * radius, 1e-9)
        else:
            sel = c > m - window
        cnt = int(sel.sum())
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 3000:
            thr = np.sort(c[sel])[-3000]
            sel = c >= thr
            cnt = int(sel.sum())
        I, J = iu[0][sel], iu[1][sel]
        p = cnt
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d),
                                   J[:, None] * d + np.arange(d),
                                   np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
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
        y = X + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True) + 1e-30
        v = _max_cos(y)
        if v < best:
            X, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def _tangent_bases(X):
    # 7x8 orthonormal tangent basis per point via SVD, stored as (n,8,7)
    n, d = X.shape
    T = np.empty((n, d, d - 1))
    for i in range(n):
        xi = X[i:i+1, :]
        try:
            _, _, Vh = np.linalg.svd(xi, full_matrices=True)
            # Vh is 8x8, first row ~ xi direction
            Ti = Vh[1:, :].T  # 8x7
        except Exception:
            # fallback: QR of random complement
            q, _ = np.linalg.qr(np.random.randn(d, d))
            # project out xi
            q = q - np.outer(xi.ravel(), (q.T @ xi.ravel()))
            q, _ = np.linalg.qr(q)
            # ensure orthogonal to xi: take 7 cols least aligned
            dots = np.abs(q.T @ xi.ravel())
            idx = np.argsort(dots)[:d-1]
            Ti = q[:, idx]
        # re-orthogonalize against xi numerically
        Ti = Ti - np.outer(X[i], (X[i] @ Ti))
        # normalize columns
        try:
            Ti, _ = np.linalg.qr(Ti)
        except Exception:
            pass
        T[i] = Ti
    return T

def solve():
    t0 = time.time()
    rng = np.random.default_rng(0)
    n, d = 162, 8
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.array(a, dtype=float)
            a /= np.linalg.norm(a, axis=1, keepdims=True) + 1e-30
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                cands.append(a)
        except Exception:
            pass
    if len(cands) == 0:
        X0 = rng.standard_normal((n, d))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        cands.append(X0)
    vals = [_max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    anchor = cands[bi].copy()
    anchor_val = float(vals[bi])
    bestX = anchor.copy()
    bestV = anchor_val

    DEADLINE = 178.0
    CRAWL_END = t0 + 154.0
    STEPS = [0.002, 0.01, 0.03, 0.08]
    SIGNED = []
    for s in STEPS:
        SIGNED.append(s)
        SIGNED.append(-s)

    outer = 0
    # quick touch: no heavy initial polish, already jammed
    while time.time() - t0 < CRAWL_END:
        outer += 1
        if time.time() - t0 > CRAWL_END - 3.0:
            break
        # build tangent bases by SVD
        try:
            Tstack = _tangent_bases(anchor)
        except Exception:
            Tstack = None
        # contact Jacobian A for slack < 1e-6
        try:
            G = anchor @ anchor.T
            iu_up = np.triu_indices(n, 1)
            cfull = G[iu_up]
            m0 = float(cfull.max())
            slack = m0 - cfull
            sel = slack < 1e-6
            Iall, Jall = iu_up[0][sel], iu_up[1][sel]
            # cap to tightest 1600 for SVD speed
            if len(Iall) > 1600:
                order = np.argsort(cfull[sel])[::-1][:1600]
                Iall, Jall = Iall[order], Jall[order]
            p = len(Iall)
        except Exception:
            p = 0
            Iall, Jall = np.array([], dtype=int), np.array([], dtype=int)
        # form dense Jacobian A (p x 1134)
        Ntan = n * (d - 1)
        null_idx = None
        Vt = None
        if p > 0 and Tstack is not None:
            try:
                A = np.zeros((p, Ntan))
                # vectorized row construction
                # need T projections: T_i^T x_j
                for k in range(p):
                    i, j = int(Iall[k]), int(Jall[k])
                    pi = Tstack[i].T @ anchor[j]  # 7
                    pj = Tstack[j].T @ anchor[i]  # 7
                    A[k, i*7:(i+1)*7] = pi
                    A[k, j*7:(j+1)*7] = pj
                # dense SVD for smallest singular directions
                _, S, Vth = np.linalg.svd(A, full_matrices=True)
                Vt = Vth  # 1134 x 1134
                Sext = np.zeros(Vt.shape[0])
                Sext[:len(S)] = S
                cand_null = np.where(Sext < 1e-8)[0]
                if len(cand_null) == 0:
                    # lowest available: 24 smallest
                    order = np.argsort(Sext)
                    cand_null = order[:min(24, len(Sext))]
                # cap huge null space to 120 lowest for stable random combos
                if len(cand_null) > 120:
                    # Sext all ~0, pick random subset of 120
                    cand_null = rng.choice(cand_null, size=120, replace=False)
                null_idx = cand_null
            except Exception:
                null_idx = None
                Vt = None
        if null_idx is None or Vt is None or len(null_idx) == 0:
            # fallback: random tangent field, still test steps
            null_idx = None
        # generate random null combos
        KCOMB = 6
        trials = []  # (smooth_val, X_smooth)
        for kc in range(KCOMB):
            if time.time() - t0 > CRAWL_END - 1.0:
                break
            # random null direction
            try:
                if null_idx is not None:
                    coeff = rng.standard_normal(len(null_idx))
                    v = coeff @ Vt[null_idx, :]
                else:
                    v = rng.standard_normal(Ntan)
                    # project to tangent already (any vector is tangent coords)
                nv = float(np.linalg.norm(v)) + 1e-30
                v /= nv
                # map to ambient dx
                vblk = v.reshape(n, d-1)
                dx = np.einsum('nij,nj->ni', Tstack, vblk) if Tstack is not None else rng.standard_normal((n, d))
                # remove radial component numerically
                dx = dx - np.sum(dx * anchor, axis=1, keepdims=True) * anchor
                rms = float(np.sqrt(np.mean(np.sum(dx*dx, axis=1)))) + 1e-30
                dx /= rms
            except Exception:
                continue
            for ss in SIGNED:
                if time.time() - t0 > CRAWL_END:
                    break
                try:
                    Y0 = anchor + ss * dx
                    Y0 /= np.linalg.norm(Y0, axis=1, keepdims=True) + 1e-30
                    Ys = _smooth_refine(Y0, iters=280, a0=70.0, a1=420.0, s0=0.028, s1=0.0025)
                    vs = _max_cos(Ys)
                    trials.append((vs, Ys))
                except Exception:
                    continue
        if len(trials) == 0:
            continue
        trials.sort(key=lambda t: t[0])
        # corrective LP polish restricted to pairs within 1e-3 for top candidates
        NLIFT = min(3, len(trials))
        lifted = []
        for k in range(NLIFT):
            vs, Ys = trials[k]
            # only lift if within reach of anchor (avoid wasting time on bad)
            if vs > anchor_val + 0.004 and vs > bestV + 0.004:
                # still keep raw for reference
                lifted.append((vs, Ys))
                continue
            secs = 0.9
            if time.time() - t0 > CRAWL_END - 0.5:
                lifted.append((vs, Ys))
                break
            try:
                Yp = _lp_polish(Ys, secs, window=1e-3, r0=2e-3)
                vp = _max_cos(Yp)
                lifted.append((vp, Yp))
                if vp < bestV - 1e-12:
                    bestV = vp
                    bestX = Yp.copy()
            except Exception:
                lifted.append((vs, Ys))
        if len(lifted) == 0:
            continue
        lifted.sort(key=lambda t: t[0])
        best_lift_val, best_lift_X = lifted[0]
        # update global best already done; decide crawl anchor move
        if best_lift_val < anchor_val - 1e-12:
            anchor = best_lift_X.copy()
            anchor_val = float(best_lift_val)
        elif best_lift_val < anchor_val + 0.002:
            # sideways crawl along manifold to new topology
            anchor = best_lift_X.copy()
            anchor_val = float(best_lift_val)
        else:
            # stay; next outer uses new random null directions from same anchor
            # slight perturbation of anchor to avoid exact repeat? no
            pass
        # early exit if time low
        if time.time() - t0 > CRAWL_END:
            break
    # final full LP polish, reserve ~20s
    remain = DEADLINE - (time.time() - t0)
    final_sec = min(22.0, max(6.0, remain - 1.5))
    if final_sec > 1.0 and time.time() - t0 < DEADLINE - 2.0:
        try:
            bestX = _lp_polish(bestX, final_sec, window=None, r0=1e-3)
        except Exception:
            pass
    bestX /= np.linalg.norm(bestX, axis=1, keepdims=True) + 1e-30
    return bestX
