import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _proj(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    y = x / n
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    n2 = np.linalg.norm(y, axis=1, keepdims=True)
    n2[n2 == 0] = 1.0
    return y / n2

def _lp_polish(x, seconds, radius_init=1e-3, cap=2500):
    x = _proj(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = radius_init
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        # time guard per iteration
        if time.time() - start > seconds - 0.2:
            break
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > cap:
            idx = np.argpartition(-c[sel], cap)[:cap]
            I, J = I[idx], J[idx]
            csv = c[sel][idx]
            p = cap
        else:
            csv = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csv, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": 2.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _proj(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _build_modes(x, thresh=1e-3, K=24, seed=0):
    n, d = x.shape
    m = d - 1
    N = n * m
    Gfull = x @ x.T
    np.fill_diagonal(Gfull, -2.0)
    mx = float(Gfull.max())
    iu = np.triu_indices(n, 1)
    c = Gfull[iu]
    # select active pairs
    sel = c > mx - thresh
    # cap / expand to keep problem tractable
    if np.count_nonzero(sel) > 2000:
        k = 2000
        idx = np.argpartition(-c, k)[:k]
        # keep only those indices
        mask = np.zeros_like(sel)
        mask[idx] = True
        # ensure they are above reasonable bound, if not keep top-k anyway
        sel = mask
    if np.count_nonzero(sel) < 50:
        # widen to at least 200 closest pairs
        k = min(200, len(c))
        idx = np.argpartition(-c, k)[:k]
        mask = np.zeros_like(sel)
        mask[idx] = True
        sel = mask
    I, J = iu[0][sel], iu[1][sel]
    p = len(I)
    rng = np.random.default_rng(seed)
    # tangent bases: orthonormal basis of null(x_i)
    Bases = np.empty((n, d, m))
    for i in range(n):
        Gr = rng.standard_normal((d, m))
        Gr -= np.outer(x[i], x[i] @ Gr)
        Q, _ = np.linalg.qr(Gr)
        # re-orthogonalize against x[i] to kill drift
        Q -= np.outer(x[i], x[i] @ Q)
        Q, _ = np.linalg.qr(Q)
        Bases[i] = Q
    # assemble dense Jacobian A (p x N)
    A = np.empty((p, N), dtype=float)
    # vectorized fill via loop over pairs (p up to 2000, fine)
    for k in range(p):
        i, j = int(I[k]), int(J[k])
        ai = Bases[i].T @ x[j]  # (m,)
        aj = Bases[j].T @ x[i]
        row = np.zeros(N)
        row[i*m:(i+1)*m] = ai
        row[j*m:(j+1)*m] = aj
        A[k] = row
    # Gram matrix and eigendecomposition for smallest modes
    try:
        Gram = A.T @ A
        # symmetrize
        Gram = 0.5 * (Gram + Gram.T)
        w, V = np.linalg.eigh(Gram)
        # V columns ascending; take K smallest
        K = min(K, N)
        Modes = V[:, :K]  # N x K
    except Exception:
        # fallback: random orthonormal modes
        Modes = np.linalg.qr(rng.standard_normal((N, K)))[0]
        Bases = Bases  # keep
    return Bases, Modes, mx, Gfull

def solve():
    T0 = time.time()
    HARD_END = T0 + 175.0
    RESERVE_FINAL = 20.0
    rng = np.random.default_rng(0)
    try:
        b0 = np.load("best.npy").astype(float)
    except Exception:
        try:
            b0 = np.load("record.npy").astype(float)
        except Exception:
            b0 = rng.standard_normal((54, 21))
    b0 = _proj(np.nan_to_num(b0))
    if b0.shape != (54, 21):
        b0 = _proj(rng.standard_normal((54, 21)))
    try:
        r0 = _proj(np.load("record.npy").astype(float))
        if r0.shape == (54, 21):
            if _max_cos(r0) < _max_cos(b0):
                b0 = r0
    except Exception:
        pass
    n, d = 54, 21
    best = b0.copy()
    best_val = _max_cos(best)
    # incumbent graph for Hamming distance (absolute threshold)
    def contact_graph(x, thr):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        return (G > thr)
    thr_ref = best_val - 1e-3
    inc_graph = contact_graph(best, thr_ref)

    thetas = [0.05, 0.10, 0.20, 0.35, 0.50]
    tan_thetas = [float(np.tan(t)) for t in thetas]

    # short initial squeeze to be at local optimum (cheap)
    if HARD_END - time.time() > 60:
        best = _lp_polish(best, seconds=min(4.0, (HARD_END - RESERVE_FINAL - time.time()) * 0.15))
        best_val = _max_cos(best)
        inc_graph = contact_graph(best, best_val - 1e-3)
        thr_ref = best_val - 1e-3

    n_outer = 0
    # outer loop: recompute modes, trial, short-polish
    while time.time() < HARD_END - RESERVE_FINAL - 6.0:
        n_outer += 1
        if n_outer > 4:
            break
        time_left_explore = (HARD_END - RESERVE_FINAL) - time.time()
        if time_left_explore < 8:
            break
        # build null modes
        try:
            Bases, Modes, mx, _ = _build_modes(best, thresh=1e-3, K=24, seed=n_outer)
        except Exception:
            break
        K = Modes.shape[1]
        m = d - 1
        N = n * m
        # generate random combinations of 2-5 modes
        n_dir = 28
        cands = []
        # also include pure smallest modes as directions
        pure = min(6, K)
        dirs = []
        for k in range(pure):
            v = Modes[:, k].copy()
            dirs.append(v)
        for _ in range(n_dir):
            nmix = int(rng.integers(2, 6))  # 2..5
            idx = rng.choice(K, size=nmix, replace=False)
            # bias to smaller indices: with prob, resample
            coeff = rng.standard_normal(nmix)
            v = Modes[:, idx] @ coeff
            dirs.append(v)
        # line search over rotation scales
        for v in dirs:
            D = v.reshape(n, m)
            # map to ambient tangent displacements
            # dx_i = Bases[i] @ D[i]
            # compute efficiently
            # Bases: n x d x m, D: n x m -> dx: n x d
            dx = np.einsum('idm,im->id', Bases, D)
            nrms = np.linalg.norm(dx, axis=1)
            mxn = float(nrms.max())
            if mxn < 1e-12:
                continue
            for tt, th in zip(tan_thetas, thetas):
                s = tt / mxn
                y = best + s * dx
                y = _proj(y)
                val = _max_cos(y)
                # quick store; Hamming computed later only for promising
                cands.append((val, th, y))
                if time.time() > HARD_END - RESERVE_FINAL - 5.0:
                    break
            if time.time() > HARD_END - RESERVE_FINAL - 5.0:
                break
        if len(cands) == 0:
            continue
        # filter by max < 0.12
        filt = [t for t in cands if t[0] < 0.12]
        if len(filt) == 0:
            # relax to 0.20 to allow topology change
            filt = [t for t in cands if t[0] < 0.20]
            if len(filt) == 0:
                continue
        # compute Hamming distance for filt (thresholded graph vs incumbent)
        scored = []
        for val, th, y in filt:
            try:
                g = contact_graph(y, thr_ref)
                ham = int(np.count_nonzero(g != inc_graph) // 2)
            except Exception:
                ham = 0
            scored.append((val, -ham, th, y))
        # sort by true max then larger Hamming (i.e., smaller -ham? we want larger ham => smaller -ham)
        # primary val ascending, secondary ham descending => -ham ascending
        scored.sort(key=lambda t: (t[0], t[1]))
        # keep diverse: greedily select up to 6 with pairwise difference or just top with ham requirement
        selected = []
        for val, negham, th, y in scored:
            ham = -negham
            # require materially different topology unless val is excellent
            if val > best_val - 1e-6 and ham < 30:
                continue
            selected.append((val, ham, th, y))
            if len(selected) >= 7:
                break
        # if too strict, take top-3 anyway
        if len(selected) == 0:
            for val, negham, th, y in scored[:3]:
                selected.append((val, -negham, th, y))
        # LP-polish each selected for 3s (budget-aware)
        for val, ham, th, y in selected:
            time_left = (HARD_END - RESERVE_FINAL) - time.time()
            if time_left < 4.0:
                break
            polish_t = min(3.5, time_left / (len(selected) + 1))
            polish_t = max(polish_t, 1.5)
            try:
                yp = _lp_polish(y, seconds=polish_t)
            except Exception:
                continue
            vp = _max_cos(yp)
            if vp < best_val:
                best, best_val = yp, vp
                inc_graph = contact_graph(best, best_val - 1e-3)
                thr_ref = best_val - 1e-3
        # loop to recompute modes at new best
    # final polish of best with reserved time
    time_left = HARD_END - time.time()
    final_t = min(max(time_left - 1.0, 2.0), 22.0)
    if final_t > 0.5:
        try:
            best = _lp_polish(best, seconds=final_t)
        except Exception:
            pass
    best = _proj(best)
    return best
