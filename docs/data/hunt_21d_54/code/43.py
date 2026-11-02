import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _proj(x):
    nrm = np.linalg.norm(x, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return x / nrm

def _lp_polish(x, seconds, cap=2500):
    x = _proj(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p > cap:
            idx = np.argpartition(-c[sel], cap)[:cap]
            I, J = I[idx], J[idx]
            csv = c[sel][idx]
        else:
            csv = c[sel]
            if p == 0:
                break
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
                          method="highs",
                          options={"time_limit": 2.0})
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
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x

def _tangent_bases(x):
    n, d = x.shape
    dm = d - 1
    T = np.zeros((n, d, dm))
    idx_all = np.arange(d)
    for i in range(n):
        xi = x[i]
        k = int(np.argmax(np.abs(xi)))
        keep = idx_all[idx_all != k]
        B = np.zeros((d, dm))
        # columns = e_l for l != k
        B[keep, np.arange(dm)] = 1.0
        s = xi[keep]  # xi^T B
        Btan = B - xi[:, None] * s[None, :]
        Q, _ = np.linalg.qr(Btan)
        T[i] = Q
    return T

def _null_modes(x, T, thresh=1e-3, cap=2500, nmode=30):
    n, d = x.shape
    dm = d - 1
    N = n * dm
    G = x @ x.T
    iu = np.triu_indices(n, 1)
    c = G[iu]
    m = float(c.max())
    sel = c > m - thresh
    I0, J0 = iu[0][sel], iu[1][sel]
    p0 = len(I0)
    if p0 == 0:
        return None, None, m
    if p0 > cap:
        idx = np.argpartition(-c[sel], cap)[:cap]
        I, J = I0[idx], J0[idx]
        p = cap
    else:
        I, J = I0, J0
        p = p0
    # Build sparse Jacobian A (p x N)
    # row q: block i = Ti^T x_j, block j = Tj^T x_i
    rows = []
    cols = []
    data = []
    for q in range(p):
        i = int(I[q]); j = int(J[q])
        Ti = T[i]  # d x dm
        Tj = T[j]
        vi = Ti.T @ x[j]  # dm
        vj = Tj.T @ x[i]
        base_i = i * dm
        base_j = j * dm
        for a in range(dm):
            if vi[a] != 0:
                rows.append(q); cols.append(base_i + a); data.append(float(vi[a]))
            if vj[a] != 0:
                rows.append(q); cols.append(base_j + a); data.append(float(vj[a]))
    try:
        A = coo_matrix((np.array(data), (np.array(rows), np.array(cols))), shape=(p, N)).tocsr()
    except Exception:
        return None, None, m
    # Try svds for smallest, fallback to eig of AtA
    Vsmall = None
    try:
        from scipy.sparse.linalg import svds
        k = int(min(nmode, min(p, N) - 1))
        if k >= 2:
            # svds with which='SM' can be slow; try with tight limit
            u, s, vt = svds(A, k=k, which='SM', maxiter=200, tol=1e-4)
            # svds returns sorted ascending? actually ascending; smallest first?
            order = np.argsort(s)
            vt = vt[order, :]
            # vt rows are right singular vectors; we want smallest -> already
            Vsmall = vt.T  # N x k
    except Exception:
        Vsmall = None
    if Vsmall is None:
        try:
            AtA = (A.T @ A).toarray()
            # symmetrize
            AtA = 0.5 * (AtA + AtA.T)
            w, V = np.linalg.eigh(AtA)
            # w ascending, take smallest nmode
            k = int(min(nmode, N))
            Vsmall = V[:, :k]
        except Exception:
            return None, None, m
    return Vsmall, A, m

def solve():
    T0 = time.time()
    HARD = 150.0
    FINAL_POLISH = 20.0
    # total budget ~ HARD+FINAL_POLISH < 180
    rng = np.random.default_rng(0)
    try:
        b0 = np.load("best.npy")
    except Exception:
        try:
            b0 = np.load("record.npy")
        except Exception:
            b0 = rng.normal(size=(54, 21))
    b0 = _proj(np.nan_to_num(np.asarray(b0, dtype=float)))
    if b0.shape != (54, 21):
        b0 = _proj(rng.normal(size=(54, 21)))
    try:
        r0 = _proj(np.asarray(np.load("record.npy"), dtype=float))
        if r0.shape != (54, 21):
            r0 = b0
    except Exception:
        r0 = b0
    bv = _max_cos(b0)
    rv = _max_cos(r0)
    if rv < bv:
        best, bv = r0.copy(), rv
    else:
        best, bv = b0.copy(), bv

    def tl():
        return HARD + FINAL_POLISH - (time.time() - T0)

    # incumbent contact mask for Hamming distance (threshold max-1e-3)
    def contact_mask_upper(x, thr):
        G = x @ x.T
        iu = np.triu_indices(x.shape[0], 1)
        return (G[iu] > thr)

    base_thr = bv - 1e-3
    base_mask = contact_mask_upper(best, base_thr)

    angles = [0.05, 0.1, 0.2, 0.35, 0.5]
    n, d = 54, 21
    dm = d - 1

    # quick ensure best is locally polished a bit (2s) to define Jacobian
    if tl() > 60:
        best = _lp_polish(best, 2.0)
        nbv = _max_cos(best)
        if nbv < bv:
            bv = nbv
            base_thr = bv - 1e-3
            base_mask = contact_mask_upper(best, base_thr)

    outer = 0
    # explore until HARD deadline
    while (time.time() - T0) < HARD - 8:
        outer += 1
        # rebuild tangent bases and modes at current best
        try:
            T = _tangent_bases(best)
        except Exception:
            break
        Vsmall, A, mcur = _null_modes(best, T, thresh=1e-3, cap=2500, nmode=30)
        if Vsmall is None:
            # fallback random tangent directions
            if (time.time() - T0) > HARD - 15:
                break
            # random shake + polish
            y = best + 0.05 * (_proj(rng.normal(size=(n, d))) * 0.1)
            y = _proj(y)
            y = _lp_polish(y, 3.0)
            v = _max_cos(y)
            if v < bv:
                best, bv = y, v
                base_thr = bv - 1e-3
                base_mask = contact_mask_upper(best, base_thr)
            continue
        nk = Vsmall.shape[1]
        # generate trials: random combos of 2-5 modes x angles
        ncombo = 6  # combos per outer
        trials = []
        for ci in range(ncombo):
            if (time.time() - T0) > HARD - 8:
                break
            kpick = int(rng.integers(2, 6))  # 2..5
            kpick = min(kpick, nk)
            idx = rng.choice(nk, size=kpick, replace=False)
            coeff = rng.normal(size=kpick)
            # combine
            vec = Vsmall[:, idx] @ coeff  # N
            Vmat = vec.reshape(n, dm)  # tangent coords
            # map to ambient
            Delta = np.einsum('ijk,ik->ij', T, Vmat)  # n x d
            # ensure tangent (numerical)
            Delta -= (np.einsum('ij,ij->i', Delta, best)[:, None] * best)
            norms = np.linalg.norm(Delta, axis=1)
            mx = float(norms.max())
            if mx < 1e-12:
                continue
            for ang in angles:
                scale = ang / mx
                y = best + scale * Delta
                y = _proj(y)
                G = y @ y.T
                np.fill_diagonal(G, -2.0)
                vm = float(G.max())
                if vm > 0.12:
                    continue
                # Hamming distance vs incumbent at base_thr
                iu = np.triu_indices(n, 1)
                tmask = (y @ y.T)[iu] > base_thr
                ham = int(np.sum(tmask != base_mask))
                trials.append((vm, -ham, ang, y, ham))
        if not trials:
            continue
        # Score trials first by true max-cosine, second by Hamming distance
        # sort by vm ascending, ham descending
        trials.sort(key=lambda t: (t[0], t[1]))
        # keep trials with materially different topology: ham > 50
        # but always consider top-2 even if ham small
        cand = []
        for (vm, negham, ang, y, ham) in trials:
            if ham > 50 or len(cand) < 2:
                cand.append((vm, ham, y))
            if len(cand) >= 3:
                break
        # LP-polish each for 3s
        for (vm, ham, y) in cand:
            if (time.time() - T0) > HARD - 4:
                break
            yp = _lp_polish(y, 3.0)
            vp = _max_cos(yp)
            if vp < bv - 1e-12:
                best, bv = yp, vp
                base_thr = bv - 1e-3
                base_mask = contact_mask_upper(best, base_thr)
            # early exit if big win
            if bv < 0.06:
                break
        # if time low break
        if tl() < FINAL_POLISH + 5:
            break

    # final LP polish of best with reserved time
    remain = FINAL_POLISH
    # don't exceed 178s total
    elapsed = time.time() - T0
    budget = min(remain, 178.0 - elapsed - 1.0)
    if budget > 1.0:
        best = _lp_polish(best, budget)
    best = _proj(best)
    return best
