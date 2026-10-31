import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _lp_polish(x, seconds, r0=2e-3, cap=1300):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    iu0, iu1 = np.triu_indices(n, 1)
    cd = np.arange(d)
    radius = r0
    start = time.time()
    while time.time() - start < seconds and radius > 1e-13:
        # time guard per iteration
        G = x @ x.T
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = np.where(c_all > thr)[0]
        if sel.size == 0:
            break
        if sel.size > cap:
            v = c_all[sel]
            kth = np.sort(v)[-cap]
            sel = np.where(c_all >= kth)[0]
            if sel.size > cap + 300:
                sel = sel[::2][:cap]
        I = iu0[sel]; J = iu1[sel]
        c = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + cd, J[:, None] * d + cd,
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
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
            radius = min(radius * 1.6, 0.02)
        else:
            radius *= 0.5
    return x

def _tangent_bases(x):
    n, d = x.shape
    assert d == 4
    T = np.empty((n, 4, 3))
    E = np.eye(4)
    for i in range(n):
        xi = x[i]
        m = int(np.argmax(np.abs(xi)))
        cols = [k for k in range(4) if k != m]
        M = np.stack([E[:, k] - xi[k] * xi for k in cols], axis=1)  # 4x3
        Q, _ = np.linalg.qr(M)
        T[i] = Q
    return T

def _rotation_modes(T, x):
    n = x.shape[0]
    N = n * 3
    R = []
    for a in range(4):
        for b in range(a + 1, 4):
            A = np.zeros((4, 4))
            A[a, b] = 1.0
            A[b, a] = -1.0
            # dx_i = A x_i
            V = np.empty(N)
            for i in range(n):
                dxi = A @ x[i]
                V[i * 3:(i + 1) * 3] = T[i].T @ dxi
            R.append(V)
    R = np.stack(R, axis=1)  # N x 6
    Q, _ = np.linalg.qr(R)
    return Q  # N x 6 orthonormal

def _build_rigidity(x, T, I, J):
    p = len(I)
    n = x.shape[0]
    N = n * 3
    Rmat = np.empty((p, N))
    # precompute T^T x for speed? loop is fine (p~500)
    for r in range(p):
        a = int(I[r]); b = int(J[r])
        ca = T[a].T @ x[b]
        cb = T[b].T @ x[a]
        row = np.zeros(N)
        row[a * 3:(a + 1) * 3] = ca
        row[b * 3:(b + 1) * 3] = cb
        Rmat[r] = row
    return Rmat

def solve():
    rng = np.random.default_rng(0)
    t0 = time.time()
    HARD = 168.0
    FINAL_RESERVE = 20.0

    n, d = 163, 4
    xb = None; xr = None
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy")
    except Exception:
        xr = None
    cands = []
    for c in (xb, xr):
        try:
            if c is not None:
                c = np.asarray(c, dtype=float)
                if c.shape == (163, 4):
                    c = _norm(np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0))
                    # guard zero rows
                    nr = np.linalg.norm(c, axis=1)
                    if np.all(nr > 1e-9) and np.all(np.isfinite(c)):
                        cands.append(c)
        except Exception:
            pass
    if not cands:
        z = rng.normal(size=(163, 4))
        cands.append(_norm(z))
    best = None; best_val = 1e9
    for c in cands:
        v = _max_cos(c)
        if v < best_val:
            best_val = v; best = c.copy()
    incumbent = best.copy()
    incumbent_val = best_val

    iu0, iu1 = np.triu_indices(n, 1)

    def contact_info(x):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        c_all = G[iu0, iu1]
        slack = m - c_all
        sel9 = np.where(slack < 1e-9)[0]
        # second shell: slack in [1e-9, 3e-3] roughly, use for exchange detection
        return m, c_all, slack, sel9

    # build nullspace pool from incumbent/best
    def null_pool(x):
        m, c_all, slack, sel9 = contact_info(x)
        I = iu0[sel9]; J = iu1[sel9]
        # fallback thresholds if too few contacts
        if len(I) < 50:
            sel7 = np.where(slack < 1e-7)[0]
            I = iu0[sel7]; J = iu1[sel7]
        if len(I) < 10:
            # take top 400
            o = np.argsort(slack)[:400]
            I = iu0[o]; J = iu1[o]
        T = _tangent_bases(x)
        Rmat = _build_rigidity(x, T, I, J)
        Qr = _rotation_modes(T, x)
        p, N = Rmat.shape
        try:
            # full Vt to get nullspace
            U, S, Vt = np.linalg.svd(Rmat, full_matrices=True)
            # Vt shape N x N
        except Exception:
            return None, None, None, m, sel9, I, J
        K = min(24, N)
        # bottom K rows of Vt
        Vb = Vt[N - K:N, :].T.copy()  # N x K
        # project out rotations
        # Vb - Qr(Qr^T Vb)
        coeff = Qr.T @ Vb  # 6 x K
        Vb = Vb - Qr @ coeff
        # renormalize, drop tiny
        norms = np.linalg.norm(Vb, axis=0, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        Vb = Vb / norms
        # drop modes that were mostly rotation (original norm after proj small)
        # recompute raw norm factor: we already normalized; instead check projection energy
        # keep all, random combos will average out
        return Vb, T, Qr, m, sel9, I, J

    Vb, Tcur, Qr, mcur, sel9_cur, Ic, Jc = null_pool(best)
    if Vb is None:
        # fallback: just polish
        rem = HARD - (time.time() - t0)
        rem = max(rem - 2.0, 2.0)
        best = _lp_polish(best, seconds=min(rem, 25.0))
        return _norm(best)

    active_set = set(sel9_cur.tolist())
    K = Vb.shape[1]
    # precompute direction scale helper
    ALPHAS = np.array([0.004, 0.008, 0.015, 0.025, 0.04, 0.06, 0.09])

    n_modes = 32
    for mode in range(n_modes):
        elapsed = time.time() - t0
        if elapsed > HARD - FINAL_RESERVE - 3.0:
            break
        # sample random combo of near-null modes
        w = rng.normal(size=K)
        # emphasize smallest singular directions: weight last modes more
        # Vb columns ordered? Vt bottom rows: last row is most null. Our Vb columns 0..K-1 correspond to increasing nullity.
        # boost latter half
        boost = np.linspace(0.5, 1.5, K)
        w = w * boost
        w /= max(np.linalg.norm(w), 1e-12)
        v = Vb @ w  # N
        # map to 3D displacements
        Y = v.reshape(n, 3)
        D = np.empty((n, 4))
        for i in range(n):
            D[i] = Tcur[i] @ Y[i]
        # scale so max point norm = 1
        dn = np.linalg.norm(D, axis=1).max()
        if dn < 1e-12:
            continue
        D /= dn
        # line search
        found = None
        # cache active set for quick check; also need iu index mapping
        # for speed compute full Gram per alpha (163x163 cheap)
        for alpha in ALPHAS:
            Xt = _norm(best + alpha * D)
            G = Xt @ Xt.T
            np.fill_diagonal(G, -2.0)
            mt = float(G.max())
            if mt > best_val + 2e-4:
                break
            # check for newly active pair: any pair not in active_set with cosine > mt - 1e-7 becoming co-maximal?
            # use threshold 5e-7 to declare active (slightly looser for detection)
            # find indices of pairs within 1e-7 of mt
            # to avoid full scan of 13k pairs: it's cheap (13k), do it
            c_all_t = G[iu0, iu1]
            # top pairs
            # if mt improved a lot, also consider? we want topology change
            close = np.where(c_all_t > mt - 1e-7)[0]
            # if any close not in active_set -> exchange
            # quick check via set difference; close size small
            is_new = False
            for k in close:
                if k not in active_set:
                    is_new = True
                    break
            if is_new:
                found = Xt
                break
        if found is None:
            continue
        # short trust-region LP polish on exchanged state
        # strict time check
        remain = HARD - FINAL_RESERVE - (time.time() - t0)
        if remain < 2.0:
            break
        secs = min(2.2, remain)
        polished = _lp_polish(found, seconds=secs, r0=2e-3, cap=1200)
        pv = _max_cos(polished)
        if pv < best_val - 1e-12:
            best = polished
            best_val = pv
            # rebuild nullspace around new best
            try:
                Vb2, T2, Qr2, m2, sel9_2, Ic2, Jc2 = null_pool(best)
                if Vb2 is not None:
                    Vb, Tcur, Qr = Vb2, T2, Qr2
                    mcur = m2
                    active_set = set(sel9_2.tolist())
                    K = Vb.shape[1]
            except Exception:
                active_set = set(sel9_cur.tolist())
            # update ALPHAS tolerance uses new best_val automatically
        # else discard (retain only if improves)

    # final reserve polish on best found
    elapsed = time.time() - t0
    remain = HARD - elapsed - 1.0
    if remain > 1.0:
        secs = min(remain, FINAL_RESERVE + 5.0)
        # polish best; if it fails to improve keep it anyway
        try:
            pb = _lp_polish(best, seconds=secs, r0=2e-3, cap=1500)
            pv = _max_cos(pb)
            if pv < best_val:
                best, best_val = pb, pv
        except Exception:
            pass
    # fallback to incumbent if no improvement (ensure no regression)
    if best_val > incumbent_val and incumbent is not None:
        best = incumbent
    return _norm(best)
