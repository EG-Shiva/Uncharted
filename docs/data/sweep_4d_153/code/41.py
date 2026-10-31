import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from scipy.linalg import null_space


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def lp_polish(x, seconds, radius_init=1e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = radius_init
    deadline = time.time() + seconds
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        I = iu[0][sel]
        J = iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], axis=1
        ).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        r_eq = np.repeat(np.arange(n), d)
        c_eq = np.arange(n * d)
        A_eq = coo_matrix((x.ravel(), (r_eq, c_eq)), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(
                obj, A_ub=A_ub, b_ub=-c[sel],
                A_eq=A_eq, b_eq=np.zeros(n),
                bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                method="highs",
            )
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def tangent_bases(x):
    n = x.shape[0]
    T = np.empty((n, 4, 3))
    for i in range(n):
        ns = null_space(x[i:i+1, :])
        # ns is 4x3 orthonormal
        if ns.shape != (4, 3):
            # fallback via QR
            q, _ = np.linalg.qr(np.random.randn(4, 4))
            # project
            v = q - np.outer(x[i], x[i] @ q)
            q2, _ = np.linalg.qr(v[:, 1:])
            ns = q2[:, :3]
        T[i] = ns
    return T


def floppy_excursion(x, rng, tol=1e-7):
    n = x.shape[0]
    N = n * 3
    G0 = x @ x.T
    np.fill_diagonal(G0, -2.0)
    m0 = float(G0.max())
    iu0 = np.triu_indices(n, 1)
    c0 = G0[iu0]
    is_contact = c0 > m0 - 1e-9
    Iall = iu0[0][is_contact]
    Jall = iu0[1][is_contact]
    p_all = len(Iall)
    if p_all < 10:
        return None
    # contact lookup for stopping: set of contact pairs
    contact_set = set(zip(Iall.tolist(), Jall.tolist()))

    # try several deletion candidates to get 1..5 dim nullspace
    for attempt in range(25):
        mode_cluster = bool(rng.random() < 0.5)
        del_mask = np.zeros(p_all, dtype=bool)
        if mode_cluster:
            k = int(4 + rng.integers(0, 5))  # 4..8
            seed = int(rng.integers(0, n))
            sims = x @ x[seed]
            # nearest neighbours (largest dot)
            nn = np.argsort(-sims)
            # take first k (includes seed itself)
            cluster = set(nn[:k].tolist())
            for r in range(p_all):
                if Iall[r] in cluster or Jall[r] in cluster:
                    del_mask[r] = True
            # ensure not deleting everything / nothing
            nd = int(del_mask.sum())
            if nd < 5 or nd > 80:
                # adjust: if too many, subsample deletions
                if nd > 80:
                    idx = np.where(del_mask)[0]
                    keep = rng.choice(idx, size=nd - 40, replace=False)
                    del_mask[keep] = False
                else:
                    continue
        else:
            t = int(10 + rng.integers(0, 11))  # 10..20
            t = min(t, p_all - 5)
            idx = rng.choice(p_all, size=t, replace=False)
            del_mask[idx] = True
        ret_I = Iall[~del_mask]
        ret_J = Jall[~del_mask]
        pr = len(ret_I)
        if pr <= 0:
            continue
        # build tangent bases at current x
        T = tangent_bases(x)
        Jr = np.empty((pr, N))
        # vectorized fill per row loop (pr~450, fast)
        for r in range(pr):
            i = int(ret_I[r]); j = int(ret_J[r])
            Jr[r, i*3:(i+1)*3] = T[i].T @ x[j]
            Jr[r, j*3:(j+1)*3] = T[j].T @ x[i]
        try:
            # full SVD to get nullspace dim
            _, S, Vh = np.linalg.svd(Jr, full_matrices=True)
        except Exception:
            continue
        rank = int((S > tol).sum())
        dim = N - rank
        if dim < 1 or dim > 5:
            continue
        # random nullspace combination
        nulls = Vh[N-dim:, :]  # dim x N
        coeff = rng.standard_normal(dim)
        nc = np.linalg.norm(coeff)
        if nc == 0:
            continue
        coeff /= nc
        dy = coeff @ nulls  # N
        dy3 = dy.reshape(n, 3)
        dx = np.empty((n, 4))
        for i in range(n):
            dx[i] = T[i] @ dy3[i]
        mx = np.linalg.norm(dx, axis=1).max()
        if mx < 1e-12 or not np.isfinite(mx):
            continue
        dx /= mx  # max point speed = 1
        # integrate with renormalisation, micro steps
        micro = 0.004
        total_target = float(rng.uniform(0.02, 0.055))
        nsteps = int(total_target / micro)
        nsteps = max(4, min(nsteps, 14))
        xc = x.copy()
        # precompute inactive mask for stopping
        # inactive pairs: all triu not in contact_set
        for s in range(nsteps):
            xc = xc + micro * dx
            xc /= np.linalg.norm(xc, axis=1, keepdims=True)
            G = xc @ xc.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            # stopping 1: a previously inactive pair reached current max
            # find max over inactive pairs
            # cheap: check if argmax pair is inactive
            # get top pair
            flat = np.argmax(G)
            ai, aj = divmod(flat, n)
            if ai > aj:
                ai, aj = aj, ai
            # if top pair not in original contacts -> new topology hit
            if (ai, aj) not in contact_set:
                # stop if it is within 1e-9 of m0 or exceeds
                if m >= m0 - 1e-9:
                    break
            # stopping 2: deleted contact rose appreciably (opened gap)
            # compute deleted contacts values
            # sample: if max retained still ~m but deleted well below
            # break if any deleted contact dropped by >2e-4
            # evaluate quickly every 3 steps
            if (s % 3 == 2):
                vals_del = np.array([xc[Iall[k]] @ xc[Jall[k]] for k in np.where(del_mask)[0][:20]])
                if vals_del.size > 0 and float(vals_del.max()) < m0 - 5e-4:
                    break
        # ensure we actually moved and didn't blow up
        Gf = xc @ xc.T
        np.fill_diagonal(Gf, -2.0)
        mf = float(Gf.max())
        if mf > m0 + 0.01:
            # too big a jump, shorten (reject)
            continue
        # displacement check
        disp = float(np.linalg.norm(xc - x, axis=1).max())
        if disp < 0.005:
            continue
        return xc
    return None


def solve():
    t_start = time.time()
    TIME_LIMIT = 172.0
    FINAL_POLISH = 20.0
    rng = np.random.default_rng(153)
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy")
    except Exception:
        xr = None
    cands = []
    if xb is not None:
        cands.append(np.asarray(xb, float))
    if xr is not None:
        cands.append(np.asarray(xr, float))
    if not cands:
        z = rng.standard_normal((153, 4))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        cands.append(z)
    best = None
    best_val = 1e9
    for c in cands:
        c = c / np.linalg.norm(c, axis=1, keepdims=True)
        v = max_cos(c)
        if v < best_val:
            best_val = v
            best = c.copy()
    # quick initial polish very short to stabilize
    # main loop: floppy excursions + LP polish
    it = 0
    fails = 0
    while time.time() - t_start < TIME_LIMIT - FINAL_POLISH:
        it += 1
        y = floppy_excursion(best, rng)
        if y is None:
            fails += 1
            # fallback: small cluster shake (still topology-changing attempt)
            y = best.copy()
            k = int(rng.integers(4, 9))
            seed = int(rng.integers(0, 153))
            sims = y @ y[seed]
            nn = np.argsort(-sims)[:k]
            y[nn] += 0.03 * rng.standard_normal((k, 4))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        # alternate 2-5 s LP polishes (use 2.2s to allow many trials)
        polish_s = 2.2 if (it % 3 != 0) else 3.5
        # cap by remaining time
        remain = TIME_LIMIT - FINAL_POLISH - (time.time() - t_start)
        if remain < 0.5:
            break
        polish_s = min(polish_s, remain)
        y2 = lp_polish(y, seconds=polish_s)
        v2 = max_cos(y2)
        if v2 < best_val - 1e-12:
            best = y2
            best_val = v2
            fails = 0
        # else keep incumbent as fallback (do nothing), continue testing
        # different deleted clusters/contact sets
        if time.time() - t_start > TIME_LIMIT - FINAL_POLISH - 0.5:
            break
    remain = TIME_LIMIT - (time.time() - t_start) - 1.0
    if remain < 1.0:
        remain = 1.0
    remain = min(remain, 178.0 - (time.time() - t_start) - 1.0)
    if remain > 0.5:
        best = lp_polish(best, seconds=remain)
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
