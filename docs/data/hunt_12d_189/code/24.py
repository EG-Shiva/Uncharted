import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def smooth_opt(x, steps=260, a0=30.0, a1=280.0, eta0=0.05, deadline=None):
    n, d = x.shape
    x = x.copy()
    for s in range(steps):
        if deadline is not None and (s % 60 == 0) and time.time() > deadline:
            break
        frac = s / max(1, steps - 1)
        alpha = a0 * ((a1 / max(a0, 1e-9)) ** frac)
        eta = eta0 * (1.0 - 0.5 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        m = G.max()
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        S = W @ x
        wg = (W * G).sum(axis=1, keepdims=True)
        Tang = S - wg * x
        wsum = W.sum(axis=1, keepdims=True) + 1e-12
        step = Tang / wsum
        x = x - eta * step
        nrm = np.linalg.norm(x, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        x /= nrm
    return x

def lp_polish(x, deadline, max_pairs=500):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = max_cos(x)
    except Exception:
        return x
    radius = 8e-4
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline - 0.15 and radius > 1e-14:
        if time.time() > deadline - 0.35:
            break
        Gfull = x @ x.T
        c_full = Gfull[iu0]
        m = c_full.max()
        thresh = m - max(20 * radius, 1e-9)
        sel_idx = np.where(c_full > thresh)[0]
        if sel_idx.size == 0:
            radius *= 0.5
            continue
        if sel_idx.size > max_pairs:
            part = np.argpartition(c_full[sel_idx], -max_pairs)[-max_pairs:]
            sel_idx = sel_idx[part]
        I = iu0[0][sel_idx]
        J = iu0[1][sel_idx]
        p = len(I)
        b_ub = -c_full[sel_idx]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols_I = (I[:, None] * d + np.arange(d))
        cols_J = (J[:, None] * d + np.arange(d))
        cols = np.concatenate([cols_I, cols_J, np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq,
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
        nrm = np.linalg.norm(y, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        y /= nrm
        if not np.all(np.isfinite(y)):
            radius *= 0.5
            continue
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return x

def contact_set(x, tol=1e-6):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    iu = np.triu_indices(x.shape[0], 1)
    sel = np.where(G[iu] > m - tol)[0]
    s = set(zip(iu[0][sel].tolist(), iu[1][sel].tolist()))
    return s, m

def procrustes_rms(a, b):
    # optimal rotation aligning a to b, same index order
    try:
        M = a.T @ b
        U, _, Vt = np.linalg.svd(M, full_matrices=False)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        d = a @ R - b
        return float(np.linalg.norm(d) / np.sqrt(a.shape[0]))
    except Exception:
        return 1e9

def cluster_rotate_offspring(x, rng):
    n, d = x.shape
    y = x.copy()
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    adj = [np.where(G[i] > m - 1e-6)[0].tolist() for i in range(n)]
    # target size 5-15
    k = int(rng.integers(5, 16))
    seed = int(rng.integers(0, n))
    # BFS
    visited = set([seed])
    queue = [seed]
    # randomize neighbor order via rng
    qi = 0
    while len(visited) < k and qi < len(queue):
        cur = queue[qi]; qi += 1
        nbrs = adj[cur]
        if len(nbrs) == 0:
            continue
        perm = rng.permutation(len(nbrs))
        for pi in perm:
            nb = nbrs[pi]
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
                if len(visited) >= k:
                    break
        if len(visited) >= k:
            break
    if len(visited) < 5:
        # fallback: random connected via nearest neighbors at looser tol
        # just take k nearest to seed by cosine
        row = G[seed]
        nn = np.argsort(-row)[:k]
        visited = set(nn.tolist())
    cl = np.array(sorted(list(visited)), dtype=int)
    # random 2-plane
    A = rng.normal(size=(d, 2))
    Q, _ = np.linalg.qr(A)
    u = Q[:, 0]; v = Q[:, 1]
    theta = float(rng.uniform(0.05, 0.2)) * (1.0 if rng.random() < 0.5 else -1.0)
    c = np.cos(theta); s = np.sin(theta)
    for idx in cl:
        xi = y[idx]
        a = float(xi @ u); b = float(xi @ v)
        rest = xi - a * u - b * v
        a2 = a * c - b * s
        b2 = a * s + b * c
        y[idx] = rest + a2 * u + b2 * v
    nrm = np.linalg.norm(y, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    y /= nrm
    # tiny global jitter to break symmetry
    y = y + rng.normal(scale=0.0015, size=y.shape)
    y /= np.linalg.norm(y, axis=1, keepdims=True)
    return y

def jacobian_null_offspring(x, rng):
    n, d = x.shape
    N = n * d
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    iu = np.triu_indices(n, 1)
    c = G[iu]
    p_want = 350
    if c.size > p_want:
        sel = np.argpartition(c, -p_want)[-p_want:]
    else:
        sel = np.arange(c.size)
    I = iu[0][sel]; J = iu[1][sel]
    p = len(I)
    try:
        Jmat = np.zeros((p, N), dtype=float)
        # fill rows: block i gets x[J], block j gets x[I]
        for r in range(p):
            i = int(I[r]); j = int(J[r])
            Jmat[r, i * d:(i + 1) * d] = x[j]
            Jmat[r, j * d:(j + 1) * d] = x[i]
        # SVD, smallest singular vector
        _, _, Vt = np.linalg.svd(Jmat, full_matrices=False)
        # combine last 1-2 vectors randomly
        if Vt.shape[0] >= 2 and rng.random() < 0.5:
            w1 = float(rng.normal()); w2 = float(rng.normal())
            vec = w1 * Vt[-1] + w2 * Vt[-2]
        else:
            vec = Vt[-1]
        dx = vec.reshape(n, d)
        # project to tangent
        dot = (dx * x).sum(axis=1, keepdims=True)
        dx = dx - dot * x
        # scale: max row norm -> 0.05..0.12
        rn = np.linalg.norm(dx, axis=1)
        mx = float(rn.max())
        if not np.isfinite(mx) or mx < 1e-12:
            raise ValueError("degenerate")
        target = float(rng.uniform(0.05, 0.12))
        dx = dx * (target / mx)
        y = x + dx
        y = y + rng.normal(scale=0.001, size=y.shape)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        return y
    except Exception:
        # fallback random shake
        y = x + rng.normal(scale=0.03, size=x.shape)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        return y

def solve():
    t0 = time.time()
    HARD_END = t0 + 174.0
    EXPLORE_END = t0 + 140.0
    rng = np.random.default_rng(12345)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    for cc in (xb, xr):
        if cc is not None and cc.shape == (189, 12) and np.all(np.isfinite(cc)):
            nrm = np.linalg.norm(cc, axis=1, keepdims=True)
            nrm[nrm == 0] = 1.0
            cands.append(cc / nrm)
    if len(cands) == 0:
        r0 = np.random.default_rng(0)
        cc = r0.normal(size=(189, 12))
        cc /= np.linalg.norm(cc, axis=1, keepdims=True)
        cands.append(cc)
    best_x = cands[0]
    best_v = max_cos(best_x)
    for cc in cands[1:]:
        v = max_cos(cc)
        if v < best_v:
            best_x, best_v = cc.copy(), v
    # archive of up to 6
    archive_X = [best_x.copy()]
    archive_V = [best_v]
    archive_C = []
    cs, _ = contact_set(best_x, tol=1e-6)
    archive_C.append(cs)
    # add a couple of slightly perturbed copies to seed diversity if time allows? keep 1 for now
    # SA temperature schedule time-based
    def temperature(now):
        frac = (now - t0) / max(1e-9, (EXPLORE_END - t0))
        frac = min(1.0, max(0.0, frac))
        # log decay 1e-4 -> 1e-6
        return 1e-4 * (1e-2 ** frac)
    it = 0
    use_cluster = True
    while time.time() < EXPLORE_END - 2.5:
        if it > 200:
            break
        now = time.time()
        T = temperature(now)
        # pick parent: 50% best, else uniform random
        idx_best = int(np.argmin(archive_V))
        if len(archive_X) == 1 or rng.random() < 0.5:
            pidx = idx_best
        else:
            pidx = int(rng.integers(0, len(archive_X)))
        parent = archive_X[pidx]
        # generate offspring
        try:
            if use_cluster:
                y0 = cluster_rotate_offspring(parent, rng)
            else:
                y0 = jacobian_null_offspring(parent, rng)
        except Exception:
            y0 = parent + rng.normal(scale=0.02, size=parent.shape)
            y0 /= np.linalg.norm(y0, axis=1, keepdims=True)
        use_cluster = not use_cluster
        # smooth briefly
        smooth_dead = min(EXPLORE_END - 1.5, time.time() + 1.0)
        # vary sharpness a bit
        a1 = float(rng.uniform(220.0, 320.0))
        y1 = smooth_opt(y0, steps=260, a0=30.0, a1=a1, eta0=0.05, deadline=smooth_dead)
        if time.time() > EXPLORE_END - 1.8:
            # evaluate without LP
            try:
                v1 = max_cos(y1)
            except Exception:
                it += 1
                continue
            # SA accept check vs best
            vbest = float(min(archive_V))
            accept = (v1 < vbest) or (rng.random() < np.exp(-(v1 - vbest) / max(T, 1e-12)))
            if accept and np.all(np.isfinite(y1)):
                # distinctness
                cs1, _ = contact_set(y1, tol=1e-6)
                dup = False
                for ec in archive_C:
                    if ec == cs1:
                        dup = True
                        break
                if not dup:
                    # rms check
                    too_close = False
                    for ex in archive_X:
                        if procrustes_rms(y1, ex) < 0.03:
                            too_close = True
                            break
                    if not too_close:
                        if len(archive_X) < 6:
                            archive_X.append(y1); archive_V.append(v1); archive_C.append(cs1)
                        else:
                            wi = int(np.argmax(archive_V))
                            if v1 < archive_V[wi] or True:
                                # replace worst even if slightly worse to allow drift, but guard huge degradation
                                if v1 < archive_V[wi] + 5 * T + 1e-4:
                                    archive_X[wi] = y1; archive_V[wi] = v1; archive_C[wi] = cs1
            it += 1
            continue
        # briefly LP-polish, cap 2s (use 1.4s)
        lp_dead = min(EXPLORE_END - 0.8, time.time() + 1.4)
        try:
            y2 = lp_polish(y1, deadline=lp_dead, max_pairs=450)
        except Exception:
            y2 = y1
        try:
            v2 = max_cos(y2)
        except Exception:
            it += 1
            continue
        if not np.all(np.isfinite(y2)):
            it += 1
            continue
        vbest = float(min(archive_V))
        if v2 < vbest - 1e-12:
            accept = True
        else:
            try:
                prob = float(np.exp(-(v2 - vbest) / max(T, 1e-12)))
            except Exception:
                prob = 0.0
            accept = (rng.random() < prob)
        if accept:
            cs2, _ = contact_set(y2, tol=1e-6)
            dup = False
            for ec in archive_C:
                if ec == cs2:
                    dup = True
                    break
            if dup:
                it += 1
                continue
            too_close = False
            for ex in archive_X:
                if procrustes_rms(y2, ex) < 0.03:
                    too_close = True
                    break
            if too_close:
                it += 1
                continue
            if len(archive_X) < 6:
                archive_X.append(y2.copy()); archive_V.append(v2); archive_C.append(cs2)
            else:
                wi = int(np.argmax(archive_V))
                # allow replacement if not catastrophically worse
                if v2 < archive_V[wi] + 5 * T + 1e-4:
                    archive_X[wi] = y2.copy(); archive_V[wi] = v2; archive_C[wi] = cs2
                elif v2 < vbest:
                    archive_X[wi] = y2.copy(); archive_V[wi] = v2; archive_C[wi] = cs2
        it += 1
        # extra clock check
        if time.time() > EXPLORE_END - 1.0:
            break
    # pick best in archive
    bi = int(np.argmin(archive_V))
    best_x = archive_X[bi].copy()
    # final polish with remaining time
    remain = HARD_END - time.time()
    if remain > 2.5:
        fin_dead = time.time() + min(remain - 0.7, 30.0)
        # two-stage: a bit of smoothing then LP
        try:
            best_x = smooth_opt(best_x, steps=200, a0=30.0, a1=320.0, eta0=0.03, deadline=fin_dead - 4.0)
        except Exception:
            pass
        try:
            best_x = lp_polish(best_x, deadline=fin_dead, max_pairs=800)
        except Exception:
            pass
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
