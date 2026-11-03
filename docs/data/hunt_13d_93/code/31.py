import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

_IU0 = None
_IU1 = None

def _get_iu(n):
    global _IU0, _IU1
    if _IU0 is None or len(_IU0) != n * (n - 1) // 2:
        _IU0, _IU1 = np.triu_indices(n, 1)
    return _IU0, _IU1

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _rand_sphere(rng, n, d):
    A = rng.standard_normal(size=(n, d))
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    return A

def _perturb(X, sigma, rng):
    Y = X + sigma * rng.standard_normal(size=X.shape)
    return _norm(Y)

def _p_continuation(X0, steps_per_stage, p_list, lr0, lr1, shift, deadline=1e18):
    n, d = X0.shape
    iu0, iu1 = _get_iu(n)
    X = _norm(X0.copy())
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    total = len(p_list) * steps_per_stage
    tglob = 0
    best_seen = _max_cos(X)
    best_X = X.copy()
    # reusable buffer for W
    for pi, p in enumerate(p_list):
        m[:] = 0.0
        v[:] = 0.0
        tt = 0
        pm1 = float(p - 1.0)
        for s in range(steps_per_stage):
            if time.time() > deadline:
                break
            frac = tglob / max(1, total - 1)
            lr = lr0 * (lr1 / lr0) ** frac
            G = X @ X.T
            c = G[iu0, iu1]
            cmax = float(c.max())
            if cmax < best_seen:
                best_seen = cmax
                best_X = X.copy()
            # shifted weights
            # pos mask
            # manual loop to avoid extra allocs: c>shift
            pos = c > shift
            gt = None
            if np.any(pos):
                # sc for pos
                sc = c[pos] - shift
                smax = float(sc.max())
                if smax > 0 and np.isfinite(smax):
                    ratio = sc / smax
                    # clip for log
                    np.clip(ratio, 1e-300, 1.0, out=ratio)
                    logr = np.log(ratio)
                    e_pos = np.exp(pm1 * logr)
                    esum = float(e_pos.sum())
                    if esum > 0 and np.isfinite(esum):
                        w = np.zeros_like(c)
                        w[pos] = e_pos / esum
                        W = np.zeros((n, n), dtype=np.float64)
                        W[iu0, iu1] = w
                        W[iu1, iu0] = w
                        grad = W @ X
                        dot = np.einsum('ij,ij->i', grad, X)[:, None]
                        gt = grad - dot * X
            tglob += 1
            if gt is None:
                continue
            if not np.all(np.isfinite(gt)):
                continue
            ng = float(np.linalg.norm(gt))
            if ng < 1e-15:
                continue
            tt += 1
            m *= b1
            m += (1.0 - b1) * gt
            v *= b2
            v += (1.0 - b2) * (gt * gt)
            mh = m / (1.0 - b1 ** tt)
            vh = v / (1.0 - b2 ** tt)
            step = lr * mh / (np.sqrt(vh) + eps)
            X = X - step
            X = _norm(X)
        if time.time() > deadline:
            break
        cur = _max_cos(X)
        if cur < best_seen:
            best_seen = cur
            best_X = X.copy()
    # compare last vs best
    try:
        cur_last = _max_cos(X)
    except Exception:
        cur_last = 1e9
    if best_seen <= cur_last:
        return best_X, best_seen
    else:
        return X, cur_last

def _hard_polish_topK(X, iters, K0, K1, step0, deadline=1e18):
    n, d = X.shape
    iu0, iu1 = _get_iu(n)
    P = len(iu0)
    Xc = _norm(X.copy())
    best = _max_cos(Xc)
    bestX = Xc.copy()
    step = float(step0)
    for it in range(iters):
        if (it % 40 == 0) and time.time() > deadline:
            break
        frac = it / max(1, iters - 1)
        K = int(round(K0 + (K1 - K0) * frac))
        K = max(1, min(K, P))
        G = Xc @ Xc.T
        c = G[iu0, iu1]
        if K < P:
            idx = np.argpartition(-c, K)[:K]
        else:
            idx = np.arange(P)
        I = iu0[idx]
        J = iu1[idx]
        grad = np.zeros_like(Xc)
        np.add.at(grad, I, Xc[J])
        np.add.at(grad, J, Xc[I])
        grad /= float(len(idx))
        dot = np.einsum('ij,ij->i', grad, Xc)[:, None]
        gt = grad - dot * Xc
        rn = np.linalg.norm(gt, axis=1, keepdims=True)
        nz = rn[:, 0] > 1e-14
        if not np.any(nz):
            # tiny random tangent kick
            break
        direc = np.zeros_like(gt)
        direc[nz] = gt[nz] / rn[nz]
        Xt = _norm(Xc - step * direc)
        vv = _max_cos(Xt)
        if vv < best:
            best = vv
            bestX = Xt.copy()
            Xc = Xt
            step = min(step * 1.25, 0.006)
        else:
            step *= 0.55
            if step < 1e-12:
                step = float(step0) * 0.25
                # shake slightly to escape
                # small random tangent move on worst point
                Gm = Xc @ Xc.T
                np.fill_diagonal(Gm, -2.0)
                per = Gm.max(axis=1)
                wi = int(np.argmax(per))
                # random tangent
                rr = np.random.standard_normal(d)
                rr -= float(rr @ Xc[wi]) * Xc[wi]
                nn = float(np.linalg.norm(rr))
                if nn > 1e-12:
                    rr /= nn
                    Xt2 = Xc.copy()
                    Xt2[wi] = _norm((Xc[wi] - 1e-4 * rr)[None, :])[0]
                    # keep Xc unchanged, just reset step
                    pass
                if step < 1e-13:
                    step = float(step0) * 0.25
    return bestX, best

def _lp_polish(x, seconds):
    x = _norm(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        cfull = (x @ x.T)[iu]
        m = float(cfull.max())
        sel = cfull > m - max(20.0 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 5000:
            # keep most tight
            idx = np.argpartition(-cfull, 5000)[:5000]
            msk = np.zeros_like(sel)
            msk[idx] = True
            sel = msk
            I, J = iu[0][sel], iu[1][sel]
            p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cfull[sel], A_eq=A_eq,
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
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    BUDGET = 168.0
    deadline_final = t0 + BUDGET
    rng = np.random.default_rng(0)
    np.random.seed(0)

    Xb = None
    best_val = 1e9
    x_best = None
    for fn in ("best.npy", "record.npy"):
        try:
            A = np.load(fn)
            A = np.asarray(A, dtype=float)
            if A.shape != (93, 13):
                continue
            A = _norm(A)
            v = _max_cos(A)
            if v < best_val:
                best_val = v
                x_best = A.copy()
        except Exception:
            pass
    if x_best is None:
        x_best = _rand_sphere(rng, 93, 13)
        best_val = _max_cos(x_best)
    Xbest0 = x_best.copy()
    # try load record separately for second start
    Xrec = None
    try:
        A = np.load("record.npy")
        A = _norm(np.asarray(A, dtype=float))
        if A.shape == (93, 13):
            Xrec = A.copy()
    except Exception:
        Xrec = None
    if Xrec is None:
        Xrec = Xbest0.copy()

    pool = []
    pool.append((best_val, x_best.copy()))

    p_list = [8, 16, 32, 64, 128, 256]

    # build starts
    starts = []
    starts.append((Xbest0, 0.15, 3e-3, 3e-4, 350))
    starts.append((Xrec, 0.15, 3e-3, 3e-4, 350))
    sigmas = [0.15, 0.3, 0.6, 0.25, 0.4, 0.15, 0.3, 0.5]
    shifts = [0.15, 0.12, 0.18, 0.10, 0.15, 0.18, 0.12, 0.15]
    for k, sig in enumerate(sigmas):
        Xs = _perturb(Xbest0, sig, rng)
        starts.append((Xs, shifts[k % len(shifts)], 3e-3, 3e-4, 280))
    # random starts
    for k in range(3):
        Xr = _rand_sphere(rng, 93, 13)
        starts.append((Xr, [0.15, 0.10, 0.18][k % 3], 4e-3, 3e-4, 280))

    # phase A: continuations
    # reserve ~100s for phase A, but adapt to budget
    phaseA_end = t0 + 105.0
    for si, (Xs, shift, lr0, lr1, sps) in enumerate(starts):
        if time.time() > phaseA_end or time.time() > deadline_final - 45:
            break
        # adjust steps if low time
        remain = deadline_final - 45 - time.time()
        if remain < 5:
            break
        try:
            Xout, vout = _p_continuation(Xs, sps, p_list, lr0, lr1, shift,
                                         deadline=min(phaseA_end, deadline_final - 40))
        except Exception:
            continue
        pool.append((float(vout), Xout.copy()))
        if vout < best_val and np.all(np.isfinite(Xout)):
            best_val = float(vout)
            x_best = Xout.copy()
        # second wave: if improved a lot, add extra perturbed start from new best
        # (deterministic, limited)
        if si == 1 and best_val < 0.244:
            # inject one more start from current best with small sigma
            try:
                Xs2 = _perturb(x_best, 0.2, rng)
                Xout2, vout2 = _p_continuation(Xs2, 250, p_list, 3e-3, 3e-4, 0.13,
                                               deadline=min(phaseA_end, deadline_final - 40))
                pool.append((float(vout2), Xout2.copy()))
                if vout2 < best_val:
                    best_val = float(vout2)
                    x_best = Xout2.copy()
            except Exception:
                pass

    # extra wave from improved best if time remains
    if time.time() < phaseA_end - 10 and time.time() < deadline_final - 50:
        for sig, sh in [(0.2, 0.14), (0.35, 0.16), (0.12, 0.12)]:
            if time.time() > phaseA_end or time.time() > deadline_final - 45:
                break
            Xs = _perturb(x_best, sig, rng)
            try:
                Xout, vout = _p_continuation(Xs, 260, p_list, 3e-3, 3e-4, sh,
                                             deadline=min(phaseA_end, deadline_final - 40))
            except Exception:
                continue
            pool.append((float(vout), Xout.copy()))
            if vout < best_val:
                best_val = float(vout)
                x_best = Xout.copy()

    # select top 3
    pool = sorted(pool, key=lambda t: t[0])[:12]
    # deduplicate roughly? keep first occurrences
    cands = []
    for v, X in pool:
        if len(cands) >= 3:
            break
        # check distinctness
        ok = True
        for _, Y in cands:
            if float(np.linalg.norm(X - Y)) < 1e-6 or float(np.linalg.norm(X + Y)) < 1e-6:
                ok = False
                break
        if ok:
            cands.append((v, X.copy()))
    if len(cands) == 0:
        cands = [(best_val, x_best.copy())]

    # phase B: hard polish
    polished = []
    for ci, (v, X) in enumerate(cands):
        if time.time() > deadline_final - 30:
            break
        # vary K schedule per candidate
        if ci == 0:
            K0, K1, iters, s0 = 900, 25, 2600, 0.004
        elif ci == 1:
            K0, K1, iters, s0 = 500, 15, 2200, 0.003
        else:
            K0, K1, iters, s0 = 700, 40, 2000, 0.005
        try:
            Xp, vp = _hard_polish_topK(X, iters, K0, K1, s0,
                                       deadline=deadline_final - 28)
        except Exception:
            Xp, vp = X, v
        polished.append((float(vp), Xp.copy()))
        if vp < best_val:
            best_val = float(vp)
            x_best = Xp.copy()
    # also polish current best once more with margin-like small K if time
    if time.time() < deadline_final - 32:
        try:
            Xp, vp = _hard_polish_topK(x_best, 1500, 300, 8, 0.002,
                                       deadline=deadline_final - 28)
            if vp < best_val:
                best_val = float(vp)
                x_best = Xp.copy()
            polished.append((float(vp), Xp.copy()))
        except Exception:
            pass

    # phase C: LP polish remaining time
    remain = deadline_final - time.time()
    if remain > 4:
        try:
            Xlp = _lp_polish(x_best, min(remain - 1.0, 25.0))
            vlp = _max_cos(Xlp)
            if vlp < best_val and np.all(np.isfinite(Xlp)):
                best_val = vlp
                x_best = Xlp.copy()
        except Exception:
            pass

    x_best = _norm(x_best)
    # feasibility fix
    for i in range(93):
        nrm = float(np.linalg.norm(x_best[i]))
        if not np.isfinite(nrm) or nrm < 1e-12:
            x_best[i] = _rand_sphere(rng, 1, 13)[0]
    x_best = _norm(x_best)
    return x_best
