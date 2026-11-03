import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def smooth_refine(X, steps, T0, T1, lr0, lr1, mom=0.88):
    n, d = X.shape
    V = np.zeros_like(X)
    logT0 = np.log(T0)
    logT1 = np.log(T1)
    for s in range(steps):
        frac = s / (steps - 1) if steps > 1 else 1.0
        T = np.exp(logT0 * (1.0 - frac) + logT1 * frac)
        lr = lr0 * (1.0 - frac) + lr1 * frac
        G = X @ X.T
        np.fill_diagonal(G, -np.inf)
        m = np.max(G)
        E = G - m
        E /= T
        np.clip(E, -50.0, 0.0, out=E)
        np.exp(E, out=E)
        np.fill_diagonal(E, 0.0)
        S = np.sum(E)
        if not np.isfinite(S) or S == 0:
            continue
        E /= S
        Gr = E @ X
        Gr *= 2.0
        dot = np.einsum('ij,ij->i', Gr, X)[:, None]
        Gr -= dot * X
        V *= mom
        V -= lr * Gr
        X += V
        nr = np.sqrt(np.einsum('ij,ij->i', X, X))[:, None]
        nr[nr == 0] = 1.0
        X /= nr
    return X

def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
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
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def build_pool(rng):
    parts = []
    # D12-type roots: two coords +-1/sqrt2
    s2 = 1.0 / np.sqrt(2.0)
    for i in range(12):
        for j in range(i+1, 12):
            for si in (1.0, -1.0):
                for sj in (1.0, -1.0):
                    v = np.zeros(12)
                    v[i] = si * s2
                    v[j] = sj * s2
                    parts.append(v)
    # all 3-sparse sign vectors
    import itertools
    for combo in itertools.combinations(range(12), 3):
        for mask in range(8):
            v = np.zeros(12)
            sc = 1.0 / np.sqrt(3.0)
            for k, idx in enumerate(combo):
                s = 1.0 if (mask >> k) & 1 else -1.0
                v[idx] = s * sc
            parts.append(v)
    # all 4-sparse sign vectors
    for combo in itertools.combinations(range(12), 4):
        for mask in range(16):
            v = np.zeros(12)
            sc = 0.5
            for k, idx in enumerate(combo):
                s = 1.0 if (mask >> k) & 1 else -1.0
                v[idx] = s * sc
            parts.append(v)
    # coordinate vectors
    for i in range(12):
        for s in (1.0, -1.0):
            v = np.zeros(12)
            v[i] = s
            parts.append(v)
    P0 = np.array(parts, dtype=np.float64)
    # sampled {0, +-1/2, +-sqrt3/2} patterns
    sq3 = np.sqrt(3.0) / 2.0
    n_sample = 16000
    Ps = np.empty((n_sample, 12), dtype=np.float64)
    for t in range(n_sample):
        r = rng.rand(12)
        v = np.empty(12)
        for k in range(12):
            rk = r[k]
            if rk < 0.40:
                v[k] = 0.0
            elif rk < 0.60:
                v[k] = 0.5
            elif rk < 0.80:
                v[k] = -0.5
            elif rk < 0.90:
                v[k] = sq3
            else:
                v[k] = -sq3
        nrm = np.linalg.norm(v)
        if nrm < 1e-12:
            v[0] = 1.0
            nrm = 1.0
        Ps[t] = v / nrm
    pool = np.vstack([P0, Ps])
    # normalise (P0 already unit)
    pool = pool / np.linalg.norm(pool, axis=1, keepdims=True)
    return pool

def solve():
    t0 = time.time()
    rng = np.random.RandomState(0)
    Tsoft = 0.02
    CLOSE = 1e-4
    # load incumbent
    bestX = None
    bestV = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            c = np.load(fn)
            c = np.asarray(c, dtype=np.float64)
            if c.shape != (102, 12):
                continue
            c = c / np.linalg.norm(c, axis=1, keepdims=True)
            if not np.all(np.isfinite(c)):
                continue
            v = max_cos(c)
            if v < bestV:
                bestV = v
                bestX = c.copy()
        except Exception:
            continue
    if bestX is None:
        bestX = rng.randn(102, 12)
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        bestV = max_cos(bestX)
    baseX = bestX.copy()
    baseV = bestV

    # build pool >=20000
    pool = build_pool(rng)
    N = pool.shape[0]
    K = 102
    # greedy lexicographic build
    Dfull = np.empty((N, K), dtype=np.float64)
    bestMax = np.full(N, -np.inf)
    is_sel = np.zeros(N, dtype=bool)
    sel_pool = []
    M_prev = -np.inf
    # store pair dots of selected set
    pair_list = []
    # first pick index 0
    for k in range(K):
        if time.time() - t0 > 30:
            # fallback: fill remaining randomly if timeout
            if len(sel_pool) < K:
                # fill with farthest random
                pass
            break
        if k == 0:
            s = 0
        else:
            Dmat = Dfull[:, :k]
            Mi = np.maximum(M_prev, bestMax)
            Mi[is_sel] = np.inf
            Mmin = np.min(Mi)
            mask = Mi <= Mmin + 1e-12
            idxs = np.where(mask)[0]
            if len(idxs) == 1:
                s = int(idxs[0])
            else:
                Dsub = Dmat[idxs]
                thr = Mmin - CLOSE
                newClose = np.sum(Dsub > thr, axis=1)
                cmin = np.min(newClose)
                cand2 = np.where(newClose <= cmin)[0]
                if len(cand2) == 1:
                    s = int(idxs[cand2[0]])
                else:
                    # softmax tie break: sum exp
                    E = Dsub[cand2] / Tsoft
                    np.clip(E, -50, 50, out=E)
                    np.exp(E, out=E)
                    sums = np.sum(E, axis=1)
                    bi = int(np.argmin(sums))
                    s = int(idxs[cand2[bi]])
        # select s
        sel_pool.append(int(s))
        is_sel[s] = True
        newCol = pool @ pool[s]
        Dfull[:, k] = newCol
        # update bestMax
        # for non-selected, max; for selected set inf
        np.maximum(bestMax, newCol, out=bestMax)
        bestMax[s] = np.inf
        # update M_prev and pair_list
        if k > 0:
            prev = np.array(sel_pool[:-1], dtype=int)
            # dots of new point to previous: newCol[prev]
            m = float(np.max(newCol[prev]))
            if m > M_prev:
                M_prev = m
            for pi in prev:
                pair_list.append(float(newCol[pi]))
        else:
            M_prev = -np.inf
        # mark other selected as inf
        bestMax[is_sel] = np.inf
        if len(sel_pool) >= K:
            break
    # if greedy incomplete due to time, pad
    while len(sel_pool) < K:
        # pick unused index with smallest bestMax
        cand = np.where(~is_sel)[0]
        # choose one with minimal bestMax
        bm = bestMax[cand]
        # bestMax for unused may be inf? fix
        finite = np.isfinite(bm)
        if np.any(finite):
            bi = int(np.argmin(np.where(finite, bm, np.inf)))
            s = int(cand[bi])
        else:
            s = int(cand[0])
        sel_pool.append(s)
        is_sel[s] = True
        k = len(sel_pool) - 1
        newCol = pool @ pool[s]
        if k < K:
            Dfull[:, k] = newCol
        np.maximum(bestMax, newCol, out=bestMax)
        bestMax[is_sel] = np.inf

    sel_pool = np.array(sel_pool, dtype=int)
    Sel = pool[sel_pool].copy()

    def cur_objective(Gram):
        iu = np.triu_indices(102, 1)
        c = Gram[iu]
        M = float(np.max(c))
        C = int(np.sum(c > M - CLOSE))
        E = c / Tsoft
        np.clip(E, -50, 50, out=E)
        S = float(np.sum(np.exp(E)))
        return M, C, S, c

    Gram = Sel @ Sel.T
    Mcur, Ccur, Scurr, _ = cur_objective(Gram)
    # precompute Q
    Q = pool @ Sel.T  # N x 102

    # swap improvement with time budget
    order_rng = np.random.RandomState(1)
    deadline_swap = t0 + 70.0
    n_sweeps = 0
    improved = True
    while improved and time.time() < deadline_swap and n_sweeps < 8:
        improved = False
        n_sweeps += 1
        order = order_rng.permutation(102)
        for p in order:
            if time.time() > deadline_swap:
                break
            # M_rest, rest dots, S_rest
            # mask rows/cols excluding p
            # compute M_rest quickly
            # Gram excluding p: max over Gram with row/col != p
            tmp = Gram.copy()
            tmp[p, :] = -2.0
            tmp[:, p] = -2.0
            Mrest = float(np.max(tmp))
            # rest dots for counting: upper triangular of rest
            # build rest Gram dots array
            keep = np.ones(102, dtype=bool)
            keep[p] = False
            Grest = Gram[np.ix_(keep, keep)]
            iu_r = np.triu_indices(101, 1)
            restDots = Grest[iu_r]
            # S_rest
            Er = restDots / Tsoft
            np.clip(Er, -50, 50, out=Er)
            Srest = float(np.sum(np.exp(Er)))
            # Q rest
            # compute max_new for all q
            # Q columns excluding p
            # to avoid big copy, compute max via two-step: overall row max may come from p or not
            # brute copy for simplicity
            Qr = Q[:, keep]  # N x 101
            max_new = np.max(Qr, axis=1)
            Mnew = np.maximum(Mrest, max_new)
            # exclude currently selected pool indices (except p's own) by setting inf
            # find pool indices of selected set
            # is_sel marks selected; allow p's own pool idx
            p_pool = int(sel_pool[p])
            # mask out other selected: set Mnew=inf where is_sel and != p_pool
            # create copy mask
            # To avoid O(N) search each time, use is_sel array
            Mnew[is_sel] = np.inf
            Mnew[p_pool] = max(Mrest, float(np.max(Gram[p, keep])))
            Mbest = float(np.min(Mnew))
            # candidates achieving Mbest
            tol = 1e-12
            candMask = Mnew <= Mbest + tol
            candIdx = np.where(candMask)[0]
            if candIdx.size == 0:
                continue
            # if Mbest > Mcur + 1e-12: should not happen because current gives Mcur
            # compute C/S for candidates
            thr = Mbest - CLOSE
            # C_rest part
            Crest = int(np.sum(restDots > thr))
            Qsub = Qr[candIdx]
            newClose = np.sum(Qsub > thr, axis=1)
            Cnew = Crest + newClose
            # S part
            Es = Qsub / Tsoft
            np.clip(Es, -50, 50, out=Es)
            np.exp(Es, out=Es)
            Ssum = np.sum(Es, axis=1)
            # lexicographic best among candidates
            cmin = int(np.min(Cnew))
            m2 = np.where(Cnew <= cmin)[0]
            if len(m2) == 1:
                bi = m2[0]
            else:
                ss = Ssum[m2]
                bi = m2[int(np.argmin(ss))]
            q_best = int(candIdx[bi])
            Mb = float(Mbest)
            Cb = int(Cnew[bi])
            Sb = float(Srest + Ssum[bi])
            # compare to current
            better = False
            if Mb < Mcur - 1e-12:
                better = True
            elif abs(Mb - Mcur) <= 1e-12:
                if Cb < Ccur:
                    better = True
                elif Cb == Ccur and Sb < Scurr - 1e-9:
                    better = True
            if better:
                # perform swap
                new_vec = pool[q_best]
                old_pool = int(sel_pool[p])
                is_sel[old_pool] = False
                is_sel[q_best] = True
                sel_pool[p] = q_best
                Sel[p] = new_vec
                # update Gram row/col p
                gcol = Sel @ new_vec  # 102
                Gram[p, :] = gcol
                Gram[:, p] = gcol
                Gram[p, p] = 1.0
                # update Q column p
                Q[:, p] = pool @ new_vec
                Mcur, Ccur, Scurr = Mb, Cb, Sb
                improved = True
        # end for p
    discSel = Sel.copy()
    discV = Mcur

    # continuous release of discrete set
    # annealed descent
    try:
        Xd = discSel.copy()
        # short anneal; adapt steps to remaining time
        rem = 178.0 - (time.time() - t0)
        if rem > 60:
            Xd = smooth_refine(Xd, 900, 0.02, 0.001, 0.03, 0.002, 0.88)
            Xd = smooth_refine(Xd, 500, 0.004, 0.0004, 0.008, 0.001, 0.90)
        elif rem > 30:
            Xd = smooth_refine(Xd, 600, 0.015, 0.0008, 0.02, 0.0015, 0.88)
        else:
            Xd = smooth_refine(Xd, 200, 0.01, 0.001, 0.01, 0.001, 0.88)
        vd = max_cos(Xd)
    except Exception:
        Xd = discSel.copy()
        vd = discV

    # choose between discrete-polished and incumbent
    if vd < baseV:
        curX = Xd
        curV = vd
    else:
        curX = baseX.copy()
        curV = baseV
        # also give incumbent a short refine if time allows and discrete failed
        try:
            rem2 = 178.0 - (time.time() - t0)
            if rem2 > 50:
                Xt = smooth_refine(curX.copy(), 600, 0.008, 0.0005, 0.008, 0.0008, 0.90)
                vt = max_cos(Xt)
                if vt < curV and np.all(np.isfinite(Xt)):
                    curX, curV = Xt, vt
        except Exception:
            pass

    # final LP polish on chosen
    rem = 178.0 - (time.time() - t0)
    if rem > 8:
        try:
            curX = lp_polish(curX, min(rem - 2.0, 20.0))
        except Exception:
            pass
    curX = curX / np.linalg.norm(curX, axis=1, keepdims=True)
    # safety: if final worse than base, return base
    try:
        vf = max_cos(curX)
        if vf > baseV and vf > vd:
            # return better of base/Xd
            if vd < baseV:
                return Xd / np.linalg.norm(Xd, axis=1, keepdims=True)
            else:
                return baseX / np.linalg.norm(baseX, axis=1, keepdims=True)
    except Exception:
        return baseX
    return curX
