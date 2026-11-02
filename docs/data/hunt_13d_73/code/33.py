import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def _lp_polish(X, seconds):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        Cfull = X @ X.T
        c = Cfull[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            X, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X


def solve():
    t0 = time.time()
    HARD_END = t0 + 176.0
    SEARCH_END = t0 + 150.0
    rng = np.random.RandomState(20261102)
    n, d = 73, 13
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.randn(n, d)
    if xb.shape != (n, d):
        tmp = rng.randn(n, d)
        m0 = min(xb.shape[0], n)
        m1 = min(xb.shape[1], d)
        tmp[:m0, :m1] = xb[:m0, :m1]
        xb = tmp
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best = _max_cos(xb)
    Xbest = xb.copy()

    cap_cycle = [12, 14, 10, 16, 11, 13, 15, 12, 10, 14]

    def contact_info(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        cm = float(G.max())
        cnt = (G >= cm - 1e-9).sum(axis=1)
        return cnt, cm, G

    def pick_center(X, cnt, typ):
        if typ == 0:
            # rattler direction
            idx = int(np.argmin(cnt))
            return X[idx].copy()
        elif typ == 1:
            idx = int(rng.randint(n))
            return X[idx].copy()
        elif typ == 2:
            idx = int(np.argmax(cnt))
            return X[idx].copy()
        else:
            # PCA outlier
            try:
                mu = X.mean(axis=0)
                Xc = X - mu
                _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
                V2 = Vt[:2, :].T  # 13x2
                P = Xc @ V2  # 73x2
                cen = P.mean(axis=0)
                dist = np.linalg.norm(P - cen, axis=1)
                idx = int(np.argmax(dist))
                return X[idx].copy()
            except Exception:
                idx = int(rng.randint(n))
                return X[idx].copy()

    def refine_single(v0, Others, steps=120, s=300.0):
        v = v0.copy()
        m = np.zeros(d)
        vv = np.zeros(d)
        b1, b2, eps = 0.9, 0.999, 1e-8
        b1p, b2p = b1, b2
        lr0, lr1 = 0.05, 0.005
        l0, l1 = np.log(lr0), np.log(lr1)
        for k in range(steps):
            frac = k / (steps - 1) if steps > 1 else 1.0
            lr = np.exp(l0 + (l1 - l0) * frac)
            dots = Others @ v
            cmax = dots.max()
            e = np.exp(s * (dots - cmax))
            se = e.sum()
            if se == 0 or not np.isfinite(se):
                # subgradient on max
                mk = dots >= cmax - 1e-12
                g = Others[mk].mean(axis=0)
            else:
                w = e / se
                g = w @ Others
            gt = g - float(g @ v) * v
            m = b1 * m + (1 - b1) * gt
            vv = b2 * vv + (1 - b2) * (gt * gt)
            mhat = m / (1 - b1p)
            vhat = vv / (1 - b2p)
            b1p *= b1
            b2p *= b2
            step = mhat / (np.sqrt(vhat) + eps)
            v = v - lr * step
            nr = float(np.linalg.norm(v))
            if nr == 0 or not np.isfinite(nr):
                v = v0.copy()
            else:
                v /= nr
        return v

    def block_adam(X0, active_mask, steps, s0, s1, lr0, lr1):
        X = X0.copy()
        m1 = np.zeros_like(X)
        v1 = np.zeros_like(X)
        b1, b2, eps = 0.9, 0.999, 1e-8
        b1p, b2p = b1, b2
        ls0, ls1 = np.log(s0), np.log(s1)
        ll0, ll1 = np.log(lr0), np.log(lr1)
        act = active_mask[:, None]
        for k in range(steps):
            frac = k / (steps - 1) if steps > 1 else 1.0
            s = np.exp(ls0 + (ls1 - ls0) * frac)
            lr = np.exp(ll0 + (ll1 - ll0) * frac)
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            cmax = G.max()
            E = np.exp(s * (G - cmax))
            np.fill_diagonal(E, 0.0)
            sumE = E.sum() * 0.5
            if sumE == 0 or not np.isfinite(sumE):
                Mask = (G >= cmax - 1e-12).astype(float)
                np.fill_diagonal(Mask, 0.0)
                cntp = Mask.sum() * 0.5
                if cntp <= 0:
                    continue
                W = Mask / cntp
            else:
                W = E / sumE
            Gt = W @ X
            dot = (Gt * X).sum(axis=1, keepdims=True)
            gt = (Gt - dot * X) * act
            m1 = b1 * m1 + (1 - b1) * gt
            v1 = b2 * v1 + (1 - b2) * (gt * gt)
            mhat = m1 / (1 - b1p)
            vhat = v1 / (1 - b2p)
            b1p *= b1
            b2p *= b2
            step = mhat / (np.sqrt(vhat) + eps)
            X = X - lr * step * act
            nr = np.linalg.norm(X, axis=1, keepdims=True)
            nr[nr == 0] = 1.0
            X /= nr
            if time.time() > HARD_END - 2:
                break
        return X

    trial = -1
    while time.time() < SEARCH_END and time.time() < HARD_END - 5:
        trial += 1
        if time.time() > SEARCH_END - 2:
            break
        cnt, cm, _ = contact_info(Xbest)
        typ = trial % 4
        center = pick_center(Xbest, cnt, typ)
        nc = float(np.linalg.norm(center))
        if nc == 0 or not np.isfinite(nc):
            center = Xbest[int(rng.randint(n))].copy()
        else:
            center /= nc
        k = cap_cycle[trial % len(cap_cycle)]
        dots_c = Xbest @ center
        del_idx = np.argsort(-dots_c)[:k]
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[del_idx] = False
        F = Xbest[~keep_mask.__invert__() if False else keep_mask].copy() if False else Xbest[keep_mask].copy()
        # greedy rebuild one at a time, 80k candidates (4x20k)
        Rlist = []
        # randomize insertion order slightly
        order = np.arange(k)
        # shuffle order deterministically per trial
        perm = rng.permutation(k)
        # del_idx in permuted order for building; final map back
        build_seq = del_idx[perm]
        for s_idx in range(k):
            if len(Rlist) == 0:
                Others = F
            else:
                Others = np.vstack([F, np.array(Rlist)])
            batch_bests = []
            batch_scores = []
            for _b in range(4):
                Cb = rng.randn(20000, d)
                Cb /= np.linalg.norm(Cb, axis=1, keepdims=True)
                dots = Cb @ Others.T
                md = dots.max(axis=1)
                bi = int(np.argmin(md))
                batch_bests.append(Cb[bi].copy())
                batch_scores.append(float(md[bi]))
                if time.time() > HARD_END - 3:
                    break
            batch_scores = np.array(batch_scores)
            # refine best two batch winners
            take = np.argsort(batch_scores)[:2]
            best_v = batch_bests[int(take[0])]
            best_sc = float(batch_scores[int(take[0])])
            refined_best = best_v
            refined_sc = best_sc
            for ti in take:
                v0 = batch_bests[int(ti)]
                v = refine_single(v0, Others, steps=120, s=300.0)
                sc = float((Others @ v).max())
                if sc < refined_sc:
                    refined_sc = sc
                    refined_best = v.copy()
                if time.time() > HARD_END - 3:
                    break
            Rlist.append(refined_best)
            if time.time() > HARD_END - 3:
                # fill remaining randomly to avoid empty
                while len(Rlist) < k:
                    vv = rng.randn(d)
                    vv /= np.linalg.norm(vv)
                    Rlist.append(vv)
                break
        Rarr = np.array(Rlist)  # in build_seq order
        Xcand = Xbest.copy()
        for ii, idx in enumerate(build_seq):
            Xcand[int(idx)] = Rarr[ii]
        # block-only optimization (frozen obstacles in LogSumExp)
        active = np.zeros(n, dtype=float)
        active[del_idx] = 1.0
        try:
            Xcand = block_adam(Xcand, active, steps=320,
                               s0=80.0, s1=1600.0, lr0=0.02, lr1=0.0015)
        except Exception:
            pass
        # unfreeze: 1-second smooth relaxation (global, fewer steps)
        try:
            # estimate steps fitting ~1s: use 450 steps
            Xcand = block_adam(Xcand, np.ones(n), steps=450,
                               s0=150.0, s1=2500.0, lr0=0.008, lr1=0.0006)
        except Exception:
            pass
        v = _max_cos(Xcand)
        # LP polish only if within 0.001 of incumbent or better
        if v < best + 0.001:
            remain_search = SEARCH_END - time.time()
            remain_hard = HARD_END - time.time()
            if remain_hard > 8 and remain_search > 1:
                try:
                    Xp = _lp_polish(Xcand, min(6.0, remain_hard - 2.0, max(1.0, remain_search)))
                    vp = _max_cos(Xp)
                except Exception:
                    Xp, vp = Xcand, v
            else:
                Xp, vp = Xcand, v
        else:
            Xp, vp = Xcand, v
        if vp < best - 1e-12 and np.isfinite(vp):
            best = float(vp)
            Xbest = Xp / np.linalg.norm(Xp, axis=1, keepdims=True)
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest
