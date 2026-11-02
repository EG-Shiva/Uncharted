import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def _lp_polish(X, seconds, t_end=1e30):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < t_end and radius > 1e-14:
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
    HARD_END = t0 + 174.0
    SEARCH_END = t0 + 152.0
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
    Xcur = xb.copy()
    cur = best

    def adam_smooth(X0, steps, s0, s1, lr0, lr1, active=None):
        X = X0.copy()
        m1 = np.zeros_like(X)
        v1 = np.zeros_like(X)
        b1, b2, eps = 0.9, 0.999, 1e-8
        b1p, b2p = b1, b2
        ls0, ls1 = np.log(s0), np.log(s1)
        ll0, ll1 = np.log(lr0), np.log(lr1)
        if active is None:
            act = 1.0
        else:
            act = active[:, None]
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
                cnt = Mask.sum() * 0.5
                if cnt <= 0:
                    continue
                W = Mask / cnt
            else:
                W = E / sumE
            Gt = W @ X
            dot = (Gt * X).sum(axis=1, keepdims=True)
            gt = (Gt - dot * X)
            if active is not None:
                gt = gt * act
            m1 = b1 * m1 + (1 - b1) * gt
            v1 = b2 * v1 + (1 - b2) * (gt * gt)
            mhat = m1 / (1 - b1p)
            vhat = v1 / (1 - b2p)
            b1p *= b1
            b2p *= b2
            step = mhat / (np.sqrt(vhat) + eps)
            if active is not None:
                X = X - lr * step * act
            else:
                X = X - lr * step
            nr = np.linalg.norm(X, axis=1, keepdims=True)
            nr[nr == 0] = 1.0
            X /= nr
            if time.time() > HARD_END - 3:
                break
        return X

    def refine_single(v0, Others, steps=60, s=400.0):
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
            if time.time() > HARD_END - 3:
                break
        return v

    def greedy_rebuild_cap(Xsrc, k, ntry=20000):
        # random center: mix of existing point and random dir
        if rng.rand() < 0.7:
            center = Xsrc[int(rng.randint(n))].copy()
        else:
            center = rng.randn(d)
            center /= np.linalg.norm(center)
        dots_c = Xsrc @ center
        del_idx = np.argsort(-dots_c)[:k]
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[del_idx] = False
        F = Xsrc[keep_mask].copy()
        build_seq = del_idx[rng.permutation(k)]
        Rlist = []
        for _s in range(k):
            if len(Rlist) == 0:
                Others = F
            else:
                Others = np.vstack([F, np.array(Rlist)])
            Cb = rng.randn(ntry, d)
            Cb /= np.linalg.norm(Cb, axis=1, keepdims=True)
            dots = Cb @ Others.T
            md = dots.max(axis=1)
            bi = int(np.argmin(md))
            v0 = Cb[bi].copy()
            v = refine_single(v0, Others, steps=60, s=400.0)
            Rlist.append(v)
            if time.time() > HARD_END - 3:
                break
        while len(Rlist) < k:
            vv = rng.randn(d)
            vv /= np.linalg.norm(vv)
            Rlist.append(vv)
        Xcand = Xsrc.copy()
        for ii, idx in enumerate(build_seq):
            Xcand[int(idx)] = Rlist[ii]
        return Xcand, del_idx

    def kill_worst(Xsrc, k, ntry=20000):
        G = Xsrc @ X.T if False else Xsrc @ Xsrc.T
        np.fill_diagonal(G, -2.0)
        mx = G.max(axis=1)
        idxs = np.argsort(-mx)[:k]
        Z = Xsrc.copy()
        # reinsert one by one greedily
        for idx in idxs:
            others = np.delete(Z, int(idx), axis=0)
            Cb = rng.randn(ntry, d)
            Cb /= np.linalg.norm(Cb, axis=1, keepdims=True)
            dots = Cb @ others.T
            md = dots.max(axis=1)
            bi = int(np.argmin(md))
            v0 = Cb[bi].copy()
            v = refine_single(v0, others, steps=50, s=400.0)
            Z[int(idx)] = v
            if time.time() > HARD_END - 3:
                break
        return Z, idxs

    trial = 0
    Temp0, Temp1 = 0.0006, 0.00002
    while time.time() < SEARCH_END and time.time() < HARD_END - 12:
        trial += 1
        if time.time() > SEARCH_END - 1:
            break
        frac_g = (time.time() - t0) / max(1e-9, (SEARCH_END - t0))
        Temp = Temp0 * (1 - frac_g) + Temp1 * frac_g
        # periodic restart from best
        if trial % 70 == 0 and rng.rand() < 0.7:
            Xcur = Xbest.copy()
            cur = best
        r = rng.rand()
        try:
            if r < 0.42:
                k = int(rng.choice(np.array([8, 10, 12, 14])))
                Xcand, del_idx = greedy_rebuild_cap(Xcur, k, ntry=20000)
                # block relax rebuilt then short global
                active = np.zeros(n)
                active[np.array(del_idx, dtype=int)] = 1.0
                Xcand = adam_smooth(Xcand, steps=180, s0=80.0, s1=1200.0,
                                    lr0=0.02, lr1=0.0015, active=active)
                Xcand = adam_smooth(Xcand, steps=180, s0=150.0, s1=2000.0,
                                    lr0=0.008, lr1=0.0007)
            elif r < 0.64:
                k = int(3 + rng.randint(4))  # 3..6
                Xcand, _idx = kill_worst(Xcur, k, ntry=20000)
                Xcand = adam_smooth(Xcand, steps=220, s0=120.0, s1=2200.0,
                                    lr0=0.009, lr1=0.0007)
            elif r < 0.84:
                sig = 0.015 + 0.035 * rng.rand()
                Noise = rng.randn(n, d) * sig
                Xcand = Xcur + Noise
                Xcand /= np.linalg.norm(Xcand, axis=1, keepdims=True)
                # small cap-biased extra kick occasionally
                Xcand = adam_smooth(Xcand, steps=240, s0=100.0, s1=2500.0,
                                    lr0=0.010, lr1=0.0006)
            else:
                # max-pair split
                G = Xcur @ Xcur.T
                np.fill_diagonal(G, -2.0)
                ij = int(np.argmax(G))
                ii, jj = divmod(ij, n)
                Xcand = Xcur.copy()
                eps = 0.02 + 0.02 * rng.rand()
                vi = Xcand[ii] - eps * Xcand[jj]
                vj = Xcand[jj] - eps * Xcand[ii]
                # add orthogonal jitter
                oi = rng.randn(d) * 0.01
                oj = rng.randn(d) * 0.01
                oi -= float(oi @ Xcand[ii]) * Xcand[ii]
                oj -= float(oj @ Xcand[jj]) * Xcand[jj]
                vi = vi + oi
                vj = vj + oj
                vi /= np.linalg.norm(vi)
                vj /= np.linalg.norm(vj)
                Xcand[ii] = vi
                Xcand[jj] = vj
                # jitter a few neighbors of the pair
                dots_i = Xcand @ Xcand[ii]
                nbr = np.argsort(-dots_i)[1:6]
                for q in nbr:
                    if rng.rand() < 0.5:
                        pert = rng.randn(d) * 0.015
                        pert -= float(pert @ Xcand[int(q)]) * Xcand[int(q)]
                        vv = Xcand[int(q)] + pert
                        vv /= np.linalg.norm(vv)
                        Xcand[int(q)] = vv
                Xcand = adam_smooth(Xcand, steps=240, s0=120.0, s1=2500.0,
                                    lr0=0.009, lr1=0.0006)
        except Exception:
            continue
        v = _max_cos(Xcand)
        if not np.isfinite(v):
            continue
        if v < best - 1e-12:
            best = float(v)
            Xbest = Xcand / np.linalg.norm(Xcand, axis=1, keepdims=True)
            Xcur = Xcand.copy()
            cur = float(v)
        elif v < cur - 1e-12:
            Xcur = Xcand.copy()
            cur = float(v)
        else:
            # Metropolis sideways
            if Temp > 0 and v < cur + 5 * Temp:
                prob = float(np.exp(-(v - cur) / max(Temp, 1e-12)))
                if rng.rand() < prob:
                    Xcur = Xcand.copy()
                    cur = float(v)
        # early exit if big win? continue to squeeze
    # final sharp squeeze on best
    try:
        if time.time() < HARD_END - 4:
            Xbest = adam_smooth(Xbest, steps=600, s0=300.0, s1=8000.0,
                                lr0=0.005, lr1=0.0003)
            v = _max_cos(Xbest)
            if v < best and np.isfinite(v):
                best = float(v)
    except Exception:
        pass
    try:
        remain = HARD_END - time.time() - 1.0
        if remain > 3:
            Xp = _lp_polish(Xbest, min(remain, 12.0), t_end=HARD_END - 0.5)
            vp = _max_cos(Xp)
            if np.isfinite(vp) and vp < best + 1e-12:
                if vp < best:
                    best = float(vp)
                Xbest = Xp
    except Exception:
        pass
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest
