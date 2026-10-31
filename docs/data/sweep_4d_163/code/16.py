import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _row_maxs(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1), g

def _smooth(x, iters=450, T0=0.025, T1=0.0015, lr0=0.01, lr1=0.0008):
    n, d = x.shape
    for k in range(iters):
        f = k / max(iters - 1, 1)
        T = T0 * (T1 / T0) ** f
        lr = lr0 * (lr1 / lr0) ** f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp((G - m) / T)
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        rs = W.sum(axis=1, keepdims=True)
        step = gt / np.maximum(rs, 1e-12)
        sn = np.linalg.norm(step, axis=1, keepdims=True)
        step = step / np.maximum(sn, 1.0)
        x = x - lr * step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _refine_subset(y, idx, iters=120, T0=0.01, T1=0.001, lr=0.02):
    # move only idx, background frozen
    n, d = y.shape
    idx = np.asarray(idx, dtype=int)
    for k in range(iters):
        f = k / max(iters - 1, 1)
        T = T0 * (T1 / T0) ** f
        G = y @ y.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp((G - m) / T)
        np.fill_diagonal(W, 0.0)
        grad = W @ y
        dot = np.sum(grad * y, axis=1, keepdims=True)
        gt = grad - dot * y
        st = np.zeros_like(y)
        st[idx] = gt[idx]
        # normalize per point
        sn = np.linalg.norm(st[idx], axis=1, keepdims=True)
        st[idx] = st[idx] / np.maximum(sn, 1e-12)
        # clip
        y = y - lr * st
        y /= np.linalg.norm(y, axis=1, keepdims=True)
    return y

def _lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu]
        m = c_all.max()
        sel = c_all > m - max(20 * radius, 1e-9)
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 2500:
            # keep top 2500
            vals = c_all[sel]
            thr = np.partition(vals, cnt - 2500)[cnt - 2500]
            sel = c_all >= thr
            # if still too many, subsample
            idxs = np.where(sel)[0]
            if len(idxs) > 3000:
                take = np.random.choice(idxs, 3000, replace=False)
                ns = np.zeros_like(sel)
                ns[take] = True
                sel = ns
        I, J = iu[0][sel], iu[1][sel]
        c = c_all[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        valsA = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((valsA, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    rng = np.random.default_rng(7)
    t0 = time.time()
    n, d = 163, 4
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                cands.append(a / np.linalg.norm(a, axis=1, keepdims=True))
        except Exception:
            pass
    if not cands:
        z = rng.normal(size=(n, d))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        cands.append(z)
    best = None
    best_val = 1e9
    for c in cands:
        v = _max_cos(c)
        if v < best_val:
            best_val = v
            best = c.copy()
    best = best / np.linalg.norm(best, axis=1, keepdims=True)

    trial = 0
    # hopping budget ~132s, leave rest for final polish
    HOP_END = 132.0
    while time.time() - t0 < HOP_END and trial < 400:
        mode = trial % 6
        # sigma schedule
        sig_cycle = [0.015, 0.035, 0.06, 0.025, 0.05, 0.01]
        sigma = sig_cycle[trial % len(sig_cycle)]
        y = best.copy()
        if mode == 0:
            # global shake
            y = y + rng.normal(scale=sigma, size=y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth(y, iters=500)
        elif mode == 1:
            # cap-localized shake
            u = rng.normal(size=4); u /= np.linalg.norm(u)
            dots = y @ u
            # target ~25 points in cap
            thr = np.sort(dots)[-25]
            mask = dots >= thr
            noise = rng.normal(scale=sigma * 0.25, size=y.shape)
            nm = int(mask.sum())
            noise[mask] = rng.normal(scale=sigma * 2.2, size=(nm, d))
            y = y + noise
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth(y, iters=500)
        elif mode == 2 or mode == 3:
            # kill worst k and fill holes
            rm, _ = _row_maxs(y)
            k = 5 + (trial % 10)  # 5..14
            if mode == 3:
                k = 8 + (trial % 8)
            idx_bad = np.argsort(rm)[-k:]
            keep = np.ones(n, dtype=bool); keep[idx_bad] = False
            Xr = y[keep]
            # pool of candidates
            P = 7000 if k > 8 else 5000
            pool = rng.normal(size=(P, d))
            pool /= np.linalg.norm(pool, axis=1, keepdims=True)
            # score = max dot to kept set
            S = pool @ Xr.T
            scores = S.max(axis=1)
            order = np.argsort(scores)
            chosen = []
            for oi in order:
                if len(chosen) >= k:
                    break
                cnd = pool[oi]
                ok = True
                for ch in chosen:
                    if float(ch @ cnd) > 0.90:
                        ok = False
                        break
                if ok:
                    chosen.append(cnd)
                # fallback if too strict: after half scan relax
                if oi == order[P // 2] and len(chosen) == 0:
                    pass
            # if not enough (strict), fill with top ones
            if len(chosen) < k:
                for oi in order:
                    if len(chosen) >= k:
                        break
                    cnd = pool[oi]
                    # avoid duplicates
                    dup = False
                    for ch in chosen:
                        if float(ch @ cnd) > 0.999:
                            dup = True
                            break
                    if not dup:
                        # allow closer
                        already = any(float(ch @ cnd) > 0.90 for ch in chosen)
                        if already:
                            chosen.append(cnd)
            chosen = np.array(chosen[:k])
            if chosen.shape[0] == k:
                y[idx_bad] = chosen
            else:
                # random fallback
                rr = rng.normal(size=(k, d))
                rr /= np.linalg.norm(rr, axis=1, keepdims=True)
                y[idx_bad] = rr
            # small shake of background
            bg = rng.normal(scale=sigma * 0.4, size=y.shape)
            bg[idx_bad] = 0
            y = y + bg
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            # refine inserted only
            y = _refine_subset(y, idx_bad, iters=100, lr=0.03)
            # joint smooth
            y = _smooth(y, iters=450)
        elif mode == 4:
            # hub + rattler targeted shake
            rm, G = _row_maxs(y)
            m = rm.max()
            # rattlers: far below max
            ratt = np.where(rm < m - 5e-4)[0]
            hubs = np.argsort(rm)[-8:]
            noise = rng.normal(scale=sigma * 0.35, size=y.shape)
            if len(ratt) > 0:
                noise[ratt] = rng.normal(scale=sigma * 2.5, size=(len(ratt), d))
            noise[hubs] = rng.normal(scale=sigma * 2.0, size=(len(hubs), d))
            # occasionally fully relocate a rattler to best hole
            if len(ratt) > 0 and (trial % 2 == 0):
                keep = np.ones(n, dtype=bool)
                # relocate worst rattler
                r0 = ratt[int(np.argmin(rm[ratt]))]
                keep[r0] = False
                Xr = y[keep]
                pool = rng.normal(size=(4000, d))
                pool /= np.linalg.norm(pool, axis=1, keepdims=True)
                sc = (pool @ Xr.T).max(axis=1)
                y[r0] = pool[int(np.argmin(sc))]
            y = y + noise
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth(y, iters=550, T0=0.03, T1=0.001)
        else:
            # random small rotation + shake
            A = rng.normal(scale=sigma * 2.0, size=(d, d))
            A = A - A.T
            # exp approx: I + A
            R = np.eye(d) + A
            # project via QR to orthogonal
            Q, _ = np.linalg.qr(R)
            if np.linalg.det(Q) < 0:
                Q[:, 0] = -Q[:, 0]
            # apply to random subset (half)
            perm = rng.permutation(n)[: n // 2]
            y[perm] = y[perm] @ Q.T
            y = y + rng.normal(scale=sigma * 0.5, size=y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth(y, iters=450)

        v = _max_cos(y)
        # decide LP: only if promising
        thresh = best_val + (0.004 if trial < 40 else 0.002)
        if v < thresh:
            remain_hop = HOP_END - (time.time() - t0)
            if remain_hop < 0.5:
                break
            sec = 1.4 if v < best_val + 0.0005 else 0.9
            sec = min(sec, max(0.5, remain_hop / 3.0))
            y2 = _lp_polish(y, seconds=sec)
            v2 = _max_cos(y2)
            if v2 < best_val - 1e-12:
                best = y2
                best_val = v2
        trial += 1
        if time.time() - t0 > HOP_END - 1:
            break

    # final polish
    elapsed = time.time() - t0
    budget = max(5.0, min(32.0, 170.0 - elapsed))
    # alternate smooth + LP for final squeeze
    best = _smooth(best, iters=300, T0=0.008, T1=0.0005, lr0=0.004, lr1=0.0005)
    best = _lp_polish(best, seconds=budget)
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    best = np.nan_to_num(best)
    best /= np.linalg.norm(best, axis=1, keepdims=True)
    return best
