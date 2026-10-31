import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def _row_maxs(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1), g


def _lp_polish(x, seconds, r0=2e-3, cap=2200):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        m = float(c_all.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = c_all > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > cap:
            v = c_all[sel]
            kth = np.sort(v)[-cap]
            sel = c_all >= kth
            cnt2 = int(np.count_nonzero(sel))
            if cnt2 > cap + 800:
                ids = np.where(sel)[0]
                # deterministic stride subsample
                keep = ids[::2][:cap]
                ns = np.zeros_like(sel)
                ns[keep] = True
                sel = ns
        I = iu0[0][sel]
        J = iu0[1][sel]
        c = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cd = np.arange(d)
        cols = np.concatenate([I[:, None] * d + cd, J[:, None] * d + cd,
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.6, 0.05)
        else:
            radius *= 0.5
    return x


def _smooth(x, iters, T0, T1, lr0, lr1):
    n, d = x.shape
    for k in range(iters):
        frac = k / max(iters - 1, 1)
        T = T0 * (T1 / T0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
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


def _void_candidates(x, keep_idx, rng, n_out, n_samp=3000):
    # find points far from set x[keep_idx]
    base = x[keep_idx] if len(keep_idx) > 0 else x
    cand = rng.normal(size=(n_samp, 4))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    s = cand @ base.T
    sc = s.max(axis=1)
    order = np.argsort(sc)[:n_out * 6]
    pool = cand[order]
    # local repulsion refinement of pool to spread them
    out = []
    chosen = []
    for v0 in pool:
        v = v0.copy()
        # few steps pushing away from base + already chosen
        for _ in range(25):
            d1 = base @ v
            w = np.exp(40.0 * (d1 - d1.max()))
            g = w @ base
            if len(chosen) > 0:
                C = np.stack(chosen)
                d2 = C @ v
                w2 = np.exp(40.0 * (d2 - d2.max()))
                g = g + 0.5 * (w2 @ C)
            gt = g - np.dot(g, v) * v
            nrm = np.linalg.norm(gt)
            if nrm < 1e-12:
                break
            v = v - 0.10 * gt / nrm
            v /= np.linalg.norm(v)
        out.append(v)
        # greedily keep diverse
        if len(out) >= n_out * 6:
            break
    # greedily select n_out most diverse / best
    out = np.stack(out)
    # score by max dot to base
    scores = (out @ base.T).max(axis=1)
    idx = np.argsort(scores)[:n_out * 2]
    # spread among themselves
    sel = []
    rem = list(idx)
    if rem:
        sel.append(rem.pop(0))
        while len(sel) < n_out and rem:
            # pick point minimizing max dot to sel + score
            S = out[sel] @ out[rem].T
            mm = S.max(axis=0) if len(sel) > 0 else np.zeros(len(rem))
            base_s = scores[rem]
            crit = base_s + 0.3 * mm
            j = int(np.argmin(crit))
            sel.append(rem.pop(j))
    return out[np.array(sel)]


def solve():
    rng = np.random.default_rng(7)
    t0 = time.time()
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
        if c is not None:
            c = np.asarray(c, dtype=float)
            if c.shape == (163, 4):
                c = c / np.linalg.norm(c, axis=1, keepdims=True)
                cands.append(c)
    if not cands:
        z = rng.normal(size=(163, 4))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        cands.append(z)
    best = None
    best_val = 1e9
    for c in cands:
        v = _max_cos(c)
        if v < best_val:
            best_val = v
            best = c.copy()
    n, d = 163, 4
    cur = best.copy()
    cur_val = best_val

    trial = 0
    # hopping budget
    while time.time() - t0 < 138 and trial < 80:
        # metropolis temperature decays
        Tmet = 0.0012 * (0.25 ** (trial / 80.0)) + 1e-5
        mv = trial % 6
        y = cur.copy()
        if mv == 0:
            # global shake, medium-large
            sig = [0.05, 0.12, 0.03, 0.20][(trial // 6) % 4]
            y = y + rng.normal(scale=sig, size=y.shape)
        elif mv == 1:
            # cap cluster shake with large amplitude
            u = rng.normal(size=4); u /= np.linalg.norm(u)
            dots = y @ u
            k = int(rng.integers(12, 30))
            idx = np.argsort(dots)[-k:]
            noise = rng.normal(scale=0.012, size=y.shape)
            noise[idx] = rng.normal(scale=0.22, size=(k, d))
            y = y + noise
        elif mv == 2:
            # surgery on overloaded points -> voids
            rm, G = _row_maxs(y)
            m = float(rm.max())
            thr = m - 1e-7
            # contact counts
            C = (G >= thr).sum(axis=1)
            # pick 2-4 most loaded
            k = int(rng.integers(2, 5))
            worst = np.argsort(C + 0.5 * rm)[-k:]
            keep = np.array([i for i in range(n) if i not in set(worst.tolist())])
            voids = _void_candidates(y, keep, rng, k)
            for t, w in enumerate(worst):
                y[w] = voids[t]
            # small shake of rest
            y = y + rng.normal(scale=0.008, size=y.shape)
        elif mv == 3:
            # rattler + worst relocation to voids
            rm, G = _row_maxs(y)
            m = float(rm.max())
            ratt = np.where(rm < m - 5e-4)[0]
            k = min(len(ratt), int(rng.integers(1, 4)))
            target = []
            if k > 0:
                # pick k rattlers with smallest rm
                o = np.argsort(rm[ratt])[:k]
                target = ratt[o].tolist()
            # add worst point always
            w = int(np.argmax(rm))
            if w not in target:
                target.append(w)
            # occasionally add random one
            if rng.random() < 0.4:
                r = int(rng.integers(0, n))
                if r not in target:
                    target.append(r)
            target = np.array(target, dtype=int)
            keep = np.array([i for i in range(n) if i not in set(target.tolist())])
            voids = _void_candidates(y, keep, rng, len(target))
            for t, w2 in enumerate(target):
                y[w2] = voids[t]
            y = y + rng.normal(scale=0.010, size=y.shape)
        elif mv == 4:
            # random subset re-init (big jump)
            k = int(rng.integers(6, 14))
            idx = rng.choice(n, size=k, replace=False)
            rnd = rng.normal(size=(k, 4))
            rnd /= np.linalg.norm(rnd, axis=1, keepdims=True)
            y[idx] = rnd
            y = y + rng.normal(scale=0.015, size=y.shape)
        else:
            # hemisphere flip + shake: pick cap and reflect tangent
            u = rng.normal(size=4); u /= np.linalg.norm(u)
            dots = y @ u
            mask = dots > 0.55
            if np.count_nonzero(mask) < 4:
                idx = np.argsort(dots)[-10:]
                mask = np.zeros(n, dtype=bool); mask[idx] = True
            # rotate cap points by random small rotation
            A = rng.normal(size=(4, 4)) * 0.35
            A = A - A.T
            # exp approx: I + A
            R = np.eye(4) + A
            # orthonormalize via QR
            Q, _ = np.linalg.qr(R)
            if np.linalg.det(Q) < 0:
                Q[:, 0] *= -1
            y[mask] = (y[mask] @ Q.T)
            y = y + rng.normal(scale=0.02, size=y.shape)

        y /= np.linalg.norm(y, axis=1, keepdims=True)
        # descent: bigger jumps need longer anneal
        if mv in (4, 1):
            y = _smooth(y, iters=500, T0=0.035, T1=0.0025, lr0=0.012, lr1=0.0012)
        elif mv == 2:
            y = _smooth(y, iters=420, T0=0.025, T1=0.002, lr0=0.010, lr1=0.001)
        else:
            y = _smooth(y, iters=380, T0=0.022, T1=0.002, lr0=0.009, lr1=0.001)

        remain = 138 - (time.time() - t0)
        if remain > 1.0:
            sec = 1.6 if mv in (2, 3) else 1.2
            sec = min(sec, max(0.5, remain / 6.0))
            y = _lp_polish(y, seconds=sec)
        v = _max_cos(y)
        if v < best_val - 1e-12:
            best = y.copy(); best_val = v
            cur = y.copy(); cur_val = v
        else:
            # metropolis accept for exploration
            if v < cur_val or rng.random() < np.exp(-(v - cur_val) / max(Tmet, 1e-9)):
                cur = y.copy(); cur_val = v
        trial += 1
        if time.time() - t0 > 137:
            break

    # final polish of best
    elapsed = time.time() - t0
    budget = max(5.0, min(32.0, 170.0 - elapsed))
    # alternate smooth + LP for final squeeze
    t1 = time.time()
    # one more short smooth before LP
    best = _smooth(best, iters=300, T0=0.012, T1=0.0015, lr0=0.006, lr1=0.0008)
    left = budget - (time.time() - t1)
    if left > 2:
        best = _lp_polish(best, seconds=left, r0=1.5e-3)
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    best = np.nan_to_num(best)
    best /= np.linalg.norm(best, axis=1, keepdims=True)
    return best
