import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def row_info(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    rm = G.max(axis=1)
    cnt = np.count_nonzero(G >= m - 1e-9, axis=1)
    return m, rm, cnt, G


def smooth_polish(x, iters, T0, T1, lr0, lr1):
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


def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    er = np.repeat(np.arange(n), d)
    ec = np.arange(n * d)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu]
        m = float(c_all.max())
        sel = c_all > m - max(20 * radius, 1e-9)
        if np.count_nonzero(sel) > 2600:
            # keep top 2600
            vals = c_all[sel]
            thr = np.partition(vals, len(vals) - 2600)[len(vals) - 2600]
            sel = c_all >= thr
        I, J = iu[0][sel], iu[1][sel]
        c = c_all[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + ar, J[:, None] * d + ar, np.full((p, 1), n * d)], 1).ravel()
        vls = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vls, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (er, ec)), shape=(n, nv)).tocsr()
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
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def largest_holes(x, k, rng, n_cand=15000):
    cand = rng.normal(size=(n_cand, 4))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ x.T
    bd = dots.max(axis=1)
    order = np.argsort(bd)
    picked = []
    for idx in order:
        c = cand[idx]
        ok = True
        for p in picked:
            if float(c @ p) > 0.90:
                ok = False
                break
        if ok:
            picked.append(c)
            if len(picked) >= k:
                break
    if len(picked) < k:
        for idx in order:
            c = cand[idx]
            if all(not np.allclose(c, p) for p in picked):
                picked.append(c)
                if len(picked) >= k:
                    break
    return np.array(picked)


def solve():
    t0 = time.time()
    rng = np.random.default_rng(20261031)
    n, d = 163, 4
    best = None
    best_val = 1e9
    for f in ("best.npy", "record.npy"):
        try:
            c = np.asarray(np.load(f), dtype=float)
            if c.shape != (n, d):
                continue
            c = c / np.linalg.norm(c, axis=1, keepdims=True)
            v = max_cos(c)
            if v < best_val:
                best_val = v
                best = c.copy()
        except Exception:
            pass
    if best is None:
        best = rng.normal(size=(n, d))
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = max_cos(best)
    best = best / np.linalg.norm(best, axis=1, keepdims=True)

    trial = 0
    # hopping phase until ~150s
    while time.time() - t0 < 148 and trial < 300:
        trial += 1
        mode = trial % 7
        y = best.copy()
        m0, rm0, cnt0, _ = row_info(best)
        if mode == 0:
            sig = 0.012 if (trial % 2 == 0) else 0.025
            y = y + rng.normal(scale=sig, size=y.shape)
        elif mode == 1:
            # shake worst points
            order = np.argsort(-rm0)
            k = int(rng.integers(2, 6))
            idx = order[:max(k * 3, 8)]
            sel = rng.choice(idx, size=k, replace=False)
            sig = 0.04 + 0.02 * rng.random()
            y[sel] += rng.normal(scale=sig, size=(k, d))
            # tiny global jitter
            y += rng.normal(scale=0.004, size=y.shape)
        elif mode == 2:
            # hole re-insert small: rattlers / low contacts
            low = np.where(cnt0 <= 2)[0]
            k = max(len(low), 2)
            k = min(k, 6)
            if k < 2:
                k = 2
            holes = largest_holes(best, k, rng)
            # choose points to replace: low + worst
            order = np.argsort(-rm0)
            cand_rm = [i for i in order if i not in low][:k]
            rep = np.array(list(low[:k]) + cand_rm)[:k]
            if len(rep) < k:
                extra = rng.choice(n, size=k - len(rep), replace=False)
                rep = np.concatenate([rep, extra])
            for a, h in zip(rep, holes):
                y[a] = h + rng.normal(scale=0.005, size=d)
        elif mode == 3:
            # cap jitter
            u = rng.normal(size=d)
            u /= np.linalg.norm(u)
            dots = y @ u
            thr = 0.70 + 0.15 * rng.random()
            mask = dots > thr
            if np.count_nonzero(mask) < 4:
                idx = np.argsort(dots)[-12:]
                mask = np.zeros(n, dtype=bool)
                mask[idx] = True
            nm = np.count_nonzero(mask)
            y[mask] += rng.normal(scale=0.07, size=(nm, d))
            y += rng.normal(scale=0.003, size=y.shape)
        elif mode == 4:
            # bigger surgery: remove 5-8 worst/random, insert holes
            k = int(rng.integers(5, 9))
            holes = largest_holes(best, k, rng)
            order = np.argsort(-rm0)
            # mix worst and random
            nw = k // 2 + 1
            rw = rng.choice(n, size=k - nw, replace=False)
            rep = np.concatenate([order[:nw], rw])
            # dedup
            _, ui = np.unique(rep, return_index=True)
            rep = rep[np.sort(ui)]
            while len(rep) < k:
                e = int(rng.integers(0, n))
                if e not in rep:
                    rep = np.append(rep, e)
            rep = rep[:k]
            for a, h in zip(rep, holes):
                y[a] = h + rng.normal(scale=0.008, size=d)
        elif mode == 5:
            # overloaded shake: high contact points
            order = np.argsort(-cnt0)
            k = int(rng.integers(4, 9))
            sel = order[:k]
            # move them toward holes slightly + jitter
            y[sel] += rng.normal(scale=0.05, size=(k, d))
        else:
            sig = 0.045 + 0.02 * rng.random()
            y = y + rng.normal(scale=sig, size=y.shape)

        y /= np.linalg.norm(y, axis=1, keepdims=True)
        # fast smooth: params depend on mode
        if mode in (2, 4):
            it = 700
            T0_, T1_ = 0.025, 0.0012
            lr0_, lr1_ = 0.010, 0.0012
        elif mode == 6:
            it = 600
            T0_, T1_ = 0.022, 0.0010
            lr0_, lr1_ = 0.009, 0.0010
        else:
            it = 400
            T0_, T1_ = 0.015, 0.0008
            lr0_, lr1_ = 0.008, 0.0010
        # cut iters if low on time
        rem = 148 - (time.time() - t0)
        if rem < 20:
            it = min(it, 250)
        y = smooth_polish(y, iters=it, T0=T0_, T1=T1_, lr0=lr0_, lr1=lr1_)
        v = max_cos(y)
        # only LP-polish promising ones
        if v < best_val + 0.004:
            # short LP
            budget = 1.2 if v < best_val + 0.001 else 0.7
            # don't exceed hopping deadline too much
            if time.time() - t0 + budget > 152 and v >= best_val:
                continue
            yp = lp_polish(y, seconds=budget)
            vp = max_cos(yp)
            if vp < best_val - 1e-12:
                best = yp / np.linalg.norm(yp, axis=1, keepdims=True)
                best_val = vp
            elif vp < v and v < best_val and rng.random() < 0.1:
                # sideways move to new basin even if not beating best yet
                pass
        # opportunistic accept slightly worse to diversify? keep best only
        if v < best_val - 1e-12:
            best = y / np.linalg.norm(y, axis=1, keepdims=True)
            best_val = v

    # final intensive polish with remaining time
    rem = 178 - (time.time() - t0) - 1.0
    if rem > 3:
        best = lp_polish(best, seconds=min(rem, 28))
        best = best / np.linalg.norm(best, axis=1, keepdims=True)
    else:
        best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
