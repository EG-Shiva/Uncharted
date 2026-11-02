import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def _lp_polish(x, deadline, max_solves=120):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu0 = np.triu_indices(n, 1)
    solves = 0
    while time.time() < deadline and radius > 1e-14 and solves < max_solves:
        G = x @ x.T
        c = G[iu0]
        m = float(c.max())
        win = max(20.0 * radius, 1e-9)
        sel = c > m - win
        # cap size for speed
        if int(sel.sum()) > 1600:
            # keep top 1600
            idx = np.argpartition(-c, 1600)[:1600]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = sel & mask
        I = iu0[0][sel]
        J = iu0[1][sel]
        p = len(I)
        if p == 0:
            break
        cc = c[sel]
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
        except Exception:
            break
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cc, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        solves += 1
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
    return x, best


def _anneal(x, steps, T0, T1, lrfac, rng, noise=0.0, mu=0.0):
    n, d = x.shape
    vel = np.zeros_like(x) if mu > 0 else None
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        T = T0 * (T1 / T0) ** frac if T1 > 0 else T0
        G = x @ x.T
        np.fill_diagonal(G, -1.0)
        m = float(G.max())
        # avoid overflow: (G-m)<=0
        W = np.exp((G - m) / T)
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        row = W.sum(axis=1, keepdims=True) + 1e-12
        gt /= row
        lr = lrfac * T
        if mu > 0:
            vel = mu * vel + gt
            # bias correction for early steps
            bc = 1.0 - mu ** (s + 1)
            step = vel / bc
            x = x - lr * step
        else:
            x = x - lr * gt
        if noise > 0:
            N = rng.standard_normal(size=x.shape)
            Nd = np.sum(N * x, axis=1, keepdims=True)
            N -= Nd * x
            x = x + noise * np.sqrt(max(lr, 1e-12)) * N
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def _find_void(x_keep, rng, ns=15000, refine=40):
    R = rng.standard_normal(size=(ns, 4))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    S = R @ x_keep.T
    m = S.max(axis=1)
    v = R[int(np.argmin(m))].copy()
    # refine: push away from near neighbours (softmax repulsion)
    for it in range(refine):
        dots = x_keep @ v
        mm = float(dots.max())
        T = 0.004
        w = np.exp((dots - mm) / T)
        w /= (w.sum() + 1e-12)
        g = w @ x_keep
        gt = g - float(g @ v) * v
        # step to minimise max dot
        step = 0.05 / (1.0 + it * 0.15)
        v = v - step * gt
        nrm = np.linalg.norm(v)
        if nrm < 1e-12:
            v = rng.standard_normal(4)
            nrm = np.linalg.norm(v)
        v /= nrm
    return v


def _patch_from_worst(G, seed, size, m):
    # adjacency: pairs within 0.025 of max
    n = G.shape[0]
    adj = G > (m - 0.025)
    # BFS
    seen = np.zeros(n, dtype=bool)
    seen[seed] = True
    frontier = [seed]
    order = [seed]
    while frontier and len(order) < size:
        nxt = []
        for u in frontier:
            nb = np.where(adj[u])[0]
            for w in nb:
                if w == u:
                    continue
                if not seen[w]:
                    seen[w] = True
                    order.append(w)
                    nxt.append(w)
                    if len(order) >= size:
                        break
            if len(order) >= size:
                break
        frontier = nxt
        if not frontier:
            break
    if len(order) < size:
        # fill with worst per-point max
        per = G.max(axis=1)
        for w in np.argsort(-per):
            if not seen[w]:
                seen[w] = True
                order.append(int(w))
                if len(order) >= size:
                    break
    return np.array(order[:size], dtype=int)


def solve():
    t0 = time.time()
    deadline = t0 + 175.0
    explore_end = t0 + 128.0
    rng = np.random.default_rng(20260607)

    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    if xb is not None and xb.shape == (153, 4):
        cands.append(xb / np.linalg.norm(xb, axis=1, keepdims=True))
    if xr is not None and xr.shape == (153, 4):
        cands.append(xr / np.linalg.norm(xr, axis=1, keepdims=True))
    if not cands:
        R = rng.standard_normal(size=(153, 4))
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        cands.append(R)
    vals = [_max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    best_x = cands[bi].copy()
    best_val = float(vals[bi])

    # quick initial squeeze if time allows (already near optimum)
    # do not spend too much yet
    best_x, best_val = _lp_polish(best_x, min(deadline - 5, time.time() + 4.0), max_solves=30)

    cur_x = best_x.copy()
    cur_val = best_val

    trial = 0
    T_accept = 0.0012
    n = 153

    while time.time() < explore_end:
        trial += 1
        if trial % 11 == 0:
            base = best_x
            base_val = best_val
        else:
            base = cur_x
            base_val = cur_val
        if cur_val > best_val + 0.015 and (trial % 5 == 0):
            base = best_x
            base_val = best_val

        G0 = base @ base.T
        np.fill_diagonal(G0, -2.0)
        per = G0.max(axis=1)
        m0 = float(G0.max())
        mode = trial % 7

        y = base.copy()
        if mode == 0:
            # targeted jitter on worst points
            idx = np.argsort(-per)[:10]
            y[idx] += 0.07 * rng.standard_normal((10, 4))
            y += 0.004 * rng.standard_normal(y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _anneal(y, steps=380, T0=0.020, T1=0.0025, lrfac=1.25, rng=rng, noise=0.15, mu=0.85)
            y = _anneal(y, steps=140, T0=0.004, T1=0.0012, lrfac=1.0, rng=rng, noise=0.0, mu=0.0)
        elif mode == 1:
            # single void reinsertion of worst point
            w = int(np.argmax(per))
            keep = np.delete(base, w, axis=0)
            ns = 15000 if (trial % 2 == 0) else 8000
            v = _find_void(keep, rng, ns=ns, refine=35)
            y[w] = v + 0.006 * rng.standard_normal(4)
            y += 0.003 * rng.standard_normal(y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _anneal(y, steps=400, T0=0.016, T1=0.002, lrfac=1.2, rng=rng, noise=0.10, mu=0.8)
            y = _anneal(y, steps=130, T0=0.0035, T1=0.0012, lrfac=1.0, rng=rng, noise=0.0, mu=0.0)
        elif mode == 2:
            # double void reinsertion
            ws = np.argsort(-per)[:2]
            keep = np.delete(base, ws, axis=0)
            for w in ws:
                v = _find_void(keep, rng, ns=9000, refine=30)
                y[w] = v + 0.008 * rng.standard_normal(4)
                keep = np.vstack([keep, y[w] / np.linalg.norm(y[w])])
            y += 0.0035 * rng.standard_normal(y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _anneal(y, steps=420, T0=0.017, T1=0.002, lrfac=1.2, rng=rng, noise=0.12, mu=0.85)
            y = _anneal(y, steps=130, T0=0.0035, T1=0.0012, lrfac=1.0, rng=rng, noise=0.0, mu=0.0)
        elif mode == 3:
            # contact-patch shake around worst point
            seed = int(np.argmax(per))
            # need full G without -2 diag for adjacency
            Gf = base @ base.T
            np.fill_diagonal(Gf, -2.0)
            # use m0
            sz = int(rng.integers(8, 16))
            patch = _patch_from_worst(Gf, seed, sz, m0)
            y[patch] += (0.09 if (trial % 14 == 3) else 0.05) * rng.standard_normal((len(patch), 4))
            # also shake rattlers (low per) a bit more
            iso = np.argsort(per)[:3]
            y[iso] += 0.03 * rng.standard_normal((3, 4))
            y += 0.002 * rng.standard_normal(y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _anneal(y, steps=450, T0=0.019, T1=0.0022, lrfac=1.25, rng=rng, noise=0.14, mu=0.85)
            y = _anneal(y, steps=140, T0=0.004, T1=0.0012, lrfac=1.0, rng=rng, noise=0.0, mu=0.0)
        elif mode == 4:
            # global shake medium + anneal (explore)
            amp = 0.06 if (trial % 3 == 0) else 0.025
            y = y + amp * rng.standard_normal(y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _anneal(y, steps=500, T0=0.028, T1=0.0025, lrfac=1.35, rng=rng, noise=0.18, mu=0.88)
            y = _anneal(y, steps=120, T0=0.0035, T1=0.0012, lrfac=1.0, rng=rng, noise=0.0, mu=0.0)
        elif mode == 5:
            # triple void: worst + 2 random among top-15
            top = np.argsort(-per)[:15]
            ws = np.concatenate([[int(np.argmax(per))], rng.choice(top, size=2, replace=False)])
            ws = np.unique(ws)
            keep = np.delete(base, ws, axis=0)
            for w in ws:
                v = _find_void(keep, rng, ns=7000, refine=25)
                y[w] = v + 0.010 * rng.standard_normal(4)
                keep = np.vstack([keep, y[w] / np.linalg.norm(y[w])])
            y += 0.003 * rng.standard_normal(y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _anneal(y, steps=420, T0=0.016, T1=0.002, lrfac=1.2, rng=rng, noise=0.10, mu=0.8)
            y = _anneal(y, steps=120, T0=0.003, T1=0.001, lrfac=1.0, rng=rng, noise=0.0, mu=0.0)
        else:
            # small jitter + strong polish (exploit)
            y = y + 0.012 * rng.standard_normal(y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _anneal(y, steps=350, T0=0.012, T1=0.0018, lrfac=1.15, rng=rng, noise=0.05, mu=0.7)
            y = _anneal(y, steps=150, T0=0.003, T1=0.001, lrfac=1.0, rng=rng, noise=0.0, mu=0.0)

        v = _max_cos(y)

        if v < best_val - 1e-12:
            if v < best_val - 4e-6:
                y2, v2 = _lp_polish(y, min(deadline - 6.0, time.time() + 2.2), max_solves=22)
                if v2 < best_val:
                    best_x, best_val = y2, v2
                    cur_x, cur_val = y2.copy(), v2
                    continue
                elif v < best_val:
                    best_x, best_val = y, v
                    cur_x, cur_val = y.copy(), v
                    continue
            else:
                best_x, best_val = y, v
                cur_x, cur_val = y.copy(), v
                continue

        if v < cur_val:
            cur_x, cur_val = y, v
        else:
            if rng.random() < np.exp((cur_val - v) / T_accept):
                cur_x, cur_val = y, v

        if (trial % 7 == 0) and time.time() < explore_end - 5.0 and v < best_val + 0.002:
            y2, v2 = _lp_polish(y, time.time() + 1.1, max_solves=10)
            if v2 < best_val - 1e-12:
                best_x, best_val = y2, v2
                cur_x, cur_val = y2.copy(), v2
            elif v2 < cur_val:
                cur_x, cur_val = y2, v2

    # micro-basin search around best with tiny kicks + LP
    if time.time() < deadline - 2.0:
        best_x, best_val = _lp_polish(best_x, min(deadline - 0.6, time.time() + 22.0), max_solves=120)
        # tiny kicks
        rk = 0
        while time.time() < deadline - 0.8 and rk < 30:
            rk += 1
            z = best_x + (2e-4 * (0.6 ** (rk // 6))) * rng.standard_normal(best_x.shape)
            z /= np.linalg.norm(z, axis=1, keepdims=True)
            z2, v2 = _lp_polish(z, time.time() + 1.0, max_solves=12)
            if v2 < best_val:
                best_x, best_val = z2, v2

    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
