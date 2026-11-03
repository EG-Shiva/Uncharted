import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def per_point_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1)


def gd_refine(X, steps, T0, T1, lr0, lr1, mu=0.9):
    n, d = X.shape
    V = np.zeros_like(X)
    lT0 = np.log(T0)
    lT1 = np.log(T1)
    llr0 = np.log(lr0)
    llr1 = np.log(lr1)
    for s in range(steps):
        f = s / max(1, steps - 1)
        T = float(np.exp(lT0 + (lT1 - lT0) * f))
        lr = float(np.exp(llr0 + (llr1 - llr0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        mx = float(G.max())
        # softmax, avoid overflow
        W = np.exp((G - mx) / T)
        np.fill_diagonal(W, 0.0)
        sw = float(W.sum())
        if not np.isfinite(sw) or sw == 0:
            break
        W /= sw
        Grad = W @ X
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        V = mu * V + Gt
        X = X - lr * V
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def deepest_hole(others, rng, n_sample=3000, refine_steps=250):
    d = others.shape[1]
    cand = rng.normal(size=(n_sample, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ others.T
    md = dots.max(axis=1)
    j = int(np.argmin(md))
    y = cand[j].copy()
    lr = 0.15
    for _ in range(refine_steps):
        dts = others @ y
        k = int(np.argmax(dts))
        mval = float(dts[k])
        g = others[k] - mval * y
        gn = float(np.linalg.norm(g))
        if gn < 1e-12:
            break
        y = y - lr * g
        ny = float(np.linalg.norm(y))
        if ny == 0 or not np.isfinite(ny):
            break
        y /= ny
        lr *= 0.996
    y /= np.linalg.norm(y)
    return y


def lp_polish(x, seconds, radius_init=1e-3, deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = float(radius_init)
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if deadline is not None and time.time() > deadline:
            break
        if time.time() - start > seconds - 0.05:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > 2000:
            idx = np.argpartition(c, -2000)[-2000:]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
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


def solve():
    t0 = time.time()
    BUDGET = 170.0
    EXPL_T = 128.0
    rng = np.random.default_rng(7)
    np.random.seed(7)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    if xb is not None and xb.shape == (95, 13):
        xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
        cands.append(xb)
    if xr is not None and xr.shape == (95, 13):
        xr = xr / np.linalg.norm(xr, axis=1, keepdims=True)
        cands.append(xr)
    if len(cands) == 0:
        x0 = rng.normal(size=(95, 13))
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        cands.append(x0)
    vals = [max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    x_best = cands[bi].copy()
    best_val = float(vals[bi])
    # dedup elites
    elites = [(best_val, x_best.copy())]
    for c, v in zip(cands, vals):
        if v > best_val + 1e-9:
            elites.append((float(v), c.copy()))
    cur = x_best.copy()
    cur_val = best_val
    n, d = x_best.shape

    it = 0
    sigmas_global = [0.01, 0.02, 0.04, 0.08, 0.15, 0.30]
    while time.time() - t0 < EXPL_T:
        it += 1
        if time.time() - t0 > BUDGET - 40:
            break
        remain = EXPL_T - (time.time() - t0)
        if remain < 0.5:
            break
        u = rng.random()
        x_try = cur.copy()
        # default gd params
        steps = 600
        T0, T1 = 0.016, 0.0011
        lr0, lr1 = 0.10, 0.008
        if u < 0.32:
            # global gaussian shake
            sigma = sigmas_global[int(rng.integers(0, len(sigmas_global)))]
            if rng.random() < 0.3:
                ksub = int(rng.integers(4, 14))
                idx = rng.choice(n, size=ksub, replace=False)
                x_try[idx] += sigma * 2.0 * rng.normal(size=(ksub, d))
            else:
                x_try += sigma * rng.normal(size=(n, d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            steps = int(rng.choice([400, 600, 800]))
        elif u < 0.52:
            # local patch shake around contact neighbourhood
            centre = int(rng.integers(0, n))
            dots = cur[centre] @ cur.T
            order = np.argsort(-dots)
            k = int(rng.integers(6, 16))
            patch = order[1:1 + k]
            s_big = float(rng.choice([0.08, 0.15, 0.25, 0.35]))
            s_small = 0.008
            x_try[patch] += s_big * rng.normal(size=(len(patch), d))
            x_try += s_small * rng.normal(size=(n, d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            steps = int(rng.choice([500, 700, 900]))
            T0 = 0.018
        elif u < 0.78:
            # hole reinsertion of overloaded / worst points
            ppm = per_point_max(cur)
            # contact count proxy
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            cc = (G > cur_val - 0.02).sum(axis=1)
            if rng.random() < 0.6:
                k = 1 if rng.random() < 0.5 else int(rng.integers(2, 4))
                # worst by ppm then cc
                order = np.argsort(-ppm - 0.001 * cc)
                idx = order[:k].copy()
                if k > 1 and rng.random() < 0.4:
                    idx[-1] = int(rng.integers(0, n))
            elif rng.random() < 0.5:
                k = int(rng.integers(1, 4))
                idx = rng.choice(n, size=k, replace=False)
            else:
                # rattlers: lowest ppm
                k = int(rng.integers(1, 3))
                order = np.argsort(ppm)
                idx = order[:k].copy()
            for j in idx:
                mask = np.ones(n, dtype=bool)
                mask[int(j)] = False
                others = x_try[mask]
                ns = 4000 if len(idx) == 1 else 2000
                x_try[int(j)] = deepest_hole(others, rng, n_sample=ns)
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            steps = int(rng.choice([500, 700]))
        else:
            # large jump + rattler push: worst pair split
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            ij = np.unravel_index(int(np.argmax(G)), G.shape)
            a, b = int(ij[0]), int(ij[1])
            # push a away from b
            v = x_try[a] - (float(x_try[a] @ x_try[b])) * x_try[b]
            nv = float(np.linalg.norm(v))
            if nv > 1e-12:
                v /= nv
                x_try[a] = x_try[a] - 0.1 * x_try[b] + 0.05 * rng.normal(size=(d,))
                x_try[a] /= np.linalg.norm(x_try[a])
            # add global noise
            sigma = float(rng.choice([0.15, 0.30, 0.45]))
            x_try += sigma * rng.normal(size=(n, d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            steps = 800
            T0, T1 = 0.020, 0.0010
            lr0, lr1 = 0.12, 0.008

        # cheap annealed repulsion
        try:
            x_gd = gd_refine(x_try, steps, T0, T1, lr0, lr1, mu=0.9)
        except Exception:
            continue
        v = max_cos(x_gd)
        improved_best = v < best_val - 1e-12
        improved_cur = v < cur_val
        # decide LP
        do_lp = False
        lp_sec = 1.0
        if improved_best:
            do_lp = True
            lp_sec = 1.4
        elif v < best_val + 0.002 and (v < cur_val - 0.0005 or rng.random() < 0.08):
            # promising near-best, squeeze occasionally
            do_lp = True
            lp_sec = 0.9
        if do_lp:
            if time.time() - t0 > BUDGET - 38:
                do_lp = False
            else:
                try:
                    x_lp = lp_polish(x_gd, seconds=lp_sec, radius_init=0.006,
                                     deadline=t0 + BUDGET - 32)
                except Exception:
                    x_lp = x_gd
                v_lp = max_cos(x_lp)
                # keep lp result if not worse by much (lp renormalizes)
                if v_lp <= v + 1e-9:
                    x_gd, v = x_lp, v_lp
        if v < best_val - 1e-12:
            x_best = x_gd.copy()
            best_val = v
            cur = x_gd.copy()
            cur_val = v
            elites.append((v, x_gd.copy()))
            if len(elites) > 6:
                elites = sorted(elites, key=lambda e: e[0])[:6]
        elif v < cur_val:
            cur = x_gd.copy()
            cur_val = v
        else:
            if rng.random() < 0.05 and v < cur_val + 0.008:
                cur = x_gd.copy()
                cur_val = v
            if rng.random() < 0.10:
                # restart from best or random elite
                if rng.random() < 0.7 or len(elites) == 1:
                    cur = x_best.copy()
                    cur_val = best_val
                else:
                    e = elites[int(rng.integers(0, len(elites)))]
                    cur = e[1].copy()
                    cur_val = e[0]
    # final squeeze: polish top elites and pick best
    elites = sorted(elites, key=lambda e: e[0])[:3]
    # ensure x_best included
    if len(elites) == 0:
        elites = [(best_val, x_best.copy())]
    remain = BUDGET - (time.time() - t0) - 1.0
    if remain < 3:
        x_best = elites[0][1] / np.linalg.norm(elites[0][1], axis=1, keepdims=True)
        return x_best
    # allocate time across elites, most to best
    total = float(np.clip(remain, 4.0, 38.0))
    # quick gd tighten then LP
    best_final = elites[0][1].copy()
    best_final_val = elites[0][0]
    for ei, (ev, ex) in enumerate(elites):
        t_left = BUDGET - (time.time() - t0) - 1.0
        if t_left < 2:
            break
        share = total * (0.6 if ei == 0 else 0.4 / max(1, len(elites) - 1))
        share = min(share, t_left)
        if share < 1.5:
            continue
        try:
            # light anneal before LP for non-best elites
            if ei > 0:
                ex = gd_refine(ex.copy(), 400, 0.008, 0.0008, 0.05, 0.005)
            ex_p = lp_polish(ex, seconds=share * 0.85, radius_init=8e-4,
                             deadline=t0 + BUDGET - 1.0)
        except Exception:
            continue
        vv = max_cos(ex_p)
        if vv < best_final_val:
            best_final_val = vv
            best_final = ex_p.copy()
    # extra micro polish on winner if time left
    t_left = BUDGET - (time.time() - t0) - 0.8
    if t_left > 2.0:
        try:
            best_final = lp_polish(best_final, seconds=min(t_left, 12.0), radius_init=4e-4,
                                   deadline=t0 + BUDGET - 0.5)
        except Exception:
            pass
    best_final /= np.linalg.norm(best_final, axis=1, keepdims=True)
    return best_final
