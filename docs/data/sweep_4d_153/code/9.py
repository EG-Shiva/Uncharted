import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def anneal_exp(x, iters, k0, k1, lr0, lr1, mu, noise, rng):
    n, d = x.shape
    v = np.zeros_like(x)
    for t in range(iters):
        frac = t / max(1, iters - 1)
        k = k0 * (k1 / k0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        moff = G.max()
        W = np.exp(k * (G - moff))
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        grad = (W @ x) / rs
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        # normalize gradient scale to keep steps stable
        gs = np.linalg.norm(gt, axis=1, keepdims=True).mean() + 1e-12
        # clip extreme
        gt = gt / gs * min(1.0, gs * 2.0 + 0.2)
        v = mu * v + gt
        vc = v / (1.0 - mu ** (t + 1)) if mu > 0 else v
        x = x - lr * vc
        if noise > 0 and frac < 0.6:
            amp = noise * (1.0 - frac / 0.6)
            pert = rng.standard_normal(x.shape)
            pd = np.sum(pert * x, axis=1, keepdims=True)
            pert -= pd * x
            x = x + amp * pert
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x


def anneal_power(x, iters, m0, m1, p, lr0, lr1, mu, rng):
    n, d = x.shape
    v = np.zeros_like(x)
    for t in range(iters):
        frac = t / max(1, iters - 1)
        margin = m0 * (m1 / m0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        moff = G.max()
        thr = moff - margin
        S = np.maximum(0.0, (G - thr) / (margin + 1e-12))
        W = S ** p
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        grad = (W @ x) / rs
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        v = mu * v + gt
        vc = v / (1.0 - mu ** (t + 1)) if mu > 0 else v
        x = x - lr * vc
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x


def greedy_max_refine(x, iters, lr, tol0=0.02):
    n, d = x.shape
    for t in range(iters):
        frac = t / max(1, iters - 1)
        tol = tol0 * (1.0 - frac * 0.8)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        moff = G.max()
        mask = G > moff - tol
        # symmetric weight: each pair pushes both ends
        W = mask.astype(float)
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        # only move points that are in active set
        active = (rs[:, 0] > 0.5)
        if not np.any(active):
            break
        grad = (W @ x) / rs
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        step = lr * (1.0 - frac * 0.7)
        x = x - step * gt * active[:, None]
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x


def largest_void(x_keep, rng, n_sample=2000):
    # find point on S^3 maximizing min-angle (minimizing max dot) to x_keep
    nc = n_sample
    cand = rng.standard_normal((nc, 4))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ x_keep.T
    md = dots.max(axis=1)
    return cand, md


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
        vv = max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    rng = np.random.default_rng(7)
    n, d = 153, 4
    try:
        best = np.load("best.npy")
    except Exception:
        best = None
    try:
        rec = np.load("record.npy")
    except Exception:
        rec = None
    c0 = []
    if best is not None:
        best = np.asarray(best, dtype=float)
        if best.shape == (n, d):
            c0.append(best / np.linalg.norm(best, axis=1, keepdims=True))
    if rec is not None:
        rec = np.asarray(rec, dtype=float)
        if rec.shape == (n, d):
            c0.append(rec / np.linalg.norm(rec, axis=1, keepdims=True))
    if len(c0) == 0:
        tmp = rng.standard_normal((n, d))
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        c0.append(tmp)
    vals0 = [max_cos(c) for c in c0]
    bi = int(np.argmin(vals0))
    incumbent = c0[bi].copy()
    inc_val = float(vals0[bi])
    base_best = incumbent.copy()
    base_best_val = inc_val
    cands = [(inc_val, incumbent.copy())]
    for c, v in zip(c0, vals0):
        if v > inc_val + 1e-12:
            cands.append((float(v), c.copy()))
    cands = sorted(cands, key=lambda z: z[0])[:4]

    SEARCH_END = t0 + 118
    trial = 0
    MAXTRIAL = 400
    while time.time() < SEARCH_END and trial < MAXTRIAL:
        trial += 1
        rb = rng.random()
        if rb < 0.60:
            base = incumbent
        elif rb < 0.82:
            base = base_best
        else:
            base = cands[int(rng.integers(len(cands)))][1]
        x = base.copy()
        mode = rng.random()
        if mode < 0.35:
            sigma = float(rng.choice(np.array([0.015, 0.03, 0.06, 0.10, 0.16, 0.25])))
            x = x + sigma * rng.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        elif mode < 0.80:
            # overloaded removal + farthest-void reinsertion
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            moff = G.max()
            maxper = G.max(axis=1)
            cnt = (G > moff - 1e-4).sum(axis=1)
            score_pt = maxper + 0.004 * cnt
            # sometimes break a random contact edge entirely
            if rng.random() < 0.35:
                # pick random contact pair
                iu0 = np.triu_indices(n, 1)
                cc = (x @ x.T)[iu0]
                m0 = cc.max()
                selp = np.where(cc > m0 - 2e-4)[0]
                pe = selp[int(rng.integers(len(selp)))]
                i0, j0 = int(iu0[0][pe]), int(iu0[1][pe])
                k = int(rng.integers(2, 7))
                idx = np.array([i0, j0], dtype=int)
                # add more worst
                order = np.argsort(-score_pt)
                for o in order:
                    if len(idx) >= k:
                        break
                    if o not in idx:
                        idx = np.append(idx, o)
            else:
                k = int(rng.integers(2, 9))
                idx = np.argsort(-score_pt)[:k]
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[idx] = False
            Xkeep = x[keep_mask]
            # large void sampling, sequential to spread reinserted points
            for j in idx:
                nc = 2000 if rng.random() < 0.7 else 500
                cand, md = largest_void(Xkeep, rng, n_sample=nc)
                order2 = np.argsort(md)
                if rng.random() < 0.75:
                    pick = int(order2[0])
                else:
                    pick = int(order2[int(rng.integers(1, min(8, nc)))])
                x[j] = cand[pick]
                Xkeep = np.vstack([Xkeep, x[j:j + 1]])
        else:
            # regional cap shake + rattler teleport
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            moff = G.max()
            cnt = (G > moff - 1e-4).sum(axis=1)
            # rattlers: minimal maxper
            maxper = G.max(axis=1)
            if rng.random() < 0.5:
                c = rng.standard_normal(d)
                c /= np.linalg.norm(c)
                dots = x @ c
                thr = float(rng.uniform(0.60, 0.93))
                mask = dots > thr
                if int(mask.sum()) == 0:
                    mask[int(rng.integers(n))] = True
                sigma = float(rng.uniform(0.06, 0.30))
                x[mask] += sigma * rng.standard_normal((int(mask.sum()), d))
                x /= np.linalg.norm(x, axis=1, keepdims=True)
            else:
                # teleport lowest-contact points to voids
                nl = int(rng.integers(1, 4))
                low = np.argsort(cnt + maxper)[:nl]
                keep_mask = np.ones(n, dtype=bool)
                keep_mask[low] = False
                Xkeep = x[keep_mask]
                for j in low:
                    cand, md = largest_void(Xkeep, rng, n_sample=1500)
                    x[j] = cand[int(np.argmin(md))]
                    Xkeep = np.vstack([Xkeep, x[j:j + 1]])

        if time.time() > SEARCH_END - 1.5:
            break
        # choose annealer
        loss_choice = rng.random()
        try:
            if loss_choice < 0.55:
                k0 = float(rng.choice(np.array([8.0, 12.0, 18.0, 25.0])))
                k1 = float(rng.choice(np.array([150.0, 250.0, 400.0, 600.0])))
                lr0 = float(rng.choice(np.array([0.02, 0.035, 0.05])))
                lr1 = float(rng.choice(np.array([0.001, 0.002, 0.004])))
                mu = float(rng.choice(np.array([0.0, 0.8, 0.9])))
                noise = float(rng.choice(np.array([0.0, 0.0, 0.004, 0.010])))
                iters = int(rng.choice(np.array([500, 800, 1100, 1400])))
                xo = anneal_exp(x, iters, k0, k1, lr0, lr1, mu, noise, rng)
            else:
                m0 = float(rng.choice(np.array([0.03, 0.06, 0.10])))
                m1 = float(rng.choice(np.array([0.003, 0.008, 0.015])))
                p = float(rng.choice(np.array([3.0, 5.0, 8.0])))
                lr0 = float(rng.choice(np.array([0.02, 0.04])))
                lr1 = float(rng.choice(np.array([0.001, 0.003])))
                mu = float(rng.choice(np.array([0.0, 0.85])))
                iters = int(rng.choice(np.array([500, 900, 1300])))
                xo = anneal_power(x, iters, m0, m1, p, lr0, lr1, mu, rng)
            # greedy max refinement
            xo = greedy_max_refine(xo, 180, 0.02, tol0=0.02)
        except Exception:
            continue
        v = max_cos(xo)
        # keep top 5
        if v < cands[-1][0] - 1e-12 or len(cands) < 5:
            cands.append((v, xo.copy()))
            cands = sorted(cands, key=lambda z: z[0])[:5]
        if v < inc_val - 1e-12:
            inc_val = v
            incumbent = xo.copy()
        if trial % 22 == 0:
            xr = rng.standard_normal((n, d))
            xr /= np.linalg.norm(xr, axis=1, keepdims=True)
            try:
                xo2 = anneal_exp(xr, 800, 14.0, 300.0, 0.04, 0.002, 0.85, 0.0, rng)
                xo2 = greedy_max_refine(xo2, 150, 0.02)
            except Exception:
                continue
            v2 = max_cos(xo2)
            if v2 < cands[-1][0] - 1e-12:
                cands.append((v2, xo2.copy()))
                cands = sorted(cands, key=lambda z: z[0])[:5]
            if v2 < inc_val - 1e-12:
                inc_val = v2
                incumbent = xo2.copy()
        if time.time() > SEARCH_END:
            break

    cands = sorted(cands, key=lambda z: z[0])[:5]
    # final polish with remaining time
    time_left = 172 - (time.time() - t0)
    if time_left < 12:
        time_left = 12
    # allocate more to best
    per = time_left / 3.2
    polished = []
    for i, (v, xc) in enumerate(cands[:3]):
        if time.time() - t0 > 171:
            polished.append((v, xc))
            continue
        budget = per * (1.6 if i == 0 else 0.8)
        budget = min(max(budget, 5), 45)
        try:
            xp = lp_polish(xc, budget)
        except Exception:
            xp = xc
        vp = max_cos(xp)
        # rattler-void relocation loop for best candidate
        if i == 0:
            t_end = time.time() + min(12, budget * 0.5)
            while time.time() < t_end and time.time() - t0 < 171:
                G = xp @ xp.T
                np.fill_diagonal(G, -2.0)
                moff = G.max()
                # contacts with tight tol
                cnt = (G > moff - 1e-7).sum(axis=1)
                ratt = np.where(cnt == 0)[0]
                if len(ratt) == 0:
                    # try loosest points
                    maxper = G.max(axis=1)
                    ratt = np.argsort(maxper)[:1]
                    if moff - maxper[ratt[0]] < 5e-5:
                        break
                improved = False
                for r in ratt[:2]:
                    mask = np.ones(n, dtype=bool)
                    mask[r] = False
                    Xk = xp[mask]
                    cand, md = largest_void(Xk, rng, n_sample=4000)
                    best_c = cand[int(np.argmin(md))]
                    trial_x = xp.copy()
                    trial_x[r] = best_c
                    # quick re-polish 3s
                    try:
                        trial_p = lp_polish(trial_x, 3.0)
                    except Exception:
                        continue
                    vt = max_cos(trial_p)
                    if vt < vp - 2e-12:
                        xp, vp = trial_p, vt
                        improved = True
                        break
                if not improved:
                    break
        polished.append((vp, xp))
        if time.time() - t0 > 172:
            break
    # include unpolished rest
    for v, xc in cands[3:]:
        polished.append((v, xc))
    polished = sorted(polished, key=lambda z: z[0])
    best_final = polished[0][1]
    best_final = best_final / np.linalg.norm(best_final, axis=1, keepdims=True)
    # safety: return better of found and loaded best
    if base_best_val < max_cos(best_final):
        bb = base_best / np.linalg.norm(base_best, axis=1, keepdims=True)
        return bb
    return best_final
