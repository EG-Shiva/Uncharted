import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def contact_stats(x, tol=1e-9):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = G.max()
    C = G > m - tol
    per = C.sum(axis=1)
    maxper = G.max(axis=1)
    return m, per, maxper, G

def smooth_relax(x, iters, k0, k1, lr0, lr1, mu, noise, rng):
    n, d = x.shape
    v = np.zeros_like(x)
    for t in range(iters):
        frac = t / max(1, iters - 1)
        k = k0 * (k1 / k0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        moff = float(G.max())
        # exponential weights, shift for stability
        W = np.exp(k * (G - moff))
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        grad = (W @ x) / rs
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        v = mu * v + gt
        vc = v / (1.0 - mu ** (t + 1)) if mu > 0 else v
        x = x - lr * vc
        if noise > 0 and frac < 0.6:
            amp = noise * (1.0 - frac / 0.6)
            pert = rng.standard_normal(x.shape)
            pd = np.sum(pert * x, axis=1, keepdims=True)
            pert -= pd * x
            x = x + amp * pert
        nrm = np.linalg.norm(x, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        x = x / nrm
    return x

def farthest_candidates(Xkeep, k_need, rng, pool=400):
    d = Xkeep.shape[1]
    out = []
    cur = Xkeep
    for _ in range(k_need):
        cand = rng.standard_normal((pool, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        dots = cand @ cur.T
        md = dots.max(axis=1)
        # pick best, sometimes 2nd-5th for diversity
        if rng.random() < 0.8:
            pick = int(np.argmin(md))
        else:
            order = np.argsort(md)
            pick = int(order[int(rng.integers(1, min(6, pool)))])
        out.append(cand[pick])
        cur = np.vstack([cur, cand[pick:pick+1]])
    return np.array(out)

def largest_void(Xkeep, rng, pool=4000):
    d = Xkeep.shape[1]
    cand = rng.standard_normal((pool, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    md = (cand @ Xkeep.T).max(axis=1)
    return cand[int(np.argmin(md))]

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
    rng = np.random.default_rng(0)
    n, d = 153, 4
    try:
        best0 = np.load("best.npy")
    except Exception:
        try:
            best0 = np.load("record.npy")
        except Exception:
            best0 = rng.standard_normal((n, d))
    best0 = np.asarray(best0, dtype=float)
    if best0.shape != (n, d):
        best0 = rng.standard_normal((n, d))
    best0 = best0 / np.linalg.norm(best0, axis=1, keepdims=True)
    best0_val = max_cos(best0)

    incumbent = best0.copy()
    inc_val = best0_val
    best_ever = best0.copy()
    best_ever_val = best0_val
    cands = [(best0_val, best0.copy())]

    SEARCH_END = t0 + 118
    trial = 0
    # annealing temperature for sideways moves
    T0 = 0.004
    T1 = 0.0002
    while time.time() < SEARCH_END:
        trial += 1
        frac_global = (time.time() - t0) / max(1e-9, (SEARCH_END - t0))
        Temp = T0 * (T1 / T0) ** frac_global
        # choose base
        rb = rng.random()
        if rb < 0.6:
            base = incumbent
        elif rb < 0.8:
            base = best_ever
        else:
            base = cands[int(rng.integers(len(cands)))][1]
        x = base.copy()
        _, per, maxper, Gfull = contact_stats(x)

        mode = rng.random()
        if mode < 0.30:
            # global jitter, sometimes large to change topology
            sig_choices = np.array([0.03, 0.06, 0.10, 0.18, 0.30, 0.45])
            sigma = float(rng.choice(sig_choices))
            x = x + sigma * rng.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        elif mode < 0.62:
            # coordinated cluster surgery: remove cap around worst/hub point
            # pick center: worst maxper or random hub
            if rng.random() < 0.7:
                center_idx = int(np.argmax(maxper))
            else:
                hubs = np.where(per >= 7)[0]
                if len(hubs) > 0:
                    center_idx = int(hubs[int(rng.integers(len(hubs)))])
                else:
                    center_idx = int(np.argmax(maxper))
            cvec = x[center_idx]
            dots = x @ cvec
            thr = float(rng.uniform(0.55, 0.90))
            mask = dots > thr
            idx = np.where(mask)[0]
            # limit size 6..22
            if len(idx) < 6:
                order = np.argsort(-dots)
                idx = order[:int(rng.integers(6, 12))]
            elif len(idx) > 22:
                order = np.argsort(-dots)
                idx = order[:int(rng.integers(12, 22))]
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[idx] = False
            Xkeep = x[keep_mask]
            # reinsert farthest
            re = farthest_candidates(Xkeep, len(idx), rng, pool=400)
            x[idx] = re
        elif mode < 0.82:
            # rattler / low-contact void filling + hub dispersal
            # replace k worst+lowest contact points by voids
            k = int(rng.integers(2, 8))
            # score: high maxper and high per -> bad; low per but not rattler?
            # pick k indices with largest maxper
            order_bad = np.argsort(-maxper)
            # also include rattlers (per==0)
            ratt = np.where(per == 0)[0]
            idx = list(order_bad[:k])
            if len(ratt) > 0:
                for r in ratt:
                    if r not in idx:
                        idx[-1] = int(r)
                        break
            idx = np.array(idx, dtype=int)
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[idx] = False
            Xkeep = x[keep_mask]
            cur = Xkeep.copy()
            for j in idx:
                vdir = largest_void(cur, rng, pool=3000)
                # small random offset to avoid exact symmetry
                vdir = vdir + 0.02 * rng.standard_normal(d)
                vdir /= np.linalg.norm(vdir)
                x[j] = vdir
                cur = np.vstack([cur, vdir[None, :]])
        else:
            # regional shake + hub push apart
            c = rng.standard_normal(d); c /= np.linalg.norm(c)
            dots = x @ c
            thr = float(rng.uniform(0.60, 0.92))
            mask = dots > thr
            if int(mask.sum()) == 0:
                mask[int(rng.integers(n))] = True
            sigma = float(rng.uniform(0.10, 0.35))
            x[mask] += sigma * rng.standard_normal((int(mask.sum()), d))
            # push hubs away from their contact neighbours
            m_now = float(Gfull.max())
            hubs = np.where(per >= 7)[0]
            for h in hubs:
                if rng.random() < 0.5:
                    nbrs = np.where(Gfull[h] > m_now - 2e-4)[0]
                    if len(nbrs) > 0:
                        push = x[h] - x[nbrs].mean(axis=0)
                        # tangential component
                        push = push - np.dot(push, x[h]) * x[h]
                        x[h] = x[h] + float(rng.uniform(0.05, 0.2)) * push
            x /= np.linalg.norm(x, axis=1, keepdims=True)

        k0 = float(rng.choice(np.array([8.0, 12.0, 18.0])))
        k1 = float(rng.choice(np.array([120.0, 220.0, 350.0, 500.0])))
        lr0 = float(rng.choice(np.array([0.02, 0.04, 0.06])))
        lr1 = float(rng.choice(np.array([0.001, 0.002, 0.004])))
        mu = float(rng.choice(np.array([0.0, 0.8, 0.9])))
        noise = float(rng.choice(np.array([0.0, 0.0, 0.004, 0.009])))
        iters = int(rng.choice(np.array([600, 900, 1300])))

        if time.time() > SEARCH_END - 2:
            break
        try:
            xo = smooth_relax(x, iters, k0, k1, lr0, lr1, mu, noise, rng)
        except Exception:
            continue
        v = max_cos(xo)
        # update best ever
        if v < best_ever_val - 1e-12:
            best_ever_val = v
            best_ever = xo.copy()
            cands.append((v, xo.copy()))
            cands = sorted(cands, key=lambda z: z[0])[:5]
        # basin-hopping acceptance for incumbent (allow sideways/uphill)
        if v < inc_val - 1e-12:
            inc_val = v
            incumbent = xo.copy()
        else:
            # Metropolis-like
            if rng.random() < np.exp(-(v - inc_val) / max(Temp, 1e-9)):
                # only accept if not too bad (within 0.02) to avoid drifting far
                if v < inc_val + 0.02:
                    inc_val = v
                    incumbent = xo.copy()
        # periodic random restart from scratch for new basin
        if trial % 22 == 0 and time.time() < SEARCH_END - 10:
            xr = rng.standard_normal((n, d))
            xr /= np.linalg.norm(xr, axis=1, keepdims=True)
            try:
                xo2 = smooth_relax(xr, 900, 12.0, 250.0, 0.04, 0.002, 0.85, 0.0, rng)
            except Exception:
                continue
            v2 = max_cos(xo2)
            if v2 < best_ever_val - 1e-12:
                best_ever_val = v2
                best_ever = xo2.copy()
                cands.append((v2, xo2.copy()))
                cands = sorted(cands, key=lambda z: z[0])[:5]
            if v2 < inc_val + 0.005 and rng.random() < 0.3:
                inc_val = v2
                incumbent = xo2.copy()

    cands = sorted(cands, key=lambda z: z[0])
    # ensure best0 included
    vals = [c[0] for c in cands]
    if best0_val <= min(vals) + 1e-15:
        pass
    else:
        cands.append((best0_val, best0.copy()))
        cands = sorted(cands, key=lambda z: z[0])[:5]

    time_left = 170 - (time.time() - t0)
    if time_left < 12:
        time_left = 12
    per_budget = time_left / min(3, len(cands))
    polished_best = None
    polished_val = 1e9
    for i, (vv, xc) in enumerate(cands[:3]):
        budget = per_budget * (1.6 if i == 0 else 1.0)
        budget = min(budget, 55)
        if time.time() - t0 > 172:
            break
        try:
            xp = lp_polish(xc, budget)
        except Exception:
            xp = xc
        vp = max_cos(xp)
        if vp < polished_val:
            polished_val = vp
            polished_best = xp
    if polished_best is None:
        polished_best = best_ever
    polished_best = polished_best / np.linalg.norm(polished_best, axis=1, keepdims=True)
    # safety: return best among polished and loaded best
    if best0_val < max_cos(polished_best):
        return best0 / np.linalg.norm(best0, axis=1, keepdims=True)
    return polished_best
