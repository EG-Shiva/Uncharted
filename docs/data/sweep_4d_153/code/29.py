import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def contact_degrees(x, tol=1e-7):
    n = x.shape[0]
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    sel = G > (m - tol)
    # count per point (symmetric, count neighbors)
    deg = np.sum(sel, axis=1)
    # pairs
    iu = np.triu_indices(n, 1)
    nmatch = int(np.sum(G[iu] > (m - tol)))
    return deg, m, nmatch

def smooth_relax(x, iters, k0, k1, lr0, lr1, mu, noise, rng):
    n, d = x.shape
    v = np.zeros_like(x)
    for t in range(iters):
        frac = t / max(1, iters - 1)
        k = k0 * (k1 / k0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        moff = float(np.max(G))
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
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x

def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        cfull = (x @ x.T)[iu]
        m = float(np.max(cfull))
        sel = cfull > m - max(20 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cfull[sel], A_eq=A_eq, b_eq=np.zeros(n),
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
            x, best = y, vv
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def refine_hole(y0, X, iters=250, lr0=0.08, lr1=0.002):
    y = y0 / np.linalg.norm(y0)
    for t in range(iters):
        frac = t / max(1, iters - 1)
        lr = lr0 * (lr1 / lr0) ** frac
        dots = X @ y
        j = int(np.argmax(dots))
        g = X[j] - dots[j] * y
        y = y - lr * g
        nrm = np.linalg.norm(y)
        if nrm < 1e-12:
            break
        y /= nrm
    return y

def solve():
    t0 = time.time()
    rng = np.random.default_rng(7)
    n, d = 153, 4
    try:
        best = np.load("best.npy")
    except Exception:
        try:
            best = np.load("record.npy")
        except Exception:
            best = rng.standard_normal((n, d))
    best = np.asarray(best, dtype=float)
    if best.shape != (n, d):
        best = rng.standard_normal((n, d))
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    best_val = max_cos(best)
    incumbent = best.copy()
    inc_val = best_val
    cands = [(inc_val, incumbent.copy())]

    DEADLINE = t0 + 172
    SEARCH_END = t0 + 108
    Ks = [8, 12, 20, 30]
    trial = 0
    # precompute for speed
    while time.time() < SEARCH_END:
        trial += 1
        if time.time() > SEARCH_END - 3:
            break
        # choose base
        rbase = rng.random()
        if rbase < 0.7:
            base = incumbent
        elif rbase < 0.9:
            base = best
        else:
            base = cands[int(rng.integers(len(cands)))][1]
        xbase = base.copy()
        # contact analysis on base
        try:
            deg, mcur, _ = contact_degrees(xbase, tol=1e-7)
        except Exception:
            deg = np.zeros(n, dtype=int)
        rattlers = np.where(deg == 0)[0].tolist()
        order_deg = np.argsort(-deg)
        K = Ks[trial % len(Ks)]
        # small jitter
        if rng.random() < 0.3:
            K = int(K + rng.integers(-2, 3))
            K = max(6, min(32, K))
        hub_mode = (trial % 2 == 0)
        # occasional cap centered on random direction vs hub
        R = set()
        for r in rattlers:
            R.add(int(r))
        if hub_mode:
            # hub-focused removal: add top hubs + their nearest neighbours
            for idx in order_deg:
                if len(R) >= K:
                    break
                R.add(int(idx))
                if len(R) >= K:
                    break
                # add nearest neighbour of this hub (most clustered)
                dots = xbase @ xbase[int(idx)]
                dots[int(idx)] = -2
                # mask out already in R
                for qq in list(R):
                    dots[qq] = -2
                nn = int(np.argmax(dots))
                R.add(nn)
                if len(R) >= K:
                    break
            R = list(R)[:K]
            # if still short, add spatially clustered: nearest to first hub
            while len(R) < K:
                h = R[0]
                dots = xbase @ xbase[h]
                for qq in R:
                    dots[qq] = -2
                R.append(int(np.argmax(dots)))
        else:
            # cap removal: pick center
            cr = rng.random()
            if cr < 0.4 and len(rattlers) > 0:
                center = xbase[rattlers[0]]
            elif cr < 0.75:
                # overloaded hub center
                center = xbase[int(order_deg[0])]
            else:
                center = rng.standard_normal(d)
                center /= np.linalg.norm(center)
            dots = xbase @ center
            cap_idx = np.argsort(-dots)[:K].tolist()
            R = cap_idx
            # force include rattler
            if len(rattlers) > 0 and rattlers[0] not in R:
                R[-1] = int(rattlers[0])
        R = np.array(sorted(set(int(v) for v in R)), dtype=int)
        # adjust to exactly K (pad/truncate)
        if len(R) > K:
            R = R[:K]
        while len(R) < K:
            cand_extra = int(rng.integers(n))
            if cand_extra not in set(R.tolist()):
                R = np.append(R, cand_extra)
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[R] = False
        remain = xbase[keep_mask].copy()
        # annealed exponential repulsion on survivors
        iters_rem = int(rng.choice(np.array([800, 1000, 1300, 1500])))
        try:
            remain = smooth_relax(remain, iters_rem, 30.0, 300.0, 0.05, 0.003, 0.9, 0.0, rng)
        except Exception:
            continue
        cur = remain
        # reinsert one at a time
        ok = True
        for s in range(K):
            if time.time() > SEARCH_END - 1:
                ok = False
                break
            mcur_n = cur.shape[0]
            # 30000 random vectors
            cand = rng.standard_normal((30000, d))
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            dotsm = cand @ cur.T
            md = dotsm.max(axis=1)
            order = np.argsort(md)
            top = order[:8]
            best_y = None
            best_sc = 1e9
            # refine top 5
            for oi in top[:5]:
                y0 = cand[int(oi)]
                try:
                    yopt = refine_hole(y0, cur, iters=250, lr0=0.08, lr1=0.002)
                except Exception:
                    yopt = y0
                sc = float(np.max(cur @ yopt))
                if sc < best_sc:
                    best_sc = sc
                    best_y = yopt
            if best_y is None:
                best_y = cand[int(top[0])]
            cur = np.vstack([cur, best_y.reshape(1, d)])
            # short smooth relaxation
            try:
                cur = smooth_relax(cur, 150, 50.0, 150.0, 0.02, 0.005, 0.85, 0.0, rng)
            except Exception:
                pass
        if not ok:
            break
        # light smooth before LP
        try:
            cur = smooth_relax(cur, 300, 20.0, 250.0, 0.03, 0.003, 0.85, 0.0, rng)
        except Exception:
            pass
        # LP polish after full reinsertion
        budget = 2.0 if K <= 12 else 2.5
        # leave time for final polish
        if time.time() + budget > SEARCH_END + 4:
            budget = max(0.5, SEARCH_END + 4 - time.time())
        try:
            cur = lp_polish(cur, budget)
        except Exception:
            pass
        try:
            v = max_cos(cur)
        except Exception:
            continue
        if v < inc_val - 1e-12:
            inc_val = v
            incumbent = cur.copy()
            cands.append((v, cur.copy()))
            cands = sorted(cands, key=lambda z: z[0])[:4]
        elif v < cands[-1][0] - 1e-12 and len(cands) < 4:
            cands.append((v, cur.copy()))
            cands = sorted(cands, key=lambda z: z[0])[:4]
        # also try occasional pure smooth on incumbent to keep sharp
        if trial % 10 == 0 and time.time() < SEARCH_END - 5:
            try:
                tmp = incumbent.copy()
                tmp = smooth_relax(tmp, 600, 15.0, 300.0, 0.04, 0.002, 0.85, 0.0, rng)
                tmp = lp_polish(tmp, 1.0)
                vv = max_cos(tmp)
                if vv < inc_val - 1e-12:
                    inc_val = vv
                    incumbent = tmp.copy()
                    cands.append((vv, tmp.copy()))
                    cands = sorted(cands, key=lambda z: z[0])[:4]
            except Exception:
                pass

    # final polishing phase
    cands = sorted(cands, key=lambda z: z[0])
    # ensure best included
    if best_val < cands[0][0] + 1e-15:
        # dedup
        cands = [(best_val, best.copy())] + cands
        cands = sorted(cands, key=lambda z: z[0])[:4]
    polished_best = None
    polished_val = 1e9
    # split remaining time
    rem_time = DEADLINE - time.time() - 2
    if rem_time < 5:
        rem_time = 5
    per = rem_time / min(3, len(cands))
    for i, (vv, xc) in enumerate(cands[:3]):
        if time.time() > DEADLINE - 3:
            break
        budget = per * (1.6 if i == 0 else 1.0)
        budget = min(budget, 45)
        # extra smooth before final LP for diversity
        try:
            if i > 0:
                xc = smooth_relax(xc.copy(), 400, 20.0, 300.0, 0.02, 0.002, 0.85, 0.0, rng)
        except Exception:
            pass
        try:
            xp = lp_polish(xc, budget)
        except Exception:
            xp = xc
        vp = max_cos(xp)
        if vp < polished_val:
            polished_val = vp
            polished_best = xp.copy()
        if time.time() > DEADLINE - 2:
            break
    if polished_best is None:
        polished_best = incumbent
    # return best among polished, incumbent, original best
    final_cands = [(polished_val if polished_best is not None else 1e9, polished_best),
                   (inc_val, incumbent), (best_val, best)]
    # filter None
    valid = [(v, x) for v, x in final_cands if x is not None]
    valid = sorted(valid, key=lambda z: z[0])
    out = valid[0][1]
    out = out / np.linalg.norm(out, axis=1, keepdims=True)
    return out
