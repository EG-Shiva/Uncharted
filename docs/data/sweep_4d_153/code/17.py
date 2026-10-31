import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def exp_relax(x, iters, k0, k1, lr0, lr1, mu, noise_amp, rng):
    n, d = x.shape
    v = np.zeros_like(x)
    for t in range(iters):
        frac = t / max(1, iters - 1)
        k = k0 * (k1 / k0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        W = np.exp(k * (G - m))
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        grad = (W @ x) / rs
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        v = mu * v + gt
        denom = 1.0 - mu ** (t + 1) if mu > 0 else 1.0
        vc = v / denom if mu > 0 else v
        x = x - lr * vc
        if noise_amp > 0 and frac < 0.5:
            amp = noise_amp * (1.0 - frac / 0.5)
            pert = rng.standard_normal(x.shape)
            pert -= np.sum(pert * x, axis=1, keepdims=True) * x
            x = x + amp * pert
        nrm = np.linalg.norm(x, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        x = x / nrm
    return x

def hard_descent(x, steps, lr, margin, rng):
    # subgradient on near-max pairs only, approximates minimax
    n, d = x.shape
    for t in range(steps):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        thr = m - margin
        # per-point worst gradient from contacts above thr
        # build weight mask
        M = (G >= thr).astype(float)
        # symmetrize already symmetric
        rs = M.sum(axis=1, keepdims=True) + 1e-12
        grad = (M @ x) / rs
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        # decay lr
        cur = lr * (1.0 - 0.7 * t / max(1, steps - 1))
        x = x - cur * gt
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x

def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 2e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        C = (x @ x.T)[iu]
        m = C.max()
        sel = C > m - max(25.0 * radius, 2e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 3000:
            # keep tightest 3000 to bound LP size
            idx = np.argsort(-C[sel])[:3000]
            I, J = I[idx], J[idx]
            p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-C[sel][:p] if len(C[sel])==p else -(x@x.T)[I,J],
                          A_eq=A_eq, b_eq=np.zeros(n),
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
            radius = min(radius * 1.6, 0.05)
        else:
            radius *= 0.5
    return x

def void_reinsert(x, kill_idx, rng, ntry=250):
    n, d = x.shape
    keep_mask = np.ones(n, dtype=bool)
    keep_mask[kill_idx] = False
    Xkeep = x[keep_mask]
    for j in kill_idx:
        cand = rng.standard_normal((ntry, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        dots = cand @ Xkeep.T
        md = dots.max(axis=1)
        order = np.argsort(md)
        # diversity: usually best, sometimes 2nd-5th
        if rng.random() < 0.75:
            pick = int(order[0])
        else:
            pick = int(order[int(rng.integers(1, min(8, ntry)))])
        x[j] = cand[pick]
        Xkeep = np.vstack([Xkeep, x[j:j+1]])
    return x

def solve():
    t0 = time.time()
    rng = np.random.default_rng(20261031)
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
    SEARCH_END = t0 + 128
    trial = 0
    while time.time() < SEARCH_END:
        trial += 1
        if time.time() > SEARCH_END - 2:
            break
        rb = rng.random()
        if rb < 0.50:
            base = incumbent
        elif rb < 0.70:
            base = best
        elif rb < 0.85:
            base = cands[int(rng.integers(len(cands)))][1]
        else:
            # fresh random to find new basin
            base = rng.standard_normal((n, d))
            base /= np.linalg.norm(base, axis=1, keepdims=True)
        x = base.copy()
        mode = rng.random()
        if mode < 0.35:
            # gaussian shake
            sigma = float(rng.choice(np.array([0.015, 0.04, 0.07, 0.11, 0.18, 0.28])))
            x = x + sigma * rng.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            iters = int(rng.choice(np.array([700, 1000, 1400])))
        elif mode < 0.75:
            # kill overloaded / worst points
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            maxper = G.max(axis=1)
            # count contacts near max for overload bonus
            m = G.max()
            cnt = (G >= m - 1e-7).sum(axis=1)
            score = maxper + 0.002 * cnt  # prefer overloaded
            # random k
            k = int(rng.integers(4, 16))
            # sometimes kill random cap cluster instead
            if rng.random() < 0.25:
                c = rng.standard_normal(d); c /= np.linalg.norm(c)
                dots = x @ c
                idx = np.argsort(-dots)[:k]
            else:
                idx = np.argsort(-score)[:k]
                # add 1-2 random rattler-adjacent for diversity
                if rng.random() < 0.4:
                    idx[-1] = int(rng.integers(n))
            x = void_reinsert(x, idx, rng, ntry=int(rng.choice(np.array([150, 250, 400]))))
            iters = int(rng.choice(np.array([1000, 1400, 1800])))
        else:
            # cap reshuffle: replace whole cap by fresh void-optimized points
            c = rng.standard_normal(d); c /= np.linalg.norm(c)
            dots = x @ c
            thr = float(rng.uniform(0.55, 0.92))
            mask = dots > thr
            if int(mask.sum()) < 3:
                mask[int(rng.integers(n))] = True
            idx = np.where(mask)[0]
            # randomize those points completely then greedy
            for j in idx:
                x[j] = rng.standard_normal(d)
                x[j] /= np.linalg.norm(x[j])
            x = void_reinsert(x, idx[:min(len(idx), 20)], rng, ntry=200)
            iters = int(rng.choice(np.array([1000, 1500])))

        k0 = float(rng.choice(np.array([8.0, 15.0, 25.0])))
        k1 = float(rng.choice(np.array([180.0, 350.0, 700.0, 1500.0])))
        lr0 = float(rng.choice(np.array([0.02, 0.04, 0.06])))
        lr1 = float(rng.choice(np.array([0.001, 0.0025, 0.005])))
        mu = float(rng.choice(np.array([0.0, 0.8, 0.9])))
        noise = float(rng.choice(np.array([0.0, 0.0, 0.004, 0.010])))
        try:
            xo = exp_relax(x, iters, k0, k1, lr0, lr1, mu, noise, rng)
            # short hard phase to approximate true minimax
            if rng.random() < 0.6:
                xo = hard_descent(xo, steps=120, lr=0.008, margin=0.015, rng=rng)
            else:
                xo = exp_relax(xo, 250, 400.0, 2500.0, 0.005, 0.001, 0.0, 0.0, rng)
        except Exception:
            continue
        v = max_cos(xo)
        if v < inc_val - 1e-12:
            inc_val = v
            incumbent = xo.copy()
            cands.append((v, xo.copy()))
            cands = sorted(cands, key=lambda z: z[0])[:5]
            # quick intermediate LP if very promising (within 2e-4 of best) to get true local value
            if v < best_val + 5e-4 and (time.time() - t0) < 110:
                try:
                    xt = lp_polish(xo, 2.0)
                    vt = max_cos(xt)
                    if vt < inc_val:
                        inc_val = vt
                        incumbent = xt.copy()
                    cands.append((vt, xt.copy()))
                    cands = sorted(cands, key=lambda z: z[0])[:5]
                except Exception:
                    pass
        elif rng.random() < 0.02 and v < inc_val + 0.005:
            # occasional sideways move to escape
            cands.append((v, xo.copy()))
            cands = sorted(cands, key=lambda z: z[0])[:6]
        if time.time() > SEARCH_END:
            break

    cands = sorted(cands, key=lambda z: z[0])
    # ensure original best included
    vals = [c[0] for c in cands]
    if best_val < min(vals) + 1e-12:
        # already have it or better; keep best as candidate too
        cands = [(best_val, best.copy())] + cands
        cands = cands[:5]
    time_left = 172 - (time.time() - t0)
    if time_left < 8:
        time_left = 8
    per = time_left / min(3, len(cands))
    polished_best = None
    polished_val = 1e9
    for i, (vv, xc) in enumerate(cands[:3]):
        budget = per * (1.6 if i == 0 else 1.0)
        budget = min(budget, 45)
        if time.time() - t0 > 171:
            break
        try:
            xp = lp_polish(xc, budget)
        except Exception:
            xp = xc
        vp = max_cos(xp)
        if vp < polished_val:
            polished_val = vp
            polished_best = xp.copy()
    if polished_best is None:
        polished_best = incumbent
        polished_val = max_cos(polished_best)
    # return best among polished and original
    if best_val < polished_val:
        return best / np.linalg.norm(best, axis=1, keepdims=True)
    polished_best /= np.linalg.norm(polished_best, axis=1, keepdims=True)
    return polished_best
