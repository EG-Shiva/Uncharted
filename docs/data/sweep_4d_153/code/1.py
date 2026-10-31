import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def smooth_relax(x, iters, k0, k1, lr0, lr1, mu, noise, rng):
    n, d = x.shape
    v = np.zeros_like(x)
    # pre-allocate not needed
    for t in range(iters):
        frac = t / max(1, iters - 1)
        k = k0 * (k1 / k0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        moff = G.max()
        # weights, moff-shifted so max=1
        W = np.exp(k * (G - moff))
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        grad = (W @ x) / rs
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        v = mu * v + gt
        # bias correction like Adam without second moment
        # scale correction for early steps
        vc = v / (1.0 - mu ** (t + 1)) if mu > 0 else v
        x = x - lr * vc
        if noise > 0 and frac < 0.6:
            amp = noise * (1.0 - frac / 0.6)
            pert = rng.standard_normal(x.shape)
            pd = np.sum(pert * x, axis=1, keepdims=True)
            pert -= pd * x
            x = x + amp * pert
        # renormalize
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x


def lp_polish(x, seconds):
    """Trust-region LP: move points tangentially to lower the largest cosine."""
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
        best = np.load("best.npy")
    except Exception:
        try:
            best = np.load("record.npy")
        except Exception:
            best = rng.standard_normal((n, d))
    best = np.asarray(best, dtype=float)
    if best.shape != (n, d):
        # fallback: tile/truncate
        tmp = rng.standard_normal((n, d))
        best = tmp
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    best_val = max_cos(best)

    incumbent = best.copy()
    inc_val = best_val
    # keep top candidates
    cands = [(inc_val, incumbent.copy())]

    # search phase: reserve time for polish
    # total budget 170s, search until 100s
    SEARCH_END = t0 + 100
    trial = 0
    while time.time() < SEARCH_END:
        trial += 1
        # choose base: mostly incumbent, sometimes best or random cand
        rbase = rng.random()
        if rbase < 0.65:
            base = incumbent
        elif rbase < 0.85:
            base = best
        else:
            base = cands[int(rng.integers(len(cands)))][1]
        x = base.copy()

        mode = rng.random()
        if mode < 0.45:
            sigma = float(rng.choice(np.array([0.02, 0.05, 0.08, 0.12, 0.20])))
            x = x + sigma * rng.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        elif mode < 0.85:
            # kick worst points and reinsert into voids
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            maxper = G.max(axis=1)
            k = int(rng.integers(3, 10))
            idx = np.argsort(-maxper)[:k]
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[idx] = False
            Xkeep = x[keep_mask]  # (n-k, d)
            for j in idx:
                # sample candidates, pick farthest (min max dot)
                nc = 30
                cand = rng.standard_normal((nc, d))
                cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                dots = cand @ Xkeep.T
                md = dots.max(axis=1)
                # also avoid being close to already reinserted ones
                # pick best of samples
                pick = int(np.argmin(md))
                # with small prob pick 2nd best for diversity
                if rng.random() < 0.2:
                    order = np.argsort(md)
                    pick = int(order[int(rng.integers(1, min(5, nc)))])
                x[j] = cand[pick]
                # update Xkeep for next j to avoid clustering reinserted points
                Xkeep = np.vstack([Xkeep, x[j:j+1]])
        else:
            # regional shake inside a cap
            c = rng.standard_normal(d)
            c /= np.linalg.norm(c)
            dots = x @ c
            thr = float(rng.uniform(0.65, 0.95))
            mask = dots > thr
            if int(mask.sum()) == 0:
                mask[int(rng.integers(n))] = True
            sigma = float(rng.uniform(0.08, 0.30))
            x[mask] += sigma * rng.standard_normal((int(mask.sum()), d))
            x /= np.linalg.norm(x, axis=1, keepdims=True)

        # hyperparams for this trial
        k0 = float(rng.choice(np.array([10.0, 15.0, 20.0])))
        k1 = float(rng.choice(np.array([150.0, 250.0, 400.0])))
        lr0 = float(rng.choice(np.array([0.02, 0.03, 0.05])))
        lr1 = float(rng.choice(np.array([0.001, 0.002, 0.005])))
        mu = float(rng.choice(np.array([0.0, 0.8, 0.9])))
        noise = float(rng.choice(np.array([0.0, 0.0, 0.005, 0.01])))
        iters = int(rng.choice(np.array([800, 1200, 1600])))

        # quick time check: if little time left break
        if time.time() > SEARCH_END - 2:
            break
        try:
            xo = smooth_relax(x, iters, k0, k1, lr0, lr1, mu, noise, rng)
        except Exception:
            continue
        v = max_cos(xo)
        if v < inc_val - 1e-12:
            inc_val = v
            incumbent = xo.copy()
            cands.append((v, xo.copy()))
            # keep top 4
            cands = sorted(cands, key=lambda z: z[0])[:4]
        # occasional random-restart trial from scratch to explore new basin
        if trial % 18 == 0:
            xr = rng.standard_normal((n, d))
            xr /= np.linalg.norm(xr, axis=1, keepdims=True)
            try:
                xo2 = smooth_relax(xr, 1000, 15.0, 250.0, 0.04, 0.002, 0.85, 0.0, rng)
            except Exception:
                continue
            v2 = max_cos(xo2)
            if v2 < inc_val - 1e-12:
                inc_val = v2
                incumbent = xo2.copy()
                cands.append((v2, xo2.copy()))
                cands = sorted(cands, key=lambda z: z[0])[:4]

        # stop early if we already improved a lot? continue to use time
        if time.time() > SEARCH_END:
            break

    # final polishing phase with remaining time
    # sort candidates
    cands = sorted(cands, key=lambda z: z[0])
    # ensure best is included
    if best_val < cands[0][0]:
        cands = [(best_val, best.copy())] + cands
        cands = cands[:4]
    time_left = 170 - (time.time() - t0)
    if time_left < 10:
        time_left = 10
    # split time among top candidates
    per = time_left / min(3, len(cands))
    # polish best candidate longest
    polished_best = None
    polished_val = 1e9
    for i, (v, xc) in enumerate(cands[:3]):
        budget = per * (1.5 if i == 0 else 1.0)
        # cap to avoid overrun
        budget = min(budget, 60)
        try:
            xp = lp_polish(xc, budget)
        except Exception:
            xp = xc
        vp = max_cos(xp)
        if vp < polished_val:
            polished_val = vp
            polished_best = xp
        if time.time() - t0 > 172:
            break
    if polished_best is None:
        polished_best = incumbent
    # return the better of polished and original best
    if best_val < max_cos(polished_best):
        return best / np.linalg.norm(best, axis=1, keepdims=True)
    polished_best /= np.linalg.norm(polished_best, axis=1, keepdims=True)
    return polished_best
