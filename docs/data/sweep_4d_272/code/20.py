import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def lp_polish(x, seconds, t_deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while True:
        if time.time() - start >= seconds:
            break
        if t_deadline is not None and time.time() >= t_deadline:
            break
        if radius < 1e-14:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
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

def quench_momentum(X, steps, T0, T1, lr0, mu, noise0, max_step, rng):
    n, d = X.shape
    V = np.zeros_like(X)
    for s in range(steps):
        frac = s / max(1, steps - 1)
        T = T0 * (1.0 - frac) + T1 * frac
        lr = lr0 * (1.0 - 0.6 * frac)
        noise_amp = noise0 * (1.0 - frac)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gmax = float(np.max(G))
        W = np.exp((G - gmax) / T)
        np.fill_diagonal(W, 0.0)
        S = float(np.sum(W) * 0.5)
        if S < 1e-30:
            gt = np.zeros_like(X)
        else:
            W /= S
            grad = W @ X
            dot = np.sum(grad * X, axis=1, keepdims=True)
            gt = grad - dot * X
        V = mu * V + lr * gt
        # clip per-row velocity
        vn = np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
        sc = np.minimum(1.0, max_step / vn)
        V = V * sc
        X = X - V
        if noise_amp > 0:
            N = rng.standard_normal(size=X.shape)
            nd = np.sum(N * X, axis=1, keepdims=True)
            Nt = N - nd * X
            X = X + (noise_amp * 0.5) * Nt
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X

def best_hole(others, rng, n_cand=8000, refine=120):
    d = others.shape[1]
    cand = rng.standard_normal(size=(n_cand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    S = cand @ others.T
    vals = np.max(S, axis=1)
    bi = int(np.argmin(vals))
    b = cand[bi].copy()
    bv = float(vals[bi])
    step = 0.04
    for _ in range(refine):
        prop = b + step * rng.standard_normal(size=(32, d))
        prop /= np.linalg.norm(prop, axis=1, keepdims=True)
        sc = np.max(prop @ others.T, axis=1)
        j = int(np.argmin(sc))
        if float(sc[j]) < bv:
            b = prop[j].copy()
            bv = float(sc[j])
            step *= 1.06
        else:
            step *= 0.96
        if step < 5e-6:
            break
    return b

def solve():
    t0 = time.time()
    HARD = t0 + 172.0
    EXPL_END = t0 + 128.0
    rng = np.random.default_rng(20261031)
    np.random.seed(1)
    X0 = None
    bv0 = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (272, 4):
                continue
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            v = max_cos(a)
            if v < bv0:
                bv0 = v
                X0 = a.copy()
        except Exception:
            continue
    if X0 is None:
        X0 = rng.standard_normal(size=(272, 4))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        bv0 = max_cos(X0)
    Xbest = X0.copy()
    best_val = float(bv0)
    n = 272
    trial = 0
    # short LP on start to get true local optimum quickly (5s)
    if HARD - time.time() > 60:
        Xlp0 = lp_polish(Xbest, seconds=4.0, t_deadline=HARD - 1.0)
        v0 = max_cos(Xlp0)
        if v0 < best_val:
            Xbest, best_val = Xlp0, v0
    sigmas = [0.04, 0.08, 0.13, 0.20]
    while time.time() < EXPL_END and time.time() < HARD - 42:
        trial += 1
        remain = min(EXPL_END - time.time(), HARD - 42 - time.time())
        if remain < 0.8:
            break
        mode = trial % 7
        X = Xbest.copy()
        if mode == 0:
            # multi worst-point void reinsertion
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            rowmax = np.max(G, axis=1)
            order = np.argsort(-rowmax)
            k = 2 + (trial % 2)  # 2-3 points
            targets = order[:8]
            # pick k among top 8
            pick = rng.choice(targets, size=k, replace=False)
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[pick] = False
            others = X[keep_mask]
            for idx in pick:
                h = best_hole(others, rng, n_cand=6000, refine=80)
                h = h + 0.003 * rng.standard_normal(4)
                h /= np.linalg.norm(h)
                # update others for next insertion
                others = np.vstack([others, h[None, :]])
                X[idx] = h
            X = X + 0.012 * rng.standard_normal(size=X.shape)
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        elif mode == 1:
            # rattler / low-contact relocation
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            gmax = float(np.max(G))
            contacts = np.sum(G > gmax - 1e-7, axis=1)
            ratt = np.where(contacts <= 1)[0]
            if len(ratt) > 0:
                tod = ratt[:min(4, len(ratt))]
                keep_mask = np.ones(n, dtype=bool)
                keep_mask[tod] = False
                others = X[keep_mask]
                for idx in tod:
                    h = best_hole(others, rng, n_cand=6000, refine=80)
                    h /= np.linalg.norm(h)
                    others = np.vstack([others, h[None, :]])
                    X[idx] = h
                X = X + 0.010 * rng.standard_normal(size=X.shape)
                X /= np.linalg.norm(X, axis=1, keepdims=True)
            else:
                sigma = sigmas[trial % len(sigmas)]
                X = X + sigma * rng.standard_normal(size=X.shape)
                X /= np.linalg.norm(X, axis=1, keepdims=True)
        elif mode == 2:
            # cap cluster resampling: delete k nearest to random dir, regrow in voids
            r = rng.standard_normal(4)
            r /= np.linalg.norm(r)
            dots = X @ r
            k = int(rng.integers(7, 15))
            idxs = np.argsort(-dots)[:k]
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[idxs] = False
            others = X[keep_mask]
            # sequential greedy insertion
            newpts = []
            for _ in range(k):
                base = others if len(newpts) == 0 else np.vstack([others] + newpts)
                h = best_hole(base, rng, n_cand=6000, refine=60)
                h = h + 0.005 * rng.standard_normal(4)
                h /= np.linalg.norm(h)
                newpts.append(h[None, :])
            for ii, idx in enumerate(idxs):
                X[idx] = newpts[ii][0]
            X = X + 0.015 * rng.standard_normal(size=X.shape)
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        elif mode == 3:
            # break the current max pair: move one endpoint to hole + patch kick
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            ij = int(np.argmax(G))
            ii, jj = ij // n, ij % n
            # move jj to hole
            mask = np.ones(n, dtype=bool)
            mask[jj] = False
            h = best_hole(X[mask], rng, n_cand=8000, refine=100)
            X[jj] = h
            # kick neighbours of ii
            neigh = np.where(G[ii] > float(np.max(G)) - 0.02)[0]
            if len(neigh) > 0:
                X[neigh] = X[neigh] + 0.05 * rng.standard_normal(size=(len(neigh), 4))
            X = X + 0.015 * rng.standard_normal(size=X.shape)
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        elif mode == 4:
            # random subset strong kick (topology change)
            sel = rng.choice(n, size=36, replace=False)
            X[sel] = X[sel] + 0.30 * rng.standard_normal(size=(36, 4))
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        else:
            sigma = sigmas[(trial // 2) % len(sigmas)]
            if mode == 6:
                # global rotation of random half + noise
                sel = rng.choice(n, size=n // 2, replace=False)
                # random SO(4) via QR
                Q, _ = np.linalg.qr(rng.standard_normal((4, 4)))
                X[sel] = X[sel] @ Q.T
                X = X + (sigma * 0.5) * rng.standard_normal(size=X.shape)
            else:
                X = X + sigma * rng.standard_normal(size=X.shape)
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        # choose quench params
        if remain < 6:
            steps = 300
        elif remain < 15:
            steps = 450
        else:
            steps = 600 + int(rng.integers(0, 200))
        T0 = float(rng.choice([0.010, 0.015, 0.022]))
        T1 = float(rng.choice([0.0015, 0.0025, 0.004]))
        lr0 = float(rng.choice([7.0, 10.0, 14.0]))
        mu = float(rng.choice([0.0, 0.5, 0.8]))
        noise0 = float(rng.choice([0.008, 0.014, 0.020]))
        Xq = quench_momentum(X, steps=steps, T0=T0, T1=T1, lr0=lr0, mu=mu, noise0=noise0, max_step=0.02, rng=rng)
        v = max_cos(Xq)
        # if promising, short LP refine to true value
        if v < best_val + 0.0015 and time.time() < HARD - 40:
            # very short LP (1.2s) to judge basin
            Xqr = lp_polish(Xq, seconds=1.2, t_deadline=HARD - 38)
            vr = max_cos(Xqr)
            if vr < v:
                Xq, v = Xqr, vr
        if v < best_val:
            Xbest = Xq
            best_val = v
            # opportunistic slightly longer polish if big progress
            if best_val < 0.8978 and time.time() < EXPL_END - 8:
                Xp = lp_polish(Xbest, seconds=2.5, t_deadline=HARD - 35)
                vp = max_cos(Xp)
                if vp < best_val:
                    Xbest, best_val = Xp, vp
    remain = HARD - time.time() - 1.5
    if remain > 4:
        secs = min(remain, 42.0)
        Xp = lp_polish(Xbest, seconds=secs, t_deadline=HARD - 0.8)
        vp = max_cos(Xp)
        if vp < best_val and np.all(np.isfinite(Xp)):
            Xbest, best_val = Xp, vp
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    bad = ~np.all(np.isfinite(Xbest), axis=1)
    if np.any(bad):
        for i in np.where(bad)[0]:
            vv = rng.standard_normal(4)
            vv /= np.linalg.norm(vv)
            Xbest[i] = vv
    return Xbest
