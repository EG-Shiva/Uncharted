import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    end = time.time() + seconds
    iu = np.triu_indices(n, 1)
    # precompute iu arrays
    while time.time() < end and radius > 1e-14:
        c_full = (x @ x.T)[iu]
        m = float(c_full.max())
        win = max(20.0 * radius, 1e-9)
        sel = c_full > m - win
        # cap constraints for speed
        if np.count_nonzero(sel) > 4000:
            # keep tightest 4000
            idx = np.where(sel)[0]
            order = np.argsort(c_full[idx])[::-1]
            keep = np.zeros_like(sel, dtype=bool)
            keep[idx[order[:4000]]] = True
            sel = keep
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
            res = linprog(obj, A_ub=A_ub, b_ub=-c_full[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            break
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
        if time.time() >= end:
            break
    return x


def adam_smooth(y0, steps, T0, T1, lr0, lr1):
    y = y0 / np.linalg.norm(y0, axis=1, keepdims=True)
    n, d = y.shape
    m1 = np.zeros_like(y)
    v1 = np.zeros_like(y)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        T = T0 * (T1 / T0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = y @ y.T
        np.fill_diagonal(G, -np.inf)
        mval = np.max(G)
        # softmax weights
        E = np.exp((G - mval) / T)
        np.fill_diagonal(E, 0.0)
        S = E.sum()
        if not np.isfinite(S) or S == 0:
            break
        W = E / S
        Grad = W @ y
        dot = np.sum(Grad * y, axis=1, keepdims=True)
        Gt = Grad - dot * y
        m1 = b1 * m1 + (1 - b1) * Gt
        v1 = b2 * v1 + (1 - b2) * (Gt * Gt)
        mhat = m1 / (1 - b1 ** (s + 1))
        vhat = v1 / (1 - b2 ** (s + 1))
        step = lr * mhat / (np.sqrt(vhat) + eps)
        # safety clip per-point step
        y = y - step
        nrm = np.linalg.norm(y, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        y /= nrm
    return y


def row_maxima(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1), G


def solve():
    t0 = time.time()
    LIMIT = 172.0
    rng = np.random.default_rng(0)
    # load start
    best_x = None
    best_v = np.inf
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (256, 6):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = max_cos(a)
                if v < best_v:
                    best_v = v
                    best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        best_x = rng.standard_normal((256, 6))
        best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
        best_v = max_cos(best_x)
    x_best = best_x.copy()
    cur_best = best_v

    it = 0
    # reserve ~22s for final polish
    while time.time() - t0 < LIMIT - 22:
        it += 1
        remain = LIMIT - 22 - (time.time() - t0)
        if remain < 3:
            break
        # analyse contacts of x_best occasionally
        Gtmp = x_best @ x_best.T
        np.fill_diagonal(Gtmp, -2.0)
        mtmp = Gtmp.max()
        rmax = Gtmp.max(axis=1)
        order = np.argsort(rmax)  # ascending
        # choose operator
        u = rng.random()
        x_try = x_best.copy()
        if u < 0.45:
            sigmas = [0.015, 0.04, 0.10, 0.22]
            sigma = sigmas[it % len(sigmas)]
            # bias noise to high-contact points: larger noise where rmax close to mtmp
            prox = np.clip((rmax - (mtmp - 0.05)) / 0.05, 0, 1)  # 0..1
            scale = (0.5 + 1.5 * prox)[:, None]
            x_try = x_try + sigma * scale * rng.standard_normal(x_try.shape)
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            T0, T1 = 0.015, 0.002
            steps, lr0, lr1 = 700, 0.02, 0.002
            if sigma > 0.15:
                T0 = 0.025
                steps = 900
        elif u < 0.80:
            # subset reinit: pick k worst + random
            ks = [8, 16, 32, 48]
            k = ks[rng.integers(0, len(ks))]
            # pool = top 64 by rmax + 5 lowest (rattlers)
            worst_pool = order[-64:]
            rattler_pool = order[:8]
            sel = set()
            # sample most from worst
            nw = min(k, k * 3 // 4)
            idx_w = rng.choice(worst_pool, size=nw, replace=False)
            for i in idx_w:
                sel.add(int(i))
            # rest random
            while len(sel) < k:
                sel.add(int(rng.integers(0, 256)))
            # also sometimes include a rattler
            if rng.random() < 0.5:
                sel.add(int(rng.choice(rattler_pool)))
                # drop one random to keep size ~k+1 (allow +1)
            idx = np.array(sorted(sel), dtype=int)
            rnd = rng.standard_normal((len(idx), 6))
            rnd /= np.linalg.norm(rnd, axis=1, keepdims=True)
            x_try[idx] = rnd
            # small global jitter
            x_try = x_try + 0.01 * rng.standard_normal(x_try.shape)
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            T0, T1 = 0.02, 0.0015
            steps, lr0, lr1 = 1000, 0.025, 0.0015
        else:
            # hole search + rattler reinsertion: randomize low-contact points to holes
            # find holes by sampling
            S = 6000
            cand = rng.standard_normal((S, 6))
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            # score holes against fixed core (all points) : max cosine
            # use chunking for speed
            scores = np.empty(S)
            for a in range(0, S, 1000):
                b = min(a + 1000, S)
                scores[a:b] = (cand[a:b] @ x_best.T).max(axis=1)
            # pick diverse good holes via greedy
            k = int(rng.choice([6, 12, 20]))
            # worst indices to replace: lowest rmax (rattlers) + random
            rep_idx = order[:k].copy()
            # greedy pick k holes far from each other
            hole_order = np.argsort(scores)  # best first
            chosen = []
            for h in hole_order:
                c = cand[h]
                ok = True
                for q in chosen:
                    if abs(float(c @ cand[q])) > 0.7:
                        ok = False
                        break
                if ok:
                    chosen.append(int(h))
                    if len(chosen) >= k:
                        break
            for ii, h in zip(rep_idx, chosen):
                x_try[int(ii)] = cand[h]
            x_try = x_try + 0.02 * rng.standard_normal(x_try.shape)
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            T0, T1 = 0.012, 0.0015
            steps, lr0, lr1 = 800, 0.02, 0.002

        # smooth optimize
        try:
            x_opt = adam_smooth(x_try, steps=steps, T0=T0, T1=T1, lr0=lr0, lr1=lr1)
        except Exception:
            continue
        v = max_cos(x_opt)
        # decide whether to LP polish: only if promising
        thresh = cur_best + (0.004 if u < 0.8 else 0.008)
        if v < thresh:
            # short LP, time-aware
            lp_time = 1.2 if v < cur_best + 0.0005 else 0.7
            # don't exceed remaining
            lp_time = min(lp_time, max(0.4, LIMIT - 22 - (time.time() - t0) - 1.0))
            if lp_time > 0.3:
                try:
                    x_opt2 = lp_polish(x_opt, lp_time)
                except Exception:
                    x_opt2 = x_opt
                v2 = max_cos(x_opt2)
            else:
                x_opt2, v2 = x_opt, v
            if v2 < cur_best:
                cur_best = float(v2)
                x_best = x_opt2.copy()
        # opportunistic: if time is running, reduce steps next rounds? handled by time check

    # final long polish of best
    rem = LIMIT - (time.time() - t0) - 1.0
    if rem > 2:
        try:
            x_best = lp_polish(x_best, min(rem, 25.0))
            cur_best = max_cos(x_best)
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
