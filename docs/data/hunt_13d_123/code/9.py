import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _row_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1)

def _normalize(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _lp_polish(x, seconds, max_pairs=3500):
    x = _normalize(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        iu = iu0
        c = G[iu]
        m = float(c.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = c > thr
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > max_pairs:
            # keep only tightest max_pairs
            idx = np.argpartition(-c, max_pairs)[:max_pairs]
            mask = np.zeros_like(c, dtype=bool)
            mask[idx] = True
            sel = mask
            cnt = max_pairs
        I = iu[0][sel]; J = iu[1][sel]
        csel = c[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _normalize(y)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _anneal(v0, steps, beta0, beta1, lr0, lr1, mu=0.88):
    v = v0.copy()
    n, d = v.shape
    vel = np.zeros_like(v)
    for s in range(steps):
        f = s / max(steps - 1, 1)
        beta = beta0 * (beta1 / beta0) ** f
        lr = lr0 * (lr1 / lr0) ** f
        G = v @ v.T
        np.fill_diagonal(G, -10.0)
        gmax = float(G.max())
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        row = W.sum(axis=1, keepdims=True) + 1e-12
        grad = W @ v
        dot = np.sum(grad * v, axis=1, keepdims=True)
        gt = (grad - dot * v) / row
        # clip to avoid blowup
        gn = np.linalg.norm(gt, axis=1, keepdims=True) + 1e-12
        # normalize-ish: limit step
        gt = gt / np.maximum(gn, 1.0)
        vel = mu * vel + gt
        v = v - lr * vel
        v = _normalize(v)
    return v

def _void_reinsert(x, k, rng, n_samples=25000, sep=0.85):
    n, d = x.shape
    rm = _row_max(x)
    worst = np.argsort(-rm)[:k]
    keep = np.ones(n, dtype=bool); keep[worst] = False
    Y = x[keep]
    S = _normalize(rng.randn(n_samples, d))
    dots = S @ Y.T
    wd = dots.max(axis=1)
    order = np.argsort(wd)
    chosen = []
    chosen_vecs = []
    for idx in order:
        cand = S[idx]
        ok = True
        for cv in chosen_vecs:
            if float(cand @ cv) > sep:
                ok = False; break
        if ok:
            chosen.append(idx); chosen_vecs.append(cand)
            if len(chosen) >= k:
                break
    if len(chosen) < k:
        for idx in order:
            if idx not in chosen:
                chosen.append(idx)
                if len(chosen) >= k:
                    break
    xn = x.copy()
    for ii, si in zip(worst, chosen):
        xn[ii] = S[si] + 5e-4 * rng.randn(d)
    return _normalize(xn), worst

def _greedy(x, seconds, rng, sigma=0.0015):
    x = _normalize(x)
    n, d = x.shape
    best = _max_cos(x)
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    start = time.time()
    it = 0
    while time.time() - start < seconds:
        it += 1
        # focus on current worst points most of the time
        if it % 3 == 0:
            i = int(rng.randint(n))
        else:
            rm = G.max(axis=1)
            # pick among top-8 worst randomly
            w = np.argpartition(-rm, 8)[:8]
            i = int(w[rng.randint(len(w))])
        # tangent perturbation
        r = rng.randn(d)
        r = r - (r @ x[i]) * x[i]
        rn = np.linalg.norm(r)
        if rn == 0:
            continue
        r /= rn
        step = sigma * (0.5 + rng.rand())
        cand = x[i] + step * r
        cand /= np.linalg.norm(cand)
        dots = x @ cand
        dots[i] = -2.0
        md = float(dots.max())
        if md < best:
            x[i] = cand
            # update Gram row/col
            gcol = x @ cand
            G[i, :] = gcol; G[:, i] = gcol
            G[i, i] = -2.0
            best = float(G.max())
            # slowly shrink sigma on success
            sigma *= 1.001
        else:
            # occasional decay
            if it % 500 == 0:
                sigma *= 0.999
                if sigma < 2e-4:
                    sigma = 2e-4
    return x

def solve():
    t0 = time.time()
    rng = np.random.RandomState(1)
    n, d = 123, 13
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.randn(n, d)
    xb = np.array(xb, dtype=float)
    if xb.shape != (n, d):
        tmp = rng.randn(n, d)
        m = min(xb.size, n * d)
        tmp.ravel()[:m] = xb.ravel()[:m]
        xb = tmp
    xb = _normalize(xb)
    best_val = _max_cos(xb)
    x_best = xb.copy()
    trial = 0
    lp_calls = 0
    # explore, reserve time for final polish
    while time.time() - t0 < 128 and trial < 300:
        trial += 1
        mode = trial % 7
        if mode == 0:
            xt, _ = _void_reinsert(x_best, 1, rng, n_samples=25000, sep=0.85)
            xt = _normalize(xt + 0.003 * rng.randn(n, d))
            steps, b0, b1, l0, l1 = 1100, 40.0, 550.0, 0.008, 0.0007
        elif mode == 1:
            xt, _ = _void_reinsert(x_best, 2, rng, n_samples=25000, sep=0.80)
            xt = _normalize(xt + 0.003 * rng.randn(n, d))
            steps, b0, b1, l0, l1 = 1300, 35.0, 600.0, 0.009, 0.0007
        elif mode == 2:
            xt, _ = _void_reinsert(x_best, 3, rng, n_samples=20000, sep=0.75)
            xt = _normalize(xt + 0.004 * rng.randn(n, d))
            steps, b0, b1, l0, l1 = 1500, 30.0, 500.0, 0.010, 0.0008
        elif mode == 3:
            xt = _normalize(x_best + 0.008 * rng.randn(n, d))
            steps, b0, b1, l0, l1 = 900, 50.0, 550.0, 0.006, 0.0006
        elif mode == 4:
            xt = _normalize(x_best + 0.025 * rng.randn(n, d))
            steps, b0, b1, l0, l1 = 1300, 28.0, 480.0, 0.010, 0.0008
        elif mode == 5:
            xt = _normalize(x_best + 0.005 * rng.randn(n, d))
            rm = _row_max(x_best)
            widx = np.argsort(-rm)[:10]
            xt[widx] += 0.05 * rng.randn(len(widx), d)
            xt = _normalize(xt)
            steps, b0, b1, l0, l1 = 1200, 32.0, 550.0, 0.009, 0.0007
        else:
            # orthogonal-ish shake: rotate random subset
            xt = x_best.copy()
            k = 6
            idx = rng.choice(n, k, replace=False)
            xt[idx] += 0.03 * rng.randn(k, d)
            xt = _normalize(xt + 0.003 * rng.randn(n, d))
            steps, b0, b1, l0, l1 = 1200, 35.0, 600.0, 0.009, 0.0007
        if time.time() - t0 > 124:
            break
        v = _anneal(xt, steps, b0, b1, l0, l1, mu=0.88)
        sv = _max_cos(v)
        if sv < best_val:
            best_val = sv
            x_best = v.copy()
        need_lp = False
        lp_sec = 0.0
        if sv < best_val - 1e-9:
            need_lp = True; lp_sec = 5.0
        elif sv < best_val + 0.0025 and lp_calls < 20:
            if (trial % 2 == 0) or (sv < best_val + 0.0004):
                need_lp = True; lp_sec = 2.5
        if need_lp:
            remain = 131 - (time.time() - t0)
            if remain < 3:
                break
            lp_sec = min(lp_sec, remain)
            vp = _lp_polish(v, lp_sec)
            lp_calls += 1
            pv = _max_cos(vp)
            if pv < best_val:
                best_val = pv
                x_best = vp.copy()
    # final polish: alternate LP + greedy
    remain = 172 - (time.time() - t0)
    if remain > 6:
        x_best = _lp_polish(x_best, min(remain * 0.55, 22.0))
        remain2 = 174 - (time.time() - t0)
        if remain2 > 4:
            x_best = _greedy(x_best, min(remain2 * 0.4, 10.0), rng, sigma=0.0012)
            remain3 = 176 - (time.time() - t0)
            if remain3 > 4:
                x_best = _lp_polish(x_best, min(remain3 - 1.0, 10.0))
    x_best = _normalize(x_best)
    return x_best
