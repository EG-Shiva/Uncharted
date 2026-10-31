import time
import numpy as np

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _contacts(x, thr=1e-7):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = G.max()
    C = G > m - thr
    cnt = C.sum(axis=1)
    return m, cnt, G

def _anneal(x, steps, beta0, beta1, lr0, mu, rng=None):
    n, d = x.shape
    v = np.zeros_like(x)
    lb = np.log(beta1 / beta0) if beta1 > beta0 else 0.0
    for t in range(steps):
        frac = t / max(1, steps - 1)
        beta = beta0 * np.exp(lb * frac)
        lr = lr0 * (1.0 - 0.5 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = G.max()
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        S = W.sum(axis=1, keepdims=True) + 1e-300
        A = (W @ x) / S
        dot = (A * x).sum(axis=1, keepdims=True)
        Gt = A - dot * x
        v = mu * v + Gt
        vd = (v * x).sum(axis=1, keepdims=True)
        v -= vd * x
        x = x - lr * v
        nrm = np.linalg.norm(x, axis=1, keepdims=True)
        np.maximum(nrm, 1e-12, out=nrm)
        x /= nrm
    return x

def _squeeze(x, steps=250, alpha=2000.0, lr=0.02):
    n, d = x.shape
    for t in range(steps):
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        dm = G.max()
        W = np.exp(alpha * (G - dm))
        np.fill_diagonal(W, 0.0)
        S = W.sum(axis=1, keepdims=True) + 1e-300
        A = (W @ x) / S
        dot = (A * x).sum(axis=1, keepdims=True)
        Gt = A - dot * x
        x = x - lr * Gt
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    return x

def _greedy_holes(X_keep, k, rng, n_cand=15000):
    d = X_keep.shape[1]
    cand = rng.standard_normal((n_cand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    cur = X_keep
    outs = []
    # process in chunks to save memory
    for _ in range(k):
        best_s = 2.0
        best_p = None
        for a in range(0, n_cand, 3000):
            cc = cand[a:a+3000]
            dd = cc @ cur.T
            mm = dd.max(axis=1)
            j = int(np.argmin(mm))
            if float(mm[j]) < best_s:
                best_s = float(mm[j])
                best_p = cc[j].copy()
        # jitter slightly to break symmetry
        jp = best_p + 0.005 * rng.standard_normal(d)
        jp /= max(np.linalg.norm(jp), 1e-12)
        outs.append(jp)
        cur = np.vstack([cur, jp.reshape(1, d)])
        # refresh candidates a bit to avoid reuse
        if rng.random() < 0.3:
            cand = rng.standard_normal((n_cand, d))
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    return np.vstack([X_keep] + [o.reshape(1, d) for o in outs])

def _lp_polish_deadline(x, deadline):
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except Exception:
        return x
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            break
        if getattr(res, "status", 1) != 0:
            radius *= 0.5
            continue
        try:
            y = x + res.x[:-1].reshape(n, d)
        except Exception:
            radius *= 0.5
            continue
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    TLIM = 172.0
    final_reserve = 30.0
    explore_end = t0 + TLIM - final_reserve
    rng = np.random.default_rng(20261031)
    n, d = 292, 4
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                nr = np.linalg.norm(a, axis=1, keepdims=True)
                if np.min(nr) > 1e-9:
                    cands.append(a / nr)
        except Exception:
            pass
    if not cands:
        X_best = rng.standard_normal((n, d))
        X_best /= np.linalg.norm(X_best, axis=1, keepdims=True)
    else:
        vals = [_max_cos(c) for c in cands]
        X_best = cands[int(np.argmin(vals))].copy()
    best_val = _max_cos(X_best)
    # contact info for targeted kills
    try:
        _, cnt_best, _ = _contacts(X_best, thr=1e-7)
    except Exception:
        cnt_best = np.zeros(n)

    it = 0
    sigmas = [0.005, 0.012, 0.03, 0.07, 0.15, 0.35]
    while time.time() < explore_end:
        it += 1
        if explore_end - time.time() < 1.2 and it > 5:
            break
        X = X_best.copy()
        rmode = rng.random()
        # decide perturbation
        if rmode < 0.30:
            # gaussian shake + kill random -> greedy holes (topology change)
            sigma = sigmas[int(rng.integers(len(sigmas)))]
            X = X + sigma * rng.standard_normal((n, d))
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            if rng.random() < 0.85:
                k = int(rng.integers(1, 9))
                # 50% target overloaded, 50% random
                if rng.random() < 0.5:
                    order = np.argsort(-cnt_best)
                    # pick among top 40 with some randomness
                    pool = order[:40]
                    idx = rng.choice(pool, size=min(k, len(pool)), replace=False)
                    # if k larger, add random
                    if k > len(idx):
                        extra = rng.choice(n, size=k-len(idx), replace=False)
                        idx = np.concatenate([idx, extra])
                else:
                    idx = rng.choice(n, size=k, replace=False)
                keep_mask = np.ones(n, dtype=bool)
                keep_mask[idx] = False
                X_keep = X[keep_mask]
                X = _greedy_holes(X_keep, k, rng, n_cand=12000)
                X = X / np.linalg.norm(X, axis=1, keepdims=True)
        elif rmode < 0.50:
            # cluster excision: pick seed, remove neighbours
            sigma = float(rng.choice([0.01, 0.03, 0.08]))
            X = X + sigma * rng.standard_normal((n, d))
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            seed = int(rng.integers(n))
            dots = X @ X[seed]
            dots[seed] = -2
            k = int(rng.integers(3, 10))
            idx = np.argsort(-dots)[:k]
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[idx] = False
            X = _greedy_holes(X[keep_mask], k, rng, n_cand=12000)
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        elif rmode < 0.65:
            # rattler-focused: move low-contact points far
            # recompute contacts cheaply for X_best (reuse cnt_best)
            low = np.where(cnt_best <= 1)[0]
            if len(low) == 0:
                low = np.argsort(cnt_best)[:5]
            sigma = float(rng.choice([0.02, 0.08, 0.25]))
            X = X + 0.01 * rng.standard_normal((n, d))
            R = rng.standard_normal((len(low), d))
            R /= np.linalg.norm(R, axis=1, keepdims=True)
            # replace low points with hole positions
            keep_mask = np.ones(n, dtype=bool)
            # only move subset
            mv = rng.choice(low, size=min(len(low), int(rng.integers(1, 5))), replace=False)
            keep_mask[mv] = False
            X = _greedy_holes((X[keep_mask] / np.linalg.norm(X[keep_mask], axis=1, keepdims=True)), len(mv), rng, n_cand=12000)
            # add remaining shake
            X = X + sigma * 0.2 * rng.standard_normal((n, d))
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        elif rmode < 0.80:
            # pure shake, varying scale
            sigma = sigmas[int(rng.integers(len(sigmas)))]
            # occasionally very heavy shake of subset
            if rng.random() < 0.3:
                idx = rng.choice(n, size=int(rng.integers(10, 60)), replace=False)
                X[idx] = X[idx] + 0.5 * rng.standard_normal((len(idx), d))
            X = X + sigma * rng.standard_normal((n, d))
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        else:
            # fresh random restart mixed with best (different basin)
            R = rng.standard_normal((n, d))
            R /= np.linalg.norm(R, axis=1, keepdims=True)
            if rng.random() < 0.4:
                X = R
            else:
                alpha_mix = float(rng.random() * 0.7 + 0.2)
                X = (1 - alpha_mix) * X + alpha_mix * R
                # also random rotation to diversify
                Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
                if rng.random() < 0.5:
                    X = X @ Q
                X /= np.linalg.norm(X, axis=1, keepdims=True)

        # random anneal hyperparams
        rr = rng.random()
        if rr < 0.35:
            steps, b0, b1, lr0, mu = 900, 25.0, 400.0, 0.06, 0.85
        elif rr < 0.65:
            steps, b0, b1, lr0, mu = 500, 15.0, 250.0, 0.08, 0.80
        elif rr < 0.85:
            steps, b0, b1, lr0, mu = 1400, 30.0, 700.0, 0.05, 0.88
        else:
            steps, b0, b1, lr0, mu = 300, 10.0, 150.0, 0.10, 0.75
        # adapt steps to remaining time
        rem = explore_end - time.time()
        if rem < 8:
            steps = min(steps, 300)

        try:
            X = _anneal(X, steps, b0, b1, lr0, mu)
            X = _squeeze(X, steps=150, alpha=1500.0, lr=0.02)
        except Exception:
            continue
        try:
            v = _max_cos(X)
        except Exception:
            continue
        # filter: only LP-polish if promising (basin could polish below best)
        # annealed value is close to polished value; allow small slack
        thresh = best_val + (5e-05 if it > 10 else 2e-04)
        if v < thresh:
            # short LP confirm (2-3s)
            dl = min(time.time() + 2.5, explore_end)
            if dl - time.time() < 0.6:
                # no time for LP, just check raw
                if v < best_val:
                    X_best, best_val = X.copy(), v
                    try:
                        _, cnt_best, _ = _contacts(X_best, thr=1e-7)
                    except Exception:
                        pass
                continue
            try:
                Y = _lp_polish_deadline(X, dl)
                vy = _max_cos(Y)
            except Exception:
                continue
            if vy < best_val:
                X_best, best_val = Y.copy(), vy
                try:
                    _, cnt_best, _ = _contacts(X_best, thr=1e-7)
                except Exception:
                    pass
                # early exit if big win? keep going to squeeze more
                if best_val < 0.9025:
                    break
        # opportunistic raw update without LP if clearly better
        elif v < best_val:
            X_best, best_val = X.copy(), v
            try:
                _, cnt_best, _ = _contacts(X_best, thr=1e-7)
            except Exception:
                pass

    # final long polish of best
    try:
        X_best = _squeeze(X_best, steps=300, alpha=2500.0, lr=0.015)
        X_best = _lp_polish_deadline(X_best, t0 + TLIM - 1.0)
    except Exception:
        pass
    X_best = X_best / np.linalg.norm(X_best, axis=1, keepdims=True)
    # strict feasibility: replace any non-finite
    if not np.all(np.isfinite(X_best)):
        X_best = np.load("best.npy").astype(np.float64)
        X_best /= np.linalg.norm(X_best, axis=1, keepdims=True)
    return X_best
