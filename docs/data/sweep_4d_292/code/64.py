import time
import numpy as np

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _normr(A):
    n = np.linalg.norm(A, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return A / n

def _anneal(x, steps, beta0, beta1, lr0, mu, rng=None):
    n = x.shape[0]
    v = np.zeros_like(x)
    lb = np.log(beta1 / beta0) if beta1 > beta0 else 0.0
    for t in range(steps):
        frac = t / max(1, steps - 1)
        beta = beta0 * np.exp(lb * frac)
        lr = lr0 * (1.0 - 0.5 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        S = W.sum(axis=1, keepdims=True) + 1e-300
        A = (W @ x) / S
        dot = (A * x).sum(axis=1, keepdims=True)
        Gt = A - dot * x
        v = mu * v + Gt
        vd = (v * x).sum(axis=1, keepdims=True)
        v = v - vd * x
        x = x - lr * v
        x = _normr(x)
    return x

def _lp_polish_deadline(x, deadline):
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except Exception:
        return x
    x = _normr(x)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        try:
            c = (x @ x.T)[iu]
        except Exception:
            break
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 4000:
            # keep tightest only to bound LP size
            idx = np.argpartition(c[sel], -4000)[-4000:]
            I, J = I[idx], J[idx]
            p = len(I)
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel][-p:] if p != int(sel.sum()) else -c[sel], A_eq=A_eq, b_eq=np.zeros(n),
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
        y = _normr(y)
        try:
            vv = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _largest_holes(X, rng, n_sample=1500, n_refine=40):
    # find void centres: directions minimizing max dot with X
    D = rng.standard_normal((n_sample, 4))
    D = _normr(D)
    try:
        S = D @ X.T
        smax = S.max(axis=1)
    except Exception:
        return D[:4]
    # take few best (smallest smax)
    k = min(6, n_sample)
    idx = np.argpartition(smax, k - 1)[:k]
    holes = D[idx].copy()
    # refine each by gradient descent on soft-max
    beta = 60.0
    for hi in range(holes.shape[0]):
        h = holes[hi]
        for _ in range(n_refine):
            g = X @ h
            m = float(g.max())
            w = np.exp(beta * (g - m))
            w = w / (w.sum() + 1e-300)
            a = w @ X
            # tangential descent (move away from weighted mean)
            grad = a - float(a @ h) * h
            h = h - 0.15 * grad
            nh = float(np.linalg.norm(h))
            if nh < 1e-12:
                break
            h = h / nh
        holes[hi] = h
    # sort by quality (smaller max dot first)
    try:
        q = (holes @ X.T).max(axis=1)
        holes = holes[np.argsort(q)]
    except Exception:
        pass
    return holes

def solve():
    t0 = time.time()
    TLIM = 170.0
    rng = np.random.default_rng(292146)
    n, d = 292, 4
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                nr = np.linalg.norm(a, axis=1, keepdims=True)
                if float(nr.min()) > 1e-9:
                    cands.append(a / nr)
        except Exception:
            pass
    if not cands:
        Xb = _normr(rng.standard_normal((n, d)))
    else:
        vals = [_max_cos(c) for c in cands]
        Xb = cands[int(np.argmin(vals))].copy()
    best_val = _max_cos(Xb)
    X_best = Xb.copy()
    X_cur = Xb.copy()
    cur_val = best_val

    # quick initial polish of loaded best if time allows (small)
    # skip long polish now; do it at end. Just ensure best is LP vertex.
    explore_end = t0 + TLIM - 20.0
    it = 0
    T0 = 4e-04
    T1 = 2e-05
    while time.time() < explore_end:
        it += 1
        rem = explore_end - time.time()
        if rem < 1.0:
            break
        prog = 1.0 - rem / max(1.0, (explore_end - t0))
        Temp = T0 * (1.0 - prog) + T1 * prog
        # choose base: mostly walker, sometimes best
        if rng.random() < 0.85:
            base = X_cur
            base_val = cur_val
        else:
            base = X_best
            base_val = best_val
        X = base.copy()
        op = rng.random()
        try:
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            rowmax = G.max(axis=1)
            mcur = float(rowmax.max())
            # degrees with tight threshold for donor ranking
            deg = (G > mcur - 1e-7).sum(axis=1)
        except Exception:
            continue
        if op < 0.65:
            # jammed-to-void transplant: 1-3 donors
            try:
                holes = _largest_holes(X, rng, n_sample=1200, n_refine=30)
            except Exception:
                continue
            # donors: among top jammed
            try:
                order = np.argsort(-rowmax)
                nj = 1
                r = rng.random()
                if r < 0.15:
                    nj = 3
                elif r < 0.45:
                    nj = 2
                # pick from top 12 randomly
                pool = order[:12]
                donors = rng.choice(pool, size=nj, replace=False)
            except Exception:
                continue
            for di, donor in enumerate(donors):
                h = holes[di % len(holes)]
                jit = 0.015 * rng.standard_normal(4)
                jit = jit - float(jit @ h) * h
                cand = h + jit
                cand = cand / max(float(np.linalg.norm(cand)), 1e-12)
                # avoid accidental clash: if too close, add bigger jitter
                try:
                    sc = float((X @ cand).max())
                    if sc > 0.95:
                        cand = _normr((h + 0.08 * rng.standard_normal(4)).reshape(1, 4))[0]
                except Exception:
                    pass
                X[int(donor)] = cand
            steps = 260 if rem > 15 else 160
            try:
                X = _anneal(X, steps, 30.0, 500.0, 0.07, 0.86)
                v = _max_cos(X)
            except Exception:
                continue
        elif op < 0.85:
            # subset jitter on random block + low-degree boost
            try:
                k = int(rng.integers(6, 22))
                idx = rng.choice(n, size=k, replace=False)
                # include rattlers/low-degree always
                low = np.where(deg <= 2)[0]
                if len(low) > 0:
                    extra = rng.choice(low, size=min(len(low), 3), replace=False)
                    idx = np.unique(np.concatenate([idx, extra]))
                sigma = float(rng.choice([0.02, 0.05, 0.09, 0.15]))
                X[idx] = _normr(X[idx] + sigma * rng.standard_normal((len(idx), 4)))
                steps = 240 if rem > 15 else 150
                X = _anneal(X, steps, 25.0, 420.0, 0.06, 0.85)
                v = _max_cos(X)
            except Exception:
                continue
        else:
            # global small shake + anneal (basin hop)
            try:
                sigma = float(rng.choice([0.008, 0.02, 0.04]))
                X = _normr(X + sigma * rng.standard_normal(X.shape))
                steps = 300 if rem > 15 else 180
                X = _anneal(X, steps, 20.0, 380.0, 0.06, 0.85)
                v = _max_cos(X)
            except Exception:
                continue
        # walker acceptance (Metropolis, allow worsening to change topology)
        try:
            if (v < cur_val) or (rng.random() < np.exp((cur_val - v) / max(Temp, 1e-12))):
                X_cur = X.copy()
                cur_val = v
        except Exception:
            pass
        # elitist update with LP polish if promising
        if v < best_val + 3e-04:
            # only polish if close to best or better, to save time
            do_polish = (v < best_val) or (rng.random() < 0.08 and rem > 30)
            if v < best_val:
                do_polish = True
            if do_polish:
                dl = min(time.time() + 2.5, explore_end)
                if dl - time.time() > 0.5:
                    try:
                        Y = _lp_polish_deadline(X.copy(), dl)
                        vy = _max_cos(Y)
                    except Exception:
                        Y, vy = X, v
                    # update walker with polished version too
                    if vy < cur_val:
                        X_cur, cur_val = Y.copy(), vy
                    if vy < best_val - 1e-12:
                        X_best, best_val = Y.copy(), vy
                    elif v < best_val - 1e-12:
                        X_best, best_val = X.copy(), v
                else:
                    if v < best_val - 1e-12:
                        X_best, best_val = X.copy(), v
                        X_cur, cur_val = X.copy(), v
            else:
                if v < best_val - 1e-12:
                    X_best, best_val = X.copy(), v
        # periodic reset of walker to best if diverged
        if it % 25 == 0:
            if cur_val > best_val + 0.0015:
                X_cur = X_best.copy()
                cur_val = best_val
            elif rng.random() < 0.25:
                # random restart from best with medium kick to diversify
                try:
                    X_cur = _normr(X_best + 0.03 * rng.standard_normal(X_best.shape))
                    X_cur = _anneal(X_cur, 120, 25.0, 300.0, 0.06, 0.85)
                    cur_val = _max_cos(X_cur)
                except Exception:
                    pass
    # final intensive polish of champion with remaining time
    try:
        deadline = t0 + TLIM - 1.5
        if time.time() < deadline - 0.5:
            X_best = _lp_polish_deadline(X_best, deadline)
            best_val = _max_cos(X_best)
    except Exception:
        pass
    X_best = _normr(X_best)
    return X_best
