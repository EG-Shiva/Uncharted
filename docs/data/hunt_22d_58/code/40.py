import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _quench(X, steps, b0, b1, lr0, lr1, mu=0.85, clip=0.05):
    n, d = X.shape
    V = np.zeros_like(X)
    lb0, lb1 = np.log(b0), np.log(b1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        f = s / max(1, steps - 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp(np.clip(beta * (G - m), -30, 0))
        np.fill_diagonal(W, 0.0)
        sw = float(W.sum())
        if not np.isfinite(sw) or sw <= 0:
            break
        W /= sw
        Gr = W @ X
        dot = (Gr * X).sum(axis=1, keepdims=True)
        Gt = Gr - dot * X
        V = mu * V + Gt
        step = lr * V
        sdot = (step * X).sum(axis=1, keepdims=True)
        step = step - sdot * X
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        step = np.where(sn > clip, step * (clip / sn), step)
        X = _norm(X - step)
    return X

def _refine_point(v, Y, steps=60, b0=200.0, b1=2000.0, lr0=0.05, lr1=0.005):
    # minimize max dot(v, Y) with Y fixed
    lb0, lb1 = np.log(b0), np.log(b1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        f = s / max(1, steps - 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        dots = Y @ v
        m = float(dots.max())
        w = np.exp(np.clip(beta * (dots - m), -30, 0))
        sw = float(w.sum())
        if not np.isfinite(sw) or sw <= 0:
            break
        w /= sw
        g = w @ Y
        gt = g - float(g @ v) * v
        v = v - lr * gt
        nv = float(np.linalg.norm(v))
        if nv == 0 or not np.isfinite(nv):
            break
        v = v / nv
    return v

def _lp_polish(X, deadline, max_it=6, cap=350):
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except Exception:
        return X, _max_cos(X)
    X = _norm(X)
    n, d = X.shape
    best = _max_cos(X)
    iu = np.triu_indices(n, 1)
    radius = 8e-4
    it = 0
    while time.time() < deadline and it < max_it and radius > 1e-13:
        it += 1
        if time.time() >= deadline:
            break
        Gfull = (X @ X.T)[iu]
        m = float(Gfull.max())
        window = max(20.0 * radius, 4e-5)
        order = np.argsort(-Gfull)
        sel = []
        for idx in order:
            if Gfull[idx] > m - window and len(sel) < cap:
                sel.append(int(idx))
            elif Gfull[idx] <= m - window:
                break
            if len(sel) >= cap:
                break
        if len(sel) == 0:
            break
        sel = np.array(sel, dtype=int)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(sel), n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-Gfull[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": 1.5})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = _norm(X + res.x[:-1].reshape(n, d))
        v = _max_cos(y)
        if v < best - 1e-12:
            X, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return X, best

def solve():
    T0 = time.time()
    DEAD = T0 + 150.0
    LNS_END = T0 + 130.0
    rng = np.random.default_rng(7)
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            c = np.load(fn).astype(float)
            if c.shape == (58, 22) and np.all(np.isfinite(c)):
                c = _norm(c)
                cands.append(c)
        except Exception:
            pass
    if not cands:
        cands.append(_norm(rng.standard_normal((58, 22))))
    vals = [_max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    incumbent = cands[bi].copy()
    best = incumbent.copy()
    best_val = float(vals[bi])
    n, d = best.shape

    ks = [3, 4, 5, 6, 8]
    trial = 0
    lp_calls = 0
    NCAND = 8192

    while time.time() < LNS_END and trial < 200:
        trial += 1
        k = ks[(trial - 1) % len(ks)]
        mode = (trial - 1) % 3  # 0 random, 1 hubs, 2 cluster
        # --- choose removal set ---
        try:
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            mcur = float(G.max())
        except Exception:
            mcur = best_val
            G = None
        if mode == 0:
            rem = rng.choice(n, size=k, replace=False)
        elif mode == 1:
            # highest-contact hubs (gap 0.004 for near contacts)
            try:
                deg = (G > mcur - 0.004).sum(axis=1)
                order = np.argsort(-deg, kind="stable")
                # add slight randomness: swap among top 2k
                pool = order[:max(2 * k, 10)].copy()
                rng.shuffle(pool)
                rem = np.sort(pool[:k])
            except Exception:
                rem = rng.choice(n, size=k, replace=False)
        else:
            try:
                seed = int(rng.integers(0, n))
                row = G[seed].copy()
                row[seed] = -2.0
                nn = np.argsort(-row, kind="stable")[:k - 1]
                rem = np.sort(np.concatenate([[seed], nn]))
            except Exception:
                rem = rng.choice(n, size=k, replace=False)
        rem = np.array(sorted(set(int(x) for x in rem)), dtype=int)
        if len(rem) != k:
            # fill up randomly if dedup issue
            extra = [i for i in rng.permutation(n) if i not in set(rem.tolist())]
            rem = np.sort(np.concatenate([rem, extra[:k - len(rem)]]))
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[rem] = False
        # sequential reinsertion
        X = best.copy()
        # temporarily store retained points
        order_ins = rem.copy()
        rng.shuffle(order_ins)
        for vidx in order_ins:
            if time.time() > LNS_END:
                break
            Y = X[keep_mask]
            # sample candidates
            C = rng.standard_normal((NCAND, d))
            C = _norm(C)
            dots = C @ Y.T
            worst = dots.max(axis=1)
            bi2 = int(np.argmin(worst))
            v = C[bi2]
            v = _refine_point(v, Y, steps=60)
            X[vidx] = v
            keep_mask[vidx] = True  # now retained for next insertion
        if time.time() > LNS_END:
            break
        # short all-point soft-max quench
        X = _quench(X, steps=180, b0=200.0, b1=1500.0, lr0=0.005, lr1=0.0012, mu=0.85, clip=0.05)
        vq = _max_cos(X)
        # decide LP
        need_lp = False
        if vq < best_val - 1e-12:
            need_lp = True
        elif vq < best_val + 0.003 and lp_calls < 12 and (trial % 2 == 0):
            need_lp = True
        if need_lp and time.time() < LNS_END - 2.0:
            lp_calls += 1
            Xp, vp = _lp_polish(X, deadline=min(time.time() + 2.0, LNS_END), max_it=5, cap=350)
            if vp < vq:
                X, vq = Xp, vp
        if vq < best_val - 1e-12:
            best, best_val = X, vq
            incumbent = best.copy()

    # reserve final 20s for LP polishing of best
    try:
        if time.time() < DEAD - 2.0:
            Xf, vf = _lp_polish(best, deadline=DEAD - 1.0, max_it=12, cap=400)
            if vf < best_val:
                best, best_val = Xf, vf
    except Exception:
        pass
    best = _norm(best)
    # guaranteed fallback: never worse than loaded best
    try:
        bl = np.load("best.npy").astype(float)
        if bl.shape == (58, 22):
            bl = _norm(np.where(np.isfinite(bl), bl, 0.0))
            if _max_cos(bl) < _max_cos(best):
                return bl
    except Exception:
        pass
    return best
