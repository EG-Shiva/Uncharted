import time
import numpy as np
from scipy.optimize import linprog, linear_sum_assignment
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _fast_polish(X, steps, lr0=0.06, T0=0.015, T1=0.0015):
    X = _norm(X.copy())
    lt0, lt1 = np.log(T0), np.log(T1)
    for s in range(steps):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        frac = s / max(1, steps - 1)
        T = float(np.exp(lt0 + (lt1 - lt0) * frac))
        W = np.exp((G - m) / T)
        np.fill_diagonal(W, 0.0)
        Grad = W @ X
        dot = np.einsum('ij,ij->i', Grad, X)[:, None]
        Gt = Grad - dot * X
        rs = W.sum(axis=1, keepdims=True) + 1.0
        lr = lr0 * (1.0 - 0.7 * frac)
        X = X - (lr / rs) * Gt
        X = _norm(X)
    return X

def _lp_polish(X, seconds, radius0=2e-3):
    X = _norm(X.copy())
    n, d = X.shape
    best = _max_cos(X)
    radius = radius0
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = X @ X.T
        c_all = G[iu0]
        m = float(c_all.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c_all > thresh
        if int(sel.sum()) > 3000:
            idx = np.where(sel)[0]
            part = np.argpartition(c_all[idx], len(idx) - 3000)[-3000:]
            mask = np.zeros_like(sel)
            mask[idx[part]] = True
            sel = mask
        I = iu0[0][sel]
        J = iu0[1][sel]
        c = c_all[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
        cols = np.concatenate([(I[:, None] * d + np.arange(d)),
                               (J[:, None] * d + np.arange(d)),
                               np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        rows = np.repeat(np.arange(p), 2 * d + 1)
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def _edges(X, slack=1e-7):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    iu = np.triu_indices(X.shape[0], 1)
    return (G[iu] >= m - slack), m

def _contact_counts(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    cnt = (G >= m - 1e-9).sum(axis=1)
    return m, cnt, G

def _recombine(Xa, Xb, rng, noise=0.01):
    # Procrustes align Xb to Xa
    M = Xa.T @ Xb
    try:
        U, _, Vt = np.linalg.svd(M)
        R = U @ Vt
        XbR = Xb @ R.T
    except Exception:
        XbR = Xb
    S = Xa @ XbR.T
    # maximize dot -> minimize -dot
    try:
        ri, ci = linear_sum_assignment(-S)
    except Exception:
        return None
    perm = np.empty_like(ri)
    perm[ri] = ci
    Yp = XbR[perm]
    # spatial coherent crossover
    u = rng.normal(size=(5,))
    u = u / (np.linalg.norm(u) + 1e-12)
    proj = Xa @ u
    med = np.median(proj)
    flip = rng.integers(2)
    child = np.empty_like(Xa)
    if flip == 0:
        mask = proj > med
    else:
        mask = proj <= med
    child[mask] = Xa[mask]
    child[~mask] = Yp[~mask]
    child = child + noise * rng.normal(size=child.shape)
    return _norm(child)

def solve():
    t0 = time.time()
    BUDGET = 172.0
    deadline = t0 + BUDGET
    explore_end = t0 + BUDGET - 28.0
    rng = np.random.default_rng(7)
    n, d = 173, 5
    PMAX = 14

    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (n, d):
                an = _norm(a)
                v = _max_cos(an)
                if v < best_val:
                    best_val = v
                    best = an.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.normal(size=(n, d)))
        best_val = _max_cos(best)

    # pool: list of [v, X, edges]
    pool = []
    e0, _ = _edges(best)
    pool.append([best_val, best.copy(), e0])

    # init diverse pool from record with strong perturbations + fast polish + short LP
    sigmas = [0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.03, 0.06, 0.10, 0.15, 0.22, 0.30, 0.04]
    for k in range(PMAX - 1):
        if time.time() > t0 + 30:
            break
        sig = sigmas[k % len(sigmas)]
        y = pool[0][1] + sig * rng.normal(size=(n, d))
        # occasional block reinit to break topology
        if k % 4 == 3:
            idx = rng.choice(n, 4, replace=False)
            y[idx] = rng.normal(size=(4, d))
        y = _norm(y)
        yf = _fast_polish(y, steps=300, lr0=0.06, T0=0.025, T1=0.0015)
        # short LP
        yl = _lp_polish(yf, seconds=0.7)
        vl = _max_cos(yl)
        el, _ = _edges(yl)
        pool.append([vl, yl, el])
        if vl < best_val:
            best_val = vl
            best = yl.copy()

    pool.sort(key=lambda z: z[0])
    # truncate
    pool = pool[:PMAX]
    # current walker starts from a middle member to avoid immediate greed
    cur_idx = min(2, len(pool) - 1)
    cur_X = pool[cur_idx][1].copy()
    cur_v = pool[cur_idx][0]

    logT0, logT1 = np.log(2e-4), np.log(1e-6)
    trial = 0
    iu_all = np.triu_indices(n, 1)

    while time.time() < explore_end:
        trial += 1
        now = time.time()
        prog = (now - t0) / max(1e-9, (explore_end - t0))
        prog = min(1.0, max(0.0, prog))
        T = float(np.exp(logT0 + (logT1 - logT0) * prog))

        # pick base: mostly cur, sometimes random pool member
        if rng.random() < 0.75:
            base = cur_X
            base_v = cur_v
        else:
            bi = int(rng.integers(len(pool)))
            base = pool[bi][1]
            base_v = pool[bi][0]

        r = rng.random()
        y = None
        steps = 350
        T0a = 0.015
        if r < 0.18:
            # recombination
            if len(pool) >= 2:
                i1, i2 = rng.choice(len(pool), 2, replace=False)
                y = _recombine(pool[i1][1], pool[i2][1], rng, noise=0.008 + 0.01 * rng.random())
                steps = 400
                T0a = 0.02
            if y is None:
                y = _norm(base + 0.03 * rng.normal(size=(n, d)))
        elif r < 0.32:
            # jammed hub kick
            _, cnt, _ = _contact_counts(base)
            order = np.argsort(-cnt)
            k = 5
            victims = order[:k]
            y = base.copy()
            y[victims] = y[victims] + 0.20 * rng.normal(size=(k, d))
            rr = rng.choice(n, 3, replace=False)
            y[rr] = y[rr] + 0.05 * rng.normal(size=(3, d))
            y = _norm(y)
            steps = 350
            T0a = 0.025
        elif r < 0.44:
            # rattler / worst reinsertion
            _, cnt, _ = _contact_counts(base)
            ratt = np.where(cnt <= 1)[0]
            if len(ratt) > 0 and rng.random() < 0.6:
                victim = int(ratt[rng.integers(len(ratt))])
            else:
                order = np.argsort(-cnt)
                victim = int(order[rng.integers(min(10, n))])
            yb = base.copy()
            others = np.delete(np.arange(n), victim)
            Xo = yb[others]
            Nc = 2000
            C = _norm(rng.normal(size=(Nc, d)))
            dots = C @ Xo.T
            md = dots.max(axis=1)
            bi2 = int(np.argmin(md))
            yb[victim] = C[bi2]
            yb[others] = yb[others] + 0.005 * rng.normal(size=(n - 1, d))
            y = _norm(yb)
            steps = 500
            T0a = 0.015
        elif r < 0.58:
            # subset shake
            y = base.copy()
            k = 20
            idx = rng.choice(n, k, replace=False)
            y[idx] = y[idx] + 0.12 * rng.normal(size=(k, d))
            y = _norm(y)
            steps = 350
            T0a = 0.02
        elif r < 0.72:
            # low-contact shake
            _, cnt, _ = _contact_counts(base)
            order = np.argsort(cnt)
            k = 8
            victims = order[:k]
            y = base.copy()
            y[victims] = y[victims] + 0.10 * rng.normal(size=(k, d))
            y = _norm(y)
            steps = 350
        elif r < 0.85:
            y = _norm(base + 0.030 * rng.normal(size=(n, d)))
            steps = 300
        else:
            # large sparse reinit
            y = base.copy()
            k = 3
            idx = rng.choice(n, k, replace=False)
            y[idx] = rng.normal(size=(k, d))
            y = y + 0.01 * rng.normal(size=(n, d))
            y = _norm(y)
            steps = 500
            T0a = 0.03

        yf = _fast_polish(y, steps=steps, lr0=0.06, T0=T0a, T1=0.0012)
        vf = _max_cos(yf)

        # decide LP: need pool worst
        pool.sort(key=lambda z: z[0])
        worst = pool[-1][0]
        do_lp = False
        if vf < worst + 0.002:
            do_lp = True
        elif rng.random() < 0.07:
            do_lp = True
        # always LP if better than cur by margin? vf already promising
        if vf < best_val:
            do_lp = True
        if not do_lp:
            continue
        # time check
        if time.time() > explore_end - 1.0:
            break
        yl = _lp_polish(yf, seconds=0.65)
        vl = _max_cos(yl)
        el, _ = _edges(yl)

        # pool insertion with diversity
        # hamming to members
        dmins = []
        for _, _, e in pool:
            dmins.append(int(np.sum(e ^ el)))
        dmin = min(dmins) if dmins else 1e9
        imin = int(np.argmin(dmins)) if dmins else 0
        if vl < best_val:
            best_val = vl
            best = yl.copy()
        inserted = False
        if len(pool) < PMAX:
            if dmin >= 20 or vl < pool[imin][0]:
                pool.append([vl, yl.copy(), el])
                inserted = True
        else:
            if dmin < 20:
                if vl < pool[imin][0]:
                    pool[imin] = [vl, yl.copy(), el]
                    inserted = True
            else:
                if vl < worst:
                    # replace worst
                    iw = int(np.argmax([z[0] for z in pool]))
                    pool[iw] = [vl, yl.copy(), el]
                    inserted = True
                elif vl < worst + 0.001 and len(pool) < 20:
                    # allow growth for diversity (cap 20)
                    pool.append([vl, yl.copy(), el])
                    inserted = True
        # Metropolis move of cur (use LP value)
        if vl < cur_v:
            cur_X, cur_v = yl.copy(), vl
        else:
            if rng.random() < np.exp(-(vl - cur_v) / max(T, 1e-12)):
                cur_X, cur_v = yl.copy(), vl
        # also occasionally reset cur to diverse pool member
        if trial % 40 == 0 and rng.random() < 0.3:
            bi3 = int(rng.integers(len(pool)))
            cur_X, cur_v = pool[bi3][1].copy(), pool[bi3][0]

    # final stage: LP polish pool best few
    pool.sort(key=lambda z: z[0])
    if pool[0][0] < best_val:
        best_val = pool[0][0]
        best = pool[0][1].copy()
    time_left = deadline - time.time()
    if time_left < 4:
        return _norm(best)
    # budgets for top-3
    K = min(3, len(pool))
    # allocate: best gets most
    if time_left > 20 and K >= 3:
        bds = [time_left * 0.18, time_left * 0.22, time_left - 5.0 - time_left * 0.40]
    elif time_left > 12 and K >= 2:
        bds = [time_left * 0.3, time_left - 5.0 - time_left * 0.3]
    else:
        bds = [time_left - 3.0]
        K = 1
    for k in range(min(K, len(bds))):
        bd = float(bds[k])
        if bd < 0.8:
            continue
        if time.time() + bd > deadline - 2.0:
            bd = (deadline - 2.0) - time.time()
            if bd < 0.8:
                continue
        cand = pool[k][1]
        pol = _lp_polish(cand, seconds=bd)
        v = _max_cos(pol)
        if v < best_val:
            best_val = v
            best = pol.copy()
    rem = deadline - time.time() - 1.5
    if rem > 2.0:
        best = _lp_polish(best, seconds=float(min(rem, 10.0)), radius0=1e-3)
    best = _norm(best)
    if not np.all(np.isfinite(best)):
        try:
            best = _norm(np.load("best.npy").astype(np.float64))
        except Exception:
            best = _norm(rng.normal(size=(n, d)))
    return best
