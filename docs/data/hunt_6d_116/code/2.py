import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def _lp_polish(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 3e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        cmat = x @ x.T
        c = cmat[iu]
        m = float(np.max(c))
        thr = m - max(20.0 * radius, 1e-9)
        sel = c > thr
        if int(np.sum(sel)) == 0:
            break
        I = iu[0][sel]
        J = iu[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        # columns: I-block, J-block, t
        base_I = (I[:, None] * d + np.arange(d)).ravel()
        base_J = (J[:, None] * d + np.arange(d)).ravel()
        cols_IJ = np.concatenate([base_I, base_J])
        # build vals/cols in order per row: xJ, xI, -1
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        vals = np.empty(p * (2 * d + 1), dtype=np.float64)
        # fill row by row with vectorization
        xJ = x[J]
        xI = x[I]
        # interleave
        for k in range(p):
            s = k * (2 * d + 1)
            cols[s:s + d] = I[k] * d + np.arange(d)
            vals[s:s + d] = xJ[k]
            cols[s + d:s + 2 * d] = J[k] * d + np.arange(d)
            vals[s + d:s + 2 * d] = xI[k]
            cols[s + 2 * d] = n * d
            vals[s + 2 * d] = -1.0
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
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best


def _smooth_descent(x, steps=280, T0=0.02, T1=0.004, lr0=2.0, lr1=0.2):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        T = T0 * (T1 / T0) ** frac
        lr = lr0 * (1 - frac) + lr1 * frac
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        M = np.max(G)
        E = np.exp((G - M) / T)
        # diag is 0 already (exp(-inf)=0)
        S = np.sum(E)
        if not np.isfinite(S) or S <= 0:
            break
        Grad = (E @ x) / S
        dot = np.sum(Grad * x, axis=1, keepdims=True)
        Gt = Grad - dot * x
        # clip extreme step
        gn = np.linalg.norm(Gt, axis=1).max()
        if gn > 0:
            # limit max per-point move to 0.05
            maxmove = 0.05
            if gn * lr > maxmove:
                Gt = Gt * (maxmove / (gn * lr))
        x = x - lr * Gt
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    EXPL_END = t0 + 140.0
    rng = np.random.default_rng(12345)

    # load starts
    try:
        xb = np.load("best.npy").astype(np.float64)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(np.float64)
    except Exception:
        xr = None
    cands = []
    if xb is not None:
        xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
        cands.append(xb)
    if xr is not None:
        xr = xr / np.linalg.norm(xr, axis=1, keepdims=True)
        cands.append(xr)
    if len(cands) == 0:
        cands.append(rng.normal(size=(116, 6)))
    best_x = None
    best_v = 1e9
    for c in cands:
        c = c / np.linalg.norm(c, axis=1, keepdims=True)
        v = _max_cos(c)
        if v < best_v:
            best_v = v
            best_x = c.copy()
    n, d = best_x.shape  # expect 116,6
    iu0 = np.triu_indices(n, 1)

    def contacts_info(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        # contacts within 1e-7 of max (robust) for counting
        C = G > (m - 1e-7)
        per = np.sum(C, axis=1)
        return G, m, per

    trial = 0
    sigmas = [0.015, 0.04, 0.09, 0.18, 0.05, 0.12]
    while time.time() < EXPL_END:
        trial += 1
        mode = trial % 7
        X0 = best_x
        G0, m0, per0 = contacts_info(X0)
        order = np.argsort(per0)  # asc: rattlers first
        Xtry = X0.copy()
        if mode in (0, 1, 2, 3):
            sigma = sigmas[trial % len(sigmas)]
            if mode == 3:
                sigma = 0.03 + 0.02 * (trial % 3)
            noise = rng.normal(scale=sigma, size=(n, d))
            # for mode 2 make a subset kick harder
            if mode == 2:
                k = int(rng.integers(3, 9))
                idx = rng.choice(n, size=k, replace=False)
                noise[idx] *= 3.0
            Xtry = Xtry + noise
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
        elif mode == 4:
            # void-filling reinsertion: pick worst or rattler
            if trial % 2 == 0:
                idx = int(np.argmax(per0))
            else:
                ratt = np.where(per0 <= 0.5)[0]
                if len(ratt) > 0:
                    idx = int(rng.choice(ratt))
                else:
                    idx = int(order[0])
            # find best void among random candidates
            Nc = 2500
            Rc = rng.normal(size=(Nc, d))
            Rc /= np.linalg.norm(Rc, axis=1, keepdims=True)
            others = np.delete(X0, idx, axis=0)  # (n-1,d)
            dots = Rc @ others.T  # (Nc,n-1)
            worst = np.max(dots, axis=1)
            bi = int(np.argmin(worst))
            Xtry[idx] = Rc[bi]
            # small shake to others to allow relaxation
            Xtry = Xtry + rng.normal(scale=0.015, size=(n, d))
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
        elif mode == 5:
            # kick overloaded hubs harder, rattlers medium
            noise = rng.normal(scale=0.01, size=(n, d))
            # top 8 overloaded
            top = np.argsort(-per0)[:8]
            noise[top] += rng.normal(scale=0.20, size=(8, d))
            ratt = np.where(per0 <= 0.5)[0]
            if len(ratt) > 0:
                noise[ratt] += rng.normal(scale=0.10, size=(len(ratt), d))
            Xtry = Xtry + noise
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
        else:  # mode 6: randomize a few points fully
            k = int(rng.integers(2, 6))
            idx = rng.choice(n, size=k, replace=False)
            Rr = rng.normal(size=(k, d))
            Rr /= np.linalg.norm(Rr, axis=1, keepdims=True)
            Xtry[idx] = Rr
            Xtry = Xtry + rng.normal(scale=0.01, size=(n, d))
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)

        # smooth annealed descent (fast topology adjustment)
        # fewer steps for small kicks, more for large
        if mode in (0, 3):
            steps = 180
        else:
            steps = 300
        Xtry = _smooth_descent(Xtry, steps=steps, T0=0.025, T1=0.004, lr0=2.0, lr1=0.25)

        # quick check: if smooth already very bad, skip LP to save time
        v_smooth = _max_cos(Xtry)
        # allow LP even if somewhat worse (up to +0.02) since LP can fix
        if v_smooth > best_v + 0.02:
            continue
        budget = 1.6 if mode in (0, 3) else 2.2
        deadline = min(time.time() + budget, EXPL_END + 2.0, HARD_END - 12.0)
        if deadline <= time.time() + 0.1:
            break
        Xpol, vpol = _lp_polish(Xtry, deadline)
        if vpol < best_v - 5e-13:
            best_v = vpol
            best_x = Xpol.copy()
            # early exit if big win? continue to push further
        # opportunistic: if very close (within 2e-6) keep exploring from best anyway (greedy)

    # final long polish of best
    remain = HARD_END - time.time()
    if remain > 3.0:
        # use up to remain-2 seconds but at most 30
        fp_budget = min(remain - 2.0, 28.0)
        Xf, vf = _lp_polish(best_x, time.time() + fp_budget)
        if vf <= best_v:
            best_x, best_v = Xf, vf

    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
