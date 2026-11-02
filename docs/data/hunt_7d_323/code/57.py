import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _contacts(X, tol=1e-9):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    C = (G > m - tol)
    cnt = np.sum(C, axis=1)
    return m, cnt, G

def _tangential_anneal(X, steps, tau0, tau1, lr0, lr1, rng, t_start=None, HARD=None, check_every=200):
    n, d = X.shape
    X = X.copy()
    # geometric tau, linear-ish lr
    ltau0 = np.log(tau0); ltau1 = np.log(tau1)
    for s in range(steps):
        if t_start is not None and (s % check_every == 0):
            if time.time() - t_start > HARD - 30.0:
                break
        frac = s / max(1, steps - 1)
        tau = float(np.exp(ltau0 + (ltau1 - ltau0) * frac))
        lr = float(lr0 + (lr1 - lr0) * frac)
        G = X @ X.T
        # for stability subtract row max? use global max
        m = float(np.max(G - np.eye(n) * 10.0))
        # weights
        # G diag currently 1.0, kill it
        W = np.exp((G - m) / tau)
        np.fill_diagonal(W, 0.0)
        Grad = (W @ X) / tau
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        nrm = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-15
        X = X - lr * (Gt / nrm)
        X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-30)
    return X

def _best_void(X, rng, ncand=12000, chunk=3000):
    n, d = X.shape
    best_c = None
    best_m = np.inf
    # chunked sampling
    for s in range(0, ncand, chunk):
        cs = rng.standard_normal(size=(min(chunk, ncand - s), d))
        cs /= np.linalg.norm(cs, axis=1, keepdims=True)
        # max dot to X
        Gc = cs @ X.T  # (c,n)
        mvec = np.max(Gc, axis=1)
        j = int(np.argmin(mvec))
        if float(mvec[j]) < best_m:
            best_m = float(mvec[j])
            best_c = cs[j].copy()
    c = best_c / np.linalg.norm(best_c)
    # local max-min refinement: minimize max_j <c,Xj>
    lr0, lr1 = 0.02, 0.0005
    iters = 400
    for it in range(iters):
        frac = it / max(1, iters - 1)
        lr = lr0 + (lr1 - lr0) * frac
        dots = X @ c
        j = int(np.argmax(dots))
        g = X[j] - float(X[j] @ c) * c
        gn = float(np.linalg.norm(g))
        if gn < 1e-14:
            # random tangent kick
            r = rng.standard_normal(size=d)
            r = r - float(r @ c) * c
            rn = float(np.linalg.norm(r))
            if rn < 1e-14:
                break
            g = r / rn
            gn = 1.0
        else:
            g = g / gn
        c = c - lr * g
        c /= np.linalg.norm(c)
    # final score
    mfinal = float(np.max(X @ c))
    return c, min(best_m, mfinal)

def _relax(X, steps, tau0, tau1, lr0, lr1, rng, t_start, HARD):
    return _tangential_anneal(X, steps, tau0, tau1, lr0, lr1, rng, t_start, HARD)

def _lp_polish(x, seconds, t_start, HARD):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    best_x = x.copy()
    radius = 1e-3
    t0 = time.time()
    iuI_all, iuJ_all = np.triu_indices(n, 1)
    it = 0
    while (time.time() - t0 < seconds) and (time.time() - t_start < HARD - 4.0) and radius > 1e-14:
        it += 1
        G = x @ x.T
        c = G[iuI_all, iuJ_all]
        m = float(np.max(c))
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        cnt = int(np.sum(sel))
        if cnt == 0:
            break
        if cnt > 2500:
            idx = np.where(sel)[0]
            cv = c[idx]
            part = np.argpartition(cv, -2500)[-2500:]
            sel2 = np.zeros_like(sel, dtype=bool)
            sel2[idx[part]] = True
            sel = sel2
            cnt = 2500
        I = iuI_all[sel]
        J = iuJ_all[sel]
        csel = c[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        vals = np.empty(p * (2 * d + 1), dtype=np.float64)
        crow = cols.reshape(p, 2 * d + 1)
        vrow = vals.reshape(p, 2 * d + 1)
        ar = np.arange(d)
        crow[:, :d] = I[:, None] * d + ar
        crow[:, d:2 * d] = J[:, None] * d + ar
        crow[:, 2 * d] = n * d
        vrow[:, :d] = x[J]
        vrow[:, d:2 * d] = x[I]
        vrow[:, 2 * d] = -1.0
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv)
        obj[-1] = 1.0
        remain = min(seconds - (time.time() - t0), HARD - 4.0 - (time.time() - t_start))
        if remain <= 0.25:
            break
        tl = float(min(5.0, max(0.6, remain - 0.2)))
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": tl})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            y = x + res.x[:-1].reshape(n, d)
            yn = np.linalg.norm(y, axis=1, keepdims=True)
            yn[yn == 0] = 1.0
            y = y / yn
            v = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if v < best:
            x = y
            best = v
            best_x = y.copy()
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if it > 60:
            break
    return best_x, best

def solve():
    t_start = time.time()
    HARD = 165.0
    rng = np.random.default_rng(7)

    def tl():
        return HARD - (time.time() - t_start)

    # load incumbent
    best_X = None
    best_val = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (323, 7):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if not np.all(np.isfinite(a)):
                    continue
                v = _max_cos(a)
                if v < best_val:
                    best_val = v
                    best_X = a.copy()
        except Exception:
            continue
    if best_X is None:
        best_X = rng.standard_normal(size=(323, 7))
        best_X /= np.linalg.norm(best_X, axis=1, keepdims=True)
        best_val = _max_cos(best_X)
    incumbent_X = best_X.copy()
    incumbent_val = best_val

    # analyse contacts to find rattlers/hubs
    try:
        m0, cnt0, G0 = _contacts(incumbent_X, tol=1e-9)
        order_asc = np.argsort(cnt0, kind="stable")
        order_desc = np.argsort(-cnt0, kind="stable")
    except Exception:
        order_asc = np.arange(323)
        order_desc = np.arange(323)[::-1]
        m0 = best_val

    candidates = []  # list of (X,val)

    # helper: shrink-regrow one route
    def shrink_regrow(n_remove, seed, small_steps=1400, relax_steps=220, ncand=10000):
        if tl() < 25.0:
            return None
        r2 = np.random.default_rng(seed)
        # pick removal set: 2 rattlers + hubs + random to fill
        rattlers = list(order_asc[:2])
        hubs = list(order_desc[:max(0, min(n_remove - 2, 8))])
        rem = set(rattlers) | set(hubs)
        # fill randomly if needed
        while len(rem) < n_remove:
            k = int(r2.integers(0, 323))
            rem.add(k)
        rem = sorted(list(rem))
        Xs = np.delete(incumbent_X, rem, axis=0)
        # freedom kick tangential
        kick = r2.standard_normal(size=Xs.shape) * 0.02
        # make tangential
        kick = kick - np.sum(kick * Xs, axis=1, keepdims=True) * Xs
        Xs = Xs + kick
        Xs /= np.linalg.norm(Xs, axis=1, keepdims=True)
        # anneal small code to well below record
        Xs = _tangential_anneal(Xs, small_steps, 0.02, 0.0025, 0.015, 0.0012, r2, t_start, HARD)
        if tl() < 20.0:
            # still try regrow but with fewer steps
            pass
        # regrow one by one
        Xcur = Xs
        for _ in range(n_remove):
            if time.time() - t_start > HARD - 28.0:
                break
            c, _ = _best_void(Xcur, r2, ncand=ncand)
            Xcur = np.vstack([Xcur, c[None, :]])
            # allow all to move
            Xcur = _tangential_anneal(Xcur, relax_steps, 0.006, 0.0018, 0.006, 0.001, r2, t_start, HARD, check_every=300)
        if Xcur.shape[0] != 323:
            return None
        Xcur /= np.linalg.norm(Xcur, axis=1, keepdims=True)
        return Xcur

    # Route A: N=313 (remove 10)
    try:
        if tl() > 30.0:
            Xa = shrink_regrow(10, seed=101, small_steps=1300, relax_steps=200, ncand=9000)
            if Xa is not None:
                va = _max_cos(Xa)
                candidates.append((Xa, va))
    except Exception:
        pass
    # Route B: N=309 (remove 14) with different random fill
    try:
        if tl() > 30.0:
            Xb = shrink_regrow(14, seed=202, small_steps=1100, relax_steps=160, ncand=7000)
            if Xb is not None:
                vb = _max_cos(Xb)
                candidates.append((Xb, vb))
    except Exception:
        pass

    # Reverse route: overfull N=330 then delete 7 worst
    try:
        if tl() > 30.0:
            r3 = np.random.default_rng(303)
            Xover = incumbent_X.copy()
            # add 7 voids iteratively (no relaxation between, fast)
            for _ in range(7):
                if time.time() - t_start > HARD - 30.0:
                    break
                c, _ = _best_void(Xover, r3, ncand=7000)
                Xover = np.vstack([Xover, c[None, :]])
            # small kick
            kick = r3.standard_normal(size=Xover.shape) * 0.008
            kick = kick - np.sum(kick * Xover, axis=1, keepdims=True) * Xover
            Xover = Xover + kick
            Xover /= np.linalg.norm(Xover, axis=1, keepdims=True)
            # optimise overfull
            Xover = _tangential_anneal(Xover, 1200, 0.015, 0.0025, 0.010, 0.001, r3, t_start, HARD)
            if Xover.shape[0] == 330:
                G = Xover @ Xover.T
                np.fill_diagonal(G, -2.0)
                per_max = np.max(G, axis=1)
                # contact count near global max
                gm = float(np.max(per_max))
                per_cnt = np.sum(G > gm - 0.005, axis=1)
                score = per_max + 0.0005 * per_cnt
                worst = np.argsort(-score, kind="stable")[:7]
                keep = np.ones(330, dtype=bool)
                keep[worst] = False
                Xpr = Xover[keep]
                # light relax after prune
                Xpr = _tangential_anneal(Xpr, 500, 0.006, 0.002, 0.005, 0.001, r3, t_start, HARD)
                Xpr /= np.linalg.norm(Xpr, axis=1, keepdims=True)
                vp = _max_cos(Xpr)
                candidates.append((Xpr, vp))
    except Exception:
        pass

    # select best regrown: retain only if competitive (unpolished within margin)
    # margin: within 0.01 of incumbent (generous) else keep incumbent
    regrown_best_X = None
    regrown_best_v = np.inf
    for (Xc, vc) in candidates:
        if np.all(np.isfinite(Xc)) and Xc.shape == (323, 7):
            if vc < regrown_best_v:
                regrown_best_v = vc
                regrown_best_X = Xc
    # decide target for final polish
    target_X = incumbent_X
    target_v = incumbent_val
    # if regrown competitive, prefer it (even if slightly worse, different basin may polish better)
    # use threshold: unpolished within +0.004
    if regrown_best_X is not None and regrown_best_v < incumbent_val + 0.004:
        # pick the best regrown; if multiple, already best
        target_X = regrown_best_X
        target_v = regrown_best_v
        # also keep incumbent as fallback candidate after polish comparison
    else:
        target_X = incumbent_X
        target_v = incumbent_val

    # one bounded LP polish before deadline
    polished_X = None
    polished_v = np.inf
    try:
        if tl() > 8.0:
            secs = min(28.0, max(5.0, tl() - 4.0))
            polished_X, polished_v = _lp_polish(target_X, seconds=secs, t_start=t_start, HARD=HARD)
    except Exception:
        polished_X = None

    # final selection among incumbent, regrown, polished
    best_out = incumbent_X
    best_v = incumbent_val
    if regrown_best_X is not None:
        try:
            v = _max_cos(regrown_best_X)
            if v < best_v:
                best_v = v
                best_out = regrown_best_X
        except Exception:
            pass
    if polished_X is not None:
        try:
            v = _max_cos(polished_X)
            if np.all(np.isfinite(polished_X)) and v < best_v:
                best_v = v
                best_out = polished_X
        except Exception:
            pass
    # also consider target if not yet
    try:
        v = _max_cos(target_X)
        if v < best_v:
            best_v = v
            best_out = target_X
    except Exception:
        pass
    best_out = best_out / np.linalg.norm(best_out, axis=1, keepdims=True)
    if not np.all(np.isfinite(best_out)):
        best_out = incumbent_X / np.linalg.norm(incumbent_X, axis=1, keepdims=True)
    return best_out
