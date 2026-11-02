import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _rowmax(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1), g

def _lp_once(x, radius, cap=3000, thr_mult=20.0, thr_min=2e-12):
    n, d = x.shape
    iu0, iu1 = np.triu_indices(n, 1)
    G = x @ x.T
    cf = G[iu0, iu1]
    m = float(cf.max())
    thr = max(thr_mult * radius, thr_min)
    sel = cf > m - thr
    cnt = int(sel.sum())
    if cnt == 0:
        return x, m, False
    if cnt > cap:
        # keep largest
        idxp = np.argpartition(-cf, cap)[:cap]
        mk = np.zeros_like(sel)
        mk[idxp] = True
        sel = mk
    c = cf[sel]
    I = iu0[sel]
    J = iu1[sel]
    p = len(I)
    nv = n * d + 1
    rows = np.repeat(np.arange(p), 2 * d + 1)
    cols = np.concatenate([I[:, None] * d + np.arange(d),
                           J[:, None] * d + np.arange(d),
                           np.full((p, 1), n * d)], 1).ravel()
    valsA = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
    A_ub = coo_matrix((valsA, (rows, cols)), shape=(p, nv)).tocsr()
    A_eq = coo_matrix((x.ravel(),
                       (np.repeat(np.arange(n), d), np.arange(n * d))),
                      shape=(n, nv)).tocsr()
    obj = np.zeros(nv)
    obj[-1] = 1.0
    try:
        res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq,
                      b_eq=np.zeros(n),
                      bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                      method="highs")
    except Exception:
        return x, m, False
    if res.status != 0:
        return x, m, False
    y = x + res.x[:-1].reshape(n, d)
    yn = np.linalg.norm(y, axis=1, keepdims=True)
    yn[yn == 0] = 1.0
    y = y / yn
    v = _max_cos(y)
    return y, v, True

def _lp_polish(x, deadline, radius_init=5e-4, radius_min=1e-14, grow=1.5, shrink=0.5, cap=3000):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    best = _max_cos(x)
    best_x = x.copy()
    radius = radius_init
    while time.time() < deadline and radius > radius_min:
        Gfull = x @ x.T
        iu0, iu1 = np.triu_indices(x.shape[0], 1)
        cf = Gfull[iu0, iu1]
        m = float(cf.max())
        y, v, ok = _lp_once(x, radius, cap=cap)
        if not ok:
            radius *= shrink
            continue
        # v is max of proposed y
        if v < best:
            x, best = y, v
            best_x = y.copy()
            radius = min(radius * grow, 0.02)
        else:
            # accept move if it tracks best_x drift? only shrink
            # if y equals x (no move) shrink
            # check if y moved
            if v < m + 1e-15:
                # stuck, shrink to refine
                x = y
                radius *= shrink
            else:
                radius *= shrink
        if best < _max_cos(best_x):
            best_x = x.copy()
    return best_x, best

def _micro_polish(x, deadline, radii=(2e-4, 1e-4, 5e-5, 2e-5, 1e-5, 5e-6, 2e-6, 1e-6)):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    best = _max_cos(x)
    best_x = x.copy()
    for r in radii:
        it = 0
        while time.time() < deadline and it < 30:
            it += 1
            y, v, ok = _lp_once(x, r, cap=3000, thr_mult=30.0, thr_min=5e-11)
            if not ok:
                break
            v = _max_cos(y)
            if v < best - 1e-15:
                best = v
                best_x = y.copy()
                x = y
            elif v < _max_cos(x) - 1e-15:
                x = y
            else:
                # no improvement, try one more from best_x?
                if _max_cos(x) > best + 1e-14:
                    x = best_x.copy()
                else:
                    break
    return best_x, best

def _make_lse_fun(n, d, iu0, iu1, beta):
    Iall = iu0
    Jall = iu1
    def fun(flat):
        X = flat.reshape(n, d)
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        Y = X / nr
        G = Y @ Y.T
        c = G[iu0, iu1]
        m = float(c.max())
        e = np.exp(beta * (c - m))
        s = float(e.sum())
        if not np.isfinite(s) or s <= 0:
            return float(m), np.zeros_like(flat)
        f = m + np.log(s) / beta
        w = e / s
        W = np.zeros((n, n))
        W[Iall, Jall] = w
        W[Jall, Iall] = w
        GY = W @ Y
        dot = (GY * Y).sum(axis=1, keepdims=True)
        GX = (GY - dot * Y) / nr
        return float(f), GX.ravel()
    return fun

def _lbfgs_anneal(X, betas, maxiters, deadline):
    n, d = X.shape
    iu = np.triu_indices(n, 1)
    Xf = X.ravel().copy()
    for beta, mi in zip(betas, maxiters):
        if time.time() > deadline:
            break
        fun = _make_lse_fun(n, d, iu[0], iu[1], float(beta))
        try:
            res = minimize(fun, Xf, jac=True, method="L-BFGS-B",
                           options={"maxiter": int(mi), "ftol": 1e-20, "gtol": 1e-14, "maxfun": int(mi * 4)})
            Xf = res.x
        except Exception:
            break
    X = Xf.reshape(n, d)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _refine_single(v0, Others, beta=800.0):
    # Others: (m,d) fixed, optimize v on sphere to minimize max dot
    def fun(flat):
        nr = max(np.linalg.norm(flat), 1e-12)
        y = flat / nr
        dots = Others @ y
        m = float(dots.max())
        e = np.exp(beta * (dots - m))
        s = float(e.sum())
        f = m + np.log(s) / beta
        w = e / s
        # grad w.r.t y
        gy = Others.T @ w  # (d,)
        # chain to flat
        gx = (gy - (gy @ y) * y) / nr
        return float(f), gx
    try:
        res = minimize(fun, v0.copy(), jac=True, method="L-BFGS-B",
                       options={"maxiter": 150, "ftol": 1e-20, "gtol": 1e-14})
        v = res.x
        v = v / np.linalg.norm(v)
        return v
    except Exception:
        return v0 / np.linalg.norm(v0)

def _reinsert_worst(Y, idxs, rng, trials=20000, beta=800.0):
    n, d = Y.shape
    # shared random candidates
    C = rng.standard_normal((trials, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    # D = C @ Y.T  (trials x n)
    D = C @ Y.T
    best_Y = Y.copy()
    best_val = _max_cos(Y)
    improved = False
    for idx in idxs:
        # row max for candidates excluding idx
        D[:, idx] = -2.0  # ensure ignored? actually D is C vs Y, col idx is dot with old point at idx; ignore
        # for candidate evaluation, max over j != idx
        cm = D.max(axis=1)
        # take top few
        order = np.argpartition(cm, 5)[:5]
        # order them
        order = order[np.argsort(cm[order])]
        for k in order:
            v0 = C[k].copy()
            Others = np.delete(best_Y, idx, axis=0)
            vref = _refine_single(v0, Others, beta=beta)
            Y2 = best_Y.copy()
            Y2[idx] = vref
            v = _max_cos(Y2)
            if v < best_val - 1e-12:
                best_Y, best_val = Y2, v
                improved = True
                # update D column for next idx? recompute D
                D = C @ best_Y.T
                break
        # refresh D col ignore
        D[:, idx] = -2.0
    return best_Y, best_val, improved

def solve():
    t0 = time.time()
    BUDGET = 172.0
    DEADLINE = t0 + BUDGET
    np.random.seed(1)
    rng = np.random.default_rng(1)
    n, d = 121, 8
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
        X0 = rng.standard_normal((n, d))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        cands = [X0]
    vals = [_max_cos(a) for a in cands]
    best_x = cands[int(np.argmin(vals))].copy()
    best = float(min(vals))

    # Phase 1: coarse LP polish on loaded best (short)
    try:
        bx, bv = _lp_polish(best_x, DEADLINE - 155, radius_init=4e-4)
        if bv < best:
            best_x, best = bx, bv
    except Exception:
        pass

    # Phase 2: L-BFGS anneal from best
    if time.time() < DEADLINE - 140:
        try:
            X = _lbfgs_anneal(best_x, betas=[120, 350, 900, 2200], maxiters=[120, 120, 120, 150], deadline=DEADLINE - 135)
            v = _max_cos(X)
            if v < best:
                best_x, best = X, v
            else:
                # keep LBFGS result if close, LP may improve it
                # try LP on X briefly
                bx2, bv2 = _lp_polish(X, time.time() + 4.0, radius_init=2e-4)
                if bv2 < best:
                    best_x, best = bx2, bv2
        except Exception:
            pass

    # Phase 3: targeted re-insertion of overloaded / worst points on best
    if time.time() < DEADLINE - 120:
        try:
            G = best_x @ best_x.T
            np.fill_diagonal(G, -2.0)
            rm = G.max(axis=1)
            # contacts within 1e-9 of max
            thr = best - 1e-9
            contacts = (G >= thr).sum(axis=1)
            # order: worst rowmax first, then high contacts
            order_rm = np.argsort(-rm)
            order_ct = np.argsort(-contacts)
            # try worst 6
            cand_idx = list(order_rm[:6])
            for i in order_ct[:4]:
                if i not in cand_idx:
                    cand_idx.append(int(i))
            Y2, v2, imp = _reinsert_worst(best_x, cand_idx[:8], rng, trials=25000, beta=600.0)
            if v2 < best:
                best_x, best = Y2, v2
                bx, bv = _lp_polish(best_x, time.time() + 5.0, radius_init=2e-4)
                if bv < best:
                    best_x, best = bx, bv
        except Exception:
            pass

    # Phase 4: kick-and-refine basin hopping
    sigmas = [0.008, 0.015, 0.03, 0.05, 0.012, 0.022]
    it = 0
    while time.time() < DEADLINE - 28:
        it += 1
        sig = sigmas[(it - 1) % len(sigmas)]
        try:
            yp = best_x + sig * rng.standard_normal((n, d))
            yp /= np.linalg.norm(yp, axis=1, keepdims=True)
            # short L-BFGS
            X = _lbfgs_anneal(yp, betas=[250, 800], maxiters=[80, 100], deadline=DEADLINE - 26)
            v = _max_cos(X)
            # greedy reinsertion on worst 3 with smaller trials for speed
            if v < best + 0.005:
                G = X @ X.T
                np.fill_diagonal(G, -2.0)
                rm = G.max(axis=1)
                order = np.argsort(-rm)[:4]
                X2, v2, imp = _reinsert_worst(X, order, rng, trials=15000, beta=600.0)
                if v2 < v:
                    X, v = X2, v2
            # short LP if promising
            if v < best + 0.002:
                Xp, vp = _lp_polish(X, time.time() + 3.0, radius_init=2e-4)
                if vp < v:
                    X, v = Xp, vp
            if v < best - 1e-12:
                best_x, best = X, v
        except Exception:
            continue

    # Phase 5: final micro polish
    if time.time() < DEADLINE - 20:
        try:
            bx, bv = _lp_polish(best_x, DEADLINE - 12, radius_init=1e-4)
            if bv < best:
                best_x, best = bx, bv
        except Exception:
            pass
    if time.time() < DEADLINE - 10:
        try:
            bx, bv = _micro_polish(best_x, DEADLINE - 1.0)
            if bv < best:
                best_x, best = bx, bv
        except Exception:
            pass

    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
