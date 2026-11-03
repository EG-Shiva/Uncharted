import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _contacts(X, m=None, tol=1e-7):
    G = X @ X.T
    if m is None:
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        np.fill_diagonal(G, 1.0)
    else:
        np.fill_diagonal(G, 1.0)
    return np.sum(G > m - tol, axis=1) - 1, m

def _smooth_opt(Y, steps, T0, T1, lr0=0.08, lr1=0.02, mu=0.6):
    n, d = Y.shape
    V = np.zeros_like(Y)
    lT0 = np.log(T0); lT1 = np.log(T1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    for s in range(steps):
        frac = s / max(1, steps - 1)
        T = float(np.exp(lT0 + (lT1 - lT0) * frac))
        lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
        G = Y @ Y.T
        np.fill_diagonal(G, -1.0)
        m = float(np.max(G))
        # softmax weights
        W = (G - m) / T
        np.clip(W, -20.0, 0.0, out=W)
        np.exp(W, out=W)
        np.fill_diagonal(W, 0.0)
        rs = np.sum(W, axis=1)
        WY = W @ Y
        inv = np.zeros_like(rs)
        mk = rs > 1e-12
        inv[mk] = 1.0 / rs[mk]
        avg = WY * inv[:, None]
        dot = np.sum(avg * Y, axis=1, keepdims=True)
        Gt = avg - dot * Y
        Gt[~mk] = 0.0
        V *= mu
        V += Gt
        nm = np.mean(np.linalg.norm(V, axis=1)) + 1e-12
        Y = Y - (lr / nm) * V
        nrm = np.linalg.norm(Y, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        Y /= nrm
    return Y

def _void_reinsert(Y, rng, n_try=4000, n_victims=4):
    n, d = Y.shape
    G = Y @ Y.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    np.fill_diagonal(G, 1.0)
    cont = np.sum(G > m - 1e-7, axis=1) - 1
    # victims: highest contacts (+ any rattler)
    order = np.argsort(-cont)
    victims = list(order[:n_victims])
    # add rattlers explicitly
    for i in range(n):
        if cont[i] == 0 and i not in victims:
            victims.append(i)
    victims = victims[:max(n_victims, 2)]
    for v in victims:
        keep = np.ones(n, dtype=bool); keep[v] = False
        Xk = Y[keep]
        C = rng.standard_normal((n_try, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        S = C @ Xk.T
        sc = S.max(axis=1)
        bi = int(np.argmin(sc))
        # local push-away refine
        vv = C[bi].copy()
        for _ in range(25):
            dv = Xk @ vv
            mm = float(np.max(dv))
            w = np.exp(np.clip((dv - mm) / 0.015, -20, 0))
            g = w @ Xk
            g = g - np.dot(g, vv) * vv
            ng = np.linalg.norm(g)
            if ng < 1e-12:
                break
            g /= ng
            vv = vv - 0.02 * g
            vv /= np.linalg.norm(vv)
        # accept if improves global max
        Ynew = Y.copy(); Ynew[v] = vv
        if _max_cos(Ynew) < _max_cos(Y) - 1e-12:
            Y = Ynew
    return Y

def _lp_polish(X, budget, cap=1400):
    t0 = time.time()
    x = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = 1.5e-3
    iu = np.triu_indices(n, 1)
    it = 0
    while time.time() - t0 < budget and radius > 1e-14:
        it += 1
        if time.time() - t0 + 0.25 > budget and it > 3:
            break
        Gfull = x @ x.T
        c = Gfull[iu]
        m = float(np.max(c))
        thr = max(20.0 * radius, 2e-9)
        sel = c > m - thr
        cnt = int(np.sum(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > cap:
            # keep tightest cap
            idx = np.where(sel)[0]
            vals = c[idx]
            k = np.argsort(-vals)[:cap]
            msk = np.zeros_like(c, dtype=bool)
            msk[idx[k]] = True
            sel = msk
        I = iu[0][sel]; J = iu[1][sel]
        csel = c[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        valsA = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((valsA, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            y = x + res.x[:-1].reshape(n, d)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            v = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    BUDGET = 168.0
    rng = np.random.default_rng(7)
    n, d = 181, 9
    base = None; base_val = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    v = _max_cos(a)
                    if v < base_val:
                        base_val = v; base = a.copy()
        except Exception:
            pass
    if base is None:
        base = rng.standard_normal((n, d))
        base /= np.linalg.norm(base, axis=1, keepdims=True)
        base_val = _max_cos(base)
    # build diverse starts (fast, no LP)
    starts = []
    # 1: strongly perturbed record (new basin attempt)
    for sig in (0.25, 0.45):
        y = base + rng.standard_normal((n, d)) * sig
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        starts.append(y)
    # 2: pure random starts
    for _ in range(4):
        y = rng.standard_normal((n, d))
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        starts.append(y)
    # 3: centrosymmetric start: 90 pairs + 1 pole
    y = rng.standard_normal((90, d)); y /= np.linalg.norm(y, axis=1, keepdims=True)
    sym = np.vstack([y, -y, rng.standard_normal((1, d)) / np.linalg.norm(rng.standard_normal((1, d)))])
    sym = sym + rng.standard_normal(sym.shape) * 0.05
    sym /= np.linalg.norm(sym, axis=1, keepdims=True)
    starts.append(sym)
    # 4: uniform Coulomb-like warm start from random (reuse one)
    best_cand = base.copy()
    best_cval = base_val
    # optimize each start with two-phase softmax; time-boxed
    # allocate ~90s for this phase
    smooth_deadline = t0 + 105.0
    optimized = []
    for si, s in enumerate(starts):
        if time.time() > smooth_deadline:
            break
        Y = s.copy()
        # phase A: uniformity (high T)
        Y = _smooth_opt(Y, steps=450, T0=0.04, T1=0.012, lr0=0.10, lr1=0.05, mu=0.5)
        if time.time() > smooth_deadline + 5:
            v = _max_cos(Y)
            optimized.append((v, Y))
            break
        # phase B: sharpen to min-max (low T)
        Y = _smooth_opt(Y, steps=650, T0=0.012, T1=0.0015, lr0=0.06, lr1=0.015, mu=0.65)
        # void surgery once
        try:
            Y = _void_reinsert(Y, rng, n_try=3000, n_victims=4)
            # short re-sharpen after surgery
            Y = _smooth_opt(Y, steps=250, T0=0.006, T1=0.0012, lr0=0.04, lr1=0.01, mu=0.6)
        except Exception:
            pass
        v = _max_cos(Y)
        optimized.append((v, Y))
        if v < best_cval:
            best_cval = v; best_cand = Y.copy()
    # also consider base itself
    optimized.append((base_val, base.copy()))
    # rank and keep top 2 for LP
    optimized.sort(key=lambda t: t[0])
    top = optimized[:2]
    # LP polish top candidates with remaining time
    rem = BUDGET - (time.time() - t0)
    # reserve 1s for return
    lp_total = max(8.0, min(45.0, rem - 2.0))
    per = lp_total / len(top)
    final_X = top[0][1].copy(); final_V = top[0][0]
    # if best overall is base and base already LP-optimal, still lightly polish challenger
    for (v, X) in top:
        if time.time() - t0 > BUDGET - 2.0:
            break
        try:
            Xp = _lp_polish(X, budget=max(4.0, per))
            vp = _max_cos(Xp)
        except Exception:
            continue
        if vp < final_V:
            final_V = vp; final_X = Xp.copy()
    # ensure we never regress vs base
    if final_V > base_val and base is not None:
        final_X = base.copy(); final_V = base_val
    final_X = final_X / np.linalg.norm(final_X, axis=1, keepdims=True)
    return final_X
