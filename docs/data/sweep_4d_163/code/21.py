import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _normalize(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _degrees(x, tol=1e-7):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    mask = G >= (m - tol)
    deg = mask.sum(axis=1)
    rowmax = G.max(axis=1)
    return deg, rowmax, m, G

def _smooth_anneal(X0, steps, a0, a1, lr0, lr1, mu, rng, noise0=0.0):
    X = X0.copy()
    V = np.zeros_like(X)
    for s in range(steps):
        frac = s / max(1, steps - 1)
        alpha = a0 * (a1 / a0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = np.exp(alpha * (G - m))
        np.fill_diagonal(E, 0.0)
        S = float(E.sum())
        if S == 0 or not np.isfinite(S):
            S = 1.0
        Grad = (E @ X) / S
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        if s == 0:
            V = (1.0 - mu) * Gt
        else:
            V = mu * V + Gt
        X = X - lr * V
        if noise0 > 0:
            nz = noise0 * (1.0 - frac)
            if nz > 1e-12:
                R = rng.standard_normal(size=X.shape)
                R -= np.sum(R * X, axis=1, keepdims=True) * X
                X = X + nz * R
        X = _normalize(X)
    return X

def _hard_polish(X0, steps, lr0, decay=0.997):
    cur = X0.copy()
    best = cur.copy()
    bv = _max_cos(cur)
    n = cur.shape[0]
    for s in range(steps):
        G = cur @ cur.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        delta = 2e-3 * (0.985 ** s) + 1e-7
        mask = (G > (m - delta)).astype(float)
        deg = mask.sum(axis=1, keepdims=True) + 1.0
        Grad = mask @ cur
        dot = np.sum(Grad * cur, axis=1, keepdims=True)
        Gt = (Grad - dot * cur) / deg
        lr = lr0 * (decay ** s)
        cur = _normalize(cur - lr * Gt)
        if (s % 10 == 0) or s == steps - 1:
            v = _max_cos(cur)
            if v < bv:
                bv = v
                best = cur.copy()
    return best, bv

_IU = None
def _lp_polish(x, seconds, r0=2e-3):
    global _IU
    x = _normalize(x.copy())
    n, d = x.shape
    if _IU is None or _IU[0].shape[0] != n * (n - 1) // 2:
        _IU = np.triu_indices(n, 1)
    iu = _IU
    best = _max_cos(x)
    bx = x.copy()
    radius = r0
    start = time.time()
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu]
        m = float(c_all.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = c_all > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 2200:
            vals = c_all[sel]
            kth = np.sort(vals)[-2200]
            sel = c_all >= kth
            cnt = int(np.count_nonzero(sel))
            if cnt > 3000:
                ids = np.where(sel)[0][::2][:3000]
                ns = np.zeros_like(sel)
                ns[ids] = True
                sel = ns
        I = iu[0][sel]; J = iu[1][sel]
        c = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cd = np.arange(d)
        cols = np.concatenate([I[:, None] * d + cd, J[:, None] * d + cd, np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = _normalize(x + res.x[:-1].reshape(n, d))
        v = _max_cos(y)
        if v < best:
            x, best, bx = y, v, y.copy()
            radius = min(radius * 1.6, 0.05)
        else:
            radius *= 0.5
        if time.time() - start >= seconds:
            break
    return bx, best

def _void_point(X, exclude, rng, n_cand=15000):
    cand = _normalize(rng.standard_normal(size=(n_cand, 4)))
    D = cand @ X.T
    if exclude is not None and len(exclude) > 0:
        D[:, list(exclude)] = -2.0
    cm = D.max(axis=1)
    bi = int(np.argmin(cm))
    y = cand[bi].copy()
    Xn = X
    ii_ex = -1
    if exclude is not None and len(exclude) == 1:
        ii_ex = list(exclude)[0]
    for t in range(200):
        d = Xn @ y
        if ii_ex >= 0:
            d[ii_ex] = -2.0
        j = int(np.argmax(d))
        gvec = Xn[j] - float(Xn[j] @ y) * y
        step = 0.08 * (0.99 ** t)
        y = y - step * gvec
        ny = float(np.linalg.norm(y))
        if ny == 0 or not np.isfinite(ny):
            y = cand[bi].copy()
            break
        y = y / ny
    return y

def _reinsert(X, idx_list, rng, n_cand=15000):
    Xn = X.copy()
    for ii in idx_list:
        y = _void_point(Xn, [ii], rng, n_cand=n_cand)
        Xn[ii] = y
    return Xn

def solve():
    t0 = time.time()
    BUDGET = 168.0
    rng = np.random.default_rng(12345)
    cands = []
    try:
        xb = np.load("best.npy").astype(float)
        cands.append(_normalize(xb))
    except Exception:
        pass
    try:
        xr = np.load("record.npy").astype(float)
        cands.append(_normalize(xr))
    except Exception:
        pass
    if not cands:
        cands.append(_normalize(rng.standard_normal(size=(163, 4))))
    best = None
    best_val = 2.0
    for c in cands:
        v = _max_cos(c)
        if v < best_val:
            best_val = v
            best = c.copy()
    try:
        pb, pv = _hard_polish(best, 200, 0.002)
        if pv < best_val:
            best, best_val = pb, pv
    except Exception:
        pass
    # quick LP to reach exact local optimum
    try:
        rem = BUDGET - (time.time() - t0)
        if rem > 30:
            lb, lv = _lp_polish(best, seconds=4.0, r0=2e-3)
            if lv < best_val and np.all(np.isfinite(lb)):
                best, best_val = lb, lv
    except Exception:
        pass

    sigmas = [0.008, 0.02, 0.05, 0.10, 0.015, 0.035, 0.07, 0.005, 0.025, 0.06]
    restart = 0
    while True:
        if time.time() - t0 > BUDGET - 9.0:
            break
        restart += 1
        sigma = sigmas[(restart - 1) % len(sigmas)]
        mode = restart % 7
        Xt = best.copy()
        try:
            deg, rowmax, mval, G = _degrees(Xt, tol=1e-7)
        except Exception:
            deg, rowmax = None, None
        try:
            if mode == 0:
                # reinsert 3 worst + jitter, coherent cluster shift
                order = np.argsort(rowmax)[-3:]
                Xt = _reinsert(Xt, order.tolist(), rng, n_cand=12000)
                Xt = _normalize(Xt + sigma * rng.standard_normal(size=Xt.shape))
                # coherent shift of jammed cluster
                seed = int(np.argmax(deg))
                dots = (Xt @ Xt[seed])
                dots[seed] = -2
                nbr = np.argsort(dots)[-6:]
                shift = rng.standard_normal(size=(4,)) * 0.05
                Xt[nbr] = _normalize(Xt[nbr] + shift)
                steps, a0, a1, lr0, lr1, mu = 1400, 8.0, 350.0, 0.6, 0.02, 0.85
                nz = 0.0
            elif mode == 1:
                order = np.argsort(rowmax)[-2:]
                Xt = _reinsert(Xt, order.tolist(), rng, n_cand=12000)
                Xt = _normalize(Xt + sigma * rng.standard_normal(size=Xt.shape))
                steps, a0, a1, lr0, lr1, mu = 1200, 12.0, 300.0, 0.5, 0.015, 0.9
                nz = 0.0
            elif mode == 2:
                Xt = _normalize(Xt + sigma * rng.standard_normal(size=Xt.shape))
                if restart % 10 == 5 and deg is not None:
                    low = np.argsort(rowmax)[:3]
                    Xt[low] = _normalize(rng.standard_normal(size=(3, 4)))
                steps, a0, a1, lr0, lr1, mu = 1100, 15.0, 400.0, 0.4, 0.01, 0.88
                nz = 0.0
            elif mode == 3:
                Xt = _normalize(Xt + sigma * rng.standard_normal(size=Xt.shape))
                # coherent move: shift a random neighborhood together
                seed = int(rng.integers(0, 163))
                dots = (Xt @ Xt[seed])
                dots[seed] = -2
                nbr = np.argsort(dots)[-8:]
                shift = rng.standard_normal(size=(4,)) * 0.08
                Xt[nbr] = _normalize(Xt[nbr] + shift + 0.02 * rng.standard_normal(size=(len(nbr), 4)))
                steps, a0, a1, lr0, lr1, mu = 1600, 6.0, 250.0, 0.7, 0.02, 0.82
                nz = 0.002
            elif mode == 4:
                # rattlers to voids
                ratt = np.where(deg <= 1)[0]
                if len(ratt) > 0:
                    k = min(len(ratt), 3)
                    Xt = _reinsert(Xt, ratt[:k].tolist(), rng, n_cand=20000)
                else:
                    order = np.argsort(rowmax)[-1:]
                    Xt = _reinsert(Xt, order.tolist(), rng, n_cand=20000)
                Xt = _normalize(Xt + 0.01 * rng.standard_normal(size=Xt.shape))
                steps, a0, a1, lr0, lr1, mu = 1200, 10.0, 350.0, 0.5, 0.012, 0.9
                nz = 0.0
            elif mode == 5:
                # heavy randomize overloaded + neighbors
                hi = np.argsort(deg)[-4:]
                Xt[hi] = _normalize(rng.standard_normal(size=(len(hi), 4)))
                Xt = _normalize(Xt + sigma * rng.standard_normal(size=Xt.shape))
                steps, a0, a1, lr0, lr1, mu = 1300, 10.0, 300.0, 0.55, 0.015, 0.87
                nz = 0.001
            else:
                Xt = _normalize(Xt + sigma * rng.standard_normal(size=Xt.shape))
                steps, a0, a1, lr0, lr1, mu = 900, 20.0, 500.0, 0.3, 0.008, 0.9
                nz = 0.0
        except Exception:
            Xt = _normalize(best + sigma * rng.standard_normal(size=best.shape))
            steps, a0, a1, lr0, lr1, mu, nz = 900, 12.0, 300.0, 0.5, 0.015, 0.9, 0.0
        rem = BUDGET - 9.0 - (time.time() - t0)
        if rem < 12.0:
            steps = min(steps, 400)
        try:
            Xo = _smooth_anneal(Xt, steps, a0, a1, lr0, lr1, mu, rng, noise0=nz)
        except Exception:
            continue
        try:
            Xp, vp = _hard_polish(Xo, 250, 0.0025)
        except Exception:
            continue
        if vp < best_val + 2e-4:
            try:
                Xp2, vp2 = _hard_polish(Xp, 250, 0.0008, decay=0.998)
                Xp, vp = Xp2, vp2
            except Exception:
                pass
        improved = (vp < best_val - 1e-12) and np.all(np.isfinite(Xp))
        if improved:
            # LP to snap to local optimum
            try:
                rem2 = BUDGET - 9.0 - (time.time() - t0)
                if rem2 > 6.0:
                    Xl, vl = _lp_polish(Xp, seconds=min(3.0, rem2 * 0.2), r0=1.5e-3)
                    if vl < vp and np.all(np.isfinite(Xl)):
                        Xp, vp = Xl, vl
            except Exception:
                pass
            if vp < best_val:
                best = Xp.copy()
                best_val = vp
                if best_val < 0.856415430312 - 5e-5:
                    try:
                        Xf, vf = _hard_polish(best, 500, 0.0012, decay=0.9985)
                        if vf < best_val:
                            best, best_val = Xf, vf
                    except Exception:
                        pass
        if time.time() - t0 > BUDGET - 9.0:
            break
    try:
        rem = BUDGET - (time.time() - t0) - 1.0
        if rem > 5.0:
            Xl, vl = _lp_polish(best, seconds=min(6.0, rem * 0.5), r0=1.5e-3)
            if vl < best_val and np.all(np.isfinite(Xl)):
                best, best_val = Xl, vl
        rem = BUDGET - (time.time() - t0) - 0.5
        if rem > 2.0:
            s = min(800, int(rem * 110))
            Xf, vf = _hard_polish(best, s, 0.0012, decay=0.9985)
            if vf < best_val:
                best, best_val = Xf, vf
    except Exception:
        pass
    best = _normalize(best)
    return best
