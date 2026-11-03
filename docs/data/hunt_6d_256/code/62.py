import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _c0_c100(x):
    n = x.shape[0]
    G = x @ x.T
    iu = np.triu_indices(n, 1)
    c = G[iu[0], iu[1]]
    # partial sort for top 101
    T = 101
    idx = np.argpartition(-c, T - 1)[:T]
    top = np.sort(c[idx])[::-1]
    return float(top[0]), float(top[T - 1])

def _adam_lse(x0, steps, K0, K1, lr, deadline=1e18):
    x = x0.copy()
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    la0, la1 = np.log(K0), np.log(K1)
    best_val = _max_cos(x)
    best_x = x.copy()
    for t in range(1, steps + 1):
        if time.time() > deadline and t > 10:
            break
        K = float(np.exp(la0 + (la1 - la0) * t / steps))
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        W = np.exp(K * (G - gmax))
        np.fill_diagonal(W, 0.0)
        S = float(W.sum() * 0.5)
        if not np.isfinite(S) or S <= 0:
            continue
        grad = (W @ x) / S
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        b1p *= b1
        b2p *= b2
        m *= b1
        m += (1 - b1) * gt
        v *= b2
        v += (1 - b2) * (gt * gt)
        mhat = m / (1 - b1p)
        vhat = v / (1 - b2p)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        sdot = np.sum(step * x, axis=1, keepdims=True)
        step = step - sdot * x
        x = x - step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        if t % 200 == 0 or t == steps:
            val = _max_cos(x)
            if val < best_val:
                best_val = val
                best_x = x.copy()
    return best_x, best_val

def _adam_riesz(x0, steps, p, lr, deadline=1e18):
    x = x0.copy()
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    expc = float(p * 0.5 - 1.0)
    best_val = _max_cos(x)
    best_x = x.copy()
    for t in range(1, steps + 1):
        if time.time() > deadline and t > 10:
            break
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        umax = (1.0 + gmax) * 0.5
        if umax <= 1e-12:
            umax = 0.5
        logUmax = np.log(umax)
        U = (1.0 + G) * 0.5
        np.fill_diagonal(U, 1.0)
        logU = np.log(np.maximum(U, 1e-300))
        E = expc * (logU - logUmax)
        np.fill_diagonal(E, -100.0)
        np.clip(E, -50.0, 0.0, out=E)
        W = np.exp(E)
        np.fill_diagonal(W, 0.0)
        S = float(W.sum() * 0.5)
        if not np.isfinite(S) or S <= 0:
            continue
        grad = (W @ x) / S
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        b1p *= b1
        b2p *= b2
        m *= b1
        m += (1 - b1) * gt
        v *= b2
        v += (1 - b2) * (gt * gt)
        mhat = m / (1 - b1p)
        vhat = v / (1 - b2p)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        sdot = np.sum(step * x, axis=1, keepdims=True)
        step = step - sdot * x
        x = x - step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        if t % 100 == 0 or t == steps:
            val = _max_cos(x)
            if val < best_val:
                best_val = val
                best_x = x.copy()
    return best_x, best_val

def _lp_polish(x, deadline, r0=2.5e-3, cap=3500):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    iuI, iuJ = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iuI, iuJ]
        mv = float(c.max())
        margin = max(20.0 * radius, 5e-5)
        sel = c > (mv - margin)
        I = iuI[sel]; J = iuJ[sel]
        csel = c[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > cap:
            part = np.argpartition(-csel, cap)[:cap]
            I = I[part]; J = J[part]; csel = csel[part]
            p = cap
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, n * d + 1)).tocsr()
        except Exception:
            break
        obj = np.zeros(n * d + 1)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        val = _max_cos(y)
        if val < best:
            x, best = y, val
            radius = min(radius * 1.4, 0.02)
        else:
            radius *= 0.5
    return x

def _refine_single(y0, F, steps=60, K=2500.0, lr=0.02):
    y = y0.copy()
    y = y / np.linalg.norm(y)
    m1 = np.zeros_like(y)
    v1 = np.zeros_like(y)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    best_y = y.copy()
    best_val = float((F @ y).max())
    for t in range(1, steps + 1):
        g = F @ y
        gmax = float(g.max())
        w = np.exp(K * (g - gmax))
        s = float(w.sum())
        if not np.isfinite(s) or s <= 0:
            break
        w = w / s
        grad = w @ F
        gt = grad - float(grad @ y) * y
        b1p *= b1; b2p *= b2
        m1 *= b1; m1 += (1 - b1) * gt
        v1 *= b2; v1 += (1 - b2) * (gt * gt)
        mhat = m1 / (1 - b1p); vhat = v1 / (1 - b2p)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        step = step - float(step @ y) * y
        y = y - step
        y = y / np.linalg.norm(y)
        if t % 20 == 0 or t == steps:
            gv = float((F @ y).max())
            if gv < best_val:
                best_val = gv
                best_y = y.copy()
    return best_y, best_val

def solve():
    import time as _time
    import numpy as _np
    t0 = _time.time()
    DEADLINE = t0 + 178.0
    rng = _np.random.default_rng(20261103)
    n, d = 256, 6
    base = None
    base_v = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = _np.load(fn)
            a = _np.asarray(a, dtype=float)
            if a.shape == (256, 6):
                a = a / _np.linalg.norm(a, axis=1, keepdims=True)
                if _np.all(_np.isfinite(a)):
                    vv = _max_cos(a)
                    if vv < base_v:
                        base_v = vv
                        base = a.copy()
        except Exception:
            pass
    if base is None:
        rs = _np.random.default_rng(0)
        base = rs.normal(size=(n, d))
        base /= _np.linalg.norm(base, axis=1, keepdims=True)
        base_v = _max_cos(base)
    inc_x = base.copy()
    inc_c0, inc_c100 = _c0_c100(inc_x)
    best_x = inc_x.copy()
    best_c0, best_c100 = inc_c0, inc_c100

    NCAND = 200000
    Z = rng.normal(size=(NCAND, d))
    Z /= _np.linalg.norm(Z, axis=1, keepdims=True)
    CHUNK = 20000

    def _participation68(x, thr=0.68):
        G = x @ x.T
        _np.fill_diagonal(G, -2.0)
        part = _np.sum(G > thr, axis=1)
        rowmax = G.max(axis=1)
        return part, rowmax

    k_list = [32, 16, 48]
    for tnum, k in enumerate(k_list):
        if _time.time() > DEADLINE - 40:
            break
        try:
            part, rowmax = _participation68(inc_x, 0.68)
        except Exception:
            continue
        nhub = k // 2
        order = _np.argsort(-part, kind='stable')
        hub = order[:nhub].tolist()
        hubset = set(hub)
        # random half among remainder
        rem = _np.array([i for i in range(n) if i not in hubset], dtype=int)
        rrg = _np.random.default_rng(9000 + k * 131 + tnum * 17)
        nrand = k - nhub
        rand_idx = rrg.choice(rem, size=nrand, replace=False).tolist()
        del_idx = _np.array(hub + rand_idx, dtype=int)
        keep = _np.ones(n, dtype=bool)
        keep[del_idx] = False
        S = inc_x[keep].copy()
        # hard optimize survivors to lower density: Riesz annealing p 20->200 + LP
        try:
            S, _ = _adam_riesz(S, 220, 20.0, 0.012, deadline=DEADLINE - 25)
        except Exception:
            pass
        if _time.time() > DEADLINE - 30:
            continue
        try:
            S, _ = _adam_riesz(S, 220, 80.0, 0.010, deadline=DEADLINE - 25)
        except Exception:
            pass
        try:
            S, _ = _adam_riesz(S, 200, 200.0, 0.008, deadline=DEADLINE - 25)
        except Exception:
            pass
        if _time.time() < DEADLINE - 28:
            try:
                S = _lp_polish(S, min(_time.time() + 4.0, DEADLINE - 24), r0=3.0e-3, cap=3500)
            except Exception:
                pass
        try:
            s_c0, s_c100 = _c0_c100(S)
        except Exception:
            continue
        # require at least 3e-3 below incumbent max, else extra LSE push
        target = inc_c0 - 3.0e-3
        if s_c0 > target:
            try:
                S2, _ = _adam_lse(S, 300, 300.0, 5000.0, 0.008, deadline=DEADLINE - 24)
                nc, _ = _c0_c100(S2)
                if nc < s_c0:
                    S = S2
                    s_c0 = nc
            except Exception:
                pass
        # regrow one at a time
        F = S.copy()
        ok = True
        for s in range(k):
            if _time.time() > DEADLINE - 28:
                ok = False
                break
            gbest = 1e9
            gbi = -1
            # chunked search minimizing max dot to F
            for a0 in range(0, NCAND, CHUNK):
                Zb = Z[a0:a0 + CHUNK]
                Cm = (Zb @ F.T).max(axis=1)
                bi = int(_np.argmin(Cm))
                bv = float(Cm[bi])
                if bv < gbest:
                    gbest = bv
                    gbi = a0 + bi
            y0 = Z[gbi].copy()
            try:
                yref, _ = _refine_single(y0, F, steps=60, K=2500.0, lr=0.02)
            except Exception:
                yref = y0
            F = _np.vstack([F, yref[None, :]])
            try:
                cur_c0 = _max_cos(F)
            except Exception:
                cur_c0 = 1e9
            # backtracking: if insertion pushes max above 0.7055, joint re-optimise
            if cur_c0 > 0.7055:
                try:
                    F, _ = _adam_lse(F, 180, 300.0, 4000.0, 0.008, deadline=DEADLINE - 22)
                    cur_c0 = _max_cos(F)
                except Exception:
                    pass
                if cur_c0 > 0.7055 and _time.time() < DEADLINE - 24:
                    try:
                        F = _lp_polish(F, min(_time.time() + 2.0, DEADLINE - 22), r0=2.0e-3, cap=3000)
                        cur_c0 = _max_cos(F)
                    except Exception:
                        pass
                # retry different void if still above: replace last point by perturbed far candidate
                if cur_c0 > 0.7055:
                    try:
                        # drop last, pick candidate far from failed point
                        F = F[:-1]
                        yfail = yref
                        # score candidates by max dot to F plus penalty for proximity to yfail
                        bbest = 1e9
                        bbi = -1
                        for a0 in range(0, NCAND, CHUNK):
                            Zb = Z[a0:a0 + CHUNK]
                            Cm = (Zb @ F.T).max(axis=1)
                            prox = Zb @ yfail
                            score = Cm + 0.15 * np.maximum(prox, 0.0)
                            bi = int(_np.argmin(score))
                            bv = float(score[bi])
                            if bv < bbest:
                                bbest = bv
                                bbi = a0 + bi
                        y0b = Z[bbi].copy()
                        yref2, _ = _refine_single(y0b, F, steps=60, K=2500.0, lr=0.02)
                        F = _np.vstack([F, yref2[None, :]])
                    except Exception:
                        F = _np.vstack([F, yref[None, :]])
            # joint LSE+LP refit after every batch of 8 insertions
            if (s + 1) % 8 == 0 or s == k - 1:
                if _time.time() > DEADLINE - 26:
                    continue
                try:
                    F, _ = _adam_lse(F, 260, 300.0, 5000.0, 0.008, deadline=DEADLINE - 20)
                except Exception:
                    pass
                if _time.time() < DEADLINE - 22:
                    try:
                        F = _lp_polish(F, min(_time.time() + 2.5, DEADLINE - 20), r0=2.0e-3, cap=3000)
                    except Exception:
                        pass
        if not ok:
            continue
        if F.shape[0] != 256:
            continue
        try:
            F = F / _np.linalg.norm(F, axis=1, keepdims=True)
            nc0, nc100 = _c0_c100(F)
        except Exception:
            continue
        # score on (c0, c100): strict c0 win, or plateau win within tolerance
        take = False
        if nc0 < best_c0 - 1e-12:
            take = True
        elif abs(nc0 - best_c0) <= 2.0e-6 and nc100 < best_c100 - 1.0e-6:
            take = True
        if take:
            best_x = F.copy()
            best_c0, best_c100 = nc0, nc100
            inc_x = best_x.copy()
            inc_c0, inc_c100 = best_c0, best_c100
    # final polish best
    try:
        if _time.time() < DEADLINE - 10:
            bx, _ = _adam_lse(best_x, 350, 500.0, 8000.0, 0.006, deadline=DEADLINE - 5)
            if _max_cos(bx) < best_c0:
                best_x = bx
                best_c0, best_c100 = _c0_c100(best_x)
    except Exception:
        pass
    try:
        if _time.time() < DEADLINE - 4:
            bx2 = _lp_polish(best_x, DEADLINE - 1.0, r0=2.0e-3, cap=3500)
            if _max_cos(bx2) < best_c0:
                best_x = bx2
    except Exception:
        pass
    # guaranteed fallback to lab best
    try:
        b = _np.load("best.npy")
        b = _np.asarray(b, dtype=float)
        if b.shape == (256, 6):
            b = b / _np.linalg.norm(b, axis=1, keepdims=True)
            if _max_cos(b) < _max_cos(best_x):
                best_x = b
    except Exception:
        pass
    best_x = best_x / _np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
