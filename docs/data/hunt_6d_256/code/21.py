import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _gram(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return G

def _adam_sm(x0, steps, K0, K1, lr):
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
        mhat = m / (1.0 - b1p)
        vhat = v / (1.0 - b2p)
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

def _lp_polish(x, deadline, r0=2.5e-3, cap=4000):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    iuI, iuJ = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iuI, iuJ]
        m = float(c.max())
        margin = max(20.0 * radius, 1.2e-4)
        sel = c > (m - margin)
        I = iuI[sel]; J = iuJ[sel]; csel = c[sel]
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
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, n * d + 1)).tocsr()
        except Exception:
            break
        obj = np.zeros(n * d + 1)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
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
    return x, best

def _greedy_sweep(x, best, rng, deadline, top_k=28, batch=2048):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    sigmas = (0.035, 0.012, 0.004)
    while time.time() < deadline:
        G = _gram(x)
        prow = G.max(axis=1)
        order = np.argsort(-prow)[:top_k]
        moved = False
        for ii in order:
            if time.time() >= deadline:
                break
            i = int(ii)
            cur = float(prow[i])
            if cur < best - 8e-4:
                continue
            Gm = G.copy()
            Gm[i, :] = -2.0
            Gm[:, i] = -2.0
            other_max = float(Gm.max())
            if other_max >= best - 1e-12 and cur <= other_max + 1e-12:
                # not a unique worst point; only try if we can beat best
                # need new_local < best and other_max < best (impossible if other_max==best)
                if other_max >= best - 1e-12:
                    # moving i alone cannot beat best unless other_max < best
                    # still try to reduce cur to allow future moves? skip for speed
                    # only try the very worst few
                    if cur < best - 1e-9:
                        continue
                    if other_max >= best - 1e-12 and cur >= best - 1e-12:
                        # multiple points tied at best; try anyway with one sigma
                        pass
                    else:
                        continue
            xi = x[i].copy()
            for sigma in sigmas:
                if time.time() >= deadline:
                    break
                R = rng.normal(size=(batch, d))
                coeff = R @ xi
                R -= coeff[:, None] * xi[None, :]
                nr = np.linalg.norm(R, axis=1, keepdims=True) + 1e-12
                R /= nr
                Y = xi[None, :] + sigma * R
                Y /= np.linalg.norm(Y, axis=1, keepdims=True)
                D = Y @ x.T
                D[:, i] = -2.0
                nl = D.max(axis=1)
                k = int(np.argmin(nl))
                nl_best = float(nl[k])
                cand = nl_best if nl_best > other_max else other_max
                if cand < best - 1e-13:
                    x[i] = Y[k]
                    best = cand
                    # refresh G for remaining indices in this sweep
                    # update row/col i efficiently
                    row = x @ x[i]
                    row[i] = -2.0
                    G[i, :] = row
                    G[:, i] = row
                    prow[i] = float(row.max())
                    # also prow of others may change; recompute prow cheaply
                    # full recompute every success is safer but slower; do cheap update:
                    # recompute prow as max of G
                    prow = G.max(axis=1)
                    moved = True
                    break
            if moved:
                break
        if not moved:
            break
    return x, best

def _void_sample(x, idx_list, rng, n_samp=30000):
    n, d = x.shape
    Z = rng.normal(size=(n_samp, d))
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    for r in idx_list:
        S = Z @ x.T
        S[:, int(r)] = -2.0
        m2 = S.max(axis=1)
        pick = int(np.argmin(m2))
        x[int(r)] = Z[pick]
    return x

def solve():
    import time as _time
    import numpy as _np
    t0 = _time.time()
    T_END = t0 + 174.0
    ANNEAL_END = t0 + 132.0
    LP_END = t0 + 163.0
    rng = _np.random.default_rng(7)
    x_best = None
    best = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = _np.load(fn)
            a = _np.asarray(a, dtype=float)
            if a.shape == (256, 6) and _np.all(_np.isfinite(a)):
                a = a / _np.linalg.norm(a, axis=1, keepdims=True)
                vv = _max_cos(a)
                if vv < best:
                    best = vv
                    x_best = a.copy()
        except Exception:
            continue
    if x_best is None:
        x_best = rng.normal(size=(256, 6))
        x_best /= _np.linalg.norm(x_best, axis=1, keepdims=True)
        best = _max_cos(x_best)
    n, d = 256, 6
    # quick initial squeeze
    try:
        if _time.time() < ANNEAL_END:
            x_best, best = _greedy_sweep(x_best, best, rng, min(_time.time() + 4.0, ANNEAL_END))
    except Exception:
        pass
    try:
        if _time.time() < ANNEAL_END:
            x_best, best = _lp_polish(x_best, min(_time.time() + 5.0, ANNEAL_END))
    except Exception:
        pass
    cycle = 0
    stall = 0
    while _time.time() < ANNEAL_END:
        cycle += 1
        rem = ANNEAL_END - _time.time()
        if rem < 2.0:
            break
        x = x_best.copy()
        try:
            G = x @ x.T
            _np.fill_diagonal(G, -2.0)
            prow = G.max(axis=1)
            order = _np.argsort(-prow)
            flat = int(_np.argmax(G))
            ai = flat // n; aj = flat % n
        except Exception:
            ai, aj, order = 0, 1, _np.arange(n)
        mode = cycle % 7
        try:
            if mode == 0:
                # worst-pair split: reinsert worst point to largest void then jitter partner
                worst = int(order[0])
                x = _void_sample(x, [worst], rng, n_samp=30000)
                # jitter partner neighbourhood
                dots = x @ x[worst]
                near = _np.argsort(-dots)[:6]
                x[near] += 0.06 * rng.normal(size=(len(near), d))
                x /= _np.linalg.norm(x, axis=1, keepdims=True)
                steps, K0, K1, lr = 650, 60.0, 1600.0, 0.010
            elif mode == 1:
                # double void reinsert
                idx = [int(order[0]), int(order[1])]
                x = _void_sample(x, idx, rng, n_samp=25000)
                sel = _np.argsort(-prow)[:8]
                x[sel] += 0.04 * rng.normal(size=(len(sel), d))
                x /= _np.linalg.norm(x, axis=1, keepdims=True)
                steps, K0, K1, lr = 700, 50.0, 1500.0, 0.010
            elif mode == 2:
                # coherent cap shift around worst point
                c = x[int(order[0])]
                dots = x @ c
                cap = _np.where(dots > 0.35)[0]
                if len(cap) < 4:
                    cap = _np.argsort(-dots)[:10]
                v = rng.normal(size=d); v /= (_np.linalg.norm(v) + 1e-12)
                sig = 0.18
                for ii in cap:
                    xi = x[int(ii)]
                    vt = v - float(v @ xi) * xi
                    x[int(ii)] = xi + sig * vt
                x /= _np.linalg.norm(x, axis=1, keepdims=True)
                # add small individual noise
                x[cap] += 0.02 * rng.normal(size=(len(cap), d))
                x /= _np.linalg.norm(x, axis=1, keepdims=True)
                steps, K0, K1, lr = 750, 35.0, 1100.0, 0.011
            elif mode == 3:
                # patch shatter around random center
                c0 = int(rng.integers(n))
                dots = x @ x[c0]
                nearest = _np.argsort(-dots)[:14]
                x[nearest] += 0.38 * rng.normal(size=(len(nearest), d))
                x /= _np.linalg.norm(x, axis=1, keepdims=True)
                steps, K0, K1, lr = 800, 30.0, 1000.0, 0.012
            elif mode == 4:
                x += 0.035 * rng.normal(size=x.shape)
                x /= _np.linalg.norm(x, axis=1, keepdims=True)
                steps, K0, K1, lr = 600, 80.0, 2200.0, 0.008
            elif mode == 5:
                pick = rng.choice(n, size=20, replace=False)
                x[pick] += 0.32 * rng.normal(size=(20, d))
                x /= _np.linalg.norm(x, axis=1, keepdims=True)
                steps, K0, K1, lr = 700, 40.0, 1300.0, 0.011
            else:
                # attack both endpoints of worst pair with orthogonal kicks
                for tt in (ai, aj):
                    pert = rng.normal(size=d)
                    pert = pert - float(pert @ x[tt]) * x[tt]
                    pert /= (_np.linalg.norm(pert) + 1e-12)
                    x[tt] += 0.10 * pert
                x /= _np.linalg.norm(x, axis=1, keepdims=True)
                x += 0.08 * rng.normal(size=x.shape)
                x /= _np.linalg.norm(x, axis=1, keepdims=True)
                steps, K0, K1, lr = 850, 25.0, 900.0, 0.013
            if rem < 25:
                steps = min(steps, 400)
            xb, vb = _adam_sm(x, steps, K0, K1, lr)
            # short coordinate sweep to snap to local minimax
            xb, vb = _greedy_sweep(xb, vb, rng, _time.time() + 0.9, top_k=24, batch=1536)
            if vb < best - 1e-12 and _np.all(_np.isfinite(xb)):
                x_best = xb
                best = vb
                stall = 0
                # opportunistic LP when we improve
                if cycle % 3 == 0 and _time.time() < ANNEAL_END - 6:
                    try:
                        xl, vl = _lp_polish(x_best, min(_time.time() + 2.5, ANNEAL_END))
                        xl, vl = _greedy_sweep(xl, vl, rng, min(_time.time() + 0.7, ANNEAL_END), top_k=20, batch=1024)
                        if vl < best - 1e-13:
                            x_best, best = xl, vl
                    except Exception:
                        pass
            else:
                stall += 1
                # occasional LP on stall to re-equalize and try different topology via larger radius
                if stall % 12 == 0 and _time.time() < ANNEAL_END - 6:
                    try:
                        xl, vl = _lp_polish(x_best, min(_time.time() + 2.0, ANNEAL_END), r0=4e-3)
                        if vl < best - 1e-13:
                            x_best, best = xl, vl
                            stall = 0
                    except Exception:
                        pass
        except Exception:
            continue
    # final polish: alternate LP and greedy
    while _time.time() < LP_END:
        try:
            xl, vl = _lp_polish(x_best, min(_time.time() + 4.0, LP_END))
            if vl < best:
                x_best, best = xl, vl
            else:
                break
            xg, vg = _greedy_sweep(x_best, best, rng, min(_time.time() + 2.0, LP_END), top_k=30, batch=2048)
            if vg < best - 1e-14:
                x_best, best = xg, vg
            else:
                if vl >= best - 1e-14:
                    break
        except Exception:
            break
    try:
        if _time.time() < T_END:
            x_best, best = _greedy_sweep(x_best, best, rng, T_END, top_k=32, batch=2048)
    except Exception:
        pass
    x_best = x_best / _np.linalg.norm(x_best, axis=1, keepdims=True)
    x_best = _np.where(_np.isfinite(x_best), x_best, 0.0)
    nr = _np.linalg.norm(x_best, axis=1, keepdims=True)
    nr = _np.where(nr < 1e-12, 1.0, nr)
    x_best = x_best / nr
    return x_best
