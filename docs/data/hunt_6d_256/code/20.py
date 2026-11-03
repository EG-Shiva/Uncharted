import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

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
        mhat = m / (1 - b1p)
        vhat = v / (1 - b2p)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        sdot = np.sum(step * x, axis=1, keepdims=True)
        step = step - sdot * x
        x = x - step
        nrm = np.linalg.norm(x, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        x = x / nrm
        if t % 150 == 0 or t == steps:
            val = _max_cos(x)
            if val < best_val:
                best_val = val
                best_x = x.copy()
    return best_x, best_val

def _lp_polish(x, deadline, r0=3e-3, cap=5000, mfac=25.0, base=3e-4):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    iu0 = np.triu_indices(n, 1)
    iuI = iu0[0]
    iuJ = iu0[1]
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iuI, iuJ]
        m = float(c.max())
        margin = max(mfac * radius, base)
        sel = c > (m - margin)
        I = iuI[sel]
        J = iuJ[sel]
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
    return x

def _micro_hill(x, deadline, rng, step0=2e-3, trials=250):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    step = step0
    while time.time() < deadline:
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        flat = int(np.argmax(G))
        ai = flat // n
        aj = flat % n
        improved = False
        for target in (ai, aj):
            if time.time() >= deadline:
                break
            # other max not involving target
            # compute quickly
            # mask: max of G excluding row/col target
            # use copy (256x256 cheap)
            Gm = G.copy()
            Gm[target, :] = -2.0
            Gm[:, target] = -2.0
            other_max = float(Gm.max())
            if other_max >= best - 1e-15:
                # moving target alone cannot beat best if other_max already >= best
                # but if both ai,aj pairs share, other_max may still be best; skip target with larger other?
                # still try if other_max < best + 1e-12 and step small? skip to save time
                if other_max >= best:
                    continue
            xt = x[target]
            # random tangent directions
            R = rng.normal(size=(trials, d))
            R -= (R @ xt)[:, None] * xt[None, :]
            nr = np.linalg.norm(R, axis=1, keepdims=True)
            nr[nr == 0] = 1.0
            R /= nr
            best_local = best
            best_cand = None
            # batch evaluate with vectorization in chunks
            # candidate = norm(xt + step*R)
            C = xt[None, :] + step * R
            C /= np.linalg.norm(C, axis=1, keepdims=True)
            D = C @ x.T  # trials x n
            # ignore self column
            D[:, target] = -2.0
            lm = D.max(axis=1)
            k = int(np.argmin(lm))
            if float(lm[k]) < best - 1e-14 and float(lm[k]) < other_max or float(lm[k]) < best - 1e-14 and float(max(float(lm[k]), other_max)) < best - 1e-14:
                # accept
                x[target] = C[k]
                best = float(max(float(lm[k]), other_max))
                improved = True
                break
        if improved:
            # keep step
            continue
        else:
            step *= 0.6
            if step < 1e-7:
                # restart with larger trials on worst only or break
                step = step0 * 0.5
                # if no progress for a while, do a tiny random walk on worst point to escape flat?
                # just continue a bit
                # check deadline loop; if step cycled, break to avoid waste
                # do one exploratory move that doesn't worsen beyond best+1e-6?
                pass
            # if step very small and no improvement, we can exit early
            # use stagnation detection via G max unchanged
            # simple: if step < 5e-6 after decay, break
            if step < 3e-6 and not improved:
                # try to shrink further once
                # break after a few decays
                # count decays
                pass
        # termination: if time low or step tiny after many loops, break
        # we limit iterations by checking improvement of best
        # quick exit if step < 1e-6 and no improvement in last pass
        if step < 1e-6:
            break
    return x

def _rand_orth_close(rng, d, t):
    M = np.eye(d) + t * rng.normal(size=(d, d))
    Q, R = np.linalg.qr(M)
    # fix signs to keep close to I
    s = np.sign(np.diag(R))
    s[s == 0] = 1.0
    Q = Q * s[None, :]
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q

def solve():
    import time as _time
    import numpy as _np
    t0 = _time.time()
    T_END = t0 + 172.0
    ANNEAL_END = t0 + 132.0
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
    n, d = 256, 6
    if x_best is None:
        x_best = rng.normal(size=(n, d))
        x_best /= _np.linalg.norm(x_best, axis=1, keepdims=True)
        best = _max_cos(x_best)
    try:
        if _time.time() < ANNEAL_END:
            x_best = _lp_polish(x_best, min(_time.time() + 6.0, ANNEAL_END), r0=2e-3)
            best = _max_cos(x_best)
    except Exception:
        pass
    try:
        if _time.time() < ANNEAL_END:
            x_best = _micro_hill(x_best, min(_time.time() + 3.0, ANNEAL_END), rng)
            best = _max_cos(x_best)
    except Exception:
        pass

    cycle = 0
    stall = 0
    while _time.time() < ANNEAL_END:
        cycle += 1
        rem = ANNEAL_END - _time.time()
        if rem < 3.0:
            break
        x = x_best.copy()
        G = x @ x.T
        _np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        flat = int(_np.argmax(G))
        ai = flat // n
        aj = flat % n
        prow = G.max(axis=1)
        cnt = _np.sum(G > gmax - 3e-4, axis=1)
        order_cnt = _np.argsort(-cnt)
        order_row = _np.argsort(-prow)
        mode = cycle % 6
        if mode == 0:
            # worst-pair split + jitter overloaded patch
            eps = 0.03 + 0.02 * rng.random()
            x[ai] = x[ai] - eps * x[aj]
            x[aj] = x[aj] - eps * x[ai]
            # orthogonal jitter
            x[ai] += 0.01 * rng.normal(size=(6,))
            x[aj] += 0.01 * rng.normal(size=(6,))
            # jitter top overloaded
            over = order_row[:6]
            x[over] += 0.02 * rng.normal(size=(6, 6))
            x /= _np.linalg.norm(x, axis=1, keepdims=True)
            steps, a0, a1, lr = (550, 80.0, 2200.0, 0.008)
        elif mode == 1:
            # coherent cap rotation around worst midpoint
            c = x[ai] + x[aj]
            c /= _np.linalg.norm(c)
            dots = x @ c
            thr = 0.60 + 0.15 * rng.random()
            idx = _np.where(dots > thr)[0]
            if idx.size < 2 or idx.size > 200:
                idx = _np.argsort(-dots)[:30]
            t_ang = 0.04 + 0.08 * rng.random()
            Q = _rand_orth_close(rng, d, t_ang)
            # apply same Q to cap (row vectors)
            x[idx] = (x[idx] @ Q.T)
            x[idx] /= _np.linalg.norm(x[idx], axis=1, keepdims=True)
            # small jitter outside to diversify
            rnd = rng.choice(n, size=4, replace=False)
            x[rnd] += 0.015 * rng.normal(size=(4, 6))
            x /= _np.linalg.norm(x, axis=1, keepdims=True)
            steps, a0, a1, lr = (550, 100.0, 2500.0, 0.007)
        elif mode == 2:
            # void reinsertion for worst points
            krep = 4
            worst = order_row[:krep]
            for r in worst:
                Z = rng.normal(size=(6000, 6))
                Z /= _np.linalg.norm(Z, axis=1, keepdims=True)
                S = Z @ x.T
                S[:, r] = -2.0
                mm = S.max(axis=1)
                k2 = int(_np.argmin(mm))
                if mm[k2] < prow[r] - 5e-5:
                    x[r] = Z[k2]
            # light jitter on neighbours of worst
            nb = _np.argsort(-G[ai])[:8]
            x[nb] += 0.01 * rng.normal(size=(8, 6))
            x /= _np.linalg.norm(x, axis=1, keepdims=True)
            steps, a0, a1, lr = (500, 60.0, 1800.0, 0.009)
        elif mode == 3:
            # patch shatter around worst point
            nb = _np.argsort(-G[ai])[:12]
            x[nb] += 0.12 * rng.normal(size=(12, 6))
            rnd = rng.choice(n, size=4, replace=False)
            x[rnd] += 0.05 * rng.normal(size=(4, 6))
            x /= _np.linalg.norm(x, axis=1, keepdims=True)
            steps, a0, a1, lr = (650, 40.0, 1200.0, 0.011)
        elif mode == 4:
            # global moderate kick on overloaded + random
            over = order_cnt[:10]
            x[over] += 0.05 * rng.normal(size=(10, 6))
            rnd = rng.choice(n, size=8, replace=False)
            x[rnd] += 0.03 * rng.normal(size=(8, 6))
            x /= _np.linalg.norm(x, axis=1, keepdims=True)
            steps, a0, a1, lr = (500, 70.0, 2000.0, 0.008)
        else:
            # tiny fine kick
            over = order_row[:8]
            x[over] += 0.012 * rng.normal(size=(8, 6))
            x /= _np.linalg.norm(x, axis=1, keepdims=True)
            steps, a0, a1, lr = (450, 150.0, 3000.0, 0.005)
        if rem < 25:
            steps = min(steps, 350)
        try:
            xb, vb = _adam_sm(x, steps, a0, a1, lr)
        except Exception:
            stall += 1
            continue
        if vb < best - 1e-12:
            x_best = xb
            best = vb
            stall = 0
            # quick LP squeeze + micro on improvement
            try:
                if _time.time() < ANNEAL_END - 4:
                    x_best = _lp_polish(x_best, min(_time.time() + 2.5, ANNEAL_END), r0=2e-3)
                    best = _max_cos(x_best)
            except Exception:
                pass
            try:
                if _time.time() < ANNEAL_END - 2:
                    x_best = _micro_hill(x_best, min(_time.time() + 1.5, ANNEAL_END), rng)
                    best = _max_cos(x_best)
            except Exception:
                pass
        else:
            stall += 1
        if stall >= 8 and _time.time() < ANNEAL_END - 8:
            # big diversification
            x2 = x_best.copy()
            pick = rng.choice(n, size=20, replace=False)
            x2[pick] += 0.35 * rng.normal(size=(20, 6))
            x2 /= _np.linalg.norm(x2, axis=1, keepdims=True)
            try:
                xb2, vb2 = _adam_sm(x2, 400, 40.0, 1200.0, 0.012)
                if vb2 < best - 1e-12:
                    x_best = xb2
                    best = vb2
                    stall = 0
                else:
                    stall = 0  # reset anyway to vary modes
            except Exception:
                pass
    # final intensive polish
    try:
        if _time.time() < T_END - 4:
            xb, vb = _adam_sm(x_best, 600, 400.0, 3500.0, 0.004)
            if vb < best:
                x_best, best = xb, vb
    except Exception:
        pass
    try:
        if _time.time() < T_END - 3:
            x_best = _micro_hill(x_best, min(_time.time() + 6.0, T_END - 2.0), rng, step0=1.5e-3, trials=400)
            best = _max_cos(x_best)
    except Exception:
        pass
    try:
        if _time.time() < T_END - 1.5:
            x_best = _lp_polish(x_best, T_END - 0.8, r0=1e-3, cap=6000, mfac=20.0, base=1e-4)
    except Exception:
        pass
    x_best = x_best / _np.linalg.norm(x_best, axis=1, keepdims=True)
    x_best = _np.where(_np.isfinite(x_best), x_best, 0.0)
    nr = _np.linalg.norm(x_best, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    x_best = x_best / nr
    return x_best
