import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _row_max_and_G(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1), G

def _adam_softmax(x0, steps, K0, K1, lr):
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
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        if t % 200 == 0 or t == steps:
            val = _max_cos(x)
            if val < best_val:
                best_val = val
                best_x = x.copy()
    return best_x, best_val

def _lp_polish(x, deadline, r0=2e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    iu0 = np.triu_indices(n, 1)
    iuI, iuJ = iu0[0], iu0[1]
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iuI, iuJ]
        m = float(c.max())
        margin = max(15.0 * radius, 8e-5)
        sel = c > (m - margin)
        I = iuI[sel]; J = iuJ[sel]
        csel = c[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > 4000:
            part = np.argpartition(-csel, 4000)[:4000]
            I = I[part]; J = J[part]; csel = csel[part]
            p = 4000
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

def _greedy_point_sweep(x, rng, deadline, sigma0=0.02, top_k=24, batch=512):
    # coordinate descent: re-optimize worst points one at a time
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    while time.time() < deadline:
        prow, G = _row_max_and_G(x)
        order = np.argsort(-prow)[:top_k]
        any_imp = False
        for ii in order:
            if time.time() >= deadline:
                break
            i = int(ii)
            cur_row = float(prow[i])
            # max over pairs not involving i
            G2 = G.copy()
            G2[i, :] = -2.0
            G2[:, i] = -2.0
            other_max = float(G2.max())
            target = min(cur_row, best)
            # if this point is far below best, skip unless it is very loose (skip)
            if target < best - 1e-3 and cur_row < best - 1e-3:
                continue
            # build candidates: random tangent + repulsion-directed
            # repulsion direction from worst neighbours
            nbr = np.where(G[i] > target - 5e-3)[0]
            rep = np.zeros(d)
            if len(nbr) > 0:
                w = np.exp(800.0 * (G[i, nbr] - target))
                w /= w.sum()
                rep = w @ x[nbr]
                # project to tangent
                rep = rep - np.dot(rep, x[i]) * x[i]
                rn = np.linalg.norm(rep)
                if rn > 1e-12:
                    rep /= rn
                else:
                    rep = np.zeros(d)
            # batch random
            R = rng.normal(size=(batch, d))
            R -= np.outer(R @ x[i], x[i])
            rn = np.linalg.norm(R, axis=1, keepdims=True) + 1e-12
            R /= rn
            # mix scales
            scales = np.empty(batch)
            # half at sigma0, quarter smaller, quarter larger
            scales[:batch//2] = sigma0
            scales[batch//2:3*batch//4] = sigma0 * 0.25
            scales[3*batch//4:] = sigma0 * 2.5
            # deterministic repulsion candidates overwrite first few rows
            if np.linalg.norm(rep) > 1e-12:
                for k, sc in enumerate([sigma0*0.5, sigma0, sigma0*2.0, sigma0*4.0]):
                    if k < batch:
                        R[k] = rep
                        scales[k] = sc
            Y = x[i][None, :] + R * scales[:, None]
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            D = Y @ x.T
            D[:, i] = -2.0
            new_rows = D.max(axis=1)
            new_globals = np.maximum(new_rows, other_max)
            kbest = int(np.argmin(new_globals))
            if new_globals[kbest] < best - 1e-12:
                x[i] = Y[kbest]
                # update G row/col quickly (recompute fully for simplicity)
                best = float(new_globals[kbest])
                _, G = _row_max_and_G(x)
                prow, _ = _row_max_and_G(x)
                any_imp = True
            elif new_rows[kbest] < cur_row - 1e-9 and new_globals[kbest] <= best + 1e-12:
                # slack-creating move: improves row without hurting global
                x[i] = Y[kbest]
                _, G = _row_max_and_G(x)
                prow, _ = _row_max_and_G(x)
                any_imp = True
        if not any_imp:
            # decay sigma and try again; if sigma tiny, break
            sigma0 *= 0.5
            if sigma0 < 2e-4:
                break
    return x

def solve():
    import time as _time
    import numpy as _np
    t0 = _time.time()
    T_END = t0 + 172.0
    GREEDY_END = t0 + 55.0
    ANNEAL_END = t0 + 135.0
    rng = _np.random.default_rng(1)
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
    # Phase 1: greedy coordinate descent on loaded best (attacks isolated pair)
    try:
        x_best = _greedy_point_sweep(x_best, rng, min(GREEDY_END, T_END - 100.0), sigma0=0.015, top_k=32, batch=512)
        best = _max_cos(x_best)
    except Exception:
        pass
    try:
        x_best = _lp_polish(x_best, min(_time.time() + 8.0, ANNEAL_END), r0=1.5e-3)
        best = _max_cos(x_best)
    except Exception:
        pass
    # Phase 2: alternate small global kicks + Adam + greedy
    cycle = 0
    while _time.time() < ANNEAL_END:
        cycle += 1
        rem = ANNEAL_END - _time.time()
        if rem < 4:
            break
        x = x_best.copy()
        n, d = x.shape
        G = x @ x.T
        _np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        prow = G.max(axis=1)
        cnt = _np.sum(G > gmax - 3e-4, axis=1)
        order = _np.argsort(-cnt)
        # kick sizes cycling
        if cycle % 5 == 0:
            sigma, steps, K0, K1, lr = 0.10, 550, 40.0, 900.0, 0.011
        elif cycle % 2 == 0:
            sigma, steps, K0, K1, lr = 0.025, 450, 100.0, 2000.0, 0.007
        else:
            sigma, steps, K0, K1, lr = 0.008, 400, 150.0, 2500.0, 0.006
        if rem < 25:
            steps = min(steps, 300)
        over = order[:8]
        x[over] += sigma * rng.normal(size=(8, 6))
        rnd = rng.choice(n, size=4, replace=False)
        x[rnd] += (sigma * 0.5) * rng.normal(size=(4, 6))
        x /= _np.linalg.norm(x, axis=1, keepdims=True)
        try:
            xb, vb = _adam_softmax(x, steps, K0, K1, lr)
        except Exception:
            continue
        # greedy refine of Adam result (short)
        try:
            xb = _greedy_point_sweep(xb, rng, _time.time() + 2.5, sigma0=0.008, top_k=16, batch=256)
            vb = _max_cos(xb)
        except Exception:
            vb = _max_cos(xb)
        if vb < best - 1e-12:
            x_best = xb
            best = vb
    # Final polish: fine Adam + greedy + LP
    try:
        if _time.time() < T_END - 8:
            xb, vb = _adam_softmax(x_best, 500, 400.0, 3000.0, 0.004)
            if vb < best:
                x_best, best = xb, vb
    except Exception:
        pass
    try:
        if _time.time() < T_END - 5:
            x_best = _greedy_point_sweep(x_best, rng, T_END - 4.0, sigma0=0.004, top_k=24, batch=512)
    except Exception:
        pass
    try:
        if _time.time() < T_END - 2:
            x_best = _lp_polish(x_best, T_END - 1.0, r0=8e-4)
    except Exception:
        pass
    x_best = x_best / _np.linalg.norm(x_best, axis=1, keepdims=True)
    x_best = _np.where(_np.isfinite(x_best), x_best, 0.0)
    nr = _np.linalg.norm(x_best, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    x_best = x_best / nr
    return x_best
