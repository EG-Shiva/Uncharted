import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def adam_quench(x, steps, b0, b1, lr0, lr1):
    n, d = x.shape
    m1 = np.zeros_like(x)
    m2 = np.zeros_like(x)
    b_1 = 0.9
    b_2 = 0.999
    eps = 1e-8
    lb0, lb1 = np.log(b0), np.log(b1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        f = s / max(1, steps - 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        # weights, G-gmax <= 0
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        sW = float(W.sum())
        if not np.isfinite(sW) or sW == 0:
            break
        W /= sW
        Grad = W @ x
        dot = np.sum(Grad * x, axis=1, keepdims=True)
        Gt = Grad - dot * x
        m1 = b_1 * m1 + (1 - b_1) * Gt
        m2 = b_2 * m2 + (1 - b_2) * (Gt * Gt)
        c1 = 1.0 - b_1 ** (s + 1)
        c2 = 1.0 - b_2 ** (s + 1)
        m1h = m1 / c1
        m2h = m2 / c2
        step = lr * m1h / (np.sqrt(m2h) + eps)
        sdot = np.sum(step * x, axis=1, keepdims=True)
        step = step - sdot * x
        # clip extreme step
        sn = np.linalg.norm(step, axis=1, keepdims=True)
        # limit to 0.05 per step
        scale = np.ones_like(sn)
        over = sn > 0.05
        scale[over] = 0.05 / (sn[over] + 1e-12)
        step = step * scale
        x = x - step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x = x / nr
    return x

def lp_polish(x, seconds, t_deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 5e-4
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while True:
        if time.time() - start >= seconds:
            break
        if t_deadline is not None and time.time() >= t_deadline:
            break
        if radius < 1e-13:
            break
        G = x @ x.T
        c = G[iu0]
        m = float(c.max())
        window = max(20.0 * radius, 5e-5)
        # keep top pairs only to bound LP size
        order = np.argsort(-c)
        # take up to 700 within window
        sel_order = []
        for idx in order:
            if c[idx] > m - window and len(sel_order) < 700:
                sel_order.append(idx)
            elif c[idx] <= m - window:
                break
            if len(sel_order) >= 700:
                break
        if len(sel_order) == 0:
            break
        sel = np.array(sel_order)
        I = iu0[0][sel]
        J = iu0[1][sel]
        p = len(sel)
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                          options={"time_limit": 5.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        y = y / nr
        v = max_cos(y)
        if v < best - 1e-15:
            x = y
            best = v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    TIME_LIMIT = 165.0  # leave margin under 180
    rng = np.random.default_rng(7)
    n, d = 58, 22
    # load start
    best = None
    best_val = 1e9
    for f in ("best.npy", "record.npy"):
        try:
            v = np.load(f).astype(float)
            if v.shape != (58, 22):
                continue
            v = v / np.linalg.norm(v, axis=1, keepdims=True)
            c = max_cos(v)
            if c < best_val:
                best_val = c
                best = v.copy()
        except Exception:
            continue
    if best is None:
        z = rng.standard_normal((n, d))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        best = z
        best_val = max_cos(best)

    # quick initial LP squeeze (2s) to get true local optimum value
    try:
        best = lp_polish(best, 2.0, t0 + TIME_LIMIT - 5)
        best_val = max_cos(best)
    except Exception:
        pass

    hop_end = t0 + 135.0
    # clamp hop_end to leave time for final polish
    if hop_end > t0 + TIME_LIMIT - 20:
        hop_end = t0 + TIME_LIMIT - 20

    it = 0
    # fast cycles
    while time.time() < hop_end:
        it += 1
        if time.time() - t0 > TIME_LIMIT - 25:
            break
        y = best.copy()
        mode = it % 5
        if mode == 0:
            sig = 0.004
            y = y + sig * rng.standard_normal(y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        elif mode == 1:
            sig = 0.008
            y = y + sig * rng.standard_normal(y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        elif mode == 2:
            k = 1 + (it % 2)
            idx = rng.choice(n, size=k, replace=False)
            y = y + 0.003 * rng.standard_normal(y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            r = rng.standard_normal((k, d))
            r /= np.linalg.norm(r, axis=1, keepdims=True)
            y[idx] = r
        elif mode == 3:
            # kick worst points
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            # worst pair
            ij = int(np.argmax(G))
            i0, j0 = ij // n, ij % n
            cand = [int(i0), int(j0)]
            # add top contacts of i0
            row = G[int(i0)].copy()
            top = np.argsort(-row)[:3]
            for t_ in top:
                if int(t_) not in cand:
                    cand.append(int(t_))
            cand = np.array(cand[:5])
            y[cand] = y[cand] + 0.03 * rng.standard_normal((len(cand), d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        else:
            k = 3
            idx = rng.choice(n, size=k, replace=False)
            r = rng.standard_normal((k, d))
            r /= np.linalg.norm(r, axis=1, keepdims=True)
            y[idx] = r
            y /= np.linalg.norm(y, axis=1, keepdims=True)

        # quench: choose schedule by mode
        try:
            if mode in (0, 1):
                y = adam_quench(y, 900, 80, 600, 0.004, 0.0006)
                y = adam_quench(y, 300, 800, 2000, 0.001, 0.0003)
            elif mode == 2:
                y = adam_quench(y, 1100, 100, 800, 0.003, 0.0004)
                y = adam_quench(y, 300, 1000, 2500, 0.0008, 0.0002)
            elif mode == 3:
                y = adam_quench(y, 1000, 60, 500, 0.004, 0.0005)
                y = adam_quench(y, 300, 800, 2000, 0.001, 0.0003)
            else:
                y = adam_quench(y, 1200, 80, 700, 0.0035, 0.0004)
                y = adam_quench(y, 300, 1000, 2500, 0.0008, 0.0002)
        except Exception:
            continue
        v = max_cos(y)
        if v < best_val - 1e-12:
            # squeeze immediately with short LP
            try:
                y2 = lp_polish(y, 1.5, t0 + TIME_LIMIT - 5)
                v2 = max_cos(y2)
                if v2 < v:
                    y, v = y2, v2
            except Exception:
                pass
            best_val = v
            best = y.copy()
        if it > 200:
            break

    # final sharpen on best
    remain = (t0 + TIME_LIMIT) - time.time() - 1.0
    if remain > 4:
        lp_sec = min(remain - 1.0, 18.0)
        try:
            polished = lp_polish(best, lp_sec, t0 + TIME_LIMIT - 1.0)
            vp = max_cos(polished)
            if vp < best_val:
                best = polished
                best_val = vp
        except Exception:
            pass

    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    best = np.nan_to_num(best, nan=0.0, posinf=0.0, neginf=0.0)
    nr = np.linalg.norm(best, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    best = best / nr
    return best
