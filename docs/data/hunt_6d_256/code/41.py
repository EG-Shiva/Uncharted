import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _stats(x):
    n = x.shape[0]
    G = x @ x.T
    iu = np.triu_indices(n, 1)
    c = G[iu[0], iu[1]]
    cs = np.sort(c)[::-1]
    m = float(cs[0])
    c100 = float(cs[99]) if len(cs) > 99 else m
    c1000 = float(cs[999]) if len(cs) > 999 else m
    top = float(cs[:1500].mean())
    return m, c100, c1000, top, cs

def _adam_lse(x0, steps, K0, K1, lr, deadline=1e18, track=100):
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
        if t % 50 == 0 and time.time() > deadline:
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
        b1p *= b1; b2p *= b2
        m *= b1; m += (1 - b1) * gt
        v *= b2; v += (1 - b2) * (gt * gt)
        mh = m / (1 - b1p); vh = v / (1 - b2p)
        step = lr * mh / (np.sqrt(vh) + eps)
        sdot = np.sum(step * x, axis=1, keepdims=True)
        step = step - sdot * x
        x = x - step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        if t % track == 0 or t == steps:
            val = _max_cos(x)
            if val < best_val:
                best_val = val
                best_x = x.copy()
    return best_x, best_val

def _lp_polish(x, deadline, r0=3.0e-3, cap=2500):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    bx = x.copy()
    radius = r0
    iuI, iuJ = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 2e-13:
        G = x @ x.T
        c = G[iuI, iuJ]
        mv = float(c.max())
        margin = max(25.0 * radius, 1.2e-4)
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
        obj = np.zeros(n * d + 1); obj[-1] = 1.0
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
            x, best, bx = y, val, y.copy()
            radius = min(radius * 1.5, 0.015)
        else:
            radius *= 0.5
    return bx, best

def _void_greedy(cur_frozen, rng, n_cand=20000):
    d = cur_frozen.shape[1]
    Z = rng.normal(size=(n_cand, d))
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    if cur_frozen.shape[0] == 0:
        return Z[0]
    S = Z @ cur_frozen.T
    cm = S.max(axis=1)
    i = int(np.argmin(cm))
    return Z[i]

def _refine_void(y0, F, steps=60, K=1500.0, lr=0.03):
    y = y0 / np.linalg.norm(y0)
    if F.shape[0] == 0:
        return y
    m1 = np.zeros_like(y); v1 = np.zeros_like(y)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    best_y = y.copy()
    best_v = float((F @ y).max())
    for t in range(1, steps + 1):
        g = F @ y
        gmax = float(g.max())
        w = np.exp(K * (g - gmax))
        s = float(w.sum())
        if not np.isfinite(s) or s <= 0:
            break
        w /= s
        grad = w @ F
        gt = grad - float(grad @ y) * y
        b1p *= b1; b2p *= b2
        m1 *= b1; m1 += (1 - b1) * gt
        v1 *= b2; v1 += (1 - b2) * (gt * gt)
        mh = m1 / (1 - b1p); vh = v1 / (1 - b2p)
        step = lr * mh / (np.sqrt(vh) + eps)
        step = step - float(step @ y) * y
        y = y - step
        y /= np.linalg.norm(y)
        if t % 20 == 0 or t == steps:
            gv = float((F @ y).max())
            if gv < best_v:
                best_v = gv
                best_y = y.copy()
    return best_y

def solve():
    t0 = time.time()
    T_END = t0 + 172.0
    CENSUS_END = t0 + 55.0
    SURG_END = t0 + 150.0
    rng0 = np.random.default_rng(0)
    x_best = None; best = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (256, 6) and np.all(np.isfinite(a)):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                vv = _max_cos(a)
                if vv < best:
                    best = vv; x_best = a.copy()
        except Exception:
            continue
    if x_best is None:
        x_best = rng0.normal(size=(256, 6))
        x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
        best = _max_cos(x_best)
    # ---- de novo balanced census (moderate K, diverse) ----
    parents = [(best, x_best.copy())]
    b = 0
    while time.time() < CENSUS_END and b < 10:
        rng = np.random.default_rng(2000 + b)
        x = rng.normal(size=(256, 6))
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        lr = [0.012, 0.009, 0.015][b % 3]
        try:
            x, _ = _adam_lse(x, 350, 80.0, 900.0, lr, deadline=CENSUS_END, track=175)
            if time.time() < CENSUS_END:
                x, vv = _adam_lse(x, 250, 400.0, 2200.0, 0.007, deadline=CENSUS_END, track=125)
            else:
                vv = _max_cos(x)
        except Exception:
            continue
        parents.append((vv, x.copy()))
        if vv < best:
            best = vv; x_best = x.copy()
        b += 1
    parents.sort(key=lambda t: t[0])
    # walking state: start from 2nd best to force different basin sometimes
    cur = parents[0][1].copy()
    cur_val = parents[0][0]
    if len(parents) > 1 and time.time() < SURG_END:
        # keep diversity: occasionally start walk from a worse but distinct parent
        cur = parents[min(1, len(parents)-1)][1].copy()
        cur_val = parents[min(1, len(parents)-1)][0]
    rng = np.random.default_rng(999)
    it = 0
    n = 256
    while time.time() < SURG_END:
        it += 1
        # choose kick size: cycle 6..28
        k = int([8, 12, 16, 24, 10, 20][it % 6])
        try:
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            rowmax = G.max(axis=1)
            # participation in top 1500
            iuI, iuJ = np.triu_indices(n, 1)
            c = G[iuI, iuJ]
            thr = float(np.sort(c)[::-1][min(1500, len(c)-1)])
            part = np.zeros(n)
            selm = c >= thr - 1e-12
            np.add.at(part, iuI[selm], 1); np.add.at(part, iuJ[selm], 1)
            score = rowmax + 0.002 * (part / max(1, part.max()))
            # mostly worst, sometimes random to diversify
            if it % 4 == 0:
                worst = np.argsort(score)[::-1][:k]
                # replace half by random indices
                rnd = rng.choice(n, size=k // 2, replace=False)
                worst[:k//2] = rnd
            else:
                worst = np.argsort(score)[::-1][:k]
            keep = np.ones(n, dtype=bool); keep[worst] = False
            frozen = cur[keep]
            # greedy sequential refill into voids
            newpts = []
            F = frozen.copy()
            for j in range(k):
                y0 = _void_greedy(F, rng, n_cand=20000)
                y0 = _refine_void(y0, F, steps=50, K=1200.0, lr=0.03)
                newpts.append(y0)
                F = np.vstack([F, y0[None, :]])
            child = np.empty_like(cur)
            child[keep] = frozen
            child[~keep] = np.array(newpts)
            child /= np.linalg.norm(child, axis=1, keepdims=True)
            # joint re-optimisation: moderate K first (equalize plateau), then sharper
            child, _ = _adam_lse(child, 180, 250.0, 900.0, 0.009, deadline=SURG_END, track=180)
            if time.time() > SURG_END:
                vv = _max_cos(child)
                if vv < best:
                    best = vv; x_best = child.copy()
                cur, cur_val = child, vv
                break
            child, vv = _adam_lse(child, 180, 600.0, 2500.0, 0.006, deadline=SURG_END, track=180)
            m2, _, _, top2, _ = _stats(child)
            # accept if max improves best, else walk if plateau improves
            if vv < best - 1e-12:
                best = vv; x_best = child.copy()
                cur, cur_val = child.copy(), vv
                # quick LP to lock equalized gain
                if time.time() < SURG_END - 3:
                    try:
                        lp_x, lp_v = _lp_polish(child, min(time.time()+4.0, SURG_END), r0=2.0e-3, cap=2500)
                        if lp_v < best:
                            best = lp_v; x_best = lp_x.copy()
                            cur, cur_val = lp_x.copy(), lp_v
                    except Exception:
                        pass
            elif vv < cur_val or top2 < _stats(cur)[3]:
                # sideways basin walk
                cur, cur_val = child.copy(), vv
            else:
                # small random jitter of cur to avoid stagnation
                if it % 5 == 0:
                    jrng = rng.normal(size=cur.shape) * 0.008
                    cur = cur + jrng
                    cur /= np.linalg.norm(cur, axis=1, keepdims=True)
                    cur, cur_val = _adam_lse(cur, 120, 300.0, 1200.0, 0.007, deadline=SURG_END, track=120)
        except Exception:
            continue
    # ---- final polish of global best ----
    try:
        if time.time() < T_END - 2:
            x_best, bv = _adam_lse(x_best, 300, 500.0, 3000.0, 0.005, deadline=T_END-8, track=150)
            best = bv
    except Exception:
        pass
    try:
        if time.time() < T_END - 2:
            x_best, best = _lp_polish(x_best, T_END-0.5, r0=2.5e-3, cap=3000)
    except Exception:
        pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
