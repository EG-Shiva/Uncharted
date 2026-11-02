import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _smooth_descent(x0, steps, T0, T1, lr0, mu, rng):
    x = x0.copy()
    n, d = x.shape
    v = np.zeros_like(x)
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    best_val = float(G.max())
    best = x.copy()
    if T0 > 0 and T1 > 0:
        log_ratio = np.log(T1 / T0)
    else:
        log_ratio = 0.0
    for k in range(steps):
        frac = k / max(1, steps - 1)
        T = T0 * np.exp(log_ratio * frac)
        if T < 1e-12:
            T = 1e-12
        lr = lr0 * (1.0 - 0.55 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        S = E.sum() * 0.5
        if S < 1e-300:
            S = 1e-300
        grad = (E @ x) / S
        dot = (grad * x).sum(axis=1, keepdims=True)
        grad -= dot * x
        v = mu * v + (1.0 - mu) * grad
        x = x - lr * v
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        if m < best_val:
            best_val = m
            best = x.copy()
    return best, best_val

def _refine_hole(Xf, h, steps=160, lr0=0.08, T=0.006):
    h = h / max(1e-12, np.linalg.norm(h))
    lr = lr0
    for _ in range(steps):
        d = Xf @ h
        m = float(d.max())
        w = np.exp((d - m) / T)
        g = w @ Xf
        g = g - np.dot(g, h) * h
        ng = np.linalg.norm(g)
        if ng < 1e-12:
            break
        g /= ng
        hn = h - lr * g
        hn /= max(1e-12, np.linalg.norm(hn))
        # accept if improves max dot
        vn = float((Xf @ hn).max())
        vo = float((Xf @ h).max())
        if vn < vo:
            h = hn
        else:
            lr *= 0.9
            if lr < 1e-4:
                break
            continue
        lr *= 0.995
    return h

def _find_hole(cur_set, rng, n_cand=8000):
    R = rng.randn(n_cand, 4)
    R /= np.maximum(np.linalg.norm(R, axis=1, keepdims=True), 1e-12)
    D = R @ cur_set.T
    vals = D.max(axis=1)
    i = int(np.argmin(vals))
    h = R[i].copy()
    h = _refine_hole(cur_set, h)
    v = float((cur_set @ h).max())
    return h, v

def _lp_polish(x, seconds, t_deadline=None):
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    n, d = x.shape
    best = _max_cos(x)
    bx = x.copy()
    radius = 1.2e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if t_deadline is not None and time.time() > t_deadline:
            break
        if radius < 1e-14:
            break
        G = x @ x.T
        cfull = G[iu]
        m = float(cfull.max())
        sel = cfull > m - max(20 * radius, 1e-9)
        if int(sel.sum()) > 3000:
            idx = np.argsort(-cfull)
            sel = np.zeros_like(sel, dtype=bool)
            sel[idx[:3000]] = True
        I = iu[0][sel]; J = iu[1][sel]
        c = cfull[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        tmp = np.empty((p, 2 * d + 1), dtype=int)
        tmp[:, :d] = I[:, None] * d + np.arange(d)
        tmp[:, d:2 * d] = J[:, None] * d + np.arange(d)
        tmp[:, 2 * d] = n * d
        cols = tmp.ravel()
        cc = np.empty((p, 2 * d + 1), dtype=float)
        cc[:, :d] = x[J]
        cc[:, d:2 * d] = x[I]
        cc[:, 2 * d] = -1.0
        vals = cc.ravel()
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
        y = x + res.x[:-1].reshape(n, d)
        y /= np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
        v = _max_cos(y)
        if v < best:
            x, best, bx = y, v, y.copy()
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return bx, best

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    SEARCH_END = t0 + 138.0
    rng = np.random.RandomState(12345)
    n = 163
    best_x = None
    best_val = 2.0
    for fn in ("best.npy", "record.npy"):
        try:
            v = np.load(fn).astype(float)
            if v.shape == (n, 4) and np.all(np.isfinite(v)):
                v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
                c = _max_cos(v)
                if c < best_val:
                    best_val = c
                    best_x = v.copy()
        except Exception:
            pass
    if best_x is None:
        best_x = rng.randn(n, 4)
        best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
        best_val = _max_cos(best_x)

    trial = 0
    stag = 0
    while time.time() < SEARCH_END:
        if time.time() > DEADLINE - 35:
            break
        trial += 1
        # occasional fresh random basin
        if trial % 27 == 0:
            cur = rng.randn(n, 4)
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            cur_b, cur_v = _smooth_descent(cur, steps=3500, T0=0.020, T1=0.0006, lr0=0.050, mu=0.85, rng=rng)
            if cur_v < best_val - 1e-12:
                best_x, best_val = cur_b, cur_v
                stag = 0
            else:
                stag += 1
            continue
        # ruin size adaptive
        if stag < 10:
            k = int(rng.randint(6, 13))
        elif stag < 25:
            k = int(rng.randint(10, 20))
        else:
            k = int(rng.randint(15, 28))
        r = rng.rand()
        if r < 0.35:
            G = best_x @ best_x.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            deg = (G > m - 0.003).sum(axis=1)
            idx = np.argsort(deg)[-k:]
        elif r < 0.65:
            u = rng.randn(4); u /= np.linalg.norm(u)
            dots = best_x @ u
            idx = np.argsort(dots)[-k:]
        elif r < 0.85:
            c0 = int(rng.randint(n))
            dots = best_x @ best_x[c0]
            idx = np.argsort(dots)[-k:]
        else:
            idx = rng.choice(n, k, replace=False)
        idx = np.array(idx, dtype=int)
        rng.shuffle(idx)
        keep = np.ones(n, dtype=bool)
        keep[idx] = False
        cur_set = best_x[keep].copy()
        cur = best_x.copy()
        # sequential deep-hole reinsertion
        for w in idx:
            h, _ = _find_hole(cur_set, rng, n_cand=7000)
            cur[w] = h
            cur_set = np.vstack([cur_set, h.reshape(1, 4)])
        # tiny jitter on survivors to break symmetry
        jit = rng.randn(n, 4) * 0.002
        # don't jitter too much the reinserted? jitter all slightly
        cur = cur + jit
        cur /= np.maximum(np.linalg.norm(cur, axis=1, keepdims=True), 1e-12)
        steps = 3000 if k <= 12 else 3800
        T0 = 0.013 + 0.004 * rng.rand()
        cur_b, cur_v = _smooth_descent(cur, steps=steps, T0=T0, T1=0.0005, lr0=0.042, mu=0.86, rng=rng)
        if cur_v < best_val - 1e-12:
            if cur_v < best_val - 2.5e-5 and time.time() < DEADLINE - 42:
                try:
                    pol, pv = _lp_polish(cur_b, seconds=2.0, t_deadline=DEADLINE - 32)
                except Exception:
                    pol, pv = cur_b, cur_v
                if pv < best_val:
                    best_x, best_val = pol, pv
                elif cur_v < best_val:
                    best_x, best_val = cur_b, cur_v
            else:
                best_x, best_val = cur_b, cur_v
            stag = 0
            if best_val < 0.84:
                break
        else:
            stag += 1

    remain = DEADLINE - time.time() - 1.5
    if remain > 3:
        try:
            px, pv = _lp_polish(best_x, seconds=min(remain, 28.0), t_deadline=DEADLINE - 1.0)
            if pv < best_val:
                best_x, best_val = px, pv
            else:
                # keep polished geometry even if tie (tighter contacts)
                if pv <= best_val + 1e-12:
                    best_x = px
        except Exception:
            pass
    best_x = best_x / np.maximum(np.linalg.norm(best_x, axis=1, keepdims=True), 1e-12)
    return best_x
