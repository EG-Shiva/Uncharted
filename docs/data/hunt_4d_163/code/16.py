import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds, t_end=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while True:
        if time.time() - start >= seconds:
            break
        if t_end is not None and time.time() >= t_end:
            break
        if radius < 1e-14:
            break
        G = x @ x.T
        cf = G[iu]
        m = float(cf.max())
        sel = cf > m - max(20.0 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        c = cf[sel]
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
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def _softmax_desc(x0, steps, T0, T1, lr0, mu, rng, noise0=0.0, t_end=None):
    x = x0.copy()
    n, d = x.shape
    v = np.zeros_like(x)
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    best_val = float(G.max())
    best = x.copy()
    lratio = np.log(T1 / T0) if T0 > 0 and T1 > 0 else 0.0
    for k in range(steps):
        if t_end is not None and (k % 400 == 0) and time.time() > t_end:
            break
        frac = k / max(1, steps - 1)
        T = T0 * np.exp(lratio * frac)
        if T < 1e-12:
            T = 1e-12
        lr = lr0 * (1.0 - 0.6 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if S < 1e-300:
            S = 1e-300
        grad = (E @ x) / S
        grad -= (grad * x).sum(axis=1, keepdims=True) * x
        v = mu * v + (1.0 - mu) * grad
        x = x - lr * v
        if noise0 > 0:
            amp = noise0 * (1.0 - frac)
            if amp > 0:
                nz = rng.randn(n, d) * amp
                nz -= (nz * x).sum(axis=1, keepdims=True) * x
                x = x + nz
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        if m < best_val:
            best_val = m
            best = x.copy()
    return best, best_val

def _riesz_desc(x0, s0, s1, steps, lr0, mu, t_end=None):
    x = x0.copy()
    n, d = x.shape
    vel = np.zeros_like(x)
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    best_val = float(G.max())
    best = x.copy()
    for k in range(steps):
        if t_end is not None and (k % 400 == 0) and time.time() > t_end:
            break
        frac = k / max(1, steps - 1)
        s = s0 * (s1 / s0) ** frac if s0 > 0 and s1 > 0 else s0
        lr = lr0 * (1.0 - 0.6 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        D = 2.0 - 2.0 * G
        np.fill_diagonal(D, 1.0)
        np.maximum(D, 1e-12, out=D)
        # min off-diag for normalization
        np.fill_diagonal(D, 1e100)
        dmin = float(D.min())
        np.fill_diagonal(D, 1.0)
        pw = s / 2.0 + 1.0
        # W = (dmin/D)^pw in stable range
        with np.errstate(over='ignore', under='ignore'):
            W = np.exp(pw * (np.log(dmin) - np.log(D)))
        np.fill_diagonal(W, 0.0)
        Gx = W @ x
        grad = Gx - (Gx * x).sum(axis=1, keepdims=True) * x
        gn = float(np.sqrt((grad * grad).sum() / n))
        if gn > 1e-12:
            grad /= max(1.0, gn * 2.0)
        vel = mu * vel + (1.0 - mu) * grad
        x = x - lr * vel
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        if m < best_val:
            best_val = m
            best = x.copy()
    return best, best_val

def _greedy_holes(fixed, k, rng, n_samples=12000):
    outs = []
    cur = fixed.copy()
    for _ in range(k):
        Y = rng.randn(n_samples, 4)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        D = Y @ cur.T
        vals = D.max(axis=1)
        idx = int(np.argmin(vals))
        h = Y[idx]
        outs.append(h)
        cur = np.vstack([cur, h.reshape(1, 4)])
    return np.array(outs)

def _refine_holes(fixed, y_init, steps, T0, T1, lr0, mu=0.86, t_end=None):
    y = y_init.copy()
    k = y.shape[0]
    vel = np.zeros_like(y)
    lratio = np.log(T1 / T0)
    for it in range(steps):
        if t_end is not None and (it % 400 == 0) and time.time() > t_end:
            break
        frac = it / max(1, steps - 1)
        T = T0 * np.exp(lratio * frac)
        lr = lr0 * (1.0 - 0.6 * frac)
        D1 = y @ fixed.T
        G2 = y @ y.T
        np.fill_diagonal(G2, -2.0)
        m = float(max(D1.max(), G2.max()))
        E1 = np.exp((D1 - m) / T)
        E2 = np.exp((G2 - m) / T)
        np.fill_diagonal(E2, 0.0)
        S = float(E1.sum() + E2.sum() * 0.5)
        if S < 1e-300:
            S = 1e-300
        grad = (E1 @ fixed + E2 @ y) / S
        grad -= (grad * y).sum(axis=1, keepdims=True) * y
        vel = mu * vel + (1.0 - mu) * grad
        y = y - lr * vel
        y /= np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
    return y

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    SEARCH_END = t0 + 130.0
    rng = np.random.RandomState(987654321)
    n = 163
    try:
        bx = np.load("best.npy")
    except Exception:
        try:
            bx = np.load("record.npy")
        except Exception:
            bx = rng.randn(n, 4)
    bx = np.asarray(bx, dtype=float)
    if bx.shape != (n, 4) or not np.all(np.isfinite(bx)):
        bx = rng.randn(n, 4)
    bx = bx / np.maximum(np.linalg.norm(bx, axis=1, keepdims=True), 1e-12)
    best_x = bx.copy()
    best_val = _max_cos(best_x)
    pool = [(best_val, best_x.copy())]
    trial = 0
    while time.time() < SEARCH_END:
        if time.time() > DEADLINE - 36:
            break
        trial += 1
        mode = trial % 7
        # pick base: mostly best, sometimes second best for diversity
        if len(pool) > 1 and (trial % 5 == 0):
            base = pool[1][1]
        else:
            base = best_x
        if mode == 0:
            sigma = [0.015, 0.05, 0.12, 0.25][(trial // 7) % 4]
            cur = base + rng.randn(n, 4) * sigma
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            cb, cv = _softmax_desc(cur, steps=4200, T0=0.010, T1=0.00055, lr0=0.036, mu=0.87, rng=rng, noise0=0.0015, t_end=SEARCH_END)
        elif mode == 1:
            # joint hole-cluster surgery on worst-degree points
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            deg = (G > m - 0.003).sum(axis=1)
            krep = 3 + (trial % 5)  # 3..7
            worst = np.argsort(deg)[-krep:]
            keep = np.ones(n, dtype=bool); keep[worst] = False
            fixed = base[keep].copy()
            y0 = _greedy_holes(fixed, krep, rng, n_samples=10000)
            y1 = _refine_holes(fixed, y0, steps=1500, T0=0.008, T1=0.001, lr0=0.05, t_end=SEARCH_END)
            cur = base.copy()
            cur[worst] = y1
            cur += rng.randn(n, 4) * 0.004
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            cb, cv = _softmax_desc(cur, steps=4800, T0=0.011, T1=0.0005, lr0=0.038, mu=0.87, rng=rng, noise0=0.001, t_end=SEARCH_END)
            # follow with short high-s Riesz squeeze
            if cv < best_val + 2e-4 and time.time() < SEARCH_END:
                cb2, cv2 = _riesz_desc(cb, s0=20.0, s1=120.0, steps=1200, lr0=0.02, mu=0.9, t_end=SEARCH_END)
                if cv2 < cv:
                    cb, cv = cb2, cv2
        elif mode == 2:
            # cap shake
            u = rng.randn(4); u /= np.linalg.norm(u)
            dots = base @ u
            thr = np.quantile(dots, 0.86)
            idx = np.where(dots > thr)[0]
            cur = base.copy()
            cur[idx] += rng.randn(len(idx), 4) * 0.30
            cur += rng.randn(n, 4) * 0.008
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            cb, cv = _softmax_desc(cur, steps=4200, T0=0.014, T1=0.0007, lr0=0.040, mu=0.84, rng=rng, noise0=0.003, t_end=SEARCH_END)
        elif mode == 3:
            # Riesz hopping from small jitter
            cur = base + rng.randn(n, 4) * 0.03
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            cb, cv = _riesz_desc(cur, s0=8.0, s1=150.0, steps=2500, lr0=0.025, mu=0.90, t_end=SEARCH_END)
            if time.time() < SEARCH_END:
                cb, cv = _softmax_desc(cb, steps=2500, T0=0.006, T1=0.0004, lr0=0.030, mu=0.88, rng=rng, t_end=SEARCH_END)
        elif mode == 4:
            # rattler/hub targeted (deg at tight tol)
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            deg = (G > m - 1e-9).sum(axis=1)
            hub = np.argsort(deg)[-8:]
            cur = base.copy()
            cur[hub] += rng.randn(len(hub), 4) * 0.14
            # also kick single worst pair apart
            ij = np.unravel_index(int(np.argmax(G)), G.shape)
            a, b = ij[0], ij[1]
            cur[a] -= 0.05 * cur[b]
            cur[b] -= 0.05 * cur[a]
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            cb, cv = _softmax_desc(cur, steps=4300, T0=0.009, T1=0.0005, lr0=0.034, mu=0.88, rng=rng, noise0=0.0015, t_end=SEARCH_END)
        elif mode == 5:
            # global rotation + medium jitter
            cur = base + rng.randn(n, 4) * 0.09
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            A = rng.randn(4, 4) * 0.06
            A = A - A.T
            Q, _ = np.linalg.qr(np.eye(4) + A)
            cur = cur @ Q.T
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            cb, cv = _softmax_desc(cur, steps=4000, T0=0.015, T1=0.0006, lr0=0.042, mu=0.85, rng=rng, noise0=0.0025, t_end=SEARCH_END)
        else:
            # pure exploration random start, Riesz then softmax
            cur = rng.randn(n, 4)
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            cb, cv = _riesz_desc(cur, s0=6.0, s1=80.0, steps=2000, lr0=0.03, mu=0.88, t_end=SEARCH_END)
            if time.time() < SEARCH_END:
                cb, cv = _softmax_desc(cb, steps=3500, T0=0.016, T1=0.0007, lr0=0.045, mu=0.85, rng=rng, noise0=0.003, t_end=SEARCH_END)
        if cv < best_val - 1e-12:
            if cv < best_val - 2.5e-5 and time.time() < DEADLINE - 42:
                try:
                    pol, pv = _lp_polish(cb, seconds=2.2, t_end=DEADLINE - 32)
                except Exception:
                    pol, pv = cb, cv
                if pv < best_val:
                    best_x, best_val = pol, pv
                else:
                    best_x, best_val = cb, cv
            else:
                best_x, best_val = cb, cv
            pool.append((best_val, best_x.copy()))
            pool = sorted(pool, key=lambda z: z[0])[:3]
        elif cv < best_val + 8e-4:
            # keep diverse second best
            if len(pool) < 3 or cv < pool[-1][0]:
                # diversity check
                dup = False
                for _, px in pool:
                    if float(np.abs(px - cb).max()) < 1e-3:
                        dup = True
                        break
                if not dup:
                    pool.append((cv, cb.copy()))
                    pool = sorted(pool, key=lambda z: z[0])[:3]
        if best_val < 0.84:
            break
    remain = DEADLINE - time.time() - 1.5
    if remain > 3:
        try:
            px, pv = _lp_polish(best_x, seconds=min(remain - 1.0, 30.0), t_end=DEADLINE - 1.0)
            if pv < best_val:
                best_x, best_val = px, pv
            else:
                best_x = px
        except Exception:
            pass
        # polish runner-up and take winner
        if len(pool) > 1 and remain > 12:
            try:
                qx, qv = _lp_polish(pool[1][1], seconds=5.0, t_end=DEADLINE - 1.0)
                if qv < best_val:
                    best_x, best_val = qx, qv
            except Exception:
                pass
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
