import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def lp_polish(x, seconds, init_radius=0.008):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = init_radius
    start = time.time()
    iu = np.triu_indices(n, 1)
    while True:
        now = time.time()
        if now - start >= seconds or radius < 1e-14:
            break
        if seconds - (now - start) < 0.12:
            break
        G = x @ x.T
        c_all = G[iu]
        m = c_all.max()
        thr = m - max(20.0 * radius, 1e-9)
        sel_idx = np.where(c_all > thr)[0]
        if len(sel_idx) > 2500:
            part = np.argpartition(c_all, -2500)[-2500:]
            sel_idx = part
        I = iu[0][sel_idx]
        J = iu[1][sel_idx]
        csel = c_all[sel_idx]
        p = len(sel_idx)
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix(
            (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
            shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq,
                          b_eq=np.zeros(n),
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
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def exp_smooth(x, steps, alpha, lr, max_step=0.02):
    for _ in range(steps):
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        m = G.max()
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        step = lr * gt
        nm = np.linalg.norm(step, axis=1).max()
        if nm > max_step and nm > 0:
            step *= (max_step / nm)
        x = x - step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def pinch(x, steps, width, lr, max_step=0.02):
    for _ in range(steps):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        W = G - (m - width)
        W[W < 0] = 0.0
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        step = lr * gt
        nm = np.linalg.norm(step, axis=1).max()
        if nm > max_step and nm > 0:
            step *= (max_step / nm)
        x = x - step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def deepest_hole(Xkeep, rng, n_cand=4000):
    d = Xkeep.shape[1]
    cand = rng.standard_normal(size=(n_cand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ Xkeep.T
    worst = dots.max(axis=1)
    return cand[np.argmin(worst)]

def solve():
    t0 = time.time()
    LIMIT = 172.0
    rng = np.random.default_rng(0)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = np.load("record.npy").astype(float)
    try:
        xr = np.load("record.npy").astype(float)
        xr /= np.linalg.norm(xr, axis=1, keepdims=True)
        vr = max_cos(xr)
    except Exception:
        vr = 1e9
    xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    vb = max_cos(xb)
    if vr < vb:
        xb, vb = xr, vr
    n, d = xb.shape
    x_best = xb.copy()
    best_val = vb
    x_cur = xb.copy()
    cur_val = vb

    sigmas = [0.03, 0.08, 0.15, 0.05, 0.12, 0.02, 0.20, 0.06, 0.10, 0.04]
    it = 0
    HORIZON = LIMIT - 28.0
    while time.time() - t0 < HORIZON:
        it += 1
        elapsed = time.time() - t0
        frac = elapsed / max(HORIZON, 1e-9)
        # linear cooling 0.0025 -> 0.0002
        Temp = 0.0025 * (1.0 - frac) + 0.0002
        sigma = sigmas[(it - 1) % len(sigmas)]
        y = x_cur.copy()
        choice = rng.random()
        if choice < 0.35:
            y = y + sigma * rng.standard_normal(size=y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        elif choice < 0.60:
            center = int(rng.integers(0, n))
            dots = y[center] @ y.T
            nn = np.argsort(-dots)[:int(rng.integers(12, 30))]
            noise = np.zeros_like(y)
            noise[nn] = (sigma * 2.0) * rng.standard_normal(size=(len(nn), d))
            mask = np.ones(n, dtype=bool)
            mask[nn] = False
            noise[mask] = (sigma * 0.2) * rng.standard_normal(size=(int(mask.sum()), d))
            y = y + noise
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        elif choice < 0.85:
            # jammed-core surgery: kill most-contacted points
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            m = G.max()
            cnt = (G > m - 0.01).sum(axis=1)
            k = int(rng.integers(2, 5))
            # bias to high-contact
            order = np.argsort(-cnt - rng.random(n) * 2.0)[:k]
            victims = order
            if rng.random() < 0.3:
                # include one random rattler-adjacent victim
                victims[-1] = int(rng.integers(0, n))
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[victims] = False
            Xkeep = y[keep_mask]
            for v in victims:
                h = deepest_hole(Xkeep, rng, n_cand=4000)
                h = h + 0.01 * rng.standard_normal(size=d)
                h /= np.linalg.norm(h)
                y[v] = h
                Xkeep = np.vstack([Xkeep, h.reshape(1, -1)])
        else:
            # pair-split: push apart a worst pair
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            ij = np.argmax(G)
            a = ij // n; b = ij % n
            diff = y[a] - y[b]
            nn = np.linalg.norm(diff)
            if nn > 1e-12:
                diff /= nn
            push = sigma * 1.5
            y[a] = y[a] + push * diff + 0.01 * rng.standard_normal(size=d)
            y[b] = y[b] - push * diff + 0.01 * rng.standard_normal(size=d)
            y = y + 0.002 * rng.standard_normal(size=y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)

        if it % 2 == 0:
            y = exp_smooth(y, steps=300, alpha=30.0, lr=0.004)
            y = pinch(y, steps=180, width=0.025, lr=0.30)
        else:
            y = exp_smooth(y, steps=220, alpha=18.0, lr=0.006)
            y = pinch(y, steps=250, width=0.04, lr=0.25)

        remain = HORIZON - (time.time() - t0)
        if remain < 1.5:
            break
        budget = min(3.5, remain)
        y = lp_polish(y, seconds=budget, init_radius=0.008)
        v = max_cos(y)
        # SA acceptance on current, elite on best
        if v < best_val - 1e-12:
            x_best = y.copy()
            best_val = v
            x_cur = y.copy()
            cur_val = v
        elif v < cur_val or rng.random() < np.exp((cur_val - v) / max(Temp, 1e-9)):
            # avoid drifting too far
            if v < cur_val + 0.008:
                x_cur = y.copy()
                cur_val = v
        # opportunistic rattler fill on best every 6 iters
        if it % 6 == 0 and time.time() - t0 < HORIZON - 4.0:
            G = x_best @ x_best.T
            np.fill_diagonal(G, -2.0)
            m = G.max()
            cnt0 = (G > m - 1e-9).sum(axis=1)
            ratt = np.where(cnt0 == 0)[0]
            if len(ratt) > 0:
                yt = x_best.copy()
                keep_m = np.ones(n, dtype=bool)
                keep_m[ratt] = False
                Xk = yt[keep_m]
                for vv in ratt:
                    h = deepest_hole(Xk, rng, n_cand=5000)
                    trial = float((Xk @ h).max())
                    if trial < m - 1e-9:
                        yt[vv] = h
                        Xk = np.vstack([Xk, h.reshape(1, -1)])
                vt = max_cos(yt)
                if vt <= best_val:
                    rb = min(2.5, HORIZON - (time.time() - t0))
                    if rb > 0.8:
                        yt = lp_polish(yt, seconds=rb, init_radius=0.003)
                        vt2 = max_cos(yt)
                        if vt2 < best_val - 1e-12:
                            x_best, best_val = yt, vt2
                            x_cur, cur_val = yt.copy(), vt2
                    elif vt < best_val - 1e-12:
                        x_best, best_val = yt, vt

    rem = LIMIT - (time.time() - t0) - 1.0
    if rem > 2.0:
        x_best = lp_polish(x_best, seconds=rem, init_radius=0.004)
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
