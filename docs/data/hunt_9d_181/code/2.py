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
        if seconds - (now - start) < 0.15:
            break
        G = x @ x.T
        c_all = G[iu]
        m = c_all.max()
        thr = m - max(20.0 * radius, 1e-9)
        sel_idx = np.where(c_all > thr)[0]
        if len(sel_idx) > 3000:
            # keep tightest 3000
            part = np.argpartition(c_all, -3000)[-3000:]
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
    n, d = x.shape
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


def deepest_hole(Xkeep, rng, n_cand=8000):
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

    # quick contact info for strategy choice (not strictly needed)
    x_best = xb.copy()
    best_val = vb

    sigmas = [0.03, 0.08, 0.15, 0.05, 0.12, 0.02, 0.20, 0.06, 0.10, 0.04]
    it = 0
    # reserve ~25s for final polish
    while time.time() - t0 < LIMIT - 30.0:
        it += 1
        sigma = sigmas[(it - 1) % len(sigmas)]
        y = x_best.copy()
        choice = rng.random()
        if choice < 0.45:
            # global shake
            y = y + sigma * rng.standard_normal(size=y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        elif choice < 0.75:
            # local cluster shake around random center
            center = int(rng.integers(0, n))
            dots = y[center] @ y.T
            nn = np.argsort(-dots)[:22]
            noise = np.zeros_like(y)
            noise[nn] = (sigma * 2.0) * rng.standard_normal(size=(len(nn), d))
            mask = np.ones(n, dtype=bool)
            mask[nn] = False
            noise[mask] = (sigma * 0.2) * rng.standard_normal(size=(int(mask.sum()), d))
            y = y + noise
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        else:
            # void reinsertion of 2-4 low-contact points
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            m = G.max()
            cnt = (G > m - 0.01).sum(axis=1)
            k = int(rng.integers(2, 5))
            victims = np.argsort(cnt)[:k]
            # also randomly add one high-contact victim sometimes
            if rng.random() < 0.5:
                G2 = y @ y.T
                np.fill_diagonal(G2, -2.0)
                m2 = G2.max()
                cnt2 = (G2 > m2 - 1e-7).sum(axis=1)
                victims[0] = int(np.argmax(cnt2))
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[victims] = False
            Xkeep = y[keep_mask]
            for v in victims:
                h = deepest_hole(Xkeep, rng, n_cand=6000)
                # small jitter around hole
                h = h + 0.01 * rng.standard_normal(size=d)
                h /= np.linalg.norm(h)
                y[v] = h
                Xkeep = np.vstack([Xkeep, h.reshape(1, -1)])

        # smoothing: alternate parameters deterministically
        if it % 2 == 0:
            y = exp_smooth(y, steps=350, alpha=28.0, lr=0.004)
            y = pinch(y, steps=200, width=0.025, lr=0.30)
        else:
            y = exp_smooth(y, steps=250, alpha=18.0, lr=0.006)
            y = pinch(y, steps=300, width=0.04, lr=0.25)

        # short LP polish
        remain = LIMIT - 30.0 - (time.time() - t0)
        if remain < 2.0:
            break
        budget = min(5.0, remain)
        y = lp_polish(y, seconds=budget, init_radius=0.008)
        v = max_cos(y)
        if v < best_val - 1e-12:
            x_best = y
            best_val = v
        # also try rattler relocation preserving score: move 0-contact points to holes
        # cheap opportunistic step every few iters
        if it % 5 == 0 and time.time() - t0 < LIMIT - 35.0:
            G = x_best @ x_best.T
            np.fill_diagonal(G, -2.0)
            m = G.max()
            cnt = (G > m - 1e-9).sum(axis=1)
            ratt = np.where(cnt == 0)[0]
            if len(ratt) > 0:
                yt = x_best.copy()
                keep_m = np.ones(n, dtype=bool)
                keep_m[ratt] = False
                Xk = yt[keep_m]
                for v in ratt:
                    h = deepest_hole(Xk, rng, n_cand=8000)
                    # accept only if it does not worsen max
                    trial = float((Xk @ h).max())
                    if trial < m - 1e-9:
                        yt[v] = h
                        Xk = np.vstack([Xk, h.reshape(1, -1)])
                vt = max_cos(yt)
                if vt <= best_val:
                    # polish lightly
                    yt = lp_polish(yt, seconds=min(3.0, LIMIT - 30.0 - (time.time() - t0)), init_radius=0.003)
                    vt2 = max_cos(yt)
                    if vt2 < best_val - 1e-12:
                        x_best, best_val = yt, vt2

    # final long polish with remaining time
    rem = LIMIT - (time.time() - t0) - 1.0
    if rem > 2.0:
        x_best = lp_polish(x_best, seconds=rem, init_radius=0.004)
        best_val = max_cos(x_best)

    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
