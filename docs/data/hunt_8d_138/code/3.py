import time
import numpy as np

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def soft_polish(x, steps, beta0, beta1, lr):
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    denom = max(1, steps - 1)
    for t in range(1, steps + 1):
        beta = beta0 + (beta1 - beta0) * (t - 1) / denom
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        mv = G.max()
        W = np.exp(beta * (G - mv))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        nd = np.sum(grad * x, axis=1, keepdims=True)
        grad_t = grad - nd * x
        m *= b1
        m += (1.0 - b1) * grad_t
        v *= b2
        v += (1.0 - b2) * (grad_t * grad_t)
        bc1 = 1.0 - b1 ** t
        bc2 = 1.0 - b2 ** t
        mhat = m / bc1
        vhat = v / bc2
        upd = mhat / (np.sqrt(vhat) + eps)
        nd2 = np.sum(upd * x, axis=1, keepdims=True)
        upd = upd - nd2 * x
        x = x - lr * upd
        nrm = np.sqrt(np.sum(x * x, axis=1, keepdims=True)) + 1e-30
        x = x / nrm
    return x

def lp_polish(x, seconds):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        mval = c.max()
        sel = c > mval - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        vv = max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    import time
    import numpy as np
    t0 = time.time()
    np.random.seed(0)
    rng = np.random.RandomState(12345)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.randn(138, 8)
    n, d = 138, 8
    if xb.shape != (n, d):
        # adapt if shape differs
        tmp = rng.randn(n, d)
        m0 = min(xb.shape[0], n)
        tmp[:m0, :min(xb.shape[1], d)] = xb[:m0, :min(xb.shape[1], d)]
        xb = tmp
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best = max_cos(xb)
    x_best = xb.copy()
    cur = xb.copy()
    cur_val = best
    stall = 0
    it = 0
    EXPL = 138.0
    # quick initial soft squeeze of loaded code (in case not fully polished)
    # keep it short to save time for hopping
    try:
        if time.time() - t0 < EXPL:
            cand0 = soft_polish(cur.copy(), 300, 30.0, 200.0, 0.004)
            v0 = max_cos(cand0)
            if v0 < best:
                best = v0
                x_best = cand0.copy()
                cur = cand0.copy()
                cur_val = v0
    except Exception:
        pass
    while time.time() - t0 < EXPL:
        it += 1
        mode = it % 3
        try:
            if mode == 0:
                sigma = 0.25 if (it % 12 == 0) else (0.12 if (it % 6 == 0) else 0.03)
                steps = 600 if sigma > 0.1 else 400
                prop = cur + sigma * rng.randn(n, d)
                prop /= np.linalg.norm(prop, axis=1, keepdims=True)
                polished = soft_polish(prop, steps, 15.0, 120.0, 0.008)
            elif mode == 1:
                G = cur @ cur.T
                np.fill_diagonal(G, -2.0)
                cmax = G.max()
                cnt = (G > cmax - 0.01).sum(axis=1)
                order = np.argsort(-cnt)
                pool = order[:8]
                k = 3
                chosen = pool[rng.choice(len(pool), size=k, replace=False)]
                prop = cur.copy()
                prop[chosen] += 0.25 * rng.randn(k, d)
                prop += 0.01 * rng.randn(n, d)
                prop /= np.linalg.norm(prop, axis=1, keepdims=True)
                polished = soft_polish(prop, 600, 15.0, 150.0, 0.008)
            else:
                G = cur @ cur.T
                np.fill_diagonal(G, -2.0)
                cmax = G.max()
                cnt = (G > cmax - 0.008).sum(axis=1)
                order = np.argsort(-cnt)
                pool = order[:10]
                victim = int(pool[rng.randint(len(pool))])
                Xrem = np.delete(cur, victim, axis=0)
                Ncand = 4000
                cand = rng.randn(Ncand, d)
                cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                smax = (cand @ Xrem.T).max(axis=1)
                bi = int(np.argmin(smax))
                prop = cur.copy()
                prop[victim] = cand[bi]
                prop += 0.005 * rng.randn(n, d)
                prop /= np.linalg.norm(prop, axis=1, keepdims=True)
                polished = soft_polish(prop, 600, 15.0, 150.0, 0.008)
        except Exception:
            stall += 1
            continue
        try:
            val = max_cos(polished)
        except Exception:
            stall += 1
            continue
        if val < cur_val - 1e-12:
            cur = polished
            cur_val = val
            stall = 0
            if val < best - 1e-12:
                best = val
                x_best = polished.copy()
        else:
            stall += 1
            if rng.rand() < 0.02 and val < cur_val + 0.01:
                cur = polished
                cur_val = val
            if stall >= 15:
                # restart with big jump from best
                try:
                    tmp = x_best + 0.15 * rng.randn(n, d)
                    tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
                    tmp = soft_polish(tmp, 500, 15.0, 120.0, 0.008)
                    cur_val = max_cos(tmp)
                    cur = tmp
                    if cur_val < best - 1e-12:
                        best = cur_val
                        x_best = cur.copy()
                except Exception:
                    cur = x_best.copy()
                    cur_val = best
                stall = 0
        # safety: if time almost up break
        if time.time() - t0 >= EXPL:
            break
    # final intensive squeeze
    try:
        tleft = 175.0 - (time.time() - t0)
        if tleft > 8:
            steps = 1200 if tleft > 32 else 600
            x_best = soft_polish(x_best.copy(), steps, 30.0, 350.0, 0.004)
            best = max_cos(x_best)
    except Exception:
        pass
    try:
        tleft2 = 178.0 - (time.time() - t0)
        if tleft2 > 5:
            x_best = lp_polish(x_best, min(tleft2 - 1.0, 25.0))
    except Exception:
        pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    # nan guard
    if not np.all(np.isfinite(x_best)):
        x_best = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    return x_best
