import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
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
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def _e8_roots():
    pts = []
    for i in range(8):
        for j in range(i + 1, 8):
            for si in (1.0, -1.0):
                for sj in (1.0, -1.0):
                    v = np.zeros(8)
                    v[i] = si
                    v[j] = sj
                    pts.append(v)
    for mask in range(256):
        signs = np.ones(8)
        nm = 0
        for k in range(8):
            if (mask >> k) & 1:
                signs[k] = -1.0
                nm += 1
        if nm % 2 == 0:
            pts.append(0.5 * signs)
    a = np.array(pts, dtype=float)
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    return a


def solve():
    t0 = time.time()
    BUDGET = 172.0
    rng = np.random.default_rng(12345)
    n, d = 168, 8
    try:
        best0 = np.load("best.npy").astype(float)
    except Exception:
        best0 = None
    try:
        rec0 = np.load("record.npy").astype(float)
    except Exception:
        rec0 = None
    cands = []
    if best0 is not None and best0.shape == (n, d):
        cands.append(best0 / np.linalg.norm(best0, axis=1, keepdims=True))
    if rec0 is not None and rec0.shape == (n, d):
        cands.append(rec0 / np.linalg.norm(rec0, axis=1, keepdims=True))
    if not cands:
        c = rng.normal(size=(n, d))
        c /= np.linalg.norm(c, axis=1, keepdims=True)
        cands.append(c)
    best_x = min(cands, key=_max_cos)
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    best_val = _max_cos(best_x)
    e8 = _e8_roots()

    def adam_run(x_init, steps, a0, a1, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        x = x_init.copy()
        m1 = np.zeros_like(x)
        m2 = np.zeros_like(x)
        b1p = 1.0
        b2p = 1.0
        la0 = np.log(a0)
        la1 = np.log(a1)
        for s in range(steps):
            if s % 400 == 0 and time.time() - t0 > BUDGET - 42:
                break
            frac = s / max(steps - 1, 1)
            alpha = float(np.exp(la0 + (la1 - la0) * frac))
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            mv = G.max()
            # global softmax weights
            W = np.exp(alpha * (G - mv))
            np.fill_diagonal(W, 0.0)
            S = W.sum() * 0.5 + 1e-300
            A = (W @ x) / S
            dot = np.sum(A * x, axis=1, keepdims=True)
            At = A - dot * x
            b1p *= beta1
            b2p *= beta2
            m1 = beta1 * m1 + (1.0 - beta1) * At
            m2 = beta2 * m2 + (1.0 - beta2) * (At * At)
            m1h = m1 / (1.0 - b1p)
            m2h = m2 / (1.0 - b2p)
            step = lr * m1h / (np.sqrt(m2h) + eps)
            sdot = np.sum(step * x, axis=1, keepdims=True)
            step_t = step - sdot * x
            x = x - step_t
            nr = np.linalg.norm(x, axis=1, keepdims=True)
            nr[nr == 0] = 1.0
            x /= nr
        return x

    def refine_single(x0, Y, steps=350, lr0=0.08, a0=30.0, a1=250.0):
        x = x0 / np.linalg.norm(x0)
        la0 = np.log(a0)
        la1 = np.log(a1)
        for s in range(steps):
            frac = s / max(steps - 1, 1)
            alpha = float(np.exp(la0 + (la1 - la0) * frac))
            lr = lr0 * (1.0 - frac * 0.9)
            g = Y @ x
            mv = g.max()
            w = np.exp(alpha * (g - mv))
            sw = w.sum() + 1e-300
            grad = (w @ Y) / sw
            gt = grad - np.dot(grad, x) * x
            x = x - lr * gt
            nn = np.linalg.norm(x)
            if nn == 0:
                x = x0 / np.linalg.norm(x0)
            else:
                x /= nn
        return x

    def reinsert_op(x_cur, k, rng_loc):
        G = x_cur @ x_cur.T
        np.fill_diagonal(G, -2.0)
        mv = G.max()
        cnt = (G > mv - 0.004).sum(axis=1)
        worst = G.max(axis=1)
        score = worst + 0.002 * (cnt / max(cnt.max(), 1))
        order = np.argsort(-score)
        pool = order[:max(2 * k, 10)]
        sel = rng_loc.choice(pool, size=min(k, len(pool)), replace=False)
        keep = np.ones(n, dtype=bool)
        keep[sel] = False
        Y = x_cur[keep]
        # insert one by one
        cur = Y
        for _ in range(k):
            Nc = 4000
            C = rng_loc.normal(size=(Nc, d))
            C /= np.linalg.norm(C, axis=1, keepdims=True)
            S = C @ cur.T
            vals = S.max(axis=1)
            bi = np.argsort(vals)[:6]
            best_c = None
            best_v = 1e9
            for b in bi:
                r = refine_single(C[b], cur, steps=300,
                                  lr0=float(rng_loc.uniform(0.04, 0.10)),
                                  a0=30.0, a1=float(rng_loc.uniform(150, 300)))
                v = float((cur @ r).max())
                if v < best_v:
                    best_v = v
                    best_c = r
            cur = np.vstack([cur, best_c[None, :]])
        return cur

    # quick E8 diverse starts
    for _ in range(2):
        if time.time() - t0 > BUDGET - 50:
            break
        idx = rng.choice(240, size=n, replace=False)
        x0 = e8[idx] + rng.normal(size=(n, d)) * 0.02
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        xo = adam_run(x0, 2200, 25.0, 220.0, 0.004)
        v = _max_cos(xo)
        if v < best_val:
            best_val = v
            best_x = xo

    trial = 0
    while time.time() - t0 < BUDGET - 40:
        trial += 1
        r = rng.random()
        if r < 0.10:
            x0 = rng.normal(size=(n, d))
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            steps = int(rng.integers(2200, 3800))
            a0 = float(rng.uniform(20, 45)); a1 = float(rng.uniform(150, 300))
            lr = float(rng.uniform(0.002, 0.006))
            xo = adam_run(x0, steps, a0, a1, lr)
        elif r < 0.45:
            scale = 10.0 ** float(rng.uniform(-2.8, -0.5))
            x0 = best_x + rng.normal(size=(n, d)) * scale
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            steps = int(rng.integers(1200, 2800))
            a0 = float(rng.uniform(20, 50)); a1 = float(rng.uniform(120, 300))
            lr = float(rng.uniform(0.002, 0.007))
            xo = adam_run(x0, steps, a0, a1, lr)
        elif r < 0.75:
            k = int(rng.integers(2, 12))
            try:
                x0 = reinsert_op(best_x, k, rng)
            except Exception:
                continue
            x0 += rng.normal(size=(n, d)) * 0.004
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            steps = int(rng.integers(1500, 3000))
            a0 = float(rng.uniform(25, 55)); a1 = float(rng.uniform(150, 320))
            lr = float(rng.uniform(0.002, 0.006))
            xo = adam_run(x0, steps, a0, a1, lr)
        else:
            G = best_x @ best_x.T
            np.fill_diagonal(G, -2.0)
            mv = G.max()
            cnt = (G > mv - 0.004).sum(axis=1)
            order = np.argsort(-cnt)
            pool = order[:30]
            k = int(rng.integers(4, 12))
            sel = rng.choice(pool, size=min(k, len(pool)), replace=False)
            x0 = best_x.copy()
            sb = float(rng.uniform(0.12, 0.45))
            x0[sel] += rng.normal(size=(len(sel), d)) * sb
            x0 += rng.normal(size=(n, d)) * 0.006
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            steps = int(rng.integers(1500, 3000))
            a0 = float(rng.uniform(25, 60)); a1 = float(rng.uniform(150, 320))
            lr = float(rng.uniform(0.002, 0.006))
            xo = adam_run(x0, steps, a0, a1, lr)
        v = _max_cos(xo)
        if v < best_val - 1e-12:
            # snap with short LP to get true value
            xo2 = _lp_polish(xo, 1.5)
            v2 = _max_cos(xo2)
            if v2 < best_val:
                best_val = v2
                best_x = xo2
            else:
                # keep adam version if LP failed to improve but adam did
                if v < best_val:
                    best_val = v
                    best_x = xo
            # intensify
            if time.time() - t0 < BUDGET - 50:
                xo3 = adam_run(best_x, 1000, 80.0, 300.0, 0.002)
                v3 = _max_cos(xo3)
                if v3 < best_val:
                    best_val = v3
                    best_x = xo3
        if time.time() - t0 > BUDGET - 40:
            break

    remain = BUDGET - (time.time() - t0) - 1.5
    if remain > 4:
        best_x = _lp_polish(best_x, min(remain, 37.0))
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
