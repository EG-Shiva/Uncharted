import time
import numpy as np

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _worst_vec(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1)

def _smooth(x, steps, lr0, lr1, tau0, tau1, mu=0.8):
    n, d = x.shape
    v = np.zeros_like(x)
    for s in range(steps):
        f = s / max(1, steps - 1)
        tau = tau0 * (tau1 / tau0) ** f
        lr = lr0 * (lr1 / lr0) ** f
        G = x @ x.T
        np.fill_diagonal(G, -1.0)
        M = float(G.max())
        E = np.exp((G - M) / tau)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if not np.isfinite(S) or S <= 0:
            break
        W = E / S
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        gn = np.linalg.norm(gt, axis=1, keepdims=True) + 1e-12
        gt = gt / np.maximum(1.0, gn * 5.0)
        v = mu * v + gt
        x = x - lr * v
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _void_replace(y, to_replace, rng, n_cand=8000):
    n, d = y.shape
    fixed_mask = np.ones(n, dtype=bool)
    fixed_mask[np.asarray(to_replace)] = False
    Xf = y[fixed_mask]
    for k in to_replace:
        cand = rng.standard_normal(size=(n_cand, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        worst = (cand @ Xf.T).max(axis=1)
        bi = int(np.argmin(worst))
        y[k] = cand[bi]
        Xf = np.vstack([Xf, y[k][None, :]])
    return y

def solve():
    t0 = time.time()
    DEAD = t0 + 168.0
    rng = np.random.default_rng(7)
    x0 = None; b0 = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (323, 7):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    v = _max_cos(a)
                    if v < b0:
                        b0 = v; x0 = a.copy()
        except Exception:
            pass
    if x0 is None:
        a = rng.standard_normal(size=(323, 7))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        x0 = a; b0 = _max_cos(a)
    x_best = x0.copy(); best = float(b0)
    n, d = 323, 7
    it = 0
    while time.time() < DEAD - 8.0:
        x = x_best.copy()
        s = it % 6
        if s == 0:
            x = x + rng.standard_normal(size=x.shape) * 0.12
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _smooth(x, 400, 0.03, 0.003, 0.012, 0.002)
        elif s == 1:
            x = x + rng.standard_normal(size=x.shape) * 0.05
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _smooth(x, 350, 0.025, 0.003, 0.010, 0.002)
        elif s == 2:
            w = _worst_vec(x)
            k = 30
            idx = np.argpartition(w, -k)[-k:]
            x[idx] += rng.standard_normal(size=(k, d)) * 0.15
            x[idx] /= np.linalg.norm(x[idx], axis=1, keepdims=True)
            x = _smooth(x, 400, 0.03, 0.003, 0.012, 0.0018)
        elif s == 3:
            m = _max_cos(x)
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            cc = np.sum(G > m - 1e-3, axis=1)
            low = np.where(cc <= 2)[0]
            order = np.argsort(-cc)
            torep = [int(v) for v in low[:4]]
            for o in order:
                if len(torep) >= 8:
                    break
                if int(o) not in torep:
                    torep.append(int(o))
            torep = torep[:8]
            if len(torep) > 0:
                x = _void_replace(x, torep, rng, n_cand=8000)
            x += rng.standard_normal(size=x.shape) * 0.004
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _smooth(x, 450, 0.025, 0.002, 0.010, 0.0018)
        elif s == 4:
            k = 40
            idx = rng.choice(n, size=k, replace=False)
            x[idx] += rng.standard_normal(size=(k, d)) * 0.25
            x[idx] /= np.linalg.norm(x[idx], axis=1, keepdims=True)
            x = _smooth(x, 400, 0.03, 0.003, 0.013, 0.002)
        else:
            x = x + rng.standard_normal(size=x.shape) * 0.30
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _smooth(x, 450, 0.035, 0.003, 0.015, 0.002)
        # cheap single-point hill on worst point
        if time.time() < DEAD - 8.0:
            for _ in range(1):
                w = _worst_vec(x)
                k = int(np.argmax(w))
                oth = np.delete(x, k, axis=0)
                cur_w = float((oth @ x[k]).max())
                trial = x[k][None, :] + rng.standard_normal(size=(1500, d)) * 0.02
                trial /= np.linalg.norm(trial, axis=1, keepdims=True)
                dw = (trial @ oth.T).max(axis=1)
                bi = int(np.argmin(dw))
                if dw[bi] < cur_w:
                    x[k] = trial[bi]
        v = _max_cos(x)
        if v < best - 1e-12 and np.all(np.isfinite(x)):
            best = v; x_best = x.copy()
        it += 1
    # final greedy polish on worst points (no LP)
    try:
        x = x_best.copy()
        for _ in range(30):
            if time.time() >= DEAD - 1.5:
                break
            w = _worst_vec(x)
            k = int(np.argmax(w))
            if w[k] < best:
                best = float(w[k])
                x_best = x.copy()
            oth = np.delete(x, k, axis=0)
            cur_w = float((oth @ x[k]).max())
            trial = x[k][None, :] + rng.standard_normal(size=(2000, d)) * 0.008
            trial /= np.linalg.norm(trial, axis=1, keepdims=True)
            dw = (trial @ oth.T).max(axis=1)
            bi = int(np.argmin(dw))
            if dw[bi] < cur_w:
                x[k] = trial[bi]
        v = _max_cos(x)
        if v < best and np.all(np.isfinite(x)):
            best = v; x_best = x.copy()
    except Exception:
        pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
