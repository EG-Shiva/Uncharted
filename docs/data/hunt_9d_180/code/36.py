import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _lp_polish(x, seconds, hard_deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while True:
        now = time.time()
        if now - start >= seconds or now >= hard_deadline or radius <= 1e-14:
            break
        G = x @ x.T
        cfull = G[iu0]
        m = float(np.max(cfull))
        tol = max(20.0 * radius, 1e-9)
        sel = cfull > m - tol
        I = iu0[0][sel]; J = iu0[1][sel]
        csub = cfull[sel]
        p = len(I)
        if p == 0:
            break
        # cap constraints for speed
        if p > 4000:
            # keep most tight
            idx = np.argpartition(-csub, 4000)[:4000]
            I = I[idx]; J = J[idx]; csub = csub[idx]
            p = 4000
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr() if False else (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csub, A_eq=A_eq, b_eq=np.zeros(n),
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

def _smooth_anneal(x, steps, T0, T1, lr0, mu, rng):
    n, d = x.shape
    x = x.copy()
    bv = _max_cos(x)
    best = x.copy()
    vel = np.zeros_like(x)
    lT = np.log(T1 / T0) if T0 > 0 and T1 > 0 else 0.0
    for s in range(steps):
        frac = s / max(1, steps - 1)
        T = T0 * np.exp(lT * frac)
        lr = lr0 * (1.0 - 0.5 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = float(np.max(G))
        # weights
        E = np.exp((G - gmax) / T)
        np.fill_diagonal(E, 0.0)
        row = np.sum(E, axis=1, keepdims=True) + 1e-12
        grad = E @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gtan = grad - dot * x
        step = gtan / row
        nm = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        step = step * np.minimum(1.0, 0.5 / nm)
        vel = mu * vel + step
        x = x - lr * vel
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        if (s & 63) == 63 or s == steps - 1:
            v = _max_cos(x)
            if v < bv:
                bv = v
                best = x.copy()
    return best, bv

def _random_rotation_small(d, rng, theta):
    a = rng.randn(d)
    a /= np.linalg.norm(a) + 1e-12
    b = rng.randn(d)
    b = b - np.dot(b, a) * a
    b /= np.linalg.norm(b) + 1e-12
    c = np.cos(theta); s = np.sin(theta)
    R = np.eye(d) + (c - 1.0) * (np.outer(a, a) + np.outer(b, b)) + s * (np.outer(a, b) - np.outer(b, a))
    return R

def _best_void(others, rng, n_try=3000, n_refine=250):
    d = others.shape[1]
    cand = rng.randn(n_try, d)
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ others.T
    m = np.max(dots, axis=1)
    bi = int(np.argmin(m))
    y = cand[bi].copy()
    cur = float(m[bi])
    for k in range(n_refine):
        frac = k / max(1, n_refine - 1)
        sig = 0.03 * (0.001 / 0.03) ** frac
        prop = y + sig * rng.randn(d)
        prop /= np.linalg.norm(prop) + 1e-12
        pm = float(np.max(prop @ others.T))
        if pm < cur:
            y = prop
            cur = pm
    return y

def solve():
    t0 = time.time()
    hard_deadline = t0 + 175.0
    explore_end = t0 + 125.0
    rng = np.random.RandomState(0)

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            c = np.asarray(np.load(fn), dtype=float)
            if c.shape == (180, 9):
                c = c / np.linalg.norm(c, axis=1, keepdims=True)
                if np.all(np.isfinite(c)):
                    cands.append((_max_cos(c), c))
        except Exception:
            pass
    if len(cands) == 0:
        x_best = rng.randn(180, 9)
        x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
        b_best = _max_cos(x_best)
    else:
        cands.sort(key=lambda t: t[0])
        b_best, x_best = cands[0][0], cands[0][1].copy()

    n, d = (180, 9)
    n_trials = 70

    for trial in range(n_trials):
        if time.time() >= explore_end or time.time() >= hard_deadline - 40:
            break
        x = x_best.copy()
        mode = trial % 5  # 0..3 cluster, 4 void
        if mode < 4:
            # coherent cap rotation
            u = rng.randn(d); u /= np.linalg.norm(u) + 1e-12
            proj = x @ u
            k = int(rng.randint(15, 36))
            if rng.rand() < 0.5:
                cap = np.argpartition(proj, n - k)[n - k:]
            else:
                cap = np.argpartition(-proj, n - k)[n - k:]
            theta = float(rng.uniform(0.02, 0.13)) * (1.0 if rng.rand() < 0.5 else -1.0)
            R = _random_rotation_small(d, rng, theta)
            x[cap] = (R @ x[cap].T).T
            # small global jitter to break exact rigidity
            sig_all = float(rng.uniform(0.003, 0.010))
            x = x + sig_all * rng.randn(n, d)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            # optionally re-seat 1 rattler-like point to a void
            if rng.rand() < 0.4:
                Gtmp = x @ x.T
                np.fill_diagonal(Gtmp, -2.0)
                pmax = np.max(Gtmp, axis=1)
                wi = int(np.argmax(pmax))
                others = np.delete(x, wi, axis=0)
                y = _best_void(others, rng, n_try=2000, n_refine=150)
                x[wi] = y
            steps = 350
            T0 = float(rng.uniform(0.008, 0.018))
            T1 = float(rng.uniform(0.001, 0.003))
            lr0 = float(rng.uniform(0.05, 0.11))
            mu = 0.85
        else:
            # worst-point void re-insertion (1-2 points)
            Gtmp = x @ x.T
            np.fill_diagonal(Gtmp, -2.0)
            pmax = np.max(Gtmp, axis=1)
            order = np.argsort(-pmax)
            nk = 2 if rng.rand() < 0.5 else 1
            for t in range(nk):
                wi = int(order[t])
                others = np.delete(x, wi, axis=0)
                y = _best_void(others, rng, n_try=3500, n_refine=250)
                x[wi] = y
            # tiny jitter rest
            sig_all = 0.003
            noise = sig_all * rng.randn(n, d)
            # don't jitter just-inserted points too much
            x = x + noise
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps = 400
            T0 = 0.012; T1 = 0.0012; lr0 = 0.08; mu = 0.85

        try:
            xb, vb = _smooth_anneal(x, steps, T0, T1, lr0, mu, rng)
        except Exception:
            continue
        if vb < b_best - 1e-12 and np.all(np.isfinite(xb)):
            # strict improvement
            x_best = xb / np.linalg.norm(xb, axis=1, keepdims=True)
            b_best = vb
            # early exit if big win? keep going

    # final capped LP polish
    remain = hard_deadline - time.time() - 2.0
    if remain > 5.0:
        secs = min(32.0, remain)
        try:
            x_best = _lp_polish(x_best, secs, hard_deadline - 1.0)
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    # safety: replace non-finite
    if not np.all(np.isfinite(x_best)):
        x_best = np.asarray(np.load("best.npy"), dtype=float)
        x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
