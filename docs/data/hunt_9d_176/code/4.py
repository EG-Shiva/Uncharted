import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def lp_polish_deadline(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = max_cos(x)
    except Exception:
        return x
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        Gfull = x @ x.T
        c = Gfull[iu]
        m = float(c.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = c > thr
        # cap to tightest 3000 to keep LP fast
        if np.count_nonzero(sel) > 3000:
            idx = np.where(sel)[0]
            order = np.argsort(-c[idx])[:3000]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
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
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def soft_quench(X, k0, k1, steps, lr0, deadline):
    n, d = X.shape
    m1 = np.zeros_like(X)
    m2 = np.zeros_like(X)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    b1p = 1.0; b2p = 1.0
    for t in range(1, steps + 1):
        if time.time() > deadline - 0.15:
            break
        frac = t / steps
        k = k0 + (k1 - k0) * frac
        # cosine decay lr
        lr = lr0 * (0.15 + 0.85 * 0.5 * (1.0 + np.cos(np.pi * frac)))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = float(G.max())
        # exp stable
        E = np.exp(k * (G - M))
        np.fill_diagonal(E, 0.0)
        S = E.sum() * 0.5
        if not np.isfinite(S) or S <= 0:
            break
        P = E / S
        grad = P @ X
        dot = np.sum(grad * X, axis=1, keepdims=True)
        g = grad - dot * X
        # adam
        b1p *= b1; b2p *= b2
        # use in-place-ish
        m1 = b1 * m1 + (1.0 - b1) * g
        m2 = b2 * m2 + (1.0 - b2) * (g * g)
        m1h = m1 / (1.0 - b1p)
        m2h = m2 / (1.0 - b2p)
        step = lr * m1h / (np.sqrt(m2h) + eps)
        # clip per-row step to avoid blowup
        sn = np.linalg.norm(step, axis=1, keepdims=True)
        cap = 0.02
        over = sn > cap
        if np.any(over):
            step = np.where(over, step * (cap / np.maximum(sn, 1e-12)), step)
        X = X - step
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X

def solve():
    t0 = time.time()
    HARD_END = t0 + 178.0
    SEARCH_END = t0 + 155.0
    rng = np.random.default_rng(0)
    # load start
    X_best = None
    best_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (176, 9):
                an = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = max_cos(an)
                if v < best_val:
                    best_val = v
                    X_best = an.copy()
        except Exception:
            continue
    if X_best is None:
        X_best = rng.standard_normal((176, 9))
        X_best /= np.linalg.norm(X_best, axis=1, keepdims=True)
        best_val = max_cos(X_best)
    n, d = 176, 9
    sigmas = [0.008, 0.015, 0.025, 0.04, 0.065, 0.10, 0.02, 0.05]
    k_sched = [(40.0, 120.0), (80.0, 200.0), (120.0, 280.0), (60.0, 160.0)]
    lr_sched = [0.0025, 0.004, 0.0015, 0.003]
    steps_sched = [900, 1300, 1600, 1100]
    trial = 0
    # precompute contacts for weighting (update when best improves)
    Gbest = X_best @ X_best.T
    np.fill_diagonal(Gbest, -2.0)
    contacts = np.sum(Gbest > best_val - 1e-7, axis=1).astype(float)
    while time.time() < SEARCH_END:
        trial += 1
        if HARD_END - time.time() < 3.0:
            break
        # big reshuffle every 17th trial
        X_try = X_best.copy()
        if trial % 17 == 0:
            idx = rng.choice(n, size=8, replace=False)
            R = rng.standard_normal((8, d))
            R /= np.linalg.norm(R, axis=1, keepdims=True)
            X_try[idx] = R
        else:
            sigma = sigmas[(trial - 1) % len(sigmas)]
            # contact-weighted magnitudes: jammed points get bigger kick
            mu = contacts.mean()
            sd = contacts.std() + 1e-9
            w = 1.0 + 0.7 * np.clip((contacts - mu) / (sd + 1.0), -1.0, 2.0)
            # extra boost for single most jammed point occasionally
            if trial % 5 == 0:
                j = int(np.argmax(contacts))
                w[j] *= 2.5
            noise = rng.standard_normal((n, d)) * (sigma * w[:, None])
            # project to tangent to stay on sphere
            dn = np.sum(noise * X_best, axis=1, keepdims=True)
            noise = noise - dn * X_best
            X_try = X_best + noise
            X_try /= np.linalg.norm(X_try, axis=1, keepdims=True)
        ki = k_sched[(trial - 1) % len(k_sched)]
        lr0 = lr_sched[(trial - 1) % len(lr_sched)]
        steps = steps_sched[(trial - 1) % len(steps_sched)]
        # soft quench with its own mini-deadline
        q_dead = min(time.time() + 1.2, SEARCH_END)
        X_q = soft_quench(X_try, ki[0], ki[1], steps, lr0, q_dead)
        try:
            vq = max_cos(X_q)
        except Exception:
            continue
        if vq < best_val - 1e-12:
            # direct win without LP: snap with LP to squeeze further
            lp_dead = min(time.time() + 2.0, HARD_END - 1.0)
            X_p = lp_polish_deadline(X_q, lp_dead)
            try:
                vp = max_cos(X_p)
            except Exception:
                continue
            if vp < best_val:
                X_best = X_p; best_val = vp
                Gbest = X_best @ X_best.T
                np.fill_diagonal(Gbest, -2.0)
                contacts = np.sum(Gbest > best_val - 1e-7, axis=1).astype(float)
            continue
        # if promising, LP polish
        # threshold looser for small-sigma trials
        thresh = best_val + (0.004 if trial % 17 != 0 else 0.015)
        if vq < thresh:
            lp_dead = min(time.time() + 1.6, SEARCH_END + 2.0, HARD_END - 1.0)
            if lp_dead - time.time() < 0.3:
                continue
            X_p = lp_polish_deadline(X_q, lp_dead)
            try:
                vp = max_cos(X_p)
            except Exception:
                continue
            if vp < best_val - 1e-12:
                X_best = X_p; best_val = vp
                Gbest = X_best @ X_best.T
                np.fill_diagonal(Gbest, -2.0)
                contacts = np.sum(Gbest > best_val - 1e-7, axis=1).astype(float)
    # final squeeze of remaining time
    if time.time() < HARD_END - 0.5:
        X_best = lp_polish_deadline(X_best, HARD_END - 0.2)
    X_best = X_best / np.linalg.norm(X_best, axis=1, keepdims=True)
    return X_best
