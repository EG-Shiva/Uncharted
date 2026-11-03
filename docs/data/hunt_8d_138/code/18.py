import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def per_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1)


def contact_counts(x, mv, tol=1e-7):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return np.sum(g > mv - tol, axis=1)


def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start >= seconds:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
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


def soft_anneal(x, steps, beta0, beta1, lr, rng, noise_amp=0.0, mu=0.8):
    n, d = x.shape
    vel = np.zeros_like(x)
    for k in range(steps):
        frac = k / max(1, steps - 1)
        beta = beta0 + (beta1 - beta0) * frac
        lr_k = lr * (1.0 - 0.5 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        grad = (W @ x) / rs
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        vel = mu * vel + (1.0 - mu) * gt
        x = x - lr_k * vel
        if noise_amp > 0:
            nz = rng.standard_normal(x.shape)
            nd = np.sum(nz * x, axis=1, keepdims=True)
            nz -= nd * x
            x = x + noise_amp * (1.0 - frac) * nz
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def refine_void(v0, Xrem, beta=120.0, steps=150, lr=0.08):
    v = v0 / (np.linalg.norm(v0) + 1e-30)
    for t in range(steps):
        dots = Xrem @ v
        mv = float(dots.max())
        W = np.exp(beta * (dots - mv))
        S = float(W.sum()) + 1e-300
        g = (W @ Xrem) / S
        gt = g - float(np.dot(g, v)) * v
        st = lr / (1.0 + t * 0.02)
        v = v - st * gt
        v /= (np.linalg.norm(v) + 1e-30)
    return v


def best_void_for_set(Xrem, rng, n_sample=20000, beta=120.0):
    d = Xrem.shape[1]
    cand = rng.standard_normal((n_sample, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    smax = (cand @ Xrem.T).max(axis=1)
    bi = int(np.argmin(smax))
    v0 = cand[bi]
    try:
        v0 = refine_void(v0, Xrem, beta=beta, steps=150, lr=0.08)
    except Exception:
        pass
    return v0


def recenter_rattlers(x, mv, rng):
    # move 0-contact points to void centers without worsening mv
    try:
        cc = contact_counts(x, mv, tol=1e-7)
    except Exception:
        return x
    idx = np.where(cc == 0)[0]
    if len(idx) == 0:
        return x
    # random order
    order = idx.copy()
    rng.shuffle(order)
    for vi in order:
        Xrem = np.delete(x, vi, axis=0)
        v0 = best_void_for_set(Xrem, rng, n_sample=12000, beta=120.0)
        # check it does not worsen global max
        dots = Xrem @ v0
        if float(dots.max()) < mv - 1e-12:
            x[vi] = v0
        # else keep old (still rattler)
    return x


def solve():
    t0 = time.time()
    T_LIMIT = 170.0
    rng = np.random.default_rng(0)

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (138, 8):
                a /= np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if len(cands) == 0:
        a = rng.standard_normal((138, 8))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    best = None
    best_val = np.inf
    for c in cands:
        v = max_cos(c)
        if v < best_val:
            best_val = v
            best = c.copy()

    try:
        cc = contact_counts(best, best_val)
    except Exception:
        cc = np.ones(138)

    trial = 0
    sigmas = [0.30, 0.18, 0.10, 0.06, 0.035, 0.018, 0.010]
    # quick initial recenter to explore flat manifold (no score change)
    try:
        if time.time() - t0 < 150:
            best = recenter_rattlers(best.copy(), best_val, rng)
            best /= np.linalg.norm(best, axis=1, keepdims=True)
            best_val = max_cos(best)
    except Exception:
        pass

    while time.time() - t0 < 122.0:
        trial += 1
        tl = T_LIMIT - (time.time() - t0)
        if tl < 14.0:
            break
        if trial % 6 == 1:
            try:
                cc = contact_counts(best, best_val)
            except Exception:
                pass
        mode = trial % 5
        x_try = None
        try:
            if mode == 0 or mode == 1:
                # Gaussian basin hop (full or jammed-biased subset)
                sigma = float(sigmas[(trial - 1) % len(sigmas)]) * (0.7 + 0.6 * rng.random())
                if trial % 2 == 0:
                    x_try = best + sigma * rng.standard_normal(best.shape)
                else:
                    k = int(10 + rng.integers(0, 16))
                    w = (cc.astype(float) + 1.0)
                    w /= w.sum()
                    idx = rng.choice(138, size=k, replace=False, p=w)
                    x_try = best + 0.015 * rng.standard_normal(best.shape)
                    x_try[idx] += (sigma * 2.0) * rng.standard_normal((k, 8))
                x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
                beta0 = 10.0 + rng.random() * 20.0
                beta1 = 80.0 + rng.random() * 140.0
                lr = 0.020 + rng.random() * 0.025
                steps = 1800 if (time.time() - t0 < 90) else 1200
                noise_amp = sigma * 0.05 * rng.random()
                x_try = soft_anneal(x_try, steps, beta0, beta1, lr, rng,
                                    noise_amp=noise_amp, mu=0.8)
            elif mode == 2 or mode == 3:
                # excision - relax - reinsert
                pm = per_max(best)
                # victims: mostly worst per_max
                order = np.argsort(-pm)
                k = int(rng.choice([1, 2, 2, 3, 3, 4, 5]))
                # mix worst + random jammed
                pool = order[:12].tolist()
                victims = []
                for _ in range(k):
                    if rng.random() < 0.75 and len(pool) > 0:
                        j = int(rng.integers(0, len(pool)))
                        victims.append(pool.pop(j))
                    else:
                        # random jammed-biased
                        w = (cc.astype(float) + 1.0)
                        w /= w.sum()
                        victims.append(int(rng.choice(138, p=w)))
                victims = list(dict.fromkeys(victims))
                k = len(victims)
                mask = np.ones(138, dtype=bool)
                mask[victims] = False
                Xrem = best[mask].copy()
                # relax thinner core briefly
                beta0 = 15.0 + rng.random() * 15.0
                beta1 = 100.0 + rng.random() * 100.0
                lr = 0.02 + rng.random() * 0.02
                rsteps = 600 + int(rng.integers(0, 500))
                Xrem = soft_anneal(Xrem, rsteps, beta0, beta1, lr, rng,
                                   noise_amp=0.0, mu=0.8)
                # optional micro LP on core if time allows (skip often for speed)
                # reinsert greedily
                full = Xrem
                for _ in range(k):
                    v0 = best_void_for_set(full, rng, n_sample=16000,
                                           beta=100.0 + rng.random() * 80.0)
                    full = np.vstack([full, v0.reshape(1, 8)])
                # need to interleave? just anneal full set to blend seam
                full = soft_anneal(full, 700 + int(rng.integers(0, 400)),
                                   20.0, 140.0 + rng.random() * 60.0,
                                   0.02 + rng.random() * 0.015, rng,
                                   noise_amp=0.0, mu=0.8)
                x_try = full
                x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            else:
                # mode 4: rattler reshuffle + void centering + light anneal
                # or occasional far restart
                if trial % 20 == 0:
                    # far random restart (different basin)
                    x_try = rng.standard_normal(best.shape)
                    x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
                    x_try = soft_anneal(x_try, 2200, 8.0, 160.0, 0.03, rng,
                                        noise_amp=0.0, mu=0.8)
                else:
                    x_try = best.copy()
                    # relocate rattlers to fresh random voids
                    try:
                        ridx = np.where(cc == 0)[0]
                    except Exception:
                        ridx = np.array([], dtype=int)
                    if len(ridx) > 0:
                        # randomize half of rattlers coarsely
                        nr = max(1, len(ridx) // 2)
                        pick = rng.choice(ridx, size=nr, replace=False)
                        for vi in pick:
                            Xrem = np.delete(x_try, vi, axis=0)
                            v0 = best_void_for_set(Xrem, rng, n_sample=8000, beta=100.0)
                            x_try[vi] = v0
                    # small kick to jammed part
                    x_try = x_try + 0.02 * rng.standard_normal(x_try.shape)
                    x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
                    x_try = soft_anneal(x_try, 900, 20.0, 150.0, 0.025, rng,
                                        noise_amp=0.0, mu=0.8)
        except Exception:
            continue
        try:
            v_try = max_cos(x_try)
        except Exception:
            continue
        if v_try < best_val + 0.012:
            polish_sec = 1.4
            if v_try < best_val:
                polish_sec = 2.2
            tl2 = T_LIMIT - (time.time() - t0)
            polish_sec = min(polish_sec, max(0.3, tl2 - 10.0))
            if polish_sec < 0.25:
                # no time for snap; direct accept check
                if v_try < best_val - 1e-12:
                    best, best_val = x_try, v_try
                    try:
                        cc = contact_counts(best, best_val)
                    except Exception:
                        pass
                continue
            try:
                x_pol = lp_polish(x_try, polish_sec)
                v_pol = max_cos(x_pol)
            except Exception:
                continue
            if v_pol < best_val - 1e-12:
                best = x_pol
                best_val = v_pol
                try:
                    cc = contact_counts(best, best_val)
                except Exception:
                    pass
                # exploit flat manifold: recenter rattlers for free
                try:
                    nb = recenter_rattlers(best.copy(), best_val, rng)
                    nb /= np.linalg.norm(nb, axis=1, keepdims=True)
                    nv = max_cos(nb)
                    if nv <= best_val + 1e-12:
                        best = nb
                        best_val = nv
                except Exception:
                    pass

    tl = T_LIMIT - (time.time() - t0)
    if tl > 2.5:
        final_sec = max(1.0, min(30.0, tl - 1.5))
        try:
            best = lp_polish(best, final_sec)
            best_val = max_cos(best)
        except Exception:
            pass
    while time.time() - t0 < T_LIMIT - 1.0:
        try:
            sigma = 0.004
            x_try = best + sigma * rng.standard_normal(best.shape)
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            x_pol = lp_polish(x_try, 0.7)
            v_pol = max_cos(x_pol)
            if v_pol < best_val:
                best, best_val = x_pol, v_pol
            else:
                break
        except Exception:
            break
    best /= np.linalg.norm(best, axis=1, keepdims=True)
    return best
