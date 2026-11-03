import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def per_point_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1)


def gd_refine(X, steps, T0, T1, lr0, lr1, mu=0.9):
    n, d = X.shape
    V = np.zeros_like(X)
    lT0 = np.log(T0)
    lT1 = np.log(T1)
    ll0 = np.log(lr0)
    ll1 = np.log(lr1)
    for s in range(steps):
        f = s / max(1, steps - 1)
        T = float(np.exp(lT0 + (lT1 - lT0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        mx = float(G.max())
        W = np.exp((G - mx) / T)
        np.fill_diagonal(W, 0.0)
        sw = float(W.sum())
        if not np.isfinite(sw) or sw <= 0:
            break
        W /= sw
        Grad = W @ X
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        # clip extreme
        nv = float(np.linalg.norm(Gt))
        if not np.isfinite(nv) or nv == 0:
            break
        if nv > 1.0:
            Gt *= 1.0 / nv
        V = mu * V + Gt
        X = X - lr * V
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def refine_hole(v, others, steps=80, T=0.004, lr=0.15):
    v = v / np.linalg.norm(v)
    for _ in range(steps):
        dots = others @ v
        mx = float(dots.max())
        w = np.exp((dots - mx) / T)
        sw = float(w.sum())
        if not np.isfinite(sw) or sw <= 0:
            break
        w /= sw
        g = w @ others
        gt = g - float(g @ v) * v
        nrm = float(np.linalg.norm(gt))
        if nrm < 1e-12 or not np.isfinite(nrm):
            break
        gt /= nrm
        v = v - lr * gt
        v /= np.linalg.norm(v)
        lr *= 0.98
    return v / np.linalg.norm(v)


def best_hole(others, rng, n_samples=4000, refine=True):
    n, d = others.shape
    C = rng.standard_normal((n_samples, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    dots = C @ others.T
    md = dots.max(axis=1)
    j = int(np.argmin(md))
    v = C[j]
    if refine:
        try:
            v = refine_hole(v, others)
        except Exception:
            pass
    return v


def lp_polish(x, seconds, radius_init=5e-3, cap=2000, deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = float(radius_init)
    start = time.time()
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    while True:
        now = time.time()
        if now - start > seconds:
            break
        if deadline is not None and now > deadline:
            break
        if radius < 1e-14:
            break
        if seconds - (now - start) < 0.05:
            break
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        cnt = int(sel.sum())
        if cnt > cap:
            idx = np.argpartition(c, -cap)[-cap:]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx] = True
            sel = mask
            cnt = cap
        if cnt == 0:
            break
        I = iu[0][sel]
        J = iu[1][sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + ar, J[:, None] * d + ar,
                               np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
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


def solve():
    t0 = time.time()
    BUDGET = 172.0
    EXPL_END = t0 + 128.0
    rng = np.random.default_rng(7)

    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None

    inits = []
    if xb is not None and xb.shape == (95, 13):
        inits.append(xb / np.linalg.norm(xb, axis=1, keepdims=True))
    if xr is not None and xr.shape == (95, 13):
        inits.append(xr / np.linalg.norm(xr, axis=1, keepdims=True))
    if not inits:
        z = rng.standard_normal((95, 13))
        inits.append(z / np.linalg.norm(z, axis=1, keepdims=True))
    vals = [max_cos(c) for c in inits]
    bi = int(np.argmin(vals))
    x_best = inits[bi].copy()
    best_val = float(vals[bi])

    # light initial squeeze to ensure high precision baseline
    if time.time() - t0 < 20:
        try:
            x_best = lp_polish(x_best, seconds=2.0, radius_init=1e-3)
            best_val = max_cos(x_best)
        except Exception:
            pass

    cur = x_best.copy()
    cur_val = best_val
    n, d = 95, 13
    elites = [x_best.copy()]
    elite_vals = [best_val]

    sigmas = [0.01, 0.02, 0.04, 0.08, 0.15, 0.30, 0.50]
    it = 0
    while time.time() < EXPL_END and time.time() - t0 < BUDGET - 40:
        it += 1
        # pick base: mostly best, sometimes cur or other elite
        u0 = rng.random()
        if u0 < 0.55:
            base = x_best
            base_val = best_val
        elif u0 < 0.80:
            base = cur
            base_val = cur_val
        else:
            ei = int(rng.integers(0, len(elites)))
            base = elites[ei]
            base_val = elite_vals[ei]
        Xt = base.copy()
        mode = rng.random()
        if mode < 0.38:
            # global shake + GD
            sigma = sigmas[int(rng.integers(0, len(sigmas)))]
            if rng.random() < 0.30:
                ksub = int(rng.integers(5, 21))
                idx = rng.choice(n, size=ksub, replace=False)
                Xt[idx] += sigma * 2.0 * rng.standard_normal((ksub, d))
            else:
                Xt += sigma * rng.standard_normal((n, d))
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            T0 = float(rng.choice([0.012, 0.018, 0.025]))
            T1 = float(rng.choice([0.0008, 0.0012, 0.002]))
            lr0 = float(rng.choice([0.08, 0.12, 0.16]))
            lr1 = float(rng.choice([0.005, 0.01, 0.02]))
            steps = int(rng.integers(350, 800))
            Xt = gd_refine(Xt, steps, T0, T1, lr0, lr1, mu=0.9)
        elif mode < 0.62:
            # subset strong shake + GD
            ksub = int(rng.integers(3, 12))
            ppm = per_point_max(Xt)
            # bias to worst points
            if rng.random() < 0.6:
                idx = np.argsort(-ppm)[:ksub]
                if rng.random() < 0.4:
                    idx = idx.copy()
                    idx[-1] = int(rng.integers(0, n))
            else:
                idx = rng.choice(n, size=ksub, replace=False)
            sigma = float(rng.choice([0.1, 0.2, 0.35, 0.6]))
            Xt[idx] += sigma * rng.standard_normal((ksub, d))
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = gd_refine(Xt, int(rng.integers(300, 650)), 0.015, 0.001, 0.10, 0.008, mu=0.88)
        elif mode < 0.88:
            # kill k worst / random / rattlers and hole-reinsert
            if rng.random() < 0.5:
                k = 1
            elif rng.random() < 0.7:
                k = 2
            else:
                k = int(rng.integers(3, 6))
            ppm = per_point_max(Xt)
            # contact counts approx
            G = Xt @ Xt.T
            np.fill_diagonal(G, -2.0)
            mc = float(G.max())
            cc = (G > mc - 0.02).sum(axis=1)
            pick = rng.random()
            if pick < 0.55:
                idx = np.argsort(-cc)[:k]
                if k > 1 and rng.random() < 0.5:
                    idx = idx.copy()
                    idx[-1] = int(rng.integers(0, n))
            elif pick < 0.80:
                idx = rng.choice(n, size=k, replace=False)
            else:
                idx = np.argsort(cc)[:k]
            # also sometimes include worst per-point max
            if rng.random() < 0.3:
                w = int(np.argmax(ppm))
                idx = np.array(list(idx) + [w])[:k] if k > 1 else np.array([w])
            for j in idx:
                if rng.random() < 0.8:
                    mask = np.ones(n, dtype=bool)
                    mask[j] = False
                    others = Xt[mask]
                    ns = 5000 if k == 1 else 2500
                    Xt[j] = best_hole(others, rng, n_samples=ns, refine=True)
                else:
                    v = rng.standard_normal(d)
                    v /= np.linalg.norm(v)
                    Xt[j] = v
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            # short GD to settle, freezing others a bit? just full GD short
            Xt = gd_refine(Xt, int(rng.integers(250, 550)), 0.012, 0.001, 0.08, 0.008, mu=0.9)
        else:
            # very large global restart + GD (basin jump)
            sigma = float(rng.choice([0.4, 0.6, 0.9]))
            Xt = Xt + sigma * rng.standard_normal((n, d))
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = gd_refine(Xt, int(rng.integers(500, 900)), 0.020, 0.001, 0.14, 0.008, mu=0.9)

        v_fast = max_cos(Xt)
        # screen: only LP if within margin
        margin = 0.012 if (time.time() - t0 < 60) else 0.006
        if v_fast > best_val + margin and v_fast > cur_val + 0.008:
            # occasional sideways drift
            if rng.random() < 0.02:
                cur = Xt.copy()
                cur_val = v_fast
            continue

        # short LP polish
        remain = EXPL_END - (time.time() - t0)
        if remain < 1.0:
            break
        ps = min(1.1, max(0.55, remain * 0.08))
        # larger radius if far
        rinit = 0.010 if v_fast > best_val + 0.002 else 0.005
        try:
            Xp = lp_polish(Xt, seconds=ps, radius_init=rinit)
        except Exception:
            continue
        v = max_cos(Xp)
        if v < best_val - 1e-12:
            x_best = Xp.copy()
            best_val = v
            cur = Xp.copy()
            cur_val = v
            elites.append(Xp.copy())
            elite_vals.append(v)
            # keep top 4
            oi = np.argsort(np.array(elite_vals))
            oi = oi[:4]
            elites = [elites[i] for i in oi]
            elite_vals = [elite_vals[i] for i in oi]
        elif v < cur_val:
            cur = Xp.copy()
            cur_val = v
        else:
            if rng.random() < 0.05 and v < cur_val + 0.008:
                cur = Xp.copy()
                cur_val = v
            if rng.random() < 0.10:
                cur = x_best.copy()
                cur_val = best_val

    # refine elites briefly
    for ei in range(len(elites)):
        if time.time() - t0 > BUDGET - 30:
            break
        try:
            elites[ei] = lp_polish(elites[ei], seconds=2.0, radius_init=2e-3)
            elite_vals[ei] = max_cos(elites[ei])
            if elite_vals[ei] < best_val:
                best_val = elite_vals[ei]
                x_best = elites[ei].copy()
        except Exception:
            pass

    # final long polish
    remain = BUDGET - (time.time() - t0)
    final_sec = float(np.clip(remain - 1.5, 4.0, 32.0))
    if final_sec > 1.0:
        try:
            x_best = lp_polish(x_best, seconds=final_sec * 0.65, radius_init=1e-3, cap=2500,
                               deadline=t0 + BUDGET - 1.0)
            rest = (t0 + BUDGET - 1.0) - time.time()
            if rest > 2.0:
                x_best = lp_polish(x_best, seconds=min(rest, 14.0), radius_init=5e-4, cap=2500,
                                   deadline=t0 + BUDGET - 0.5)
        except Exception:
            pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
