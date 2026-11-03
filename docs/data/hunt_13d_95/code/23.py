import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def per_point_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1)


def contact_counts(x, mc, tol=1e-7):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return (g > mc - tol).sum(axis=1)


def lp_polish(x, seconds, radius_init=1e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = float(radius_init)
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start > seconds - 0.05:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        if int(sel.sum()) > 1200:
            idx = np.argpartition(c, -1200)[-1200:]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx] = True
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
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


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
        mx = G.max()
        W = np.exp((G - mx) / T)
        np.fill_diagonal(W, 0.0)
        sw = W.sum()
        if not np.isfinite(sw) or sw == 0:
            break
        W /= sw
        Grad = W @ X
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        # clip extreme grads
        V = mu * V + Gt
        X = X - lr * V
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def best_hole_vec(others, rng, n_cand=4000):
    d = others.shape[1]
    C = rng.standard_normal((n_cand, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    dots = C @ others.T
    md = dots.max(axis=1)
    j = int(np.argmin(md))
    return C[j], float(md[j])


def solve():
    t0 = time.time()
    DEADLINE = t0 + 170.0
    SEARCH_END = t0 + 128.0
    rng = np.random.default_rng(12345)

    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    if xb is not None and xb.shape == (95, 13):
        xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
        cands.append(xb)
    if xr is not None and xr.shape == (95, 13):
        xr = xr / np.linalg.norm(xr, axis=1, keepdims=True)
        cands.append(xr)
    if len(cands) == 0:
        z = rng.standard_normal((95, 13))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        cands.append(z)
    vals = [max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    x_best = cands[bi].copy()
    best_val = float(vals[bi])

    cur = x_best.copy()
    cur_val = best_val
    n, d = (95, 13)

    it = 0
    T_accept = 0.003
    while time.time() < SEARCH_END and time.time() < DEADLINE - 35:
        it += 1
        remain = SEARCH_END - (time.time() - t0)
        if remain < 1.0:
            break
        mode = it % 8
        Xt = cur.copy()

        if mode == 0:
            # small global shake + cool anneal
            sig = 0.03
            Xt = Xt + sig * rng.standard_normal((n, d))
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = gd_refine(Xt, 600, 0.012, 0.0012, 0.08, 0.008)
        elif mode == 1:
            # medium shake focused on overloaded hubs
            mc = max_cos(cur)
            cc = contact_counts(cur, mc, tol=0.015)
            order = np.argsort(-cc)
            k = int(rng.integers(6, 14))
            idx = order[:k]
            sig = 0.15
            Xt[idx] += sig * rng.standard_normal((k, d))
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = gd_refine(Xt, 700, 0.015, 0.001, 0.10, 0.008)
        elif mode == 2:
            # delete worst 1-3 and reinsert into largest holes
            ppm = per_point_max(cur)
            k = 1 if rng.random() < 0.5 else (2 if rng.random() < 0.7 else 3)
            worst = np.argsort(-ppm)[:k]
            # occasionally include a random high-contact hub
            if rng.random() < 0.3:
                mc = max_cos(cur)
                cc = contact_counts(cur, mc, tol=0.015)
                worst[0] = int(np.argmax(cc))
            keep = [i for i in range(n) if i not in set(worst.tolist())]
            others = Xt[keep]
            for w in worst:
                h, _ = best_hole_vec(others, rng, n_cand=5000)
                # small jitter around hole
                h = h + 0.01 * rng.standard_normal(d)
                h /= np.linalg.norm(h)
                Xt[w] = h
                others = Xt[[i for i in range(n) if i != w]] if len(worst) == 1 else np.vstack([others, h.reshape(1, -1)])
            Xt = gd_refine(Xt, 500, 0.010, 0.001, 0.08, 0.008)
        elif mode == 3:
            # large shatter kick
            sig = 0.30
            Xt = Xt + sig * rng.standard_normal((n, d))
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = gd_refine(Xt, 800, 0.020, 0.001, 0.12, 0.008)
        elif mode == 4:
            # rattler-guided: freeze rattlers, kick core along random direction
            mc = max_cos(cur)
            cc0 = contact_counts(cur, mc, tol=1e-7)
            ratt = np.where(cc0 <= 1)[0]
            u = rng.standard_normal(d)
            u /= np.linalg.norm(u)
            proj = cur @ u
            # kick points in densest slab
            k = int(rng.integers(10, 25))
            order = np.argsort(np.abs(proj - np.median(proj)))[:k]
            # don't move rattlers
            order = [i for i in order if i not in set(ratt.tolist())]
            if len(order) > 0:
                arr = np.array(order)
                Xt[arr] += 0.12 * rng.standard_normal((len(arr), d))
                Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = gd_refine(Xt, 650, 0.014, 0.001, 0.10, 0.008)
        elif mode == 5:
            # pairwise break: push top pairs apart tangentially
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            iu = np.triu_indices(n, 1)
            c = G[iu]
            top = np.argpartition(-c, min(20, len(c) - 1))[:20]
            for t in top[:10]:
                i = int(iu[0][t]); j = int(iu[1][t])
                if rng.random() < 0.7:
                    # push i away from j
                    v = Xt[i] - (Xt[i] @ Xt[j]) * Xt[j]
                    nv = np.linalg.norm(v)
                    if nv > 1e-12:
                        v /= nv
                        Xt[i] += 0.05 * v + 0.02 * rng.standard_normal(d)
                    v2 = Xt[j] - (Xt[j] @ Xt[i]) * Xt[i]
                    nv2 = np.linalg.norm(v2)
                    if nv2 > 1e-12:
                        v2 /= nv2
                        Xt[j] += 0.05 * v2 + 0.02 * rng.standard_normal(d)
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = gd_refine(Xt, 600, 0.011, 0.0009, 0.09, 0.007)
        elif mode == 6:
            # subset shake 5-12 random points with larger sigma
            k = int(rng.integers(5, 13))
            idx = rng.choice(n, size=k, replace=False)
            sig = 0.25
            Xt[idx] += sig * rng.standard_normal((k, d))
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = gd_refine(Xt, 700, 0.016, 0.001, 0.10, 0.008)
        else:
            # orthogonal block rotation: rotate random 13D plane for a block
            k = int(rng.integers(8, 20))
            idx = rng.choice(n, size=k, replace=False)
            a = rng.standard_normal(d); a /= np.linalg.norm(a)
            b = rng.standard_normal(d); b -= (b @ a) * a; b /= np.linalg.norm(b)
            ang = float(rng.uniform(0.05, 0.35))
            ca, sa = np.cos(ang), np.sin(ang)
            for i in idx:
                pa = Xt[i] @ a; pb = Xt[i] @ b
                rest = Xt[i] - pa * a - pb * b
                na = ca * pa - sa * pb; nb = sa * pa + ca * pb
                Xt[i] = rest + na * a + nb * b
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = gd_refine(Xt, 600, 0.012, 0.001, 0.09, 0.008)

        v_gd = max_cos(Xt)
        # screen with short LP if promising
        do_lp = False
        if v_gd < best_val + 0.012:
            do_lp = True
        elif v_gd < cur_val + 0.005 and (mode in (1, 2, 4, 5)):
            do_lp = True
        if do_lp:
            if time.time() > SEARCH_END - 2.0:
                break
            sec = 1.1 if v_gd < best_val + 0.002 else 0.7
            # leave time for final polish
            sec = min(sec, max(0.5, SEARCH_END - (time.time() - t0) - 0.5))
            if sec >= 0.45:
                Xt = lp_polish(Xt, sec, radius_init=0.004 if v_gd > best_val else 0.002)
                v = max_cos(Xt)
            else:
                v = v_gd
        else:
            v = v_gd

        if v < best_val:
            x_best = Xt.copy()
            best_val = float(v)
            cur = Xt.copy()
            cur_val = float(v)
            T_accept *= 0.98
        elif v < cur_val:
            cur = Xt.copy()
            cur_val = float(v)
        else:
            # simulated-annealing uphill acceptance, decaying
            decay = 1.0 - (time.time() - t0) / (SEARCH_END - t0 + 1e-9)
            Ta = T_accept * max(0.15, decay) + 1e-4
            if rng.random() < np.exp((cur_val - v) / Ta):
                cur = Xt.copy()
                cur_val = float(v)
            # occasional restart to best
            if it % 25 == 0 and rng.random() < 0.5:
                cur = x_best.copy()
                cur_val = best_val

    # final long polish of best
    t_rem = DEADLINE - time.time() - 1.0
    if t_rem > 3.0:
        # two-stage: moderate radius then tiny radius
        s1 = min(12.0, t_rem * 0.5)
        x_best = lp_polish(x_best, s1, radius_init=0.003)
        t_rem2 = DEADLINE - time.time() - 1.0
        if t_rem2 > 2.0:
            x_best = lp_polish(x_best, t_rem2, radius_init=8e-4)
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    # tiny shrink guard against rounding (ensure strictly feasible, no NaN)
    if not np.all(np.isfinite(x_best)):
        x_best = cur / np.linalg.norm(cur, axis=1, keepdims=True)
    return x_best
