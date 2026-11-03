import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def per_point_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)


def contact_counts(X, mc, tol=0.02):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return (G > mc - tol).sum(axis=1)


def gd_refine(X, steps, T0, T1, lr0, lr1, mu=0.85):
    n, d = X.shape
    V = np.zeros_like(X)
    lT0 = np.log(T0); lT1 = np.log(T1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    for s in range(steps):
        f = s / max(1, steps - 1)
        T = float(np.exp(lT0 + (lT1 - lT0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        mx = float(G.max())
        W = np.exp((G - mx) / T)
        np.fill_diagonal(W, 0.0)
        sw = W.sum()
        if not np.isfinite(sw) or sw == 0:
            continue
        W /= sw
        Grad = W @ X
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        V = mu * V + Gt
        X = X - lr * V
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def refine_hole(y, others, steps=200, T=0.005, lr=0.25, mu=0.5):
    v = np.zeros_like(y)
    for s in range(steps):
        dots = others @ y
        mx = float(dots.max())
        w = np.exp((dots - mx) / T)
        sw = w.sum()
        if sw == 0 or not np.isfinite(sw):
            break
        w /= sw
        g = w @ others
        gt = g - float(g @ y) * y
        v = mu * v + gt
        y = y - lr * v
        nrm = float(np.linalg.norm(y))
        if nrm == 0 or not np.isfinite(nrm):
            break
        y /= nrm
    return y


def best_hole_for(others, rng, n_sample=8000, n_refine=2):
    d = others.shape[1]
    C = rng.standard_normal((n_sample, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    dots = C @ others.T
    md = dots.max(axis=1)
    idx = np.argpartition(md, n_refine)[:n_refine]
    best_y = C[idx[0]]
    best_v = float(md[idx[0]])
    for j in idx:
        y0 = C[j].copy()
        y1 = refine_hole(y0, others, steps=250, T=0.004, lr=0.25, mu=0.5)
        v1 = float((others @ y1).max())
        if v1 < best_v:
            best_v = v1
            best_y = y1
    return best_y, best_v


def lp_polish(x, seconds, radius_init=1e-3, cap=2000, deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = float(radius_init)
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if deadline is not None and time.time() > deadline:
            break
        if time.time() - start > seconds - 0.05:
            break
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        thresh = m - max(20 * radius, 1e-9)
        sel = c > thresh
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > cap:
            kth = np.argpartition(c, -cap)[-cap:]
            mask = np.zeros_like(sel, dtype=bool)
            mask[kth] = True
            sel = mask
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
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
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
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    BUDGET = 172.0
    deadline = t0 + BUDGET
    EXPL_END = t0 + 122.0

    rng = np.random.default_rng(0)

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

    n, d = 95, 13
    cur = x_best.copy()
    cur_val = best_val

    trial = 0
    while time.time() < EXPL_END and time.time() < deadline - 35:
        trial += 1
        remain = EXPL_END - (time.time() - t0)
        if remain < 0.5:
            break
        mode = trial % 7
        Xt = cur.copy()
        if mode == 0:
            sig = 0.008 + 0.004 * rng.random()
            Xt = Xt + sig * rng.standard_normal((n, d))
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = gd_refine(Xt, 500, 0.008, 0.0008, 0.06, 0.006, mu=0.85)
        elif mode == 1:
            sig = 0.03 + 0.03 * rng.random()
            Xt = Xt + sig * rng.standard_normal((n, d))
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = gd_refine(Xt, 700, 0.012, 0.001, 0.08, 0.007, mu=0.88)
        elif mode == 2:
            ksub = int(rng.integers(6, 16))
            idx = rng.choice(n, size=ksub, replace=False)
            sig = 0.10 + 0.15 * rng.random()
            Xt[idx] = Xt[idx] + sig * rng.standard_normal((ksub, d))
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = gd_refine(Xt, 700, 0.012, 0.0009, 0.09, 0.007, mu=0.88)
        elif mode == 3:
            # single worst-point reinsertion
            ppm = per_point_max(cur)
            order = np.argsort(-ppm)
            k = 1
            # mostly worst, sometimes random among top-10
            if rng.random() < 0.7:
                victims = order[:1]
            else:
                victims = rng.choice(order[:12], size=1, replace=False)
            keep = np.ones(n, dtype=bool); keep[victims] = False
            others = cur[keep]
            for vi in victims:
                y, _ = best_hole_for(others, rng, n_sample=8000, n_refine=2)
                Xt[vi] = y
                others = np.vstack([others, y.reshape(1, -1)])
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = gd_refine(Xt, 600, 0.010, 0.0008, 0.07, 0.006, mu=0.86)
        elif mode == 4:
            # multi reinsertion of overloaded points
            cc = contact_counts(cur, cur_val, tol=0.02)
            # weight by contacts
            w = cc.astype(float) + 1.0
            w /= w.sum()
            k = int(rng.integers(2, 5))
            if rng.random() < 0.5:
                victims = rng.choice(n, size=k, replace=False, p=w)
            else:
                ppm = per_point_max(cur)
                order = np.argsort(-ppm)
                victims = np.concatenate([order[:1], rng.choice(order[1:20], size=k-1, replace=False)])
            keep = np.ones(n, dtype=bool); keep[victims] = False
            others = cur[keep]
            for vi in victims:
                y, _ = best_hole_for(others, rng, n_sample=6000, n_refine=2)
                # small jitter to avoid exact repeat
                y = y + 0.005 * rng.standard_normal(d)
                y /= np.linalg.norm(y)
                Xt[vi] = y
                others = np.vstack([others, y.reshape(1, -1)])
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = gd_refine(Xt, 800, 0.013, 0.0009, 0.09, 0.006, mu=0.88)
        elif mode == 5:
            # low-contact / rattler relocation + neighbour shake
            cc = contact_counts(cur, cur_val, tol=1e-7)
            low = np.where(cc <= 5)[0]
            if len(low) == 0:
                low = np.argsort(cc)[:5]
            vi = int(rng.choice(low))
            keep = np.ones(n, dtype=bool); keep[vi] = False
            others = cur[keep]
            y, _ = best_hole_for(others, rng, n_sample=10000, n_refine=2)
            Xt[vi] = y
            # shake its new neighbours slightly
            dots = others @ y
            nb = np.argsort(-dots)[:8]
            # map nb indices in `others` to full indices: need careful; just shake random 8
            ridx = rng.choice(n, size=8, replace=False)
            Xt[ridx] += 0.02 * rng.standard_normal((8, d))
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = gd_refine(Xt, 600, 0.009, 0.0007, 0.07, 0.006, mu=0.86)
        else:
            # coherent cluster move: pick worst point and its neighbours, apply common shift
            ppm = per_point_max(cur)
            s = int(np.argmax(ppm))
            G = cur @ cur[s]
            G[s] = -2
            nb = np.argsort(-G)[:int(rng.integers(5, 11))]
            shift = rng.standard_normal(d)
            shift -= float(shift @ cur[s]) * cur[s]
            shift /= (np.linalg.norm(shift) + 1e-12)
            amp = 0.05 + 0.10 * rng.random()
            Xt[nb] += amp * shift[None, :]
            Xt[s] -= amp * 0.7 * shift
            Xt += 0.01 * rng.standard_normal((n, d))
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = gd_refine(Xt, 800, 0.014, 0.001, 0.10, 0.007, mu=0.88)

        v = max_cos(Xt)
        # acceptance
        if v < best_val - 1e-12:
            # promising record: short LP squeeze
            t_left = deadline - time.time()
            if t_left < 8:
                x_best, best_val = Xt, v
                cur, cur_val = Xt, v
            else:
                Xs = lp_polish(Xt, min(2.5, max(0.8, t_left * 0.05)), radius_init=0.006)
                vs = max_cos(Xs)
                if vs < best_val:
                    x_best, best_val = Xs, vs
                    cur, cur_val = Xs, vs
                elif vs < cur_val + 0.002:
                    cur, cur_val = Xs, vs
                else:
                    # keep raw if LP failed but raw is better
                    if v < best_val:
                        x_best, best_val = Xt, v
                    cur, cur_val = Xt, v
        elif v < cur_val:
            cur, cur_val = Xt, v
            # occasional LP to check hidden improvement when close to best
            if v < best_val + 0.0015 and (trial % 6 == 0) and time.time() < deadline - 40:
                Xs = lp_polish(Xt, 1.2, radius_init=0.005)
                vs = max_cos(Xs)
                if vs < best_val - 1e-12:
                    x_best, best_val = Xs, vs
                    cur, cur_val = Xs, vs
        else:
            # allow uphill drift to change basin
            if v < cur_val + 0.008 and rng.random() < 0.15:
                cur, cur_val = Xt, v
            # periodic restart from best to avoid wandering too far
            if trial % 25 == 0:
                cur, cur_val = x_best.copy(), best_val

    # final squeeze
    t_left = deadline - time.time()
    if t_left > 2:
        x_best = lp_polish(x_best, max(2.0, t_left - 1.0), radius_init=0.004)
        best_val = max_cos(x_best)
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
