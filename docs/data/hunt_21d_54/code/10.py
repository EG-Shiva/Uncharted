import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def proj_unit(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    x = x / n
    x = np.nan_to_num(x)
    n2 = np.linalg.norm(x, axis=1, keepdims=True)
    n2[n2 == 0] = 1.0
    return x / n2

def lp_polish(x, seconds):
    x = proj_unit(x)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        csv = c[sel]
        if p > 3000:
            idx = np.argpartition(-csv, 3000)[:3000]
            I, J = I[idx], J[idx]
            csv = csv[idx]
            p = 3000
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
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=-csv, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                              method="highs", options={"time_limit": 5.0})
            except Exception:
                res = linprog(obj, A_ub=A_ub, b_ub=-csv, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                              method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = proj_unit(y)
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def contact_info(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    C = (G > m - 1e-9).sum(axis=1)
    return C, m, G

def repulsion_smooth(x, steps, lr):
    y = x.copy()
    n, d = y.shape
    for s in range(steps):
        G = y @ y.T
        np.fill_diagonal(G, -10.0)
        m = G.max()
        T = max(0.005, 0.02 * (1 - s / max(1, steps)))
        W = np.exp((G - m) / T)
        np.fill_diagonal(W, 0.0)
        grad = W @ y
        grad -= (grad * y).sum(axis=1, keepdims=True) * y
        gn = np.linalg.norm(grad, axis=1, keepdims=True) + 1e-12
        grad /= gn
        step = lr * (1 - 0.5 * s / max(1, steps))
        y = y - step * grad
        y = proj_unit(y)
    return y

def refine_single(Xf, y, steps=300, lr0=0.05):
    y = y / (np.linalg.norm(y) + 1e-12)
    lr = lr0
    k = 3
    kk = min(k, Xf.shape[0])
    for _ in range(steps):
        dots = Xf @ y
        idx = np.argpartition(-dots, kk - 1)[:kk]
        T = 0.008
        mm = dots[idx].max()
        w = np.exp((dots[idx] - mm) / T)
        w /= (w.sum() + 1e-16)
        g = w @ Xf[idx]
        g = g - (g @ y) * y
        gn = np.linalg.norm(g) + 1e-12
        g /= gn
        y = y - lr * g
        ny = np.linalg.norm(y)
        if ny < 1e-12:
            break
        y /= ny
        lr *= 0.997
    return y

def best_insertion(Xf, rng, n_sample=15000, refine_steps=250):
    d = Xf.shape[1]
    best_y = None
    best_s = 1e9
    # try antipodes of existing points first
    Y0 = -Xf.copy()
    s0 = (Xf @ Y0.T).max(axis=0)
    i0 = int(np.argmin(s0))
    best_y = Y0[i0].copy()
    best_s = float(s0[i0])
    batch = 5000
    nb = max(1, n_sample // batch)
    for _ in range(nb):
        Y = rng.normal(size=(batch, d))
        Y /= (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12)
        S = Xf @ Y.T
        s = S.max(axis=0)
        i = int(np.argmin(s))
        if float(s[i]) < best_s:
            best_s = float(s[i])
            best_y = Y[i].copy()
    best_y = refine_single(Xf, best_y, steps=refine_steps)
    return best_y

def shake_points(x, rng, strength, n_kick):
    n, d = x.shape
    y = x.copy()
    cnts, _, _ = contact_info(x)
    order_low = np.argsort(cnts)
    order_high = np.argsort(-cnts)
    targets = set()
    targets.update(order_low[:n_kick // 3 + 1].tolist())
    targets.update(order_high[:n_kick // 3 + 1].tolist())
    while len(targets) < n_kick:
        targets.add(int(rng.integers(0, n)))
    for i in list(targets):
        v = rng.normal(size=d)
        v -= (v @ y[i]) * y[i]
        nv = np.linalg.norm(v)
        if nv < 1e-12:
            continue
        v /= nv
        ang = strength * (0.5 + rng.random())
        y[i] = np.cos(ang) * y[i] + np.sin(ang) * v
    noise = rng.normal(size=(n, d)) * (strength * 0.2)
    noise -= (noise * y).sum(axis=1, keepdims=True) * y
    y = y + noise
    return proj_unit(y)

def solve():
    import time as _t
    T0 = _t.time()
    LIMIT = 168.0
    elapsed = lambda: _t.time() - T0
    rng = np.random.default_rng(20261101)
    try:
        x0 = np.load("best.npy")
    except Exception:
        x0 = np.load("record.npy")
    x0 = proj_unit(np.asarray(x0, dtype=float))
    n, d = x0.shape
    # ensure shape (54,21)
    if x0.shape != (54, 21):
        # fallback: random
        Y = rng.normal(size=(54, 21))
        x0 = proj_unit(Y)
        n, d = 54, 21
    best = x0.copy()
    best_val = max_cos(best)
    # initial polish to local optimum
    if elapsed() < LIMIT - 20:
        b = min(8.0, LIMIT - 20 - elapsed())
        if b > 0.5:
            best = lp_polish(best, b)
            best_val = max_cos(best)
    it = 0
    strengths = [0.03, 0.08, 0.18, 0.35, 0.05, 0.12, 0.25]
    while elapsed() < LIMIT - 22:
        it += 1
        time_left = LIMIT - 22 - elapsed()
        if time_left < 2:
            break
        # alternate cavity vs shake
        if it % 2 == 1:
            # cavity move: remove k jammed neighbors
            k = int(rng.integers(2, 5))  # 2..4
            cnts, m, G = contact_info(best)
            order_top = np.argsort(-cnts)
            v0 = int(rng.choice(order_top[:6]))
            nbr = np.argsort(-G[v0])
            # G[v0,v0]=-2 so self is last, safe
            victims = [v0]
            for cand in nbr.tolist():
                if cand != v0 and cand not in victims:
                    victims.append(int(cand))
                if len(victims) >= k:
                    break
            if rng.random() < 0.25:
                victims = rng.choice(n, size=k, replace=False).tolist()
            mask = np.ones(n, dtype=bool)
            mask[np.array(victims, dtype=int)] = False
            Xrem = best[mask].copy()
            # tiny jitter to break symmetry near hole
            jn = rng.normal(size=Xrem.shape) * 5e-4
            jn -= (jn * Xrem).sum(axis=1, keepdims=True) * Xrem
            Xrem = proj_unit(Xrem + jn)
            b1 = min(1.5, time_left * 0.25)
            if b1 > 0.4:
                Xrem = lp_polish(Xrem, b1)
            Xnew = Xrem
            for _ in range(k):
                y_star = best_insertion(Xnew, rng, n_sample=12000, refine_steps=220)
                Xnew = np.vstack([Xnew, y_star])
            Xnew = proj_unit(Xnew)
            Xnew = repulsion_smooth(Xnew, steps=50, lr=0.02)
            b2 = min(4.0, LIMIT - 22 - elapsed())
            if b2 < 0.8:
                continue
            Xnew = lp_polish(Xnew, b2)
            v = max_cos(Xnew)
            if v < best_val - 1e-12:
                best, best_val = Xnew, v
        else:
            s = strengths[(it // 2) % len(strengths)]
            nk = int(rng.integers(3, 10))
            cand = shake_points(best, rng, strength=s, n_kick=nk)
            cand = repulsion_smooth(cand, steps=70, lr=0.02)
            b2 = min(4.5, LIMIT - 22 - elapsed())
            if b2 < 0.8:
                continue
            cand = lp_polish(cand, b2)
            v = max_cos(cand)
            if v < best_val - 1e-12:
                best, best_val = cand, v
    # final intensive polish
    rem = LIMIT - elapsed() - 0.5
    # allow a bit beyond LIMIT but within 180s total
    rem = min(rem, 178.0 - elapsed())
    if rem > 0.5:
        best = lp_polish(best, rem)
    best = proj_unit(best)
    return best
