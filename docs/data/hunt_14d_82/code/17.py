import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def lp_polish_dl(x, deadline, radius_init=1.5e-3, max_pairs=1500):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = radius_init
    iu = np.triu_indices(n, 1)
    niter = 0
    while time.time() < deadline - 0.4 and radius > 1e-14 and niter < 40:
        niter += 1
        Gfull = x @ x.T
        c = Gfull[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 2e-7)
        cnt = int(np.sum(sel))
        if cnt == 0:
            break
        if cnt > max_pairs:
            # keep largest
            idx = np.argpartition(-c, max_pairs)[:max_pairs]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx] = True
            sel = sel & mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix(
                (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
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
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return x

def hybrid_quench(X_init, T, steps, lr0, lr1, tau0, tau1, deadline):
    X = X_init.copy()
    n, d = X.shape
    m1 = np.zeros_like(X)
    m2 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    l0, l1 = np.log(lr0), np.log(lr1)
    lt0, lt1 = np.log(tau0), np.log(tau1)
    for s in range(steps):
        if s % 60 == 0 and time.time() > deadline - 1.0:
            break
        frac = s / max(steps - 1, 1)
        lr = float(np.exp(l0 + (l1 - l0) * frac))
        tau = float(np.exp(lt0 + (lt1 - lt0) * frac))
        p = 2.0 + 5.0 * frac
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        D = G - T
        pos = D > 0
        if np.any(pos):
            W = np.zeros_like(G)
            W[pos] = p * (D[pos] ** (p - 1.0))
            Gr = W @ X
        else:
            m = float(np.max(G))
            # softmax descent when feasible
            E = np.exp((G - m) / tau)
            np.fill_diagonal(E, 0.0)
            S = float(np.sum(E))
            if not np.isfinite(S) or S == 0:
                break
            E /= S
            Gr = E @ X
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        b1p *= b1
        b2p *= b2
        m1 = b1 * m1 + (1.0 - b1) * Gt
        m2 = b2 * m2 + (1.0 - b2) * (Gt * Gt)
        mh = m1 / (1.0 - b1p)
        vh = m2 / (1.0 - b2p)
        upd = mh / (np.sqrt(vh) + eps)
        X = X - lr * upd
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X = X / nr
    return X

def hole_best(X, v, rng, nc=2000):
    n, d = X.shape
    others = np.delete(X, v, axis=0)
    cand = rng.standard_normal((nc, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ others.T
    worst = dots.max(axis=1)
    return cand[int(np.argmin(worst))]

def solve():
    t0 = time.time()
    LIMIT = 166.0
    deadline_final = t0 + LIMIT
    rng = np.random.default_rng(12345)
    np.random.seed(12345)
    n, d = 82, 14
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.standard_normal((n, d))
    if xb.shape != (n, d):
        xb = rng.standard_normal((n, d))
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    if time.time() < deadline_final - 30:
        xb = lp_polish_dl(xb, min(time.time() + 2.5, deadline_final - 20), radius_init=1.5e-3)
    best_val = max_cos(xb)
    X_best = xb.copy()

    it = 0
    while time.time() < deadline_final - 7.0:
        it += 1
        time_left = deadline_final - (time.time() - t0) - 7.0
        if time_left < 2:
            break
        # target gap
        r = it % 10
        if r == 0:
            delta = float(np.exp(rng.uniform(np.log(8e-4), np.log(3.5e-3))))
        elif r % 3 == 0:
            delta = float(np.exp(rng.uniform(np.log(2e-4), np.log(8e-4))))
        else:
            delta = float(np.exp(rng.uniform(np.log(2e-5), np.log(2.5e-4))))
        T = best_val - delta

        # diagnose bottleneck
        G = X_best @ X_best.T
        np.fill_diagonal(G, -2.0)
        per_max = G.max(axis=1)
        order_low = np.argsort(per_max, kind="stable")
        order_high = order_low[::-1]
        flat = int(np.argmax(G))
        wa, wb = divmod(flat, n)

        X = X_best.copy()
        mode = it % 6
        sigma = float(10.0 ** rng.uniform(-2.4, -0.7))
        victims = []
        if mode == 0:
            # break worst pair: move one endpoint to hole
            v = wa if rng.random() < 0.5 else wb
            # occasionally move both
            if rng.random() < 0.25:
                X[wa] = hole_best(X, wa, rng, nc=2000)
                X[wb] = hole_best(X, wb, rng, nc=1200)
                victims = [wa, wb]
            else:
                X[v] = hole_best(X, v, rng, nc=2500)
                victims = [v]
            # shake neighbours of the pair
            neigh = np.argsort(-(G[wa] + G[wb]), kind="stable")[:4]
            for u in neigh:
                if u != wa and u != wb and rng.random() < 0.6:
                    X[u] += rng.standard_normal(d) * (sigma * 0.5)
            victims = list(victims) + [int(u) for u in neigh[:2]]
        elif mode == 1:
            # pairwise separation along difference
            eps = float(10.0 ** rng.uniform(-2.2, -1.0)) * 0.5
            da = X[wa] - X[wb]
            da -= np.dot(da, X[wa]) * X[wa]
            db = -da
            X[wa] += da * eps * 5.0
            X[wb] += db * eps * 5.0
            X[wa] /= np.linalg.norm(X[wa])
            X[wb] /= np.linalg.norm(X[wb])
            # small global jitter
            X += rng.standard_normal((n, d)) * (sigma * 0.2)
            victims = [wa, wb]
        elif mode == 2:
            k = int(rng.integers(2, 6))
            victims = order_low[:k].tolist()
            for v in victims:
                if rng.random() < 0.85:
                    X[int(v)] = hole_best(X, int(v), rng, nc=1800)
                else:
                    z = rng.standard_normal(d); z /= np.linalg.norm(z)
                    X[int(v)] = z
            X[victims] += rng.standard_normal((len(victims), d)) * (sigma * 0.4)
        elif mode == 3:
            k = int(rng.integers(2, 7))
            victims = rng.choice(n, size=k, replace=False).tolist()
            for v in victims:
                if rng.random() < 0.7:
                    X[int(v)] = hole_best(X, int(v), rng, nc=1500)
            X += rng.standard_normal((n, d)) * (sigma * 0.22)
        elif mode == 4:
            k = int(rng.integers(2, 5))
            victims = order_high[:k].tolist()
            for v in victims:
                if rng.random() < 0.5:
                    X[int(v)] = hole_best(X, int(v), rng, nc=1500)
            X[victims] += rng.standard_normal((len(victims), d)) * (sigma * 0.7)
            X += rng.standard_normal((n, d)) * (sigma * 0.15)
        else:
            gsig = float(10.0 ** rng.uniform(-2.0, -0.9))
            X += rng.standard_normal((n, d)) * gsig

        X /= np.linalg.norm(X, axis=1, keepdims=True)

        if time_left > 40:
            steps = 550
        elif time_left > 20:
            steps = 380
        else:
            steps = 240
        lr0 = float(0.012 + rng.random() * 0.018)
        lr1 = lr0 * 0.10
        tau0, tau1 = 0.004, 0.0006
        qdead = deadline_final - 5.0
        X2 = hybrid_quench(X, T, steps=steps, lr0=lr0, lr1=lr1, tau0=tau0, tau1=tau1, deadline=qdead)
        qv = max_cos(X2)
        if qv > best_val + 0.03 and time.time() < deadline_final - 25:
            continue
        if time.time() > deadline_final - 6.0:
            # no time for LP, just check
            if qv < best_val - 1e-12:
                X_best = X2 / np.linalg.norm(X2, axis=1, keepdims=True)
                best_val = qv
            break
        if qv < best_val:
            psec = 1.4 if time_left > 30 else 0.8
            psec = min(psec + 0.6, 2.2)
        else:
            psec = 0.9 if time_left > 30 else 0.55
        lp_dead = min(time.time() + psec, deadline_final - 5.0)
        if lp_dead - time.time() < 0.25:
            if qv < best_val - 1e-12:
                X_best = X2; best_val = qv
            continue
        Xp = lp_polish_dl(X2, lp_dead, radius_init=1.5e-3)
        pv = max_cos(Xp)
        if pv < best_val - 1e-12:
            X_best = Xp
            best_val = pv

    rem = (deadline_final - 1.0) - time.time()
    if rem > 1.2:
        X_best = lp_polish_dl(X_best, time.time() + min(rem, 6.0), radius_init=8e-4)
    X_best = X_best / np.linalg.norm(X_best, axis=1, keepdims=True)
    return X_best
