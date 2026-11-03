import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _per_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1)


def _anneal(x, steps, beta0, beta1, lr0, lr1, mu=0.6):
    n, d = x.shape
    vel = np.zeros_like(x)
    for s in range(steps):
        f = s / max(steps - 1, 1)
        beta = beta0 + (beta1 - beta0) * f
        lr = lr0 + (lr1 - lr0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        Grad = W @ x
        dot = (Grad * x).sum(axis=1, keepdims=True)
        Tan = Grad - dot * x
        # normalize scale to avoid explosion
        sc = np.sqrt((Tan * Tan).sum() / n) + 1e-30
        Tan = Tan / sc
        vel = mu * vel + (1.0 - mu) * Tan
        x = x - lr * sc * 0.0 - lr * vel * sc * 0.0  # placeholder
        # use standard step with scale-invariant lr
        # actually step = lr * Tan smoothed by vel
        # recompute: we already normalized, so use lr*vel*mean_scale
        # simpler: x = x - lr * vel_scaled
        # vel already normalized scale ~1, multiply by typical scale
        x = x  # will update below correctly
        # redo cleanly: use unnormalized Tan for physics
        # To keep code simple, do momentum on unnormalized Tan:
        break
    # clean momentum anneal (second loop, correct)
    vel = np.zeros_like(x)
    for s in range(steps):
        f = s / max(steps - 1, 1)
        beta = beta0 + (beta1 - beta0) * f
        lr = lr0 + (lr1 - lr0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        Grad = W @ x
        dot = (Grad * x).sum(axis=1, keepdims=True)
        Tan = Grad - dot * x
        vel = mu * vel + Tan
        # bias correction style damping
        step = vel / (1.0 + s * 0.002)
        # clip step norm per point
        nrm = np.sqrt((step * step).sum(axis=1, keepdims=True)) + 1e-30
        # limit max move per step to lr*5
        scale = np.minimum(1.0, (lr * 5.0) / (nrm + 1e-30))
        # mean step ~ lr
        x = x - lr * step * 0.2 - lr * Tan * 0.8
        x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-30)
    return x


def _refine_void(v0, Xrem, beta=150.0, steps=120, lr=0.15):
    v = v0 / (np.linalg.norm(v0) + 1e-30)
    for t in range(steps):
        dots = Xrem @ v
        mv = dots.max()
        W = np.exp(beta * (dots - mv))
        S = W.sum() + 1e-300
        g = (W @ Xrem) / S
        gt = g - np.dot(g, v) * v
        st = lr / (1.0 + t * 0.03)
        v = v - st * gt
        nv = np.linalg.norm(v) + 1e-30
        v = v / nv
        if not np.all(np.isfinite(v)):
            return v0 / (np.linalg.norm(v0) + 1e-30)
    return v


def _find_void(Xrem, rng, ncand=25000, beta=150.0):
    n1, d = Xrem.shape
    cand = rng.standard_normal(size=(ncand, d))
    cand = cand / (np.linalg.norm(cand, axis=1, keepdims=True) + 1e-30)
    # batched to save memory
    best_val = 1e9
    best_v = cand[0]
    bs = 5000
    for a in range(0, ncand, bs):
        b = min(ncand, a + bs)
        sm = (cand[a:b] @ Xrem.T).max(axis=1)
        i = int(np.argmin(sm))
        if sm[i] < best_val:
            best_val = float(sm[i])
            best_v = cand[a + i].copy()
    try:
        best_v = _refine_void(best_v, Xrem, beta=beta, steps=120, lr=0.12)
    except Exception:
        pass
    return best_v


def _jam_smooth(x_init, betas, miters):
    n, d = x_init.shape
    Y = x_init.copy()
    Y = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-30)
    iu = np.triu_indices(n, 1)
    for beta, mi in zip(betas, miters):
        beta = float(beta)

        def fun(Yf):
            Ym = Yf.reshape(n, d)
            nr = np.sqrt(np.sum(Ym * Ym, axis=1, keepdims=True)) + 1e-30
            X = Ym / nr
            G = X @ X.T
            c = G[iu]
            mv = c.max()
            E = np.exp(beta * (c - mv))
            S = E.sum() + 1e-300
            f = mv + np.log(S) / beta
            W = np.zeros((n, n))
            W[iu] = E
            W = W + W.T
            GX = (W @ X) / S
            dot = np.sum(GX * X, axis=1, keepdims=True)
            Gt = GX - dot * X
            GY = Gt / nr
            if not np.isfinite(f):
                f = float(mv)
            return float(f), GY.ravel()
        try:
            res = minimize(fun, Y.ravel(), jac=True, method='L-BFGS-B',
                           options={'maxiter': int(mi), 'ftol': 1e-14,
                                    'gtol': 1e-7, 'maxfun': int(mi * 30), 'maxcor': 12})
            Y = res.x.reshape(n, d)
        except Exception:
            pass
        Y = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-30)
    X = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-30)
    return X


def _lp_polish(x, deadline):
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-30)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        Gfull = x @ x.T
        c = Gfull[iu]
        m = c.max()
        thr = m - max(20.0 * radius, 1e-9)
        sel = c > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 2800:
            idx = np.where(sel)[0]
            order = np.argsort(-c[idx])[:2800]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], 1
        ).ravel()
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
            res = linprog(
                obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
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
        y = y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-30)
        if not np.all(np.isfinite(y)):
            radius *= 0.5
            continue
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    rng = np.random.default_rng(7)
    t0 = time.time()
    deadline = t0 + 168.0
    n, d = 138, 8
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            v = np.load(fn).astype(float)
            if v.shape == (n, d):
                v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-30)
                c = _max_cos(v)
                if c < best_val:
                    best_val = c
                    best = v.copy()
        except Exception:
            pass
    if best is None:
        best = rng.standard_normal(size=(n, d))
        best = best / (np.linalg.norm(best, axis=1, keepdims=True) + 1e-30)
        best_val = _max_cos(best)

    it = 0
    # quick deep polish of incumbent if time allows (1s)
    # skip: already polished; go straight to hopping
    while time.time() < deadline - 7.0 and it < 300:
        it += 1
        now = time.time()
        time_left = deadline - 7.0 - now
        if time_left <= 0:
            break
        mod = it % 6
        if mod == 0 and it % 12 == 0:
            # fresh random basin (rare) for diversity
            y = rng.standard_normal(size=(n, d))
            y = y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-30)
            y = _jam_smooth(y, betas=[50.0, 150.0, 350.0], miters=[60, 60, 60])
            y = _lp_polish(y, min(deadline - 7.0, time.time() + 2.0))
            v = _max_cos(y)
            if v < best_val - 1e-12:
                best_val = v
                best = y.copy()
            continue
        y = best.copy()
        per = _per_max(y)
        # rattlers: clearly below current max
        rattlers = np.where(per < best_val - 1e-6)[0]
        worst_order = np.argsort(-per)
        try:
            if mod == 1:
                # worst-point replacement + neighbour kick
                w = int(worst_order[0])
                Xrem = np.delete(y, w, axis=0)
                nv_ = _find_void(Xrem, rng, ncand=25000, beta=150.0)
                y[w] = nv_
                dots = y @ y[w]
                neigh = np.where(dots > best_val - 0.06)[0]
                neigh = neigh[neigh != w]
                if len(neigh) > 0:
                    y[neigh] = y[neigh] + 0.06 * rng.standard_normal(size=(len(neigh), d))
                    y[neigh] = y[neigh] / (np.linalg.norm(y[neigh], axis=1, keepdims=True) + 1e-30)
                y = _jam_smooth(y, betas=[120.0, 350.0], miters=[50, 50])
                y = _anneal(y, steps=250, beta0=20.0, beta1=250.0, lr0=0.015, lr1=0.002)
                y = _lp_polish(y, min(deadline - 7.0, time.time() + 1.6))
            elif mod == 2:
                # multi rattler rebirth (up to 6)
                if len(rattlers) >= 2:
                    k = min(6, len(rattlers))
                    victims = rng.choice(rattlers, size=k, replace=False)
                else:
                    k = 4
                    victims = worst_order[:k]
                for vv in list(victims):
                    Xrem = np.delete(y, int(vv), axis=0)
                    y[int(vv)] = _find_void(Xrem, rng, ncand=18000, beta=130.0)
                # small kick to worst cluster to make room
                w = int(worst_order[0])
                dots = y @ y[w]
                neigh = np.where(dots > best_val - 0.04)[0]
                if len(neigh) > 0:
                    y[neigh] = y[neigh] + 0.03 * rng.standard_normal(size=(len(neigh), d))
                    y[neigh] = y[neigh] / (np.linalg.norm(y[neigh], axis=1, keepdims=True) + 1e-30)
                y = _jam_smooth(y, betas=[100.0, 300.0], miters=[60, 60])
                y = _lp_polish(y, min(deadline - 7.0, time.time() + 1.6))
            elif mod == 3:
                # global medium kick + 2 void replacements
                y = y + 0.10 * rng.standard_normal(size=(n, d))
                y = y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-30)
                for vv in worst_order[:2]:
                    Xrem = np.delete(y, int(vv), axis=0)
                    y[int(vv)] = _find_void(Xrem, rng, ncand=15000, beta=120.0)
                y = _anneal(y, steps=600, beta0=15.0, beta1=280.0, lr0=0.02, lr1=0.002)
                y = _lp_polish(y, min(deadline - 7.0, time.time() + 1.4))
            elif mod == 4:
                # small kick, fast anneal
                y = y + 0.03 * rng.standard_normal(size=(n, d))
                y = y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-30)
                # random re-insert 1 rattler/worst
                if len(rattlers) > 0 and rng.random() < 0.7:
                    vv = int(rng.choice(rattlers))
                else:
                    vv = int(worst_order[rng.integers(0, 5)])
                Xrem = np.delete(y, vv, axis=0)
                y[vv] = _find_void(Xrem, rng, ncand=15000, beta=120.0)
                y = _anneal(y, steps=500, beta0=25.0, beta1=300.0, lr0=0.015, lr1=0.0015)
                y = _lp_polish(y, min(deadline - 7.0, time.time() + 1.4))
            else:  # mod == 5 cluster dispersal
                w = int(worst_order[rng.integers(0, 3)])
                dots = y @ y[w]
                neigh = np.where(dots > best_val - 0.08)[0]
                if len(neigh) < 4:
                    # expand neighbourhood
                    order = np.argsort(-(y @ y[w]))
                    neigh = order[:12]
                # disperse: push away from centroid
                cent = y[neigh].mean(axis=0)
                cent = cent / (np.linalg.norm(cent) + 1e-30)
                for idx in neigh:
                    y[idx] = y[idx] - 0.05 * cent + 0.05 * rng.standard_normal(d)
                y[neigh] = y[neigh] / (np.linalg.norm(y[neigh], axis=1, keepdims=True) + 1e-30)
                # replace worst with void
                Xrem = np.delete(y, w, axis=0)
                y[w] = _find_void(Xrem, rng, ncand=20000, beta=140.0)
                y = _anneal(y, steps=550, beta0=18.0, beta1=280.0, lr0=0.018, lr1=0.002)
                y = _lp_polish(y, min(deadline - 7.0, time.time() + 1.5))
        except Exception:
            continue
        try:
            v = _max_cos(y)
        except Exception:
            continue
        if v < best_val - 1e-12:
            best_val = v
            best = y.copy()
    try:
        best = _lp_polish(best, min(deadline, time.time() + 5.0))
    except Exception:
        pass
    best = best / (np.linalg.norm(best, axis=1, keepdims=True) + 1e-30)
    # tiny shrink guard not needed; ensure finite
    best = np.where(np.isfinite(best), best, rng.standard_normal(size=(n, d)))
    best = best / (np.linalg.norm(best, axis=1, keepdims=True) + 1e-30)
    return best
