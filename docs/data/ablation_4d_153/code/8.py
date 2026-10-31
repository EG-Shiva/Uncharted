import time
import numpy as np


def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))


def _random_sphere(n, d, rng):
    x = rng.standard_normal(size=(n, d))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def _smooth_opt(x, steps, tau0, tau1, lr0, lr1, mom=0.88):
    n, d = x.shape
    vel = np.zeros_like(x)
    lt0, lt1 = np.log(tau0), np.log(tau1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for t in range(steps):
        f = t / max(1, steps - 1)
        tau = float(np.exp(lt0 + (lt1 - lt0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        m = float(np.max(G))
        W = np.exp((G - m) / tau)
        np.fill_diagonal(W, 0.0)
        row = np.sum(W, axis=1, keepdims=True)
        row = np.maximum(row, 1e-12)
        grad = (W @ x) / row
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad = grad - dot * x
        gn = np.linalg.norm(grad, axis=1, keepdims=True)
        grad = grad * np.where(gn > 2.0, 2.0 / np.maximum(gn, 1e-12), 1.0)
        vel = mom * vel + (1.0 - mom) * grad
        bc = 1.0 - mom ** (t + 1)
        x = x - lr * (vel / bc)
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x = x / nr
    return x


def _sharp_opt(x, steps, lr0, lr1, d0, d1, mom=0.85):
    n, d = x.shape
    vel = np.zeros_like(x)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    ld0, ld1 = np.log(d0), np.log(d1)
    for t in range(steps):
        f = t / max(1, steps - 1)
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        delta = float(np.exp(ld0 + (ld1 - ld0) * f))
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        m = float(np.max(G))
        thresh = m - delta
        M = G > thresh
        # linear weight emphasizing closest
        W = np.where(M, 1.0 + 8.0 * np.maximum(G - thresh, 0.0) / max(delta, 1e-12), 0.0)
        # degree normalisation
        deg = np.sum(M, axis=1, keepdims=True)
        deg = np.maximum(deg, 1.0)
        grad = (W @ x) / deg
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad = grad - dot * x
        gn = np.linalg.norm(grad, axis=1, keepdims=True)
        grad = grad * np.where(gn > 2.0, 2.0 / np.maximum(gn, 1e-12), 1.0)
        vel = mom * vel + (1.0 - mom) * grad
        bc = 1.0 - mom ** (t + 1)
        x = x - lr * (vel / bc)
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x = x / nr
    return x


def _hole_replace(x0, worst, rng, ncand=3000):
    x = x0.copy()
    n, d = x.shape
    for w in np.atleast_1d(worst):
        cand = rng.standard_normal(size=(ncand, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        dots = cand @ x.T
        dots[:, w] = -2.0
        md = np.max(dots, axis=1)
        j = int(np.argmin(md))
        x[w] = cand[j]
    return x


def _lp_polish(x, deadline):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        idx = np.where(c > thresh)[0]
        p0 = len(idx)
        if p0 == 0:
            break
        if p0 > 2500:
            order = np.argpartition(c[idx], -2500)[-2500:]
            idx = idx[order]
            p0 = 2500
        I = iu[0][idx]
        J = iu[1][idx]
        csel = c[idx]
        p = p0
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cI = (I[:, None] * d + np.arange(d))
        cJ = (J[:, None] * d + np.arange(d))
        cols = np.concatenate([cI, cJ, np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x = y
            best = v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() >= deadline:
            break
    return x, best


def solve():
    t0 = time.time()
    HARD_END = t0 + 175.0
    rng = np.random.default_rng(12345)
    n, d = 153, 4
    best_x = None
    best_v = 1e9
    for fname in ("best.npy", "record.npy"):
        try:
            a = np.load(fname)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(a)
                if v < best_v:
                    best_v = v
                    best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        best_x = _random_sphere(n, d, rng)
        best_v = _max_cos(best_x)
    cur_x = best_x.copy()
    cur_v = best_v

    sigmas = [0.004, 0.01, 0.025, 0.06, 0.12, 0.22, 0.38, 0.6]
    trial = 0
    # exploration phase, reserve ~35s for final polish
    while time.time() < HARD_END - 35.0:
        trial += 1
        if trial % 13 == 0:
            # pure random restart
            x0 = _random_sphere(n, d, rng)
            x1 = _smooth_opt(x0, 3800, 0.06, 0.003, 0.09, 0.003, mom=0.88)
            x1 = _sharp_opt(x1, 700, 0.03, 0.002, 0.03, 0.002, mom=0.85)
        elif trial % 3 == 0:
            # hole reinsertion of worst points
            base = cur_x if rng.random() < 0.65 else best_x
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            indiv = np.max(G, axis=1)
            k = int(rng.integers(1, 4))
            worst = np.argsort(indiv)[-k:]
            x0 = _hole_replace(base, worst, rng, ncand=3000)
            # small global jitter
            x0 = x0 + 0.008 * rng.standard_normal(size=(n, d))
            x0 = x0 / np.linalg.norm(x0, axis=1, keepdims=True)
            # occasional extra random swap
            if rng.random() < 0.3:
                kk = int(rng.integers(1, 4))
                idx = rng.choice(n, size=kk, replace=False)
                rp = _random_sphere(kk, d, rng)
                x0[idx] = rp
            x1 = _smooth_opt(x0, 2600, 0.03, 0.0022, 0.07, 0.0025, mom=0.88)
            x1 = _sharp_opt(x1, 500, 0.025, 0.002, 0.02, 0.0015, mom=0.85)
        else:
            sigma = sigmas[(trial + int(rng.integers(0, 8))) % len(sigmas)]
            base = cur_x if (trial % 4 != 2) else best_x
            x0 = base + sigma * rng.standard_normal(size=(n, d))
            x0 = x0 / np.linalg.norm(x0, axis=1, keepdims=True)
            if sigma > 0.15 and rng.random() < 0.5:
                k = int(rng.integers(1, 6))
                idx = rng.choice(n, size=k, replace=False)
                x0[idx] = _random_sphere(k, d, rng)
            if sigma < 0.03:
                x1 = _smooth_opt(x0, 2000, 0.022, 0.002, 0.055, 0.0025, mom=0.88)
                x1 = _sharp_opt(x1, 400, 0.02, 0.0015, 0.015, 0.0012, mom=0.85)
            elif sigma < 0.15:
                x1 = _smooth_opt(x0, 2600, 0.035, 0.0025, 0.075, 0.0025, mom=0.88)
                x1 = _sharp_opt(x1, 500, 0.025, 0.0018, 0.02, 0.0015, mom=0.85)
            else:
                x1 = _smooth_opt(x0, 3000, 0.05, 0.003, 0.085, 0.003, mom=0.88)
                x1 = _sharp_opt(x1, 600, 0.03, 0.002, 0.025, 0.0018, mom=0.85)
        v1 = _max_cos(x1)
        if v1 < best_v + 0.004 and time.time() < HARD_END - 35.0:
            dl = min(time.time() + 1.0, HARD_END - 33.0)
            if dl > time.time() + 0.15:
                try:
                    x1p, v1p = _lp_polish(x1, dl)
                    if v1p < v1:
                        x1, v1 = x1p, v1p
                except Exception:
                    pass
        if v1 < best_v:
            best_x = x1.copy()
            best_v = v1
        tol = 0.003 * (1.0 - min(trial, 60) / 60.0) + 0.0004
        if v1 < cur_v + tol:
            cur_x = x1.copy()
            cur_v = v1
            if cur_v > best_v + 0.008:
                cur_x = best_x.copy()
                cur_v = best_v
        else:
            if rng.random() < 0.12:
                cur_x = best_x.copy()
                cur_v = best_v
        if trial > 200:
            break

    try:
        best_x, best_v = _lp_polish(best_x, HARD_END - 1.0)
    except Exception:
        pass
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
