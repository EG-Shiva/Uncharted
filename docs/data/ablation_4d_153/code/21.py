import time
import numpy as np


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def _random_sphere(n, d, rng):
    x = rng.standard_normal(size=(n, d))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def _smooth_softmax(x, steps, tau0, tau1, lr0, lr1, mom=0.88):
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


def _smooth_riesz(x, steps, s0, s1, lr0, lr1, mom=0.88):
    n, d = x.shape
    vel = np.zeros_like(x)
    for t in range(steps):
        f = t / max(1, steps - 1)
        s = s0 + (s1 - s0) * f
        lr = float(np.exp(np.log(lr0) + (np.log(lr1) - np.log(lr0)) * f))
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        r2 = np.maximum(2.0 - 2.0 * G, 1e-3)
        # power weights, diagonal -> 0
        W = np.power(r2, -s / 2.0 - 1.0)
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


def _refine_hole(fixed, y, steps=350, lr0=0.03, lr1=0.002, tau=0.004):
    y = y / np.linalg.norm(y)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for t in range(steps):
        f = t / max(1, steps - 1)
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        dots = fixed @ y
        m = float(np.max(dots))
        w = np.exp((dots - m) / tau)
        s = float(np.sum(w))
        if s <= 0:
            break
        g = (w[:, None] * fixed).sum(axis=0) / s
        gt = g - float(np.dot(g, y)) * y
        y = y - lr * gt
        n = float(np.linalg.norm(y))
        if n == 0:
            break
        y = y / n
    return y


def _hole_position(fixed, rng, n_sample=25000, tau=0.004):
    d = fixed.shape[1]
    cand = rng.standard_normal(size=(n_sample, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    scores = cand @ fixed.T
    best_score = scores.max(axis=1)
    k = min(3, n_sample)
    idx = np.argpartition(best_score, k - 1)[:k]
    best_y = None
    best_v = 1e9
    for j in idx:
        y = _refine_hole(fixed, cand[j], tau=tau)
        v = float(np.max(fixed @ y))
        if v < best_v:
            best_v = v
            best_y = y
    return best_y


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
        sel = c > thresh
        p0 = int(np.count_nonzero(sel))
        if p0 == 0:
            break
        idx = np.where(sel)[0]
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
    rng = np.random.default_rng(153)
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

    sigmas = [0.004, 0.01, 0.025, 0.06, 0.12, 0.22, 0.4, 0.7]
    it = 0
    while time.time() < HARD_END - 35.0:
        it += 1
        r = float(rng.random())
        if r < 0.13:
            # fresh random start, strong anneal
            x0 = _random_sphere(n, d, rng)
            use_riesz = bool(rng.random() < 0.4)
            if use_riesz:
                x1 = _smooth_riesz(x0, 2600, 2.0, 22.0, 0.09, 0.003)
            else:
                x1 = _smooth_softmax(x0, 3000, 0.06, 0.0035, 0.10, 0.003, mom=0.88)
        elif r < 0.42:
            # hole reinsertion move
            base = cur_x if float(rng.random()) < 0.7 else best_x
            x0 = base.copy()
            G = x0 @ x0.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            prow = np.max(G, axis=1)
            deg = np.sum(G > m - 5e-4, axis=1)
            k = int(rng.integers(2, 7))
            # build replacement list: mix of rattlers and worst
            order_worst = np.argsort(-prow)
            order_rattle = np.argsort(deg)
            chosen = []
            for j in range(k):
                if float(rng.random()) < 0.5:
                    # rattler / low degree
                    for cand in order_rattle:
                        if int(cand) not in chosen:
                            chosen.append(int(cand))
                            break
                else:
                    # worst point, random among top 15
                    top = order_worst[:15]
                    # pick random not yet chosen
                    perm = top[rng.permutation(len(top))]
                    picked = False
                    for cand in perm:
                        if int(cand) not in chosen:
                            chosen.append(int(cand))
                            picked = True
                            break
                    if not picked:
                        for cand in order_worst:
                            if int(cand) not in chosen:
                                chosen.append(int(cand))
                                break
            # sequential greedy hole fill
            tau_h = float(rng.choice(np.array([0.003, 0.005, 0.008])))
            for idxr in chosen:
                mask = np.ones(n, dtype=bool)
                mask[idxr] = False
                fixed = x0[mask]
                y = _hole_position(fixed, rng, n_sample=22000, tau=tau_h)
                x0[idxr] = y
            if float(rng.random()) < 0.6:
                s1 = float(rng.choice(np.array([14.0, 20.0, 26.0])))
                x1 = _smooth_riesz(x0, 1800, 3.0, s1, 0.07, 0.003)
            else:
                tau1 = float(rng.choice(np.array([0.0025, 0.0035, 0.005])))
                x1 = _smooth_softmax(x0, 2000, 0.035, tau1, 0.08, 0.003, mom=0.88)
        else:
            # gaussian basin hop
            sigma = sigmas[int(rng.integers(0, len(sigmas)))]
            base = cur_x if (it % 3 != 2) else best_x
            # occasionally restart walker from best if drifting
            if cur_v > best_v + 0.012 and float(rng.random()) < 0.5:
                base = best_x
            noise = rng.standard_normal(size=(n, d))
            x0 = base + sigma * noise
            x0 = x0 / np.linalg.norm(x0, axis=1, keepdims=True)
            if float(rng.random()) < 0.35:
                k = int(rng.integers(1, 4))
                idx = rng.choice(n, size=k, replace=False)
                rp = rng.standard_normal(size=(k, d))
                rp /= np.linalg.norm(rp, axis=1, keepdims=True)
                x0[idx] = rp
            if sigma < 0.05:
                x1 = _smooth_softmax(x0, 1800, 0.025, 0.0025, 0.06, 0.003, mom=0.9)
            elif sigma < 0.2:
                if float(rng.random()) < 0.4:
                    x1 = _smooth_riesz(x0, 2000, 2.0, 18.0, 0.08, 0.003)
                else:
                    x1 = _smooth_softmax(x0, 2200, 0.035, 0.003, 0.08, 0.003, mom=0.88)
            else:
                x1 = _smooth_softmax(x0, 2600, 0.05, 0.0035, 0.09, 0.003, mom=0.88)
        v1 = _max_cos(x1)
        if v1 < best_v + 0.004 and time.time() < HARD_END - 35.0:
            dl = min(time.time() + 0.9, HARD_END - 34.0)
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
        tol = 0.0035 * (1.0 - min(1.0, it / 80.0)) + 0.0004
        if v1 < cur_v + tol:
            cur_x = x1.copy()
            cur_v = v1
            if cur_v > best_v + 0.01:
                cur_x = best_x.copy()
                cur_v = best_v
        else:
            if float(rng.random()) < 0.12:
                cur_x = best_x.copy()
                cur_v = best_v
        if time.time() >= HARD_END - 35.0:
            break

    try:
        best_x, best_v = _lp_polish(best_x, HARD_END - 1.0)
    except Exception:
        pass
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
