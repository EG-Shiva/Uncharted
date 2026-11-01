import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def smooth_descent(x, steps, beta0, beta1, lr0, lr1, mu=0.85, rng=None):
    n, d = x.shape
    vel = np.zeros_like(x)
    for s in range(steps):
        t = s / max(steps - 1, 1)
        beta = beta0 * (beta1 / beta0) ** t if beta0 > 0 else beta1
        lr = lr0 + (lr1 - lr0) * t
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        Grad = W @ x
        proj = np.sum(Grad * x, axis=1, keepdims=True)
        Tg = Grad - proj * x
        rms = np.sqrt(np.mean(Tg * Tg)) + 1e-12
        Tg = Tg / rms
        vel = mu * vel + (1.0 - mu) * Tg
        # bias correction-ish scaling
        x = x - lr * vel / (1.0 - mu ** (s + 2)) * 0.2 - lr * Tg * 0.8
        # project and renormalize
        nrm = np.linalg.norm(x, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        x = x / nrm
    return x


def lp_polish(x, seconds, deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while True:
        now = time.time()
        if now - start > seconds:
            break
        if deadline is not None and now > deadline - 0.3:
            break
        if radius < 1e-14:
            break
        Gfull = x @ x.T
        c = Gfull[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cI = np.arange(d)
        cols = np.concatenate([I[:, None] * d + cI, J[:, None] * d + cI, np.full((p, 1), n * d)], 1).ravel()
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
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def worst_order(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    per = G.max(axis=1)
    return np.argsort(-per), per


def best_hole(x_rest, rng, ncand=4000):
    d = x_rest.shape[1]
    cand = rng.standard_normal((ncand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    scores = (cand @ x_rest.T).max(axis=1)
    return cand[int(np.argmin(scores))]


def micro_hill(x, seconds, rng):
    # single-point greedy tangent search on exact max
    start = time.time()
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-4
    it = 0
    while time.time() - start < seconds:
        it += 1
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        per = G.max(axis=1)
        i = int(np.argmax(per))
        # partner direction to move away from
        # random tangent perturbation
        r = rng.standard_normal(d)
        r = r - np.dot(r, x[i]) * x[i]
        nr = np.linalg.norm(r)
        if nr < 1e-12:
            continue
        r /= nr
        step = radius * (0.5 + rng.random())
        y = x.copy()
        v = x[i] + step * r
        v /= np.linalg.norm(v)
        y[i] = v
        val = max_cos(y)
        if val < best:
            x = y
            best = val
            radius = min(radius * 1.3, 1e-3)
        else:
            # decay radius occasionally
            if it % 50 == 0:
                radius *= 0.7
                if radius < 1e-12:
                    radius = 1e-4
    return x


def solve():
    t0 = time.time()
    LIMIT = 172.0
    deadline = t0 + LIMIT
    rng = np.random.default_rng(6812)

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            a /= np.linalg.norm(a, axis=1, keepdims=True)
            if a.shape == (68, 12) and np.all(np.isfinite(a)):
                cands.append(a)
        except Exception:
            pass
    if len(cands) == 0:
        a = rng.standard_normal((68, 12))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    # pick best loaded
    best_val = 1e9
    x_best = cands[0].copy()
    for c in cands:
        v = max_cos(c)
        if v < best_val:
            best_val = v
            x_best = c.copy()
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)

    n, d = 68, 12
    # reserve final polish
    explore_end = t0 + LIMIT - 32.0

    trial = 0
    # quick initial LP to ensure tight baseline (short)
    # do not waste time if already tight
    while time.time() < explore_end:
        trial += 1
        x = x_best.copy()
        u = rng.random()
        order, per = worst_order(x)
        if u < 0.32:
            # global Gaussian shake, log-uniform sigma
            ls = rng.uniform(-2.4, -0.85)
            sigma = 10.0 ** ls
            x = x + sigma * rng.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            b0 = float(rng.uniform(30, 80))
            b1 = float(rng.uniform(300, 850))
            lr0 = float(rng.uniform(0.02, 0.05))
            lr1 = float(rng.uniform(0.002, 0.008))
            steps = int(rng.integers(350, 800))
            x = smooth_descent(x, steps, b0, b1, lr0, lr1, mu=0.85)
        elif u < 0.58:
            # kick k points: worst-biased + random, large re-randomize
            k = int(rng.integers(1, 5))
            idx = set()
            nw = min(k, 2)
            for j in range(nw):
                # sample among top-8 worst with randomness
                cand_pool = order[:8]
                idx.add(int(cand_pool[rng.integers(0, len(cand_pool))]))
            while len(idx) < k:
                idx.add(int(rng.integers(0, n)))
            idx = list(idx)
            for ii in idx:
                if rng.random() < 0.6:
                    v = rng.standard_normal(d)
                    v /= np.linalg.norm(v)
                    x[ii] = v
                else:
                    x[ii] = x[ii] + 0.35 * rng.standard_normal(d)
                    x[ii] /= np.linalg.norm(x[ii])
            b0 = float(rng.uniform(40, 90))
            b1 = float(rng.uniform(350, 900))
            steps = int(rng.integers(400, 800))
            x = smooth_descent(x, steps, b0, b1, 0.035, 0.004, mu=0.85)
        elif u < 0.78:
            # delete worst 1-2 and reinsert at best hole
            k = 1 if rng.random() < 0.7 else 2
            rem = list(order[:k]) if rng.random() < 0.8 else list(rng.choice(n, k, replace=False))
            keep = [i for i in range(n) if i not in rem]
            x_rest = x[keep]
            # refine rest a bit first
            x_rest = smooth_descent(x_rest, 150, 60.0, 400.0, 0.02, 0.005, mu=0.8)
            newpts = []
            xr = x_rest
            for _ in range(k):
                h = best_hole(xr, rng, ncand=5000)
                # local jitter around hole: pick best of jittered
                jit = h[None, :] + 0.02 * rng.standard_normal((200, d))
                jit /= np.linalg.norm(jit, axis=1, keepdims=True)
                sc = (jit @ xr.T).max(axis=1)
                h = jit[int(np.argmin(sc))]
                newpts.append(h)
                xr = np.vstack([xr, h[None, :]])
            x = xr  # already 68
            x = smooth_descent(x, 500, 60.0, 600.0, 0.03, 0.004, mu=0.85)
        else:
            # subspace rotation of a small cluster + hub-targeted jitter
            # pick hub (point 0-like = most contacts) and its nearest neighbors
            G = x @ x.T
            np.fill_diagonal(G, -1.0)
            hub = int(order[0])
            neigh = np.argsort(-G[hub])[:6]
            # random small rotation in random plane applied to cluster
            A = rng.standard_normal((d, d))
            Q, _ = np.linalg.qr(A)
            ang = float(rng.uniform(0.02, 0.18))
            # Givens rotation in first two coords of Q basis
            R = np.eye(d)
            R[0, 0] = np.cos(ang); R[0, 1] = -np.sin(ang)
            R[1, 0] = np.sin(ang); R[1, 1] = np.cos(ang)
            Rot = Q @ R @ Q.T
            for ii in list(neigh) + [hub]:
                if rng.random() < 0.8:
                    x[ii] = Rot @ x[ii]
            # extra jitter on hub
            x[hub] = x[hub] + float(rng.uniform(0.01, 0.06)) * rng.standard_normal(d)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = smooth_descent(x, 600, 50.0, 700.0, 0.03, 0.003, mu=0.88)

        v_smooth = max_cos(x)
        # only LP-probe if promising (avoid wasting time on bad basins)
        # threshold generous early, tighter later
        if v_smooth < best_val + 0.012:
            # short probe
            probe_time = 1.2 if v_smooth < best_val + 0.003 else 0.6
            # cap by remaining explore time
            remain = explore_end - time.time()
            if remain < 0.8:
                break
            probe_time = min(probe_time, remain - 0.1)
            if probe_time > 0.25:
                x = lp_polish(x, probe_time)
                v = max_cos(x)
                if v < best_val - 1e-12:
                    best_val = v
                    x_best = x.copy()
        # occasional micro acceptance of equal (to move laterally)
        # continue

        if time.time() > explore_end:
            break

    # final intensive polish on best
    remain = deadline - time.time()
    if remain > 5:
        # alternate LP and micro + smooth touch
        # first long LP
        lp_time = max(remain - 6.0, remain * 0.6)
        x_best = lp_polish(x_best, lp_time, deadline=deadline - 5.0)
        best_val = max_cos(x_best)
        # micro hill
        if deadline - time.time() > 4:
            x_best = micro_hill(x_best, min(3.0, (deadline - time.time()) * 0.4), rng)
            v = max_cos(x_best)
            if v < best_val:
                best_val = v
            # final LP squeeze
            if deadline - time.time() > 2:
                x_best = lp_polish(x_best, deadline - time.time() - 0.5)

    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
