import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def lp_polish(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    best_x = x.copy()
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
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
            x, best, best_x = y, v, y.copy()
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return best_x


def smooth_relax(x, steps=600):
    n, d = x.shape
    for s in range(steps):
        frac = s / (steps - 1) if steps > 1 else 1.0
        alpha = 30.0 + (400.0 - 30.0) * frac
        lr = 0.025 * (1.0 - frac) + 0.002 * frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = np.max(G)
        G -= m
        G *= alpha
        np.clip(G, -50, 0, out=G)
        np.fill_diagonal(G, -50.0)
        np.exp(G, out=G)
        np.fill_diagonal(G, 0.0)
        S = np.sum(G) * 0.5
        if S < 1e-300:
            continue
        G /= S  # P
        grad = G @ x
        proj = np.sum(grad * x, axis=1, keepdims=True)
        grad -= proj * x
        gnorm = np.sqrt(np.sum(grad * grad, axis=1, keepdims=True)) + 1e-12
        w = np.sum(G, axis=1, keepdims=True)
        wmax = np.max(w) + 1e-12
        scale = w / wmax
        x -= (lr * scale / gnorm) * grad
        x /= np.sqrt(np.sum(x * x, axis=1, keepdims=True))
    return x


def greedy_insert(x, idx, rng, M=3000):
    n, d = x.shape
    mask = np.ones(n, dtype=bool)
    mask[idx] = False
    others = x[mask]
    C = rng.standard_normal((M, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    D = C @ others.T
    scores = np.max(D, axis=1)
    x[idx] = C[np.argmin(scores)]
    return x


def hill_climb(x, rng, iters=1500, sigma=0.03):
    n, d = x.shape
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    cur_max = float(np.max(G))
    flat = int(np.argmax(G))
    ai, aj = divmod(flat, n)
    for k in range(iters):
        i = int(rng.integers(n))
        sig = sigma * (1.0 - 0.5 * k / iters)
        prop = x[i] + sig * rng.standard_normal(d)
        nr = np.linalg.norm(prop)
        if nr < 1e-12:
            continue
        prop /= nr
        dots = x @ prop
        dots[i] = -2.0
        new_local = float(np.max(dots))
        if i == ai or i == aj:
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            max_rest = float(np.max(G[np.ix_(mask, mask)]))
        else:
            max_rest = cur_max
        new_global = new_local if new_local > max_rest else max_rest
        if new_global < cur_max - 1e-12:
            x[i] = prop
            G[i, :] = dots
            G[:, i] = dots
            G[i, i] = -2.0
            cur_max = float(np.max(G))
            flat = int(np.argmax(G))
            ai, aj = divmod(flat, n)
        elif new_global <= cur_max + 1e-12:
            if rng.random() < 0.05:
                x[i] = prop
                G[i, :] = dots
                G[:, i] = dots
                G[i, i] = -2.0
                cur_max = float(np.max(G))
                flat = int(np.argmax(G))
                ai, aj = divmod(flat, n)
    return x


def solve():
    t0 = time.time()
    rng = np.random.default_rng(0)
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy")
    except Exception:
        xr = None
    cands = []
    for c in (xb, xr):
        if c is not None:
            c = np.asarray(c, dtype=float)
            if c.shape == (116, 6):
                c = c / np.linalg.norm(c, axis=1, keepdims=True)
                cands.append(c)
    if len(cands) == 0:
        c = rng.standard_normal((116, 6))
        c /= np.linalg.norm(c, axis=1, keepdims=True)
        cands.append(c)
    vals = [max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    x_best = cands[bi].copy()
    best_val = float(vals[bi])

    search_deadline = t0 + 135.0
    hard_deadline = t0 + 170.0

    it = 0
    n = 116
    while time.time() < search_deadline:
        it += 1
        # contacts of best
        Gc = x_best @ x_best.T
        np.fill_diagonal(Gc, -2.0)
        m0 = float(np.max(Gc))
        tol = 1e-7
        cnt = np.sum(Gc > m0 - tol, axis=1)
        order_over = np.argsort(-cnt)
        ratt = np.where(cnt == 0)[0]
        x_try = x_best.copy()
        if it % 3 == 0:
            sigma_g = 0.05 if (it % 6 == 0) else 0.02
            x_try = x_try + sigma_g * rng.standard_normal(x_try.shape)
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        else:
            nk = 3 + (it % 3)  # 3..5
            kick = set()
            # 1-2 overloaded
            n_over = min(2, nk)
            top_pool = order_over[:10]
            # deterministic pick via rng
            perm = rng.permutation(len(top_pool))[:n_over]
            for k in perm:
                kick.add(int(top_pool[k]))
            # 1-2 rattlers
            if len(ratt) > 0:
                n_r = min(len(ratt), 2 if nk > 3 else 1)
                rp = rng.permutation(len(ratt))[:n_r]
                for k in rp:
                    kick.add(int(ratt[k]))
            while len(kick) < nk:
                kick.add(int(rng.integers(n)))
            for idx in list(kick):
                greedy_insert(x_try, idx, rng, M=2500)
                if time.time() > search_deadline:
                    break
        # relax
        x_try = smooth_relax(x_try, steps=550)
        if time.time() > search_deadline:
            break
        x_try = hill_climb(x_try, rng, iters=1200, sigma=0.03)
        v = max_cos(x_try)
        if v < best_val - 1e-12:
            # short LP snap
            x_snapped = lp_polish(x_try, min(time.time() + 2.0, hard_deadline))
            v2 = max_cos(x_snapped)
            if v2 < best_val:
                x_best = x_snapped
                best_val = v2
        if time.time() > search_deadline:
            break

    x_best = lp_polish(x_best, hard_deadline)
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
