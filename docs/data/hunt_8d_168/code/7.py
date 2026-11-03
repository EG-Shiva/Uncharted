import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _contact_degrees(x, m, tol):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    mask = G > m - tol
    # count upper only then distribute
    deg = (mask).sum(axis=1)
    iu = np.triu_indices(len(x), 1)
    pairs = np.column_stack((iu[0][mask[iu]], iu[1][mask[iu]])) if False else None
    return deg, G

def _lp_polish(x, seconds, t_end_limit=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while True:
        if time.time() - start > seconds:
            break
        if t_end_limit is not None and time.time() > t_end_limit:
            break
        if radius < 1e-14:
            break
        Gfull = x @ x.T
        iu_c = Gfull[iu]
        m = iu_c.max()
        sel = iu_c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-iu_c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def _soft_run(x_init, steps, a0, a1, lr0, lr1, mu, rng, jitter_scale=0.0):
    x = x_init.copy()
    n, d = x.shape
    vel = np.zeros_like(x)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        alpha = a0 + (a1 - a0) * frac
        if lr0 > 0 and lr1 > 0:
            lr = lr0 * (lr1 / lr0) ** frac
        else:
            lr = lr0
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        # cap alpha*(G-m) to avoid overflow: G-m <=0
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        S = W.sum(axis=1, keepdims=True) + 1e-12
        A = (W @ x) / S
        dot = np.sum(A * x, axis=1, keepdims=True)
        At = A - dot * x
        # normalize gradient scale
        gnorm = np.linalg.norm(At) + 1e-12
        # keep scale independent: use At directly
        vel = mu * vel + At
        x = x - lr * vel
        # tiny extra noise annealing to escape (optional)
        if jitter_scale > 0:
            x = x + rng.normal(size=x.shape) * (jitter_scale * (1 - frac))
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x /= nr
    return x

def _greedy_fill(x_keep, k, rng, n_try=300):
    # x_keep: (n-k, d) normalized, generate k new points greedily max-min
    d = x_keep.shape[1]
    new_pts = []
    cur = x_keep.copy()
    for _ in range(k):
        cand = rng.normal(size=(n_try, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        # max dot to cur (and to already added)
        dots = cand @ cur.T
        md = dots.max(axis=1)
        best_i = np.argmin(md)
        best = cand[best_i]
        new_pts.append(best)
        cur = np.vstack([cur, best.reshape(1, -1)])
    return np.vstack([x_keep, np.array(new_pts)])

def solve():
    t0 = time.time()
    BUDGET = 172.0
    rng = np.random.default_rng(7)
    n, d = 168, 8
    try:
        best0 = np.load("best.npy").astype(float)
    except Exception:
        best0 = None
    try:
        rec0 = np.load("record.npy").astype(float)
    except Exception:
        rec0 = None
    cands = []
    if best0 is not None and best0.shape == (n, d):
        cands.append(best0 / np.linalg.norm(best0, axis=1, keepdims=True))
    if rec0 is not None and rec0.shape == (n, d):
        cands.append(rec0 / np.linalg.norm(rec0, axis=1, keepdims=True))
    if not cands:
        c = rng.normal(size=(n, d))
        c /= np.linalg.norm(c, axis=1, keepdims=True)
        cands.append(c)
    best_x = min(cands, key=_max_cos)
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    best_val = _max_cos(best_x)

    # quick short LP to ensure baseline tight (2 sec)
    if time.time() - t0 < BUDGET - 60:
        bx, bv = _lp_polish(best_x, 2.0)
        if bv < best_val:
            best_x, best_val = bx, bv

    it = 0
    # precompute loop until time for final polish
    while time.time() - t0 < BUDGET - 25:
        it += 1
        r = rng.random()
        # choose perturbation type
        base = best_x
        G = base @ base.T
        np.fill_diagonal(G, -2.0)
        m0 = G.max()
        # degrees for tight contacts
        tol_deg = 0.002 if rng.random() < 0.5 else 0.01
        deg = (G > m0 - tol_deg).sum(axis=1)

        if r < 0.28:
            # global jitter, log-uniform sigma
            sigma = float(np.exp(rng.uniform(np.log(1e-4), np.log(0.25))))
            x0 = base + rng.normal(size=base.shape) * sigma
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            k_replace = 0
        elif r < 0.48:
            # overloaded hub shake: pick top hubs
            nhub = int(rng.integers(3, 12))
            order = np.argsort(-deg)
            hubs = order[:nhub]
            # also include random neighbors of hubs
            sigma_hub = float(np.exp(rng.uniform(np.log(0.02), np.log(0.5))))
            sigma_rest = float(np.exp(rng.uniform(np.log(1e-5), np.log(1e-3))))
            noise = rng.normal(size=base.shape) * sigma_rest
            noise[hubs] += rng.normal(size=(nhub, d)) * sigma_hub
            # occasionally shake 2nd-order neighbors
            if rng.random() < 0.5:
                # neighbors with large dot to hubs
                for h in hubs[:2]:
                    nbrs = np.where(G[h] > m0 - 0.05)[0]
                    if len(nbrs) > 0:
                        sel = rng.choice(nbrs, size=min(len(nbrs), 4), replace=False)
                        noise[sel] += rng.normal(size=(len(sel), d)) * sigma_hub * 0.5
            x0 = base + noise
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            k_replace = 0
        elif r < 0.68:
            # subset random jitter (random subset bigger move)
            ksub = int(rng.integers(5, 30))
            idx = rng.choice(n, size=ksub, replace=False)
            sigma = float(np.exp(rng.uniform(np.log(0.01), np.log(0.4))))
            x0 = base.copy()
            x0[idx] += rng.normal(size=(ksub, d)) * sigma
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            k_replace = 0
        elif r < 0.88:
            # kill-and-fill: remove worst hubs, greedy reinsert
            krep = int(rng.integers(2, 12))
            order = np.argsort(-deg)
            kill = order[:krep]
            # occasionally kill random instead
            if rng.random() < 0.3:
                kill = rng.choice(n, size=krep, replace=False)
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[kill] = False
            x_keep = base[keep_mask]
            # greedy fill with random candidates
            x0 = _greedy_fill(x_keep, krep, rng, n_try=400)
            # small jitter on all
            x0 += rng.normal(size=x0.shape) * 1e-3
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            k_replace = krep
        else:
            # fresh random or interpolated start
            if rng.random() < 0.7:
                x0 = rng.normal(size=(n, d))
                x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
                # seed part of best to keep some structure?
                if rng.random() < 0.5:
                    nmix = int(rng.integers(100, 160))
                    idx = rng.choice(n, size=nmix, replace=False)
                    x0[idx] = base[idx] + rng.normal(size=(nmix, d)) * 0.05
                    x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            else:
                # orthogonal transform of best + jitter (different basin orientation is same but jitter breaks)
                Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
                x0 = base @ Q.T
                x0 += rng.normal(size=x0.shape) * 0.02
                x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            k_replace = 0

        # random soft hyperparams
        steps = int(rng.choice([1200, 1800, 2500, 3500]))
        a0 = float(rng.uniform(15, 60))
        a1 = float(rng.uniform(120, 350))
        if a1 < a0:
            a0, a1 = a1, a0
        lr0 = float(np.exp(rng.uniform(np.log(0.02), np.log(0.15))))
        lr1 = float(np.exp(rng.uniform(np.log(0.001), np.log(0.02))))
        mu = float(rng.uniform(0.75, 0.92))
        # time guard
        if time.time() - t0 > BUDGET - 28:
            break
        xopt = _soft_run(x0, steps, a0, a1, lr0, lr1, mu, rng, jitter_scale=0.0)
        v = _max_cos(xopt)
        # fast acceptance: if soft already beats best, polish it
        if v < best_val:
            # short LP snap (allocate 2-4 sec)
            remain = BUDGET - 22 - (time.time() - t0)
            sec = min(4.0, max(1.0, remain / 4))
            xp, vp = _lp_polish(xopt, sec)
            if vp < best_val:
                best_x, best_val = xp, vp
            elif v < best_val:
                # keep soft even if LP worse (should not happen)
                best_x, best_val = xopt, v
        elif v < best_val + 0.0015 and (it % 7 == 0 or rng.random() < 0.08):
            # occasional polish of near-miss to discover new basin
            remain = BUDGET - 22 - (time.time() - t0)
            if remain > 5:
                xp, vp = _lp_polish(xopt, 1.5)
                if vp < best_val:
                    best_x, best_val = xp, vp
        # else discard

    # final long polish on best
    remaining = BUDGET - (time.time() - t0) - 1.0
    if remaining > 2:
        xp, vp = _lp_polish(best_x, min(remaining, 22.0))
        if vp < best_val:
            best_x, best_val = xp, vp
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
