import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _norm(x):
    x = np.asarray(x, dtype=float)
    x = np.where(np.isfinite(x), x, 0.0)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n < 1e-12] = 1.0
    return x / n

def _personal_worst(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def _void_for(others, rng, n_cand=6000):
    cand = rng.standard_normal((n_cand, 4))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    s = cand @ others.T
    sc = s.max(axis=1)
    bi = int(np.argmin(sc))
    return cand[bi].copy()

def _adam_anneal(x, steps, tau0, tau1, lr0, lr1):
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for k in range(steps):
        frac = k / (steps - 1) if steps > 1 else 0.0
        tau = tau0 * (tau1 / tau0) ** frac
        lr = lr0 + (lr1 - lr0) * frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gm = G.max()
        G -= gm
        G /= tau
        # clip for exp overflow safety
        np.clip(G, -50.0, 0.0, out=G)
        np.exp(G, out=G)
        np.fill_diagonal(G, 0.0)
        S = G.sum()
        if not np.isfinite(S) or S <= 0:
            break
        G /= S
        grad = G @ x
        dot = (grad * x).sum(axis=1, keepdims=True)
        g = grad - dot * x
        m *= b1
        m += (1.0 - b1) * g
        v *= b2
        v += (1.0 - b2) * (g * g)
        bc1 = 1.0 - b1 ** (k + 1)
        bc2 = 1.0 - b2 ** (k + 1)
        mh = m / bc1
        vh = v / bc2
        step = mh / (np.sqrt(vh) + eps)
        np.clip(step, -2.0, 2.0, out=step)
        x = x - lr * step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _lp_polish(x, seconds, deadline):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < deadline and radius > 1e-14:
        c = (x @ x.T)[iu]
        mcur = c.max()
        sel = c > mcur - max(20.0 * radius, 1e-9)
        # cap constraints for speed
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > 3000:
            # keep top 3000
            idx = np.where(sel)[0]
            order = np.argpartition(-c[idx], 3000)[:3000]
            keep = np.zeros_like(sel)
            keep[idx[order]] = True
            sel = keep
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
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
        y = _norm(y)
        vv = _max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    deadline = t0 + 172.0
    rng = np.random.default_rng(12345)
    try:
        x0 = np.load("best.npy")
    except Exception:
        try:
            x0 = np.load("record.npy")
        except Exception:
            x0 = rng.standard_normal((272, 4))
    x0 = _norm(np.asarray(x0, dtype=float))
    if x0.shape != (272, 4):
        tmp = rng.standard_normal((272, 4))
        x0 = _norm(tmp)
    best = x0.copy()
    best_val = _max_cos(best)
    cur = best.copy()
    cur_val = best_val

    sigmas = [0.03, 0.012, 0.06, 0.02, 0.09, 0.008, 0.045]
    taus = [(0.015, 0.0015), (0.02, 0.002), (0.008, 0.0008), (0.03, 0.003), (0.012, 0.0006)]
    lrs = [(0.008, 0.0008), (0.012, 0.001), (0.005, 0.0005), (0.015, 0.0015)]

    trial = 0
    # basin hopping until ~140s, reserve rest for LP
    while time.time() < t0 + 138.0 and time.time() < deadline - 30.0:
        trial += 1
        sigma = sigmas[trial % len(sigmas)]
        tau0, tau1 = taus[trial % len(taus)]
        lr0, lr1 = lrs[trial % len(lrs)]
        # start from best or cur (alternate to diversify)
        if trial % 5 == 0:
            x = best.copy()
        else:
            x = cur.copy()
        # global kick
        x = x + sigma * rng.standard_normal((272, 4))
        x = _norm(x)
        # topology change: reinsert worst hubs into voids
        nkill = 2 + (trial % 3)  # 2..4
        if trial % 2 == 0:
            for _ in range(nkill):
                pw = _personal_worst(x)
                # pick randomly among top-8 worst for diversity
                order = np.argsort(-pw)
                pool = order[:8]
                wi = int(pool[rng.integers(0, len(pool))])
                others = np.delete(x, wi, axis=0)
                b = _void_for(others, rng, n_cand=6000)
                x[wi] = b
        # anneal length adapts to remaining time
        steps = 260
        if trial % 4 == 3:
            steps = 380
        if trial % 7 == 0:
            steps = 150
        x = _adam_anneal(x, steps, tau0, tau1, lr0, lr1)
        val = _max_cos(x)
        if val < best_val - 1e-13:
            best = x.copy()
            best_val = val
            cur = x.copy()
            cur_val = val
        elif val < cur_val:
            cur = x.copy()
            cur_val = val
        else:
            # simulated-annealing style acceptance to change basin
            T = 0.0008
            if rng.random() < np.exp(-(val - cur_val) / T):
                cur = x.copy()
                cur_val = val
            # occasionally reset cur to best to avoid drift
            if trial % 25 == 0:
                cur = best.copy()
                cur_val = best_val

    # final tightening
    try:
        best = _lp_polish(best, 28.0, deadline - 1.0)
    except Exception:
        pass
    best = _norm(best)
    return best
